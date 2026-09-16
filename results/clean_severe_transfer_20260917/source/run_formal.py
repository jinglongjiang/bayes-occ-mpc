"""Queues D and E: formal evaluation at frozen operating points.

D  clean six-scene comparison, bayes vs age_margin
E  fair severe-noise comparison, five arms on the 20-person square

Operating points come from a frozen JSON written by the development sweep; this
script never selects them.  Blocks are (arm, scene) and each writes its own
file, so a machine can be given any subset and a failed block re-runs alone.
"""
from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from dataclasses import asdict, replace
from functools import partial
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

SRC = os.environ.get("SRC", "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, SRC)

import numpy as np
from continuous_mpc_gate import run_episode, DEFAULT_CROWDNAV
from evaluate_matched_safety import MatchedAdapter, AgeMarginAdapter, config

CROWDNAV = Path(os.environ.get("CROWDNAV_ROOT", str(DEFAULT_CROWDNAV)))
SCENES = (
    ("baseline_circle", 5, "circle_crossing", 4.0, None),
    ("baseline_square", 10, "square_crossing", None, 10.0),
    ("dense_circle", 10, "circle_crossing", 4.0, None),
    ("dense_square", 20, "square_crossing", None, 10.0),
    ("large_circle", 12, "circle_crossing", 6.0, None),
    ("large_square", 20, "square_crossing", None, 14.0),
)
KEEP = ("case_id", "event", "success", "collision", "timeout", "nav_time",
        "scored_time", "path_length", "min_clearance", "actual_min_clearance",
        "actual_overlap_steps", "collision_union", "success_without_overlap",
        "mean_plan_ms", "p95_plan_ms")
STATE = {}


def init(calibration, noise, points, frozen_sensor=False):
    STATE.update(calibration=calibration, noise=noise, points=points, frozen_sensor=frozen_sensor)


def adapter_factory(*args, arm, **kwargs):
    point = STATE['points'][arm]
    cls = AgeMarginAdapter if arm == 'age_margin' else MatchedAdapter
    extra = {} if arm == 'age_margin' else {'method': arm}
    adapter = cls(*args, calibration=STATE['calibration'], point=point, **extra, **kwargs)
    if STATE['frozen_sensor']:
        p, v, d = STATE['calibration']['noise']
        adapter.rfs.cfg = replace(adapter.rfs.cfg, position_measurement_std=p,
                                  velocity_measurement_std=v, detection_probability=min(.98, d))
        assert adapter.position_noise_std == STATE['noise'][0]
        assert adapter.rfs.cfg.position_measurement_std == p
    return adapter


def one(task):
    arm, scene_index, case = task
    _, count, sim, radius, width = SCENES[scene_index]
    point = STATE["points"][arm]
    factory = partial(adapter_factory, arm=arm)
    r = run_episode(CROWDNAV, "bayes", count, sim, case, config(point),
                    radius, width, 0, *STATE["noise"], time_limit=25,
                    adapter_factory=factory)
    d = asdict(r)
    return {"arm": arm, "point": point, "scene": scene_index,
            **{k: d[k] for k in KEEP if k in d}}


def main():
    blocks, workers = sys.argv[1], int(sys.argv[2])
    frozen_path, calib_path = sys.argv[3], sys.argv[4]
    case_base, count = int(sys.argv[5]), int(sys.argv[6])
    out = Path(sys.argv[7]); out.mkdir(parents=True, exist_ok=True)
    frozen = json.load(open(frozen_path))
    calib = json.load(open(calib_path))
    noise = tuple(calib.get("noise", [0.0, 0.0, 1.0]))
    if os.environ.get('TEST_NOISE'):
        noise = tuple(json.loads(os.environ['TEST_NOISE']))
    frozen_sensor = os.environ.get('FREEZE_SENSOR') == '1'
    points = frozen["points"]
    print(f"冻结工作点={points}  标定={calib_path}  噪声={noise}  "
          f"case={case_base}-{case_base + count - 1}", flush=True)
    manifest = {'test_noise': noise, 'frozen_sensor': frozen_sensor,
                'calibration_noise': calib.get('noise'), 'points':points,
                'cases':[case_base,case_base+count-1], 'blocks':blocks,
                'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__),Path(SRC)/'continuous_mpc_gate.py',
                              Path(SRC)/'evaluate_matched_safety.py',Path(SRC)/'bayesian_rfs.py',
                              CROWDNAV/'crowd_sim/envs/crowd_sim.py',Path(frozen_path),Path(calib_path)]}}
    manifest_path=out/'protocol.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError('existing protocol mismatch')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    for spec in blocks.split(","):
        arm, scene_index = spec.split(":")
        scene_index = int(scene_index)
        path = out / f"{arm}_sc{scene_index}.json"
        if path.exists():
            old=json.loads(path.read_text())
            assert sorted(r['case_id'] for r in old['episodes']) == list(range(case_base,case_base+count))
            assert old['noise'] == list(noise) and old.get('frozen_sensor',False) == frozen_sensor
            print(f"  跳过 {path.name}", flush=True)
            continue
        start = time.time()
        with Pool(workers, initializer=init, initargs=(calib, noise, points, frozen_sensor)) as pool:
            rows = pool.map(one, [(arm, scene_index, case_base + i) for i in range(count)],
                            chunksize=2)
        n = len(rows)
        path.write_text(json.dumps(
            {"arm": arm, "point": points[arm], "scene": scene_index,
             "noise": list(noise), "cases": [case_base, case_base + count - 1],
             "frozen_sensor": frozen_sensor,
             "frozen_points": frozen, "calibration": calib_path,
             "episodes": rows}, indent=1))
        print(f"  {path.name}  n={n}"
              f"  审计SR={sum(r['success_without_overlap'] for r in rows)/n*100:5.1f}%"
              f"  审计CR={sum(r['collision_union'] for r in rows)/n*100:5.1f}%"
              f"  时间={np.mean([r['scored_time'] for r in rows]):6.2f}s"
              f"  用时={time.time()-start:5.1f}s", flush=True)
    print("本机分块完成", flush=True)


if __name__ == "__main__":
    main()
