"""Measure real per-episode cost so the ETA is computed, not extrapolated.

Order section 13.10 requires the estimate to come from measured rates on both
sparse and dense scenes, per method, under the resource budget the formal run
will actually use -- not from one 5-human episode scaled up.

Each measurement runs one episode and reports wall time, peak RSS of this
process tree, and the number of control steps, so the cost per step is visible
separately from the fixed start-up cost of bringing a bridge up.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")

import numpy as np                                                    # noqa: E402
from continuous_mpc_gate import DEFAULT_CROWDNAV, run_episode          # noqa: E402
from unicycle_mpc_gate import UnicycleCEMMPC                           # noqa: E402
from evaluate_matched_safety import config                             # noqa: E402
from modern_worker import bridge_planner_factory                       # noqa: E402

SWEEP = Path("/home/abc/temp/modern/risk_sweep")

# One sparse and one dense scene from the registered six.
PROBES = [("baseline_circle", "circle_crossing", 5, 4.0, None),
          ("dense_square", "square_crossing", 20, None, 10.0)]

ARMS = [
    ("bayes_full", lambda cfg: UnicycleCEMMPC),
    ("tmpc_shared", lambda cfg: bridge_planner_factory(
        "tmpc", settings=str(SWEEP / "tmpc_own0.35.yaml"))),
    ("shmpc_shared", lambda cfg: bridge_planner_factory(
        "shmpc", settings=str(SWEEP / "shmpc_own0.42.yaml"))),
]


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    cfg = config(3)
    crowdnav = Path(os.environ.get("CROWDNAV_ROOT", str(DEFAULT_CROWDNAV)))
    results = []
    print(f"{'臂':14s} {'场景':16s} {'人数':>4s} {'墙钟s':>8s} {'步数':>5s} "
          f"{'每步ms':>8s} {'峰值RSS_MB':>11s} {'结局':>10s}")
    for arm, factory in ARMS:
        for scene_id, scenario, humans, radius, width in PROBES:
            before = peak_rss_mb()
            start = time.perf_counter()
            try:
                r = run_episode(crowdnav, "bayes", humans, scenario, 3700, cfg,
                                radius, width, 0, 0.0, 0.0, 1.0, time_limit=25,
                                planner_type=factory(cfg), robot_kinematics="unicycle")
                event, steps = r.event, r.solver_steps
            except Exception as exc:
                event, steps = f"error:{type(exc).__name__}", 0
            wall = time.perf_counter() - start
            peak = max(peak_rss_mb(), before)
            per_step = 1000.0 * wall / steps if steps else float("nan")
            print(f"{arm:14s} {scene_id:16s} {humans:4d} {wall:8.2f} {steps:5d} "
                  f"{per_step:8.1f} {peak:11.1f} {event:>10s}", flush=True)
            results.append({"arm": arm, "scene_id": scene_id, "humans": humans,
                            "wall_s": wall, "steps": steps, "ms_per_step": per_step,
                            "peak_rss_mb": peak, "event": event})

    Path("/home/abc/temp/modern/snapshot/throughput.json").write_text(
        json.dumps(results, indent=1))

    # Cost of the formal matrix, from these rates.
    sparse = {r["arm"]: r["wall_s"] for r in results if r["humans"] == 5}
    dense = {r["arm"]: r["wall_s"] for r in results if r["humans"] == 20}
    # Six scenes: 5,10,10,20,12,20 humans -> two sparse-ish, four denser.
    # Weight by a linear interpolation in crowd size between the two probes.
    sizes = [5, 10, 10, 20, 12, 20]
    print("\n每方法跑完一遍六场景 100 布局 × 3 repeat（1800 回合）的估计墙钟：")
    total = {}
    for arm in sparse:
        per_scene = []
        for n in sizes:
            t = sparse[arm] + (dense[arm] - sparse[arm]) * (n - 5) / 15.0
            per_scene.append(t * 100 * 3)
        total[arm] = sum(per_scene)
        print(f"  {arm:14s} {total[arm]/3600:8.1f} 机时")
    print(f"\n  T-OCC 四臂（bayes_r1 按 bayes_full 计）: "
          f"{(total.get('bayes_full',0)*2 + total.get('tmpc_shared',0) + total.get('shmpc_shared',0))/3600:.1f} 机时")
    print(f"  T-FULL 三臂: "
          f"{(total.get('bayes_full',0) + total.get('tmpc_shared',0) + total.get('shmpc_shared',0))/3600:.1f} 机时")


if __name__ == "__main__":
    main()
