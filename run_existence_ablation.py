"""Separate the existence probability from the position covariance.

Three arms share the filter, the track set, the trajectory means, the position
covariances, the planner, the operating point, the cases and the audit rule.
They differ only in what the risk term is allowed to use:

  bayes          r_i * P(collision | present)      full posterior
  existence_one  1.0 * P(collision | present)      position uncertainty only
  posterior_mean 1.0 * 1{mean inside disc}         neither

So bayes vs existence_one isolates the existence probability, and
existence_one vs posterior_mean isolates the position uncertainty.  The
previous posterior_mean ablation removed both at once.

Protocol matches the formal occlusion result: six scenes, point=3,
cases 5000-5099, occlusion on, no sensor noise.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from continuous_mpc_gate import run_episode, DEFAULT_CROWDNAV
from evaluate_matched_safety import MatchedAdapter, config

SCENES = (
    ("baseline_circle", 5, "circle_crossing", 4.0, None),
    ("baseline_square", 10, "square_crossing", None, 10.0),
    ("dense_circle", 10, "circle_crossing", 4.0, None),
    ("dense_square", 20, "square_crossing", None, 10.0),
    ("large_circle", 12, "circle_crossing", 6.0, None),
    ("large_square", 20, "square_crossing", None, 14.0),
)
POINT = 3
CASE_BASE = 5000


def main():
    per_scene = int(sys.argv[1])
    calib = json.load(open(sys.argv[2]))
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    base = config(POINT)
    arms = {"bayes": base, "existence_one": replace(base, existence_override=1.0)}
    for arm, cfg in arms.items():
        rows = []
        for si, (label, count, sim, radius, width) in enumerate(SCENES):
            for i in range(per_scene):
                r = run_episode(DEFAULT_CROWDNAV, "bayes", count, sim,
                                CASE_BASE + i, cfg, radius, width, 0,
                                0.0, 0.0, 1.0, time_limit=25,
                                adapter_factory=partial(MatchedAdapter, method="bayes",
                                                        calibration=calib, point=POINT))
                d = asdict(r)
                rows.append({"scene": si, **{k: d[k] for k in
                             ("case_id", "event", "success", "collision", "timeout",
                              "nav_time", "scored_time", "path_length",
                              "min_clearance", "actual_min_clearance",
                              "actual_overlap_steps", "collision_union",
                              "success_without_overlap")}})
            n = len(rows)
            print(f"  [{arm}] 场景{si} 完成，累计 {n} 回合", flush=True)
        (out / f"{arm}.json").write_text(json.dumps(
            {"arm": arm, "point": POINT, "existence_override": cfg.existence_override,
             "cases": [CASE_BASE, CASE_BASE + per_scene - 1], "episodes": rows}, indent=1))
        n = len(rows)
        aud = sum(r["success_without_overlap"] for r in rows)
        print(f"{arm:14s} n={n:4d}  审计SR={aud/n*100:5.2f}%  "
              f"CR={sum(r['collision'] for r in rows)/n*100:5.2f}%  "
              f"TR={sum(r['timeout'] for r in rows)/n*100:5.2f}%  "
              f"T={np.mean([r['scored_time'] for r in rows]):6.3f}s", flush=True)


if __name__ == "__main__":
    main()
