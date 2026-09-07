"""Closed-loop development check: shipped Gaussian vs the frozen process mixture.

Only the collision-probability computation differs.  The filter, the trajectory
means, gamma, the chance limits and the CEM search are untouched, and the
mixture parameters are frozen at the values fitted offline on episodes
8000-8014; nothing is tuned here.  Cases are fresh: they were used neither for
fitting nor for model selection.

This is a go/no-go on whether the mixture is worth a larger evaluation, not a
safety claim.
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

SCENE = ("dense_square", 20, "square_crossing", None, 10.0)
POINT = 2
CASE_BASE = 9000
NOISE = {"clean": (0.0, 0.0, 1.0), "severe": (0.1, 0.2, 0.8)}
# Frozen from the constrained offline fit on dev episodes 8000-8014.
MIX = dict(mixture_pi=0.0340, mixture_a=0.0001, mixture_b=1.2363)


def main():
    count = int(sys.argv[1])
    calib = json.load(open(sys.argv[2]))
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    base = config(POINT)
    arms = {"gaussian": base, "mixture": replace(base, **MIX)}
    for label, noise in NOISE.items():
        for arm, cfg in arms.items():
            rows = []
            for i in range(count):
                r = run_episode(DEFAULT_CROWDNAV, "bayes", SCENE[1], SCENE[2],
                                CASE_BASE + i, cfg, SCENE[3], SCENE[4], 0, *noise,
                                time_limit=25,
                                adapter_factory=partial(MatchedAdapter, method="bayes",
                                                        calibration=calib, point=POINT))
                d = asdict(r)
                rows.append({k: d[k] for k in
                             ("case_id", "event", "success", "collision", "timeout",
                              "nav_time", "scored_time", "path_length",
                              "min_clearance", "actual_min_clearance", "collision_union")})
            name = f"{label}_{arm}"
            (out / f"{name}.json").write_text(json.dumps(
                {"noise": label, "arm": arm, "mixture": MIX if arm == "mixture" else None,
                 "cases": [CASE_BASE, CASE_BASE + count - 1], "episodes": rows}, indent=1))
            n = len(rows)
            print(f"{name:18s} n={n:3d}  SR={sum(r['success'] for r in rows)/n*100:5.1f}%  "
                  f"CR={sum(r['collision'] for r in rows)/n*100:5.1f}%  "
                  f"TR={sum(r['timeout'] for r in rows)/n*100:5.1f}%  "
                  f"T={np.mean([r['scored_time'] for r in rows]):6.3f}s", flush=True)


if __name__ == "__main__":
    main()
