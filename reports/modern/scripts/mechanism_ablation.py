"""Which part of the planner produces its advantage? (A1, A2, A4)

The manuscript currently claims the belief and disowns the planner, which throws
away the larger part of what was built.  Claiming the planner instead requires
saying *which* of its components does the work, and these are the three that can
be isolated without writing a new algorithm.  All of them run the frozen arm --
`bayes_h16_p4`, continuous accelerating unicycle executor, occluded,
robot.visible=false -- and change exactly one thing.

A1  Risk schedule.  The planner spends a per-step risk budget that grows along
    the horizon (near 0.30 to far 0.90, gamma 2).  The comparators use one
    constant risk at every step.  Two endpoint controls are not enough to show
    the *shape* matters, because beating an over-strict and an over-permissive
    constant is easy; a constant tuned on the development set is the control that
    can actually refute the claim, so the sweep is part of the experiment.

A2  Route separation.  The population is split between the warm start and three
    route seeds, fitted independently so opposite evasions cannot average into a
    motion nobody evaluated.  The control is a genuine single-distribution CEM:
    same population, same iterations, same seeds, same rule for executing the
    best trajectory; only the grouping changes.

A4  Degraded policy.  When no candidate is feasible even at the first step, this
    planner keeps the least-bad executable action.  The comparators brake.  Since
    braking is what turned their infeasible solves into collisions in an
    environment where pedestrians do not yield, the policy itself may account for
    part of the gap, and it has to be measured rather than assumed.

Development split only (cases 20100-20199, the two five-person dev scenes).  No
test layouts are touched.
"""
from __future__ import annotations
import json, os, sys, time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")

import numpy as np
from continuous_mpc_gate import DEFAULT_CROWDNAV, run_episode
from evaluate_matched_safety import config as point_config
from modern_dynamics import ContinuousUnicycleMPC

OUT = Path("/home/abc/temp/modern/runs/mechanism")
CASES = list(range(20100, 20200))
SCENES = [("dev_circle", 5, "circle_crossing", 4.0, None),
          ("dev_square", 5, "square_crossing", None, 10.0)]
POINT = 4                       # the frozen working point


def single_mode(base):
    class SingleModeCEM(base):
        """One distribution over the whole population; no route separation."""
        def _route_seeds(self, obs):
            return np.empty((0, self.cfg.horizon, 2), dtype=np.float64)
    return SingleModeCEM


def braking(base):
    class BrakingFallback(base):
        """Brake when nothing is feasible, as the comparators natively do."""
        def _degraded_choice(self, costs, first_physical, params, obs):
            # The slowest first action available in the evaluated population is
            # the closest thing this sampler has to the comparators' hard brake.
            return int(np.argmin(np.abs(params[:, 0, 0])))
    return BrakingFallback


def arms():
    base = point_config(POINT)
    out = {
        "A1_shaped":       (base, ContinuousUnicycleMPC),
        "A2_single_mode":  (base, single_mode(ContinuousUnicycleMPC)),
        "A4_brake":        (base, braking(ContinuousUnicycleMPC)),
    }
    # The tuned-constant control: a flat budget at each registered level, so the
    # shaped schedule is compared against the best constant, not only the ends.
    for level in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90):
        out[f"A1_flat_{level:.2f}"] = (
            replace(base, near_chance_limit=level, chance_limit=level),
            ContinuousUnicycleMPC)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (cfg, planner) in arms().items():
        for scene_id, humans, generator, radius, width in SCENES:
            target = OUT / f"{name}__{scene_id}.jsonl"
            if target.exists():
                continue
            rows, started = [], time.time()
            for case in CASES:
                r = run_episode(DEFAULT_CROWDNAV, "bayes", humans, generator, case, cfg,
                                radius, width, 0, 0.0, 0.0, 1.0, time_limit=25,
                                planner_type=planner, robot_kinematics="unicycle",
                                occlusion=True)
                rows.append({
                    "arm": name, "scene_id": scene_id, "case_id": case,
                    "event": r.event, "nav_time": r.nav_time,
                    "success_without_overlap": r.success_without_overlap,
                    "collision_union": r.collision_union, "timeout": r.timeout,
                    "path_length": r.path_length,
                    "degraded_steps": sum(1 for s in r.steps
                                          if s.get("feasibility_class", 0) >= 2),
                    "steps": len(r.steps)})
            tmp = target.with_suffix(".partial")
            tmp.write_text("".join(json.dumps(x) + "\n" for x in rows))
            tmp.replace(target)
            print(f"  {name:16s} {scene_id:12s} {len(rows):3d} 回合 "
                  f"{time.time()-started:6.1f}s", flush=True)
    report()


def report():
    rows = []
    for p in sorted(OUT.glob("*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        return
    print(f"\n=== 机制消融（冻结 bayes_h16_p4，开发集 100 布局 × 2 场景）===")
    print(f"{'臂':18s} {'n':>4s} {'SR%':>7s} {'CR%':>7s} {'TR%':>7s} {'罚时s':>7s} {'退化步%':>8s}")
    summary = {}
    for name in sorted({r["arm"] for r in rows}):
        per = []
        for scene_id, *_ in SCENES:
            g = [r for r in rows if r["arm"] == name and r["scene_id"] == scene_id]
            if not g:
                continue
            per.append([
                np.mean([r["success_without_overlap"] and not r["collision_union"] for r in g]),
                np.mean([bool(r["collision_union"]) for r in g]),
                np.mean([r["timeout"] and not r["collision_union"] for r in g]),
                np.mean([r["nav_time"] if (r["success_without_overlap"] and not r["collision_union"])
                         else 25.0 for r in g]),
                np.sum([r["degraded_steps"] for r in g]) / max(np.sum([r["steps"] for r in g]), 1)])
        if not per:
            continue
        m = np.mean(per, axis=0)
        n = sum(1 for r in rows if r["arm"] == name)
        summary[name] = {"n": n, "sr": 100*m[0], "cr": 100*m[1], "tr": 100*m[2],
                         "time": m[3], "degraded_step_frac": 100*m[4]}
        print(f"{name:18s} {n:4d} {100*m[0]:7.2f} {100*m[1]:7.2f} {100*m[2]:7.2f} "
              f"{m[3]:7.2f} {100*m[4]:8.2f}")
    Path("/home/abc/temp/modern/snapshot/mechanism_ablation.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print("-> snapshot/mechanism_ablation.json")


if __name__ == "__main__":
    main()
