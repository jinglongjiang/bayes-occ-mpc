"""Does the horizon-shaped risk budget account for the method's advantage?

The manuscript states the budget

    eps_k = eps_near + (eps_far - eps_near) * ((k-1)/(H-1))^gamma

as a configuration detail.  An analysis of the shared posterior suggests it is
actually the operative mechanism: the constant-velocity covariance grows to about
1.27 m of positional standard deviation by the four-second horizon, so a constant
risk level must respect a large disc at every step, whereas a shaped budget spends
its allowance late and replans before the late steps arrive.

That is an argument, not evidence.  This ablation supplies the evidence by holding
everything else fixed -- same filter, same collision-mass evaluation, same sampler,
same layouts -- and changing only the budget's shape:

    shaped        eps_near -> eps_far with gamma = 2   (the method as published)
    flat_near     eps_k = eps_near for all k           (uniformly strict)
    flat_far      eps_k = eps_far   for all k          (uniformly permissive)

If `shaped` beats both flat arms, the shape itself carries the benefit and the
claim is about risk allocation over the horizon.  If a flat arm matches it, the
shape is incidental and the manuscript should not be reorganised around it.
"""
from __future__ import annotations
import json, os, sys, time, collections
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")

import numpy as np
from continuous_mpc_gate import DEFAULT_CROWDNAV, run_episode, ContinuousCEMMPC
from evaluate_matched_safety import SCENES, config as point_config

OUT = Path("/home/abc/temp/modern/runs/risk_shape")
POINT = 2                      # the manuscript's completed holonomic study point


def arms():
    base = point_config(POINT)
    return {
        "shaped":    replace(base),                                    # near 0.15 -> far 0.50
        "flat_near": replace(base, chance_limit=base.near_chance_limit),
        "flat_far":  replace(base, near_chance_limit=base.chance_limit),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = json.loads(Path("/home/abc/temp/modern/snapshot/case_registry.json").read_text())
    configs = arms()
    print("风险剖面（H=16）：")
    for name, cfg in configs.items():
        planner = ContinuousCEMMPC(cfg)
        lim = planner._belief_risk_limits()
        print(f"  {name:10s} k=0 {lim[0]:.3f}  k=7 {lim[7]:.3f}  k=15 {lim[15]:.3f}  "
              f"平均 {lim.mean():.3f}")
    for name, cfg in configs.items():
        for scene_id, humans, generator, radius, width in SCENES:
            target = OUT / f"{name}__{scene_id}.jsonl"
            if target.exists():
                continue
            cases = reg["splits"][f"T_{scene_id}"]["cases"]
            rows, started = [], time.time()
            for case in cases:
                r = run_episode(DEFAULT_CROWDNAV, "bayes", humans, generator, case, cfg,
                                radius, width, 0, 0.0, 0.0, 1.0, time_limit=25,
                                planner_type=ContinuousCEMMPC,
                                robot_kinematics="holonomic")
                rows.append({"arm": name, "scene_id": scene_id, "case_id": case,
                             "event": r.event,
                             "success_without_overlap": r.success_without_overlap,
                             "collision_union": r.collision_union,
                             "timeout": r.timeout, "nav_time": r.nav_time,
                             "path_length": r.path_length})
            tmp = target.with_suffix(".partial")
            tmp.write_text("".join(json.dumps(x) + "\n" for x in rows))
            tmp.replace(target)
            print(f"  {name:10s} {scene_id:16s} {len(rows):3d} 回合  "
                  f"{time.time()-started:6.1f}s", flush=True)
    report()


def report():
    rows = []
    for p in sorted(OUT.glob("*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        return
    print(f"\n=== 风险剖面消融（holonomic，point {POINT}，六场景各 100 布局）===")
    print(f"{'臂':10s} {'n':>5s} {'SR%':>7s} {'CR%':>7s} {'TR%':>7s} {'罚时s':>7s}")
    summary = {}
    for name in ("shaped", "flat_near", "flat_far"):
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
                         else 25.0 for r in g])])
        if not per:
            continue
        m = np.mean(per, axis=0)
        n = sum(1 for r in rows if r["arm"] == name)
        summary[name] = {"n": n, "sr": 100*m[0], "cr": 100*m[1], "tr": 100*m[2], "time": m[3]}
        print(f"{name:10s} {n:5d} {100*m[0]:7.2f} {100*m[1]:7.2f} {100*m[2]:7.2f} {m[3]:7.2f}")
    Path("/home/abc/temp/modern/snapshot/risk_shape_ablation.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print("-> snapshot/risk_shape_ablation.json")


if __name__ == "__main__":
    main()
