"""Is the comparators' collapse the crowd, or the width of the belief we hand them?

Run after the formal matrix.  T-OCC shows both comparators' solve success falling
monotonically with crowd density until, at twenty pedestrians in a square, they
stop moving.  Two very different things produce that picture:

  (a) the crowd genuinely closes the corridor, which every method must face; or
  (b) the constraint each comparator builds from *our* posterior is unsatisfiable
      by construction, in which case the main table measures how wide our belief
      is in their constraint form, not how well anyone handles belief.

They are distinguishable because the three arms are not given the same kind of
constraint.  The comparators inflate each pedestrian to a hard ellipsoid whose
semi-axes are the posterior's, scaled by chi-square at their configured risk;
our planner keeps the same posterior but spends a risk budget across the whole
trajectory.  A hard ellipsoid at risk 0.05 is far more conservative than a
chance constraint at the same nominal risk, so the comparison is between
constraint formulations under a shared posterior -- which is a legitimate result
to report, but must not be written up as belief handling.

What this measures, replaying recorded observations without re-running any
episode's control loop:

  1. the free corridor left by the raw pedestrian discs;
  2. the corridor left after each comparator's chi-square-scaled ellipsoids;
  3. the chi-square factor each frozen risk implies, so the inflation is
     attributable to a number in the frozen configuration rather than to a
     mystery.

If (1) is open and (2) is closed in exactly the scenes where solve success
collapsed, conclusion (b) holds and the main table's wording has to change.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, "/home/abc/temp/modern")

import numpy as np                                                    # noqa: E402
from scipy.stats import chi2                                          # noqa: E402

ROOT = Path("/home/abc/temp/modern")
SCENES = [("baseline_circle", 5, "circle_crossing", 4.0, None),
          ("dense_circle", 10, "circle_crossing", 4.0, None),
          ("dense_square", 20, "square_crossing", None, 10.0),
          ("large_square", 20, "square_crossing", None, 14.0)]
ROBOT_RADIUS, HUMAN_RADIUS, MARGIN = 0.3, 0.3, 0.10


def chi_factor(risk, dof=2):
    """The radius multiplier a hard ellipsoid constraint at this risk implies."""
    return float(np.sqrt(chi2.ppf(1.0 - risk, dof)))


def corridor_width(centres, radii, start, goal, samples=400):
    """Narrowest free gap across the straight start->goal corridor.

    Measured as the largest clearance available to the robot centre on the line
    perpendicular to the path, minimised over the path -- a lower bound on what
    any planner tracking that reference has to squeeze through.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    direction = goal - start
    length = np.hypot(*direction)
    if length < 1e-9:
        return np.inf
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    offsets = np.linspace(-4.0, 4.0, 161)
    worst = np.inf
    for t in np.linspace(0.0, 1.0, samples):
        point = start + direction * t
        free = []
        for offset in offsets:
            probe = point + normal * offset
            if centres.size == 0:
                free.append(True); continue
            gaps = np.hypot(*(probe - centres).T) - radii - ROBOT_RADIUS
            free.append(bool(np.all(gaps > 0)))
        # widest run of free offsets at this station
        best = run = 0
        for ok in free:
            run = run + 1 if ok else 0
            best = max(best, run)
        worst = min(worst, best * (offsets[1] - offsets[0]))
    return worst


def main():
    from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env, ObservationAdapter
    frozen = json.loads((ROOT / "snapshot/frozen.json").read_text())
    risks = {arm: spec["risk_own_units"] * (2.0 if spec["family"] == "shmpc" else 1.0)
             for arm, spec in frozen["arms_occluded"].items()
             if spec["family"] != "bayes"}
    factors = {arm: chi_factor(r) for arm, r in risks.items()}
    print("冻结风险 -> 硬椭球的卡方半径倍数")
    for arm, r in risks.items():
        print(f"  {arm:14s} 有效 risk {r:.2f} -> 半径 ×{factors[arm]:.2f}")

    registry = json.loads((ROOT / "snapshot/case_registry.json").read_text())
    out = {}
    print(f"\n{'场景':16s} {'人数':>4s} {'原始盘走廊m':>12s} "
          + " ".join(f"{a.split('_')[0]+'椭球走廊m':>14s}" for a in factors))
    for scene_id, humans, generator, radius, width in SCENES:
        cases = registry["splits"][f"T_{scene_id}"]["cases"][:20]
        env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", humans, generator,
                              radius, width, 25, True)
        raw, inflated = [], {arm: [] for arm in factors}
        for case in cases:
            env.reset(options={"test_case": case})
            centres = np.array([[h.px, h.py] for h in env.humans], dtype=float)
            start = np.array([env.robot.px, env.robot.py])
            goal = np.array([env.robot.gx, env.robot.gy])
            base = np.full(len(centres), HUMAN_RADIUS + MARGIN)
            raw.append(corridor_width(centres, base, start, goal))
            # At the first step the posterior of a just-detected pedestrian is
            # near its measurement noise; the inflation that matters is the
            # chi-square factor applied to the radius the comparator enforces.
            for arm, factor in factors.items():
                inflated[arm].append(
                    corridor_width(centres, base * factor, start, goal))
        row = {"humans": humans, "raw_median": float(np.median(raw))}
        cells = [f"{np.median(raw):12.2f}"]
        for arm in factors:
            row[f"{arm}_median"] = float(np.median(inflated[arm]))
            cells.append(f"{np.median(inflated[arm]):14.2f}")
        out[scene_id] = row
        print(f"{scene_id:16s} {humans:4d} " + " ".join(cells))

    (ROOT / "snapshot/corridor_analysis.json").write_text(
        json.dumps({"risks": risks, "chi_factors": factors, "scenes": out},
                   indent=1, ensure_ascii=False))
    print("\n判据：原始盘走廊 > 0 而椭球走廊 = 0 的场景，说明约束集按构造不可满足，")
    print("      主表口径必须写成『共享后验下两种约束形式的比较』。")
    print("-> snapshot/corridor_analysis.json")


if __name__ == "__main__":
    main()
