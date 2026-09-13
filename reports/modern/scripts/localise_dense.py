"""Where exactly does the dense-scene constraint set close? (localisation)

The repair report's strongest clue: on the same failing observation, 3000 sampled
executable trajectories satisfied our risk criterion but none satisfied all of
the comparators' hard ellipse constraints.  My earlier corridor analysis missed
this because it inflated the pedestrian radius by the chi-square factor at k=0
only, ignoring that the posterior covariance grows along the horizon.  The
constraint a comparator enforces at horizon step k is

    radius_k = r_robot + r_human + margin + chi(risk) * sigma_k

and sigma_k grows.  This measures sigma_k from the project's own adapter on the
real dense scenes, converts it into the radius each frozen arm enforces, and
compares the total blocked width against the corridor the robot has to cross.
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, "/home/abc/temp/modern")
import numpy as np
from scipy.stats import chi2
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env
from evaluate_matched_safety import MatchedAdapter, config as point_config
from unicycle_mpc_gate import unicycle_config

ROBOT_R, HUMAN_R, MARGIN = 0.3, 0.3, 0.10
CHI = {"tmpc_shared": float(np.sqrt(chi2.ppf(1 - 0.35, 2))),
       "shmpc_shared": float(np.sqrt(chi2.ppf(1 - 0.40, 2)))}

def measure(scene_id, humans, generator, radius, width, cases, horizon=16):
    cfg = unicycle_config(point_config(2), horizon=horizon)
    env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", humans, generator, radius, width, 25, True)
    sigmas = [[] for _ in range(horizon)]
    counts = []
    for case in cases:
        env.reset(options={"test_case": case})
        # Constructed exactly as run_episode does, so the covariance measured
        # here is the one the arms were actually handed.
        ad = MatchedAdapter("bayes", cfg.horizon, cfg.human_margin, cfg.dt,
                            cfg.chance_limit, cfg.fixed_uncertainty_radius,
                            cfg.acceleration_std, 0.0, 0.0, 1.0,
                            case * 1000003 + 7919,
                            cfg.conformal_visible_radii, cfg.conformal_hidden_radii)
        for _ in range(6):                      # a few steps in, so tracks mature
            obs = ad.read(env)
            env.step(_zero_action(env))
        obs = ad.read(env)
        cov = obs.human_position_covariance
        if cov is None or np.asarray(cov).size == 0:
            continue
        cov = np.asarray(cov)                   # (H, N, 2, 2)
        counts.append(cov.shape[0])
        eig = np.linalg.eigvalsh(cov)           # ascending
        major = np.sqrt(np.maximum(eig[..., 1], 0.0))   # (H, N)
        for k in range(min(horizon, major.shape[1])):
            sigmas[k].extend(major[:, k].tolist())
    return sigmas, counts

def _zero_action(env):
    # The environment is built with its default holonomic robot here; the robot
    # is held still for a few steps only so the tracker matures.  What is being
    # measured is the predicted covariance the arms receive, not a trajectory.
    from continuous_mpc_gate import _load_modules
    _, _, ActionXY, _, _ = _load_modules(DEFAULT_CROWDNAV)
    return ActionXY(0.0, 0.0)

def main():
    reg = json.load(open("/home/abc/temp/modern/snapshot/case_registry.json"))
    for scene_id, humans, generator, radius, width, span in (
            ("baseline_circle", 5, "circle_crossing", 4.0, None, 8.0),
            ("dense_square", 20, "square_crossing", None, 10.0, 10.0)):
        cases = reg["splits"][f"T_{scene_id}"]["cases"][:8]
        sigmas, counts = measure(scene_id, humans, generator, radius, width, cases)
        seen = np.mean(counts) if counts else 0
        print(f"\n=== {scene_id}（{humans} 人，平均可见 {seen:.1f} 个，走廊宽约 {span:.0f} m）===")
        print(f"{'k':>3s} {'时刻s':>6s} {'sigma_k m':>10s} "
              + " ".join(f"{a.split('_')[0]+'半径m':>13s} {a.split('_')[0]+'占宽m':>13s}" for a in CHI))
        for k in (0, 3, 7, 11, 15):
            if not sigmas[k]:
                continue
            s = float(np.median(sigmas[k]))
            cells = []
            for arm, chi in CHI.items():
                r = ROBOT_R + HUMAN_R + MARGIN + chi * s
                cells += [f"{r:13.2f}", f"{min(2*r*seen, 99):13.1f}"]
            print(f"{k:3d} {(k+1)*0.25:6.2f} {s:10.3f} " + " ".join(cells))
        print(f"  （占宽 = 2×半径×可见人数，是最坏情形的完全不重叠上界；"
              f"与走廊宽 {span:.0f} m 比较）")

if __name__ == "__main__":
    main()
