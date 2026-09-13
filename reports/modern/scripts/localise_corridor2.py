"""The free corridor at each horizon step, with the radius each arm enforces there.

The earlier "blocked width" figure assumed non-overlapping ellipses, which is a
loose upper bound.  This measures the real thing: at horizon step k, place every
tracked pedestrian at its predicted mean and inflate it to the radius the
comparator's hard ellipse constraint enforces at that step, then find the widest
free lane the robot centre could still use across the start-to-goal corridor.

A hard-constraint planner must find one trajectory that is admissible at every k
simultaneously.  So the binding quantity is the *minimum over k* of this width.
A chance-constrained planner spends a risk budget instead and may accept a
trajectory that violates the far-horizon ellipses, because it replans each step.
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, "/home/abc/temp/modern")
import numpy as np
from scipy.stats import chi2
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env, _load_modules
from evaluate_matched_safety import MatchedAdapter, config as point_config
from unicycle_mpc_gate import unicycle_config

ROBOT_R, HUMAN_R, MARGIN = 0.3, 0.3, 0.10
CHI = {"tmpc": float(np.sqrt(chi2.ppf(1 - 0.35, 2))),
       "shmpc": float(np.sqrt(chi2.ppf(1 - 0.40, 2)))}

def free_lane(centres, radii, start, goal, stations=60, half=6.0, step=0.05):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    d = goal - start; L = np.hypot(*d)
    if L < 1e-9: return np.inf
    u = d / L; n = np.array([-u[1], u[0]])
    offsets = np.arange(-half, half + step, step)
    worst = np.inf
    for t in np.linspace(0.05, 0.95, stations):
        p = start + d * t
        probes = p[None, :] + n[None, :] * offsets[:, None]
        if centres.size == 0:
            worst = min(worst, 2 * half); continue
        gaps = np.linalg.norm(probes[:, None, :] - centres[None, :, :], axis=2) - radii[None, :]
        free = np.all(gaps > ROBOT_R, axis=1)
        best = run = 0
        for ok in free:
            run = run + 1 if ok else 0
            best = max(best, run)
        worst = min(worst, best * step)
    return worst

def main():
    reg = json.load(open("/home/abc/temp/modern/snapshot/case_registry.json"))
    _, _, ActionXY, _, _ = _load_modules(DEFAULT_CROWDNAV)
    for scene_id, humans, gen, radius, width in (
            ("baseline_circle", 5, "circle_crossing", 4.0, None),
            ("dense_circle", 10, "circle_crossing", 4.0, None),
            ("dense_square", 20, "square_crossing", None, 10.0)):
        cases = reg["splits"][f"T_{scene_id}"]["cases"][:6]
        cfg = unicycle_config(point_config(2), horizon=16)
        env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", humans, gen, radius, width, 25, True)
        per_k = {k: [] for k in (0, 3, 7, 11, 15)}
        for case in cases:
            env.reset(options={"test_case": case})
            ad = MatchedAdapter("bayes", cfg.horizon, cfg.human_margin, cfg.dt,
                                cfg.chance_limit, cfg.fixed_uncertainty_radius,
                                cfg.acceleration_std, 0.0, 0.0, 1.0,
                                case * 1000003 + 7919,
                                cfg.conformal_visible_radii, cfg.conformal_hidden_radii)
            for _ in range(6):
                ad.read(env); env.step(ActionXY(0.0, 0.0))
            obs = ad.read(env)
            cov, ends = obs.human_position_covariance, obs.human_segment_end
            if cov is None or np.asarray(cov).size == 0 or ends is None:
                continue
            cov = np.asarray(cov); ends = np.asarray(ends)
            sig = np.sqrt(np.maximum(np.linalg.eigvalsh(cov)[..., 1], 0.0))
            for k in per_k:
                if k >= ends.shape[1]: continue
                c = ends[:, k, :2]
                r = HUMAN_R + MARGIN + CHI["tmpc"] * sig[:, k]
                per_k[k].append(free_lane(c, r, obs.robot_xy, obs.goal_xy))
        print(f"\n=== {scene_id}（{humans} 人）自由通道宽度（tmpc 的硬椭球，risk 0.35）===")
        print(f"{'k':>3s} {'预测时刻s':>9s} {'sigma_k m':>10s} {'强制半径m':>10s} {'自由通道m':>10s}")
        base = None
        for k in (0, 3, 7, 11, 15):
            v = [x for x in per_k[k] if np.isfinite(x)]
            if not v: continue
            s = float(np.median(sig[:, k])) if k < sig.shape[1] else float('nan')
            print(f"{k:3d} {(k+1)*0.25:9.2f} {s:10.3f} "
                  f"{HUMAN_R+MARGIN+CHI['tmpc']*s:10.2f} {np.median(v):10.2f}")
        print(f"  机器人需要 {ROBOT_R:.1f} m 半径；硬约束规划器必须在**所有 k 上同时**可行")

if __name__ == "__main__":
    main()
