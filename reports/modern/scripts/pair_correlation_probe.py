"""Is there any inter-pedestrian motion correlation to estimate here?

The proposed upgrade rests on Var(x_j - x_i) = Var(x_i) + Var(x_j) - 2Cov(x_i,x_j):
the cross term is what an independent per-track filter throws away.  But in this
system that cross term is not merely ignored, it is exactly zero by construction,
because each track is a constant-velocity Kalman filter driven by its own
independent acceleration noise.  Any joint posterior therefore has to *estimate*
the coupling from observed history.

So the question that decides the whole plan is empirical: do CrowdSim pedestrians,
which follow ORCA towards independent goals, actually produce correlated
prediction residuals?  This measures it directly -- for every pair of pedestrians
alive at the same time, the correlation between their constant-velocity
prediction errors at a fixed lookahead.
"""
from __future__ import annotations
import os, sys, itertools, collections
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
import numpy as np
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env, _load_modules

DT, LOOKAHEAD = 0.25, 8          # 2 s ahead, the horizon's middle
SCENES = [("baseline_circle", 5, "circle_crossing", 4.0, None),
          ("dense_circle", 10, "circle_crossing", 4.0, None),
          ("dense_square", 20, "square_crossing", None, 10.0)]


def episode_residuals(env, steps=60):
    """Constant-velocity residual of every pedestrian at a fixed lookahead."""
    _, _, ActionXY, _, _ = _load_modules(DEFAULT_CROWDNAV)
    track = []
    for _ in range(steps):
        track.append(np.array([[h.px, h.py, h.vx, h.vy] for h in env.humans]))
        _, _, term, trunc, _ = env.step(ActionXY(0.0, 0.0))
        if term or trunc:
            break
    track = np.stack(track)                       # (T, H, 4)
    T, H, _ = track.shape
    res = []
    for t in range(T - LOOKAHEAD):
        pred = track[t, :, :2] + track[t, :, 2:] * (LOOKAHEAD * DT)
        res.append(track[t + LOOKAHEAD, :, :2] - pred)
    return np.stack(res) if res else None         # (N, H, 2)


def main():
    print("行人之间的常速度预测残差相关性（前瞻 2 s）")
    print(f"{'场景':16s} {'样本对':>7s} {'|rho| 中位':>10s} {'|rho|>0.3 占比':>13s} "
          f"{'近邻(<2m) |rho| 中位':>20s}")
    reg_cases = list(range(12000, 12012))
    for scene_id, humans, gen, radius, width in SCENES:
        env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", humans, gen, radius, width, 25, True)
        rhos, near_rhos = [], []
        for case in reg_cases:
            env.reset(options={"test_case": case})
            r = episode_residuals(env)
            if r is None or r.shape[0] < 12:
                continue
            positions = np.array([[h.px, h.py] for h in env.humans])
            for i, j in itertools.combinations(range(r.shape[1]), 2):
                for d in (0, 1):                  # x and y components
                    a, b = r[:, i, d], r[:, j, d]
                    if a.std() < 1e-9 or b.std() < 1e-9:
                        continue
                    rho = float(np.corrcoef(a, b)[0, 1])
                    if not np.isfinite(rho):
                        continue
                    rhos.append(abs(rho))
                    if np.linalg.norm(positions[i] - positions[j]) < 2.0:
                        near_rhos.append(abs(rho))
        rhos = np.array(rhos)
        near = np.array(near_rhos) if near_rhos else np.array([np.nan])
        print(f"{scene_id:16s} {len(rhos):7d} {np.median(rhos):10.3f} "
              f"{100*np.mean(rhos > 0.3):12.1f}% {np.nanmedian(near):20.3f}")
    print("\n判据：若 |rho| 中位数接近 0 且高相关比例很低，则本基准里没有可估计的关联，")
    print("      Var(x_j-x_i) 的交叉项在这里恒等于噪声，联合滤波无信息可用。")


if __name__ == "__main__":
    main()
