"""Two questions the |rho| number cannot answer, and both decide the proposal.

(1) Sign.  The gap variance is Var_i + Var_j - 2Cov along the line joining the
    pair.  Positive correlation shrinks it (gaps hold), negative correlation
    inflates it (gaps close).  Which one this benchmark produces decides whether
    the independent model is over- or under-confident about gaps.

(2) Systematic or stochastic.  A residual correlation can come from a bias every
    pedestrian shares at that instant -- everyone slows near the centre of a
    circle crossing -- rather than from pairwise coupling.  If it is the shared
    bias, the missing information is in the *mean* motion model and a joint
    covariance is the wrong repair.  Removing the per-timestep cross-sectional
    mean separates the two.
"""
from __future__ import annotations
import os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
import numpy as np
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env, _load_modules
DT, LOOK = 0.25, 8

def residuals(env, steps=60):
    _, _, ActionXY, _, _ = _load_modules(DEFAULT_CROWDNAV)
    tr = []
    for _ in range(steps):
        tr.append(np.array([[h.px, h.py, h.vx, h.vy] for h in env.humans]))
        _, _, a, b, _ = env.step(ActionXY(0.0, 0.0))
        if a or b: break
    tr = np.stack(tr); T = tr.shape[0]
    out = [tr[t+LOOK, :, :2] - (tr[t, :, :2] + tr[t, :, 2:] * (LOOK*DT))
           for t in range(T-LOOK)]
    return (np.stack(out), tr) if out else (None, None)

def main():
    print(f"{'场景':16s} {'沿连线 rho 中位':>15s} {'去掉共同偏置后':>15s} {'相关性来自共同偏置':>18s}")
    for scene, n, gen, r_, w_ in (("baseline_circle",5,"circle_crossing",4.0,None),
                                  ("dense_circle",10,"circle_crossing",4.0,None),
                                  ("dense_square",20,"square_crossing",None,10.0)):
        env, _, _ = build_env(DEFAULT_CROWDNAV,"bayes",n,gen,r_,w_,25,True)
        raw, dem = [], []
        for case in range(12000, 12012):
            env.reset(options={"test_case": case})
            res, tr = residuals(env)
            if res is None or res.shape[0] < 12: continue
            centred = res - res.mean(axis=1, keepdims=True)   # drop shared bias
            for i, j in itertools.combinations(range(res.shape[1]), 2):
                axis = tr[0, j, :2] - tr[0, i, :2]
                nrm = np.linalg.norm(axis)
                if nrm < 1e-6: continue
                u = axis / nrm
                for arr, bag in ((res, raw), (centred, dem)):
                    a, b = arr[:, i, :] @ u, arr[:, j, :] @ u
                    if a.std() < 1e-9 or b.std() < 1e-9: continue
                    rho = float(np.corrcoef(a, b)[0, 1])
                    if np.isfinite(rho): bag.append(rho)
        raw, dem = np.array(raw), np.array(dem)
        drop = 100*(1 - abs(np.median(dem))/max(abs(np.median(raw)), 1e-9))
        print(f"{scene:16s} {np.median(raw):15.3f} {np.median(dem):15.3f} {drop:17.0f}%")
    print("\n沿连线 rho > 0 -> 间隙比独立假设更稳定（当前模型过度保守）")
    print("沿连线 rho < 0 -> 间隙比独立假设更易关闭（当前模型过度自信）")

if __name__ == "__main__":
    main()
