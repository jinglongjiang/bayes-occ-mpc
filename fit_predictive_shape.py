"""Offline predictive-shape comparison on recorded forecast residuals.

The filter's covariance recursion P_h = F P_{h-1} F^T + Q is linear, so the
predictive covariance separates exactly into two parts:

    P_h = P_h^state + P_h^process
        = F^h P_0 (F^h)^T  +  sum_{k<h} F^k Q (F^k)^T

The first is irreducible uncertainty about where the pedestrian is now; the
second is uncertainty about what they will do next.  Only the second should
carry a heavy tail, so the mixture scales the process term alone:

    core:  Sigma = P^state + a P^process     a < 1
    tail:  Sigma = P^state + b P^process     b > 1
    p(x)   = pi N(mu, Sigma_core) + (1 - pi) N(mu, Sigma_tail)

Scaling the total covariance instead shrinks measurement uncertainty that is
genuinely there, which is what broke the previous version at short horizons.

Both blocks are exactly isotropic here (F and Q are x/y symmetric and the
measurement noise is isotropic; verified lambda1/lambda2 = 1.000), so every
quantity reduces to scalars and the mixture's level sets remain circles.

Protocol: development and test episodes are disjoint (splitting at the forecast
level would leak, since one episode's forecasts are strongly correlated); one
global (pi, a, b) is fitted on development episodes pooled across noise
conditions and then scored unchanged on every test stratum.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

LEVELS = (0.50, 0.68, 0.90, 0.95)
LOG2PI = math.log(2.0 * math.pi)
COL = {n: i for i, n in enumerate(
    ["case", "track", "h", "visible", "existence", "err_x", "err_y",
     "P00", "P01", "P11", "d2", "state_var", "process_var"])}


def nll_single(e2, var):
    """2-D isotropic Gaussian: |Sigma| = var^2, so 0.5 log|Sigma| = log var."""
    return LOG2PI + np.log(var) + e2 / (2.0 * var)


def nll_mixture(e2, s, q, pi, a, b):
    vc, vt = s + a * q, s + b * q
    lc = math.log(pi) - np.log(vc) - e2 / (2.0 * vc)
    lt = math.log1p(-pi) - np.log(vt) - e2 / (2.0 * vt)
    m = np.maximum(lc, lt)
    return LOG2PI - (m + np.log(np.exp(lc - m) + np.exp(lt - m)))


def radius2_single(level, var):
    return -2.0 * var * math.log1p(-level)


def radius2_mixture(level, s, q, pi, a, b):
    """Per-sample squared radius of the mixture's own level set (a circle)."""
    vc, vt = s + a * q, s + b * q
    lo = np.zeros_like(s)
    hi = -2.0 * np.maximum(vc, vt) * math.log1p(-level) * 4.0 + 1e-12
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cov = pi * (1.0 - np.exp(-mid / (2.0 * vc))) + (1.0 - pi) * (1.0 - np.exp(-mid / (2.0 * vt)))
        below = cov < level
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def fit(e2, s, q):
    def objective(z):
        pi = 1.0 / (1.0 + math.exp(-z[0]))
        # Enforce a < 1 < b: the core must be sharper and the tail heavier than
        # nominal.  Without b > 1 the optimiser makes both components narrower
        # than P_h, which lowers NLL but collapses the 90/95 coverage the safety
        # constraint depends on.
        a = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z[1]))))
        b = 1.0 + math.exp(max(-700.0, min(30.0, z[2])))
        return float(np.mean(nll_mixture(e2, s, q, pi, a, b)))
    best = min((minimize(objective, x0, method="Nelder-Mead",
                         options={"maxiter": 8000, "xatol": 1e-7, "fatol": 1e-9})
                for x0 in ([1.0, -1.5, -0.7], [0.0, -3.0, 0.0], [2.0, -0.8, 0.7],
                           [-0.5, -5.0, -1.5], [0.5, -2.0, 1.2])), key=lambda r: r.fun)
    return (1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, best.x[0])))),
            1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, best.x[1])))),
            1.0 + math.exp(max(-700.0, min(30.0, best.x[2]))))


def strata(h, vis, tag):
    yield f"{tag}/all", np.ones(h.size, bool)
    for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 16)):
        band = (h >= lo) & (h <= hi)
        for v, name in ((1, "vis"), (0, "mis")):
            yield f"{tag}/h{lo}-{hi} {name}", band & (vis == v)


def main():
    data = {}
    for p in sys.argv[1:]:
        r = np.load(p)["rows"]
        e2 = r[:, COL["err_x"]] ** 2 + r[:, COL["err_y"]] ** 2
        var = r[:, COL["state_var"]] + r[:, COL["process_var"]]
        keep = np.isfinite(e2) & np.isfinite(var) & (var > 0)
        data[Path(p).stem.replace("resid2_", "").replace("resid_", "")] = r[keep]

    cases = sorted({int(c) for r in data.values() for c in r[:, COL["case"]]})
    dev_cases = sorted(cases)[: len(cases) // 2]
    test_cases = sorted(set(cases) - set(dev_cases))
    print(f"开发集 episode {min(dev_cases)}-{max(dev_cases)} ({len(dev_cases)} 个)   "
          f"测试集 episode {min(test_cases)}-{max(test_cases)} ({len(test_cases)} 个)   "
          f"条件: {', '.join(data)}")

    dev = np.concatenate([r[np.isin(r[:, COL["case"]].astype(int), dev_cases)]
                          for r in data.values()])
    de2 = dev[:, COL["err_x"]] ** 2 + dev[:, COL["err_y"]] ** 2
    ds, dq = dev[:, COL["state_var"]], dev[:, COL["process_var"]]
    pi, a, b = fit(de2, ds, dq)
    print(f"全局参数（只用开发集，两种噪声合并，n={dev.shape[0]}）")
    print(f"  process-scale mixture   pi={pi:.4f}   a={a:.4f}   b={b:.4f}")
    print(f"  即  core = P_state + {a:.3f}*P_process     tail = P_state + {b:.3f}*P_process\n")

    print(f"{'分层':22s} {'模型':12s} {'n':>8s} {'NLL':>8s} "
          f"{'cov50':>6s} {'cov68':>6s} {'cov90':>6s} {'cov95':>6s} {'面积50':>7s} {'面积90':>7s}")
    print(f"{'（名义）':22s} {'':12s} {'':>8s} {'':>8s} {50.0:6.1f} {68.0:6.1f} "
          f"{90.0:6.1f} {95.0:6.1f} {1.0:7.2f} {1.0:7.2f}\n")

    fails = []
    for tag, rows in data.items():
        test = rows[~np.isin(rows[:, COL["case"]].astype(int), dev_cases)]
        e2 = test[:, COL["err_x"]] ** 2 + test[:, COL["err_y"]] ** 2
        s, q = test[:, COL["state_var"]], test[:, COL["process_var"]]
        h = test[:, COL["h"]].astype(int)
        vis = test[:, COL["visible"]].astype(int)
        for name, mask in strata(h, vis, tag):
            if mask.sum() < 400:
                continue
            E, S, Q = e2[mask], s[mask], q[mask]
            V = S + Q
            rows_out = []
            g = {"nll": float(np.mean(nll_single(E, V)))}
            m = {"nll": float(np.mean(nll_mixture(E, S, Q, pi, a, b)))}
            for lv in LEVELS:
                rg = radius2_single(lv, V)
                rm = radius2_mixture(lv, S, Q, pi, a, b)
                k = int(lv * 100)
                g[f"cov{k}"] = float(np.mean(E <= rg))
                m[f"cov{k}"] = float(np.mean(E <= rm))
                g[f"area{k}"] = 1.0
                m[f"area{k}"] = float(np.mean(rm / rg))     # area ∝ radius²
            for label, r in (("gaussian", g), ("process-mix", m)):
                print(f"{name if label == 'gaussian' else '':22s} {label:12s} {mask.sum():8d} "
                      f"{r['nll']:8.4f} {r['cov50']*100:6.1f} {r['cov68']*100:6.1f} "
                      f"{r['cov90']*100:6.1f} {r['cov95']*100:6.1f} "
                      f"{r['area50']:7.2f} {r['area90']:7.2f}")
            if m["nll"] > g["nll"]:
                fails.append(f"{name}: NLL 比高斯差 ({m['nll']:.4f} > {g['nll']:.4f})")
            if m["cov90"] < 0.87 or m["cov95"] < 0.92:
                fails.append(f"{name}: 90/95 欠覆盖 {m['cov90']*100:.1f}/{m['cov95']*100:.1f}")
            if m["area50"] > 1.0 or m["area90"] > 1.0:
                fails.append(f"{name}: 区域比高斯大 {m['area50']:.2f}/{m['area90']:.2f}")
            print()

    print("=== 分层诊断拟合（只看参数跨条件稳不稳，不用于打分）===")
    for tag, rows in data.items():
        dv = np.isin(rows[:, COL["case"]].astype(int), dev_cases)
        e2 = rows[:, COL["err_x"]] ** 2 + rows[:, COL["err_y"]] ** 2
        s, q = rows[:, COL["state_var"]], rows[:, COL["process_var"]]
        h = rows[:, COL["h"]].astype(int)
        vis = rows[:, COL["visible"]].astype(int)
        for name, mask in strata(h, vis, tag):
            sel = mask & dv
            if sel.sum() < 2000:
                continue
            p2, a2, b2 = fit(e2[sel], s[sel], q[sel])
            print(f"  {name:22s} pi={p2:.3f}  a={a2:.3f}  b={b2:.3f}")

    print("\n=== 预注册门禁 ===")
    print("全部通过" if not fails else "未通过:\n  " + "\n  ".join(sorted(set(fails))))


if __name__ == "__main__":
    main()
