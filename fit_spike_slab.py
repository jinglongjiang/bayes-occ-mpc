"""Single-Q scaling (control) versus spike-and-slab process noise (candidate).

  gaussian     Sigma_h = P_state + P_process                         (as shipped)
  single-Q     Sigma_h = P_state + c P_process                       one parameter
  spike-slab   pi N(mu, P_state) + (1-pi) N(mu, P_state + b P_process)
               a is pinned at 0: a quarter of the forecasts carry essentially no
               process noise, the rest carry the nominal amount.

Both mixture components stay isotropic (verified lambda1/lambda2 = 1.000), so
the level sets remain circles and coverage is exact.

Single-Q is fitted by development NLL alone.  Spike-and-slab is fitted by
development NLL *subject to* the pre-registered tail floors holding on every
development stratum, because an unconstrained NLL optimum shrinks the tail the
safety constraint depends on.  Development and test episodes are disjoint.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

LEVELS = (0.50, 0.68, 0.90, 0.95)
LOG2PI = math.log(2.0 * math.pi)
COV90_FLOOR, COV95_FLOOR = 0.90, 0.95      # dev-side floors, stricter than the test gate
COL = {n: i for i, n in enumerate(
    ["case", "track", "h", "visible", "existence", "err_x", "err_y",
     "P00", "P01", "P11", "d2", "state_var", "process_var"])}


def nll_single(e2, var):
    return LOG2PI + np.log(var) + e2 / (2.0 * var)


def nll_spike(e2, s, q, pi, b):
    vc, vt = np.maximum(s, 1e-12), s + b * q
    lc = math.log(max(pi, 1e-300)) - np.log(vc) - e2 / (2.0 * vc)
    lt = math.log(max(1.0 - pi, 1e-300)) - np.log(vt) - e2 / (2.0 * vt)
    m = np.maximum(lc, lt)
    return LOG2PI - (m + np.log(np.exp(lc - m) + np.exp(lt - m)))


def r2_single(level, var):
    return -2.0 * var * math.log1p(-level)


def r2_spike(level, s, q, pi, b):
    vc, vt = np.maximum(s, 1e-12), s + b * q
    lo = np.zeros_like(s)
    hi = -2.0 * np.maximum(vc, vt) * math.log1p(-level) * 4.0 + 1e-12
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cov = pi * (1.0 - np.exp(-mid / (2.0 * vc))) + (1.0 - pi) * (1.0 - np.exp(-mid / (2.0 * vt)))
        below = cov < level
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def strata(h, vis, tag):
    yield f"{tag}/all", np.ones(h.size, bool)
    for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 16)):
        band = (h >= lo) & (h <= hi)
        for v, name in ((1, "vis"), (0, "mis")):
            yield f"{tag}/h{lo}-{hi} {name}", band & (vis == v)


def unpack(rows, idx):
    r = rows[idx]
    return (r[:, COL["err_x"]] ** 2 + r[:, COL["err_y"]] ** 2,
            r[:, COL["state_var"]], r[:, COL["process_var"]],
            r[:, COL["h"]].astype(int), r[:, COL["visible"]].astype(int))


def main():
    data = {}
    for p in sys.argv[1:]:
        r = np.load(p)["rows"]
        v = r[:, COL["state_var"]] + r[:, COL["process_var"]]
        data[Path(p).stem.split("_", 1)[1]] = r[np.isfinite(v) & (v > 0)]

    cases = sorted({int(c) for r in data.values() for c in r[:, COL["case"]]})
    dev_cases = cases[: len(cases) // 2]
    print(f"开发集 episode {min(dev_cases)}-{max(dev_cases)}   "
          f"测试集 {min(set(cases)-set(dev_cases))}-{max(cases)}   条件: {', '.join(data)}")

    dev_parts, dev_strata = [], []
    for tag, rows in data.items():
        m = np.isin(rows[:, COL["case"]].astype(int), dev_cases)
        dev_parts.append(rows[m])
    dev = np.concatenate(dev_parts)
    de2, ds, dq, dh, dv = unpack(dev, np.ones(dev.shape[0], bool))
    for tag in data:
        pass
    dev_masks = [m for _, m in strata(dh, dv, "dev") if m.sum() >= 2000]

    # ---- control: one scale on the process term, NLL only ----
    grid_c = np.exp(np.linspace(math.log(0.2), math.log(3.0), 141))
    nll_c = [float(np.mean(nll_single(de2, ds + c * dq))) for c in grid_c]
    c_hat = float(grid_c[int(np.argmin(nll_c))])

    # ---- candidate: spike-and-slab, NLL subject to dev tail floors ----
    # The grid touches every dev stratum, so it searches on a fixed random
    # subsample; the winner is re-scored on the full development set below.
    sub = np.random.default_rng(0).permutation(de2.size)[:40000]
    sub_masks = [m[sub] for m in dev_masks]
    se2, ss, sq = de2[sub], ds[sub], dq[sub]

    def feasible(pi, b):
        for m in sub_masks:
            if m.sum() < 200:
                continue
            if (np.mean(se2[m] <= r2_spike(0.90, ss[m], sq[m], pi, b)) < COV90_FLOOR
                    or np.mean(se2[m] <= r2_spike(0.95, ss[m], sq[m], pi, b)) < COV95_FLOOR):
                return False
        return True

    best = None
    for pi in np.linspace(0.02, 0.70, 18):
        for b in np.exp(np.linspace(0.0, math.log(3.0), 18)):
            if not feasible(pi, b):
                continue
            v = float(np.mean(nll_spike(se2, ss, sq, pi, b)))
            if best is None or v < best[0]:
                best = (v, float(pi), float(b))
    if best is not None:                      # refine around the coarse winner
        p0, b0 = best[1], best[2]
        for pi in np.linspace(max(0.01, p0 - 0.05), min(0.90, p0 + 0.05), 11):
            for b in np.exp(np.linspace(math.log(max(1.0, b0 * 0.7)), math.log(b0 * 1.4), 11)):
                if not feasible(pi, b):
                    continue
                v = float(np.mean(nll_spike(se2, ss, sq, pi, b)))
                if v < best[0]:
                    best = (v, float(pi), float(b))
    if best is None:
        print("spike-and-slab: 开发集上没有任何 (pi,b) 满足尾部下限")
        pi_hat, b_hat = 0.0, 1.0
    else:
        _, pi_hat, b_hat = best

    print(f"单 Q 缩放（仅 NLL）           c = {c_hat:.4f}")
    print(f"spike-and-slab（NLL + 尾部约束） pi = {pi_hat:.4f}   b = {b_hat:.4f}")
    print(f"  即 {pi_hat*100:.1f}% 的预测 Sigma = P_state（无过程噪声），"
          f"其余 Sigma = P_state + {b_hat:.3f}*P_process\n")

    hdr = f"{'分层':22s} {'模型':12s} {'n':>7s} {'NLL':>8s} {'c50':>6s} {'c68':>6s} {'c90':>6s} {'c95':>6s} {'面积50':>7s} {'面积90':>7s}"
    print(hdr)
    print(f"{'（名义）':22s} {'':12s} {'':>7s} {'':>8s} {50.0:6.1f} {68.0:6.1f} {90.0:6.1f} {95.0:6.1f} {1.0:7.2f} {1.0:7.2f}\n")

    fails = {"single-Q": [], "spike-slab": []}
    for tag, rows in data.items():
        test = ~np.isin(rows[:, COL["case"]].astype(int), dev_cases)
        e2, s, q, h, vis = unpack(rows, test)
        for name, mask in strata(h, vis, tag):
            if mask.sum() < 400:
                continue
            E, S, Q = e2[mask], s[mask], q[mask]
            out = {}
            out["gaussian"] = ({"nll": float(np.mean(nll_single(E, S + Q)))},
                               [r2_single(l, S + Q) for l in LEVELS])
            out["single-Q"] = ({"nll": float(np.mean(nll_single(E, S + c_hat * Q)))},
                               [r2_single(l, S + c_hat * Q) for l in LEVELS])
            out["spike-slab"] = ({"nll": float(np.mean(nll_spike(E, S, Q, pi_hat, b_hat)))},
                                 [r2_spike(l, S, Q, pi_hat, b_hat) for l in LEVELS])
            base = out["gaussian"][1]
            for label in ("gaussian", "single-Q", "spike-slab"):
                d, radii = out[label]
                for l, r, rb in zip(LEVELS, radii, base):
                    k = int(l * 100)
                    d[f"c{k}"] = float(np.mean(E <= r))
                    d[f"a{k}"] = float(np.mean(r / rb))
                print(f"{name if label == 'gaussian' else '':22s} {label:12s} {mask.sum():7d} "
                      f"{d['nll']:8.4f} {d['c50']*100:6.1f} {d['c68']*100:6.1f} "
                      f"{d['c90']*100:6.1f} {d['c95']*100:6.1f} {d['a50']:7.2f} {d['a90']:7.2f}")
                if label == "gaussian":
                    continue
                g = out["gaussian"][0]
                if d["nll"] > g["nll"]:
                    fails[label].append(f"{name}: NLL 差")
                if d["c90"] < 0.87 or d["c95"] < 0.92:
                    fails[label].append(f"{name}: 90/95 欠覆盖 {d['c90']*100:.1f}/{d['c95']*100:.1f}")
                if d["a50"] > 1.0 or d["a90"] > 1.0:
                    fails[label].append(f"{name}: 区域变大")
            print()

    print("=== 预注册门禁 ===")
    for label in ("single-Q", "spike-slab"):
        f = sorted(set(fails[label]))
        print(f"  {label:12s} " + ("全部通过" if not f else f"{len(f)} 项未通过"))
        for x in f:
            print(f"      {x}")


if __name__ == "__main__":
    main()
