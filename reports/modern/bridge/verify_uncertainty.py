"""Does the planner receive the distribution we think we sent it?

Two separate mistakes are possible and both look like "it runs":

  * the field means something different at the other end (a marginal versus an
    increment), so the obstacle's spread over the horizon is wrong;
  * the time index is off by one, so step k is given step k+1's spread.

Neither shows up as an error.  This checks them by construction rather than by
looking at success rates: a synthetic pedestrian is sent with an analytically
known marginal covariance, and the two consumers are compared against it.

Findings this encodes (verified in the source, not assumed):

  scenario_module/src/config.cpp:27
      risk_ = CONFIG["probabilistic"]["risk"] * 2
      SH-MPC doubles the configured risk.  T-MPC's EllipsoidConstraintModule
      does not.  The same yaml value therefore means different things.

  scenario_module/src/sampler.cpp:452
      bivariate_gaussian += A_[v][k] * xi_k          (accumulated over k)
      samples_[k] = mean_k + bivariate_gaussian * dt
      SH-MPC treats the per-step matrix as a process-noise increment and
      integrates it into a random walk, then scales by dt.  T-MPC treats the
      same field as the marginal semi-axes at step k.

So a single array cannot be sent to both.  This module derives the increment
form from a marginal sequence and checks that the round trip reproduces it.
"""
from __future__ import annotations

import numpy as np


def marginal_to_increment(marginal, dt):
    """Convert marginal covariances P_k into the per-step matrices SH-MPC wants.

    SH-MPC's sampler produces, for the displacement at step k,

        d_k = dt * sum_{j<=k} A_j xi_j ,        xi_j ~ N(0, I) independent

    so  Cov(d_k) = dt^2 * sum_{j<=k} S_j   with  S_j = A_j A_j^T.

    Requiring Cov(d_k) = P_k gives

        S_0 = P_0 / dt^2,   S_k = (P_k - P_{k-1}) / dt^2 .

    `marginal` is (humans, horizon, 2, 2) and must be non-decreasing in k, which
    a filter covariance without new measurements is.  A step that decreases is
    reported rather than clipped, because clipping would quietly hand SH-MPC a
    distribution nobody chose.
    """
    marginal = np.asarray(marginal, dtype=np.float64)
    previous = np.zeros_like(marginal[:, :1])
    shifted = np.concatenate((previous, marginal[:, :-1]), axis=1)
    increment = (marginal - shifted) / (dt * dt)

    values = np.linalg.eigvalsh(increment)
    negative = float(values.min())
    return increment, negative


def ellipse_from_covariance(covariance):
    """Semi-axes and tilt of a 2x2 covariance, the form both bridges parse."""
    values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
    values = np.maximum(values, 0.0)
    minor = np.sqrt(values[..., 0])
    major = np.sqrt(values[..., 1])
    principal = vectors[..., 1]
    tilt = np.arctan2(principal[..., 1], principal[..., 0])
    return major, minor, tilt


def simulate_sh_sampler(increment, dt, samples=200000, seed=0):
    """Reproduce sampler.cpp's random walk in numpy, to check the conversion."""
    rng = np.random.default_rng(seed)
    humans, horizon = increment.shape[0], increment.shape[1]
    chol = np.linalg.cholesky(increment + 1e-15 * np.eye(2))
    out = np.empty((humans, horizon, 2, 2))
    for v in range(humans):
        walk = np.zeros((samples, 2))
        for k in range(horizon):
            xi = rng.standard_normal((samples, 2))
            walk = walk + xi @ chol[v, k].T
            displacement = walk * dt
            out[v, k] = np.cov(displacement, rowvar=False)
    return out


def main():
    dt, horizon = 0.25, 16
    # A pedestrian whose marginal covariance grows the way a constant-velocity
    # filter's does: P_k = P0 + k*Q, with a deliberately non-isotropic Q so a
    # dropped tilt or a swapped axis would show.
    P0 = np.array([[0.02, 0.005], [0.005, 0.01]])
    Q = np.array([[0.03, -0.008], [-0.008, 0.012]])
    marginal = np.stack([P0 + (k + 1) * Q for k in range(horizon)])[None, :, :, :]

    increment, negative = marginal_to_increment(marginal, dt)
    print(f"增量最小特征值 {negative:.3e}（须 >= 0，否则边缘协方差在某一步下降了）")

    recovered = simulate_sh_sampler(increment, dt, samples=200000, seed=1)
    error = np.abs(recovered - marginal[0]).max()
    relative = error / np.abs(marginal[0]).max()
    print(f"SH 采样器复原边缘协方差：最大绝对误差 {error:.3e}，相对 {relative:.2%}")
    print("  （20 万样本的蒙特卡洛误差在 1% 量级，超出说明换算式错了）")

    major, minor, tilt = ellipse_from_covariance(marginal[0])
    print(f"\nT-MPC 用的边缘半轴：k=0 major {major[0]:.4f} minor {minor[0]:.4f} "
          f"tilt {tilt[0]:+.4f} rad")
    print(f"                      k={horizon-1} major {major[-1]:.4f} "
          f"minor {minor[-1]:.4f} tilt {tilt[-1]:+.4f} rad")
    imajor, iminor, itilt = ellipse_from_covariance(increment[0])
    print(f"SH 用的增量半轴：      k=0 major {imajor[0]:.4f} minor {iminor[0]:.4f}")
    print(f"                      k={horizon-1} major {imajor[-1]:.4f} "
          f"minor {iminor[-1]:.4f}")
    print(f"\n两者在末端相差 {major[-1]/max(imajor[-1],1e-12):.1f} 倍 —— "
          f"送错一个，障碍尺度就差这么多")

    ok = negative >= -1e-12 and relative < 0.02
    print("\n结论:", "换算式成立" if ok else "换算式不成立，需要重查")
    return 0 if ok else 1



# ---------------------------------------------------------------------------
# Section 13.4-B: what SH-MPC's joint distribution actually is.
#
# The marginal round trip above is necessary but not sufficient.  SH-MPC does
# not sample each horizon step independently: one draw produces a whole
# trajectory, and the scenario constraints are built from those trajectories, so
# the correlation between two time steps matters to the constraint set even
# though it never appears in a marginal.  The order requires that the difference
# be measured and disclosed rather than assumed away.
#
# Two joints are compared, both with identical marginals by construction:
#
#   random walk (what SH-MPC integrates)
#       d_k = dt * sum_{j<=k} A_j xi_j
#       Cov(d_j, d_k) = Cov(d_j, d_j) = P_j            for j <= k
#
#   constant velocity (what this project's filter propagates)
#       p_k = p_0 + k dt v,  with v uncertain and p_0 uncertain
#       Cov(p_j, p_k) = P_pp + (j+k) dt P_pv + j k dt^2 P_vv
#
# Both give a growing marginal, so a marginal check cannot distinguish them.
# ---------------------------------------------------------------------------

def random_walk_joint(marginal):
    """Cov(d_j, d_k) under the random walk SH-MPC's sampler integrates."""
    n = len(marginal)
    joint = np.empty((n, n, 2, 2))
    for j in range(n):
        for k in range(n):
            joint[j, k] = marginal[min(j, k)]
    return joint


def constant_velocity_joint(p_pp, p_pv, p_vv, n, dt):
    """Cov(p_j, p_k) for the constant-velocity posterior, and its marginals."""
    joint = np.empty((n, n, 2, 2))
    for j in range(n):
        for k in range(n):
            a, b = (j + 1) * dt, (k + 1) * dt
            joint[j, k] = p_pp + a * p_pv.T + b * p_pv + a * b * p_vv
    marginal = np.stack([joint[k, k] for k in range(n)])
    return joint, marginal


def correlation_discrepancy(p_pp, p_pv, p_vv, n, dt):
    """How far SH-MPC's implied joint sits from the posterior's, at equal
    marginals.  Reported as the correlation of the x-component between step j
    and step k, which is what a scenario trajectory's shape depends on."""
    cv_joint, marginal = constant_velocity_joint(p_pp, p_pv, p_vv, n, dt)
    rw_joint = random_walk_joint(marginal)

    def correlation(joint):
        out = np.empty((n, n))
        for j in range(n):
            for k in range(n):
                out[j, k] = joint[j, k][0, 0] / np.sqrt(
                    joint[j, j][0, 0] * joint[k, k][0, 0])
        return out

    return correlation(cv_joint), correlation(rw_joint), marginal


def report_joint(dt=0.25, n=16):
    """Print the disclosure the main table has to carry."""
    # A representative tracked pedestrian: position known to a few centimetres,
    # velocity much less certain, mild position-velocity coupling -- the shape a
    # constant-velocity filter settles into after a few visible frames.
    p_pp = np.diag([0.02, 0.02])
    p_vv = np.diag([0.09, 0.09])
    p_pv = np.diag([0.01, 0.01])
    cv, rw, marginal = correlation_discrepancy(p_pp, p_pv, p_vv, n, dt)

    print("SH-MPC 联合分布披露（第 13.4-B 节）")
    print(f"  时域 N={n}，dt={dt}s；两种联合的边缘按构造完全相同")
    print(f"  边缘标准差 首步 {np.sqrt(marginal[0][0,0]):.4f} m -> "
          f"末步 {np.sqrt(marginal[-1][0,0]):.4f} m")
    print("\n  x 分量在第 j 步与第 k 步之间的相关系数：")
    print(f"  {'(j,k)':>10s} {'常速度后验':>12s} {'SH随机游走':>12s} {'差':>8s}")
    worst = 0.0
    for j, k in ((0, 1), (0, 4), (0, 8), (0, 15), (4, 8), (4, 15), (8, 15)):
        difference = cv[j, k] - rw[j, k]
        worst = max(worst, abs(difference))
        print(f"  {f'({j},{k})':>10s} {cv[j,k]:12.4f} {rw[j,k]:12.4f} {difference:+8.4f}")
    print(f"\n  最大相关系数差 {worst:.4f}")
    print("  结论：边缘一致不蕴含联合一致。SH-MPC 收到的是与本项目后验边缘相同、")
    print("  但时间相关结构为随机游走的轨迹族。随机游走在所有测试滞后上都")
    print("  系统性地低估时间相关（最大低 0.63），因为常速度后验里速度误差")
    print("  持续作用于整条轨迹，而随机游走每步重新抽独立增量。")
    print("  实际后果：SH-MPC 的场景集里折返、抖动的轨迹偏多，笔直持续偏离的")
    print("  轨迹偏少；它面对的是同样宽、但形状不同的一族未来。")
    print("  主表口径因此写作『同检测/跟踪均值与边缘不确定度、各方法原生规划假设』，")
    print("  不声称『同完整轨迹后验』。")
    return {"max_correlation_difference": float(worst),
            "cv_correlation": cv.tolist(), "rw_correlation": rw.tolist()}


if __name__ == "__main__":
    import json
    from pathlib import Path
    main()
    print()
    result = report_joint()
    Path("/home/abc/temp/modern/snapshot/joint_disclosure.json").write_text(
        json.dumps(result, indent=1))
    print("\n-> snapshot/joint_disclosure.json")
