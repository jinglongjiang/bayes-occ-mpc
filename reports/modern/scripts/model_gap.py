"""Quantify the one-cycle gap between the native model and the environment
(order section 13.3).

The comparators optimise a second-order unicycle:

    x' = v cos psi,  y' = v sin psi,  psi' = w,  v' = a

with state (x, y, psi, v, spline) and input (a, w), integrated over the horizon
by the generated solver.  The bridge then commands the pair the upstream
controller commands: the predicted speed at stage 1 and the input w at stage 0.

CrowdSim advances a non-holonomic agent differently:

    theta_1 = theta_0 + r,      p_1 = p_0 + v [cos theta_1, sin theta_1] dt

that is, it turns first and then translates at a constant speed along the new
heading.  The solver instead turns continuously while translating, and its speed
ramps from v_0 to v_1 under the acceleration a.

Both are legitimate discretisations of the same vehicle, but they are not the
same map, so one control period of the same command lands the robot in two
slightly different places.  The order requires that difference be measured, not
waved away with "the box constraints are the same".

Nothing here is a fix: the environment is the arbiter for every arm alike, and
the Bayesian planner's own rollout already matches the environment exactly.
What this establishes is how much prediction error the comparators carry from
the mismatch, so the main table can state it.
"""
from __future__ import annotations

import json
import numpy as np

DT = 0.25
V_MAX, A_MAX, W_MAX = 1.0, 2.0, 0.8


def native_step(state, control, dt=DT, substeps=400):
    """The continuous model the solver integrates, by fine RK4."""
    x, y, psi, v = state
    a, w = control
    h = dt / substeps

    def derivative(s):
        return np.array([s[3] * np.cos(s[2]), s[3] * np.sin(s[2]), w, a])

    s = np.array([x, y, psi, v], dtype=np.float64)
    for _ in range(substeps):
        k1 = derivative(s)
        k2 = derivative(s + 0.5 * h * k1)
        k3 = derivative(s + 0.5 * h * k2)
        k4 = derivative(s + h * k3)
        s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def environment_step(state, v_command, w, dt=DT):
    """CrowdSim's ActionRot: turn by the increment, then translate."""
    x, y, psi, _ = state
    psi_1 = psi + w * dt
    return np.array([x + v_command * np.cos(psi_1) * dt,
                     y + v_command * np.sin(psi_1) * dt, psi_1, v_command])


def main():
    rows = []
    for v0 in (0.0, 0.25, 0.5, 0.75, 1.0):
        for a in (-A_MAX, -1.0, 0.0, 1.0, A_MAX):
            for w in (-W_MAX, -0.4, 0.0, 0.4, W_MAX):
                v1 = float(np.clip(v0 + a * DT, 0.0, V_MAX))
                native = native_step((0.0, 0.0, 0.0, v0), (a, w))
                # The bridge commands the solver's stage-1 speed and stage-0 w.
                environment = environment_step((0.0, 0.0, 0.0, v0), v1, w)
                position_gap = float(np.hypot(*(native[:2] - environment[:2])))
                heading_gap = float(abs(native[2] - environment[2]))
                rows.append({"v0": v0, "a": a, "w": w, "v1": v1,
                             "position_gap_m": position_gap,
                             "heading_gap_rad": heading_gap})

    gaps = np.array([r["position_gap_m"] for r in rows])
    headings = np.array([r["heading_gap_rad"] for r in rows])
    worst = max(rows, key=lambda r: r["position_gap_m"])

    print("原生二阶模型 与 环境推进 的一周期偏差（同一命令，dt=0.25 s）")
    print(f"  样本 {len(rows)} 个（v0 × a × w 的注册取值网格）")
    print(f"  位置偏差  中位 {np.median(gaps)*1000:.2f} mm  "
          f"p95 {np.percentile(gaps,95)*1000:.2f} mm  最大 {gaps.max()*1000:.2f} mm")
    print(f"  朝向偏差  最大 {headings.max():.2e} rad")
    print(f"  最差点：v0={worst['v0']} a={worst['a']} w={worst['w']} "
          f"-> {worst['position_gap_m']*1000:.2f} mm")
    print(f"\n  对照：一步最大位移 {V_MAX*DT*1000:.0f} mm，"
          f"机器人半径 300 mm，成功半径 250 mm")
    print(f"  即最大偏差为一步位移的 {100*gaps.max()/(V_MAX*DT):.1f}%，"
          f"为成功半径的 {100*gaps.max()/0.25:.1f}%")
    print("\n  说明：朝向偏差为 0，两种离散化的转角完全一致；差异全部来自")
    print("  '边转边走'与'先转后走'的位置积分，以及速度在这一周期内是斜坡还是阶跃。")
    print("  环境对所有臂是同一个仲裁者，贝叶斯规划器的 rollout 与环境逐位一致，")
    print("  两个对照方法则带着这个量级的单周期预测误差。")

    payload = {"dt": DT, "samples": rows,
               "position_gap_median_m": float(np.median(gaps)),
               "position_gap_p95_m": float(np.percentile(gaps, 95)),
               "position_gap_max_m": float(gaps.max()),
               "heading_gap_max_rad": float(headings.max())}
    from pathlib import Path
    Path("/home/abc/temp/modern/snapshot/model_gap.json").write_text(
        json.dumps(payload, indent=1))
    print("\n-> snapshot/model_gap.json")


if __name__ == "__main__":
    main()
