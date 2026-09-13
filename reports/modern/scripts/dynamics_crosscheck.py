"""Does the model asymmetry drive the comparators' low scores? (13.3 acceptance)

The registered one-cycle gap is up to 67 mm: the comparators predict with a
continuous second-order unicycle, while CrowdSim turns first and then translates,
and the Bayesian planner's rollout matches CrowdSim exactly.  Disclosing that
asymmetry is not the same as knowing whether it explains the gap in scores, so
this measures it three ways.

Part 1 decomposes the one-cycle gap into its two causes -- integrating the
heading along an arc instead of using the post-turn heading, and letting the
speed ramp under acceleration instead of stepping to its end value -- so the
next part can reproduce the one that matters.

Part 2 compounds it over a horizon: a 67 mm one-step error is only interesting
if it accumulates, and a controller replans every step, so the relevant quantity
is how far the horizon-16 prediction has drifted by its end.

Part 3 is the empirical control the acceptance actually needs.  The Bayesian
planner is given the comparators' mismatch -- it keeps every cost, the same
risk model, the same tracker, and only rolls out with the continuous model
while the environment still turns-then-translates -- and is re-scored on the
development layouts.  If its results collapse towards the comparators', the
asymmetry is a dominant cause and the main table cannot be read as a comparison
of belief handling.  If they barely move, it is a disclosed but minor term.

This is a diagnostic.  The handicapped planner is not an arm in the formal
matrix, and no arm's dynamics are changed by running it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, "/home/abc/temp/modern")

import numpy as np                                                   # noqa: E402

DT = 0.25
V_MAX, A_MAX, W_MAX = 1.0, 2.0, 0.8


# ----------------------------------------------------------------- part 1

def arc_displacement(psi, speed, omega, dt=DT):
    """Exact displacement when the heading turns at a constant rate while the
    vehicle translates -- the comparators' assumption."""
    if abs(omega) < 1e-12:
        return np.array([speed * np.cos(psi) * dt, speed * np.sin(psi) * dt])
    return np.array([
        speed / omega * (np.sin(psi + omega * dt) - np.sin(psi)),
        -speed / omega * (np.cos(psi + omega * dt) - np.cos(psi))])


def chord_displacement(psi, speed, omega, dt=DT):
    """CrowdSim: turn by the whole increment, then translate at constant speed."""
    psi_1 = psi + omega * dt
    return np.array([speed * np.cos(psi_1) * dt, speed * np.sin(psi_1) * dt])


def decompose():
    """Split the one-cycle gap into the heading term and the speed-ramp term."""
    heading_only, ramp_only, both = [], [], []
    for v0 in (0.0, 0.25, 0.5, 0.75, 1.0):
        for a in (-A_MAX, -1.0, 0.0, 1.0, A_MAX):
            for w in (-W_MAX, -0.4, 0.0, 0.4, W_MAX):
                v1 = float(np.clip(v0 + a * DT, 0.0, V_MAX))
                # heading term: same (end-of-interval) speed, arc vs chord
                heading_only.append(np.linalg.norm(
                    arc_displacement(0.0, v1, w) - chord_displacement(0.0, v1, w)))
                # ramp term: same chord geometry, ramped mean speed vs end speed
                mean_speed = 0.5 * (v0 + v1)
                ramp_only.append(np.linalg.norm(
                    chord_displacement(0.0, mean_speed, w)
                    - chord_displacement(0.0, v1, w)))
                # both, integrating the ramp along the arc
                steps = 200
                h = DT / steps
                position, psi, speed = np.zeros(2), 0.0, v0
                for _ in range(steps):
                    position = position + speed * np.array([np.cos(psi), np.sin(psi)]) * h
                    psi += w * h
                    speed = float(np.clip(speed + a * h, 0.0, V_MAX))
                both.append(np.linalg.norm(position - chord_displacement(0.0, v1, w)))
    return (np.array(heading_only), np.array(ramp_only), np.array(both))


# ----------------------------------------------------------------- part 2

def horizon_drift(horizon=16, trials=2000, seed=11, include_ramp=False):
    """How far apart the two integrations are after a whole horizon, under the
    same control sequence -- the quantity a planner's prediction actually uses."""
    rng = np.random.default_rng(seed)
    drift = np.empty(trials)
    for t in range(trials):
        speeds = rng.uniform(0.0, V_MAX, horizon)
        omegas = rng.uniform(-W_MAX, W_MAX, horizon)
        arc = np.zeros(2); psi_a = 0.0
        chord = np.zeros(2); psi_c = 0.0
        previous = 0.0
        for k in range(horizon):
            predicted = 0.5 * (previous + speeds[k]) if include_ramp else speeds[k]
            arc = arc + arc_displacement(psi_a, predicted, omegas[k]); psi_a += omegas[k] * DT
            chord = chord + chord_displacement(psi_c, speeds[k], omegas[k]); psi_c += omegas[k] * DT
            previous = speeds[k]
        drift[t] = np.linalg.norm(arc - chord)
    return drift


# ----------------------------------------------------------------- part 3

def handicapped_planner(arc=True, ramp=False):
    """The Bayesian planner rolled out with the comparators' model.

    Two independent handicaps, because the decomposition shows they are not the
    same size.  `arc` integrates the heading along the turn instead of using the
    post-turn heading (the smaller term, ~6 mm median).  `ramp` predicts with the
    mean speed over the interval while the environment applies the end-of-interval
    speed as a constant (the larger term, ~31 mm median) -- which is exactly what
    the comparators suffer, since the bridge commands `getSolution(1, "v")`, the
    speed at the *end* of the first step.
    """
    from unicycle_mpc_gate import UnicycleCEMMPC

    class ArcRolloutCEMMPC(UnicycleCEMMPC):
        """Identical to the formal planner except that its rollout integrates the
        heading along the arc, as the comparators' model does, while the
        environment still turns first and then translates.  Every cost, the risk
        model, the tracker and the CEM update are inherited unchanged."""

        def _rollout(self, samples, obs):
            cfg = self.cfg
            controls = self._project_controls(samples, obs.robot_velocity)
            speeds = controls[:, :, 0]
            if ramp:
                # Predict the displacement a ramping speed would produce, while
                # the environment will still apply the commanded end speed.
                previous = np.concatenate(
                    (np.full_like(speeds[:, :1],
                                  float(np.linalg.norm(obs.robot_velocity))),
                     speeds[:, :-1]), axis=1)
                speeds = 0.5 * (previous + speeds)
            turns = controls[:, :, 1]
            omegas = turns / cfg.dt
            # Heading at the start of each step.
            start = np.concatenate(
                (np.zeros_like(turns[:, :1]), np.cumsum(turns, axis=1)[:, :-1]),
                axis=1) + self._heading(obs)
            small = np.abs(omegas) < 1e-9
            safe = np.where(small, 1.0, omegas)
            end = start + turns
            if arc:
                dx = np.where(small, speeds * np.cos(start) * cfg.dt,
                              speeds / safe * (np.sin(end) - np.sin(start)))
                dy = np.where(small, speeds * np.sin(start) * cfg.dt,
                              -speeds / safe * (np.cos(end) - np.cos(start)))
            else:
                dx = speeds * np.cos(end) * cfg.dt
                dy = speeds * np.sin(end) * cfg.dt
            positions = obs.robot_xy[None, None, :] + np.cumsum(
                np.stack((dx, dy), axis=2), axis=1)
            # Downstream terms consume velocities; report the average velocity
            # over each step, which is the displacement divided by dt.
            velocities = np.stack((dx / cfg.dt, dy / cfg.dt), axis=2)
            return controls, velocities, positions

    return ArcRolloutCEMMPC


def empirical(cases, humans=5, scenario="circle_crossing", radius=4.0, width=None):
    from continuous_mpc_gate import DEFAULT_CROWDNAV, run_episode
    from evaluate_matched_safety import config as point_config
    from unicycle_mpc_gate import UnicycleCEMMPC, unicycle_config

    cfg = unicycle_config(point_config(2), horizon=16)
    out = {}
    for label, planner in (
            ("匹配环境（正式）", UnicycleCEMMPC),
            ("仅朝向失配", handicapped_planner(arc=True, ramp=False)),
            ("仅速度斜坡失配", handicapped_planner(arc=False, ramp=True)),
            ("两项都失配", handicapped_planner(arc=True, ramp=True))):
        rows = []
        for case in cases:
            rows.append(run_episode(DEFAULT_CROWDNAV, "bayes", humans, scenario, case,
                                    cfg, radius, width, 0, 0.0, 0.0, 1.0, time_limit=25,
                                    planner_type=planner, robot_kinematics="unicycle"))
        out[label] = {
            "n": len(rows),
            "sr": 100.0 * float(np.mean([r.success_without_overlap for r in rows])),
            "cr": 100.0 * float(np.mean([bool(r.collision_union) for r in rows])),
            "time": float(np.mean([r.nav_time if r.success_without_overlap else 25.0
                                   for r in rows])),
            "path": float(np.mean([r.path_length for r in rows])),
        }
    return out


def main():
    heading, ramp, both = decompose()
    print("一、单周期偏差的分解（125 个注册网格点，单位 mm）")
    print(f"  {'来源':22s} {'中位':>8s} {'p95':>8s} {'最大':>8s}")
    for label, values in (("朝向：沿弧 vs 转后走", heading),
                          ("速度：斜坡 vs 阶跃", ramp),
                          ("合计（实际偏差）", both)):
        print(f"  {label:22s} {np.median(values)*1000:8.2f} "
              f"{np.percentile(values,95)*1000:8.2f} {values.max()*1000:8.2f}")

    drift = horizon_drift(include_ramp=True)
    drift_heading = horizon_drift(include_ramp=False)
    print(f"\n二、时域末端漂移（N=16，同一控制序列，2000 次随机控制）")
    print(f"  仅朝向项  中位 {np.median(drift_heading):.3f} m  "
          f"p95 {np.percentile(drift_heading,95):.3f} m  最大 {drift_heading.max():.3f} m")
    print(f"  两项合计  中位 {np.median(drift):.3f} m  "
          f"p95 {np.percentile(drift,95):.3f} m  最大 {drift.max():.3f} m")
    print(f"  对照：时域可达距离 {V_MAX*16*DT:.1f} m，目标半径 0.25 m，人半径 0.3 m")

    cases = json.load(open("/home/abc/temp/modern/snapshot/case_registry.json"))
    circle = cases["splits"]["D1_circle"]["cases"]
    square = cases["splits"]["D1_square"]["cases"]
    print(f"\n三、经验对照：贝叶斯规划器背上对手的模型失配（D1 开发布局）")
    results = {}
    for name, kwargs in (("dev_circle", dict(cases=circle, radius=4.0, width=None)),
                         ("dev_square", dict(cases=square, scenario="square_crossing",
                                             radius=None, width=10.0))):
        results[name] = empirical(**kwargs)
        print(f"\n  {name}（{len(kwargs['cases'])} 布局）")
        print(f"    {'配置':20s} {'SR%':>7s} {'CR%':>7s} {'罚时s':>7s} {'路径m':>7s}")
        for label, values in results[name].items():
            print(f"    {label:20s} {values['sr']:7.1f} {values['cr']:7.1f} "
                  f"{values['time']:7.2f} {values['path']:7.2f}")

    macro = {}
    for label in next(iter(results.values())):
        macro[label] = {k: float(np.mean([results[s][label][k] for s in results]))
                        for k in ("sr", "cr", "time", "path")}
    print(f"\n  两场景等权 macro")
    print(f"    {'配置':20s} {'SR%':>7s} {'CR%':>7s} {'罚时s':>7s}")
    for label, values in macro.items():
        print(f"    {label:20s} {values['sr']:7.1f} {values['cr']:7.1f} {values['time']:7.2f}")
    keys = list(macro)
    baseline = macro[keys[0]]
    delta = {k: macro[k]["sr"] - baseline["sr"] for k in keys[1:]}
    print()
    for k, v in delta.items():
        print(f"    {k:16s} 相对正式配置的 SR 变化 {v:+.1f} 个百分点，"
              f"罚时 {macro[k]['time']-baseline['time']:+.2f} s")

    Path("/home/abc/temp/modern/snapshot/dynamics_crosscheck.json").write_text(
        json.dumps({
            "one_cycle_mm": {"heading_median": float(np.median(heading)*1000),
                             "ramp_median": float(np.median(ramp)*1000),
                             "total_median": float(np.median(both)*1000),
                             "total_max": float(both.max()*1000)},
            "horizon_drift_m": {"median": float(np.median(drift)),
                                "p95": float(np.percentile(drift, 95)),
                                "max": float(drift.max())},
            "empirical": results, "macro": macro,
            "sr_change_points": delta}, indent=1, ensure_ascii=False))
    print("\n-> snapshot/dynamics_crosscheck.json")


if __name__ == "__main__":
    main()
