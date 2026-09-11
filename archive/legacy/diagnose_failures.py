#!/usr/bin/env python3
"""Read-only step traces for failed Bayesian MPC episodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import chndtr

from continuous_mpc_gate import (
    DEFAULT_CROWDNAV,
    MPCConfig,
    ContinuousCEMMPC,
    ObservationAdapter,
    _load_modules,
    build_env,
)


def swept_clearance(env, velocity):
    robot = np.asarray(env.robot.get_position(), dtype=np.float64)
    robot_v = np.asarray(velocity, dtype=np.float64)
    best = (math.inf, -1)
    for identifier, human in enumerate(env.humans):
        relative_start = np.asarray(human.get_position(), dtype=np.float64) - robot
        relative_end = relative_start + (
            np.asarray(human.get_velocity(), dtype=np.float64) - robot_v
        ) * env.time_step
        segment = relative_end - relative_start
        denominator = float(segment @ segment)
        fraction = float(np.clip(-relative_start @ segment / max(denominator, 1e-12), 0.0, 1.0))
        closest = relative_start + fraction * segment
        clearance = float(np.linalg.norm(closest) - human.radius - env.robot.radius)
        if clearance < best[0]:
            best = (clearance, identifier)
    return best


def oracle_action_scan(env, config):
    """Exhaustively scan the admissible first-step velocity disc using truth."""
    axis = np.linspace(-config.v_max, config.v_max, 81)
    xx, yy = np.meshgrid(axis, axis)
    velocities = np.column_stack((xx.ravel(), yy.ravel()))
    current = np.asarray(env.robot.get_velocity(), dtype=np.float64)
    admissible = (
        (np.linalg.norm(velocities, axis=1) <= config.v_max + 1e-12)
        & (np.linalg.norm(velocities - current, axis=1)
           <= config.a_max * config.dt + 1e-12)
    )
    velocities = np.vstack((velocities[admissible], current[None, :]))
    robot = np.asarray(env.robot.get_position(), dtype=np.float64)
    clearance = np.full(velocities.shape[0], math.inf, dtype=np.float64)
    for human in env.humans:
        start = np.asarray(human.get_position(), dtype=np.float64) - robot
        end = start[None, :] + (
            np.asarray(human.get_velocity(), dtype=np.float64)[None, :] - velocities
        ) * env.time_step
        segment = end - start[None, :]
        denominator = np.square(segment).sum(axis=1)
        fraction = np.clip(
            -np.sum(start[None, :] * segment, axis=1)
            / np.maximum(denominator, 1e-12),
            0.0,
            1.0,
        )
        closest = start[None, :] + fraction[:, None] * segment
        candidate = (
            np.linalg.norm(closest, axis=1) - human.radius - env.robot.radius
        )
        clearance = np.minimum(clearance, candidate)
    best = int(np.argmax(clearance))
    return float(clearance[best]), velocities[best].tolist()


def track_probability(track, robot_xy, robot_radius, human_margin):
    if track is None:
        return None
    variance = 0.5 * float(np.trace(track.covariance[:2, :2]))
    radius = robot_radius + track.radius + human_margin
    distance2 = float(np.square(robot_xy - track.mean[:2]).sum())
    if variance < 1e-9:
        return float(track.existence if distance2 <= radius * radius else 0.0)
    return float(track.existence * chndtr(radius * radius / variance, 2.0, distance2 / variance))


def run_case(case_id, crowdnav_root, config):
    env, _, _ = build_env(Path(crowdnav_root), "bayes", 20, "circle_crossing")
    _, _, ActionXY, _ = _load_modules(Path(crowdnav_root))
    env.reset(options={"test_case": case_id})
    planner = ContinuousCEMMPC(config)
    adapter = ObservationAdapter(
        "bayes", config.horizon, config.human_margin, config.dt,
        config.chance_limit, config.fixed_uncertainty_radius,
        config.acceleration_std,
    )
    seen = set()
    rows = []
    terminated = truncated = False
    event = "timeout"
    while not (terminated or truncated):
        obs = adapter.read(env)
        visible = set(env.occlusion.visible_ids)
        seen.update(visible)
        step = int(round(env.global_time / env.time_step))
        seed = case_id * 100003 + step * 97 + 1729
        velocity, _ = planner.plan(obs, seed)
        selected_controls = planner.last_controls[None, :, :]
        selected_positions = obs.robot_xy[None, None, :] + np.cumsum(
            selected_controls * config.dt, axis=1
        )
        selected_human_clearance = (
            planner._human_clearance(selected_controls, obs)
            if obs.entities.size else None
        )
        selected_full_physical, _ = planner._physical_clearance(
            selected_controls,
            obs,
            selected_positions,
            selected_human_clearance,
        )
        selected_hazard = planner._belief_collision_hazard(
            selected_controls, obs, selected_positions
        )
        true_clearance, true_id = swept_clearance(env, velocity)
        oracle_clearance, oracle_velocity = oracle_action_scan(env, config)
        track = adapter.rfs.tracks.get(true_id)
        next_robot = obs.robot_xy + velocity * config.dt
        row = {
            "step": step,
            "time": float(env.global_time),
            "robot": obs.robot_xy.tolist(),
            "goal_distance": float(np.linalg.norm(obs.robot_xy - obs.goal_xy)),
            "velocity": velocity.tolist(),
            "speed": float(np.linalg.norm(velocity)),
            "true_first_clearance": true_clearance,
            "oracle_max_first_clearance": oracle_clearance,
            "oracle_best_velocity": oracle_velocity,
            "true_closest_id": true_id,
            "closest_visible": true_id in visible,
            "closest_seen_before": true_id in seen,
            "visible_count": len(visible),
            "belief_track_count": len(adapter.rfs.tracks),
            "belief_probability_at_next": track_probability(
                track, next_robot, obs.robot_radius, config.human_margin
            ),
            "track_existence": None if track is None else float(track.existence),
            "track_missed_steps": None if track is None else int(track.missed_steps),
            "track_position_error": None if track is None else float(
                np.linalg.norm(track.mean[:2] - np.asarray(env.humans[true_id].get_position()))
            ),
            "selected_full_physical": float(selected_full_physical[0]),
            "selected_max_hazard": None if selected_hazard is None else float(
                selected_hazard[0].max()
            ),
            "selected_sum_hazard": None if selected_hazard is None else float(
                selected_hazard[0].sum()
            ),
            **planner.last_diagnostics,
        }
        _, _, terminated, truncated, info = env.step(
            ActionXY(float(velocity[0]), float(velocity[1]))
        )
        event = str(info.get("event", "nothing"))
        row["event"] = event
        row["reported_dmin"] = float(info.get("dmin", math.inf))
        rows.append(row)
    return {"case_id": case_id, "event": event, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crowdnav-root", type=Path, default=DEFAULT_CROWDNAV)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = MPCConfig(
        population=512,
        iterations=4,
        v_max=1.2,
        human_margin=0.10,
        near_chance_limit=0.15,
        chance_limit=0.50,
        probability_weight=1.0,
    )
    tasks = [(case_id, str(args.crowdnav_root), config) for case_id in args.cases]
    if args.workers == 1:
        traces = [run_case(*task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            traces = list(executor.map(lambda_args, tasks))
    payload = {"config": config.__dict__, "traces": traces}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for trace in traces:
        close = [r for r in trace["rows"] if r["true_first_clearance"] < 0.25]
        last = trace["rows"][-1]
        print(
            trace["case_id"], trace["event"], "steps", len(trace["rows"]),
            "near", len(close), "final_goal", f"{last['goal_distance']:.3f}",
            "final_class", int(last["feasibility_class"]),
        )


def lambda_args(args):
    return run_case(*args)


if __name__ == "__main__":
    main()
