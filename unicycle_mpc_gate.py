"""Matched-unicycle variant of the Bayesian RFS planner.

The three modern MPC comparators (SH-MPC, T-MPC++, SICNav) all drive a unicycle,
so scoring them against the holonomic planner would compare turning limits as
much as it compares belief handling.  This file gives the same planner a
unicycle, and nothing else: the tracker, the collision hazard, every cost term
and every clearance test are inherited unchanged from ContinuousCEMMPC, which
sees only XY velocities and the positions they produce.

What changes:

  * the sampled parameters are (speed, heading increment) instead of (vx, vy),
  * the limits applied to them are a speed bound, an acceleration bound and a
    turn-rate bound rather than a planar acceleration bound,
  * the rollout integrates heading before position, matching exactly what
    CrowdSim's Agent.step does for a non-holonomic agent, so that positions
    remain start + cumsum(v_xy * dt) and no downstream term needs to know.

What does not change: the risk function, the feasibility classes, the elite
selection, the warm start policy, and the CEM update.  The holonomic planner is
not modified by importing this file.

CrowdSim's unicycle convention, reproduced here exactly:

    theta_{k+1} = theta_k + r_k          (r is an increment, already in radians)
    p_{k+1}     = p_k + v_k * [cos theta_{k+1}, sin theta_{k+1}] * dt

so the heading used for a step is the one *after* that step's turn.  Multiplying
r by dt anywhere would double-apply the timestep.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from continuous_mpc_gate import ContinuousCEMMPC, MPCConfig, PlannerObservation


@dataclass(frozen=True)
class UnicycleConfig(MPCConfig):
    """MPCConfig plus the two limits a unicycle needs.

    Registered before any development run and not tuned per arm:
    `omega_max` is the turn rate, `reverse` states whether negative speeds are
    allowed.  Both must be identical across every method in a matched
    comparison; they are recorded in the result metadata.
    """

    # 0.8 rad/s: the turn rate the modern MPC comparators' shipped model
    # supports.  Taking the tighter of the two keeps every method inside its
    # own design range rather than giving one an advantage it cannot match.
    omega_max: float = 0.8          # rad/s
    reverse: bool = False           # unicycle comparators do not drive backwards


def unicycle_config(base: MPCConfig, **overrides) -> UnicycleConfig:
    """The one place a unicycle arm's configuration is built.

    Every arm in the matched comparison -- ours and both comparators -- must be
    validated against the same actuator box.  Passing a plain `MPCConfig` to a
    comparator left `run_episode` checking the turn command against the
    `getattr(config, "omega_max", 1.0)` fallback instead of the registered
    0.8 rad/s, so a violation would not have been caught on those arms even
    though their solver bounds are 0.8.  Building the config here removes the
    chance of one arm being checked more loosely than another.
    """
    values = {f.name: getattr(base, f.name) for f in fields(MPCConfig)}
    if isinstance(base, UnicycleConfig):
        values["omega_max"] = base.omega_max
        values["reverse"] = base.reverse
    values.update(overrides)
    return UnicycleConfig(**values)


class UnicycleCEMMPC(ContinuousCEMMPC):
    """ContinuousCEMMPC with a second-order-free unicycle rollout."""

    def __init__(self, config: MPCConfig):
        if not isinstance(config, UnicycleConfig):
            # Accept a plain MPCConfig so callers can share one config factory.
            config = UnicycleConfig(
                **{f.name: getattr(config, f.name) for f in fields(MPCConfig)}
            )
        super().__init__(config)

    # ---------------------------------------------------------------- helpers

    def _heading(self, obs: PlannerObservation) -> float:
        if obs.robot_heading is not None:
            return float(obs.robot_heading)
        # Falling back to the velocity direction would silently invent a heading
        # for a stopped robot, so this is an error rather than a guess.
        raise ValueError(
            "unicycle planner requires observation.robot_heading; the adapter "
            "did not provide it"
        )

    def _speed(self, obs: PlannerObservation) -> float:
        return float(np.linalg.norm(obs.robot_velocity))

    def _turn_to(self, heading: float, target_angle: np.ndarray) -> np.ndarray:
        """Shortest signed turn from heading to each target angle."""
        return np.arctan2(np.sin(target_angle - heading), np.cos(target_angle - heading))

    # ------------------------------------------------- parameterisation hooks

    def _project_controls(
        self, controls: np.ndarray, initial_velocity: np.ndarray
    ) -> np.ndarray:
        """Clip (speed, heading increment) sequences to the registered limits.

        `initial_velocity` carries the current speed only; the heading enters
        through the rollout, not through the limits, because the turn bound is
        on the increment itself.
        """
        cfg = self.cfg
        controls = np.array(controls, dtype=np.float64, copy=True)
        max_turn = cfg.omega_max * cfg.dt
        max_speed_delta = cfg.a_max * cfg.dt
        speed_floor = -cfg.v_max if cfg.reverse else 0.0

        previous_speed = np.full(
            controls.shape[0], float(np.linalg.norm(initial_velocity)), dtype=np.float64
        )
        for k in range(cfg.horizon):
            speed = controls[:, k, 0]
            speed = np.clip(
                speed,
                np.maximum(previous_speed - max_speed_delta, speed_floor),
                np.minimum(previous_speed + max_speed_delta, cfg.v_max),
            )
            controls[:, k, 0] = speed
            controls[:, k, 1] = np.clip(controls[:, k, 1], -max_turn, max_turn)
            previous_speed = speed
        return controls

    def _rollout(self, samples: np.ndarray, obs: PlannerObservation):
        """Integrate the unicycle exactly as CrowdSim does, then hand the rest
        of the planner the XY velocities it expects."""
        cfg = self.cfg
        controls = self._project_controls(samples, obs.robot_velocity)

        heading = np.cumsum(controls[:, :, 1], axis=1) + self._heading(obs)
        velocities = np.stack(
            (controls[:, :, 0] * np.cos(heading),
             controls[:, :, 0] * np.sin(heading)),
            axis=2,
        )
        positions = obs.robot_xy[None, None, :] + np.cumsum(velocities * cfg.dt, axis=1)
        return controls, velocities, positions

    # ---------------------------------------------------------------- seeding

    def _initial_mean(self, obs: PlannerObservation) -> np.ndarray:
        """Warm start: last plan shifted forward, ending with a step that aims
        at the goal at full speed."""
        cfg = self.cfg
        heading = self._heading(obs)
        to_goal = obs.goal_xy - obs.robot_xy
        goal_angle = float(np.arctan2(to_goal[1], to_goal[0]))
        turn = float(np.clip(self._turn_to(heading, np.asarray(goal_angle)),
                             -cfg.omega_max * cfg.dt, cfg.omega_max * cfg.dt))
        target = np.array([cfg.v_max, turn], dtype=np.float64)
        if self._previous_mean is None:
            return np.repeat(target[None, :], cfg.horizon, axis=0)
        return np.vstack((self._previous_mean[1:], target[None, :]))

    def _seed_trajectories(self, obs: PlannerObservation) -> np.ndarray:
        """The same seven topologies the holonomic planner uses, expressed as
        turn-and-go sequences: straight at the goal, stop, turn hard either way,
        and two intermediate bends."""
        cfg = self.cfg
        heading = self._heading(obs)
        to_goal = obs.goal_xy - obs.robot_xy
        goal_angle = float(np.arctan2(to_goal[1], to_goal[0]))
        goal_turn = float(self._turn_to(heading, np.asarray(goal_angle)))
        max_turn = cfg.omega_max * cfg.dt

        def sequence(total_turn: float, speed: float) -> np.ndarray:
            """Spread a total heading change over the first steps, then hold."""
            per_step = np.clip(total_turn, -max_turn * cfg.horizon,
                               max_turn * cfg.horizon) / max(1, cfg.horizon // 2)
            per_step = float(np.clip(per_step, -max_turn, max_turn))
            turns = np.zeros(cfg.horizon, dtype=np.float64)
            turns[: max(1, cfg.horizon // 2)] = per_step
            return np.stack((np.full(cfg.horizon, speed), turns), axis=1)

        seeds = [
            sequence(goal_turn, cfg.v_max),                 # straight at the goal
            sequence(0.0, 0.0),                             # hold still
            sequence(np.pi, cfg.v_max * 0.5),               # turn around
            sequence(max_turn * cfg.horizon, cfg.v_max),    # hard left
            sequence(-max_turn * cfg.horizon, cfg.v_max),   # hard right
            sequence(goal_turn + max_turn * 4, cfg.v_max),  # bend left of the goal
            sequence(goal_turn - max_turn * 4, cfg.v_max),  # bend right of the goal
        ]
        return self._project_controls(np.stack(seeds), obs.robot_velocity)

    def _route_seeds(self, obs: PlannerObservation) -> np.ndarray:
        """Left bend, right bend, and wait-then-go, each returning to the goal
        heading afterwards.  These are the three independent CEM modes."""
        cfg = self.cfg
        heading = self._heading(obs)
        max_turn = cfg.omega_max * cfg.dt
        bend_steps = max(1, cfg.horizon // 3)
        wait_steps = max(1, cfg.horizon // 4)

        controls = np.zeros((3, cfg.horizon, 2), dtype=np.float64)
        headings = np.full(3, heading, dtype=np.float64)
        positions = np.repeat(obs.robot_xy[None, :], 3, axis=0)

        for step in range(cfg.horizon):
            to_goal = obs.goal_xy[None, :] - positions
            goal_angle = np.arctan2(to_goal[:, 1], to_goal[:, 0])
            turn = self._turn_to(headings, goal_angle)

            # Route 0 bends left first, route 1 bends right first, route 2 waits.
            if step < bend_steps:
                turn[0] = max_turn
                turn[1] = -max_turn
            turn = np.clip(turn, -max_turn, max_turn)

            speed = np.full(3, cfg.v_max, dtype=np.float64)
            if step < wait_steps:
                speed[2] = 0.0

            controls[:, step, 0] = speed
            controls[:, step, 1] = turn
            headings = headings + turn
            positions = positions + cfg.dt * np.stack(
                (speed * np.cos(headings), speed * np.sin(headings)), axis=1
            )

        return self._project_controls(controls, obs.robot_velocity)
