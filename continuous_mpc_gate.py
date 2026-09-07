#!/usr/bin/env python3
"""Paired continuous-MPC gate for occluded CrowdNav.

The controller never reads ``env.humans``.  The three arms differ only in the
observation exposed by ``OcclusionBelief``:

* sensor: visible pedestrians only;
* gt: all pedestrians through the explicit oracle mode;
* worst: visible pedestrians plus every unknown sensor-grid cell treated as
  potentially occupied.

Humans cannot observe the robot, so their trajectories are exogenous and are
paired by CrowdNav test-case id.  This file deliberately contains no learned
policy, Mamba dependency, IL, or RL.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import configparser
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt
from scipy.special import ndtr
from scipy.stats import ncx2


ROOT = Path(__file__).resolve().parent
DEFAULT_CROWDNAV = Path(
    "/home/abc/workspace/nav_data/mamba/camrl/CrowdNav"
)


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.25
    horizon: int = 16
    population: int = 768
    iterations: int = 4
    elite_fraction: float = 0.08
    v_max: float = 1.0
    a_max: float = 2.0
    human_margin: float = 0.16
    unknown_margin: float = 0.05
    goal_stage_weight: float = 0.28
    goal_terminal_weight: float = 9.0
    smooth_weight: float = 0.35
    effort_weight: float = 0.02
    collision_weight: float = 20000.0
    discomfort_weight: float = 500.0
    unknown_weight: float = 6500.0
    probability_weight: float = 2500.0
    chance_limit: float = 0.10
    near_chance_limit: float = 0.10
    occupancy_chance_limit: float = 0.15
    fixed_uncertainty_radius: float = 0.35
    stagnation_weight: float = 250.0
    min_progress: float = 0.15
    init_std: float = 0.65
    min_std: float = 0.08
    acceleration_std: float = 0.55
    conformal_visible_radii: Tuple[float, ...] = ()
    conformal_hidden_radii: Tuple[float, ...] = ()


@dataclass
class UnknownField:
    clearance: np.ndarray
    x0: float
    y0: float
    resolution: float

    def sample(self, positions: np.ndarray) -> np.ndarray:
        cols = np.floor((positions[..., 0] - self.x0) / self.resolution).astype(int)
        rows = np.floor((positions[..., 1] - self.y0) / self.resolution).astype(int)
        valid = (
            (rows >= 0)
            & (rows < self.clearance.shape[0])
            & (cols >= 0)
            & (cols < self.clearance.shape[1])
        )
        out = np.zeros(rows.shape, dtype=np.float64)
        out[valid] = self.clearance[rows[valid], cols[valid]]
        return out


@dataclass
class ProbabilityField:
    probability: np.ndarray
    x0: float
    y0: float
    resolution: float

    def sample(self, positions: np.ndarray) -> np.ndarray:
        cols = np.floor((positions[..., 0] - self.x0) / self.resolution).astype(int)
        rows = np.floor((positions[..., 1] - self.y0) / self.resolution).astype(int)
        valid = (
            (rows >= 0)
            & (rows < self.probability.shape[0])
            & (cols >= 0)
            & (cols < self.probability.shape[1])
        )
        out = np.ones(rows.shape, dtype=np.float64)
        out[valid] = self.probability[rows[valid], cols[valid]]
        return out


@dataclass
class PlannerObservation:
    robot_xy: np.ndarray
    robot_velocity: np.ndarray
    robot_radius: float
    goal_xy: np.ndarray
    entities: np.ndarray
    human_segment_start: Optional[np.ndarray]
    human_segment_end: Optional[np.ndarray]
    human_uncertainty_buffer: Optional[np.ndarray]
    human_position_covariance: Optional[np.ndarray]
    human_existence: Optional[np.ndarray]
    unknown: Optional[UnknownField]
    occupancy_probability: Optional[ProbabilityField]
    provenance: str


@dataclass
class EpisodeResult:
    arm: str
    scenario: str
    human_num: int
    case_id: int
    event: str
    success: int
    collision: int
    timeout: int
    nav_time: float
    scored_time: float
    path_length: float
    min_clearance: float
    mean_plan_ms: float
    p95_plan_ms: float
    solver_steps: int
    observation_hash: str
    actual_min_clearance: float = math.inf
    actual_overlap_steps: int = 0
    first_actual_overlap_time: Optional[float] = None
    collision_union: int = 0
    success_without_overlap: int = 0
    mean_pipeline_ms: float = 0.0
    p95_pipeline_ms: float = 0.0
    deadline_miss_fraction: float = 0.0
    plan_step_ms: List[float] = field(default_factory=list)
    pipeline_step_ms: List[float] = field(default_factory=list)


class ContinuousCEMMPC:
    """Continuous trajectory optimizer with temporally correlated CEM samples."""

    def __init__(self, config: MPCConfig):
        self.cfg = config
        self._previous_mean: Optional[np.ndarray] = None
        self.last_controls: Optional[np.ndarray] = None
        self.last_diagnostics: Dict[str, float] = {}
        if config.population < 32:
            raise ValueError("route search needs at least 32 samples")

    def reset(self) -> None:
        self._previous_mean = None
        self.last_controls = None

    def _belief_risk_limits(self) -> np.ndarray:
        """Risk budget is strict near execution and relaxed far in the horizon."""
        if self.cfg.horizon <= 1:
            return np.asarray([self.cfg.near_chance_limit], dtype=np.float64)
        phase = np.linspace(0.0, 1.0, self.cfg.horizon, dtype=np.float64)
        limits = (
            self.cfg.near_chance_limit
            + (self.cfg.chance_limit - self.cfg.near_chance_limit)
            * np.square(phase)
        )
        return limits

    def _belief_hazard_limits(self) -> np.ndarray:
        return -np.log1p(-np.clip(self._belief_risk_limits(), 0.0, 1.0 - 1e-12))

    @staticmethod
    def _goal_velocity(obs: PlannerObservation, speed: float) -> np.ndarray:
        delta = obs.goal_xy - obs.robot_xy
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return np.zeros(2, dtype=np.float64)
        return delta / distance * min(speed, distance / 0.25)

    @staticmethod
    def _active_until_goal(
        positions: np.ndarray, obs: PlannerObservation
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return task-active steps and goal distances under terminal semantics.

        CrowdSim checks collision on the step entering the goal and then ends
        the episode.  That entry step remains active; only later, nonexistent
        trajectory steps are masked out.
        """
        center_distance = np.linalg.norm(
            positions - obs.goal_xy[None, None, :], axis=2
        )
        reached = center_distance < obs.robot_radius
        reached_before = np.concatenate(
            (
                np.zeros((positions.shape[0], 1), dtype=bool),
                np.maximum.accumulate(reached[:, :-1], axis=1),
            ),
            axis=1,
        )
        return ~reached_before, reached, center_distance

    def _initial_mean(self, obs: PlannerObservation) -> np.ndarray:
        target = self._goal_velocity(obs, self.cfg.v_max)
        if self._previous_mean is None:
            return np.repeat(target[None, :], self.cfg.horizon, axis=0)
        shifted = np.vstack((self._previous_mean[1:], target[None, :]))
        return shifted

    def _seed_trajectories(self, obs: PlannerObservation) -> np.ndarray:
        """Deterministic topology seeds; never average their final actions."""
        goal = self._goal_velocity(obs, self.cfg.v_max)
        goal_norm = max(float(np.linalg.norm(goal)), 1e-12)
        forward = goal / goal_norm
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        targets = [
            goal,
            np.zeros(2, dtype=np.float64),
            -self.cfg.v_max * forward,
            self.cfg.v_max * left,
            -self.cfg.v_max * left,
            self.cfg.v_max * (0.70 * forward + 0.70 * left),
            self.cfg.v_max * (0.70 * forward - 0.70 * left),
        ]
        raw = np.stack(
            [np.repeat(target[None, :], self.cfg.horizon, axis=0) for target in targets]
        )
        return self._project_controls(raw, obs.robot_velocity)

    def _project_controls(
        self, controls: np.ndarray, initial_velocity: np.ndarray
    ) -> np.ndarray:
        controls = controls.copy()
        previous = np.broadcast_to(initial_velocity, (controls.shape[0], 2)).copy()
        max_delta = self.cfg.a_max * self.cfg.dt
        for k in range(self.cfg.horizon):
            delta = controls[:, k] - previous
            delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-12))
            controls[:, k] = previous + delta * scale
            speed = np.linalg.norm(controls[:, k], axis=1, keepdims=True)
            controls[:, k] *= np.minimum(
                1.0, self.cfg.v_max / np.maximum(speed, 1e-12)
            )
            previous = controls[:, k]
        return controls

    def _route_seeds(self, obs: PlannerObservation) -> np.ndarray:
        """Left/right bends and wait-then-go, returning toward the actual goal."""
        forward = self._goal_velocity(obs, 1.0)
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        left = np.array([-forward[1], forward[0]])
        controls = np.zeros((3, self.cfg.horizon, 2))
        positions = np.repeat(obs.robot_xy[None, :], 3, axis=0)
        previous = np.repeat(obs.robot_velocity[None, :], 3, axis=0)
        bend_steps = max(1, self.cfg.horizon // 3)
        for step in range(self.cfg.horizon):
            target = obs.goal_xy[None, :] - positions
            distance = np.linalg.norm(target, axis=1, keepdims=True)
            target *= np.minimum(self.cfg.v_max / np.maximum(distance, 1e-12),
                                 1.0 / self.cfg.dt)
            if step < bend_steps:
                target[0] = self.cfg.v_max * (0.5 * forward + np.sqrt(0.75) * left)
                target[1] = self.cfg.v_max * (0.5 * forward - np.sqrt(0.75) * left)
            if step < max(1, self.cfg.horizon // 4):
                target[2] = 0.0
            delta = target - previous
            delta *= np.minimum(1.0, self.cfg.a_max * self.cfg.dt /
                                np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12))
            previous = previous + delta
            controls[:, step] = previous
            positions += self.cfg.dt * previous
        return controls

    @staticmethod
    def _elite_indices(costs, full_clearance, first_clearance, count):
        # Fit each route to feasible samples first, not a mix across obstacles.
        feasible = full_clearance >= 0.0
        first_feasible = first_clearance >= 0.0
        category = np.where(feasible, 0, np.where(first_feasible, 1, 2))
        violation = np.where(feasible, 0.0,
                             np.where(first_feasible, -full_clearance, -first_clearance))
        return np.lexsort((costs, violation, category))[:count]

    def _human_clearance(
        self, controls: np.ndarray, obs: PlannerObservation
    ) -> np.ndarray:
        """Swept clearance for every population/human/horizon segment."""
        if not obs.entities.size:
            return np.empty((controls.shape[0], 0, self.cfg.horizon))
        cfg = self.cfg
        robot_end = obs.robot_xy[None, None, :] + np.cumsum(
            controls * cfg.dt, axis=1
        )
        robot_start = np.concatenate(
            (
                np.broadcast_to(obs.robot_xy, (controls.shape[0], 1, 2)),
                robot_end[:, :-1],
            ),
            axis=1,
        )
        if obs.human_segment_start is not None:
            human_start = obs.human_segment_start
            human_end = obs.human_segment_end
        else:
            times = cfg.dt * np.arange(1, cfg.horizon + 1, dtype=np.float64)
            human_end = (
                obs.entities[:, None, 0:2]
                + obs.entities[:, None, 2:4] * times[None, :, None]
            )
            human_start = np.concatenate(
                (obs.entities[:, None, 0:2], human_end[:, :-1]), axis=1
            )
        rel_start = human_start[None, :, :, :] - robot_start[:, None, :, :]
        rel_end = human_end[None, :, :, :] - robot_end[:, None, :, :]
        segment = rel_end - rel_start
        denominator = np.square(segment).sum(axis=3)
        fraction = np.clip(
            -np.sum(rel_start * segment, axis=3)
            / np.maximum(denominator, 1e-12),
            0.0,
            1.0,
        )
        closest = rel_start + fraction[:, :, :, None] * segment
        radii = obs.robot_radius + obs.entities[None, :, None, 4]
        return np.linalg.norm(closest, axis=3) - radii

    def _cost(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: np.ndarray,
        human_clearance: Optional[np.ndarray],
        occupancy_probability: Optional[np.ndarray],
        belief_hazard: Optional[np.ndarray],
    ) -> np.ndarray:
        cfg = self.cfg
        active, reached, center_distance = self._active_until_goal(positions, obs)
        # CrowdSim terminates when the robot enters its goal-radius disc.  The
        # optimizer must target that same set instead of an unnecessarily
        # stricter (and sometimes occupied) zero-radius point.
        goal_dist = np.maximum(center_distance - obs.robot_radius, 0.0)
        terminal_goal_dist = np.where(
            reached.any(axis=1), 0.0, goal_dist[:, -1]
        )
        cost = (
            cfg.goal_stage_weight * (goal_dist * active).sum(axis=1)
            + cfg.goal_terminal_weight * terminal_goal_dist
            + cfg.effort_weight
            * (np.square(controls) * active[:, :, None]).sum(axis=(1, 2))
        )
        previous = np.concatenate(
            (
                np.broadcast_to(obs.robot_velocity, (controls.shape[0], 1, 2)),
                controls[:, :-1],
            ),
            axis=1,
        )
        cost += cfg.smooth_weight * (
            np.square(controls - previous) * active[:, :, None]
        ).sum(axis=(1, 2))

        initial_distance = max(
            float(np.linalg.norm(obs.robot_xy - obs.goal_xy)) - obs.robot_radius,
            0.0,
        )
        progress = initial_distance - terminal_goal_dist
        shortfall = np.maximum(cfg.min_progress - progress, 0.0)
        cost += cfg.stagnation_weight * np.square(shortfall)

        if obs.entities.size:
            clearance = human_clearance
            penetration = np.maximum(-clearance, 0.0)
            margin = cfg.human_margin
            if obs.human_uncertainty_buffer is not None:
                margin = margin + obs.human_uncertainty_buffer[None, :, :]
            discomfort = np.maximum(margin - clearance, 0.0)
            human_active = active[:, None, :]
            cost += cfg.collision_weight * (
                np.square(penetration) * human_active
            ).sum(axis=(1, 2))
            cost += cfg.discomfort_weight * (
                np.square(discomfort) * human_active
            ).sum(axis=(1, 2))
            cost += 1e5 * (
                np.where(human_active, penetration, 0.0).max(axis=(1, 2)) > 0.02
            )

        if obs.unknown is not None:
            clearance = obs.unknown.sample(positions)
            required = obs.robot_radius + cfg.unknown_margin
            penetration = np.maximum(required - clearance, 0.0)
            cost += cfg.unknown_weight * (np.square(penetration) * active).sum(axis=1)
            cost += 1e5 * (
                np.where(active, penetration, 0.0).max(axis=1) > 0.05
            )

        if obs.occupancy_probability is not None:
            probability = occupancy_probability
            cost += cfg.probability_weight * (
                -np.log1p(-np.clip(probability, 0.0, 1.0 - 1e-9)) * active
            ).sum(axis=1)
            cost += 1e5 * (
                np.where(active, probability, 0.0).max(axis=1)
                > cfg.occupancy_chance_limit
            )

        if belief_hazard is not None:
            cost += cfg.probability_weight * (belief_hazard * active).sum(axis=1)
            limits = self._belief_hazard_limits()[None, :]
            cost += 1e5 * (
                np.where(active, belief_hazard - limits, -math.inf).max(axis=1)
                > 0.0
            )

        return cost

    def _belief_collision_hazard(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Bernoulli-Gaussian collision hazard per step.

        The current filter's position covariance is isotropic, so squared
        radial distance follows a non-central chi-square distribution.  Its
        CDF gives the exact probability mass inside the collision disc.  The
        additive negative log-survival form preserves ordering when union
        probabilities round to one in dense crowds.
        """
        if (
            obs.human_position_covariance is None
            or obs.human_existence is None
            or not obs.entities.size
        ):
            return None
        if positions is None:
            positions = obs.robot_xy[None, None, :] + np.cumsum(
                controls * self.cfg.dt, axis=1
            )
        human_positions = obs.human_segment_end
        relative = positions[:, None, :, :] - human_positions[None, :, :, :]
        distance = np.linalg.norm(relative, axis=3)
        variance = 0.5 * np.trace(
            obs.human_position_covariance, axis1=2, axis2=3
        )
        collision_radius = (
            obs.robot_radius
            + obs.entities[None, :, None, 4]
            + self.cfg.human_margin
        )
        scaled_radius = np.square(collision_radius) / np.maximum(
            variance[None, :, :], 1e-9
        )
        noncentrality = np.square(distance) / np.maximum(
            variance[None, :, :], 1e-9
        )
        # A collision disc lies inside its near-side tangent half-plane.
        # Beyond 8 sigma use the upper bound Phi(-8), never drop a person.
        # Per-step hazard error is at most N * 6.23e-16 (roundoff aside).
        far = (distance - collision_radius) >= 8.0 * np.sqrt(variance)[None, :, :]
        conditional = np.full(distance.shape, ndtr(-8.0))
        near = ~far
        conditional[near] = ncx2.cdf(
            np.broadcast_to(scaled_radius, distance.shape)[near],
            2.0, noncentrality[near],
        )
        deterministic = variance < 1e-9
        if np.any(deterministic):
            conditional = np.where(
                deterministic[None, :, :],
                distance <= collision_radius,
                conditional,
            )
        component = conditional * obs.human_existence[None, :, None]
        return -np.log1p(-np.clip(component, 0.0, 1.0 - 1e-12)).sum(axis=1)

    def _belief_collision_probability(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        hazard = self._belief_collision_hazard(controls, obs, positions)
        return None if hazard is None else -np.expm1(-hazard)

    def _combined_clearance(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: np.ndarray,
        human_clearance: Optional[np.ndarray],
        occupancy_probability: Optional[np.ndarray],
        belief_hazard: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Minimum robust clearance over the horizon and over the first step."""
        population = controls.shape[0]
        active, _, _ = self._active_until_goal(positions, obs)
        full = np.full(population, math.inf, dtype=np.float64)
        first = np.full(population, math.inf, dtype=np.float64)
        if obs.entities.size:
            margin = self.cfg.human_margin
            if obs.human_uncertainty_buffer is not None:
                margin = margin + obs.human_uncertainty_buffer[None, :, :]
            human = np.where(
                active[:, None, :], human_clearance - margin, math.inf
            )
            full = np.minimum(full, human.min(axis=(1, 2)))
            first = np.minimum(first, human[:, :, 0].min(axis=1))
        if obs.unknown is not None:
            unknown = np.where(
                active,
                obs.unknown.sample(positions)
                - (obs.robot_radius + self.cfg.unknown_margin),
                math.inf,
            )
            full = np.minimum(full, unknown.min(axis=1))
            first = np.minimum(first, unknown[:, 0])
        if obs.occupancy_probability is not None:
            probability = occupancy_probability
            slack = np.where(
                active, self.cfg.occupancy_chance_limit - probability, math.inf
            )
            full = np.minimum(full, slack.min(axis=1))
            first = np.minimum(first, slack[:, 0])
        if belief_hazard is not None:
            slack = np.where(
                active,
                self._belief_hazard_limits()[None, :] - belief_hazard,
                math.inf,
            )
            full = np.minimum(full, slack.min(axis=1))
            first = np.minimum(first, slack[:, 0])
        return full, first

    def _physical_clearance(
        self,
        controls: np.ndarray,
        obs: PlannerObservation,
        positions: np.ndarray,
        human_clearance: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Geometric clearance used when every chance constraint is infeasible."""
        population = controls.shape[0]
        active, _, _ = self._active_until_goal(positions, obs)
        full = np.full(population, math.inf, dtype=np.float64)
        first = np.full(population, math.inf, dtype=np.float64)
        if obs.entities.size:
            margin = self.cfg.human_margin
            if obs.human_uncertainty_buffer is not None:
                margin = margin + obs.human_uncertainty_buffer[None, :, :]
            human = np.where(
                active[:, None, :], human_clearance - margin, math.inf
            )
            full = np.minimum(full, human.min(axis=(1, 2)))
            first = np.minimum(first, human[:, :, 0].min(axis=1))
        if obs.unknown is not None:
            unknown = np.where(
                active,
                obs.unknown.sample(positions)
                - (obs.robot_radius + self.cfg.unknown_margin),
                math.inf,
            )
            full = np.minimum(full, unknown.min(axis=1))
            first = np.minimum(first, unknown[:, 0])
        return full, first

    def plan(self, obs: PlannerObservation, seed: int) -> Tuple[np.ndarray, float]:
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        mean = self._initial_mean(obs)
        start = time.perf_counter()
        seeds = self._seed_trajectories(obs)
        route_seeds = self._route_seeds(obs)
        means = np.concatenate((mean[None, :, :], route_seeds), axis=0)
        deviations = np.full_like(means, cfg.init_std)
        # Retain half the budget around the warm start. Fit alternatives
        # independently so left/right turns cannot cancel out.
        remaining = cfg.population - cfg.population // 2
        counts = [cfg.population // 2] + [remaining // 3] * 3
        for i in range(remaining % 3):
            counts[i + 1] += 1
        bounds = np.cumsum([0] + counts)
        mode_ids = np.repeat(np.arange(4), counts)
        best_controls = seeds[0].copy()
        best_cost = math.inf
        best_class = 3
        best_first_clearance = -math.inf
        best_full_clearance = -math.inf
        best_first_physical = -math.inf

        for _ in range(cfg.iterations):
            noise = rng.standard_normal((cfg.population, cfg.horizon, 2))
            # Low-pass noise gives coherent curves instead of jittering controls.
            noise[:, 1:] = 0.68 * noise[:, :-1] + 0.32 * noise[:, 1:]
            samples = means[mode_ids] + deviations[mode_ids] * noise
            samples[:len(seeds)] = seeds
            samples[len(seeds)] = mean
            for route in range(1, 4):
                samples[bounds[route]] = means[route]
                samples[bounds[route] + 1] = route_seeds[route - 1]
            controls = self._project_controls(samples, obs.robot_velocity)
            positions = obs.robot_xy[None, None, :] + np.cumsum(
                controls * cfg.dt, axis=1
            )
            human_clearance = (
                self._human_clearance(controls, obs) if obs.entities.size else None
            )
            occupancy_probability = (
                obs.occupancy_probability.sample(positions)
                if obs.occupancy_probability is not None else None
            )
            belief_hazard = self._belief_collision_hazard(
                controls, obs, positions
            )
            costs = self._cost(
                controls,
                obs,
                positions,
                human_clearance,
                occupancy_probability,
                belief_hazard,
            )
            full_clearance, first_clearance = self._combined_clearance(
                controls,
                obs,
                positions,
                human_clearance,
                occupancy_probability,
                belief_hazard,
            )
            _, first_physical = self._physical_clearance(
                controls, obs, positions, human_clearance
            )
            # Clearances already include the configured human/unknown buffers.
            full_feasible = full_clearance >= 0.0
            first_feasible = first_clearance >= 0.0
            if full_feasible.any():
                pool = np.flatnonzero(full_feasible)
                iteration_best = int(pool[np.argmin(costs[pool])])
                candidate_class = 0
            elif first_feasible.any():
                pool = np.flatnonzero(first_feasible)
                safest = float(full_clearance[pool].max())
                near_safest = pool[full_clearance[pool] >= safest - 0.02]
                iteration_best = int(near_safest[np.argmin(costs[near_safest])])
                candidate_class = 1
            else:
                safest = float(first_physical.max())
                near_safest = np.flatnonzero(first_physical >= safest - 0.02)
                iteration_best = int(
                    near_safest[np.argmin(costs[near_safest])]
                )
                candidate_class = 2
            candidate_cost = float(costs[iteration_best])
            candidate_first = float(first_clearance[iteration_best])
            replace = candidate_class < best_class
            if candidate_class == best_class == 2:
                candidate_physical = float(first_physical[iteration_best])
                replace = (
                    candidate_physical > best_first_physical + 0.02
                    or (
                        candidate_physical >= best_first_physical - 0.02
                        and candidate_cost < best_cost
                    )
                )
            elif candidate_class == best_class == 1:
                replace = (
                    float(full_clearance[iteration_best]) > best_full_clearance + 0.02
                    or (
                        float(full_clearance[iteration_best])
                        >= best_full_clearance - 0.02
                        and candidate_cost < best_cost
                    )
                )
            elif candidate_class == best_class:
                replace = candidate_cost < best_cost
            if replace:
                best_class = candidate_class
                best_cost = candidate_cost
                best_first_clearance = candidate_first
                best_full_clearance = float(full_clearance[iteration_best])
                best_first_physical = float(first_physical[iteration_best])
                best_controls = controls[iteration_best].copy()
            for route, count in enumerate(counts):
                begin, end = bounds[route:route + 2]
                indices = self._elite_indices(
                    costs[begin:end], full_clearance[begin:end],
                    first_clearance[begin:end], max(4, round(count * cfg.elite_fraction)),
                ) + begin
                elite = controls[indices]
                means[route] = 0.22 * means[route] + 0.78 * elite.mean(axis=0)
                deviations[route] = np.maximum(cfg.min_std, elite.std(axis=0))

        # The average of trajectories passing on opposite sides can pass through
        # the pedestrian.  Execute the best optimized trajectory, not that mean.
        action = best_controls[0]
        self._previous_mean = best_controls
        # Read-only diagnostic snapshot.  Keeping the selected trajectory makes
        # failure attribution possible without changing the optimizer contract.
        self.last_controls = best_controls.copy()
        self.last_diagnostics = {
            "feasibility_class": float(best_class),
            "first_clearance": best_first_clearance,
            "horizon_clearance": best_full_clearance,
            "first_physical_clearance": best_first_physical,
            "cost": best_cost,
        }
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return action, elapsed_ms


def policy_observation_frame(env):
    """One permitted observation interface for full and occluded evaluation."""
    state = env.get_policy_state()
    if env.occlusion_enabled():
        return (state, list(state.policy_entities or []), set(env.occlusion.visible_ids),
                env.occlusion.sensor_grid, env.occlusion._mesh, env.occlusion.res)
    entities = [dict(id=i, px=h.px, py=h.py, vx=h.vx, vy=h.vy, radius=h.radius)
                for i, h in enumerate(state.human_states)]
    state.policy_entities = entities
    state.visible_ids = list(range(len(entities)))
    state.provenance = "full_observation"
    resolution, extent = .25, 5.
    axis = resolution * (np.arange(int(2 * extent / resolution)) + .5) - extent
    mesh = np.meshgrid(axis + state.self_state.px, axis + state.self_state.py)
    sensor = np.zeros(mesh[0].shape, dtype=np.float32)
    for entity in entities:
        occupied = ((mesh[0] - entity['px']) ** 2 + (mesh[1] - entity['py']) ** 2
                    <= entity['radius'] ** 2)
        sensor[occupied] = 1.
    return state, entities, set(state.visible_ids), sensor, mesh, resolution


class ObservationAdapter:
    def __init__(
        self,
        arm: str,
        horizon: int = MPCConfig().horizon,
        safety_margin: float = MPCConfig().human_margin,
        dt: float = MPCConfig().dt,
        chance_limit: float = MPCConfig().chance_limit,
        fixed_uncertainty_radius: float = MPCConfig().fixed_uncertainty_radius,
        acceleration_std: float = MPCConfig().acceleration_std,
        position_noise_std: float = 0.0,
        velocity_noise_std: float = 0.0,
        detection_probability: float = 1.0,
        observation_seed: int = 0,
        conformal_visible_radii: Sequence[float] = (),
        conformal_hidden_radii: Sequence[float] = (),
    ):
        if arm not in {
            "sensor", "gt", "worst", "deterministic", "fixed", "bayes",
            "bayes_poisson", "conformal",
        }:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.horizon = horizon
        self.safety_margin = safety_margin
        self.dt = float(dt)
        if position_noise_std < 0.0 or velocity_noise_std < 0.0:
            raise ValueError("observation noise standard deviations must be non-negative")
        if not 0.0 < detection_probability <= 1.0:
            raise ValueError("detection_probability must be in (0, 1]")
        self.position_noise_std = float(position_noise_std)
        self.velocity_noise_std = float(velocity_noise_std)
        self.detection_probability = float(detection_probability)
        self.observation_seed = int(observation_seed)
        self.detected_entities = []
        self.reported_ids = []
        self.conformal_visible_radii = np.asarray(
            conformal_visible_radii, dtype=np.float64
        )
        self.conformal_hidden_radii = np.asarray(
            conformal_hidden_radii, dtype=np.float64
        )
        if arm == "conformal":
            if (
                self.conformal_visible_radii.shape != (horizon,)
                or self.conformal_hidden_radii.shape != (horizon,)
            ):
                raise ValueError(
                    "conformal requires one visible and hidden radius per horizon"
                )
            if (
                np.any(self.conformal_visible_radii < 0.0)
                or np.any(self.conformal_hidden_radii < 0.0)
            ):
                raise ValueError("conformal radii must be non-negative")
        self._observation_rng = (
            np.random.default_rng(observation_seed)
            if position_noise_std > 0.0
            or velocity_noise_std > 0.0
            or detection_probability < 1.0
            else None
        )
        self.rfs = None
        if arm in {
            "deterministic", "fixed", "bayes", "bayes_poisson", "conformal",
        }:
            from bayesian_rfs import BayesianRFSBelief, RFSConfig

            self.rfs = BayesianRFSBelief(
                RFSConfig(
                    dt=dt,
                    chance_limit=chance_limit,
                    fixed_uncertainty_radius=fixed_uncertainty_radius,
                    acceleration_std=acceleration_std,
                    use_poisson_birth=arm == "bayes_poisson",
                    position_measurement_std=position_noise_std,
                    velocity_measurement_std=velocity_noise_std,
                    detection_probability=min(0.98, detection_probability),
                ),
                mode=(
                    "bayes" if arm == "bayes_poisson"
                    else "deterministic" if arm == "conformal"
                    else arm
                ),
            )

    def _corrupt_detections(self, entities, timestamp):
        observed = []
        step = int(round(timestamp / self.dt))
        for source in entities:
            # Key noise by object and time, not by controller-dependent visibility order.
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.observation_seed, step, int(source["id"]) & 0xffffffff]))
            draw = rng.standard_normal(5)
            if float(ndtr(draw[0])) > self.detection_probability:
                continue
            entity = dict(source)
            entity["px"] += float(draw[1] * self.position_noise_std)
            entity["py"] += float(draw[2] * self.position_noise_std)
            entity["vx"] += float(draw[3] * self.velocity_noise_std)
            entity["vy"] += float(draw[4] * self.velocity_noise_std)
            observed.append(entity)
        return observed

    def read(self, env) -> PlannerObservation:
        state, entities, visible_ids, sensor_grid, mesh, resolution = policy_observation_frame(env)
        self.sensor_grid, self.sensor_mesh = sensor_grid, mesh

        if self.arm in {
            "sensor", "worst", "deterministic", "fixed", "bayes",
            "bayes_poisson", "conformal",
        }:
            leaked = [e["id"] for e in entities if e["id"] not in visible_ids]
            if leaked:
                raise RuntimeError(f"hidden-state leakage in {self.arm}: {leaked}")
        elif len(entities) != env.human_num:
            raise RuntimeError(
                f"GT arm expected {env.human_num} entities, received {len(entities)}"
            )

        if self._observation_rng is not None and self.arm != "gt":
            entities = self._corrupt_detections(entities, env.global_time)
        self.detected_entities = entities

        entity_array = np.asarray(
            [[e["px"], e["py"], e["vx"], e["vy"], e["radius"]] for e in entities],
            dtype=np.float64,
        )
        if entity_array.size == 0:
            entity_array = np.empty((0, 5), dtype=np.float64)

        unknown = None
        occupancy_probability = None
        human_uncertainty_buffer = None
        human_position_covariance = None
        human_existence = None
        if self.arm == "worst":
            sensor = sensor_grid
            unknown_mask = sensor == 0.5
            clearance = distance_transform_edt(~unknown_mask) * resolution
            mx, my = mesh
            unknown = UnknownField(
                clearance=clearance,
                x0=float(mx[0, 0] - 0.5 * resolution),
                y0=float(my[0, 0] - 0.5 * resolution),
                resolution=float(resolution),
            )

        robot = state.self_state
        human_segment_start = human_segment_end = None
        if self.arm == "gt":
            human_segment_start, human_segment_end = rollout_human_segments(
                env, self.horizon
            )
        elif self.rfs is not None:
            self.rfs.update(
                entities,
                sensor_grid,
                mesh,
                env.global_time,
            )
            output = self.rfs.output(
                sensor_grid,
                mesh,
                self.horizon,
                float(robot.radius),
                self.safety_margin,
            )
            entity_array = output.entities
            self.reported_ids = [track.identifier for track in self.rfs.tracks.values()
                                 if track.visible or track.existence >= self.rfs.cfg.report_existence]
            if len(self.reported_ids) != len(entity_array):
                raise RuntimeError("track/output ordering mismatch")
            human_segment_start = output.segment_start
            human_segment_end = output.segment_end
            if self.arm == "conformal":
                selected = [
                    track for track in self.rfs.tracks.values()
                    if track.visible
                    or track.existence >= self.rfs.cfg.report_existence
                ]
                if len(selected) != len(entity_array):
                    raise RuntimeError("conformal track/output ordering mismatch")
                visible = np.asarray(
                    [track.visible for track in selected], dtype=bool
                )[:, None]
                human_uncertainty_buffer = np.where(
                    visible,
                    self.conformal_visible_radii[None, :],
                    self.conformal_hidden_radii[None, :],
                )
            elif self.arm == "fixed":
                human_uncertainty_buffer = output.uncertainty_buffer
            elif self.arm != "deterministic":
                human_position_covariance = output.position_covariance
                human_existence = output.existence
            mx, my = mesh
            if np.any(output.occupancy_probability > 0.0):
                occupancy_probability = ProbabilityField(
                    probability=output.occupancy_probability,
                    x0=float(mx[0, 0] - 0.5 * resolution),
                    y0=float(my[0, 0] - 0.5 * resolution),
                    resolution=float(resolution),
                )
        payload = np.concatenate(
            ([robot.px, robot.py, robot.vx, robot.vy, robot.gx, robot.gy],
             entity_array.reshape(-1))
        )
        if occupancy_probability is not None:
            payload = np.concatenate(
                (payload, occupancy_probability.probability.reshape(-1))
            )
        if human_position_covariance is not None:
            payload = np.concatenate(
                (payload, human_position_covariance.reshape(-1), human_existence)
            )
        obs_hash = hashlib.sha256(payload.astype(np.float64).tobytes()).hexdigest()
        return PlannerObservation(
            robot_xy=np.array([robot.px, robot.py], dtype=np.float64),
            robot_velocity=np.array([robot.vx, robot.vy], dtype=np.float64),
            robot_radius=float(robot.radius),
            goal_xy=np.array([robot.gx, robot.gy], dtype=np.float64),
            entities=entity_array,
            human_segment_start=human_segment_start,
            human_segment_end=human_segment_end,
            human_uncertainty_buffer=human_uncertainty_buffer,
            human_position_covariance=human_position_covariance,
            human_existence=human_existence,
            unknown=unknown,
            occupancy_probability=occupancy_probability,
            provenance=f"{state.provenance}:{obs_hash}",
        )


def rollout_human_segments(env, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Roll out the exogenous ORCA crowd without mutating the real environment.

    CrowdSim checks each step's swept collision using the velocity stored at the
    beginning of that step, then advances humans using their newly computed ORCA
    action.  Separate segment starts/ends preserve that exact convention.
    """
    humans = []
    for source in env.humans:
        human = copy.copy(source)
        human.policy = copy.copy(source.policy)
        human.policy.sim = None
        humans.append(human)

    starts = np.empty((len(humans), horizon, 2), dtype=np.float64)
    ends = np.empty_like(starts)
    for step in range(horizon):
        starts[:, step] = np.asarray(
            [[human.px, human.py] for human in humans], dtype=np.float64
        )
        velocities = np.asarray(
            [[human.vx, human.vy] for human in humans], dtype=np.float64
        )
        ends[:, step] = starts[:, step] + velocities * env.time_step
        actions = []
        for human in humans:
            observation = [
                other.get_observable_state() for other in humans if other is not human
            ]
            actions.append(human.act(observation))
        for human, action in zip(humans, actions):
            human.step(action)
    return starts, ends


def _load_modules(crowdnav_root: Path):
    root_text = str(crowdnav_root.resolve())
    sys.path = [p for p in sys.path if "soc-nav-training/CrowdNav" not in p]
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.policy.policy_factory import NonePolicy
    from crowd_sim.envs.utils.action import ActionXY
    from crowd_sim.envs.utils.robot import Robot

    return CrowdSim, NonePolicy, ActionXY, Robot


def build_env(
    crowdnav_root: Path,
    arm: str,
    human_num: int,
    scenario: str,
    circle_radius: Optional[float] = None,
    square_width: Optional[float] = None,
    time_limit: Optional[int] = None,
    occlusion: bool = True,
):
    CrowdSim, NonePolicy, _, Robot = _load_modules(crowdnav_root)
    source = crowdnav_root / "crowd_nav/configs/env.config"
    config = configparser.RawConfigParser()
    if not config.read(source):
        raise FileNotFoundError(source)
    config.set("sim", "human_num", str(human_num))
    config.set("sim", "test_sim", scenario)
    if circle_radius is not None:
        config.set("sim", "circle_radius", str(circle_radius))
    if square_width is not None:
        config.set("sim", "square_width", str(square_width))
    if time_limit is not None:
        if time_limit <= 0.0:
            raise ValueError("time_limit must be positive")
        config.set("env", "time_limit", str(time_limit))
    config.set("env", "randomize_attributes", "false")
    config.set("robot", "visible", "false")
    config.set("robot", "policy", "none")
    config.set("occlusion", "enabled", "true" if occlusion else "false")
    config.set("occlusion", "mode", "gt" if arm == "gt" else "sensor")
    config.set("occlusion", "max_entities", "10")
    config.set("occlusion", "token_contract", "belief_v3")
    config.set("occlusion", "hidden_slots", "10")

    env = CrowdSim()
    env.configure(config)
    policy = NonePolicy()
    policy.multiagent_training = True
    policy.kinematics = "holonomic"
    robot = Robot(config, "robot")
    robot.set_policy(policy)
    robot.visible = False
    env.set_robot(robot)
    env.phase = "test"
    return env, config, source


def swept_min_clearance(robot_start, robot_end, human_start, human_end, radii):
    """Independent audit of linear motion actually executed within one simulator step."""
    relative = np.asarray(human_start) - robot_start
    delta = np.asarray(human_end) - human_start - (robot_end - robot_start)
    length2 = np.einsum("ij,ij->i", delta, delta)
    fraction = np.clip(-np.einsum("ij,ij->i", relative, delta) / np.maximum(length2, 1e-30), 0., 1.)
    clearance = np.linalg.norm(relative + fraction[:, None] * delta, axis=1) - radii
    return float(clearance.min()) if len(clearance) else math.inf


def run_episode(
    crowdnav_root: Path,
    arm: str,
    human_num: int,
    scenario: str,
    case_id: int,
    planner_config: MPCConfig,
    circle_radius: Optional[float] = None,
    square_width: Optional[float] = None,
    planner_seed_offset: int = 0,
    position_noise_std: float = 0.0,
    velocity_noise_std: float = 0.0,
    detection_probability: float = 1.0,
    time_limit: Optional[int] = None,
    planner_type=ContinuousCEMMPC,
    adapter_factory=ObservationAdapter,
    step_observer=None,
    occlusion: bool = True,
) -> EpisodeResult:
    env, _, _ = build_env(
        crowdnav_root, arm, human_num, scenario, circle_radius, square_width,
        time_limit, occlusion,
    )
    _, _, ActionXY, _ = _load_modules(crowdnav_root)
    env.reset(options={"test_case": case_id})
    if env.robot.visible:
        raise RuntimeError("protocol violation: robot.visible must be false")

    planner = planner_type(planner_config)
    adapter = adapter_factory(
        arm,
        planner_config.horizon,
        planner_config.human_margin,
        planner_config.dt,
        planner_config.chance_limit,
        planner_config.fixed_uncertainty_radius,
        planner_config.acceleration_std,
        position_noise_std,
        velocity_noise_std,
        detection_probability,
        case_id * 1000003 + 7919,
        planner_config.conformal_visible_radii,
        planner_config.conformal_hidden_radii,
    )
    path_length = 0.0
    minimum_clearance = math.inf
    plan_times: List[float] = []
    pipeline_times = []
    actual_minimum = math.inf
    overlap_steps = 0
    first_overlap = None
    hashes: List[str] = []
    event = "timeout"
    terminated = truncated = False

    while not (terminated or truncated):
        cycle_start = time.perf_counter()
        observation = adapter.read(env)
        hashes.append(observation.provenance)
        step = int(round(env.global_time / env.time_step))
        seed = (
            case_id * 100003 + step * 97 + 1729
            + planner_seed_offset * 10000019
        )
        velocity, plan_ms = planner.plan(observation, seed)
        pipeline_times.append((time.perf_counter() - cycle_start) * 1000.)
        if not np.all(np.isfinite(velocity)) or np.linalg.norm(velocity) > planner_config.v_max + 1e-6:
            raise RuntimeError("invalid planner velocity")
        plan_times.append(plan_ms)
        if step_observer is not None:
            step_observer(env, observation, adapter, step)
        previous = np.array(env.robot.get_position(), dtype=np.float64)
        human_start = np.asarray([h.get_position() for h in env.humans], dtype=np.float64)
        radii = np.asarray([h.radius + env.robot.radius for h in env.humans])
        _, _, terminated, truncated, info = env.step(
            ActionXY(float(velocity[0]), float(velocity[1]))
        )
        current = np.array(env.robot.get_position(), dtype=np.float64)
        physical = swept_min_clearance(previous, current, human_start,
                    np.asarray([h.get_position() for h in env.humans]), radii)
        actual_minimum = min(actual_minimum, physical)
        if physical < -1e-9:
            overlap_steps += 1
            if first_overlap is None:
                first_overlap = float(env.global_time)
        path_length += float(np.linalg.norm(current - previous))
        minimum_clearance = min(minimum_clearance, float(info.get("dmin", math.inf)))
        event = str(info.get("event", "nothing"))

    success = int(event == "reach_goal")
    collision = int(event == "collision")
    timeout = int(event == "timeout")
    scored_time = float(env.global_time if success else env.time_limit)
    return EpisodeResult(
        arm=arm,
        scenario=scenario,
        human_num=human_num,
        case_id=case_id,
        event=event,
        success=success,
        collision=collision,
        timeout=timeout,
        nav_time=float(env.global_time),
        scored_time=scored_time,
        path_length=path_length,
        min_clearance=minimum_clearance,
        mean_plan_ms=float(np.mean(plan_times)),
        p95_plan_ms=float(np.percentile(plan_times, 95)),
        solver_steps=len(plan_times),
        observation_hash=hashlib.sha256("|".join(hashes).encode()).hexdigest(),
        actual_min_clearance=actual_minimum,
        actual_overlap_steps=overlap_steps,
        first_actual_overlap_time=first_overlap,
        collision_union=int(collision or overlap_steps > 0),
        success_without_overlap=int(success and overlap_steps == 0),
        mean_pipeline_ms=float(np.mean(pipeline_times)),
        p95_pipeline_ms=float(np.percentile(pipeline_times, 95)),
        deadline_miss_fraction=float(np.mean(np.asarray(pipeline_times) > planner_config.dt * 1000)),
        plan_step_ms=plan_times,
        pipeline_step_ms=pipeline_times,
    )


def verify_pairing(
    crowdnav_root: Path,
    human_num: int,
    scenario: str,
    circle_radius: Optional[float] = None,
    square_width: Optional[float] = None,
) -> Dict[str, str]:
    """Prove that arm configuration cannot change exogenous human motion."""
    hashes: Dict[str, str] = {}
    for arm in ("sensor", "gt", "worst"):
        env, _, _ = build_env(
            crowdnav_root, arm, human_num, scenario, circle_radius, square_width
        )
        _, _, ActionXY, _ = _load_modules(crowdnav_root)
        env.reset(options={"test_case": 0})
        trajectory = []
        for _ in range(8):
            trajectory.append(
                np.asarray(
                    [
                        [h.px, h.py, h.gx, h.gy, h.vx, h.vy, h.radius, h.v_pref]
                        for h in env.humans
                    ],
                    dtype=np.float64,
                )
            )
            env.step(ActionXY(0.0, 0.0))
        hashes[arm] = hashlib.sha256(np.stack(trajectory).tobytes()).hexdigest()
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"arm exogenous-trajectory mismatch: {hashes}")
    return hashes


def verify_access_contract(
    crowdnav_root: Path,
    human_num: int,
    scenario: str,
    circle_radius: Optional[float] = None,
    square_width: Optional[float] = None,
) -> Dict[str, object]:
    observations = {}
    visible_ids = None
    for arm in ("sensor", "gt", "worst"):
        env, _, _ = build_env(
            crowdnav_root, arm, human_num, scenario, circle_radius, square_width
        )
        env.reset(options={"test_case": 0})
        observation = ObservationAdapter(arm).read(env)
        observations[arm] = observation
        if arm == "sensor":
            visible_ids = tuple(sorted(env.occlusion.visible_ids))
            if observation.entities.shape[0] != len(visible_ids):
                raise RuntimeError("sensor entity count does not match visible IDs")
        if arm == "worst" and observation.unknown is None:
            raise RuntimeError("worst arm did not receive unknown-space geometry")
    if not np.array_equal(observations["sensor"].entities, observations["worst"].entities):
        raise RuntimeError("sensor and worst arms received different visible entities")
    if observations["gt"].entities.shape[0] != human_num:
        raise RuntimeError("GT arm did not receive every pedestrian")
    return {
        "visible_ids": visible_ids,
        "sensor_entities": int(observations["sensor"].entities.shape[0]),
        "gt_entities": int(observations["gt"].entities.shape[0]),
        "sensor_worst_equal": True,
    }


def summarize(results: Sequence[EpisodeResult]) -> List[Dict[str, float]]:
    output: List[Dict[str, float]] = []
    keys = sorted({(r.arm, r.scenario, r.human_num) for r in results})
    for arm, scenario, human_num in keys:
        rows = [
            r
            for r in results
            if (r.arm, r.scenario, r.human_num) == (arm, scenario, human_num)
        ]
        successful_times = [r.nav_time for r in rows if r.success]
        output.append(
            {
                "arm": arm,
                "scenario": scenario,
                "human_num": human_num,
                "episodes": len(rows),
                "sr": float(np.mean([r.success for r in rows])),
                "cr": float(np.mean([r.collision for r in rows])),
                "tr": float(np.mean([r.timeout for r in rows])),
                "arrival_time_success": (
                    float(np.mean(successful_times)) if successful_times else math.nan
                ),
                "failure_penalized_time": float(np.mean([r.scored_time for r in rows])),
                "path_length": float(np.mean([r.path_length for r in rows])),
                "min_clearance": float(np.mean([r.min_clearance for r in rows])),
                "mean_plan_ms": float(np.mean([r.mean_plan_ms for r in rows])),
                "p95_plan_ms": float(np.percentile([r.p95_plan_ms for r in rows], 95)),
            }
        )
    return output


def parse_ints(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> Tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _run_episode_task(task) -> EpisodeResult:
    return run_episode(*task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crowdnav-root", type=Path, default=DEFAULT_CROWDNAV)
    parser.add_argument("--arms", default="sensor,gt,worst")
    parser.add_argument("--human-counts", default="5")
    parser.add_argument("--scenarios", default="circle_crossing")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--population", type=int, default=768)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--v-max", type=float, default=1.0)
    parser.add_argument("--chance-limit", type=float, default=0.10)
    parser.add_argument("--near-chance-limit", type=float, default=None)
    parser.add_argument("--occupancy-chance-limit", type=float, default=0.15)
    parser.add_argument("--probability-weight", type=float, default=2500.0)
    parser.add_argument("--human-margin", type=float, default=0.16)
    parser.add_argument("--fixed-uncertainty-radius", type=float, default=0.35)
    parser.add_argument("--unknown-margin", type=float, default=0.05)
    parser.add_argument("--acceleration-std", type=float, default=0.55)
    parser.add_argument("--circle-radius", type=float, default=None)
    parser.add_argument("--square-width", type=float, default=None)
    parser.add_argument("--planner-seed-offset", type=int, default=0)
    parser.add_argument("--position-noise-std", type=float, default=0.0)
    parser.add_argument("--velocity-noise-std", type=float, default=0.0)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--conformal-visible-radii", default="")
    parser.add_argument("--conformal-hidden-radii", default="")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gate.json")
    args = parser.parse_args()

    arms = parse_strings(args.arms)
    human_counts = parse_ints(args.human_counts)
    scenarios = parse_strings(args.scenarios)
    cfg = MPCConfig(
        population=args.population,
        iterations=args.iterations,
        v_max=args.v_max,
        chance_limit=args.chance_limit,
        near_chance_limit=(
            args.chance_limit
            if args.near_chance_limit is None
            else args.near_chance_limit
        ),
        occupancy_chance_limit=args.occupancy_chance_limit,
        probability_weight=args.probability_weight,
        human_margin=args.human_margin,
        fixed_uncertainty_radius=args.fixed_uncertainty_radius,
        unknown_margin=args.unknown_margin,
        acceleration_std=args.acceleration_std,
        conformal_visible_radii=parse_floats(args.conformal_visible_radii),
        conformal_hidden_radii=parse_floats(args.conformal_hidden_radii),
    )
    pairing = {
        f"{scenario}:{count}": verify_pairing(
            args.crowdnav_root, count, scenario,
            args.circle_radius, args.square_width,
        )
        for scenario in scenarios
        for count in human_counts
    }
    access_contract = {
        f"{scenario}:{count}": verify_access_contract(
            args.crowdnav_root, count, scenario,
            args.circle_radius, args.square_width,
        )
        for scenario in scenarios
        for count in human_counts
    }
    tasks = [
        (
            args.crowdnav_root,
            arm,
            human_num,
            scenario,
            case_id,
            cfg,
            args.circle_radius,
            args.square_width,
            args.planner_seed_offset,
            args.position_noise_std,
            args.velocity_noise_std,
            args.detection_probability,
            args.time_limit,
        )
        for scenario in scenarios
        for human_num in human_counts
        for case_id in range(args.case_offset, args.case_offset + args.episodes)
        for arm in arms
    ]
    results: List[EpisodeResult] = []
    started = time.time()
    if args.workers == 1:
        iterator = map(_run_episode_task, tasks)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        )
        iterator = executor.map(_run_episode_task, tasks)
    try:
        for row in iterator:
            results.append(row)
            print(
                f"arm={row.arm:6s} scenario={row.scenario:15s} n={row.human_num:2d} "
                f"case={row.case_id:04d} event={row.event:10s} "
                f"time={row.nav_time:5.2f} path={row.path_length:5.2f} "
                f"plan={row.mean_plan_ms:6.1f}ms",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    source_config = args.crowdnav_root / "crowd_nav/configs/env.config"
    payload = {
        "schema": 1,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "protocol": {
            "robot_visible": False,
            "learned_controller": False,
            "paired_test_cases": True,
            "geometry": {
                "circle_radius": args.circle_radius,
                "square_width": args.square_width,
            },
            "planner_seed_offset": args.planner_seed_offset,
            "observation_model": {
                "position_noise_std": args.position_noise_std,
                "velocity_noise_std": args.velocity_noise_std,
                "detection_probability": args.detection_probability,
            },
            "time_limit": args.time_limit,
            "workers": args.workers,
            "failure_time": "environment time_limit",
            "crowdnav_root": str(args.crowdnav_root.resolve()),
            "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "belief_sha256": hashlib.sha256(
                (ROOT / "bayesian_rfs.py").read_bytes()
            ).hexdigest(),
            "planner": asdict(cfg),
            "pairing_hashes": pairing,
            "access_contract": access_contract,
            "elapsed_seconds": time.time() - started,
        },
        "episodes": [asdict(r) for r in results],
        "summary": summarize(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
