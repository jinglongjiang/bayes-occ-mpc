"""One holonomic CEM loop for exact and envelope-guided evaluation."""
from __future__ import annotations
import math
import time
from typing import Dict, Optional, Tuple
import numpy as np
from .contracts import MPCConfig, PlannerObservation
from .risk import CompiledDiscRisk, exact_hazard, _keys, _le_key


class MPCPlanner:
    """Production defaults to Hermite; experiments can supply an envelope or None."""

    def __init__(self, config: MPCConfig, envelope=CompiledDiscRisk):
        self.cfg = config
        if config.population < 32:
            raise ValueError("route search needs at least 32 samples")
        if config.probability_weight < 0 or not np.isfinite(config.probability_weight):
            raise ValueError("finite nonnegative risk weight required")
        self.envelope = envelope
        self._previous_mean = None
        self.last_controls = None
        self.last_diagnostics = {}
        self.trace_hook = None
        self.bound_fallbacks = 0

    def _validate_observation(self, obs):
        for name in ("robot_xy", "robot_velocity", "goal_xy", "entities",
                     "human_segment_start", "human_segment_end",
                     "human_position_covariance", "human_existence",
                     "human_uncertainty_buffer"):
            value = getattr(obs, name)
            if value is not None and not np.isfinite(value).all():
                raise ValueError("nonfinite observation: " + name)
        if not np.isfinite(obs.robot_radius) or obs.robot_radius < 0:
            raise ValueError("invalid robot radius")
        if obs.entities.size and (obs.entities[:, 4] < 0).any():
            raise ValueError("negative human radius")
        if obs.human_existence is not None:
            if ((obs.human_existence < 0) | (obs.human_existence > 1)).any():
                raise ValueError("existence outside [0, 1]")
        if obs.human_position_covariance is not None:
            covariance = obs.human_position_covariance
            if covariance.shape != (len(obs.entities), self.cfg.horizon, 2, 2):
                raise ValueError("invalid covariance shape")
            if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-12, rtol=0):
                raise ValueError("asymmetric covariance")
            if (np.linalg.eigvalsh(covariance) < -1e-12).any():
                raise ValueError("negative covariance")
            if not np.allclose(covariance, covariance[..., :1, :1]*np.eye(2), atol=1e-12, rtol=0):
                raise ValueError("Gaussian-disc evaluator requires isotropic covariance")
        if self.cfg.existence_override is not None:
            if not 0 <= self.cfg.existence_override <= 1:
                raise ValueError("existence override outside [0, 1]")

    def _build_envelope(self, obs):
        if (self.envelope is None or not obs.entities.size
                or obs.human_position_covariance is None or obs.human_existence is None):
            return None
        try:
            return self.envelope(self.cfg, obs)
        except (FloatingPointError, OverflowError):
            self.bound_fallbacks += 1
            return None

    def _belief_collision_hazard(self, controls, obs, positions=None):
        self.risk_rows += len(controls)
        h = exact_hazard(self.cfg, controls, obs, positions)
        if h is not None and not np.isfinite(h).all():
            raise ValueError("nonfinite exact hazard")
        return h

    def _score_from_hazard(self, base_cost, geo_full, geo_first, active, limits, h):
        cost = base_cost.copy()
        if h is None:
            return cost, geo_full, geo_first
        cost += self.cfg.probability_weight*(h*active).sum(axis=1)
        cost += 1e5*(np.where(active, h-limits, -math.inf).max(axis=1) > 0.)
        slack = np.where(active, limits-h, math.inf)
        full = np.minimum(geo_full, slack.min(axis=1))
        first = np.minimum(geo_first, slack[:, 0])
        return cost, full, first


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
            * np.power(phase, self.cfg.risk_gamma)
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

    def _rollout(self, samples: np.ndarray, obs: PlannerObservation):
        """Sampled parameters to (executable parameters, XY velocities, positions).

        The holonomic model is parameterised directly by its velocities, so the
        first two returns are the same array.  A different kinematic model
        overrides this and leaves every cost and clearance term untouched: they
        only ever see XY velocities and the positions they produce.
        """
        controls = self._project_controls(samples, obs.robot_velocity)
        positions = obs.robot_xy[None, None, :] + np.cumsum(
            controls * self.cfg.dt, axis=1
        )
        return controls, controls, positions

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

        return cost

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

    def _degraded_choice(self, costs, first_physical, params, obs) -> int:
        """Which candidate to execute when nothing is feasible.

        The registered policy keeps moving: among the candidates that hold the
        largest physical clearance at the first step, take the cheapest.  The
        modern comparators instead brake on an infeasible solve, and whether that
        difference matters is an experimental question, so subclasses override
        this rather than the whole optimiser.
        """
        safest = float(first_physical.max())
        near_safest = np.flatnonzero(first_physical >= safest - 0.02)
        return int(near_safest[np.argmin(costs[near_safest])])

    def _evaluate_batch(self, controls, obs, positions, clearance, occupancy, counts, bounds):
        cfg = self.cfg
        # Every non-probabilistic quantity is evaluated as in the reference.
        base_cost = self._cost(controls, obs, positions, clearance, occupancy, None)
        geo_full, geo_first = self._combined_clearance(
            controls, obs, positions, clearance, occupancy, None)
        _, first_physical = self._physical_clearance(controls, obs, positions, clearance)
        active, _, _ = self._active_until_goal(positions, obs)
        limits = self._belief_hazard_limits()[None, :]
        n = len(controls)
        exact = np.zeros(n, bool)
        self._exact_mask = exact

        def evaluate(h):
            return self._score_from_hazard(
                base_cost, geo_full, geo_first, active, limits, h)

        if self._compiled is None:
            h = self._belief_collision_hazard(controls, obs, positions)
            exact[:] = True
            cost, full, first = evaluate(h)
            return cost, full, first, first_physical
        try:
            hlo, hhi = self._compiled.bounds(positions)
            valid = (hlo.shape == active.shape and hhi.shape == active.shape
                     and np.isfinite(hlo).all() and np.isfinite(hhi).all()
                     and (hlo <= hhi).all() and (hlo >= 0.).all())
        except (FloatingPointError, OverflowError):
            valid = False
        if not valid:
            self.bound_fallbacks += 1
            self._compiled = None
            h = self._belief_collision_hazard(controls, obs, positions)
            exact[:] = True
            cost, full, first = evaluate(h)
            return cost, full, first, first_physical

        def refine(mask):
            ids = np.flatnonzero(mask & ~exact)
            if not len(ids):
                return
            self.refinement_batches += 1
            h = self._belief_collision_hazard(controls[ids], obs, positions[ids])
            hlo[ids] = h
            hhi[ids] = h
            exact[ids] = True

        clo, flo, slo = evaluate(hlo)  # optimistic cost/clearance
        chi, fhi, shi = evaluate(hhi)  # pessimistic cost/clearance
        lower_keys = _keys(clo, flo, slo)
        upper_keys = _keys(chi, fhi, shi)
        keep = np.zeros(n, bool)
        for route, count in enumerate(counts):
            begin, end = bounds[route:route+2]
            k = max(4, round(count*cfg.elite_fraction))
            ck,vk,jk = (a[begin:end] for a in upper_keys)
            kth = np.lexsort((jk,vk,ck))[k-1] + begin
            cut = tuple(a[kth] for a in upper_keys)
            keep[begin:end] |= _le_key(tuple(a[begin:end] for a in lower_keys), cut)

        # Also retain the possible iteration winner, independently of elites.
        # This matters because final selection and elite selection differ when
        # the reference uses a 0.02 clearance tolerance.
        possible_full = flo >= 0.
        definite_full = fhi >= 0.
        if definite_full.any():
            ceiling = chi[definite_full].min()
            keep |= possible_full & (clo <= ceiling)
        else:
            keep |= possible_full
        refine(keep)
        cost, full, first = evaluate(hlo)
        if not (full >= 0.).any():
            # All potentially full-feasible candidates have been resolved.
            # Preserve the class-1 maximum-clearance and tolerance pool.
            chi, fhi, shi = evaluate(hhi)
            possible_first = first >= 0.
            definite_first = shi >= 0.
            if possible_first.any():
                best_lower = fhi[definite_first].max() if definite_first.any() else -math.inf
                refine(possible_first & (full >= best_lower - .02))
                cost, full, first = evaluate(hlo)
            if not (first >= 0.).any():
                safest = float(first_physical.max())
                physical_pool = first_physical >= safest - .02
                upper_cost, _, _ = evaluate(hhi)
                ceiling = upper_cost[physical_pool].min()
                refine(physical_pool & (cost <= ceiling))
                cost, full, first = evaluate(hlo)
        # Unrefined rows retain optimistic bounds. The certificates establish
        # that these rows cannot be selected as elites or as iteration winner.
        return cost, full, first, first_physical

    def plan(self, obs: PlannerObservation, seed: int) -> Tuple[np.ndarray,float]:
        self.elite_trace = []
        self.risk_rows = 0
        self.refinement_batches = 0
        self.bound_fallbacks = 0
        self._validate_observation(obs)
        cfg = self.cfg
        self.total_rows = cfg.population*cfg.iterations
        rng = np.random.default_rng(seed)
        mean = self._initial_mean(obs)
        start = time.perf_counter()
        self._compiled = self._build_envelope(obs)
        self.table_cdf_values = self._compiled.table_cdf_values if self._compiled is not None else 0
        seeds = self._seed_trajectories(obs)
        route_seeds = self._route_seeds(obs)
        means = np.concatenate((mean[None,:,:], route_seeds), axis=0)
        deviations = np.full_like(means, cfg.init_std)
        n_routes = int(route_seeds.shape[0])
        if n_routes:
            remaining = cfg.population - cfg.population//2
            counts = [cfg.population//2] + [remaining//n_routes]*n_routes
            for i in range(remaining % n_routes): counts[i+1] += 1
        else:
            counts = [cfg.population]
        bounds = np.cumsum([0]+counts)
        mode_ids = np.repeat(np.arange(len(counts)), counts)
        best_controls, best_cost, best_class = seeds[0].copy(), math.inf, 3
        best_first_clearance = best_full_clearance = best_first_physical = -math.inf
        for _ in range(cfg.iterations):
            noise = rng.standard_normal((cfg.population,cfg.horizon,2))
            noise[:,1:] = .68*noise[:,:-1] + .32*noise[:,1:]
            samples = means[mode_ids]+deviations[mode_ids]*noise
            samples[:len(seeds)] = seeds
            samples[len(seeds)] = mean
            for route in range(1,len(counts)):
                samples[bounds[route]] = means[route]
                samples[bounds[route]+1] = route_seeds[route-1]
            params, controls, positions = self._rollout(samples,obs)
            human_clearance = self._human_clearance(controls,obs) if obs.entities.size else None
            occupancy = obs.occupancy_probability.sample(positions) if obs.occupancy_probability is not None else None
            costs, full, first, first_physical = self._evaluate_batch(
                controls,obs,positions,human_clearance,occupancy,counts,bounds)
            full_feasible, first_feasible = full >= 0., first >= 0.
            if full_feasible.any():
                pool = np.flatnonzero(full_feasible)
                ibest = int(pool[np.argmin(costs[pool])]); iclass = 0
            elif first_feasible.any():
                pool = np.flatnonzero(first_feasible)
                safest = float(full[pool].max())
                near = pool[full[pool] >= safest-.02]
                ibest = int(near[np.argmin(costs[near])]); iclass = 1
            else:
                ibest = self._degraded_choice(costs,first_physical,params,obs); iclass = 2
            if not self._exact_mask[ibest]:
                raise RuntimeError("iteration winner was not exactly evaluated")
            candidate_cost, candidate_first = float(costs[ibest]), float(first[ibest])
            replace = iclass < best_class
            if iclass == best_class == 2:
                physical = float(first_physical[ibest])
                replace = physical > best_first_physical+.02 or (physical >= best_first_physical-.02 and candidate_cost < best_cost)
            elif iclass == best_class == 1:
                replace = float(full[ibest]) > best_full_clearance+.02 or (float(full[ibest]) >= best_full_clearance-.02 and candidate_cost < best_cost)
            elif iclass == best_class:
                replace = candidate_cost < best_cost
            if replace:
                best_class, best_cost = iclass, candidate_cost
                best_first_clearance, best_full_clearance = candidate_first, float(full[ibest])
                best_first_physical = float(first_physical[ibest])
                best_controls = params[ibest].copy()
            for route,count in enumerate(counts):
                begin,end = bounds[route:route+2]
                ids = self._elite_indices(costs[begin:end], full[begin:end], first[begin:end],
                                          max(4,round(count*cfg.elite_fraction)))+begin
                if not self._exact_mask[ids].all():
                    raise RuntimeError("elite was not exactly evaluated")
                elite = params[ids]
                means[route] = .22*means[route]+.78*elite.mean(axis=0)
                deviations[route] = np.maximum(cfg.min_std,elite.std(axis=0))
            if self.trace_hook is not None:
                self.trace_hook(locals())
        action = best_controls[0]
        self._previous_mean = best_controls
        self.last_controls = best_controls.copy()
        self.last_diagnostics = dict(feasibility_class=float(best_class),
            first_clearance=best_first_clearance,horizon_clearance=best_full_clearance,
            first_physical_clearance=best_first_physical,cost=best_cost,
            exact_risk_rows=float(self.risk_rows), total_candidate_rows=float(self.total_rows),
            refinement_batches=float(self.refinement_batches),
            table_cdf_values=float(self.table_cdf_values))
        return action,(time.perf_counter()-start)*1000.
