"""Reference swept-posterior risk interface for the spatial-belief experiment.

No probability-field cache or GPU approximation is used. Conditional on the finite
weighted trajectory set, every sample is checked over the execution interval.
Chord error is bounded geometrically; particle approximation error is separate.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from modern_dynamics import ContinuousUnicycleMPC, ramp_displacement


@njit
def segment_distance(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    t = 0. if denominator < 1e-24 else min(1., max(0., -(ax * dx + ay * dy) / denominator))
    return ((ax + t * dx) ** 2 + (ay + t * dy) ** 2) ** .5


@njit(parallel=True)
def swept_statistics(robot_nodes, robot_acceleration_bound, human_nodes,
                     human_acceleration_bound, weights, existence, radii,
                     margin, dt):
    population, horizon, nodes, _ = robot_nodes.shape
    people, samples = human_nodes.shape[:2]
    probabilities = np.zeros((population, people, horizon))
    penetration_cost = np.zeros_like(probabilities)
    discomfort_cost = np.zeros_like(probabilities)
    # Bounding boxes only skip cases with identically zero sample cost/risk.
    bounds = np.empty((people, horizon, 4))
    for i in range(people):
        for k in range(horizon):
            bounds[i, k, 0] = np.min(human_nodes[i, :, k, :, 0])
            bounds[i, k, 1] = np.max(human_nodes[i, :, k, :, 0])
            bounds[i, k, 2] = np.min(human_nodes[i, :, k, :, 1])
            bounds[i, k, 3] = np.max(human_nodes[i, :, k, :, 1])
    for p in prange(population):
        for k in range(horizon):
            rx0 = np.min(robot_nodes[p, k, :, 0])
            rx1 = np.max(robot_nodes[p, k, :, 0])
            ry0 = np.min(robot_nodes[p, k, :, 1])
            ry1 = np.max(robot_nodes[p, k, :, 1])
            rb = robot_acceleration_bound[p, k]
            for i in range(people):
                max_human_bound = np.max(human_acceleration_bound[i, :, k])
                sub_error = (rb + max_human_bound) * (dt / (nodes - 1)) ** 2 / 8.
                dx = max(0., bounds[i, k, 0] - rx1, rx0 - bounds[i, k, 1])
                dy = max(0., bounds[i, k, 2] - ry1, ry0 - bounds[i, k, 3])
                if (dx * dx + dy * dy) ** .5 > radii[i] + margin + sub_error:
                    continue
                for s in range(samples):
                    if weights[i, s] <= 0.:
                        continue
                    error = (rb + human_acceleration_bound[i, s, k]) * dt ** 2 / 8.
                    a = robot_nodes[p, k, 0] - human_nodes[i, s, k, 0]
                    b = robot_nodes[p, k, -1] - human_nodes[i, s, k, -1]
                    coarse = segment_distance(a[0], a[1], b[0], b[1])
                    if coarse > radii[i] + margin + error:
                        continue
                    distance = np.inf
                    for j in range(nodes - 1):
                        a = robot_nodes[p, k, j] - human_nodes[i, s, k, j]
                        b = robot_nodes[p, k, j + 1] - human_nodes[i, s, k, j + 1]
                        distance = min(distance, segment_distance(a[0], a[1], b[0], b[1]))
                    clearance = distance - error / (nodes - 1) ** 2 - radii[i]
                    w = weights[i, s] * existence[i]
                    if clearance <= margin:
                        probabilities[p, i, k] += w
                    penetration_cost[p, i, k] += w * max(-clearance, 0.) ** 2
                    discomfort_cost[p, i, k] += w * max(margin - clearance, 0.) ** 2
    return probabilities, penetration_cost, discomfort_cost


def swept_statistics_torch(robot_nodes, robot_acceleration_bound, human_nodes,
                           human_acceleration_bound, weights, existence, radii,
                           margin, dt, device="cuda", chunk_size=48):
    """Vectorized accelerator implementation of :func:`swept_statistics`.

    The finite posterior, curve nodes and conservative chord-error correction
    are identical to the CPU reference. Population chunks only bound temporary
    GPU memory; they do not change the calculation.
    """
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA swept-risk backend requested but CUDA is unavailable")
    dtype = torch.float32
    target = torch.device(device)
    human = torch.as_tensor(human_nodes, dtype=dtype, device=target)
    human_bound = torch.as_tensor(
        human_acceleration_bound, dtype=dtype, device=target)
    sample_weight = torch.as_tensor(
        weights * existence[:, None], dtype=dtype, device=target)
    collision_radius = torch.as_tensor(radii, dtype=dtype, device=target)
    outputs = [[], [], []]
    subdivisions = robot_nodes.shape[2] - 1
    error_scale = float(dt) ** 2 / (8. * subdivisions ** 2)

    with torch.inference_mode():
        for begin in range(0, len(robot_nodes), chunk_size):
            end = min(begin + chunk_size, len(robot_nodes))
            robot = torch.as_tensor(
                robot_nodes[begin:end], dtype=dtype, device=target)
            robot_bound = torch.as_tensor(
                robot_acceleration_bound[begin:end], dtype=dtype, device=target)
            relative = robot[:, None, None] - human[None]
            start = relative[..., :-1, :]
            segment = relative[..., 1:, :] - start
            denominator = torch.sum(segment * segment, dim=-1).clamp_min_(1e-24)
            fraction = (-torch.sum(start * segment, dim=-1) / denominator).clamp_(0., 1.)
            closest = start + fraction[..., None] * segment
            distance = torch.sqrt(torch.sum(closest * closest, dim=-1).min(dim=-1).values)
            error = (robot_bound[:, None, None, :] + human_bound[None]) * error_scale
            clearance = distance - error - collision_radius[None, :, None, None]
            weighted = sample_weight[None, :, :, None]
            probability = ((clearance <= margin) * weighted).sum(dim=2)
            penetration = (torch.clamp(-clearance, min=0.) ** 2 * weighted).sum(dim=2)
            discomfort = (torch.clamp(margin - clearance, min=0.) ** 2 * weighted).sum(dim=2)
            for bucket, value in zip(outputs, (probability, penetration, discomfort)):
                bucket.append(value.cpu().numpy().astype(np.float64, copy=False))
    return tuple(np.concatenate(parts, axis=0) for parts in outputs)


def human_path_nodes(obs, dt, subdivisions):
    times = np.linspace(0., dt, subdivisions + 1)
    state = obs.posterior_states[:, :, :-1]
    return (state[:, :, :, None, :2] + state[:, :, :, None, 2:] * times[None, None, None, :, None] +
            .5 * obs.posterior_accelerations[:, :, :, None] * times[None, None, None, :, None] ** 2)


class SpatialRiskMPC(ContinuousUnicycleMPC):
    """The same CEM with one coherent uncertainty-consuming obstacle interface."""

    def __init__(self, config, subdivisions=8, backend="numba"):
        super().__init__(config)
        if subdivisions < 2:
            raise ValueError("swept audit needs at least two subdivisions")
        if backend not in ("numba", "torch_cuda"):
            raise ValueError(f"unknown swept-risk backend {backend!r}")
        self.subdivisions = subdivisions
        self.backend = backend
        self._stats = None

    def _rollout(self, samples, obs):
        params, controls, positions = super()._rollout(samples, obs)
        self._stats = None
        times = np.linspace(0., 1., self.subdivisions + 1)
        headings = self._heading(obs) + np.cumsum(params[:, :, 1], axis=1) - params[:, :, 1]
        speeds = np.concatenate((np.full((len(params), 1), self._speed(obs)), params[:, :-1, 0]), axis=1)
        starts = np.concatenate((np.broadcast_to(obs.robot_xy, (len(params), 1, 2)), positions[:, :-1]), axis=1)
        self._robot_nodes = starts[:, :, None] + ramp_displacement(
            headings[:, :, None], speeds[:, :, None],
            speeds[:, :, None] + times * (params[:, :, 0] - speeds)[:, :, None],
            times * params[:, :, 1, None], times * self.cfg.dt)
        self._robot_bound = np.hypot((params[:, :, 0] - speeds) / self.cfg.dt,
                                    np.maximum(params[:, :, 0], speeds) * params[:, :, 1] / self.cfg.dt)
        return params, controls, positions

    def _evaluate(self, obs):
        if self._stats is None:
            people = len(obs.entities)
            if people:
                function = (swept_statistics_torch if self.backend == "torch_cuda"
                            else swept_statistics)
                self._stats = function(
                    self._robot_nodes, self._robot_bound, self._human_nodes,
                    self._human_bound, obs.posterior_weights, obs.human_existence,
                    obs.robot_radius + obs.entities[:, 4], self.cfg.human_margin, self.cfg.dt)
            else:
                shape = (len(self._robot_nodes), 0, self.cfg.horizon)
                self._stats = tuple(np.zeros(shape) for _ in range(3))
        return self._stats

    def _human_clearance(self, controls, obs):
        # Nominal geometry remains only for presently detected entities. Hidden
        # means are not an additional obstacle in any of the three consumers.
        clearance = super()._human_clearance(controls, obs)
        clearance[:, ~obs.human_visible, :] = np.inf
        return clearance

    def _belief_collision_hazard(self, controls, obs, positions=None):
        probability = self._evaluate(obs)[0]
        return -np.log1p(-np.clip(probability, 0., 1. - 1e-12)).sum(axis=1)

    def _cost(self, controls, obs, positions, human_clearance, occupancy_probability, belief_hazard):
        cost = super()._cost(controls, obs, positions, human_clearance,
                             occupancy_probability, belief_hazard)
        _, penetration, discomfort = self._evaluate(obs)
        active = self._active_until_goal(positions, obs)[0][:, None, :]
        hidden = ~obs.human_visible
        cost += self.cfg.collision_weight * (penetration[:, hidden] * active).sum(axis=(1, 2))
        cost += self.cfg.discomfort_weight * (discomfort[:, hidden] * active).sum(axis=(1, 2))
        return cost

    def _physical_clearance(self, controls, obs, positions, human_clearance):
        visible_full, visible_first = super()._physical_clearance(
            controls, obs, positions, human_clearance)
        hazard = self._belief_collision_hazard(controls, obs, positions)
        probability = -np.expm1(-hazard)
        active = self._active_until_goal(positions, obs)[0]
        # This is a dimensionless fallback score, not a claimed physical
        # clearance: first avoid nominal visible contacts, then minimize risk.
        full_score = -2. * (visible_full < 0.) - np.where(active, probability, 0.).max(axis=1)
        first_score = -2. * (visible_first < 0.) - probability[:, 0]
        return full_score, first_score

    def plan(self, obs, seed):
        if obs.human_visible is None or obs.human_existence is None:
            raise ValueError("spatial risk requires explicit track visibility/existence")
        self._human_nodes = human_path_nodes(obs, self.cfg.dt, self.subdivisions)
        self._human_bound = np.linalg.norm(obs.posterior_accelerations, axis=-1)
        action, elapsed = super().plan(obs, seed)
        self.last_diagnostics["fallback_score_is_dimensionless"] = 1.
        self.last_diagnostics["posterior_samples"] = float(obs.posterior_count)
        self.last_diagnostics["sweep_subdivisions"] = float(self.subdivisions)
        self.last_diagnostics["sweep_backend_cuda"] = float(self.backend == "torch_cuda")
        return action, elapsed
