"""Experimental visibility-conditioned track posterior; not the frozen method.

The simulated sensor grid distinguishes observed free space, occupied returns and
unknown space.  A missed identity is negative spatial evidence only where the grid
explicitly reports free space.  In particular, the footprint of a visible occluder
is not treated as evidence against another pedestrian behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from bayesian_rfs import BayesianRFSBelief, BernoulliTrack
from continuous_mpc_gate import ObservationAdapter


def keyed_seed(*parts):
    data = ":".join(map(str, parts)).encode("ascii")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


def normals(count, dimension, seed):
    if count < 8 or count & (count - 1):
        raise ValueError("particle count must be a power of two >= 8")
    u = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(
        int(np.log2(count)))
    z = ndtri(np.clip(u, 1e-12, 1. - 1e-12))
    # Exact first two ensemble moments keep a Gaussian projection from adding
    # a mean/scale perturbation to the representation intervention.
    z -= z.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(z.T @ z / count)
    return z @ ((vectors / np.sqrt(eigenvalues)) @ vectors.T)


def covariance_root(covariance):
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.)
    if values.min() < -1e-8:
        raise ValueError("non-positive-semidefinite covariance")
    return vectors * np.sqrt(np.maximum(values, 0.))


def moments(states, weights):
    mean = weights @ states
    residual = states - mean
    return mean, (residual * weights[:, None]).T @ residual


def systematic_indices(weights, seed):
    n = len(weights)
    points = (np.arange(n) + np.random.default_rng(seed).random()) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    return np.searchsorted(cdf, points, side="right")


def missed_update(existence, weights, detection):
    likelihood = 1. - np.asarray(detection, dtype=float)
    if np.any((likelihood < 0.) | (likelihood > 1.)):
        raise ValueError("invalid detection probability")
    unnormalized = weights * likelihood
    eta = float(unnormalized.sum())
    evidence = 1. - existence + existence * eta
    if evidence <= 0.:
        raise ValueError("observation has zero probability under the prior")
    posterior_existence = existence * eta / evidence
    return posterior_existence, (unnormalized / eta if eta > 0. else None)


class DetectionSupport:
    """Conservative support for spatial negative evidence.

    A hypothetical pedestrian is contradicted only if part of its footprint lies
    in observed free space.  Unknown cells provide no evidence, while occupied
    cells may belong to an occluder and therefore cannot disprove the hypothesis.
    """

    def __init__(self, grid, mesh):
        grid = np.asarray(grid)
        if not np.all(np.isin(grid, (0., .5, 1.))):
            raise ValueError("sensor grid must contain only free/unknown/occupied labels")
        self.free = grid == 0.
        self.observed = grid != .5
        mx, my = mesh
        self.x0, self.y0 = float(mx[0, 0]), float(my[0, 0])
        self.dx, self.dy = float(mx[0, 1] - mx[0, 0]), float(my[1, 0] - my[0, 0])

    def _footprint_overlaps(self, mask, positions, radius):
        positions = np.asarray(positions)
        cx = np.floor((positions[:, 0] - self.x0) / self.dx + .5).astype(int)
        cy = np.floor((positions[:, 1] - self.y0) / self.dy + .5).astype(int)
        out = np.zeros(len(positions), dtype=bool)
        for dy in range(-int(np.ceil(radius / self.dy)) - 1,
                        int(np.ceil(radius / self.dy)) + 2):
            for dx in range(-int(np.ceil(radius / self.dx)) - 1,
                            int(np.ceil(radius / self.dx)) + 2):
                col, row = cx + dx, cy + dy
                inside = ((col >= 0) & (col < self.free.shape[1]) &
                          (row >= 0) & (row < self.free.shape[0]))
                distance2 = ((self.x0 + col * self.dx - positions[:, 0]) ** 2 +
                             (self.y0 + row * self.dy - positions[:, 1]) ** 2)
                idx = np.flatnonzero(inside & (distance2 <= radius * radius))
                out[idx] |= mask[row[idx], col[idx]]
        return out

    def eligible(self, positions, radius):
        """Whether the hypothesis overlaps explicit free-space evidence."""
        return self._footprint_overlaps(self.free, positions, radius)

    def observable(self, positions, radius):
        """Whether any footprint cell is in the sensor's observed region."""
        return self._footprint_overlaps(self.observed, positions, radius)


@dataclass
class ParticleState:
    states: np.ndarray
    weights: np.ndarray
    negative_streak: np.ndarray


class SpatialBelief(BayesianRFSBelief):
    """Sequential Monte Carlo with an optimal linear-Gaussian detection proposal.

    C and D run this identical filter. Projection for C happens only in the
    observation adapter. E uses a separately declared consecutive-miss deletion
    rule; it is not described as a Bayesian posterior.
    """

    def __init__(self, config, count=64, seed=0, hard_streak=None):
        self.count, self.seed, self.hard_streak = count, seed, hard_streak
        self.particles = {}
        self.diagnostics = {}
        super().__init__(config, mode="bayes")

    def reset(self):
        super().reset()
        self.particles = {}
        self.diagnostics = dict(missed_updates=0, mixed_support=0,
                                hard_exhaustions=0, resamples=0,
                                min_ess_fraction=1., visible_support_misses=0)

    def _draw(self, mean, covariance, identifier, step, label):
        return mean + normals(self.count, 4, keyed_seed(
            self.seed, identifier, step, label)) @ covariance_root(covariance).T

    def update(self, visible_detections, sensor_grid, mesh, timestamp):
        detections = list(visible_detections)
        by_id = {int(d["id"]): d for d in detections}
        if len(by_id) != len(detections):
            raise ValueError("duplicate identity in detections")
        step = int(round(timestamp / self.cfg.dt))
        elapsed = (0 if self._last_time is None else
                   int(round((timestamp - self._last_time) / self.cfg.dt)))
        if self._last_time is not None and elapsed <= 0:
            raise ValueError("duplicate or decreasing observation timestamp")
        transition, process = self._transition()
        measurement = np.diag([self.cfg.position_measurement_std ** 2] * 2 +
                              [self.cfg.velocity_measurement_std ** 2] * 2)
        support = DetectionSupport(sensor_grid, mesh)
        final_predictive = {}
        for j in range(elapsed):
            for identifier, track in self.tracks.items():
                particle = self.particles[identifier]
                predicted = particle.states @ transition.T
                if j == elapsed - 1:
                    final_predictive[identifier] = (predicted.copy(), particle.weights.copy())
                noise = normals(self.count, 2, keyed_seed(
                    self.seed, identifier, step - elapsed + j + 1, "motion"))
                accel = noise * self.cfg.acceleration_std
                particle.states = predicted + np.column_stack((
                    .5 * self.cfg.dt ** 2 * accel, self.cfg.dt * accel))
                track.existence *= self.cfg.survival_probability
                track.missed_steps += 1
                track.visible = False

        for identifier, detection in by_id.items():
            value = np.array([detection[k] for k in ("px", "py", "vx", "vy")])
            radius = float(detection["radius"])
            if not support.observable(value[None, :2], radius)[0]:
                # Noisy centroids can fall outside the eligibility mask. Record
                # this model mismatch instead of claiming a perfect sensor.
                self.diagnostics["visible_support_misses"] += 1
            if identifier not in self.tracks:
                self.tracks[identifier] = BernoulliTrack(
                    identifier, value.copy(), measurement.copy(), .999, radius=radius)
                states = self._draw(value, measurement, identifier, step, "birth")
            elif not np.any(measurement):
                states = np.broadcast_to(value, (self.count, 4)).copy()
            else:
                predicted, prior_weights = final_predictive[identifier]
                innovation = process + measurement
                delta = value - predicted
                log_weights = (np.log(np.maximum(prior_weights, 1e-300)) -
                               .5 * np.sum(delta * np.linalg.solve(innovation, delta.T).T, axis=1))
                weights = np.exp(log_weights - logsumexp(log_weights))
                gain = np.linalg.solve(innovation, process.T).T
                residual_gain = np.eye(4) - gain
                covariance = (residual_gain @ process @ residual_gain.T +
                              gain @ measurement @ gain.T)
                means = predicted + delta @ gain.T
                indices = systematic_indices(weights, keyed_seed(
                    self.seed, identifier, step, "detection_ancestors"))
                states = self._draw(means[indices], covariance, identifier, step, "detection")
            self.particles[identifier] = ParticleState(
                states, np.full(self.count, 1. / self.count), np.zeros(self.count, dtype=int))
            track = self.tracks[identifier]
            track.existence, track.missed_steps, track.visible, track.radius = .999, 0, True, radius

        for identifier, track in list(self.tracks.items()):
            particle = self.particles[identifier]
            if identifier not in by_id:
                eligible = support.eligible(particle.states[:, :2], track.radius)
                fraction = float(particle.weights @ eligible)
                self.diagnostics["missed_updates"] += 1
                self.diagnostics["mixed_support"] += int(.05 < fraction < .95)
                if self.hard_streak is None:
                    detection = eligible * self.cfg.detection_probability
                else:
                    particle.negative_streak = np.where(eligible, particle.negative_streak + 1, 0)
                    detection = (particle.negative_streak >= self.hard_streak).astype(float)
                existence, weights = missed_update(track.existence, particle.weights, detection)
                track.existence = existence
                if weights is None:
                    self.diagnostics["hard_exhaustions"] += 1
                    del self.tracks[identifier]
                    del self.particles[identifier]
                    continue
                particle.weights = weights
                ess = 1. / np.square(weights).sum()
                self.diagnostics["min_ess_fraction"] = min(
                    self.diagnostics["min_ess_fraction"], float(ess / self.count))
                if ess < self.count / 2:
                    indices = systematic_indices(weights, keyed_seed(
                        self.seed, identifier, step, "miss_resample"))
                    particle.states = particle.states[indices].copy()
                    particle.negative_streak = particle.negative_streak[indices].copy()
                    particle.weights = np.full(self.count, 1. / self.count)
                    self.diagnostics["resamples"] += 1
            track.mean, track.covariance = moments(particle.states, particle.weights)
            if (track.existence < self.cfg.minimum_existence or
                    track.missed_steps > self.cfg.maximum_missed_steps):
                del self.tracks[identifier]
                del self.particles[identifier]
        self._last_time = float(timestamp)


class SpatialAdapter(ObservationAdapter):
    def __init__(self, *args, representation="shape", count=64,
                 hard_streak=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rfs is None or self.arm != "bayes":
            raise ValueError("spatial prototype requires the legal-history Bayes arm")
        if representation not in ("original_gaussian", "projected", "shape"):
            raise ValueError(representation)
        self.representation, self.count = representation, count
        if representation != "original_gaussian":
            self.rfs = SpatialBelief(self.rfs.cfg, count, self.observation_seed, hard_streak)

    def read(self, env):
        obs = super().read(env)
        step = int(round(env.global_time / self.dt))
        transition, _ = self.rfs._transition()
        n = len(self.reported_ids)
        states = np.empty((n, self.count, self.horizon + 1, 4))
        weights = np.empty((n, self.count))
        accelerations = np.empty((n, self.count, self.horizon, 2))
        for i, identifier in enumerate(self.reported_ids):
            track = self.rfs.tracks[identifier]
            if self.representation == "shape":
                particle = self.rfs.particles[identifier]
                initial, weight = particle.states.copy(), particle.weights.copy()
            else:
                initial = track.mean + normals(self.count, 4, keyed_seed(
                    self.observation_seed, identifier, step, "projection")) @ covariance_root(track.covariance).T
                weight = np.full(self.count, 1. / self.count)
            states[i, :, 0] = initial
            weights[i] = weight
            for k in range(self.horizon):
                a = self.rfs.cfg.acceleration_std * normals(self.count, 2, keyed_seed(
                    self.observation_seed, identifier, step, k, "forecast"))
                accelerations[i, :, k] = a
                states[i, :, k + 1] = states[i, :, k] @ transition.T + np.column_stack((
                    .5 * self.dt ** 2 * a, self.dt * a))
        obs.posterior_states = states
        obs.posterior_weights = weights
        obs.posterior_accelerations = accelerations
        obs.posterior_count = self.count
        return obs
