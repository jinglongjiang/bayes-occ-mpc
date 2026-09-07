"""One-event, binary re-detection contingency MPC research pilot.

Only one tracked, currently missed pedestrian is branched. Planning deliberately
coarsens the future observation to detection/no-detection; it does not pretend
to know the future measured position. Other tracks use their predictive means
as occluders. This angular visibility approximation is not the sensor renderer.
"""
from dataclasses import dataclass, replace
import math
import time

import numpy as np
from scipy.special import ndtr
from scipy.stats import ncx2

from continuous_mpc_gate import ContinuousCEMMPC, PlannerObservation
from evaluate_matched_safety import MatchedAdapter


@dataclass
class ReobservationObservation(PlannerObservation):
    state_covariances: np.ndarray = None
    missed: np.ndarray = None
    detection_probability: float = 1.0
    view_radius: float = 5.0
    occlusion_enabled: bool = True


class ReobservationAdapter(MatchedAdapter):
    def read(self, env):
        observation = super().read(env)
        tracks = [self.rfs.tracks[i] for i in self.reported_ids]
        return ReobservationObservation(
            **vars(observation),
            state_covariances=np.array([t.covariance for t in tracks]).reshape(-1, 4, 4),
            missed=np.array([not t.visible for t in tracks], dtype=bool),
            detection_probability=self.detection_probability,
            view_radius=float(env.occlusion.fov_radius) if env.occlusion_enabled() else math.inf,
            occlusion_enabled=env.occlusion_enabled(),
        )


def predictive_cloud(mean, covariance, horizon, dt, acceleration_std, seed, count=128):
    """Moment-matched joint CV trajectories, including correlated velocities."""
    dimensions = 4 + 2 * horizon
    if count <= dimensions:
        raise ValueError('Not enough samples to preserve the joint prior moments')
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((count, dimensions))
    noise -= noise.mean(axis=0)
    noise = np.linalg.qr(noise, mode='reduced')[0] * np.sqrt(count)
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-8:
        raise ValueError('Non-PSD track covariance')
    root = vectors * np.sqrt(np.maximum(values, 0.))
    state = mean + noise[:, :4] @ root.T
    states = [state.copy()]
    for k in range(horizon):
        acceleration = noise[:, 4+2*k:6+2*k] * acceleration_std
        state[:, :2] += state[:, 2:] * dt + .5 * dt**2 * acceleration
        state[:, 2:] += dt * acceleration
        states.append(state.copy())
    return np.stack(states, axis=1)


def detection_likelihood(robot_positions, targets, target_radius, occluders,
                         radii, view_radius, detection_probability, occlusion=True):
    """Binary sensor likelihood using disc angular coverage, never true states."""
    delta = targets[None, :, :] - robot_positions[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    detectable = distance <= view_radius + target_radius
    if occlusion and len(occluders):
        others = occluders[None, :, :] - robot_positions[:, None, :]
        other_distance = np.linalg.norm(others, axis=-1)
        angle = np.arctan2(delta[..., 1], delta[..., 0])
        other_angle = np.arctan2(others[..., 1], others[..., 0])
        width = np.arcsin(np.minimum(1., target_radius / np.maximum(distance, 1e-9)))
        other_width = np.arcsin(np.minimum(1., radii / np.maximum(other_distance, 1e-9)))
        gap = np.abs(np.arctan2(np.sin(angle[:, :, None] - other_angle[:, None, :]),
                               np.cos(angle[:, :, None] - other_angle[:, None, :])))
        covered = ((gap + width[:, :, None] <= other_width[:, None, :]) &
                   (other_distance[:, None, :] < distance[:, :, None]))
        detectable &= ~covered.any(axis=-1)
    return detectable.astype(float) * detection_probability


def condition_cloud(cloud, likelihood, existence):
    """Bayes update for re-detection / no detection, including absence mass."""
    sample_likelihood = np.stack((likelihood, 1. - likelihood), axis=1)
    normalizer = sample_likelihood.mean(axis=-1)
    probabilities = existence * normalizer
    probabilities[:, 1] += 1. - existence
    posterior_existence = np.divide(existence * normalizer, probabilities,
                                   out=np.zeros_like(probabilities), where=probabilities > 0.)
    weights = sample_likelihood / np.maximum(sample_likelihood.sum(axis=-1, keepdims=True), 1e-15)
    empty = normalizer == 0.
    weights[empty] = 1. / len(cloud)
    xy = cloud[:, :, :2]
    means = np.einsum('tbw,whd->tbhd', weights, xy)
    seconds = np.einsum('tbw,whd,whe->tbhde', weights, xy, xy)
    covariances = seconds - means[..., :, None] * means[..., None, :]
    covariances = .5 * (covariances + covariances.swapaxes(-1, -2))
    return probabilities, posterior_existence, means, covariances


class ReobservationCEMMPC(ContinuousCEMMPC):
    event_step = 2
    branch_count = 2

    def _condition(self, cloud, likelihood, existence):
        return condition_cloud(cloud, likelihood, existence)

    def _target(self, obs):
        if not isinstance(obs, ReobservationObservation):
            raise TypeError('Re-observation planning requires a causal track snapshot')
        candidates = np.flatnonzero(obs.missed)
        if not len(candidates):
            return None
        distance = np.linalg.norm(obs.human_segment_end[candidates] - obs.robot_xy, axis=-1).min(axis=1)
        return int(candidates[np.argmin(distance)])

    def _branch_metrics(self, trees, obs, target, cloud):
        cfg, k = self.cfg, self.event_step
        count = len(trees)
        controls = trees.reshape(-1, cfg.horizon, 2)
        positions = obs.robot_xy + np.cumsum(controls * cfg.dt, axis=1)
        root_positions = positions.reshape(count, 2, cfg.horizon, 2)[:, 0, k-1]
        others = np.arange(len(obs.entities)) != target
        others &= obs.human_existence >= .5
        likelihood = detection_likelihood(root_positions, cloud[:, k, :2], obs.entities[target, 4],
            obs.human_segment_end[others, k-1], obs.entities[others, 4], obs.view_radius,
            obs.detection_probability, obs.occlusion_enabled)
        probabilities, existence, means, covariances = self._condition(
            cloud, likelihood, obs.human_existence[target])
        means = means.reshape(-1, cfg.horizon+1, 2)
        covariances = covariances.reshape(-1, cfg.horizon+1, 2, 2)
        posterior_existence = existence.reshape(-1)

        human_clearance = super()._human_clearance(controls, obs)
        robot_start = np.concatenate((np.broadcast_to(obs.robot_xy, (len(controls), 1, 2)),
                                      positions[:, :-1]), axis=1)
        relative_start = means[:, :-1] - robot_start
        relative_end = means[:, 1:] - positions
        segment = relative_end - relative_start
        fraction = np.clip(-np.sum(relative_start * segment, axis=-1) /
                           np.maximum(np.sum(segment**2, axis=-1), 1e-12), 0., 1.)
        target_clearance = np.linalg.norm(relative_start + fraction[..., None]*segment, axis=-1)
        target_clearance -= obs.robot_radius + obs.entities[target, 4]
        target_clearance[posterior_existence < .15] = math.inf
        # Before the event neither geometry nor risk can use future information.
        human_clearance[:, target, k:] = target_clearance[:, k:]

        hazard = super()._belief_collision_hazard(controls, obs, positions)
        one = replace(obs, entities=obs.entities[target:target+1],
            human_segment_end=obs.human_segment_end[target:target+1],
            human_position_covariance=obs.human_position_covariance[target:target+1],
            human_existence=obs.human_existence[target:target+1])
        old_target = super()._belief_collision_hazard(controls, one, positions)
        variance = np.maximum(.5*np.trace(covariances[:, 1:], axis1=-2, axis2=-1), 0.)
        distance = np.linalg.norm(positions - means[:, 1:], axis=-1)
        radius = obs.robot_radius + obs.entities[target, 4] + cfg.human_margin
        # Evaluate only nonsingular, near-field distributions. The ncx2 backend
        # can abort on enormous noncentrality for a collapsed conditional cloud.
        deterministic = variance < 1e-9
        conditional = np.where(deterministic, distance <= radius, ndtr(-8.)).astype(float)
        near = (~deterministic) & ((distance-radius) < 8.*np.sqrt(variance))
        conditional[near] = ncx2.cdf(radius**2 / variance[near], 2.,
                                    distance[near]**2 / variance[near])
        new_target = -np.log1p(-np.clip(conditional*posterior_existence[:, None], 0., 1.-1e-12))
        hazard[:, k:] = np.maximum(0., hazard[:, k:] - old_target[:, k:] + new_target[:, k:])
        costs = self._cost(controls, obs, positions, human_clearance, None, hazard)
        full, first = self._combined_clearance(controls, obs, positions, human_clearance, None, hazard)
        _, physical = self._physical_clearance(controls, obs, positions, human_clearance)
        costs = np.sum(costs.reshape(count, 2)*probabilities, axis=1)
        full = np.where(probabilities > 0., full.reshape(count, 2), math.inf).min(axis=1)
        first = first.reshape(count, 2)[:, 0]
        physical = physical.reshape(count, 2)[:, 0]
        return costs, full, first, physical, probabilities

    def plan(self, obs, seed):
        start = time.perf_counter()
        self.last_tree = None
        target = self._target(obs)
        if target is None or self.cfg.horizon <= self.event_step:
            return super().plan(obs, seed)
        cfg = self.cfg
        if cfg.population % 2 or cfg.population < 64:
            raise ValueError('Two-branch search requires an even budget >= 64')
        if obs.human_position_covariance is None or obs.occupancy_probability is not None:
            raise ValueError('Pilot supports the existing Gaussian track posterior only')
        cloud = predictive_cloud(obs.entities[target, :4], obs.state_covariances[target],
            cfg.horizon, cfg.dt, cfg.acceleration_std, seed + 87017)
        rng = np.random.default_rng(seed)
        population = cfg.population // 2
        mean = self._initial_mean(obs)
        seeds = self._seed_trajectories(obs)
        routes = self._route_seeds(obs)
        means = np.repeat(np.concatenate((mean[None], routes))[:, None], 2, axis=1)
        deviations = np.full_like(means, cfg.init_std)
        remainder = population-population//2
        counts = [population//2]+[remainder//3]*3
        for i in range(remainder%3): counts[i+1] += 1
        bounds = np.cumsum([0]+counts)
        mode_ids = np.repeat(np.arange(4), counts)
        best_key = (3, math.inf, math.inf)
        best_tree = None
        best_probabilities = None
        k = self.event_step
        for _ in range(cfg.iterations):
            noise = rng.standard_normal((population, 2, cfg.horizon, 2))
            noise[:, :, 1:] = .68*noise[:, :, :-1]+.32*noise[:, :, 1:]
            samples = means[mode_ids]+deviations[mode_ids]*noise
            samples[:len(seeds)] = seeds[:, None]
            samples[len(seeds)] = mean
            for route in range(1, 4):
                samples[bounds[route]] = means[route]
                samples[bounds[route]+1] = routes[route-1]
            samples[:, 1, :k] = samples[:, 0, :k]
            trees = self._project_controls(samples.reshape(-1, cfg.horizon, 2),
                obs.robot_velocity).reshape(population, 2, cfg.horizon, 2)
            if not np.array_equal(trees[:, 0, :k], trees[:, 1, :k]):
                raise RuntimeError('Non-anticipativity violation')
            costs, full, first, physical, probabilities = self._branch_metrics(trees, obs, target, cloud)
            classes = np.where(full >= 0., 0, np.where(first >= 0., 1, 2))
            safety = np.where(classes == 1, full, np.where(classes == 2, physical, 0.))
            order = np.lexsort((costs, -safety, classes))
            chosen = int(order[0])
            key = (int(classes[chosen]), -float(safety[chosen]), float(costs[chosen]))
            if key < best_key:
                best_key = key
                best_tree = trees[chosen].copy()
                best_probabilities = probabilities[chosen].copy()
            for route, count in enumerate(counts):
                left, right = bounds[route:route+2]
                indices = self._elite_indices(costs[left:right], full[left:right], first[left:right],
                    max(4, round(count*cfg.elite_fraction)))+left
                elite = trees[indices]
                means[route] = .22*means[route]+.78*elite.mean(axis=0)
                deviations[route] = np.maximum(cfg.min_std, elite.std(axis=0))
        self.last_tree = best_tree
        self.last_branch_probabilities = best_probabilities
        self.last_controls = best_tree[int(np.argmax(best_probabilities))].copy()
        self._previous_mean = self.last_controls.copy()
        self.last_diagnostics = {'feasibility_class':float(best_key[0]), 'cost':best_key[2],
            'target_index':float(target), 'event_step':float(k),
            'trajectory_evaluations':float(population*2*cfg.iterations),
            'redetection_probability':float(best_probabilities[0])}
        self.branch_steps = getattr(self, 'branch_steps', 0) + 1
        return best_tree[0, 0].copy(), (time.perf_counter()-start)*1000.


class BlindContingencyCEMMPC(ReobservationCEMMPC):
    """Same tree/search budget, but no information from the future observation."""

    def _condition(self, cloud, likelihood, existence):
        return condition_cloud(cloud, np.zeros_like(likelihood), existence)
