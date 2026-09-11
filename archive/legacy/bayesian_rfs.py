"""Recursive finite-set belief for pedestrians hidden by geometric occlusion.

The filter consumes only visible, identity-associated detections and the sensor
grid.  Detected pedestrians are Bernoulli-Gaussian tracks.  Pedestrians that
have never been detected are represented by a discounted Gamma-Poisson point
process over the unknown part of the field of view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve
from scipy.stats import chi2


@dataclass(frozen=True)
class RFSConfig:
    dt: float = 0.25
    survival_probability: float = 0.995
    detection_probability: float = 0.98
    acceleration_std: float = 0.55
    # CrowdSim exposes exact coordinate state in this protocol.  Process
    # uncertainty belongs in Q; inventing centimetre-scale measurement noise
    # makes visible tracks lag abrupt ORCA manoeuvres.
    position_measurement_std: float = 0.0
    velocity_measurement_std: float = 0.0
    minimum_existence: float = 0.08
    report_existence: float = 0.15
    maximum_missed_steps: int = 40
    chance_limit: float = 0.10
    fixed_uncertainty_radius: float = 0.35
    density_shape_prior: float = 2.0
    density_rate_prior: float = 8.0
    density_forgetting: float = 0.90
    fixed_density: float = 0.25
    pedestrian_radius_prior: float = 0.30
    use_poisson_birth: bool = False


@dataclass
class BernoulliTrack:
    identifier: int
    mean: np.ndarray
    covariance: np.ndarray
    existence: float
    missed_steps: int = 0
    visible: bool = True
    radius: float = 0.30


@dataclass
class RFSOutput:
    entities: np.ndarray
    segment_start: np.ndarray
    segment_end: np.ndarray
    uncertainty_buffer: np.ndarray
    position_covariance: np.ndarray
    existence: np.ndarray
    visible: np.ndarray
    occupancy_probability: np.ndarray
    density_mean: float
    density_std: float
    track_count: int
    hidden_track_count: int


class BayesianRFSBelief:
    """Discounted Gamma-Poisson / Bernoulli-Gaussian multi-person filter."""

    def __init__(self, config: RFSConfig = RFSConfig(), mode: str = "bayes"):
        if mode not in {"bayes", "fixed", "deterministic"}:
            raise ValueError(f"unsupported RFS mode {mode!r}")
        self.cfg = config
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.tracks: Dict[int, BernoulliTrack] = {}
        self.density_shape = self.cfg.density_shape_prior
        self.density_rate = self.cfg.density_rate_prior
        self._last_time: Optional[float] = None

    def _transition(self) -> Tuple[np.ndarray, np.ndarray]:
        dt = self.cfg.dt
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        base = np.array(
            [[dt ** 4 / 4, 0.0, dt ** 3 / 2, 0.0],
             [0.0, dt ** 4 / 4, 0.0, dt ** 3 / 2],
             [dt ** 3 / 2, 0.0, dt ** 2, 0.0],
             [0.0, dt ** 3 / 2, 0.0, dt ** 2]],
            dtype=np.float64,
        )
        process = base * self.cfg.acceleration_std ** 2
        return transition, process

    def _predict_tracks(self) -> None:
        transition, process = self._transition()
        for track in self.tracks.values():
            track.mean = transition @ track.mean
            track.covariance = transition @ track.covariance @ transition.T + process
            track.existence *= self.cfg.survival_probability
            track.missed_steps += 1
            track.visible = False

    def _measurement_update(self, track: BernoulliTrack, measurement: np.ndarray) -> None:
        measurement_covariance = np.diag(
            [self.cfg.position_measurement_std ** 2] * 2
            + [self.cfg.velocity_measurement_std ** 2] * 2
        )
        if not np.any(measurement_covariance):
            track.mean = measurement.copy()
            track.covariance = np.zeros((4, 4), dtype=np.float64)
        else:
            innovation_covariance = track.covariance + measurement_covariance
            gain = np.linalg.solve(
                innovation_covariance.T, track.covariance.T
            ).T
            track.mean = track.mean + gain @ (measurement - track.mean)
            identity = np.eye(4)
            track.covariance = (
                (identity - gain) @ track.covariance @ (identity - gain).T
                + gain @ measurement_covariance @ gain.T
            )
        track.existence = 0.999
        track.missed_steps = 0
        track.visible = True

    def _new_track(self, detection: dict) -> BernoulliTrack:
        mean = np.array(
            [detection["px"], detection["py"], detection["vx"], detection["vy"]],
            dtype=np.float64,
        )
        covariance = np.diag(
            [self.cfg.position_measurement_std ** 2] * 2
            + [self.cfg.velocity_measurement_std ** 2] * 2
        )
        return BernoulliTrack(
            identifier=int(detection["id"]),
            mean=mean,
            covariance=covariance,
            existence=0.999,
            radius=float(detection["radius"]),
        )

    @staticmethod
    def _grid_index(mesh, position: np.ndarray) -> Optional[Tuple[int, int]]:
        mx, my = mesh
        if mx.shape[1] < 2 or my.shape[0] < 2:
            return None
        resolution_x = float(mx[0, 1] - mx[0, 0])
        resolution_y = float(my[1, 0] - my[0, 0])
        col = int(np.floor((position[0] - mx[0, 0]) / resolution_x + 0.5))
        row = int(np.floor((position[1] - my[0, 0]) / resolution_y + 0.5))
        if row < 0 or row >= mx.shape[0] or col < 0 or col >= mx.shape[1]:
            return None
        return row, col

    def update(
        self,
        visible_detections: Iterable[dict],
        sensor_grid: np.ndarray,
        mesh,
        timestamp: float,
    ) -> None:
        detections = list(visible_detections)
        if self._last_time is not None:
            elapsed_steps = max(
                1, int(round((timestamp - self._last_time) / self.cfg.dt))
            )
            for _ in range(elapsed_steps):
                self._predict_tracks()
        self._last_time = float(timestamp)

        detected_ids = set()
        for detection in detections:
            identifier = int(detection["id"])
            detected_ids.add(identifier)
            measurement = np.array(
                [detection["px"], detection["py"],
                 detection["vx"], detection["vy"]], dtype=np.float64
            )
            if identifier not in self.tracks:
                self.tracks[identifier] = self._new_track(detection)
            else:
                self._measurement_update(self.tracks[identifier], measurement)
                self.tracks[identifier].radius = float(detection["radius"])

        for identifier, track in list(self.tracks.items()):
            if identifier in detected_ids:
                continue
            index = self._grid_index(mesh, track.mean[:2])
            predicted_visible = (
                index is not None and sensor_grid[index] != 0.5
            )
            if predicted_visible:
                prior = track.existence
                missed = 1.0 - self.cfg.detection_probability
                track.existence = prior * missed / max(1.0 - prior + prior * missed, 1e-9)
            if (
                track.existence < self.cfg.minimum_existence
                or track.missed_steps > self.cfg.maximum_missed_steps
            ):
                del self.tracks[identifier]

        known_area = float(np.count_nonzero(sensor_grid != 0.5))
        mx, my = mesh
        resolution = float(mx[0, 1] - mx[0, 0])
        known_area *= resolution ** 2
        rho = self.cfg.density_forgetting
        self.density_shape = (
            rho * self.density_shape
            + (1.0 - rho) * self.cfg.density_shape_prior
            + len(detections)
        )
        self.density_rate = (
            rho * self.density_rate
            + (1.0 - rho) * self.cfg.density_rate_prior
            + known_area
        )

    def _prediction_for_track(
        self, track: BernoulliTrack, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        transition, process = self._transition()
        mean = track.mean.copy()
        covariance = track.covariance.copy()
        starts = np.empty((horizon, 2), dtype=np.float64)
        ends = np.empty_like(starts)
        buffers = np.empty(horizon, dtype=np.float64)
        covariances = np.empty((horizon, 2, 2), dtype=np.float64)
        alpha = self.cfg.chance_limit
        for step in range(horizon):
            starts[step] = mean[:2]
            next_mean = transition @ mean
            next_covariance = transition @ covariance @ transition.T + process
            ends[step] = next_mean[:2]
            if self.mode == "fixed":
                fixed_std = self.cfg.fixed_uncertainty_radius / np.sqrt(
                    chi2.ppf(1.0 - alpha, df=2)
                )
                covariances[step] = np.eye(2) * fixed_std ** 2
            elif self.mode == "deterministic":
                covariances[step] = np.zeros((2, 2), dtype=np.float64)
            else:
                covariances[step] = next_covariance[:2, :2]
            if self.mode == "deterministic":
                buffers[step] = 0.0
            elif self.mode == "fixed":
                buffers[step] = self.cfg.fixed_uncertainty_radius
            else:
                conditional_mass = 1.0 - min(0.999, alpha / max(track.existence, alpha))
                quantile = float(chi2.ppf(max(conditional_mass, 0.5), df=2))
                radial_std = float(
                    np.sqrt(max(np.linalg.eigvalsh(next_covariance[:2, :2]).max(), 0.0))
                )
                buffers[step] = np.sqrt(quantile) * radial_std
            mean, covariance = next_mean, next_covariance
        return starts, ends, buffers, covariances

    def _poisson_probability(
        self,
        sensor_grid: np.ndarray,
        mesh,
        robot_radius: float,
        safety_margin: float,
    ) -> np.ndarray:
        unknown = sensor_grid == 0.5
        mx, my = mesh
        resolution = float(mx[0, 1] - mx[0, 0])
        density = (
            self.density_shape / self.density_rate
            if self.mode == "bayes" else self.cfg.fixed_density
        )
        intensity = unknown.astype(np.float64) * density * resolution ** 2

        for track in self.tracks.values():
            if track.visible or track.existence < self.cfg.report_existence:
                continue
            distance2 = (mx - track.mean[0]) ** 2 + (my - track.mean[1]) ** 2
            suppression_radius = track.radius + 2.0 * np.sqrt(
                max(np.linalg.eigvalsh(track.covariance[:2, :2]).max(), 0.0)
            )
            intensity[distance2 <= suppression_radius ** 2] = 0.0

        collision_radius = (
            robot_radius + self.cfg.pedestrian_radius_prior + safety_margin
        )
        cells = int(np.ceil(collision_radius / resolution))
        yy, xx = np.mgrid[-cells:cells + 1, -cells:cells + 1]
        kernel = ((xx * resolution) ** 2 + (yy * resolution) ** 2
                  <= collision_radius ** 2).astype(np.float64)
        expected_count = convolve(intensity, kernel, mode="constant", cval=0.0)
        return 1.0 - np.exp(-expected_count)

    def output(
        self,
        sensor_grid: np.ndarray,
        mesh,
        horizon: int,
        robot_radius: float,
        safety_margin: float,
    ) -> RFSOutput:
        selected = [
            track for track in self.tracks.values()
            if track.visible or track.existence >= self.cfg.report_existence
        ]
        entities: List[List[float]] = []
        starts, ends, buffers, covariances, existences = [], [], [], [], []
        for track in selected:
            entities.append(
                [track.mean[0], track.mean[1], track.mean[2], track.mean[3], track.radius]
            )
            start, end, buffer, covariance = self._prediction_for_track(track, horizon)
            starts.append(start)
            ends.append(end)
            buffers.append(buffer)
            covariances.append(covariance)
            existences.append(track.existence)

        entity_array = np.asarray(entities, dtype=np.float64).reshape(-1, 5)
        start_array = np.asarray(starts, dtype=np.float64).reshape(-1, horizon, 2)
        end_array = np.asarray(ends, dtype=np.float64).reshape(-1, horizon, 2)
        buffer_array = np.asarray(buffers, dtype=np.float64).reshape(-1, horizon)
        covariance_array = np.asarray(covariances, dtype=np.float64).reshape(
            -1, horizon, 2, 2
        )
        existence_array = np.asarray(existences, dtype=np.float64)
        probability = (
            self._poisson_probability(
                sensor_grid, mesh, robot_radius, safety_margin
            )
            if self.cfg.use_poisson_birth
            else np.zeros(sensor_grid.shape, dtype=np.float64)
        )
        density_mean = self.density_shape / self.density_rate
        density_std = np.sqrt(self.density_shape) / self.density_rate
        return RFSOutput(
            entities=entity_array,
            segment_start=start_array,
            segment_end=end_array,
            uncertainty_buffer=buffer_array,
            position_covariance=covariance_array,
            existence=existence_array,
            visible=np.asarray([t.visible for t in selected], dtype=bool),
            occupancy_probability=probability,
            density_mean=float(density_mean),
            density_std=float(density_std),
            track_count=len(selected),
            hidden_track_count=sum(not track.visible for track in selected),
        )
