"""Bayes-only recursive pedestrian belief; no density or buffer side models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BeliefConfig:
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
class BeliefOutput:
    entities: np.ndarray
    segment_start: np.ndarray
    segment_end: np.ndarray
    position_covariance: np.ndarray
    existence: np.ndarray
    visible: np.ndarray
    track_count: int
    hidden_track_count: int


class BayesianBelief:
    """The audited Bernoulli-Gaussian track recursion without unused outputs."""

    def __init__(self, config: BeliefConfig = BeliefConfig()):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self.tracks: Dict[int, BernoulliTrack] = {}
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


    def _prediction_for_track(
        self, track: BernoulliTrack, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        transition, process = self._transition()
        mean = track.mean.copy()
        covariance = track.covariance.copy()
        starts = np.empty((horizon, 2), dtype=np.float64)
        ends = np.empty_like(starts)
        covariances = np.empty((horizon, 2, 2), dtype=np.float64)
        for step in range(horizon):
            starts[step] = mean[:2]
            next_mean = transition @ mean
            next_covariance = transition @ covariance @ transition.T + process
            ends[step] = next_mean[:2]
            covariances[step] = next_covariance[:2, :2]
            mean, covariance = next_mean, next_covariance
        return starts, ends, covariances

    def output(self, horizon: int) -> BeliefOutput:
        selected = [
            track for track in self.tracks.values()
            if track.visible or track.existence >= self.cfg.report_existence
        ]
        entities: List[List[float]] = []
        starts, ends, covariances, existences = [], [], [], []
        for track in selected:
            entities.append(
                [track.mean[0], track.mean[1], track.mean[2], track.mean[3], track.radius]
            )
            start, end, covariance = self._prediction_for_track(track, horizon)
            starts.append(start)
            ends.append(end)
            covariances.append(covariance)
            existences.append(track.existence)

        entity_array = np.asarray(entities, dtype=np.float64).reshape(-1, 5)
        start_array = np.asarray(starts, dtype=np.float64).reshape(-1, horizon, 2)
        end_array = np.asarray(ends, dtype=np.float64).reshape(-1, horizon, 2)
        covariance_array = np.asarray(covariances, dtype=np.float64).reshape(
            -1, horizon, 2, 2
        )
        existence_array = np.asarray(existences, dtype=np.float64)
        return BeliefOutput(
            entities=entity_array,
            segment_start=start_array,
            segment_end=end_array,
            position_covariance=covariance_array,
            existence=existence_array,
            visible=np.asarray([t.visible for t in selected], dtype=bool),
            track_count=len(selected),
            hidden_track_count=sum(not track.visible for track in selected),
        )
