"""Data contracts retained from the audited holonomic protocol."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

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
    risk_gamma: float = 2.0
    near_chance_limit: float = 0.10
    occupancy_chance_limit: float = 0.15
    fixed_uncertainty_radius: float = 0.35
    stagnation_weight: float = 250.0
    min_progress: float = 0.15
    init_std: float = 0.65
    min_std: float = 0.08
    acceleration_std: float = 0.55
    # Ablation: pin the existence probability used by the risk term while
    # leaving the track set, the means and the covariances untouched.  This
    # separates "is this person still there" from "where exactly are they".
    existence_override: Optional[float] = None
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
    human_visible: Optional[np.ndarray]
    unknown: Optional[UnknownField]
    occupancy_probability: Optional[ProbabilityField]
    provenance: str
    # Only the unicycle planner reads this; the holonomic path never touches it.
    robot_heading: Optional[float] = None
