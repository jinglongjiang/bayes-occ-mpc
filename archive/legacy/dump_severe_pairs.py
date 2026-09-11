"""Per-frame dump for severe-noise pairs where Bayes times out and static_cov succeeds.

Read-only instrumentation: the planner subclass records the arrays the optimizer
already computes and changes no decision. Verified by replaying the frozen event.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from continuous_mpc_gate import ContinuousCEMMPC, run_episode, DEFAULT_CROWDNAV
from evaluate_matched_safety import MatchedAdapter, config

SCENE = ("dense_square", 20, "square_crossing", None, 10.0)
NOISE = (0.1, 0.2, 0.8)          # severe: position std, velocity std, detection prob
POINT = 2


class Instrumented(ContinuousCEMMPC):
    """Capture the last CEM iteration's population arrays and the selected row."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture = None

    def _combined_clearance(self, controls, obs, positions, human_clearance,
                            occupancy_probability, belief_hazard):
        full, first = super()._combined_clearance(
            controls, obs, positions, human_clearance,
            occupancy_probability, belief_hazard)
        limits = self._belief_hazard_limits()
        cap = {"full_feasible_frac": float(np.mean(full >= 0.0)),
               "first_feasible_frac": float(np.mean(first >= 0.0)),
               "population": int(full.shape[0])}
        if belief_hazard is not None:
            violate = belief_hazard > limits[None, :]
            any_violate = violate.any(axis=1)
            # First horizon step at which each violating candidate crosses its limit.
            first_h = np.where(any_violate, violate.argmax(axis=1), -1)
            hist = np.bincount(first_h[first_h >= 0], minlength=self.cfg.horizon)
            cap.update(
                belief_violate_frac=float(np.mean(any_violate)),
                first_violating_h_hist=hist.astype(int).tolist(),
                pop_max_risk_ratio_med=float(np.median(
                    (belief_hazard / np.maximum(limits[None, :], 1e-12)).max(axis=1))),
            )
        self.capture = cap
        return full, first

    def plan(self, obs, seed):
        action, ms = super().plan(obs, seed)
        cap = dict(self.capture or {})
        controls = self.last_controls[None, :, :]
        positions = obs.robot_xy[None, None, :] + np.cumsum(controls * self.cfg.dt, axis=1)
        hazard = self._belief_collision_hazard(controls, obs, positions)
        limits = self._belief_hazard_limits()
        if hazard is not None:
            h = hazard[0]
            ratio = h / np.maximum(limits, 1e-12)
            cap.update(
                sel_hazard=[round(float(x), 6) for x in h],
                risk_limit=[round(float(x), 6) for x in limits],
                sel_risk_ratio=[round(float(x), 4) for x in ratio],
                sel_sum_hazard=float(h.sum()),
                sel_max_hazard=float(h.max()),
                sel_max_risk_ratio=float(ratio.max()),
                sel_first_violating_h=int(np.argmax(ratio > 1.0)) if (ratio > 1.0).any() else -1,
                soft_belief_risk_cost=float(self.cfg.probability_weight * h.sum()),
            )
        # Cost decomposition for the executed trajectory only.
        active, reached, center = self._active_until_goal(positions, obs)
        goal_dist = np.maximum(center - obs.robot_radius, 0.0)
        terminal = 0.0 if reached.any() else float(goal_dist[0, -1])
        initial = max(float(np.linalg.norm(obs.robot_xy - obs.goal_xy)) - obs.robot_radius, 0.0)
        shortfall = max(self.cfg.min_progress - (initial - terminal), 0.0)
        cap.update(
            goal_cost=float(self.cfg.goal_stage_weight * (goal_dist * active).sum()
                            + self.cfg.goal_terminal_weight * terminal),
            stagnation_cost=float(self.cfg.stagnation_weight * shortfall ** 2),
        )
        cap.update(self.last_diagnostics)
        cap["plan_ms"] = round(ms, 2)
        self.capture = cap
        return action, ms


def dump(case_id, method, calibration, frames):
    def observer(env, observation, adapter, step):
        p = env.robot
        cap = dict(planner_ref[0].capture or {})
        cap.update(
            step=step,
            time=float(env.global_time),
            speed=float(np.hypot(p.vx, p.vy)),
            goal_distance=float(np.hypot(p.gx - p.px, p.gy - p.py)),
        )
        missed = []
        if observation.human_existence is not None and observation.human_position_covariance is not None:
            cov = observation.human_position_covariance
            for i in range(cov.shape[0]):
                var = 0.5 * float(np.trace(cov[i, 0]))
                missed.append({"i": i,
                               "existence": round(float(observation.human_existence[i]), 4),
                               "var_h1": round(var, 5),
                               "var_h16": round(0.5 * float(np.trace(cov[i, -1])), 5)})
        cap["tracks"] = missed
        frames.append(cap)

    planner_ref = [None]

    class Bound(Instrumented):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            planner_ref[0] = self

    result = run_episode(
        DEFAULT_CROWDNAV, "bayes", SCENE[1], SCENE[2], case_id, config(POINT),
        SCENE[3], SCENE[4], 0, *NOISE, time_limit=25,
        planner_type=Bound,
        adapter_factory=partial(MatchedAdapter, method=method,
                                calibration=calibration, point=POINT),
        step_observer=observer,
    )
    return asdict(result)


def main():
    cases = [int(x) for x in sys.argv[1].split(",")]
    calib = json.load(open(sys.argv[2]))
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for case_id in cases:
        for method in ("bayes", "static_cov"):
            frames = []
            res = dump(case_id, method, calib, frames)
            path = out / f"case{case_id}_{method}.json"
            path.write_text(json.dumps(
                {"case": case_id, "method": method, "noise": list(NOISE),
                 "scene": list(SCENE), "point": POINT,
                 "result": {k: v for k, v in res.items()
                            if k in ("event", "success", "collision", "timeout",
                                     "nav_time", "path_length", "min_clearance",
                                     "actual_min_clearance")},
                 "frames": frames}, indent=1, allow_nan=False))
            print(f"  case {case_id} {method:11s} {res['event']:10s} "
                  f"t={res['nav_time']:5.2f} steps={len(frames)}", flush=True)


if __name__ == "__main__":
    main()
