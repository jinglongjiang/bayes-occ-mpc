"""Risk-limit ramp sweep: gamma in {2, 1} x {bayes, static_cov}, fresh dev cases.

Only the exponent of the per-step chance-limit ramp changes:
    eps(h) = near + (far - near) * phase ** gamma
gamma=2.0 reproduces the frozen configuration bit-for-bit.

Every candidate's hazard is additive across pedestrians, so the visible and
missed contributions are an exact split of the total, not an attribution.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from continuous_mpc_gate import ContinuousCEMMPC, run_episode, DEFAULT_CROWDNAV
from evaluate_matched_safety import MatchedAdapter, config

SCENE = ("dense_square", 20, "square_crossing", None, 10.0)
NOISE = (0.1, 0.2, 0.8)          # severe
POINT = 2
CASE_BASE = 7000                 # never used in any prior cohort


class Logged(ContinuousCEMMPC):
    """Read-only: records the arrays the optimizer already computes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame = None

    def _hazard_split(self, controls, obs, positions):
        """Exact visible/missed split of the additive per-human hazard."""
        if obs.human_visible is None or obs.human_existence is None:
            return None
        vis = np.asarray(obs.human_visible, dtype=bool)
        if vis.size != obs.entities.shape[0]:
            return None
        parts = {}
        for name, mask in (("visible", vis), ("missed", ~vis)):
            if not mask.any():
                parts[name] = np.zeros(self.cfg.horizon)
                continue
            sub = replace(
                obs, entities=obs.entities[mask],
                human_segment_end=obs.human_segment_end[mask],
                human_position_covariance=obs.human_position_covariance[mask],
                human_existence=obs.human_existence[mask],
            )
            parts[name] = self._belief_collision_hazard(controls, sub, positions)[0]
        return parts

    def _combined_clearance(self, controls, obs, positions, human_clearance,
                            occupancy_probability, belief_hazard):
        full, first = super()._combined_clearance(
            controls, obs, positions, human_clearance,
            occupancy_probability, belief_hazard)
        cap = {"full_feasible_frac": float(np.mean(full >= 0.0)),
               "first_feasible_frac": float(np.mean(first >= 0.0))}
        if belief_hazard is not None:
            limits = self._belief_hazard_limits()
            violate = belief_hazard > limits[None, :]
            any_v = violate.any(axis=1)
            first_h = np.where(any_v, violate.argmax(axis=1), -1)
            cap.update(
                belief_violate_frac=float(np.mean(any_v)),
                first_violating_h_hist=np.bincount(
                    first_h[first_h >= 0], minlength=self.cfg.horizon).astype(int).tolist(),
            )
        self.frame = cap
        return full, first

    def plan(self, obs, seed):
        action, ms = super().plan(obs, seed)
        cap = dict(self.frame or {})
        controls = self.last_controls[None, :, :]
        positions = obs.robot_xy[None, None, :] + np.cumsum(controls * self.cfg.dt, axis=1)
        hazard = self._belief_collision_hazard(controls, obs, positions)
        if hazard is not None:
            h = hazard[0]
            limits = self._belief_hazard_limits()
            ratio = h / np.maximum(limits, 1e-12)
            over = np.flatnonzero(ratio > 1.0)
            first_h = int(over[0]) if over.size else -1
            cap.update(
                hazard_total=[round(float(x), 6) for x in h],
                risk_limit=[round(float(x), 6) for x in limits],
                sel_max_risk_ratio=float(ratio.max()),
                sel_first_violating_h=first_h,
            )
            split = self._hazard_split(controls, obs, positions)
            if split is not None:
                cap["hazard_visible"] = [round(float(x), 6) for x in split["visible"]]
                cap["hazard_missed"] = [round(float(x), 6) for x in split["missed"]]
                # Contribution shares at the first horizon step that crosses.
                k = first_h if first_h >= 0 else int(np.argmax(ratio))
                total = split["visible"][k] + split["missed"][k]
                if total > 1e-12:
                    cap["share_visible_at_violation"] = float(split["visible"][k] / total)
                    cap["share_missed_at_violation"] = float(split["missed"][k] / total)
        cap.update(self.last_diagnostics)
        cap["plan_ms"] = round(ms, 2)
        self.frame = cap
        return action, ms


def one(case_id, method, gamma, calibration):
    cfg = replace(config(POINT), risk_gamma=gamma)
    frames = []
    ref = [None]

    class Bound(Logged):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            ref[0] = self

    def observer(env, observation, adapter, step):
        p = env.robot
        f = dict(ref[0].frame or {})
        f.update(step=step, speed=float(np.hypot(p.vx, p.vy)),
                 goal_distance=float(np.hypot(p.gx - p.px, p.gy - p.py)))
        frames.append(f)

    res = run_episode(
        DEFAULT_CROWDNAV, "bayes", SCENE[1], SCENE[2], case_id, cfg,
        SCENE[3], SCENE[4], 0, *NOISE, time_limit=25,
        planner_type=Bound,
        adapter_factory=partial(MatchedAdapter, method=method,
                                calibration=calibration, point=POINT),
        step_observer=observer)
    return asdict(res), frames


def main():
    count = int(sys.argv[1])
    calib = json.load(open(sys.argv[2]))
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    for gamma in (2.0, 1.0):
        for method in ("bayes", "static_cov"):
            rows, agg = [], []
            for i in range(count):
                res, frames = one(CASE_BASE + i, method, gamma, calib)
                rows.append({k: res[k] for k in
                             ("case_id", "event", "success", "collision", "timeout",
                              "nav_time", "scored_time", "path_length",
                              "min_clearance", "actual_min_clearance",
                              "collision_union")})
                sp = [f["speed"] for f in frames]
                ff = [f.get("full_feasible_frac", float("nan")) for f in frames]
                bv = [f.get("belief_violate_frac") for f in frames
                      if f.get("belief_violate_frac") is not None]
                sv = [f.get("share_visible_at_violation") for f in frames
                      if f.get("share_visible_at_violation") is not None]
                hh = np.sum([f.get("first_violating_h_hist", [0] * 16)
                             for f in frames], axis=0)
                agg.append(dict(case=CASE_BASE + i, speed_med=float(np.median(sp)),
                                full_feasible_med=float(np.median(ff)),
                                violate_med=float(np.median(bv)) if bv else None,
                                share_visible_med=float(np.median(sv)) if sv else None,
                                first_violating_h_hist=hh.astype(int).tolist(),
                                steps=len(frames)))
            name = f"g{gamma:g}_{method}"
            (out / f"{name}.json").write_text(json.dumps(
                {"gamma": gamma, "method": method, "point": POINT,
                 "noise": list(NOISE), "scene": list(SCENE),
                 "cases": [CASE_BASE, CASE_BASE + count - 1],
                 "episodes": rows, "per_case": agg}, indent=1, allow_nan=False))
            n = len(rows)
            print(f"{name:20s} SR={sum(r['success'] for r in rows)/n*100:5.2f}% "
                  f"CR={sum(r['collision'] for r in rows)/n*100:5.2f}% "
                  f"TR={sum(r['timeout'] for r in rows)/n*100:5.2f}% "
                  f"T={np.mean([r['scored_time'] for r in rows]):6.3f}s "
                  f"speed={np.median([a['speed_med'] for a in agg]):.3f}", flush=True)


if __name__ == "__main__":
    main()
