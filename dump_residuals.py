"""Dump (prediction error, predictive covariance) pairs for offline shape fitting.

Truth is read only after the control is selected and is never returned to the
adapter or planner, so this changes no decision.  Rows carry the Bayes filter's
own per-track covariance, which the existing calibration records do not.
"""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from continuous_mpc_gate import run_episode, DEFAULT_CROWDNAV
from evaluate_matched_safety import MatchedAdapter, config

SCENE = ("dense_square", 20, "square_crossing", None, 10.0)
POINT = 2
NOISE = {"clean": (0.0, 0.0, 1.0), "severe": (0.1, 0.2, 0.8)}


class Recorder:
    """Pair each forecast with the truth that arrives h steps later."""

    def __init__(self, case):
        self.case = case
        self.pending = {}
        self.rows = []

    def __call__(self, env, observation, adapter, step):
        for (ident, h, visible, mean, cov, existence,
             state_var, process_var) in self.pending.pop(step, []):
            truth = np.asarray(env.humans[ident].get_position())
            error = truth - mean
            # Whitened squared radius under the filter's own covariance.
            try:
                d2 = float(error @ np.linalg.solve(cov, error))
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(d2):
                continue
            self.rows.append([self.case, ident, h, visible, existence,
                              error[0], error[1], cov[0, 0], cov[0, 1], cov[1, 1], d2,
                              state_var, process_var])
        if observation.human_position_covariance is None:
            return
        transition, process = adapter.rfs._transition()
        for i, ident in enumerate(adapter.reported_ids):
            track = adapter.rfs.tracks[ident]
            vis = int(track.visible)
            ex = float(observation.human_existence[i])
            # P_h = F^h P_0 (F^h)^T  +  sum_{k<h} F^k Q (F^k)^T.  The filter's
            # recursion is linear, so the two terms separate exactly: the first
            # is irreducible state uncertainty, the second is future process
            # noise.  Only the latter should carry the heavy tail.
            state_cov = track.covariance.copy()
            process_cov = np.zeros_like(state_cov)
            state_vars, process_vars = [], []
            for _ in range(adapter.horizon):
                state_cov = transition @ state_cov @ transition.T
                process_cov = transition @ process_cov @ transition.T + process
                state_vars.append(float(state_cov[0, 0]))
                process_vars.append(float(process_cov[0, 0]))
            for h in range(adapter.horizon):
                self.pending.setdefault(step + h + 1, []).append((
                    ident, h + 1, vis,
                    observation.human_segment_end[i, h].copy(),
                    observation.human_position_covariance[i, h].copy(), ex,
                    state_vars[h], process_vars[h]))


def main():
    first, count, label = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    calib = json.load(open(sys.argv[4]))
    out = Path(sys.argv[5])
    out.parent.mkdir(parents=True, exist_ok=True)
    noise = NOISE[label]
    rows = []
    for case in range(first, first + count):
        rec = Recorder(case)
        run_episode(DEFAULT_CROWDNAV, "bayes", SCENE[1], SCENE[2], case,
                    config(POINT), SCENE[3], SCENE[4], 0, *noise, time_limit=25,
                    adapter_factory=partial(MatchedAdapter, method="bayes",
                                            calibration=calib, point=POINT),
                    step_observer=rec)
        rows.extend(rec.rows)
        print(f"  case {case}: {len(rec.rows):6d} forecasts (total {len(rows)})", flush=True)
    array = np.asarray(rows, dtype=np.float64)
    np.savez_compressed(out, rows=array, columns=np.array(
        ["case", "track", "h", "visible", "existence", "err_x", "err_y",
         "P00", "P01", "P11", "d2", "state_var", "process_var"]), noise=np.array(noise),
        cases=np.array([first, first + count - 1]))
    print(f"saved {array.shape} -> {out}")


if __name__ == "__main__":
    main()
