#!/usr/bin/env python3
"""Read-only calibration audit for the Bayesian pedestrian belief."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from array import array
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from continuous_mpc_gate import (
    DEFAULT_CROWDNAV,
    MPCConfig,
    ContinuousCEMMPC,
    ObservationAdapter,
    _load_modules,
    build_env,
)


QUANTILES = (0.50, 0.68, 0.90, 0.95)


METRICS = ("error", "nll", "d2", "sharpness")


def new_bucket() -> dict:
    return {name: array("d") for name in METRICS}


def summarize(buckets: list) -> dict:
    if not buckets or not any(len(bucket["error"]) for bucket in buckets):
        return {"n": 0}
    values = {
        name: np.concatenate([
            np.frombuffer(bucket[name], dtype=np.float64)
            for bucket in buckets if len(bucket[name])
        ])
        for name in METRICS
    }
    error = values["error"]
    nll = values["nll"]
    d2 = values["d2"]
    sharpness = values["sharpness"]
    output = {
        "n": len(error),
        "mean_error": float(error.mean()),
        "median_error": float(np.median(error)),
        "mean_nll": float(nll.mean()),
        "median_mahalanobis2": float(np.median(d2)),
        "mean_sharpness_sqrt_det": float(sharpness.mean()),
    }
    for quantile in (0.50, 0.68, 0.85, 0.90, 0.95):
        output[f"radial_error_q{int(100 * quantile)}"] = float(
            np.quantile(error, quantile)
        )
    for quantile in QUANTILES:
        threshold = float(chi2.ppf(quantile, df=2))
        output[f"coverage_{int(100 * quantile)}"] = float(np.mean(d2 <= threshold))
    return output


def score_forecast(entry: dict, truth: np.ndarray) -> dict:
    covariance = np.asarray(entry["covariance"], dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 1e-9)) @ eigenvectors.T
    error_vector = truth - entry["mean"]
    inverse = np.linalg.inv(covariance)
    d2 = float(error_vector @ inverse @ error_vector)
    _, logdet = np.linalg.slogdet(covariance)
    nll = 0.5 * (d2 + logdet + 2.0 * math.log(2.0 * math.pi))
    nll -= math.log(max(float(entry["existence"]), 1e-12))
    return {
        "horizon": int(entry["horizon"]),
        "origin_visibility": entry["origin_visibility"],
        "error": float(np.linalg.norm(error_vector)),
        "d2": d2,
        "nll": nll,
        "sharpness": float(np.sqrt(max(np.linalg.det(covariance), 0.0))),
    }


class PairedCalibrationAudit:
    """Score several estimators on the same observations, after action selection."""

    METHODS = ('bayes', 'static_cov', 'ewma', 'conformal')

    def __init__(self, config, calibration, noise, case_id, point=2):
        from evaluate_matched_safety import MatchedAdapter
        self.cfg = config
        self.calibration = calibration
        args = ('bayes', config.horizon, config.human_margin, config.dt,
                config.chance_limit, config.fixed_uncertainty_radius,
                config.acceleration_std, *noise, case_id * 1000003 + 7919,
                config.conformal_visible_radii, config.conformal_hidden_radii)
        self.shadows = {name: MatchedAdapter(*args, method=name,
                       calibration=calibration, point=point) for name in self.METHODS[1:]}
        self.pending = defaultdict(list)
        self.buckets = {}
        self.previous_step = -1
        self.scored = 0
        self.forecasts = 0
        self.mean_parity_checks = 0

    def __call__(self, env, observation, adapter, step):
        if step <= self.previous_step:
            raise RuntimeError('Calibration observer called twice or out of order')
        self.previous_step = step
        outputs = {'bayes': observation}
        for name, shadow in self.shadows.items():
            value = shadow.read(env)
            if (shadow.reported_ids != adapter.reported_ids
                    or not np.array_equal(value.human_segment_end, observation.human_segment_end)):
                raise RuntimeError('Calibration methods did not receive identical track means')
            outputs[name] = value
            self.mean_parity_checks += 1

        # Only the evaluator reads these labels. Nothing is returned to the planner.
        for entry in self.pending.pop(step, []):
            ids, horizon, visible, age, means, covariances, radii = entry
            truth = np.asarray([env.humans[i].get_position() for i in ids])
            error = truth - means
            error2 = np.square(error).sum(axis=1)
            for name in self.METHODS:
                if name == 'conformal':
                    hits = error2[:, None] <= np.square(radii)
                    areas = np.pi * np.square(radii)
                    nll = None
                else:
                    covariance = covariances[name]
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    covariance = (eigenvectors * np.maximum(eigenvalues, 1e-9)[:, None, :]) @ np.swapaxes(eigenvectors, -1, -2)
                    d2 = np.einsum('ni,ni->n', error, np.linalg.solve(covariance, error[..., None])[..., 0])
                    logdet = np.linalg.slogdet(covariance)[1]
                    thresholds = chi2.ppf(QUANTILES, 2)
                    hits = d2[:, None] <= thresholds
                    areas = np.pi * np.exp(.5 * logdet)[:, None] * thresholds
                    nll = .5 * (d2 + logdet + 2 * math.log(2 * math.pi))
                for i in range(len(ids)):
                    key = (name, horizon, int(visible[i]), int(age[i]))
                    bucket = self.buckets.setdefault(key, {
                        'n': 0, 'coverage_hits': np.zeros(4, dtype=np.int64),
                        'area_sum': np.zeros(4), 'error_sum': 0., 'conditional_nll_sum': 0.})
                    bucket['n'] += 1
                    bucket['coverage_hits'] += hits[i]
                    bucket['area_sum'] += areas[i]
                    bucket['error_sum'] += float(np.sqrt(error2[i]))
                    if nll is not None:
                        bucket['conditional_nll_sum'] += float(nll[i])
            self.scored += len(ids)

        ids = list(adapter.reported_ids)
        if not ids:
            return
        visible = np.array([int(adapter.rfs.tracks[i].visible) for i in ids])
        age = np.array([0 if adapter.rfs.tracks[i].missed_steps == 0 else
                        (1 if adapter.rfs.tracks[i].missed_steps <= 4 else 2) for i in ids])
        reference_radii = np.asarray(self.calibration['audit_conformal_quantile_radii'])
        for h in range(self.cfg.horizon):
            covariance = {name: outputs[name].human_position_covariance[:, h].copy()
                          for name in self.METHODS[:-1]}
            self.pending[step+h+1].append((ids, h+1, visible, age,
                observation.human_segment_end[:, h].copy(), covariance,
                reference_radii[visible, h].copy()))
            self.forecasts += len(ids)

    def result(self):
        rows = []
        for (method, horizon, visible, age), bucket in sorted(self.buckets.items()):
            row = {'method': method, 'horizon': horizon, 'detected_at_origin': visible,
                   'missed_bin': age, 'n': bucket['n'],
                   'coverage_hits': bucket['coverage_hits'].tolist(),
                   'area_sum': bucket['area_sum'].tolist(),
                   'error_sum': bucket['error_sum'],
                   'conditional_nll_sum': bucket['conditional_nll_sum'] if method != 'conformal' else None}
            rows.append(row)
        return {'strata': rows, 'scored_forecasts': self.scored,
                'terminal_censored_forecasts': self.forecasts-self.scored,
                'mean_parity_checks': self.mean_parity_checks,
                'quantiles': list(QUANTILES),
                'scope': 'Conditional location coverage of reported, identity-associated tracks; not a trajectory safety guarantee or an existence calibration test.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crowdnav-root", type=Path, default=DEFAULT_CROWDNAV)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--case-offset", type=int, default=2100)
    parser.add_argument("--human-count", type=int, default=20)
    parser.add_argument("--scenario", default="circle_crossing")
    parser.add_argument("--population", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--v-max", type=float, default=1.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = MPCConfig(
        population=args.population,
        iterations=args.iterations,
        v_max=args.v_max,
        chance_limit=0.50,
        near_chance_limit=0.15,
        probability_weight=1.0,
        human_margin=0.10,
    )
    _, _, ActionXY, _ = _load_modules(args.crowdnav_root)
    # A dict per forecast costs hundreds of bytes and can exceed 1 GB for a
    # 100-episode audit.  Four compact double arrays per stratum preserve the
    # exact statistics while keeping memory bounded to the numeric payload.
    scores = defaultdict(new_bucket)
    score_count = 0
    recall_counts = defaultdict(int)
    outcomes = defaultdict(int)
    plan_times = []

    for case_id in range(args.case_offset, args.case_offset + args.episodes):
        env, _, _ = build_env(
            args.crowdnav_root, "bayes", args.human_count, args.scenario
        )
        env.reset(options={"test_case": case_id})
        planner = ContinuousCEMMPC(cfg)
        adapter = ObservationAdapter(
            "bayes", cfg.horizon, cfg.human_margin, cfg.dt,
            cfg.chance_limit, cfg.fixed_uncertainty_radius,
            cfg.acceleration_std,
        )
        pending = defaultdict(list)
        seen_ids = set()
        event = "timeout"
        terminated = truncated = False
        step = 0

        while not (terminated or truncated):
            observation = adapter.read(env)
            visible_ids = set(env.occlusion.visible_ids)
            seen_ids.update(visible_ids)

            for entry in pending.pop(step, []):
                human = env.humans[entry["identifier"]]
                truth = np.asarray([human.px, human.py], dtype=np.float64)
                scored = score_forecast(entry, truth)
                bucket = scores[(scored["horizon"], scored["origin_visibility"])]
                for name in METRICS:
                    bucket[name].append(scored[name])
                score_count += 1

            hidden_ids = set(range(len(env.humans))) - visible_ids
            reported_ids = {
                identifier for identifier, track in adapter.rfs.tracks.items()
                if track.visible or track.existence >= adapter.rfs.cfg.report_existence
            }
            robot_xy = np.asarray(env.robot.get_position(), dtype=np.float64)
            for identifier in hidden_ids:
                human = env.humans[identifier]
                clearance = float(np.linalg.norm(
                    robot_xy - np.asarray([human.px, human.py], dtype=np.float64)
                ) - env.robot.radius - human.radius)
                category = "seen_hidden" if identifier in seen_ids else "never_seen_hidden"
                recall_counts[f"{category}_total"] += 1
                recall_counts[f"{category}_reported"] += int(identifier in reported_ids)
                if clearance <= 2.0:
                    recall_counts[f"near_{category}_total"] += 1
                    recall_counts[f"near_{category}_reported"] += int(identifier in reported_ids)

            for identifier, track in adapter.rfs.tracks.items():
                _, means, _, covariances = adapter.rfs._prediction_for_track(
                    track, cfg.horizon
                )
                origin_visibility = "visible" if identifier in visible_ids else "hidden"
                for horizon in range(1, cfg.horizon + 1):
                    pending[step + horizon].append({
                        "identifier": identifier,
                        "horizon": horizon,
                        "mean": means[horizon - 1].copy(),
                        "covariance": covariances[horizon - 1].copy(),
                        "existence": float(track.existence),
                        "origin_visibility": origin_visibility,
                    })

            solver_seed = case_id * 100003 + step * 97 + 1729
            velocity, plan_ms = planner.plan(observation, solver_seed)
            plan_times.append(plan_ms)
            _, _, terminated, truncated, info = env.step(
                ActionXY(float(velocity[0]), float(velocity[1]))
            )
            event = str(info.get("event", "nothing"))
            step += 1
        outcomes[event] += 1
        print(
            f"case={case_id} event={event} scores={score_count}",
            flush=True,
        )

    grouped_horizon = {
        str(horizon): summarize([
            scores[(horizon, visibility)]
            for visibility in ("visible", "hidden")
        ])
        for horizon in range(1, cfg.horizon + 1)
    }
    grouped_visibility = {
        visibility: summarize([
            scores[(horizon, visibility)]
            for horizon in range(1, cfg.horizon + 1)
        ])
        for visibility in ("visible", "hidden")
    }
    risk_limits = ContinuousCEMMPC(cfg)._belief_risk_limits()
    def conformal_radii_for(visibility: str) -> list:
        radii = []
        for horizon, alpha in enumerate(risk_limits, start=1):
            values = np.sort(np.frombuffer(
                scores[(horizon, visibility)]["error"], dtype=np.float64
            ))
            if not len(values):
                radii.append(None)
                continue
            rank = min(
                len(values),
                int(math.ceil((len(values) + 1) * (1.0 - alpha))),
            )
            radii.append(float(values[rank - 1]))
        return radii

    conformal_visible_radii = conformal_radii_for("visible")
    conformal_hidden_radii = conformal_radii_for("hidden")
    recall = {}
    for category in (
        "seen_hidden", "never_seen_hidden",
        "near_seen_hidden", "near_never_seen_hidden",
    ):
        total = recall_counts[f"{category}_total"]
        reported = recall_counts[f"{category}_reported"]
        recall[category] = {
            "reported": reported,
            "total": total,
            "recall": float(reported / total) if total else None,
        }

    source_config = args.crowdnav_root / "crowd_nav/configs/env.config"
    output = {
        "protocol": {
            "evaluation_only_truth": True,
            "truth_read_after_action_observation": True,
            "controller_truth_access": False,
            "case_ids": [args.case_offset, args.case_offset + args.episodes - 1],
            "episodes": args.episodes,
            "scenario": args.scenario,
            "human_count": args.human_count,
            "planner": asdict(cfg),
            "controller_sha256": hashlib.sha256(
                (Path(__file__).parent / "continuous_mpc_gate.py").read_bytes()
            ).hexdigest(),
            "belief_sha256": hashlib.sha256(
                (Path(__file__).parent / "bayesian_rfs.py").read_bytes()
            ).hexdigest(),
            "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        },
        "outcomes": dict(outcomes),
        "forecast_all": summarize(list(scores.values())),
        "forecast_by_horizon": grouped_horizon,
        "forecast_by_origin_visibility": grouped_visibility,
        "split_conformal": {
            "calibration_only": True,
            "finite_sample_order_statistic": True,
            "mondrian_strata": ["visible", "hidden"],
            "miscoverage_by_horizon": [float(value) for value in risk_limits],
            "visible_radial_buffer_by_horizon": conformal_visible_radii,
            "hidden_radial_buffer_by_horizon": conformal_hidden_radii,
        },
        "hidden_track_recall": recall,
        "planning_latency_ms": {
            "n": len(plan_times),
            "p50": float(np.percentile(plan_times, 50)),
            "p95": float(np.percentile(plan_times, 95)),
            "p99": float(np.percentile(plan_times, 99)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
