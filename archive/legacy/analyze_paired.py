#!/usr/bin/env python3
"""Analyze arbitrary paired arms without selecting on test outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom_test


def paired_bootstrap(delta: np.ndarray, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 2000):
        stop = min(samples, start + 2000)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_delta_bayes_minus_baseline": float(delta.mean()),
        "ci95": [float(low), float(high)],
    }


def binary_pair(bayes: np.ndarray, baseline: np.ndarray) -> dict:
    wins = int(np.sum((bayes == 1) & (baseline == 0)))
    losses = int(np.sum((bayes == 0) & (baseline == 1)))
    discordant = wins + losses
    p_value = (
        float(binom_test(min(wins, losses), discordant, 0.5, alternative="two-sided"))
        if discordant else 1.0
    )
    return {
        "wins": wins,
        "losses": losses,
        "discordant": discordant,
        "mcnemar_exact_p": p_value,
    }


def proportion_ci(successes: int, total: int) -> list:
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return [low, high]


def parse_input(value: str) -> tuple:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    return label, Path(path)


def holm_adjust(values: dict) -> dict:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for index, (label, p_value) in enumerate(ordered):
        running = max(running, (total - index) * p_value)
        adjusted[label] = min(1.0, float(running))
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--case-offset", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--bootstrap", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=91573)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len({label for label, _ in args.input}) != len(args.input):
        raise RuntimeError("duplicate input label")
    payloads = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in args.input
    }
    if "bayes" not in payloads:
        raise RuntimeError("a bayes input is required")

    script_hashes = {p["protocol"]["script_sha256"] for p in payloads.values()}
    belief_hashes = {p["protocol"].get("belief_sha256") for p in payloads.values()}
    source_hashes = {
        p["protocol"]["source_config_sha256"] for p in payloads.values()
    }
    if (
        len(script_hashes) != 1
        or len(source_hashes) != 1
        or len(belief_hashes) != 1
    ):
        raise RuntimeError("arms did not use identical controller/source code")

    invariant_keys = (
        "dt", "horizon", "population", "iterations", "elite_fraction",
        "v_max", "a_max", "human_margin", "goal_stage_weight",
        "goal_terminal_weight", "smooth_weight", "effort_weight",
        "collision_weight", "discomfort_weight", "probability_weight",
        "chance_limit", "near_chance_limit", "occupancy_chance_limit",
        "stagnation_weight", "min_progress", "init_std", "min_std",
        "acceleration_std",
    )
    planners = {label: p["protocol"]["planner"] for label, p in payloads.items()}
    reference = planners["bayes"]
    for label, planner in planners.items():
        mismatch = [key for key in invariant_keys if planner[key] != reference[key]]
        if mismatch:
            raise RuntimeError(f"planner mismatch for {label}: {mismatch}")

    expected_cases = set(range(args.case_offset, args.case_offset + args.episodes))
    rows = {}
    summaries = {}
    for label, payload in payloads.items():
        keyed = {int(row["case_id"]): row for row in payload["episodes"]}
        if set(keyed) != expected_cases or len(keyed) != args.episodes:
            raise RuntimeError(f"{label} has an incomplete or wrong case range")
        if len(payload["summary"]) != 1:
            raise RuntimeError(f"{label} must contain exactly one setting")
        rows[label] = keyed
        summaries[label] = payload["summary"][0]

    ordered_cases = sorted(expected_cases)
    bayes = rows["bayes"]
    comparisons = {}
    for comparison_index, label in enumerate(payloads):
        if label == "bayes":
            continue
        success_bayes = np.asarray([bayes[i]["success"] for i in ordered_cases])
        success_other = np.asarray([rows[label][i]["success"] for i in ordered_cases])
        safe_bayes = 1 - np.asarray([bayes[i]["collision"] for i in ordered_cases])
        safe_other = 1 - np.asarray([rows[label][i]["collision"] for i in ordered_cases])
        scored_delta = np.asarray([
            bayes[i]["scored_time"] - rows[label][i]["scored_time"]
            for i in ordered_cases
        ], dtype=np.float64)
        path_delta = np.asarray([
            bayes[i]["path_length"] - rows[label][i]["path_length"]
            for i in ordered_cases
        ], dtype=np.float64)
        both_success = [
            i for i in ordered_cases
            if bayes[i]["success"] and rows[label][i]["success"]
        ]
        arrival_delta = np.asarray([
            bayes[i]["nav_time"] - rows[label][i]["nav_time"]
            for i in both_success
        ], dtype=np.float64)
        bootstrap_seed = args.seed + 10 * comparison_index
        comparisons[label] = {
            "success": binary_pair(success_bayes, success_other),
            "collision_free": binary_pair(safe_bayes, safe_other),
            "failure_penalized_time": paired_bootstrap(
                scored_delta, args.bootstrap, bootstrap_seed
            ),
            "path_length": paired_bootstrap(
                path_delta, args.bootstrap, bootstrap_seed + 1
            ),
            "both_success_episodes": len(both_success),
            "arrival_time_both_success": (
                paired_bootstrap(arrival_delta, args.bootstrap, bootstrap_seed + 2)
                if len(arrival_delta) else None
            ),
        }

    adjusted = {
        family: holm_adjust({
            label: comparison[family]["mcnemar_exact_p"]
            for label, comparison in comparisons.items()
        })
        for family in ("success", "collision_free")
    }
    for family, family_values in adjusted.items():
        for label, p_value in family_values.items():
            comparisons[label][family]["holm_adjusted_p"] = p_value

    bayes_successes = int(sum(row["success"] for row in bayes.values()))
    output = {
        "protocol_audit": {
            "script_sha256": next(iter(script_hashes)),
            "source_config_sha256": next(iter(source_hashes)),
            "belief_sha256": next(iter(belief_hashes)),
            "case_ids": [args.case_offset, args.case_offset + args.episodes - 1],
            "episodes_per_arm": args.episodes,
            "paired": True,
            "selection_on_test": False,
        },
        "summary": summaries,
        "bayes_success_ci95_clopper_pearson": proportion_ci(
            bayes_successes, args.episodes
        ),
        "bayes_paired_comparisons": comparisons,
        "multiple_comparison_correction": {
            "method": "Holm",
            "families": ["success", "collision_free"],
            "scope": "all reported Bayes-vs-baseline binary comparisons",
        },
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
