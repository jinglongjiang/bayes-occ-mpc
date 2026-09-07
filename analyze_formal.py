#!/usr/bin/env python3
"""Paired analysis for the frozen continuous-MPC experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom_test


LABELS = (
    "sensor", "deterministic", "fixed_035", "fixed_065", "bayes", "worst", "gt"
)


def paired_bootstrap(delta: np.ndarray, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 2000
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
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
    return {"wins": wins, "losses": losses, "discordant": discordant,
            "mcnemar_exact_p": p_value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bootstrap", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=91573)
    parser.add_argument("--case-offset", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payloads = {
        label: json.loads((args.directory / f"{label}.json").read_text())
        for label in LABELS
    }
    script_hashes = {p["protocol"]["script_sha256"] for p in payloads.values()}
    source_hashes = {
        p["protocol"]["source_config_sha256"] for p in payloads.values()
    }
    if len(script_hashes) != 1 or len(source_hashes) != 1:
        raise RuntimeError("formal arms did not use identical code/source config")

    rows = {}
    expected_cases = set(range(args.case_offset, args.case_offset + args.episodes))
    for label, payload in payloads.items():
        keyed = {int(row["case_id"]): row for row in payload["episodes"]}
        if set(keyed) != expected_cases or len(keyed) != args.episodes:
            raise RuntimeError(f"{label} does not contain the locked 100 cases")
        rows[label] = keyed

    bayes = rows["bayes"]
    comparisons = {}
    for label in LABELS:
        if label == "bayes":
            continue
        success_bayes = np.array([bayes[i]["success"] for i in sorted(expected_cases)])
        success_other = np.array([rows[label][i]["success"] for i in sorted(expected_cases)])
        collision_free_bayes = 1 - np.array(
            [bayes[i]["collision"] for i in sorted(expected_cases)]
        )
        collision_free_other = 1 - np.array(
            [rows[label][i]["collision"] for i in sorted(expected_cases)]
        )
        scored_delta = np.array(
            [bayes[i]["scored_time"] - rows[label][i]["scored_time"]
             for i in sorted(expected_cases)], dtype=np.float64
        )
        path_delta = np.array(
            [bayes[i]["path_length"] - rows[label][i]["path_length"]
             for i in sorted(expected_cases)], dtype=np.float64
        )
        both_success = [
            i for i in sorted(expected_cases)
            if bayes[i]["success"] and rows[label][i]["success"]
        ]
        arrival_delta = np.array(
            [bayes[i]["nav_time"] - rows[label][i]["nav_time"]
             for i in both_success], dtype=np.float64
        )
        comparisons[label] = {
            "success": binary_pair(success_bayes, success_other),
            "collision_free": binary_pair(collision_free_bayes, collision_free_other),
            "failure_penalized_time": paired_bootstrap(
                scored_delta, args.bootstrap, args.seed
            ),
            "path_length": paired_bootstrap(path_delta, args.bootstrap, args.seed + 1),
            "both_success_episodes": len(both_success),
            "arrival_time_both_success": (
                paired_bootstrap(arrival_delta, args.bootstrap, args.seed + 2)
                if len(arrival_delta) else None
            ),
        }

    output = {
        "protocol_audit": {
            "script_sha256": next(iter(script_hashes)),
            "source_config_sha256": next(iter(source_hashes)),
            "case_ids": [args.case_offset, args.case_offset + args.episodes - 1],
            "episodes_per_arm": args.episodes,
            "paired": True,
        },
        "summary": {label: payloads[label]["summary"][0] for label in LABELS},
        "bayes_paired_comparisons": comparisons,
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
