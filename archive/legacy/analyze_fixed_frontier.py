#!/usr/bin/env python3
"""Select the fixed-uncertainty development frontier without test peeking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dominates(left: dict, right: dict, eps: float = 1e-12) -> bool:
    """Return whether left is no worse in SR/CR/scored time and better in one."""
    no_worse = (
        left["sr"] >= right["sr"] - eps
        and left["cr"] <= right["cr"] + eps
        and left["failure_penalized_time"]
        <= right["failure_penalized_time"] + eps
    )
    strictly_better = (
        left["sr"] > right["sr"] + eps
        or left["cr"] < right["cr"] - eps
        or left["failure_penalized_time"]
        < right["failure_penalized_time"] - eps
    )
    return no_worse and strictly_better


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--case-offset", type=int, default=500)
    args = parser.parse_args()

    candidates = []
    expected = set(range(args.case_offset, args.case_offset + args.expected_episodes))
    for path in sorted(args.directory.glob("fixed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = payload["episodes"]
        cases = {int(row["case_id"]) for row in episodes}
        if cases != expected or len(episodes) != args.expected_episodes:
            raise RuntimeError(f"incomplete or wrong case range: {path}")
        protocol = payload["protocol"]
        planner = protocol["planner"]
        if not (
            planner["v_max"] == 1.2
            and planner["human_margin"] == 0.10
            and planner["chance_limit"] == 0.50
            and planner["near_chance_limit"] == 0.15
            and planner["probability_weight"] == 1.0
            and planner["population"] == 512
            and planner["iterations"] == 4
        ):
            raise RuntimeError(f"formal planner mismatch: {path}")
        row = dict(payload["summary"][0])
        row["radius"] = float(planner["fixed_uncertainty_radius"])
        row["path"] = str(path)
        row["script_sha256"] = protocol["script_sha256"]
        row["source_config_sha256"] = protocol["source_config_sha256"]
        candidates.append(row)

    if not candidates:
        raise RuntimeError("no fixed_*.json inputs")
    if len({row["script_sha256"] for row in candidates}) != 1:
        raise RuntimeError("fixed candidates used different controller code")
    if len({row["source_config_sha256"] for row in candidates}) != 1:
        raise RuntimeError("fixed candidates used different source configuration")

    frontier = [
        row for row in candidates
        if not any(dominates(other, row) for other in candidates if other is not row)
    ]
    frontier.sort(key=lambda row: row["radius"])
    output = {
        "selection_contract": {
            "data_role": "development_only",
            "case_ids": [args.case_offset, args.case_offset + args.expected_episodes - 1],
            "criteria": ["maximize_sr", "minimize_cr", "minimize_failure_penalized_time"],
            "test_selection_forbidden": True,
        },
        "candidates": candidates,
        "frontier_radii": [row["radius"] for row in frontier],
        "frontier": frontier,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
