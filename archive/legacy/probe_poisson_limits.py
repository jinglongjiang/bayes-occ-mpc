#!/usr/bin/env python3
"""Development-only sweep of undetected-target chance limits."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

from continuous_mpc_gate import DEFAULT_CROWDNAV, MPCConfig, run_episode


CASES = (40, 518, 521, 525, 539, 544, 550, 575, 597)
LIMITS = (0.10, 0.15, 0.20)


def evaluate(limit, case_id):
    config = replace(
        MPCConfig(),
        population=512,
        iterations=4,
        chance_limit=0.50,
        occupancy_chance_limit=limit,
        probability_weight=0.0,
    )
    result = run_episode(
        DEFAULT_CROWDNAV,
        "bayes_poisson",
        20,
        "circle_crossing",
        case_id,
        config,
    )
    return limit, case_id, result


def main():
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(evaluate, limit, case) for limit in LIMITS for case in CASES]
        for future in as_completed(futures):
            limit, case, result = future.result()
            rows.append((limit, case, result))
            print(
                f"limit={limit:.2f} case={case:03d} event={result.event} "
                f"time={result.nav_time:.2f} clear={result.min_clearance:.3f}",
                flush=True,
            )
    print("\nSUMMARY")
    for limit in LIMITS:
        subset = [result for value, _, result in rows if value == limit]
        print(
            limit,
            "success", sum(result.success for result in subset),
            "collision", sum(result.collision for result in subset),
            "timeout", sum(result.timeout for result in subset),
        )


if __name__ == "__main__":
    main()
