# Repaired Modern MPC Comparison

Independent test only; 600 layouts per arm and observation condition.
The controllers share legal-history predictions and a continuous accelerating unicycle executor.
SR/CR/TR partition outcomes; collision takes precedence over timeout when both are recorded.
Penalty time is arrival time on collision-free success and 25 s on failure.

The 600 layouts comprise 100 environment-seed blocks, each containing six scenes.
Primary inference keeps each seed block intact; repeated planner seeds are not new independent environments.

## occluded

| Controller | SR (%) | CR (%) | TR (%) | Penalty time (s) |
|---|---:|---:|---:|---:|
| Bayes-MPC | 85.50 | 14.17 | 0.33 | 14.034 |
| T-MPC++ (acados, goal-adapted) | 44.67 | 53.50 | 1.83 | 19.532 |
| SH-MPC (acados, goal-adapted) | 44.00 | 31.67 | 24.33 | 21.259 |

## fully_observed

| Controller | SR (%) | CR (%) | TR (%) | Penalty time (s) |
|---|---:|---:|---:|---:|
| Bayes-MPC | 86.67 | 12.83 | 0.50 | 13.744 |
| T-MPC++ (acados, goal-adapted) | 47.83 | 49.83 | 2.33 | 19.227 |
| SH-MPC (acados, goal-adapted) | 45.83 | 29.67 | 24.50 | 21.062 |

## Limits

- These are repaired acados implementations with disclosed goal-task profiles, not claimed bitwise reproductions of the authors' published experiments.
- Synthetic acceptance was not an all-pass result: SH stopped before a stationary obstacle on a straight reference, while a known-static detour permitted arrival with the same settings. See ACCEPTANCE_REVIEW.md; the benchmark retains its fixed straight-reference protocol.
- Risk parameters describe different events; identical numerical risk or equivalent safety certificates are not asserted.
- Shared-posterior controller comparisons cannot by themselves establish that Bayesian estimation is indispensable.
- Lower penalty time may result from fewer failures. Success-only times and paired arrival times on jointly successful cases are secondary, conditional analyses, not unconditional proof of faster travel.
- Concurrent main-table timing is not the latency benchmark. Use the separate single-worker latency records.
- Native solver success is not a safety certificate; report SH slack/support status and executed collision audit.
- See analysis.json for scene-wise results, paired confidence intervals, Holm-adjusted primary tests and seed-clustered stability.
