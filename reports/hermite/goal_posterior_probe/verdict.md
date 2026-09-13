# Continuous Goal Posterior: Fixed Evaluation

12 development / 18 previously viewed evaluation episodes. No new navigation runs.
Every arm uses the frozen D forward model except CV. Goal-only Energy Score; no arm receives a residual kernel.
Configurations equal weight; confidence intervals cluster whole episodes.

| Subset | Metric | Arm | 1s | 2s | 3s | 4s |
|---|---|---|---:|---:|---:|---:|
| ordinary | errors | CV | 0.1316 | 0.3620 | 0.6908 | 1.1113 |
| ordinary | errors | C | 0.1630 | 0.3848 | 0.6376 | 0.8950 |
| ordinary | errors | D | 0.0997 | 0.2024 | 0.3208 | 0.4429 |
| ordinary | errors | Heading-FULL | 0.1515 | 0.3361 | 0.5412 | 0.7545 |
| ordinary | errors | Interaction-MAP | 0.1276 | 0.3024 | 0.5182 | 0.7495 |
| ordinary | errors | Interaction-FULL | 0.1241 | 0.2830 | 0.4771 | 0.6908 |
| ordinary | energy | CV | 0.1316 | 0.3620 | 0.6908 | 1.1113 |
| ordinary | energy | C | 0.1630 | 0.3848 | 0.6376 | 0.8950 |
| ordinary | energy | D | 0.0997 | 0.2024 | 0.3208 | 0.4429 |
| ordinary | energy | Heading-FULL | 0.1337 | 0.2869 | 0.4576 | 0.6346 |
| ordinary | energy | Interaction-MAP | 0.1276 | 0.3024 | 0.5182 | 0.7495 |
| ordinary | energy | Interaction-FULL | 0.1180 | 0.2633 | 0.4375 | 0.6267 |
| conflict | errors | CV | 0.0920 | 0.2784 | 0.5665 | 0.9467 |
| conflict | errors | C | 0.1267 | 0.3362 | 0.5936 | 0.8689 |
| conflict | errors | D | 0.0734 | 0.1793 | 0.2947 | 0.4043 |
| conflict | errors | Heading-FULL | 0.1167 | 0.2920 | 0.4868 | 0.6871 |
| conflict | errors | Interaction-MAP | 0.0939 | 0.2494 | 0.4410 | 0.6526 |
| conflict | errors | Interaction-FULL | 0.0947 | 0.2465 | 0.4318 | 0.6349 |
| conflict | energy | CV | 0.0920 | 0.2784 | 0.5665 | 0.9467 |
| conflict | energy | C | 0.1267 | 0.3362 | 0.5936 | 0.8689 |
| conflict | energy | D | 0.0734 | 0.1793 | 0.2947 | 0.4043 |
| conflict | energy | Heading-FULL | 0.1033 | 0.2547 | 0.4187 | 0.5844 |
| conflict | energy | Interaction-MAP | 0.0939 | 0.2494 | 0.4410 | 0.6526 |
| conflict | energy | Interaction-FULL | 0.0903 | 0.2317 | 0.3990 | 0.5755 |
| unknown_conflict | errors | CV | 0.0971 | 0.2970 | 0.5920 | 0.9607 |
| unknown_conflict | errors | C | 0.1403 | 0.4088 | 0.7479 | 1.1071 |
| unknown_conflict | errors | D | 0.0693 | 0.1972 | 0.3303 | 0.4477 |
| unknown_conflict | errors | Heading-FULL | 0.1322 | 0.3743 | 0.6549 | 0.9452 |
| unknown_conflict | errors | Interaction-MAP | 0.0966 | 0.3023 | 0.5589 | 0.8440 |
| unknown_conflict | errors | Interaction-FULL | 0.1023 | 0.3001 | 0.5531 | 0.8368 |
| unknown_conflict | energy | CV | 0.0971 | 0.2970 | 0.5920 | 0.9607 |
| unknown_conflict | energy | C | 0.1403 | 0.4088 | 0.7479 | 1.1071 |
| unknown_conflict | energy | D | 0.0693 | 0.1972 | 0.3303 | 0.4477 |
| unknown_conflict | energy | Heading-FULL | 0.1123 | 0.3158 | 0.5459 | 0.7734 |
| unknown_conflict | energy | Interaction-MAP | 0.0966 | 0.3023 | 0.5589 | 0.8440 |
| unknown_conflict | energy | Interaction-FULL | 0.0963 | 0.2813 | 0.5076 | 0.7460 |

Full paired intervals, subgroup counts, spread/error diagnostics and timings: summary.json.
No navigation integration. Particle approximation and misspecified independent residual likelihood remain limitations.

## Final Interpretation and Stronger Distribution Control

The descriptive 30% information-recovery target was reached: conflict FULL error
0.3520m versus C 0.4814m, recovering 53.2% of the observed D-C gap. This is not a
claim of new navigation performance or a theoretical oracle bound.

After excluding known-birth goals, 201 conflict queries in 15 episodes remain.
Their average point errors are CV 0.4867, C 0.6010, Heading-FULL 0.5267,
Interaction-MAP 0.4504, Interaction-FULL 0.4481, and D 0.2611 metres.
FULL-C has interval [-0.2550,-0.0610]m; FULL-CV [-0.0982,+0.0207]m and
FULL-MAP [-0.0447,+0.0451]m do not establish a clear improvement.

A supplementary fairness audit adds the same development-fitted D residual
Gaussian to ALL arms. It does not modify inference or any point predictions.
4096 independent Monte Carlo pairs per query/horizon estimate Energy Score,
with common random numbers across arms. This is not joint-process calibration.

| Arm | Ordinary ES | Conflict ES | Unknown-goal conflict ES |
|---|---:|---:|---:|
| CV | 0.476 | 0.381 | 0.385 |
| C | 0.413 | 0.392 | 0.475 |
| Heading-FULL | 0.340 | 0.310 | 0.400 |
| Interaction-MAP | 0.349 | 0.301 | 0.367 |
| Interaction-FULL | 0.319 | 0.285 | 0.353 |
| D | 0.236 | 0.210 | 0.225 |

On unknown-goal conflict queries, common-kernel ES intervals for FULL-MAP and
FULL-CV are [-0.0436,+0.0198] and [-0.0801,+0.0153]. Do not substitute the
more favorable goal-only ES analysis for this stronger control.

Post-hoc goal-arrival stratification finds stronger gains near destinations.
On the 104 unknown-goal conflict queries not approaching a goal, FULL-C has
interval [-0.163,+0.056]m; 4s errors CV/FULL are 0.901/0.899m. Thus routine
conflict avoidance has not shown a clear predictive improvement.

Particle convergence is not established: a fixed development square query
has 4s errors 1.325/0.813/1.347m with 64/128/256 particles. Two additional
128-particle development seeds give 0.629/0.968m. Evaluation settings were
not retuned. The originally uninformative known-birth 5-person stability
check was supplemented with an unknown-goal query; all records are retained.

All 1200 baseline predictions match the previous CV/C/D results exactly.
238 known-birth queries yield no extra inference gain. Mode weights sum to
one within 4.11e-15. Nontrivial label-permutation and causal-prefix checks pass.
The common-kernel audit preserves original point-score records byte for byte.

Total recorded episode evaluation wall time is 895.53 seconds. The 20-person
per-target query median is 639ms for the COMBINED two-FULL-plus-MAP diagnostic,
not individual-arm inference time. No 250ms end-to-end admission is claimed.

Final decision: usable developmental prediction gain, but neither a resolved
posterior-convergence problem nor established full-posterior navigation value.
Formal MPC, risk settings and the paper remain unchanged.
