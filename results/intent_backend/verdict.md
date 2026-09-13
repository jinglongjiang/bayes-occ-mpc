# Frozen-model computational backend audit

Same 12 development states and parameters; three technical repeats per backend. No navigation experiment.
The original RVO LP and constraint code is compiled through access-only friend declarations in generated headers.
History contexts cache only state-dependent constraints/TTC. Every proposed goal still replays its own complete likelihood.
Future modes have separate constraints after their states diverge. Neighbor CV predictions and physical constraints are unchanged.
Risk fusion retains SciPy CDF, Hermite nodes/bounds, per-person mode mixture, existence and clipping.

| People | Backend | Goal update ms | D rollout ms | Risk ms | MPC other ms | Total p50 ms | p95 ms | >250 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | original | 31.40 | 591.65 | 985.04 | 30.39 | 1636.67 | 3261.54 | 100.0% |
| 5 | behavior_only | 1.26 | 2.79 | 989.14 | 30.52 | 1027.25 | 1877.50 | 100.0% |
| 5 | risk_only | 31.38 | 593.03 | 303.63 | 30.19 | 951.51 | 1997.14 | 100.0% |
| 5 | combined | 1.32 | 2.80 | 303.54 | 29.52 | 338.49 | 614.00 | 50.0% |
| 10 | original | 406.77 | 1687.00 | 2243.35 | 49.97 | 4438.38 | 4806.03 | 100.0% |
| 10 | behavior_only | 5.44 | 8.39 | 2242.73 | 51.37 | 2312.89 | 2533.40 | 100.0% |
| 10 | risk_only | 404.98 | 1688.61 | 595.39 | 44.87 | 2779.72 | 3059.39 | 100.0% |
| 10 | combined | 5.49 | 8.43 | 593.81 | 44.49 | 653.36 | 891.85 | 100.0% |
| 20 | original | 1575.28 | 5691.29 | 5515.06 | 64.91 | 13010.21 | 14905.34 | 100.0% |
| 20 | behavior_only | 14.82 | 33.69 | 5488.25 | 68.21 | 5603.25 | 6338.46 | 100.0% |
| 20 | risk_only | 1577.90 | 5689.11 | 1584.49 | 66.29 | 9052.34 | 10419.61 | 100.0% |
| 20 | combined | 14.86 | 33.66 | 1583.18 | 68.10 | 1705.98 | 1764.84 | 100.0% |

Component medians do not add. Compilation, prefix restoration, deepcopy and assertions are outside timing.
Timing includes a fresh legal observation/filter update, goal update, unchanged compression, prediction, risk and complete MPC.
Primary timings start with empty historical context caches; only reuse within the current update is counted.
These are development-state latency samples, not all-step closed-loop real-time certification.
Conclusion: all four computational changes are implemented, but the unchanged full-mode method does not meet the 250 ms budget.
At 20 people the remaining dominant cost is mixture risk, not goal inference or behavior rollout. No representation redesign is included.
Previous negative scientific results and failed posterior/compression precision gates remain unchanged.
