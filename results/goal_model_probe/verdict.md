# Goal / Motion Model Factorial Gate

30 collection episodes, 12 development / 18 holdout. No training or new navigation arm.
All predictions are deterministic endpoint forecasts, not posterior calibration.
Configurations equally weighted; intervals use stratified episode bootstrap.

| Subset | Arm | 1s error m | 2s | 3s | 4s |
|---|---|---:|---:|---:|---:|
| all | CV | 0.122 | 0.342 | 0.662 | 1.073 |
| all | TURN | 0.127 | 0.355 | 0.679 | 1.092 |
| all | A | 0.301 | 0.609 | 0.913 | 1.189 |
| all | B | 0.237 | 0.462 | 0.668 | 0.845 |
| all | C | 0.154 | 0.373 | 0.625 | 0.884 |
| all | D | 0.094 | 0.198 | 0.316 | 0.435 |
| ordinary | CV | 0.132 | 0.362 | 0.691 | 1.111 |
| ordinary | TURN | 0.136 | 0.374 | 0.706 | 1.127 |
| ordinary | A | 0.308 | 0.623 | 0.932 | 1.213 |
| ordinary | B | 0.241 | 0.472 | 0.686 | 0.872 |
| ordinary | C | 0.163 | 0.385 | 0.638 | 0.895 |
| ordinary | D | 0.100 | 0.202 | 0.321 | 0.443 |
| conflict | CV | 0.092 | 0.278 | 0.566 | 0.947 |
| conflict | TURN | 0.098 | 0.294 | 0.589 | 0.975 |
| conflict | A | 0.285 | 0.574 | 0.867 | 1.137 |
| conflict | B | 0.222 | 0.428 | 0.610 | 0.754 |
| conflict | C | 0.127 | 0.336 | 0.594 | 0.869 |
| conflict | D | 0.073 | 0.179 | 0.295 | 0.404 |

| Subset | Contrast (negative better) | 1s | 2s | 3s | 4s | Improved/worse episodes |
|---|---|---:|---:|---:|---:|---:|
| all | C-A | -0.147 | -0.237 | -0.288 | -0.305 | 17/1 |
| all | D-C | -0.060 | -0.174 | -0.309 | -0.449 | 16/1 |
| all | B-A | -0.064 | -0.148 | -0.245 | -0.344 | 15/2 |
| all | C-CV | 0.032 | 0.031 | -0.037 | -0.189 | 12/6 |
| all | D-CV | -0.028 | -0.144 | -0.346 | -0.638 | 18/0 |
| all | C-TURN | 0.027 | 0.018 | -0.054 | -0.208 | 12/6 |
| all | D-TURN | -0.033 | -0.157 | -0.363 | -0.657 | 18/0 |
| ordinary | C-A | -0.145 | -0.238 | -0.295 | -0.318 | 17/1 |
| ordinary | D-C | -0.063 | -0.182 | -0.317 | -0.452 | 16/1 |
| ordinary | B-A | -0.067 | -0.151 | -0.247 | -0.341 | 15/2 |
| ordinary | C-CV | 0.031 | 0.023 | -0.053 | -0.216 | 12/6 |
| ordinary | D-CV | -0.032 | -0.160 | -0.370 | -0.668 | 18/0 |
| ordinary | C-TURN | 0.027 | 0.011 | -0.068 | -0.232 | 14/4 |
| ordinary | D-TURN | -0.037 | -0.172 | -0.385 | -0.684 | 18/0 |
| conflict | C-A | -0.158 | -0.237 | -0.273 | -0.268 | 16/2 |
| conflict | D-C | -0.053 | -0.157 | -0.299 | -0.465 | 14/1 |
| conflict | B-A | -0.063 | -0.146 | -0.257 | -0.383 | 11/4 |
| conflict | C-CV | 0.035 | 0.058 | 0.027 | -0.078 | 7/11 |
| conflict | D-CV | -0.019 | -0.099 | -0.272 | -0.542 | 17/1 |
| conflict | C-TURN | 0.028 | 0.042 | 0.005 | -0.106 | 7/11 |
| conflict | D-TURN | -0.025 | -0.115 | -0.294 | -0.571 | 17/1 |

Target-information resource gate: PASS
This does not authorize posterior training or establish closed-loop benefit.
No comparison has access to hidden neighbor truth. D/B alone receive the target goal.
Stopping at episode termination censors the last 4s. Only currently detected targets are queried.
Full confidence intervals, configuration differences and sample counts: summary.json.
The improved model is one bounded-response ORCA/TTC approximation, not a certified human model.
