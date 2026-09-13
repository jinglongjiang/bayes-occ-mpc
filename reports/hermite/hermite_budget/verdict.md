# Population-only budget pilot

| Condition | Arm | N | Success/Collision/Timeout | Penalty s | Pipeline p50/p95/p99 ms | Overrun % |
|---|---|---:|---|---:|---|---:|
| combined | exact | 640 | 50/7/3 | 15.508 | 165.69/192.69/199.46 | 0.000 |
| combined | hermite | 640 | 50/7/3 | 15.508 | 139.50/181.51/186.64 | 0.000 |
| clean | exact | 640 | 29/1/0 | 12.783 | 168.61/191.69/198.80 | 0.000 |
| clean | hermite | 640 | 29/1/0 | 12.783 | 138.86/167.33/169.40 | 0.000 |
| severe | exact | 640 | 21/6/3 | 18.233 | 162.27/193.44/200.22 | 0.000 |
| severe | hermite | 640 | 21/6/3 | 18.233 | 143.43/183.01/191.68 | 0.000 |

A failed / B succeeded: 0; A succeeded / B failed: 0.
Complete transitions: {"collision -> collision": 7, "success -> success": 50, "timeout -> timeout": 3}
Budget qualified in both conditions: True. Engineering continuation gate: False.

Verdict 2: retain computation acceleration only; do not expand this navigation-benefit experiment.

This is a 60-layout engineering pilot, not a significance or safety-equivalence claim.
Only clean/severe 20-person dense_square; synchronous soft budget, not hard real-time deployment.
No iCEM/EWMA/480-episode confirmation was started.

## Why both arms selected 640

All development counts below are over the same 20 layouts per configuration.
Navigation outcomes matched between exact and Hermite at each fixed population.

| N | Success/Collision/Timeout | Penalty s | Exact eligible | Hermite eligible |
|---:|---|---:|---|---|
| 512 | 17/2/1 | 16.613 | yes | yes |
| 640 | 18/1/1 | 16.613 | yes | yes |
| 768 | 16/2/2 | 16.475 | yes | yes |
| 896 | 14/4/2 | 17.600 | no | yes |
| 1024 | 18/1/1 | 15.113 | no | no |

Hermite admits 896 candidates within the development budget, but this configuration
has more collisions and fewer successes than 640. The faster navigation at 1024
does not qualify: its clean/severe pipeline p99 is 275.8/285.2 ms for Hermite and
325.8/337.8 ms for exact. No risk thresholds or budget criteria were relaxed.

## Final independent audit

- Baseline: f4fbb18; isolated experiment implementation: 26c7a60.
- 200 development episodes plus 120 independent pilot episodes; no code errors.
- 80 distinct new layouts checked against 30,039 historical JSON/JSONL files,
  4,955 historical case IDs reconstructed under the current dense-square generator,
  and recorded historical layout hashes. No unreadable history files.
- All 17,982 executed control steps passed the holonomic displacement check.
- 160 paired episode records matched exactly in actions, positions, physical
  clearances, outcomes and penalty; pilot contributed 60 of these pairs.
- Independent population selection reproduced 640 for both arms.
- Frozen source hashes and protocol digest remained unchanged. No differences
  from f4fbb18 in nav/ or integration/.
- Pilot timing uses 3,183 actual closed-loop control steps per arm, not averages
  of episode quantiles. All recorded pilot steps met 250 ms; this is not a hard
  real-time guarantee beyond the measured run.

最终选择：② 仅保留计算加速贡献。本轮未获得独立导航收益，停止扩展本试验。
