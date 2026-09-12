# Intent repair: end-to-end results

Frozen original D and production MPC; only experiment likelihood and risk adapter changed.
Prediction evaluation reuses viewed data. Navigation uses separately registered new layouts. No threshold/search tuning.

## Prediction: unknown-goal conflict

| Arm | 1 s | 2 s | 3 s | 4 s | Mean ES |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV | 0.0971 | 0.2970 | 0.5920 | 0.9607 | 0.3857 |
| OldFULL | 0.1023 | 0.3001 | 0.5531 | 0.8368 | 0.3543 |
| WideFULL | 0.0925 | 0.2804 | 0.5099 | 0.7757 | 0.3177 |
| NewMAP | 0.1020 | 0.3084 | 0.5785 | 0.8701 | 0.3755 |
| NewFULL | 0.0938 | 0.2822 | 0.5071 | 0.7682 | 0.3153 |
| D | 0.0693 | 0.1972 | 0.3303 | 0.4477 | 0.2266 |

## Navigation and complete runtime

| Arm | N | Success | Collision | Timeout | Penalty s | p50 ms | p95 ms | p99 ms | >250 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O | 18 | 17 | 1 | 0 | 12.708 | 68.8 | 119.7 | 122.4 | 0.0% |
| B | 18 | 17 | 0 | 1 | 11.861 | 75.2 | 129.8 | 131.4 | 0.0% |
| F0 | 18 | 17 | 0 | 1 | 12.625 | 1267.3 | 4256.1 | 5501.2 | 87.6% |
| M | 18 | 18 | 0 | 0 | 11.778 | 1397.0 | 8927.3 | 11971.2 | 84.2% |
| F | 18 | 18 | 0 | 0 | 12.208 | 4706.2 | 14128.6 | 17600.4 | 91.9% |

## Checks and limitations

- Numerical status: {"count": 256, "mh": 2, "numerical_pass": false, "note": "particle TV to fine bins is a stringent finite-sample test; failure retained, not redefined"}
- Compression status: {"k": 256, "passed": false, "fallback": "if no K passes, retain all inference particles before deduplicating identical goals; report missed runtime rather than silently lose mode accuracy"}
- Risk is conditional on a fitted isotropic marginal D error kernel, not a certified joint safety probability.
- Current CV geometry is common to all arms and may block mode-dependent routes.
- F versus F0 includes both likelihood repair and the registered particle/MH refinement; L2 versus L1 isolates the temporal likelihood at matched approximation.
- Offline action clearance uses only available future labels; records shorter than four seconds are flagged, not extended.
- The factorial diagnostic uses the actual ORCA/TTC generator without extra D rollout clipping in all four cells; D forecasts remain frozen.
- Timing is measured serially, includes inference and MPC, excludes environment truth/audit and writing.
- Risk is a subset of MPC time, not an additional summand. Component percentiles do not add.
- Goal MAP optimization is included in goal-update time for M.
- No claim of novelty from AR residuals, Rao-Blackwellization or medoids alone.

## Paired outcomes

- {"left": "F", "right": "M", "cases": 18, "transitions": {"success -> success": 18}, "penalty_delta": 0.4305555555555556, "CI": [-0.4305555555555556, 1.263888888888889], "beneficial": 0, "harmful": 0}
- {"left": "F", "right": "F0", "cases": 18, "transitions": {"success -> success": 17, "timeout -> success": 1}, "penalty_delta": -0.4166666666666667, "CI": [-1.2777777777777777, 0.3194444444444444], "beneficial": 1, "harmful": 0}
- {"left": "F", "right": "B", "cases": 18, "transitions": {"success -> success": 17, "timeout -> success": 1}, "penalty_delta": 0.3472222222222222, "CI": [-1.6805555555555556, 1.8888888888888886], "beneficial": 1, "harmful": 0}
- {"left": "B", "right": "O", "cases": 18, "transitions": {"success -> success": 16, "collision -> success": 1, "success -> timeout": 1}, "penalty_delta": -0.8472222222222222, "CI": [-2.6666666666666665, 0.986111111111111], "beneficial": 1, "harmful": 1}

Prediction contrasts and grouped latency/navigation tables are in the neighboring JSON reports.

## Final decision

The complete experiment finished: 1,200 prediction queries, 36 common-candidate action states,
and 90 closed-loop episodes on 18 new paired layouts (six per configuration). No execution
errors or partial episodes occurred. The original nav/ and integration/ trees are unchanged
from fecf501. The implemented prototype is runnable, but has NOT passed acceptance as a
real-time navigation method with demonstrated additional value from the new likelihood or FULL.

1. Prediction: on 201 unknown-goal conflict queries, NewFULL mean position error is
   0.4128 m versus CV 0.4867, OldFULL 0.4481, WideFULL 0.4146 and NewMAP 0.4647.
   The episode-clustered 95% interval for NewFULL minus CV is [-0.1376, -0.0070] m;
   versus NewMAP it is [-0.0992, -0.0023] m. These are exploratory results on reused data.
2. Specific likelihood mechanism: NewFULL minus WideFULL is -0.0018 m with interval
   [-0.0130, 0.0054]. Its Energy Score difference also crosses zero. This does not establish
   an advantage of temporal nuisance modeling over independent wider noise. NewFULL versus
   OldFULL includes particle/MH changes, so that comparison cannot isolate the likelihood.
3. Nonarrival conflicts: on the fixed 104-query subset, four-second errors are CV 0.9005,
   OldFULL 0.8987, WideFULL 0.8410, NewMAP 0.9545, NewFULL 0.8421 and D 0.5001 m.
   NewFULL minus CV averaged across horizons is -0.0128 m, interval [-0.1037, 0.0752].
   The improvement on this subset remains unresolved; the interval does not prove equality.
4. Navigation: FULL and MAP both succeeded in all 18 layouts. FULL recovered one failure
   relative to each of O, B and F0, with no lost successes. However, MAP recovered those
   same failures. FULL's penalty is 0.4306 s higher than MAP, interval [-0.4306, 1.2639].
   FULL also has a 0.3472 s higher penalty than B despite rescuing B's one timeout.
   These 18 layouts do not demonstrate a FULL-specific navigation advantage or a safety guarantee.
5. Runtime and approximation: numerical particle accuracy and compressed-mode precision gates
   both failed at their registered limits. Full-mode fallback was used, not forced compression.
   FULL total p50/p95/p99 is 4706/14129/17600 ms, with 91.9% over 250 ms. MAP also fails
   the budget. Synchronous closed-loop successes must not be described as real-time deployment.

### Where FULL spends time (20-person configuration, median milliseconds)

| Component | Median ms |
| --- | ---: |
| Legal observation and base belief | 1.8 |
| Goal inference | 1323.0 |
| Mode preparation | 1.8 |
| D future rollouts | 3267.7 |
| MPC, including mixture risk | 3875.9 |
| Mixture risk, a subset of MPC | 3819.9 |
| Complete pipeline | 8940.9 |

Component medians do not add. The computational bottleneck is not just the small nuisance
filter: full-mode D rollout and multimodal risk evaluation are both substantial. No extra
optimization, training, parameter search, new scenes, or navigation queue was opened after
this registered experiment. Keep this as a complete exploratory result, not a verified
real-time replacement for the production controller or evidence of a new Bayesian mechanism.
