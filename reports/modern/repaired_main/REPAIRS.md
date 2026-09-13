# Repair and Adaptation Ledger

## Scope

The old 11.33% / 2.50% comparator results are not accepted as evidence of
algorithmic inferiority. This evaluation uses isolated workspaces under
`build_modern/{tmpc,shmpc}_repair_n{8,16}` and new development/test layouts.
Archived experiments, manuscript files and the remote ECG task are untouched.

## Backend and Interface Defects Repaired

- The acados backend had disabled the upstream terminal objective. It now
  includes a state-only terminal objective, terminal state bounds, terminal
  obstacle constraints and the corresponding final parameter vector.
- Running/terminal objective weights now follow the upstream sum-of-node-costs
  convention, rather than acados' default running-cost-only dt scaling.
- The final warm-start input accessed u_N, which does not exist. Terminal input
  placeholders are zero; terminal input reads are rejected and regression-tested.
- Reset clears native solver memory as well as wrapper arrays.
- T-MPC's PRM no longer draws from shared random-number engines in an OpenMP
  loop. Draws are ordered; geometric checks and planner branches stay parallel.
- The reference endpoint no longer jumps between its stored terminal coordinate
  and its polynomial continuation. Native path-reference velocity is restored.
- Heading unwrapping avoids 2*pi discontinuities. Contouring's updated initial
  state is supplied to the solver. Guidance horizon and actuator limits match
  the selected solver's horizon and the registered robot.
- Braking initialization respects solver acceleration bounds. The actual common
  executor always saturates requested speed, acceleration and turning limits;
  requested saturation is recorded separately from executed motion.
- SH receives position-velocity covariance and temporally correlated CV samples,
  not marginal covariances misrepresented as independent position increments.
- SH scenario-index storage, projected warm-start loading and success-code
  conversion are repaired. Native branch reset is implemented.
- SH's slack is not fixed in the initial state, matching the upstream model's
  explicit get_xinit() exclusion. Terminal scenarios are included in validation.
- Diagnostics read actual nonlinear residuals and actual branch status rather
  than a pointer interpreted as a floating-point value or stale main-solver data.
- The common executor stops at exactly 25 s, not after an extra timeout interval.
  Arrival during the final permitted interval still counts; contact takes
  precedence over arrival. The initial development run was archived and restarted.

## Task Adaptations, Not Bug-Fix Claims

- All methods use the same continuous accelerating unicycle execution and
  collision audit. This differs from the old turn-then-translate experiment;
  old and new aggregate scores are not directly subtracted.
- Both native controllers can activate the upstream GoalModule in addition to
  their original defining modules. Contouring/guidance/scenario planning are
  not replaced with a different controller. Nonzero goal-weight selections must
  be labeled goal-task-adapted implementations in the table.
- We retain the upstream goal normalization and record its translation dependence.
- Horizons and risk units are selected separately by family on five-person
  development data. No covariance shrinkage is used to force benchmark scores.
- Shared legal-history Gaussian predictions are an input adapter, not either
  comparator's native perception system. Existence weights are not consumed by
  the two native interfaces. These experiments compare controllers, not complete
  perception pipelines or the indispensability of Bayesian estimation.

## Verification and Release

`test_modern_repair.py` checks dynamics, deadline handling, observation protocol,
terminal code generation and both native horizon builds. The C++ CV-sampling
test checks 80,000 trajectories against analytic marginal and cross-time moments.
`verify_modern_acceptance.py` tests the selected native profiles on fixed synthetic
goal-recovery and stationary-obstacle cases before the independent main table.

`modern_main_table.py develop` runs D1 and D2, then freezes candidate identities,
protocol hash, source hashes and actually loaded shared-library hashes.
`modern_main_table.py test` runs the six-scene independent table, repeated-seed
stability checks and a single-worker latency group. It checks frozen files again.
`analyze_modern_main.py` independently checks records/pairing/runtime configurations,
actual actuator limits, deadlines and raw step timings before producing the table.
Six scene variants share a CrowdSim base seed. Statistical inference therefore
uses the 100 case-id blocks intact, including any repeated planner seeds, rather
than treating all 600 scene/case combinations as independent draws.

The entry point is the CrowdNav Python environment at
`/home/abc/miniconda3/envs/crowdnav/bin/python`, working directory
`/home/abc/workspace/bayes_occ_mpc`. The queue uses two local CPU workers with at
most four native OpenMP threads each. Latency uses one worker. Neither native
solver has been represented as a GPU implementation.

Remaining performance gaps, if any, are reported rather than repaired by choosing
test layouts, weakening only one method's uncertainty, or changing the goal test.
Passing these checks does not guarantee acceptance of a paper or reproduction of
the authors' original published benchmark numbers.

## Startup-Only Revision After Development

D1 and D2 finished before this revision. The startup helper prepended identical
ROS/library search paths on every episode. Longer test workers could exceed
execve's per-variable limit. Search-path setup now removes duplicates and empty
segments, preserving the first resolution order. The previous source and freeze
are retained as modern_repair_before_idempotent.py and
frozen_before_startup_fix.json. No control, risk or selection rule changed.
Before testing, 1,000 repeated calls, native contracts, exact replay and
unchanged loaded-library hashes were verified. The same selections were re-sealed.

## Completed Evaluation and Remaining Qualification

All 6,470 planned episodes finished: D1 1,160, D2 900, independent main 3,600,
additional planning seeds 720, single-worker latency 90. Archived invalid
development records and diagnostic replays are not added to those counts.
The independent analysis passed pairing, layout, frozen-source, runtime-config,
actuation, deadline and timing-record checks. See MAIN_TABLE.md and analysis.json.

The user's requested strong-comparator outcome is not established. Occluded
six-scene SR is 85.50% for Bayes, 44.67% for T-MPC++ and 44.00% for SH-MPC.
In dense_square it is 72%, 5% and 5%, respectively. Fully observed SR remains
86.67%, 47.83% and 45.83%; the gaps cannot be attributed to occlusion inference
alone. These are qualified shared-posterior, goal-adapted implementation results,
not general claims about the authors' complete navigation stacks. A ten-point
score gap is not itself a scientific validity threshold.

Primary occluded SR differences, using the registered case-block inference:

- Bayes minus T: +40.83 pp, 95% CI [36.50,45.17], Holm p=6.51e-27.
- Bayes minus SH: +41.50 pp, 95% CI [37.50,45.50], Holm p=1.30e-25.

Statistical significance does not establish comparator transfer validity or
algorithmic novelty. The comparable score direction survives three planning
seeds, but that does not remove the reference/backend qualification.

Single-worker pipeline p95 is 169.65 ms (Bayes), 51.81 ms (T), 39.93 ms (SH),
with no sampled step exceeding 250 ms. Each arm has only 30 latency episodes;
this is a measured workload, not a worst-case real-time guarantee. Both native
comparators are computationally faster in this measurement.

Post-hoc failed-case replays exactly matched formal paths: T/dense_square/21000
and SH/dense_square/21001. On successful solver steps, maximum first predicted
position versus executed position error was 1.74e-8 and 1.62e-8 m. The T case
ended with zero usable guidance branches and a QP failure; the SH case returned
a solution with slack=0.6173 and validation status 3. Its colliding human was
already represented in legal history. This rules out a command-unit/dynamics
mismatch in these two replays, not every possible transfer defect.

No code was tuned on the independent results. The remaining work is to validate
the native reference/solver operating regime independently before presenting
these as unqualified modern-method main-table baselines. More repetitions of
the same failing configuration do not resolve that question. No covariance
shrinkage, layout selection, or forced score matching was used.
