# Pre-Test Acceptance Review, 2026-09-10 14:42 KST

The synthetic script returned a nonzero exit code. This is retained, not relabeled
as an all-pass result. All eight empty-space and goal-recovery tests passed.
T-MPC++ also passed the stationary-obstacle test. SH stopped without collision in
front of that obstacle and timed out; two lateral-start checks also timed out.

In the failed SH trace all 100 steps returned a solution, speed approached zero,
and minimum physical clearance remained positive (about 0.107 m at the end).
Nonzero slack and large stationarity residuals were recorded. This was not a
crash, a missing command, or a fallback-only episode. It does not prove the
optimizer reached the global optimum or possessed a valid safety certificate.

A diagnostic with the same frozen controller, seed, obstacle, dynamics, weights
and covariance but a known-static-obstacle reference changed the outcome to
collision-free arrival in 8.75 s. The reference samples x=1.2*sin(pi*t), y=-4+8*t
describe a detour around the one known stationary obstacle. It is diagnostic
only, not inserted into the crossing benchmark or used to select a candidate.
This demonstrates reference sensitivity in this case, not a universal cause for
all SH failures. No complete-baseline-reproduction claim follows.

Decision: proceed with the independently specified shared-posterior crossing
evaluation, retain the failed straight-reference diagnostic in the release, and
label the native methods as goal-task-adapted acados implementations. The
experiment must not be described as all synthetic tests passing or as a general
comparison of the authors' complete navigation stacks. If the results are used
in a manuscript, the fixed straight-reference protocol and this observed SH
limitation must be explicit. No test outcome or risk setting was changed here.

Artifacts: acceptance.json, acceptance/*.json, stationary_followup.log,
stationary_reference_diag.log. The original failing acceptance.log is retained.

Startup-only verification also completed: 21 targeted tests passed; three
selected controllers exactly replayed development case 20100 in x/y/heading,
speed, both action components and time. Loaded native libraries were unchanged.
Selection was unchanged after re-sealing; only modern_repair.py's idempotent
search-path helper changed in the source manifest.
