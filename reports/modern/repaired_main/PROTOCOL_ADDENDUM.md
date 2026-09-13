# Repaired Modern MPC Evaluation, 2026-09-10

This addendum is registered before the first D1 episode in this directory.
Earlier repair results are development diagnostics, never confirmation data.

## Task and Method Identity

All arms receive only visible detections and legal history, from the same fixed
Bernoulli-Gaussian filter. They use the same continuous accelerating unicycle,
v <= 1 m/s, |a| <= 2 m/s^2, |omega| <= 0.8 rad/s, 0.25 s control period,
robot.visible=false, a 25 s deadline, and the robot-radius goal test (0.3 m).
Collision auditing uses the actual robot curve and actual pedestrian motion.
The only inputs unique to the Bayesian controller are existence weights; this
comparison does not establish a better estimator or the value of existence.

T-MPC++ retains Contouring, PathReferenceVelocity, parallel topological
guidance and EllipsoidConstraints. SH-MPC retains Contouring,
PathReferenceVelocity, scenario sampling and the native scenario optimizer.
Both are repaired acados implementations, not the authors' published numbers.
The upstream GoalModule is added, rather than replacing their defining modules.
The three predeclared objective profiles are:

| Profile | contour | terminal_contouring | terminal_angle | goal |
|---|---:|---:|---:|---:|
| legacy | 0.05 | 10 | 100 | 0 |
| track | 0.5 | 50 | 100 | 1 |
| goal_track | 2 | 50 | 100 | 10 |

The unchanged upstream goal objective divides squared goal distance by
goal_x^2 + goal_y^2 + 0.01. Its lack of translation invariance is retained and
disclosed. Stage and terminal costs use unit node weights, matching the upstream
sum convention, rather than silently amplifying the terminal cost by 1/dt.
Horizon candidates are 2 and 4 seconds. Risk values are method-specific and
must not be described as equivalent probabilities across methods.

## Selection and Data

- D1: 20 five-person layouts, 10 circle + 10 square. 24 candidates for each
  native method, 10 for Bayes. Same maximum search allowance, unused Bayes
  allowance is not filled with redundant configurations.
- D2: the top three candidates per family, 100 further five-person layouts,
  50 circle + 50 square. Rank by audited success, collision, penalized time,
  then candidate id. No 10/12/20-person outcomes enter the selector.
- T: six scenarios, 100 fresh layouts per scenario, all three selected methods,
  separately under occluded and full observation. Total 3600 episodes.
- Stability: two extra planner seeds, 20 of the test layouts per scenario,
  occluded only, 720 episodes. These are repeated layouts, not new samples.
- Latency: a separate 90-episode single-worker run, recording per-step times.

All history JSON/JSONL result files in the project and temp are scanned for case
ids. Unlike the old scanner, modern/ is not excluded. New actual layouts are
hashed and checked for duplication and overlap. Test ids are 21000-21099.
Native controllers are recreated per episode and get explicit guidance seeds;
SH joint CV sampling receives the same episode/step seed convention.

## Acceptance and Reporting

Known backend, dynamics, state/control bounds and sampling-contract tests must
pass before release. The selected native profiles additionally undergo empty
space/goal-recovery checks. No requirement that a competitor's success be within
ten percentage points of ours is imposed.

Code, shared libraries, protocol and settings are hashed at freeze and checked
again after evaluation. Malformed output, missing cases, duplicate pairs or a
changed frozen source are errors, never silently counted as navigation failures.
Report all six scenes and all outcomes. SR/CR/TR are disjoint audited events;
failures receive the fixed 25 s penalty. Success-only time is supplemental.
Primary SR comparisons are Bayes vs each modern comparator, paired by layout,
with Holm correction. Before test collection, source inspection confirmed that
CrowdSim seeds NumPy using case_id without a scene offset. All six scene variants
of a case therefore stay together in one environment-seed block. Primary event
tests use exact blockwise label sign flips; confidence intervals resample those
blocks, preserving the six-scene composition. Repeated planner seeds stay in the
same block. The 600 main-table layouts are 100 independent seed blocks, not 600
independent draws. Individual-episode McNemar is not used for the primary claim.
Compute pipeline p95 from concatenated step arrays of the single-worker run.

SH's solver success is not a safety certificate. Nonzero slack, support-bound
failure, native validation status and actual collision must all be reported.
Large performance gaps, if remaining after these checks, support only this
registered shared-posterior task comparison, not universal algorithm superiority.
