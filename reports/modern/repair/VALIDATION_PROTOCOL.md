# Repair Validation, 2026-09-10

This is development validation, not a paper's held-out benchmark. Previous
formal artifacts are retained unchanged and cannot establish superiority of a
baseline integration now known to contain backend errors.

## Fixed Before This Validation

- Methods: bayes, tmpc_repair_n8, shmpc_repair_n8.
- Bayes uses the existing point 4 and horizon 16. Native methods use horizon 8.
- Continuous accelerating unicycle for every method: dt=0.25, v<=1,
  |a|<=2, |omega|<=0.8, physical radius=0.3, human_margin=0.1.
- robot.visible=false, no change to pedestrian ORCA, 25-second deadline.
- Native reference ends at the actual goal; goal_track weights were chosen
  from synthetic empty-goal recovery checks, not six-scene test scores.
- Development operating points: T-MPC++ risk=0.65; SH YAML risk=0.42
  (upstream effective risk=0.84). These are NOT comparable probability bounds.
- Points selected from 10 five-person development cases 11160-11169,
  maximum audited success, then minimum penalized time:
  T risks .35/.65/.90: success 6/8/7, collisions 4/2/3;
  SH risks .30/.42/.475: success 10/10/8, collisions 0/0/2.
- Validation: all six existing scenes, cases 11180-11184 per scene,
  both occluded and full-observation. These case identifiers overlap the
  historical development domain; do not call them independent final tests.
- 3 methods x 6 scenes x 5 cases x 2 visibility conditions = 180 episodes.
- Record per-step solver status, actual overlap, failures, trajectory,
  SH slack, and source/binary/settings/ROS provenance. Native process errors
  are errors, not scored as successful episodes or silently retried.

## Interpretation

Empty-goal and covariance tests establish contracts, not navigation quality.
Scores need not be similar to be valid. No parameter will be changed in response
to this validation and then relabeled as pre-frozen. Dense failures remain
visible in the report. Nonzero SH slack means the successful solve alone does
not provide its nominal collision-risk certificate.

Concurrent validation timing is not a deployment latency claim. No full formal
matrix is authorized by this file; it requires resolving integration and
fair native operating-point calibration first.
