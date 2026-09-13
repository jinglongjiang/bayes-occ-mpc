# Posterior depth diagnosis, 2026-09-13

## Decision

Continue the intention investigation, but do not treat particle count as the sole problem. Frozen-likelihood mismatch and particle approximation error both occur. No navigation benefit or real-time readiness has been established. Formal navigation and inference formulas were not changed in this diagnosis.

## Same 104 non-arrival queries

Unknown goal, conflict, not within 0.5 m of goal over the following four seconds; 15 episodes. Truth is used only for retrospective scoring and the D diagnostic.

| Predictor | 1 s | 2 s | 3 s | 4 s |
| --- | ---: | ---: | ---: | ---: |
| CV | .093378 | .278565 | .555798 | .900504 |
| C | .118019 | .345901 | .646818 | 1.003743 |
| D | .088184 | .237541 | .377307 | .500143 |
| FULL | .112575 | .320176 | .583921 | .898662 |

Episode-clustered four-horizon mean D-CV difference: -0.156267 m, 95% CI [-0.224790, -0.089682]. FULL-CV: +0.021773 m, CI [-0.073633, +0.111813]. Goal information has remaining predictive value away from arrival; the current inference has not captured that advantage. D is not a mathematical upper bound.

## Posterior reference

Three development states fixed before computation. Same prior, complete legal history, Student-t likelihood, and D model as the particle implementation. Practical convergence checks cover log normalizer, 32x32 binned probability mass, and future mean, not formal integration certification.

The 5- and 10-person states pass 128/256 and shifted-256 checks. Their converged likelihood strongly disfavors the real goal: true-goal versus maximum log likelihood is -68.63 versus -33.29, and -33.09 versus -8.64 respectively. More particles alone cannot correct these model-conditioned posteriors.

The 20-person state initially FAILS 128-to-256 binned TV (0.1668); that failure is retained in reference.json. A registered single 512 refinement passes against both 256 grids: TV .00618/.02407, maximum predictive-mean difference .00127/.00806 m. The true goal is plausible (about .507 posterior mass has higher density), while three particle seeds differ from the refined 4-second predictive mean by .192/.380/.034 m. Ambiguity and approximation error coexist. An iid-128 diagnostic has a .140 m 95th-percentile discrepancy; it is not a universal error bound.

## Evidence-model diagnosis

In the same three histories, frozen D true-goal one-step displacement RMS errors are .02810/.00778/.02484 m. A diagnostic using actual response memory and all actual neighbors reproduces all tested next positions exactly at recorded precision. Restoring memory alone with legal neighbors does not consistently improve prediction (.06419/.01001/.01092 m).

This isolates a concrete mismatch: pedestrians can respond to neighbors unavailable to the robot, while the likelihood conditions only on robot-visible neighbors and an approximate response state. These oracle diagnostics are not deployable inputs and do not prove every error comes solely from hidden neighbors. Do not fix memory alone or leak full neighbors into inference. Improving a forecast model does not automatically make it a calibrated inverse likelihood.

## Action sensitivity

Reconstructed six recorded states, verifying 94 preceding executed controls bitwise against the old planner, plus robot/human states and legal detections. CV repeat is identical. Same pre-query warm start, seed, covariance, existence, geometry, and search workload for each comparison.

| Case / step | MAP-CV first velocity norm | FULL-mean-CV | D-mean-CV |
| --- | ---: | ---: | ---: |
| 2000006 / 8 | .149857 | .016446 | .029185 |
| 2000007 / 12 | 0 | 0 | 0 |
| 2000015 / 28 | .094669 | .094669 | .094669 |
| 2000017 / 8 | 0 | 0 | 0 |
| 2000025 / 12 | .076905 | .083236 | .107498 |
| 2000027 / 20 | .170710 | .334048 | .110831 |

Units m/s. FULL-mean changes 4/6 actions, with 3/6 at least .05 m/s. All remain full-horizon feasible. Only visible unknown-goal predicted means are replaced. This does NOT consume the multimodal distribution or its between-mode covariance, does NOT establish improved actions, and is NOT closed-loop navigation evidence. Six conditional states do not estimate a population frequency.

## Measured runtime

Single-thread, serial, current goal update plus FULL rollout for all visible targets; excludes ordinary belief processing and MPC. These are three profiles, not a latency distribution.

| Scene people | Visible | FULL total ms | Update ms | Rollout ms |
| --- | ---: | ---: | ---: | ---: |
| 5 | 4 | 94.66 | .02 | 94.63 |
| 10 | 7 | 685.65 | 36.20 | 649.45 |
| 20 | 13 | 1446.70 | 85.30 | 1361.40 |

Current implementation fails 250 ms before MPC in the latter two profiles. The measured dominant cost is mode rollout, not a runtime inferred by dividing a combined benchmark by three. This does not prove all optimized implementations must fail.

## Corrections and next scope

Realized error need not decrease monotonically with particle count. A mixture is not automatically better under Energy Score (truth 0: point mass 0 scores 0; equal mass at -1/+1 scores .5). C-to-D recovery is not improvement over CV. FULL-CV on the key previous subset remains unresolved.

Next research priority: make the goal likelihood acknowledge unobserved interaction/response uncertainty, then test particle approximation against these fixed numerical references. Neither modification has been implemented here. Do not merely enlarge particles, add a network, or claim navigation improvement. Before navigation integration, require an information-fair multimodal risk interface and measured end-to-end budget compliance.

## Artifacts

reference.json and reference_*.npz: grids and particle comparisons; remaining_information.json: same-subset D; actions.json: replay checks and actions; profile.json: runtime; likelihood_model.json: oracle component diagnostics; protocol.json: frozen scope and refinement amendment. Original results remain intact.
