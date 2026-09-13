# Occlusion Closeout Results

4080 registered confirmation episodes completed; 100 additional development episodes are not test data.
No model, risk parameter, or sample-count adjustment based on confirmation outcomes.

## Decision

- The position-uncertainty contribution survives independent testing: Bayes has 10 collisions versus 46 for the shared-tracker posterior mean; the paired reduction is 6.0 percentage points, with a 95% seed-block interval of 4.0 to 8.0 points.
- The former freezing explanation does not survive the stronger age-margin calibration: both methods have 586 successes, and Age has three timeouts versus four for Bayes. Bayes has a shorter penalized time, not established equal safety.
- Planning existence weighting has no demonstrated additional benefit: covariance-one has seven collisions and 589 successes. The full method is not uniformly best across configurations.
- Bayes does not establish a collision advantage over EWMA, Conformal, or Age after the registered secondary comparisons. These are competitive alternatives, not failed baselines.
- The visibility subset does not establish an extra human-occlusion-specific effect beyond range restriction; the interaction intervals include zero.
- The defensible paper is a fully specified navigation system and controlled mechanism study. These experiments do not create a new Bayesian theorem or guarantee journal acceptance. No additional algorithm branch or outcome-seeking queue is launched.

## Main Table

| Configuration | Method | n | Success | Collision | Timeout | Penalty (s) |
|---|---|---:|---:|---:|---:|---:|
| baseline_circle | posterior_mean | 100 | 99 | 1 | 0 | 9.613 |
| baseline_circle | covariance | 100 | 100 | 0 | 0 | 9.818 |
| baseline_circle | bayes | 100 | 100 | 0 | 0 | 9.818 |
| baseline_circle | age_margin | 100 | 100 | 0 | 0 | 10.055 |
| baseline_circle | ewma | 100 | 100 | 0 | 0 | 9.762 |
| baseline_circle | conformal | 100 | 100 | 0 | 0 | 10.102 |
| baseline_square | posterior_mean | 100 | 93 | 7 | 0 | 10.485 |
| baseline_square | covariance | 100 | 96 | 3 | 1 | 10.595 |
| baseline_square | bayes | 100 | 97 | 3 | 0 | 10.465 |
| baseline_square | age_margin | 100 | 97 | 2 | 1 | 11.155 |
| baseline_square | ewma | 100 | 95 | 5 | 0 | 10.457 |
| baseline_square | conformal | 100 | 95 | 3 | 2 | 11.443 |
| dense_circle | posterior_mean | 100 | 95 | 5 | 0 | 12.010 |
| dense_circle | covariance | 100 | 100 | 0 | 0 | 11.492 |
| dense_circle | bayes | 100 | 100 | 0 | 0 | 11.555 |
| dense_circle | age_margin | 100 | 100 | 0 | 0 | 11.717 |
| dense_circle | ewma | 100 | 100 | 0 | 0 | 11.482 |
| dense_circle | conformal | 100 | 99 | 1 | 0 | 11.945 |
| dense_square | posterior_mean | 100 | 76 | 24 | 0 | 14.633 |
| dense_square | covariance | 100 | 94 | 3 | 3 | 13.697 |
| dense_square | bayes | 100 | 90 | 6 | 4 | 14.095 |
| dense_square | age_margin | 100 | 95 | 3 | 2 | 14.515 |
| dense_square | ewma | 100 | 90 | 7 | 3 | 13.900 |
| dense_square | conformal | 100 | 87 | 5 | 8 | 15.318 |
| large_circle | posterior_mean | 100 | 99 | 1 | 0 | 14.705 |
| large_circle | covariance | 100 | 100 | 0 | 0 | 14.873 |
| large_circle | bayes | 100 | 100 | 0 | 0 | 14.807 |
| large_circle | age_margin | 100 | 100 | 0 | 0 | 15.332 |
| large_circle | ewma | 100 | 100 | 0 | 0 | 14.850 |
| large_circle | conformal | 100 | 100 | 0 | 0 | 15.535 |
| large_square | posterior_mean | 100 | 92 | 8 | 0 | 11.182 |
| large_square | covariance | 100 | 99 | 1 | 0 | 11.435 |
| large_square | bayes | 100 | 99 | 1 | 0 | 11.402 |
| large_square | age_margin | 100 | 94 | 6 | 0 | 12.857 |
| large_square | ewma | 100 | 97 | 3 | 0 | 11.232 |
| large_square | conformal | 100 | 97 | 2 | 1 | 12.630 |

## Paired Comparisons

Intervals resample the 100 seed blocks, each containing six configurations. Secondary collision p-values use Holm correction.

Bayes minus posterior_mean: {"a": "bayes", "b": "posterior_mean", "collision_union": {"delta": -0.06, "ci95": [-0.08, -0.039999999999999994]}, "seed_block_randomization_p": 4.999750012499375e-05, "success_without_overlap": {"delta": 0.05333333333333333, "ci95": [0.03333333333333333, 0.07333333333333333]}, "penalty": {"delta": -0.08083333333333334, "ci95": [-0.3925, 0.22376041666666655]}, "A_fail_B_success": 8, "A_success_B_fail": 40, "descriptive_mcnemar_p": 4.405936238072169e-08}

Bayes minus covariance: {"a": "bayes", "b": "covariance", "collision_union": {"delta": 0.005, "ci95": [0.0, 0.011666666666666665]}, "seed_block_randomization_p": 0.2456377181140943, "success_without_overlap": {"delta": -0.005, "ci95": [-0.013333333333333332, 0.0016666666666666666]}, "penalty": {"delta": 0.03874999999999999, "ci95": [-0.05583333333333333, 0.13958333333333334]}, "A_fail_B_success": 4, "A_success_B_fail": 1, "descriptive_mcnemar_p": 0.25, "secondary_holm_p": 0.9825508724563772}

Bayes minus age_margin: {"a": "bayes", "b": "age_margin", "collision_union": {"delta": -0.0016666666666666672, "ci95": [-0.013333333333333332, 0.01]}, "seed_block_randomization_p": 1.0, "success_without_overlap": {"delta": 5.551115123125783e-19, "ci95": [-0.013333333333333332, 0.013333333333333332]}, "penalty": {"delta": -0.5816666666666667, "ci95": [-0.8020833333333331, -0.3554166666666667]}, "A_fail_B_success": 8, "A_success_B_fail": 8, "descriptive_mcnemar_p": 0.9999999999999999, "secondary_holm_p": 1.0}

Bayes minus ewma: {"a": "bayes", "b": "ewma", "collision_union": {"delta": -0.008333333333333333, "ci95": [-0.021666666666666664, 0.003333333333333333]}, "seed_block_randomization_p": 0.3136343182840858, "success_without_overlap": {"delta": 0.006666666666666667, "ci95": [-0.005, 0.02]}, "penalty": {"delta": 0.07624999999999998, "ci95": [-0.12083333333333333, 0.26]}, "A_fail_B_success": 4, "A_success_B_fail": 8, "descriptive_mcnemar_p": 0.22656250000000003, "secondary_holm_p": 0.9825508724563772}

Bayes minus conformal: {"a": "bayes", "b": "conformal", "collision_union": {"delta": -0.0016666666666666666, "ci95": [-0.013333333333333332, 0.008333333333333333]}, "seed_block_randomization_p": 1.0, "success_without_overlap": {"delta": 0.013333333333333332, "ci95": [-2.8449465006019007e-19, 0.028333333333333332]}, "penalty": {"delta": -0.805, "ci95": [-1.0225, -0.5991562500000001]}, "A_fail_B_success": 7, "A_success_B_fail": 15, "descriptive_mcnemar_p": 0.9999999999999999, "secondary_holm_p": 1.0}

## Visibility Attribution

Each row contains 120 layouts. Combined observations reuse the registered main comparison.
| Visibility | Method | Success | Collision | Timeout | Penalty (s) |
|---|---|---:|---:|---:|---:|
| range_and_occlusion | posterior_mean | 111 | 9 | 0 | 12.160 |
| range_and_occlusion | covariance | 118 | 2 | 0 | 12.110 |
| range_only | posterior_mean | 113 | 7 | 0 | 11.898 |
| range_only | covariance | 118 | 2 | 0 | 11.981 |
| full | posterior_mean | 116 | 4 | 0 | 11.510 |
| full | covariance | 118 | 2 | 0 | 11.969 |

{"comparison": "combined minus range_only", "field": "collision_union", "delta": -0.016666666666666666, "ci95": [-0.041666666666666664, 0.0]}

{"comparison": "combined minus range_only", "field": "success_without_overlap", "delta": 0.016666666666666666, "ci95": [0.0, 0.041666666666666664]}

{"comparison": "combined minus range_only", "field": "penalty", "delta": -0.13333333333333328, "ci95": [-0.5041666666666667, 0.175]}

{"comparison": "combined minus full", "field": "collision_union", "delta": -0.041666666666666664, "ci95": [-0.08333333333333334, 0.0]}

{"comparison": "combined minus full", "field": "success_without_overlap", "delta": 0.041666666666666664, "ci95": [0.0, 0.08333333333333334]}

{"comparison": "combined minus full", "field": "penalty", "delta": -0.5083333333333333, "ci95": [-1.1854166666666666, 0.1313020833333326]}

## Complete Outcome Transitions

Rows are Bayes outcomes and columns are comparator outcomes; all transitions are retained.

posterior_mean: {"success": {"success": 546, "collision": 40, "timeout": 0}, "collision": {"success": 5, "collision": 5, "timeout": 0}, "timeout": {"success": 3, "collision": 1, "timeout": 0}}

covariance: {"success": {"success": 585, "collision": 0, "timeout": 1}, "collision": {"success": 3, "collision": 7, "timeout": 0}, "timeout": {"success": 1, "collision": 0, "timeout": 3}}

age_margin: {"success": {"success": 578, "collision": 7, "timeout": 1}, "collision": {"success": 6, "collision": 4, "timeout": 0}, "timeout": {"success": 2, "collision": 0, "timeout": 2}}

ewma: {"success": {"success": 578, "collision": 8, "timeout": 0}, "collision": {"success": 3, "collision": 7, "timeout": 0}, "timeout": {"success": 1, "collision": 0, "timeout": 3}}

conformal: {"success": {"success": 571, "collision": 7, "timeout": 8}, "collision": {"success": 6, "collision": 4, "timeout": 0}, "timeout": {"success": 1, "collision": 0, "timeout": 3}}

## Data and Code Map

| Evidence | Data | Code |
|---|---|---|
| Frozen allocation and loaded parameters | confirmation_protocol.json; calibration.json | experiments/occlusion_confirmation.py |
| Added age-margin development points | development.jsonl; development_summary.json | experiments/occlusion_closeout.py |
| New main and visibility results | episodes.jsonl; confirmation_summary.json | experiments/occlusion_confirmation.py summarize |
| Covariance attribution and full event transitions | covariance_attribution.json; outcome_transitions.json | experiments/occlusion_report.py |
| Same-state candidate mechanisms | mechanism/*.pkl; mechanism_summary.json | experiments/occlusion_report.py |
| Recorded-action uncertainty trace | uncertainty_trace.json | experiments/occlusion_report.py |
| Historical manuscript and numerical tables | manuscript_before.txt; historical archive | Original versions; not new confirmation |

## Boundaries

- No safety-equivalence inference from nonsignificance.
- Opportunity counts are Gaussian quadrature diagnostics, not measured benefits of a negative-evidence algorithm.
- Same-controller controls still differ in risk channels and calibration; they are not immune to fairness criticism.
- Configuration differences combine geometry and pedestrian count, not density alone.
- Concurrent timings do not establish a real-time deployment guarantee.
- The unchanged simulator records its 25 timeout endpoints at 25.25 s on the control grid; all receive the fixed 25 s failure penalty. No success or collision terminates after 25 s.
- Scientific outcomes do not establish novelty or guarantee journal acceptance.

## Reproduction

The compressed bundle contains original episode records, calibration inputs, snapshots, figures, and the manuscript with its local assets. Its manifest records each content hash and original path.
The original protocol is not rewritten to pretend that historical absolute paths were portable. The portable analysis command resolves archived inputs explicitly, verifies their original hashes and the executing core source, and requires the recorded RVO2 binary.
```bash
mkdir -p /tmp/occlusion-closeout-check
tar -xzf results/occlusion_closeout/reproduction_bundle.tar.gz -C /tmp/occlusion-closeout-check
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/occlusion_report.py portable /tmp/occlusion-closeout-check
# In /tmp/occlusion-closeout-check/manuscript:
tectonic latex_FCS.txt --keep-logs
```
Navigation can be rerun on the original registered paths with `python experiments/occlusion_confirmation.py run --workers 6`; existing episode keys are skipped. Use an explicitly separate result copy for a fresh rerun, not a mixture of old and new timing records.
Portable reanalysis is not an independent simulator installation or a new scientific replication. See portable_analysis_check.json and the existing repository reproducibility documentation for the tested scope.

## Continuation: Risk Relevance and Submission Position

The subsequent read-only audit replayed all 600 Bayes layouts and 27,978 control states with the original seeds and individual warm starts. Every first action and executed robot position reproduced within 1e-9. The original 4,080-episode file remains unchanged.

At the registered 1-percentage-point selected-plan risk-reduction threshold, the 15-point-per-axis screen identifies 4,992 states (17.84%; descriptive seed-block interval 16.92--18.76%). This is 22.17% of states with hidden tracks and 39.85% of geometric-opportunity states. The six configuration rates are 6.93%, 7.70%, 24.45%, 28.14%, 20.45%, and 14.14% in the registered order. Exact rectangular-cell Gaussian masses, combined with monotonic set-inclusion checks, bracket the fraction at 16.47--20.80%. This bracket is numerical classification uncertainty, not a confidence interval.

Thus the 10% potential-relevance threshold passes. It measures removal of ALL eligible tracks, not a legal posterior update or actual constraint change; it does not prove that negative evidence will improve actions. The diagnostic only bounds this risk-reduction question, not effects on alternative candidates or risk increases.

The next, separately registered, developmental probe compared the old Gaussian, a one-frame spatially conditioned distribution, and its full-covariance Gaussian projection on the 12 previously fixed snapshots. All methods used the same 24-or-fewer candidates and unchanged geometry and existence weights. No new online filter, planner, or navigation episode was added. All first commands remained unchanged. Only four states activate the update; three of those meet the probability-refinement tolerance. Overall 11/12 states are numerically stable. The registered 3-state/2-layout shape-specific action-change threshold is not met. This small coverage does NOT establish absence of a spatial-evidence benefit in general or under recursive filtering. No favorable states are added after inspecting this result.

Whole-world mirroring is not a novelty test: an equivariant margin planner can mirror its actions too. The probe instead isolates spatial conditioning and Gaussian projection on fixed inputs. Its coupled distance-reflection check is only numerical self-consistency.

The repaired external-adapter table now includes configuration-wise outcomes, native solver success, and collision-interval END speeds, recomputed from 3,600 historical records. All 42 aggregate rows match the repaired analysis. T-MPC++ solve success is 40.93% in dense square, not the superseded 10.3%. SH-MPC has 35,763 successful replies in 35,769 steps, and all 190 collision-terminal replies report success and nonzero slack. Its collisions cannot all be described as solver-failure braking. These remain local unicycle adaptations, not the holonomic main experiment or universal failure boundaries.

The manuscript now explicitly states the planning-existence negative finding: Cov-1 has 589 successes/7 collisions versus Bayes 586/10, with four timeouts each. The direction is not uniform by configuration. PaS sensor-code inheritance, its learned social-inference objective, and the narrower previously-detected-track scope here are explicitly distinguished.

**Decision:** advance the current system-and-mechanism manuscript toward submission without requiring Bayesian indispensability. Do not add a negative-evidence mixture to the formal method on the strength of this limited probe. The research possibility remains open; this round supplies no new navigation or shape-specific efficacy claim. A manuscript with reproducible evidence is not a guarantee of novelty acceptance or journal acceptance. No journal submission has been made; authors must approve venue, authorship, and final submission materials.

Additional reproducible commands:
```bash
python experiments/occlusion_binding.py --workers 6
python experiments/occlusion_report.py binding
python experiments/occlusion_report.py cells
python experiments/occlusion_shape_probe.py
python experiments/occlusion_report.py external
```
The final command reads the historical repaired-adapter raw records at their recorded location; `external_scope.json` additionally releases the compact per-episode sufficient counts with raw-source hashes. Diagnostic sources, protocols, and per-state results are in the repository. The main paper remains a 16-page manuscript with configuration-stratified controls as its principal comparison.
