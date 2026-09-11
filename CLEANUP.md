# Hermite Architecture Work Log

## Scope and Recovery

Worktree: `/home/abc/workspace/bayes_occ_mpc_hermite`.
Branch: `hermite-architecture-20260912`.
Pre-change checkpoint: `ae3fad8`, tag `hermite-pre-refactor-20260912`.
The original master worktree, its uncommitted changes, and historical results remain untouched.
Use a new worktree at the checkpoint to inspect or restore old code without destroying ongoing work.

`HERMITE_BASELINE.json` registers external replay hashes, dependency versions, and the separate historical result tag. The historical 97.17% result is not reassigned to this code. The audited configuration is loaded from `archive/hermite_audit/protocol.json`, including population 512, horizon 16, four iterations and the existing risk settings; class defaults are not used as experimental configuration.

## Tasks 1 and 2

- The new `nav` package imports only its data contracts, risk implementation, NumPy, SciPy and Python's standard library. It does not import the old gate, environment, experiment runner, ROS or modern-MPC bridge.
- `nav/planner.py` maintains one CEM loop. Exact evaluation uses `envelope=None`; production defaults to the unchanged Hermite envelope. The cached rectangle baseline is injected by the experiment, not imported by the core.
- Risk-to-cost and risk-to-clearance formulas now live in `_score_from_hazard` once. Exact hazard and both envelope endpoints use it. Arithmetic ordering, risk thresholds and tail conventions are unchanged.
- Elite ordering, iteration-winner selection and cross-iteration replacement retain their different rules, each implemented once. They have not been collapsed into one lexicographic sort.
- Every selected elite and iteration winner is checked against the exact-evaluation mask. Unrefined optimistic values are exclusion evidence, not eligible final scores.
- Invalid observations are rejected. Nonfinite or reversed envelope bounds on valid inputs fall back to the original exact risk computation; this is not a safety controller.
- Read-only tracing is disabled by default. Regression tracing additionally checks projected candidates, ordered elites and their scores, updated means/deviations, iteration winner and cross-iteration best.
- The original reference and Hermite code remain immutable references. They are not dependencies of the new runtime package.
- The rejected early-skip micro-optimization is not included.

## Explicitly Not Yet Changed

The belief module and the old environment adapter remain unchanged in this task. Source inspection confirms density recursion in `BayesianRFSBelief.update`, unused Bayes buffer calculations in `_prediction_for_track`, and experimental adapter branches. Their removal requires the separate raw-observation regression in task 3; posterior-only replays cannot validate that deletion.

Visibility evidence, track management, existence probability, covariance, route seeds, warm starts, geometry, cost terms, risk profile and fallback motion remain. Unicycle code and old modern-MPC scripts are preserved outside the new import chain, not deleted from history. Configuration fields and data contracts are retained for now rather than silently breaking old serialized inputs.

## Verification and Reproduction

Run from this worktree with `/home/abc/miniconda3/envs/crowdnav/bin/python`:

```bash
python experiments/replay.py
python experiments/replay.py --edges
python experiments/replay.py --full --timing
```

The first command uses three existing representative episodes (134 control steps). The full command uses the same 30 saved episodes (no new simulation). Each of five implementations maintains its own warm start: old exact, old Hermite, new exact, cached rectangle, new Hermite. All four are compared against the untouched old exact implementation; the old Hermite is not replaced by a simultaneously modified reference.

Results are centralized in `results/hermite_architecture/`, excluded from Git; source and configuration remain versioned. Timing runs separately from tracing/assertions and uses only the three new implementations with common code and alternating order. It is not directly compared against absolute timings from another run.

Initial gate: all 134 steps passed ordered-elite, distribution, winner, control and diagnostic equality. Full regression status is recorded in `complete.json` only after all cases finish; a gate pass is not a claim that full validation or all four task packages are complete.
