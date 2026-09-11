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

## Task 3: Belief and Directory Cleanup

`nav/belief.py` now contains only the audited Bayes track recursion. Removed from this new path: density shape/rate state and updates; Gamma-Poisson convolution; zero occupancy-grid allocation; covariance eigenvalue/chi-square buffer computation and output; fixed/deterministic mode branches; tracker risk/buffer/density configuration. The original implementation is preserved in `archive/legacy/bayesian_rfs.py`.

`integration/crowdnav.py` constructs only Bayes observations. It retains the legal policy observation/visibility interface and keyed detection corruption. It has no GT rollout or sensor/fixed/conformal/Poisson dispatch. The observation-hash serialization is intentionally absent from the online path; it is audit metadata, not an input to decisions.

Six legal-observation fixtures (96 updates) separately verified track-ID order, complete means/covariances, existence, missed age, visibility, timestamps, prediction arrays and adapter fields. Both clean and nonzero measurement noise were covered, including deletion and reappearance. End-of-sequence old/new composed controls also match. These are deterministic input fixtures, not real-world perception validation or new navigation episodes.

Visibility evidence, track management, existence probability, covariance, route seeds, warm starts, geometry, cost terms, risk profile and fallback motion remain. Forty-eight old root Python/shell files, including unicycle and modern-MPC scripts, were moved unchanged to `archive/legacy`; they are not deleted from history. Only the new core and integration path are supported by this refactor's verification. Archived launch scripts are historical references, not newly revalidated launchers after relocation. Planner configuration fields and input data contracts remain compatible with old serialized observations.

## Verification and Reproduction

Run from this worktree with `/home/abc/miniconda3/envs/crowdnav/bin/python`:

```bash
python experiments/replay.py
python experiments/replay.py --edges
python experiments/replay.py --belief
python experiments/replay.py --full --timing
python experiments/replay.py --report
python experiments/replay.py --counts
```

The first command uses three existing representative episodes (134 control steps). The full command uses the same 30 saved episodes (no new simulation). Each of five implementations maintains its own warm start: old exact, old Hermite, new exact, cached rectangle, new Hermite. All four are compared against the untouched old exact implementation; the old Hermite is not replaced by a simultaneously modified reference.

Results are centralized in `results/hermite_architecture/`. Compact completion/summary/edge/belief/counter reports are committed; large per-state records and external replay inputs remain outside Git with registered provenance. Gate records have a separate subdirectory and cannot overwrite full timing records. Timing runs separately from tracing/assertions and uses only the three new implementations with common code and alternating order. It is not directly compared against absolute timings from another run.

Full regression: 30 episodes, 1358 control states passed all four old/new comparisons: 86912 ordered route-elite comparisons and 21728 iteration-state comparisons. The tested core hashes are in `complete.json` and remained unchanged after directory relocation. The relocated rectangle implementation has an identical AST to the audited cached rectangle class. Edge tests also verify invalid-input rejection and exact fallback for a failed envelope.

Same-run median exact/rectangle/Hermite milliseconds: 5-person circle 44.87/45.34/43.60; 10-person circle 80.40/81.63/72.82; 20-person square 132.22/135.17/115.79. Hermite is faster on every episode mean against both controls, but not every control step. Single-pass development timing is not an independent confirmation or proof of publication novelty.

## Using the New Runtime

From this worktree, construct `MPCConfig` using the registered protocol values, not unexamined defaults:

```python
from nav.contracts import MPCConfig
from nav.planner import MPCPlanner
from integration.crowdnav import BayesObservationAdapter

cfg = MPCConfig(**registered_config)
adapter = BayesObservationAdapter(cfg)
planner = MPCPlanner(cfg)  # Hermite is the production default.
obs = adapter.read(env)
action, plan_ms = planner.plan(obs, seed)
```

The environment continues to execute the declared holonomic velocity command. No unicycle/GPU adaptation, new success-rate experiment or external action correction is introduced.

## Task 4: Manuscript and Method

`METHOD.md` contains the shared-variable definition, real-arithmetic bound, floating-error caveats, refinement/selection argument, cost accounting and primary-source distinctions. The sole manuscript remains `/home/abc/workspace/nav_data/mamba/camrl/mamba_log/latex_FCS.txt`; it is reorganized around the computation result and keeps historical navigation evidence explicitly separate. No PDF or additional TeX/bibliography files are generated.

The previous manuscript is stored directly as a Git blob under tag `manuscript-pre-hermite-20260912`, so preserving it does not require another manuscript file. The final manuscript blob is tagged separately after checks. TeX is checked structurally only, not compiled. Universal floating-point certification, independent held-out computation confirmation and venue-level novelty acceptance remain unproved.
