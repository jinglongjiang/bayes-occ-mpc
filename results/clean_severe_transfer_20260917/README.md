# Frozen clean-to-severe transfer

Read [the Chinese report](报告.txt) for results and limitations. This is a
shared-tracker MPC risk-representation test, not IL/PPO training or a short-history
policy comparison. No new research direction passed this gate.

- 100 paired layouts (5100--5199), 20-human dense square.
- Five methods, two conditions: 1000 evaluation episodes; one separate smoke episode.
- Clean operating points, calibration tables and filter measurement assumptions frozen.
- Actual severe measurements: position std .1 m, velocity std .2 m/s, detection .8.
- All filters still assume clean measurement noise and detection .98.
- Clean/severe success: Bayes 96/61, EWMA 97/67, age 75/73,
  conformal 91/75, static covariance 99/70 (out of 100).

## Evidence

`clean/` and `severe/` contain per-episode outcomes and original protocol manifests.
`summary.json` includes paired change bootstrap intervals. `contract_checks.json`
records sensor/filter separation and environment access checks.

`source/` contains exact hashed run-time source files and frozen inputs, not edits
to the repository's production planner. `source_mapping.json` maps original
absolute paths in manifests to portable archived files. Source hashes are preserved.
`source/merge.py` is the original local analysis entry; its paths and report-writing
behavior remain unchanged. To verify this archive without the original machine:

```bash
python results/clean_severe_transfer_20260917/verify.py
```

This verifies source hashes, episode coverage and recomputes the published summary
and bootstrap intervals. It does not rerun the simulator.

## Simulation rerun

Requires the original compatible CrowdNav/RVO2 environment. CrowdNav base commit:
`775ae6b3116a5eb84061e08f6baac27bbbadb10b`; the actual locally modified
`crowd_sim.py` is archived. A fresh dependency installation was not tested.
Set `SRC` to this bundle's `source/`, `CROWDNAV_ROOT` to that environment and
`FREEZE_SENSOR=1`. Invoke `source/run_formal.py` with:

```text
bayes:3,ewma:3,age_margin:3,conformal:3,static_cov:3
7
source/frozen_clean.json
source/calibration_clean.json
5100
100
NEW_OUTPUT_DIRECTORY
```

Run once with `TEST_NOISE='[0,0,1]'`, once with
`TEST_NOISE='[0.1,0.2,0.8]'`, into separate new directories. Do not overwrite
the archived results. The suffix `:3` selects the scene, not the operating point.
