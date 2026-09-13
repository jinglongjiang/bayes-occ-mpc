# Runtime source reproduction

Baseline: `6ee89d7eeb80cdaa7ec73313bcd5d4d8bf5d853d`, the completed intent-repair
experiment. This package changes paths, import validation and provenance only.
It does not alter models, parameters, risk, geometry, recorded outcomes or papers.

## Included sources

- `vendor/CrowdNav`: byte-identical local `crowd_sim`, `crowd_nav/contracts.py`,
  package initializers, `crowd_nav/configs/env.config` and original license.
  This is the actual modified environment, not stock upstream CrowdNav.
- `vendor/Python-RVO2`: buildable C++/Cython sources and license from the local
  source tree at upstream commit `c2c46ba8d59556aa10faf03479293236efea154d`.
- `manifest.json`: file hashes, observed package versions, source origins and
  explicitly registered path-only modifications.
- `originals`: pre-portability source snapshots, including the earlier model
  protocol version. These are audit inputs, not alternate runtime implementations.
- `fixture.json`: three fixed layouts, 36 recorded actions/legal observation frames,
  effective environment configurations, MPC config, D predictions and mixture-risk
  references captured before the path edits using the original installed binary.
  No external download or original-machine data path is needed for this small test.

Original protocol JSON files remain unchanged. `runtime.verify_frozen` maps their
old path prefixes to this checkout, checks unchanged bytes directly, and checks
both the old snapshot and the registered new hash for path-edited files. It never
accepts arbitrary changed code by ignoring a mismatch. The earlier model snapshot
is also verified against its original protocol hash without requiring Git history.

## Dependencies and build

The observed interpreter is Python 3.8.10 on Linux x86-64. See `requirements.txt`
and the manifest for exact versions. PyTorch is imported by the unchanged contracts
module even though no network is trained or used by this navigation method.
Gymnasium is the active environment API; Gym is also imported by a fallback path.

In your chosen Python environment, install the listed dependencies. Older SciPy
may require a source build (C/C++/Fortran compilers and BLAS/LAPACK); wheel availability
is platform-dependent. Do not substitute newer versions and claim they were tested.
The recorded Torch build is `2.1.0+cu121`; the portable requirement specifies 2.1.0,
but other CPU/CUDA builds have not been accepted as cross-environment equivalents.

```bash
python -m pip install -r reproducibility/requirements.txt
cd vendor/Python-RVO2
python setup.py build_ext --inplace
python -m pip install --no-build-isolation --no-deps .
```

RVO requires CMake, a C++ compiler and Python development headers. Build artifacts,
virtual environments and binary extensions are deliberately not committed.

Important provenance limit: the original installed `pyrvo2==0.0.0` has no
`direct_url.json`; its binary hash differs from the located local build artifact.
We cannot prove its exact original source/build provenance. The vendored source
is a transparent rebuild candidate, checked against the original reference; a
matching small replay does not establish equivalence for every possible simulation.

## Independent-directory acceptance

Clone this branch to another directory and invoke the test from an unrelated working
directory. Clear inherited Python path overrides. No old CrowdNav source directory
is consulted, and the test checks loaded `crowd_sim`/`crowd_nav` file locations.

```bash
env -u PYTHONPATH -u PYTHONHOME python /path/to/checkout/reproducibility/smoke.py \
  --output /tmp/runtime-check.json
```

For the full checkout, also verify the existing experiment's data-loading and frozen
dependency entry points without running any experiment stages:

```bash
env -u PYTHONPATH -u PYTHONHOME python /path/to/checkout/reproducibility/smoke.py \
  --check-entry --output /tmp/runtime-entry-check.json
```

To test the freshly built extension without changing the installed environment:

```bash
env -u PYTHONPATH -u PYTHONHOME python -c '
import runpy, sys
sys.path.insert(0, "/path/to/checkout/vendor/Python-RVO2")
sys.argv = ["smoke.py", "--check-entry", "--output", "/tmp/runtime-rebuilt-check.json"]
runpy.run_path("/path/to/checkout/reproducibility/smoke.py", run_name="__main__")'
```

Acceptance is fixed: identical layout hashes/configuration/detections; bitwise
reference controls; position, D and risk absolute tolerance `1e-12`, relative
tolerance zero. Reports include actual maximum differences and imported module paths.
Do not use `--capture` to erase a failure: it was used once before portability edits
to establish the committed reference. It is not part of acceptance.

The existing `experiments/intent_repair_run.py` commands remain available. No full
queue was rerun for this packaging task. Existing stages must not be relabeled as
fresh experiments. Historical experiment tools unrelated to this branch (for example
the old budget-calibration collector's `/home/abc/temp/formal` inputs and machine-wide
unused-layout scan) remain archival; this package does not authorize new data collection
or claim those old collectors are portable. Registered intent layouts already exist.

## Scope of the evidence

See `verification.json` for actual checks. Same-host, separate-directory success is
path-independent reproduction only. Rebuilding RVO on the same host is additional
compiler/source validation, not a clean-machine or cross-environment installation test.
This package does not change the previous conclusions about MAP, global compression,
hidden-track CV fallback, navigation performance or real-time failure.
