# Bayes-MPC: reproducible intent-repair runtime

This branch packages the runtime dependencies of the completed experiment at
`6ee89d7eeb80cdaa7ec73313bcd5d4d8bf5d853d`. It does not introduce a new algorithm,
change prior results, or rerun the 90-episode navigation evaluation.

- [Runtime sources, installation and small replay](reproducibility/README.md)
- [Versions, file hashes and path-change provenance](reproducibility/manifest.json)
- [Independent-directory and rebuilt-RVO test results](reproducibility/verification.json)
- [Small self-contained reference fixture](reproducibility/fixture.json)
- [Existing intent-repair result and limitations](results/intent_repair/verdict.md)

CrowdNav runtime sources are in `vendor/CrowdNav`; C++/Cython RVO build sources
are in `vendor/Python-RVO2`. Each retains its original license. Training code,
weights, virtual environments and compiled binaries are not packaged.

The small test reproduces three layouts and 36 recorded steps. Tested scope is
same-host path independence, including a rebuilt RVO extension, not cross-machine
installation. The original installed RVO binary's exact build provenance remains
unknown; the provided source rebuild passes the fixed reference, not a universal
binary-equivalence proof. See the manifest for that distinction.
