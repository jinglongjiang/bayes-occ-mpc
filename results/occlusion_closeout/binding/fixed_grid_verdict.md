# Fixed-grid spatial evidence diagnostic

Base: f592484. No production controller, parameter, manuscript or historical result changed.

Reproduction:

```bash
OCCLUSION_FIXED_GRID=1 OCCLUSION_ALL_CANDIDATES=1 \
  /home/abc/miniconda3/envs/crowdnav/bin/python experiments/occlusion_shape_probe.py
```

Same 12 previously fixed snapshots, six configurations sharing case 3001017,
steps 8 and 24. These are not 12 independent layouts. Uniform standardized
position grids use 25, 49 and 97 cells per axis on [-6,6]^2, with Gaussian
cell masses and midpoint collision integration. Prior truncation is below
4e-9; this is not a universal posterior or floating-point error certificate.
Projection retains the full position covariance and conditional velocity
relationship. Existence and original geometric checks are held fixed to
isolate spatial conditioning. This is a one-frame intervention, not a
recursive negative-evidence tracker.

| Test | 24-candidate subset | All 512 candidates |
| --- | ---: | ---: |
| States | 12 | 12 |
| States with eligible evidence | 4 | 4 |
| Numerically stable states | 12 | 12 |
| Spatial vs original first-action changes | 0 | 1 |
| Spatial vs Gaussian projection changes | 0 | 0 |

All-candidate evaluation was added after the subset returned zero changes,
to check candidate coverage. It is exploratory, not independent confirmation.
The maximum fine-grid risk change across both new arms is 0.001383, below
the previously fixed 0.005 diagnostic tolerance, with unchanged selections.

At dense_square, step 24, the old selection is candidate 146, command
(-0.0448904683, 0.9989919148) m/s. Both updated arms select candidate 456,
command (-0.3834630539, 0.7810269483) m/s. Norm difference is 0.4026663 m/s.
Full-horizon feasible candidates increase from 13 to 15. All three grid
resolutions agree. The old subset omitted candidate 456.

Conclusion: one-frame negative evidence can alter a feasible action in this
pool. No extra action value from non-Gaussian shape is demonstrated. The
predeclared shape-specific continuation criterion is not met. Do not build
adaptive discretization on these results. No independent CEM evolution,
branch continuation, or new navigation episode was run; improved safety or
progress has not been established.
