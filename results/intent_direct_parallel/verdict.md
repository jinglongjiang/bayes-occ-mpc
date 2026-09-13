# Packed direct execution: real-input acceptance

## Decision

Numerical acceptance PASSED; stable 250 ms complete-pipeline acceptance FAILED.
Do not start the conditional 18-record full-sequence shadow replay. Neither the
production algorithm nor the previous navigation/scientific conclusions changed.
The packed implementation is retained as an optional experimental execution
backend, not a new Bayesian model or an approved real-time controller.

## Hardware and provenance

AMD Ryzen 7 5700G, 8 physical cores / 16 logical CPUs. Allowed CPUs: 0-15.
Use one logical CPU per distinct physical core: 15,14,13,12,11,10,9,8.
Root and user.slice CPU quotas were -1 (unlimited). Other desktop processes
were not killed and cores were not exclusively reserved; this is not a hard
real-time operating-system test. No other experiment queue ran concurrently.

BLAS/MKL remain single-threaded. OpenMP uses close binding, explicit places,
dynamic teams disabled, passive waiting; only independent risk outputs are
parallelized. Each row's mode/person addition order stays serial. The native
kernel records actual worker CPU IDs inside its parallel region, and every
recorded query team matched the requested distinct physical cores. Serial host
work and the original T baseline remain on CPU 15.

The legacy RVO runtime emitted a one-place initialization warning when first
loaded with the main thread pinned. It did not constrain the packed query team:
its independently recorded 1/2/4/8-worker CPU IDs and equality checks all passed.
No deployment hardware policy restricting the method to one core was assumed;
multicore measurements are explicitly extra-resource results, not single-core wins.

Input package SHA256:
`5ec34194c0c365bab18deb28b6d9f6f3feefe9dff2c500955bb04a2ef1444550`.
The supplied two runtime files were copied into `experiments/packed_direct_execution`.
Only CPU-placement instrumentation was added to those files; the query formula,
precision and sum order were not changed. Source hashes for the measured version
are in protocol.json. Original input SHA256s were:
- packed_direct.py: 50aaf1082912e2d840b4356264712e9cbd29b42a3e3709328038f610773425dd
- risk_execution.cpp: 3a7dd3ede71f32972d02d4d4f35a9e0340a213893c5972c49fa9b142a85acee8

## Complete-pipeline results

Same 12 development states, four per configuration, three balanced technical
repeats per variant. All figures below are milliseconds. Original T uses one
core. No modes, likelihood parameters, search parameters, geometry or tolerances
were changed. These percentiles describe a small development sample, not a
certified full-sequence latency distribution.

| Configuration | Original T p50 | Packed 1 p50 | Packed 2 p50 | Packed 4 p50 | Packed 8 p50 | Packed 8 p95 | Packed 8 maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 people | 155.40 | 141.65 | 104.78 | 72.68 | 58.74 | 90.24 | 93.50 |
| 10 people | 365.36 | 321.37 | 205.01 | 145.35 | 112.70 | 118.35 | 118.72 |
| 20 people | 968.93 | 830.42 | 476.25 | 300.91 | 228.99 | 265.05 | 267.61 |

At 8 threads, four of the twelve 20-person timing samples exceeded 250 ms:
case 2000020: 267.61, 262.94, 253.96 ms; case 2000022: 255.48 ms.
Across all configurations that is 4/36 samples, not four independent episodes.
The predeclared gate required all 36 measurements below 250 ms. No thread count
passed it. A below-budget median does not replace this stability requirement.

## Risk-query components (20 people)

| Variant | Table construction + packing p50 | Risk query p50 | Full pipeline p50 |
| --- | ---: | ---: | ---: |
| Original T / 1 core | 8.06 | 835.74 | 968.93 |
| Packed / 1 core | 9.16 | 701.76 | 830.42 |
| Packed / 2 cores | 8.68 | 357.30 | 476.25 |
| Packed / 4 cores | 8.65 | 182.99 | 300.91 |
| Packed / 8 cores | 8.99 | 106.34 | 228.99 |

The remaining median components at 8 threads include goal update 14.95 ms,
full rollout 33.69 ms and other MPC work 69.39 ms. Component medians do not add.
Risk parallelism is effective, but the complete computation is not just risk.

## Numerical and initialization checks

24,576 actual original-T candidate trajectories / 393,216 candidate-time
queries were checked for each of 1/2/4/8 threads. All packed hazard values were
bitwise equal to T, hence no risk-threshold differences relative to T. Ordered
elites and complete control sequences also matched in all 12 states. This
preserves the tested T approximation, not exact collision probability or physical
robot safety. It does not repair the old posterior-approximation precision gate.

Synthetic checks additionally covered zero/positive variance, heterogeneous
per-mode parameters (the non-common native path), split/permuted mode mass,
zero existence, empty crowds, closed contexts and nonfinite input rejection.
The additional heterogeneous/empty-context tests were appended after timing;
no timed evaluation code changed. Their results are in self_test.json.

The cached shared library was removed before the measured run. Fresh native
compile/load took 318.28 ms, recorded separately and not charged to steady-state
controls. The process initializes its thread teams in the numerical self-check;
small-query first-call times are recorded there separately. This is not a
measurement or guarantee of complete first-ever navigation startup latency.
Every timed state still includes observation/filter update, posterior update,
all-mode rollout, current tables and packing, thread dispatch, risk and MPC.
Prefix restoration, snapshot copies, compilation, assertions and serialization
are outside timing. No current-state packing is cached in the prefix.

## Reproduction and stopping boundary

`python experiments/intent_direct_parallel.py --self-test`

`python experiments/intent_direct_parallel.py`

The runner resumes completed states from states.json. Preserve that file as the
record of this run; a resumed command is not a new independent timing experiment.
`threads=0` in raw files denotes original T with one CPU, not zero workers.

The CPU instrumentation assumes serial calls per QueryContext. The runtime was
tested on Linux with GCC/OpenMP, not on other machines or accelerators.
CV and MAP were not run because the FULL budget gate did not pass. A future
navigation comparison must give all methods the same hardware ceiling; this
report makes no resource-mismatched method comparison. No epsilon changes,
new trees, new models, extra CPUs, or navigation episodes were introduced.
