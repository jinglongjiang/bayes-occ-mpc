# Temporal First-Hit Gate

Verdict: STOP_TEMPORAL

The registered gate requires at least six qualifying states from four episodes.
Five states from five episodes qualify. This is a resource stop, not proof that
temporal dependence has no decision value. No threshold, state selection or grid
was changed to obtain a sixth example. No environment continuations or training run.

O is the original controller; I assumes independent time events; C uses adjacent
event intersections; J uses joint trajectory sampling. 60 states from 30 existing
episodes, 1440 unique within-state candidates, 4096 selection samples and a disjoint
65536-sample verification stream on selected/critical candidates. The clean-model
P0 reconstruction uniquely matches every stored future covariance, with zero maximum
residual. This is not a reconstruction method for arbitrary noisy observations.

| epsilon | contrast | executable changes >=0.05 m/s | independently supported states | episodes | no feasible |
|---|---|---:|---:|---:|---:|
| 0.25 | I-J | 5 | 5 | 5 | 41 |
| 0.25 | C-J | 0 | 0 | 0 | 35 |
| 0.25 | O-J | 4 | 1 | 1 | 35 |
| 0.1 | I-J | 3 | 2 | 2 | 46 |
| 0.1 | C-J | 0 | 0 | 0 | 41 |
| 0.1 | O-J | 4 | 0 | 0 | 41 |
| 0.5 | I-J | 8 | 5 | 4 | 34 |
| 0.5 | C-J | 1 | 0 | 0 | 19 |
| 0.5 | O-J | 21 | 2 | 2 | 19 |

No neural training. No new navigation episodes. C timings are real quadrature costs on the small pool, not a full-workload real-time claim.
I-J isolates temporal dependence; O-J also changes risk objective and interface. J is a finite-sample model reference, not environment truth.
Protocol and all candidate risks, controls, constraints, convergence and paired MC intervals are in this directory.

## What The Five Positive Examples Mean

All five meet criterion B: more progress under the common model-risk ceiling, not
reduced risk. Their J risk is actually higher than I's; the independent upper bounds
remain below 0.25. All five select exactly the same candidate under C and J.

| Configuration / case / step | I model risk | J model risk | First velocity change (m/s) | Terminal distance improvement (m) |
|---|---:|---:|---:|---:|
| 5 circle / 23003 / 26 | 0.03223 | 0.16358 | 0.6167 | 0.4494 |
| 10 circle / 23002 / 24 | 0.10229 | 0.15686 | 0.3037 | 1.0602 |
| 10 circle / 23004 / 32 | 0.07678 | 0.11016 | 0.4731 | 0.9519 |
| 20 square / 23002 / 29 | 0.09284 | 0.21864 | 0.1949 | 0.1799 |
| 20 square / 23005 / 35 | 0.10286 | 0.17571 | 0.4753 | 1.2263 |

Relative to O, only one state satisfies the full improvement rule. Changing a
newly introduced independent-time approximation is not the same as beating the
existing controller. No real collision or arrival improvement has been measured.

## Constraint And Sampling Diagnosis

- The common G retains 869/1440 candidates. At epsilon 0.25, J has a risk-admissible
  candidate in 25/60 states; all 25 still have a candidate after G. Adding full-horizon
  mean geometry to the first-step constraints changes none of the J winners. Thus
  this gate's main bottleneck is not an otherwise useful J choice blocked by geometry.
- I has feasible candidates in 19 states, C in 26, and selection-stream J in 25.
  Six J states restore feasibility relative to I. They are recorded separately from
  paired feasible I-J action gains. Comparing J against O on those six states also
  yields no additional qualifying example; this is not a hidden sixth pass.
- One boundary state (5 / 23003 / 13) is sampling-sensitive: candidate 18 has
  J-select 0.25427, J-eval 0.24821, and C 0.24976. The new stream changes a no-feasible
  decision within the verification pool. We do not retrospectively select it into
  the gate or describe J as an exact oracle.
- Selection-stream I-J order inversions with at least 0.01 J risk separation occur
  in 50/60 states (499 candidate pairs). These are descriptive noisy rankings, not
  50 supported executable improvements.

## Classical Accuracy And Cost

On 244 common-G candidates in the independent verification pools, C is within
0.01 absolute probability of J in 231 cases (94.67%). Median error is 0.001107,
95th percentile 0.01003, maximum 0.06754. The sample-based ideal adjacent bound's
excess over J has median 0.000286, p95 0.008215 and maximum 0.06946 on the same
candidate indices in the selection stream. Nonadjacent re-entry matters in some
candidates, but it has not produced a material C-J action gain at the main threshold.

All 60 states pass the registered node-doubling convergence check. This is not a
certified upper-bound implementation: the largest pair Frechet-bound violation is
8.57e-5. Numerical integration and finite reference errors remain disclosed.

| Configuration | Last retained quadrature median (24 candidates) | All convergence levels median |
|---|---:|---:|
| 5 circle | 1.342 s | 1.973 s |
| 10 circle | 7.434 s | 11.032 s |
| 20 square | 10.390 s | 15.660 s |

These are actual conditional-Gaussian integrals, not intersections reused for free
from J. They exclude common planning/filter costs. The current implementation is
costly even on the small pool; no full 512-by-4 workload or pipeline-p99 eligibility
claim is made. This does not prove every possible classical implementation is slow,
nor does cost alone authorize a neural model after the decision gate failed.

## Decision

Keep the existing navigator and previous Hermite results unchanged. Do not train the
first-hit network or start branch rollouts in this task. Temporal information has
real but limited observed model-decision effects, and the classical adjacent method
already recovers all five qualifying choices. Neither real-world navigation benefit
nor a need for learned higher-order temporal memory has been established.

The adjacent bound is the existing chain-tree specialization, not a new theorem:
[Patil and Tanaka, Eq. (9)](https://arxiv.org/html/2110.15879v1).
