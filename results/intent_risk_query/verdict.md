# Full-mode risk query experiment

## Scope and fixed choices

Base: c711be6. Same 12 registered development states, posterior, D rollout,
256-particle cap, conditional covariance, geometry, risk limits and CEM workload.
No new navigation episodes. Three rotated technical timing repeats per state.
The complete candidate batches from the reference's four search iterations are
used for risk/threshold audit. Each evaluator also runs its own CEM iterations
from a common canonical warm start at each recorded state. This is not a
multi-step closed-loop or a sequence with independently accumulated warm starts.

Reference: c711be6 fused Hermite bounds with necessary exact evaluation.
T: direct C++ Hermite point queries over every mode, with no elite protection.
Hierarchy: per-person, per-time spatial tree over all actually expanded D modes;
query upper bounds feed the same full-evaluation scoring path as T. The method
does not return group centers to the geometric obstacle interface.

## Bound construction

For a group of mass W, center c and radius a covering all its mode means, let
d=|x-c| and q(d) denote the common isotropic Gaussian disk probability. Then
W q(d+a) <= sum_m w_m q(d_m) <= W q(max(0,d-a)).
Different people, times, radii or conditional variances cannot be merged.
Node radii enclose their members, and all leaf/group weights are retained.
Common-condition equality is checked at construction. No posterior is compressed.

Each endpoint uses the original Hermite spacing 0.125 and an interpolation error
budget sqrt(24)/768*h^4 plus 1e-12 numerical padding. Tail approximations use
the original +/-8 standard-deviation domain. The padding covers those analytic
tail conventions, but is NOT a certified bound on SciPy and floating arithmetic.
Claims of containment are conditional mathematical properties plus tested
numerical validation, not a universal implementation safety certificate.

Before testing, the per-step union-probability width budget was fixed at 1e-4.
For n people, accept a node when its conditional width <=1e-4/n, otherwise split.
Node contributions are multiplied by their mass. Summing accepted nodes gives
at most 1e-4/n per-person width, provided leaf interpolation error fits that budget.
Existence is multiplied once per person. For independent-person aggregation
P=1-product_i(1-r_i q_i), each partial derivative is at most 1, so the union
width is no larger than the sum of per-person widths. No temporal independence
or first-hit risk change is introduced. The planner uses the upper hazard;
the cost is conservatively perturbed too, not merely the feasibility test.

T has no such explicit interval decision protection. It is not required to
match the reference's internal elites. Zero observed differences do not prove
that T is universally action-consistent or safe.

## Closest mechanism and novelty limit

Park and Manocha, Sections IV-D/IV-E, equation (22) and Algorithms 2/3 already
use hierarchical weighted-sample or mixture collision evaluation, including
early termination using node mass. Our specialization bounds a smooth common-
variance disk kernel from distance intervals, rather than replacing a potentially
colliding node by its entire mass. The budget is propagated through per-person
existence and cross-person union aggregation. This is a concrete implementation
distinction, not proof of a new general hierarchy principle or strong novelty.
No claim of outperforming their complete planner or their published implementation
is supported by this experiment.

Source: https://arxiv.org/html/1902.10252v4 (IV-D, IV-E, Algorithms 2 and 3).

Gray/Moore's error-controlled kernel summation is an even closer computational
template: this prototype is a single-query-tree specialization for a radial
disk-mass kernel. Changing the kernel and appending a monotone union aggregation
does not by itself establish a new approximation principle. Their author page
describes general kernels and hard error bounds:
https://www.cs.cmu.edu/~agray/nbody.html . A competitive novelty claim would need
more than this specialization; this experiment only compares our local backends.

## Reproduction and interpretation

Run `python experiments/intent_risk_query.py`. Completed states are resumed from
`states.json`; the initial one-state smoke check is separate and is not included
in the final statistics. Compile time and restored-prefix setup are excluded;
current observation, filter, inference, all-mode rollout, table/tree building,
risk queries and full planning are included. Historical contexts start empty
for each measured update. `remaining_exact` includes exact CDF preparation and
aggregation, not only special-function CPU time. All component medians are
descriptive and do not add to the median total.

This test does not measure navigation success, FULL-versus-MAP value, or full
closed-loop p99. No latency or prediction result here upgrades those old claims.

## Results and decision

All 12 states completed. Totals below are milliseconds, three technical repeats
per state; four states per configuration. These are descriptive development
measurements, not independent episode confidence intervals.

| Configuration | Reference p50 | T p50 | Hierarchy p50 | T p95 | Hierarchy p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 people | 338.00 | 152.05 | 152.53 | 312.07 | 240.20 |
| 10 people | 654.26 | 361.38 | 349.79 | 447.53 | 680.28 |
| 20 people | 1691.50 | 951.25 | 1359.42 | 1038.83 | 1575.51 |

For 20 people, reference table building took 7.66 ms, bound queries 801.50 ms,
and remaining exact evaluation 759.56 ms (component medians). CDF-related
remaining evaluation is important but is not the entire risk bottleneck.
T query time was 837.90 ms. Hierarchy tree building was 4.80 ms but its traversal
and endpoint queries took 1224.06 ms. Building the tree is not the main problem.
In all four 20-person states, hierarchy was slower than T. Across all states,
hierarchy beat T in five and lost in seven. No adaptive switch was added.

On 24,576 common candidate trajectories / 393,216 candidate-time queries:
- T maximum absolute union-probability error: 5.82e-7; zero raw risk-threshold
  classification changes in the checked queries.
- Hierarchy maximum error: 3.82e-5; maximum interval width: 5.37e-5, below 1e-4.
  No interval containment violations or false-safe threshold classifications;
  five more-conservative candidate-time classifications. These are raw per-time
  risk tests, including candidates/steps later rejected or masked by the planner.
- In all 12 full-search checks, ordered elites and complete controls matched the
  reference. This is an observed result, not a requirement or universal guarantee.

Decision: T is a useful faster engineering alternative, but still fails the
250 ms dense-scene budget. The proposed hierarchy has not demonstrated enough
independent computational value to replace T for the dense target configuration.
The research hypothesis was tested, not confirmed. Do not tune epsilon after
these results, start a navigation queue, or claim a new real-time navigation win.
Neither experimental evaluator is made the production default.

`python experiments/intent_risk_query.py --self-test` additionally checks
zero/positive variance, split mass, mode permutation, bound containment and NaN
rejection. Invalid interpolated values must raise, not become low risk through
floating clipping. The final timed run includes these native exception guards;
the earlier pre-guard run is archived separately and excluded from this table.
The self-test command was added after timing; it changes no measured evaluation
or planner code. Frozen timed source hashes remain in protocol.json.
