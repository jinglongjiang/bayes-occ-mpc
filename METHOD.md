# Shared Collision Envelopes and Elite-Preserving CEM

## Scope

The contribution under examination is computational: retain a specified finite-budget holonomic CEM controller's decisions while reducing repeated risk evaluation. The Bernoulli-Gaussian tracker, route seeds, costs, actuator limits, risk profile, goal mask and fallback motion are inherited components, not separate novelty claims. Exact hazard remains necessary inside the accelerated implementation. This method does not require a newly trained network or a new latent-variable model.

The audit configuration is population 512, horizon 16, four iterations, elite fraction 0.08, dt 0.25, and the parameters recorded in `archive/hermite_audit/protocol.json`. The historical navigation results and this calculation experiment are different evidence cohorts. Do not assign the historical 97.17% success rate to an acceleration implementation tested only on replays.

## Shared Quantity

For a fixed planning call and isotropic position covariance, let

\[
a_{i,k}=R_i/\sigma_{i,k},\qquad
z_{j,i,k}=(d_{j,i,k}-R_i)/\sigma_{i,k}.
\]

The conditional Gaussian-disc mass is

\[
q_{j,i,k}=f_{a_{i,k}}(z_{j,i,k})
=F_{\chi'^2_2((a_{i,k}+z_{j,i,k})^2)}(a_{i,k}^2).
\]

Only z changes across robot candidates. Tables for each distinct a are shared over candidates and iterations within this call. They are rebuilt at the next observation, not extrapolated across time. No posterior compression is involved. Deterministic components and the reference's deliberate positive 8-sigma tail replacement retain their registered numerical conventions.

## Error Envelope

For a standard normal Z and translated disk A, differentiation of Gaussian measure gives a fourth derivative of the form E[1_A He_4(Z_x)]. Since E[He_4]=0, this equals E[(1_A-1/2)He_4]. Therefore

\[
|f_a^{(4)}|\leq \tfrac12 E|He_4|\leq\tfrac12\sqrt{E[He_4^2]}
=\sqrt{24}/2.
\]

The cubic Hermite remainder on an interval of width h is bounded by M_4 h^4/384. Thus the real-arithmetic interpolation error is at most sqrt(24) h^4/768 (about 1.56e-6 for h=0.125).

If endpoint values, endpoint derivatives and polynomial arithmetic have certified errors e0, e1 and er, a sufficient interpolation budget is

\[
\delta=\sqrt{24}h^4/768+e_0+(h/4)e_1+e_r.
\]

The endpoint value basis has absolute sum one, and the derivative basis absolute sum is at most 1/4. Monotone endpoint clipping must itself use valid endpoint enclosures. Aggregation rounding is a separate error. SciPy CDF/Bessel calls and the prototype's 1e-12 padding have no established universal directed-rounding guarantee. Accordingly the implementation is numerically checked, not formally interval-certified. The real disk-probability theorem must also be distinguished from preservation of the reference's deliberately substituted far tail.

## From Probability to Ordering

For nonnegative existence weights r_i in [0,1], clipped component probabilities are aggregated as

\[
h_{j,k}=-\sum_i\log(1-\operatorname{clip}(r_iq_{j,i,k},0,1-10^{-12})).
\]

This map is nondecreasing in every component. With nonnegative risk weight, the two hazard endpoints give cost bounds and reversed clearance bounds. Active-until-goal masking and the existing hard penalty retain their exact evaluation order. The single `_score_from_hazard` implements these equations for both exact and bounded evaluation.

Elite keys are (feasibility category, violation, cost), ordered lexicographically. Within each route, let tau be the kth smallest pessimistic key. A candidate is discarded from elite consideration only if its optimistic key is strictly worse than tau. At least k candidates have true keys no worse than tau; hence the discarded candidate cannot enter the exact top k. All ties survive. Remaining candidates are evaluated with the original exact hazard, and the original deterministic sorting rule decides their order.

This argument does not require recovering the order of every discarded candidate. It does require valid bounds and correct propagation through category changes and penalties.

## Final Choice Is Not the First Elite

The controller separately chooses: a lowest-cost full-feasible candidate; otherwise a first-step-feasible candidate within 0.02 of the largest full-horizon clearance; otherwise the lowest-cost candidate within 0.02 of the largest first-step physical clearance. The possible winners of these pools are also refined, independently of elite membership. Cross-iteration replacement retains the original clearance tolerance and cost order. The runtime rejects selection of an unrefined elite or winner.

Conditional preservation follows by induction: the same initial state, warm start and random numbers generate the same candidates; exact ordered elites yield the same floating-point mean/std updates in the same order; the next candidate batch is therefore unchanged. Independently preserved iteration choices and cross-iteration replacement yield the same final control sequence. This is a property of the specified finite search, not global optimality, true collision safety, or equality in asynchronous deployments with different observation arrival times.

## Algorithm

1. Validate input and build per-observation envelopes (or choose all-exact evaluation).
2. Generate candidates with the inherited route populations and correlated noise.
3. Compute common trajectory, geometry and non-risk cost terms.
4. Obtain hazard bounds and use the shared score mapping for optimistic/pessimistic keys.
5. Refine possible route elites and possible tolerance-pool winners using exact hazard.
6. Apply the unchanged elite, iteration-choice and cross-iteration replacement rules.
7. Update the route distributions in the same order and repeat; execute the best evaluated sequence's first control.

Nonfinite/reversed envelope bounds on valid inputs trigger all-exact evaluation. Invalid inputs or nonfinite exact risk are errors, not zero-risk candidates.

## Cost Accounting

\[
T_E=T_{common}+T_{all\ exact},\qquad
T_H=T_{common}+T_{table}+T_{bounds}+T_{screen}+T_{remaining\ exact}.
\]

The method is beneficial only when saved exact work exceeds table/query/screening costs. Geometry still traverses all candidates; no reduction in asymptotic population-by-human complexity is claimed. An 8% candidate-refinement fraction is not an 8% special-function count: the original exact evaluator already masks far components, and table construction invokes both CDF and scaled Bessel functions. Report those calls separately. The rejected early-mask modification is an example of extra indexing costs offsetting saved interpolation, and is not part of this algorithm.

## Closest Work and Remaining Novelty Boundary

| Prior construction (body checked) | Concrete specialization here | Proof obligation | Experiment |
|---|---|---|---|
| Thomas et al., IV-B, Theorem 2: the noncentral-chi-square special case explicitly permits table lookup. IV-C (18)--(21) assumes a finite upper support bound beta. | Reuse exactly equal a=R/sigma curves across candidates; attach two-sided interpolation envelopes. This is not the first CDF lookup. Do not transplant a bounded-support inequality to the untruncated Gaussian event. | Gaussian-measure derivative bound, Hermite remainder, separate node/derivative/arithmetic errors. | Direct point interpolation T versus envelope/refinement H, using identical tables. |
| Zhitnikov and Indelman, III-B (22)--(31): propagate lower/upper function values through constraint indicators and sampled constraints; tighten when acceptance/rejection is unresolved. | Supply a concrete shared disk-probability enclosure and propagate it through this navigator's hazard, hard penalty and lexicographic keys. This instantiates a general principle, not a capability denied to that framework. | Monotonicity of clipped negative-log survival and nonnegative risk costs; reversal for clearance bounds. | Same-selector cached rectangles R versus H; unresolved-but-unused refinements and category ambiguity. |
| Elimelech and Indelman, 2.1.2--2.2: action consistency and loss/offset relations formalize decision-preserving simplification. | Resolve enough ordering for each route's ordered top-k and the distinct clearance-tolerance winner, without reconstructing all candidate ranks or sparsifying the posterior. | Strict elimination, retained ties, exact ordered updates and separate winner preservation. Full-order consistency already implies top-k consistency. | Fixed-candidate selections, then independently evolving search and warm starts. |
| Hoerger et al., LCEOPT, 4.4: lazy sampling/evaluation and updates exploit policy-tree components needed by simulated paths. | Avoid trajectory-risk refinements rather than policy-tree components. No new general lazy-CEM claim. | Saved exact work must exceed table, query and selector overhead; common geometric work remains. | Full-call timing, special-function counts and retained slowdown cases. |

Primary sources checked: [Thomas et al.](https://arxiv.org/html/2110.06348v1), [Elimelech and Indelman](https://arxiv.org/html/1909.00885v3), [Zhitnikov and Indelman](https://arxiv.org/html/2302.06697v1), [LCEOPT](https://arxiv.org/html/2305.08049v2).

The focused E/R/T/H comparison tests this concrete construction, not empirical dominance over these complete systems. E is reference risk, R is the cached rectangle selector, T directly scores the Hermite point value, and H retains envelopes plus refinement. H-minus-T measures the cost of protecting selection; H need not beat T in latency. If natural replay inputs show no T disagreement, report that rather than coarsening the grid. A zero-disagreement sample is not a universal guarantee.

The defensible candidate contribution is this specialized shared-risk evaluation mechanism with explicit elite/tolerance-pool preservation and measured cost tradeoffs. Its publication-level novelty remains a review question, not a consequence of a positive speedup alone. The completed population-only experiment (a8c3d90) selected 640 for both arms and found no independent navigation improvement; it remains a separate evidence cohort.

## Focused Evidence (2026-09-12)

One mechanism pass on 30 existing episodes / 1358 states, followed by three balanced timing passes (16296 calls). No new navigation episodes. Raw generation commit: `78be5e1`; results: `results/hermite_focused/`.

| Configuration | E p50 ms | R p50 ms | T p50 ms | H p50 ms | Episode-paired H minus T, ms (95% interval) |
|---|---:|---:|---:|---:|---|
| 5 circle | 42.62 | 43.35 | 34.11 | 39.99 | +5.36 [5.18, 5.53] |
| 10 circle | 81.32 | 81.33 | 66.49 | 78.24 | +11.06 [10.69, 11.48] |
| 20 square | 140.84 | 140.34 | 104.11 | 125.15 | +18.21 [17.51, 18.95] |

- R/H preserve fixed-batch selections and independently evolving search on all states. There are no recorded raw hazard enclosure violations and no incorrectly excluded required candidates; this remains empirical evidence.
- T changes no feasibility categories, eight fixed ordered elite lists and seven iteration winners. Only one final output differs, by at most `1.1102230246251565e-16`. Its maximum conditional probability error is about `4.364e-7`. No material navigation benefit of protection over lookup is demonstrated.
- In 20 square, R refines 476925/1060864 candidate rows and H 85140/1060864. Ultimately unused refinements are 384973 versus 130. H still pays table and query costs, including 1860309 table CDF and equally many Bessel evaluations over 518 calls; its candidate CDF count is 15379103 versus E's 192378140.
- H beats E and R on every episode mean, but loses to T on every episode mean. All slower individual calls are retained in `summary.json`. Confidence intervals resample paired episodes, not steps or technical repetitions.
- A reporting correction excludes 12 no-risk batches (6144 candidate rows) from the interval-refinement partition. Their all-resolved masks are not CDF work. Raw records, measured times and source hashes are preserved; only reporting functions changed.

Decision: retain the computational result and its explicit protection cost. Do not claim that the natural data demonstrate practically necessary protection, improved navigation, or sufficient publication novelty. The sole manuscript includes the weaker T-discrepancy evidence and the unchanged negative budget pilot, rather than presenting only H-versus-E gains.
