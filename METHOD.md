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
T_A=T_{common}+T_{all\ exact},\qquad
T_C=T_{common}+T_{table}+T_{bounds}+T_{screen}+T_{remaining\ exact}.
\]

The method is beneficial only when saved exact work exceeds table/query/screening costs. Geometry still traverses all candidates; no reduction in asymptotic population-by-human complexity is claimed. An 8% candidate-refinement fraction is not an 8% special-function count: the original exact evaluator already masks far components, and table construction invokes both CDF and scaled Bessel functions. Report those calls separately. The rejected early-mask modification is an example of extra indexing costs offsetting saved interpolation, and is not part of this algorithm.

## Closest Work and Remaining Novelty Boundary

| Work | Existing mechanism | Specific distinction examined here |
|---|---|---|
| Thomas et al. | Exact and bounded Gaussian collision probabilities for ellipsoidal geometry | Shared fixed-a disk queries coupled to elite refinement; not a claim to invent probability bounds |
| Elimelech and Indelman | Decision simplification, action consistency and belief sparsification | Preserve the predictive belief and partially resolve candidate risk ordering; full action-order consistency could already imply elite preservation |
| Zhitnikov and Indelman | Adaptive action-sequence acceptance/rejection under belief-dependent probabilistic constraints | Fixed predictive posterior and a finite CEM search, rather than future-observation-tree expansion |
| Hoerger et al., LCEOPT | Lazy policy-tree sampling, evaluation and cross-entropy updates | Envelope-based risk refinement of trajectory candidates, not omission of unvisited policy-tree components |

Primary sources checked: [Thomas et al.](https://arxiv.org/html/2110.06348v1), [Elimelech and Indelman](https://arxiv.org/html/1909.00885v5), [Zhitnikov and Indelman](https://arxiv.org/html/2302.06697v1), [LCEOPT](https://arxiv.org/html/2305.08049v2).

The A/B/C comparison tests the value of this concrete shared envelope against a cached rectangle bound under the same selector. It does not establish empirical dominance over these complete systems. Bounded probabilities, action consistency and lazy CEM are established ideas. The defensible candidate contribution is their specialized shared-risk evaluation mechanism with explicit elite/tolerance-pool preservation and measured cost tradeoffs; sufficient publication-level novelty remains a review question, not a consequence of refactoring or acceleration alone.
