"""Elite-preserving lazy Bayesian risk evaluation for the uploaded CEM planner.

This is an isolated research prototype, not a replacement for a frozen experiment.
Place it beside continuous_mpc_gate.py (the user's uploaded reference module).
It changes risk evaluation, not tracking, costs, limits, samples, or execution.

The real-arithmetic argument uses shared cubic-Hermite Gaussian-disc tables,
a fourth-derivative error bound, and elite-set membership certificates. IEEE/SciPy calculations here are NOT interval arithmetic; a small numeric
padding is used, and numerical equivalence is tested separately. No real-world
safety guarantee, novel-inference claim, or closed-loop result is implied.
"""
from __future__ import annotations
from typing import Tuple
import math
import time
import numpy as np
from scipy.special import ndtr, ive
from scipy.stats import ncx2
from continuous_mpc_gate import ContinuousCEMMPC, MPCConfig, PlannerObservation


def disc_probability_bounds(distance, radius, variance):
    """Bounds for P(||N((d,0), s^2 I)|| <= R), including reference conventions.

    An inscribed square of half-width R/sqrt(2) is a subset of the disc;
    the circumscribed square of half-width R is a superset. Alignment with
    the mean makes the two Gaussian coordinates independent. Monotonicity
    under set inclusion supplies the bounds; no fitted surrogate is used.
    """
    d, R, v = np.broadcast_arrays(np.asarray(distance, float),
                                   np.asarray(radius, float),
                                   np.asarray(variance, float))
    if np.any(d < 0) or np.any(R < 0) or np.any(v < -1e-12):
        raise ValueError('distance, radius and variance must be nonnegative')
    s = np.sqrt(np.maximum(v, 1e-9))
    def rectangle(a, b):
        return (ndtr((a - d) / s) - ndtr((-a - d) / s)) * (2*ndtr(b/s) - 1)
    a = R / math.sqrt(2.0)
    lo = rectangle(a, a)
    hi = rectangle(R, R)
    # Numeric padding is not a rigorous IEEE directed-rounding certificate.
    lo = np.maximum(0., lo - 1e-12)
    hi = np.minimum(1., hi + 1e-12)
    # Match the reference's deliberate 8-sigma replacement, not exact tails.
    far = (d-R) >= 8.0*s
    lo = np.where(far, ndtr(-8.), lo)
    hi = np.where(far, ndtr(-8.), hi)
    deterministic = v < 1e-9
    lo = np.where(deterministic, d <= R, lo)
    hi = np.where(deterministic, d <= R, hi)
    return lo, hi


class CompiledDiscRisk:
    """Per-observation Gaussian-disc table with analytic interpolation brackets.

    q(z)=P(||N((R+sigma*z,0),sigma^2 I)||<=R).
    The fourth derivative is bounded by sqrt(24)/2, from the zero mean
    and L2 norm of the fourth probabilists' Hermite polynomial. Cubic-Hermite
    interpolation error <= sqrt(24)*h^4/768. Floating table values and
    derivatives mean this implementation is NOT interval arithmetic.
    """
    def __init__(self,cfg,obs,step=.125):
        self.cfg,self.obs,self.step=cfg,obs,float(step)
        if not np.isclose(16./step,round(16./step)):
            raise ValueError('table step must divide 16')
        self.grid=np.linspace(-8.,8.,round(16./step)+1)
        self.variance=.5*np.trace(obs.human_position_covariance,axis1=2,axis2=3)
        self.s=np.sqrt(np.maximum(self.variance,1e-9))
        self.radius=obs.robot_radius+obs.entities[:,4,None]+cfg.human_margin
        self.a=np.broadcast_to(self.radius,self.variance.shape)/self.s
        unique_a, inverse = np.unique(self.a, return_inverse=True)
        self.component_map = inverse.reshape(self.variance.shape)
        centers=unique_a[:,None]+self.grid
        self.table=ncx2.cdf(unique_a[:,None]**2,2.,centers**2)
        self.derivative = (-unique_a[:,None] * np.sign(centers)
            * np.exp(-.5*(np.abs(centers)-unique_a[:,None])**2)
            * ive(1,unique_a[:,None]*np.abs(centers)))
        self.table_cdf_values=int(self.table.size)
        # d^4/dz^4 Gaussian set mass = E[1_A He_4(Z)]. Since E He_4=0,
        # its absolute value <= .5 E|He_4| <= sqrt(4!)/2.
        # Cubic Hermite error <= M4*h^4/384.
        self.interpolation_error=np.sqrt(24.)/2.*step**4/384.

    def conditional_bounds(self,positions):
        cfg,obs=self.cfg,self.obs
        d=np.linalg.norm(positions[:,None,:,:]-obs.human_segment_end[None,:,:,:],axis=3)
        z=(d-self.radius[None,:,:])/self.s[None,:,:]
        index=(z+8.)/self.step
        left=np.clip(np.floor(index).astype(np.int64),0,len(self.grid)-2)
        frac=np.clip(index-left,0.,1.)
        component=self.component_map
        table=self.table
        qleft=table[component[None,:,:],left]
        qright=table[component[None,:,:],left+1]
        derivative=self.derivative
        gleft=derivative[component[None,:,:],left]
        gright=derivative[component[None,:,:],left+1]
        t=frac; t2=t*t; t3=t2*t
        approx=((2*t3-3*t2+1)*qleft + (t3-2*t2+t)*self.step*gleft
                +(-2*t3+3*t2)*qright + (t3-t2)*self.step*gright)
        lower=approx-self.interpolation_error
        upper=approx+self.interpolation_error
        positive_left=self.a[None,:,:]+self.grid[left]>=0.
        lower=np.where(positive_left,np.maximum(lower,qright),lower)
        upper=np.where(positive_left,np.minimum(upper,qleft),upper)
        lower=np.maximum(0.,lower-1e-12)
        upper=np.minimum(1.,upper+1e-12)
        # A radius-gap of 8sigma contains a centered 8sigma ball.
        lower=np.where(z < -8.,1.-np.exp(-32.)-1e-12,lower)
        upper=np.where(z < -8.,1.,upper)
        # Reproduce reference's far-tail numerical convention exactly.
        far=z>=8.
        lower=np.where(far,ndtr(-8.),lower)
        upper=np.where(far,ndtr(-8.),upper)
        deterministic=self.variance[None,:,:]<1e-9
        lower=np.where(deterministic,d<=self.radius,lower)
        upper=np.where(deterministic,d<=self.radius,upper)
        return lower,upper

    def bounds(self,positions):
        cfg,obs=self.cfg,self.obs
        lower,upper=self.conditional_bounds(positions)
        existence=obs.human_existence
        if cfg.existence_override is not None:
            existence=np.full_like(existence,cfg.existence_override)
        if np.any(existence<0) or np.any(existence>1):
            raise ValueError('existence must be between zero and one')
        def aggregate(p):
            return -np.log1p(-np.clip(p*existence[None,:,None],0.,1.-1e-12)).sum(axis=1)
        return aggregate(lower),aggregate(upper)


def _keys(cost, full, first):
    feasible, first_feasible = full >= 0, first >= 0
    category = np.where(feasible, 0, np.where(first_feasible, 1, 2))
    violation = np.where(feasible, 0., np.where(first_feasible, -full, -first))
    return category, violation, cost


def _le_key(keys, cut):
    category, violation, cost = keys
    c, v, j = cut
    # Keep ALL exact ties, so sorting/tie-breaking remains the reference's job.
    return ((category < c) | ((category == c) & ((violation < v) |
            ((violation == v) & (cost <= j)))))


class ReferenceTraceCEM(ContinuousCEMMPC):
    """Reference behavior with read-only elite/candidate diagnostics."""
    def __init__(self, config):
        super().__init__(config)
        self.elite_trace = []
        self.risk_rows = 0
        self.total_rows = 0
    def _elite_indices(self, costs, full, first, count):
        ids = ContinuousCEMMPC._elite_indices(costs, full, first, count)
        self.elite_trace.append(ids.copy())
        return ids
    def _belief_collision_hazard(self, controls, obs, positions=None):
        self.risk_rows += len(controls)
        return super()._belief_collision_hazard(controls, obs, positions)
    def plan(self, obs, seed):
        self.elite_trace = []
        self.risk_rows = 0
        self.total_rows = self.cfg.population*self.cfg.iterations
        return super().plan(obs, seed)


class RankCertifiedCEM(ReferenceTraceCEM):
    """Drop-in plan(obs, seed), preserving reference elites and final choices.

    It certifies top-k membership in EVERY route at EVERY CEM iteration,
    not just the best trajectory of one fixed candidate batch. It explicitly
    covers the reference's 0.02-tolerance feasibility/degraded selection rules.
    Unknown/occupancy costs, when present, remain exactly evaluated.
    """
    def __init__(self, config):
        if config.probability_weight < 0:
            raise ValueError('nonnegative risk cost is required for monotone bounds')
        super().__init__(config)
        self.refinement_batches = 0
        self.table_cdf_values = 0

    def _lazy_evaluate(self, controls, obs, positions, clearance, occupancy, counts, bounds):
        cfg = self.cfg
        # Every non-probabilistic quantity is evaluated as in the reference.
        base_cost = self._cost(controls, obs, positions, clearance, occupancy, None)
        geo_full, geo_first = self._combined_clearance(
            controls, obs, positions, clearance, occupancy, None)
        _, first_physical = self._physical_clearance(controls, obs, positions, clearance)
        hlo, hhi = self._compiled.bounds(positions) if self._compiled is not None else (None,None)
        if hlo is None:
            return base_cost, geo_full, geo_first, first_physical
        active, _, _ = self._active_until_goal(positions, obs)
        limits = self._belief_hazard_limits()[None,:]
        n = len(controls)
        exact = np.zeros(n, bool)

        def evaluate(h):
            cost = base_cost.copy()
            cost += cfg.probability_weight*(h*active).sum(axis=1)
            cost += 1e5*(np.where(active,h-limits,-math.inf).max(axis=1) > 0.)
            slack = np.where(active, limits-h, math.inf)
            full = np.minimum(geo_full, slack.min(axis=1))
            first = np.minimum(geo_first, slack[:,0])
            return cost, full, first

        def refine(mask):
            ids = np.flatnonzero(mask & ~exact)
            if not len(ids):
                return
            self.refinement_batches += 1
            h = self._belief_collision_hazard(controls[ids], obs, positions[ids])
            hlo[ids] = h
            hhi[ids] = h
            exact[ids] = True

        clo, flo, slo = evaluate(hlo)  # optimistic cost/clearance
        chi, fhi, shi = evaluate(hhi)  # pessimistic cost/clearance
        lower_keys = _keys(clo, flo, slo)
        upper_keys = _keys(chi, fhi, shi)
        keep = np.zeros(n, bool)
        for route, count in enumerate(counts):
            begin, end = bounds[route:route+2]
            k = max(4, round(count*cfg.elite_fraction))
            ck,vk,jk = (a[begin:end] for a in upper_keys)
            kth = np.lexsort((jk,vk,ck))[k-1] + begin
            cut = tuple(a[kth] for a in upper_keys)
            keep[begin:end] |= _le_key(tuple(a[begin:end] for a in lower_keys), cut)

        # Also retain the possible iteration winner, independently of elites.
        # This matters because final selection and elite selection differ when
        # the reference uses a 0.02 clearance tolerance.
        possible_full = flo >= 0.
        definite_full = fhi >= 0.
        if definite_full.any():
            ceiling = chi[definite_full].min()
            keep |= possible_full & (clo <= ceiling)
        else:
            keep |= possible_full
        refine(keep)
        cost, full, first = evaluate(hlo)
        if not (full >= 0.).any():
            # All potentially full-feasible candidates have been resolved.
            # Preserve the class-1 maximum-clearance and tolerance pool.
            chi, fhi, shi = evaluate(hhi)
            possible_first = first >= 0.
            definite_first = shi >= 0.
            if possible_first.any():
                best_lower = fhi[definite_first].max() if definite_first.any() else -math.inf
                refine(possible_first & (full >= best_lower - .02))
                cost, full, first = evaluate(hlo)
            if not (first >= 0.).any():
                safest = float(first_physical.max())
                physical_pool = first_physical >= safest - .02
                upper_cost, _, _ = evaluate(hhi)
                ceiling = upper_cost[physical_pool].min()
                refine(physical_pool & (cost <= ceiling))
                cost, full, first = evaluate(hlo)
        # Unrefined rows retain optimistic bounds. The certificates establish
        # that these rows cannot be selected as elites or as iteration winner.
        return cost, full, first, first_physical

    def plan(self, obs: PlannerObservation, seed: int) -> Tuple[np.ndarray,float]:
        self.elite_trace = []
        self.risk_rows = 0
        self.refinement_batches = 0
        cfg = self.cfg
        self.total_rows = cfg.population*cfg.iterations
        rng = np.random.default_rng(seed)
        mean = self._initial_mean(obs)
        start = time.perf_counter()
        self._compiled = (CompiledDiscRisk(cfg,obs) if obs.entities.size and
             obs.human_position_covariance is not None and obs.human_existence is not None else None)
        self.table_cdf_values = self._compiled.table_cdf_values if self._compiled is not None else 0
        seeds = self._seed_trajectories(obs)
        route_seeds = self._route_seeds(obs)
        means = np.concatenate((mean[None,:,:], route_seeds), axis=0)
        deviations = np.full_like(means, cfg.init_std)
        n_routes = int(route_seeds.shape[0])
        if n_routes:
            remaining = cfg.population - cfg.population//2
            counts = [cfg.population//2] + [remaining//n_routes]*n_routes
            for i in range(remaining % n_routes): counts[i+1] += 1
        else:
            counts = [cfg.population]
        bounds = np.cumsum([0]+counts)
        mode_ids = np.repeat(np.arange(len(counts)), counts)
        best_controls, best_cost, best_class = seeds[0].copy(), math.inf, 3
        best_first_clearance = best_full_clearance = best_first_physical = -math.inf
        for _ in range(cfg.iterations):
            noise = rng.standard_normal((cfg.population,cfg.horizon,2))
            noise[:,1:] = .68*noise[:,:-1] + .32*noise[:,1:]
            samples = means[mode_ids]+deviations[mode_ids]*noise
            samples[:len(seeds)] = seeds
            samples[len(seeds)] = mean
            for route in range(1,len(counts)):
                samples[bounds[route]] = means[route]
                samples[bounds[route]+1] = route_seeds[route-1]
            params, controls, positions = self._rollout(samples,obs)
            human_clearance = self._human_clearance(controls,obs) if obs.entities.size else None
            occupancy = obs.occupancy_probability.sample(positions) if obs.occupancy_probability is not None else None
            costs, full, first, first_physical = self._lazy_evaluate(
                controls,obs,positions,human_clearance,occupancy,counts,bounds)
            full_feasible, first_feasible = full >= 0., first >= 0.
            if full_feasible.any():
                pool = np.flatnonzero(full_feasible)
                ibest = int(pool[np.argmin(costs[pool])]); iclass = 0
            elif first_feasible.any():
                pool = np.flatnonzero(first_feasible)
                safest = float(full[pool].max())
                near = pool[full[pool] >= safest-.02]
                ibest = int(near[np.argmin(costs[near])]); iclass = 1
            else:
                ibest = self._degraded_choice(costs,first_physical,params,obs); iclass = 2
            candidate_cost, candidate_first = float(costs[ibest]), float(first[ibest])
            replace = iclass < best_class
            if iclass == best_class == 2:
                physical = float(first_physical[ibest])
                replace = physical > best_first_physical+.02 or (physical >= best_first_physical-.02 and candidate_cost < best_cost)
            elif iclass == best_class == 1:
                replace = float(full[ibest]) > best_full_clearance+.02 or (float(full[ibest]) >= best_full_clearance-.02 and candidate_cost < best_cost)
            elif iclass == best_class:
                replace = candidate_cost < best_cost
            if replace:
                best_class, best_cost = iclass, candidate_cost
                best_first_clearance, best_full_clearance = candidate_first, float(full[ibest])
                best_first_physical = float(first_physical[ibest])
                best_controls = params[ibest].copy()
            for route,count in enumerate(counts):
                begin,end = bounds[route:route+2]
                ids = self._elite_indices(costs[begin:end], full[begin:end], first[begin:end],
                                          max(4,round(count*cfg.elite_fraction)))+begin
                elite = params[ids]
                means[route] = .22*means[route]+.78*elite.mean(axis=0)
                deviations[route] = np.maximum(cfg.min_std,elite.std(axis=0))
        action = best_controls[0]
        self._previous_mean = best_controls
        self.last_controls = best_controls.copy()
        self.last_diagnostics = dict(feasibility_class=float(best_class),
            first_clearance=best_first_clearance,horizon_clearance=best_full_clearance,
            first_physical_clearance=best_first_physical,cost=best_cost,
            exact_risk_rows=float(self.risk_rows), total_candidate_rows=float(self.total_rows),
            refinement_batches=float(self.refinement_batches),
            table_cdf_values=float(self.table_cdf_values))
        return action,(time.perf_counter()-start)*1000.
