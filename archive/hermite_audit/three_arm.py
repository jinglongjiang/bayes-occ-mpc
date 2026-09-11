"""Three-arm development replay, isolated from the navigation implementation."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, str(HERE/"elite_preserving_mpc"))
import contextlib
import copy
import dataclasses
import hashlib
import itertools
import json
import pickle
import time
import numpy as np
import continuous_mpc_gate as ref
import rank_certified_cem as core
from audit_local import CFG, assert_pair

OUT = HERE/"three_arm_results"
ARMS = ("exact", "rectangle", "hermite")
HERMITE = core.CompiledDiscRisk


def save(name, data):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class RectangleRisk:
    def __init__(self, cfg, obs):
        self.cfg, self.obs = cfg, obs
        self.variance = .5*np.trace(obs.human_position_covariance, axis1=2, axis2=3)
        self.radius = obs.robot_radius+obs.entities[:, 4, None]+cfg.human_margin
        self.table_cdf_values = 0

    def bounds(self, positions):
        obs = self.obs
        distance = np.linalg.norm(positions[:, None]-obs.human_segment_end[None], axis=3)
        lower, upper = core.disc_probability_bounds(distance, self.radius[None],
                                                     self.variance[None])
        existence = obs.human_existence
        if self.cfg.existence_override is not None:
            existence = np.full_like(existence, self.cfg.existence_override)
        assert np.all((existence >= 0) & (existence <= 1))
        def aggregate(prob):
            return -np.log1p(-np.clip(prob*existence[None, :, None], 0., 1.-1e-12)).sum(axis=1)
        return aggregate(lower), aggregate(upper)


class Profiler:
    """Exclusive wall time, so nested exact-risk evaluations are not double counted."""
    def __init__(self):
        self.times = {}
        self.stack = []
    def wrap(self, name, fn):
        def call(*args, **kwargs):
            frame = [time.perf_counter(), 0.]
            self.stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter()-frame[0]
                self.stack.pop()
                self.times[name] = self.times.get(name, 0.)+(elapsed-frame[1])*1000
                if self.stack:
                    self.stack[-1][1] += elapsed
        return call


def planner(arm, profiling=False, tracing=True):
    p = core.ReferenceTraceCEM(CFG) if arm == "exact" else core.RankCertifiedCEM(CFG)
    if not tracing:
        p._elite_indices = ref.ContinuousCEMMPC._elite_indices
    p.audit_profiler = Profiler() if profiling else None
    if profiling:
        names = {"_rollout": "trajectory_geometry", "_human_clearance": "trajectory_geometry",
                 "_cost": "base_cost", "_combined_clearance": "trajectory_geometry",
                 "_physical_clearance": "trajectory_geometry",
                 "_belief_collision_hazard": "exact_risk", "_elite_indices": "screen_update",
                 "_seed_trajectories": "screen_update", "_route_seeds": "screen_update",
                 "_initial_mean": "screen_update"}
        if arm != "exact": names["_lazy_evaluate"] = "screen_update"
        for method, category in names.items():
            setattr(p, method, p.audit_profiler.wrap(category, getattr(p, method)))
    return p


@contextlib.contextmanager
def evaluator(arm, p):
    # Only the isolated module's query factory changes, never the common selector.
    original = core.CompiledDiscRisk
    cls = RectangleRisk if arm == "rectangle" else HERMITE
    def factory(cfg, obs):
        prof = p.audit_profiler
        obj = prof.wrap("bound_build", cls)(cfg, obs) if prof else cls(cfg, obs)
        if prof:
            obj.bounds = prof.wrap("bound_query", obj.bounds)
        return obj
    core.CompiledDiscRisk = factory
    try:
        yield
    finally:
        core.CompiledDiscRisk = original


def measured_plan(p, arm, obs, seed):
    if p.audit_profiler:
        p.audit_profiler.times = {}
    with evaluator(arm, p):
        start = time.perf_counter()
        action, _ = p.plan(obs, seed)
        elapsed = (time.perf_counter()-start)*1000
    if p.audit_profiler:
        profile = dict(p.audit_profiler.times)
        profile["search_other"] = elapsed-sum(profile.values())
        assert profile["search_other"] >= -1e-6
    else:
        profile = {}
    return action, elapsed, profile


def numeric():
    from validate_prototype import observation
    rng = np.random.default_rng(31223)
    obs = observation(933, 20, "crossing")
    obs.human_segment_end = rng.uniform(-3, 3, (20, 16, 2))
    variances = 10**rng.uniform(-10, 1, (20, 16))
    obs.human_position_covariance = variances[..., None, None]*np.eye(2)
    positions = rng.uniform(-4, 4, (625, 16, 2))
    from scipy.stats import ncx2
    from scipy.special import ndtr
    r = RectangleRisk(CFG, obs)
    d = np.linalg.norm(positions[:, None]-obs.human_segment_end[None], axis=3)
    lo, hi = core.disc_probability_bounds(d, r.radius[None], variances[None])
    near = (d-r.radius[None]) < 8*np.sqrt(variances)[None]
    q = np.full(d.shape, ndtr(-8.))
    q[near] = ncx2.cdf(np.broadcast_to(r.radius[None]**2/np.maximum(variances[None], 1e-9), d.shape)[near],
                       2, (d*d/np.maximum(variances[None], 1e-9))[near])
    q = np.where(variances[None] < 1e-9, d <= r.radius[None], q)
    assert np.all(lo <= q+2e-11) and np.all(hi >= q-2e-11)
    for family in ("circle", "crossing", "horizon_blocked", "first_infeasible"):
        obs = observation(2931, 20, family)
        models = {a: planner(a) for a in ARMS}
        for seed in (735, 736):
            for arm in ARMS: measured_plan(models[arm], arm, obs, seed)
            for arm in ARMS[1:]: assert_pair(models["exact"], models[arm])
    save("numeric.json", dict(rectangle_queries=int(q.size), violations=0,
                              synthetic_families=4, plans_per_arm=8))


def collect(n, case):
    tag = f"{n}_{case}"
    frames, diagnostics, observations, last_seen = [], [], [], {}
    models = {a: planner(a, profiling=True) for a in ARMS}
    permutations = list(itertools.permutations(ARMS))
    class Collector(ref.ContinuousCEMMPC):
        def plan(self, obs, seed):
            number = len(frames)
            snapshot = copy.deepcopy(obs)
            warm = {a: None if models[a]._previous_mean is None else
                    models[a]._previous_mean.copy() for a in ARMS}
            outputs = {}
            for arm in permutations[(number+case+n)%6]:
                action, elapsed, profile = measured_plan(models[arm], arm, obs, seed)
                outputs[arm] = dict(profile=profile, profiled_ms=elapsed,
                    exact_rows=models[arm].risk_rows, total_rows=models[arm].total_rows)
            try:
                for arm in ARMS[1:]: assert_pair(models["exact"], models[arm])
            except AssertionError:
                OUT.mkdir(parents=True, exist_ok=True)
                with (OUT/f"mismatch_{tag}_{number}.pkl").open("wb") as f:
                    pickle.dump(dict(obs=snapshot, seed=seed, warm=warm, config=CFG), f)
                raise
            self.last_controls = models["exact"].last_controls.copy()
            self.last_diagnostics = models["exact"].last_diagnostics.copy()
            self._previous_mean = self.last_controls.copy()
            self.last_diag = self.last_diagnostics
            observations.append(dict(obs=snapshot, seed=seed))
            frames.append(dict(step=number, class_id=self.last_diagnostics["feasibility_class"],
                hidden_count=int(np.sum(~obs.human_visible)) if obs.human_visible is not None else 0,
                hidden_variance_max=float(np.max(.5*np.trace(obs.human_position_covariance, axis1=2, axis2=3)))
                    if obs.human_position_covariance is not None and len(obs.entities) else 0.,
                elite_pairs_per_comparison=len(models["exact"].elite_trace), arms=outputs))
            return self.last_controls[0], outputs["exact"]["profiled_ms"]
    def observer(env, obs, adapter, step):
        # Belief internals are logged for analysis only and never fed into an arm.
        belief = getattr(adapter, "belief", None)
        if belief is None: belief = getattr(adapter, "rfs", None)
        tracks = getattr(belief, "tracks", {})
        items = list(tracks.values()) if isinstance(tracks, dict) else list(tracks)
        diagnostics.append(dict(step=step, track_fields=sorted(vars(items[0])) if items else []))
    scenario = "square_crossing" if n == 20 else "circle_crossing"
    result = ref.run_episode(ref.DEFAULT_CROWDNAV, "bayes", n, scenario, case, CFG,
        4. if n != 20 else None, 10. if n == 20 else None, time_limit=25,
        planner_type=Collector, step_observer=observer, occlusion=True, robot_kinematics="holonomic")
    save(f"check/{tag}.json", dict(humans=n, case=case, frames=frames,
                                  episode=dataclasses.asdict(result), auxiliary=diagnostics))
    with (OUT/f"replay_{tag}.pkl").open("wb") as f:
        pickle.dump(observations, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"CHECK_OK {tag} steps={len(frames)}", flush=True)


def timing(n, case):
    with (OUT/f"replay_{n}_{case}.pkl").open("rb") as f:
        frames = pickle.load(f)
    models = {a: planner(a, tracing=False) for a in ARMS}
    permutations = list(itertools.permutations(ARMS))
    times = []
    # One whole replay per arm, retaining each arm's own warm start; no assertions,
    # diagnostic output or serialization inside the timed calls.
    for index, frame in enumerate(frames):
        row = {}
        for arm in permutations[(index+n+case)%6]:
            _, elapsed, _ = measured_plan(models[arm], arm, frame["obs"], frame["seed"])
            row[arm] = elapsed
        times.append(row)
    save(f"timing/{n}_{case}.json", dict(humans=n, case=case, frames=times))
    print(f"TIMING_OK {n}_{case} "+json.dumps({a: round(float(np.median([r[a] for r in times])), 2)
                                            for a in ARMS}), flush=True)


def summarize(tasks):
    summary = []
    rng = np.random.default_rng(430)
    for n in (5, 10, 20):
        check = [json.loads((OUT/f"check/{h}_{c}.json").read_text()) for h,c in tasks if h == n]
        timing_rows = [json.loads((OUT/f"timing/{h}_{c}.json").read_text()) for h,c in tasks if h == n]
        frames = [f for r in timing_rows for f in r["frames"]]
        for arm in ARMS:
            durations = np.array([f[arm] for f in frames])
            group = [f["arms"][arm] for r in check for f in r["frames"]]
            profile = {key: float(np.mean([g["profile"].get(key, 0.) for g in group]))
                       for key in sorted({k for g in group for k in g["profile"]})}
            summary.append(dict(humans=n, arm=arm, episodes=len(check), steps=len(frames),
                p50_ms=float(np.median(durations)), p95_ms=float(np.quantile(durations,.95)),
                mean_ms=float(np.mean(durations)), profile_mean_ms=profile,
                exact_row_fraction=float(sum(g["exact_rows"] for g in group)/sum(g["total_rows"] for g in group))))
        for a,b in (("exact","rectangle"),("exact","hermite"),("rectangle","hermite")):
            # Bootstrap episodes, never independent-looking control steps.
            differences = np.array([np.mean([f[b]-f[a] for f in r["frames"]]) for r in timing_rows])
            estimates = np.mean(differences[rng.integers(0,len(differences),(10000,len(differences)))], axis=1)
            summary.append(dict(humans=n, comparison=f"{b}_minus_{a}",
                episode_mean_difference_ms=float(np.mean(differences)),
                exploratory_episode_bootstrap_ci95=np.quantile(estimates,[.025,.975]).tolist(),
                slower_cases=[r["case"] for r,d in zip(timing_rows,differences) if d>0],
                slower_step_fraction=float(np.mean([f[b]>f[a] for f in frames]))))
    checks = [json.loads((OUT/f"check/{n}_{c}.json").read_text()) for n,c in tasks]
    save("summary.json", dict(results=summary, episodes=len(tasks),
        steps=sum(len(r["frames"]) for r in checks),
        elite_pairs=sum(2*f["elite_pairs_per_comparison"] for r in checks for f in r["frames"]),
        all_checked_equal=True, timing_separate=True))


if __name__ == "__main__":
    files = [Path(ref.__file__), Path(ref.__file__).with_name("bayesian_rfs.py"),
             Path(core.__file__), Path(__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    initial = [(5,23000),(5,23001),(20,23000),(20,23001)]
    tasks = initial+[(n,c) for n in (5,10,20) for c in range(23000,23010) if (n,c) not in initial]
    save("protocol.json", dict(config=dataclasses.asdict(CFG), tasks=tasks,
        assets=hashes, development_only=True, independent_test=False,
        timing_threads=1, timing_repeats=1,
        order="four episode gate, then remaining development layouts, all timing serial",
        note="No navigation SR advantage tested; no runtime optimization before profiling results"))
    start = time.monotonic()
    numeric()
    for n,c in initial: collect(n,c)
    for n,c in initial: timing(n,c)
    print("FOUR_EPISODE_GATE_PASSED", flush=True)
    for n,c in tasks[len(initial):]: collect(n,c)
    for n,c in tasks[len(initial):]: timing(n,c)
    summarize(tasks)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    save("complete.json", dict(wall_seconds=time.monotonic()-start, sources_unchanged=True,
                               tasks=len(tasks), status="COMPLETE"))
    print("ALL_COMPLETE", flush=True)
