"""Fair rectangle baseline: cache only candidate-independent constants."""
import three_arm as t
import copy
import dataclasses
import hashlib
import itertools
import json
import pickle
import time
import numpy as np
from scipy.special import ndtr

NAIVE_OUT = t.OUT
FAIR_OUT = t.HERE/"fair_three_arm_results"
NaiveRectangle = t.RectangleRisk


class CachedRectangle:
    def __init__(self, cfg, obs):
        self.cfg, self.obs = cfg, obs
        self.variance = .5*np.trace(obs.human_position_covariance,axis1=2,axis2=3)
        self.s = np.sqrt(np.maximum(self.variance,1e-9))
        self.radius = obs.robot_radius+obs.entities[:,4,None]+cfg.human_margin
        self.inner = self.radius/np.sqrt(2.)
        self.inner_factor = 2*ndtr(self.inner/self.s)-1
        self.outer_factor = 2*ndtr(self.radius/self.s)-1
        self.table_cdf_values = 0

    def bounds(self, positions):
        obs = self.obs
        d = np.linalg.norm(positions[:,None]-obs.human_segment_end[None],axis=3)
        s, R, a = self.s[None], self.radius[None], self.inner[None]
        lo = (ndtr((a-d)/s)-ndtr((-a-d)/s))*self.inner_factor[None]
        hi = (ndtr((R-d)/s)-ndtr((-R-d)/s))*self.outer_factor[None]
        lo = np.maximum(0.,lo-1e-12); hi = np.minimum(1.,hi+1e-12)
        far = d-R >= 8*s
        lo = np.where(far,ndtr(-8.),lo); hi = np.where(far,ndtr(-8.),hi)
        deterministic = self.variance[None]<1e-9
        lo = np.where(deterministic,d<=R,lo); hi = np.where(deterministic,d<=R,hi)
        existence = obs.human_existence
        if self.cfg.existence_override is not None:
            existence = np.full_like(existence,self.cfg.existence_override)
        assert np.all((existence>=0)&(existence<=1))
        def aggregate(q):
            return -np.log1p(-np.clip(q*existence[None,:,None],0.,1.-1e-12)).sum(axis=1)
        return aggregate(lo),aggregate(hi)


def rectangle_equivalence():
    from validate_prototype import observation
    rng = np.random.default_rng(843)
    obs = observation(64,20,"crossing")
    obs.human_position_covariance = (10**rng.uniform(-10,1,(20,16)))[...,None,None]*np.eye(2)
    positions = rng.uniform(-5,5,(625,16,2))
    a,b = NaiveRectangle(t.CFG,obs).bounds(positions),CachedRectangle(t.CFG,obs).bounds(positions)
    for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    t.save("rectangle_equivalence.json",dict(component_queries=200000,aggregate_arrays_bitwise_equal=True,
        sole_optimization="Cache per-observation sigma and the two transverse Gaussian masses; same interval and selector"))


def replay_check(n,case):
    original_path = NAIVE_OUT/f"replay_{n}_{case}.pkl"
    with original_path.open("rb") as f: inputs = pickle.load(f)
    metadata = json.loads((NAIVE_OUT/f"check/{n}_{case}.json").read_text())
    models = {a:t.planner(a,profiling=True) for a in t.ARMS}
    permutations = list(itertools.permutations(t.ARMS))
    frames = []
    for index,frame in enumerate(inputs):
        warm = {a:None if models[a]._previous_mean is None else models[a]._previous_mean.copy() for a in t.ARMS}
        outputs = {}
        for arm in permutations[(index+n+case)%6]:
            _,elapsed,profile = t.measured_plan(models[arm],arm,frame["obs"],frame["seed"])
            outputs[arm] = dict(profile=profile,profiled_ms=elapsed,
                exact_rows=models[arm].risk_rows,total_rows=models[arm].total_rows)
        try:
            for arm in t.ARMS[1:]: t.assert_pair(models["exact"],models[arm])
        except AssertionError:
            with (FAIR_OUT/f"mismatch_{n}_{case}_{index}.pkl").open("wb") as f:
                pickle.dump(dict(obs=frame["obs"],seed=frame["seed"],warm=warm,config=t.CFG),f)
            raise
        record = copy.deepcopy(metadata["frames"][index]); record["arms"] = outputs
        frames.append(record)
    metadata["frames"] = frames
    t.save(f"check/{n}_{case}.json",metadata)
    # Preserve the exact trusted replay, not a separately generated environment.
    (FAIR_OUT/f"replay_{n}_{case}.pkl").write_bytes(original_path.read_bytes())
    print(f"FAIR_CHECK_OK {n}_{case} steps={len(inputs)}",flush=True)


if __name__ == "__main__":
    t.OUT = FAIR_OUT
    t.RectangleRisk = CachedRectangle
    initial = [(5,23000),(5,23001),(20,23000),(20,23001)]
    tasks = initial+[(n,c) for n in (5,10,20) for c in range(23000,23010) if (n,c) not in initial]
    files = [t.Path(t.ref.__file__),t.Path(t.ref.__file__).with_name("bayesian_rfs.py"),
        t.Path(t.core.__file__),t.Path(t.__file__),t.Path(__file__)]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    t.save("protocol.json",dict(config=dataclasses.asdict(t.CFG),tasks=tasks,assets=hashes,
        development_only=True,independent_test=False,timing_threads=1,timing_repeats=1,
        rectangle_baseline="cached constants; bounds bitwise equal to supplied rectangle helper",
        supersedes="Uncached rectangle run stopped before completion; retain as implementation diagnosis only"))
    start = time.monotonic()
    rectangle_equivalence()
    t.numeric()
    for group in (initial,tasks[len(initial):]):
        for n,c in group:
            if (NAIVE_OUT/f"replay_{n}_{c}.pkl").exists() and (NAIVE_OUT/f"check/{n}_{c}.json").exists():
                replay_check(n,c)
            else:
                t.collect(n,c)
        for n,c in group: t.timing(n,c)
        if group is initial: print("FAIR_FOUR_EPISODE_GATE_PASSED",flush=True)
    t.summarize(tasks)
    assert all(hashlib.sha256(t.Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    t.save("complete.json",dict(wall_seconds=time.monotonic()-start,sources_unchanged=True,
                                tasks=len(tasks),status="COMPLETE"))
    print("FAIR_ALL_COMPLETE",flush=True)
