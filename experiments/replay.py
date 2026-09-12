"""Frozen-input regression for the shared CEM loop; no environment rollouts."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
import ast
import copy
import hashlib
import inspect
import itertools
import json
import pickle
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path('/home/abc/temp/elite_audit_20260911')
DATA = AUDIT/'fair_three_arm_results'
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT/'archive/legacy'))
sys.path.insert(2, str(ROOT/'archive/hermite_audit'))
import continuous_mpc_gate as original
import rank_certified_cem as old
from experiments.controls import CachedRectangle
from nav.planner import MPCPlanner
from nav.contracts import MPCConfig
from nav.risk import CompiledDiscRisk

OUT = ROOT/'results/hermite_architecture'
CONFIG = json.loads((ROOT/'archive/hermite_audit/protocol.json').read_text())['config']
CFG = MPCConfig(**CONFIG)
ARMS = ('old_exact', 'old_hermite', 'exact', 'rectangle', 'hermite')


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def capture(local):
    winner = local.get('ibest', local.get('iteration_best'))
    return dict(params=digest(local['params']), means=digest(local['means']),
                deviations=digest(local['deviations']), best=digest(local['best_controls']),
                winner=int(winner), best_class=int(local['best_class']),
                best_cost=float(local['best_cost']),
                best_full=float(local['best_full_clearance']),
                best_first=float(local['best_first_clearance']),
                best_physical=float(local['best_first_physical']))


def make(arm, tracing):
    if arm == 'old_exact':
        p = old.ReferenceTraceCEM(CFG)
    elif arm == 'old_hermite':
        p = old.RankCertifiedCEM(CFG)
    else:
        p = MPCPlanner(CFG, envelope={'exact': None, 'rectangle': CachedRectangle,
                                      'hermite': CompiledDiscRisk}[arm])
    p.audit_elites = []
    p.audit_iterations = []
    if tracing:
        select = p._elite_indices
        def elite(cost, full, first, count):
            ids = select(cost, full, first, count)
            p.audit_elites.append(dict(ids=ids.tolist(), cost=digest(cost[ids]),
                                        full=digest(full[ids]), first=digest(first[ids])))
            return ids
        p._elite_indices = elite
        if arm not in ('old_exact', 'old_hermite'):
            p.trace_hook = lambda state: p.audit_iterations.append(capture(state))
    elif arm in ('old_exact', 'old_hermite'):
        p._elite_indices = original.ContinuousCEMMPC._elite_indices
    return p


def plan(p, arm, frame, tracing):
    p.audit_elites.clear()
    p.audit_iterations.clear()
    previous = sys.gettrace()
    if tracing and arm in ('old_exact', 'old_hermite'):
        fn = original.ContinuousCEMMPC.plan if arm == 'old_exact' else old.RankCertifiedCEM.plan
        source, start = inspect.getsourcelines(fn)
        import textwrap
        tree = ast.parse(textwrap.dedent(''.join(source)))
        loop = next(node for node in tree.body[0].body if isinstance(node, ast.For))
        line = start + loop.lineno - 1
        def trace(f, event, arg):
            if f.f_code is not fn.__code__:
                return None
            if event == 'line' and f.f_lineno == line and 'params' in f.f_locals:
                p.audit_iterations.append(capture(f.f_locals))
            return trace
        sys.settrace(trace)
    try:
        start = time.perf_counter()
        action, _ = p.plan(frame['obs'], frame['seed'])
        ms = (time.perf_counter()-start)*1000
    finally:
        if tracing and arm in ('old_exact', 'old_hermite'):
            sys.settrace(previous)
    return action, ms


def verify_assets():
    baseline = json.loads((ROOT/'HERMITE_BASELINE.json').read_text())
    for path, sha in baseline['external_assets'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise RuntimeError('asset changed: '+path)
    protocol = json.loads((ROOT/'archive/hermite_audit/protocol.json').read_text())
    for name in ('continuous_mpc_gate.py', 'bayesian_rfs.py'):
        expected = protocol['assets']['/home/abc/workspace/bayes_occ_mpc/'+name]
        assert hashlib.sha256((ROOT/'archive/legacy'/name).read_bytes()).hexdigest() == expected


def edges():
    with (DATA/'replay_20_23000.pkl').open('rb') as f:
        baseline = pickle.load(f)[0]
    passed = []
    for mode in ('empty', 'no_covariance', 'zero_existence', 'one_existence', 'at_goal'):
        frame = copy.deepcopy(baseline)
        obs = frame['obs']
        if mode == 'empty':
            obs.entities = np.empty((0,5))
            for field in ('human_segment_start', 'human_segment_end', 'human_position_covariance',
                          'human_existence', 'human_visible', 'human_uncertainty_buffer'):
                setattr(obs,field,None)
        elif mode == 'no_covariance':
            obs.human_position_covariance = None
        elif mode == 'at_goal':
            obs.goal_xy = obs.robot_xy.copy()
        else:
            obs.human_existence[:] = 0 if mode == 'zero_existence' else 1
        models = {a:make(a,True) for a in ARMS}
        for a,p in models.items():
            plan(p,a,frame,True)
        for a in ARMS[1:]:
            np.testing.assert_array_equal(models['old_exact'].last_controls,models[a].last_controls)
            assert models['old_exact'].audit_elites == models[a].audit_elites
            assert models['old_exact'].audit_iterations == models[a].audit_iterations
        passed.append(mode)

    class BrokenEnvelope:
        table_cdf_values = 0
        def __init__(self,cfg,obs):
            pass
        def bounds(self,positions):
            return np.full(positions.shape[:2],np.nan), np.full(positions.shape[:2],np.nan)
    reference = MPCPlanner(CFG,envelope=None)
    broken = MPCPlanner(CFG,envelope=BrokenEnvelope)
    reference.plan(baseline['obs'],baseline['seed'])
    broken.plan(baseline['obs'],baseline['seed'])
    np.testing.assert_array_equal(reference.last_controls,broken.last_controls)
    assert broken.bound_fallbacks == 1
    for mode in ('nonfinite', 'negative_covariance', 'invalid_existence', 'anisotropic'):
        obs = copy.deepcopy(baseline['obs'])
        if mode == 'nonfinite':
            obs.robot_xy[0] = np.nan
        elif mode == 'invalid_existence':
            obs.human_existence[0] = 1.01
        elif mode == 'anisotropic':
            obs.human_position_covariance[0,:,0,0] += .01
        else:
            obs.human_position_covariance[0,:,0,0] = -1.
        try:
            MPCPlanner(CFG).plan(obs,baseline['seed'])
        except ValueError:
            passed.append(mode)
        else:
            raise AssertionError('invalid input accepted: '+mode)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'edges.json').write_text(json.dumps(dict(passed=passed,envelope_fallback='exact controls preserved'),indent=2))
    print('EDGE_PASS '+str(passed),flush=True)


def belief_check():
    from types import SimpleNamespace
    from dataclasses import fields
    from nav.belief import BeliefConfig
    from integration.crowdnav import BayesObservationAdapter
    scenarios = [('visible',8,0.,0.,1.), ('occlusion_reappearance',12,0.,0.,1.),
                 ('visible_miss_delete',10,0.,0.,1.), ('long_occlusion_delete',46,0.,0.,1.),
                 ('measurement_noise',12,.1,.2,.8), ('timestamp_gap',8,.1,.2,1.)]
    records = []
    raw = []
    for name,length,pos_std,vel_std,detect in scenarios:
        legacy = original.ObservationAdapter('bayes', horizon=CFG.horizon,
            safety_margin=CFG.human_margin, dt=CFG.dt, chance_limit=CFG.chance_limit,
            acceleration_std=CFG.acceleration_std, position_noise_std=pos_std,
            velocity_noise_std=vel_std, detection_probability=detect, observation_seed=817)
        modern = BayesObservationAdapter(CFG, BeliefConfig(dt=CFG.dt,
            acceleration_std=CFG.acceleration_std, position_measurement_std=pos_std,
            velocity_measurement_std=vel_std,detection_probability=min(.98,detect)),
            detection_probability=detect,observation_seed=817)
        observed_counts = []
        for step in range(length):
            stamp = step*CFG.dt + (.5 if name=='timestamp_gap' and step>=3 else 0.)
            detections = [dict(id=i,px=.2*stamp+(i==7)*.8,py=.3*(i==2),
                              vx=.2,vy=0.,radius=.3) for i in (7,2)]
            hidden = name in ('occlusion_reappearance','long_occlusion_delete') and step>=2
            if name == 'occlusion_reappearance' and step>=9:
                hidden = False
            if hidden or (name=='visible_miss_delete' and step>=2):
                detections = []
            axis = np.arange(-3.,3.25,.25)
            mesh = np.meshgrid(axis,axis)
            sensor = np.full(mesh[0].shape,.5 if hidden else 0.)
            for e in detections:
                sensor[(mesh[0]-e['px'])**2+(mesh[1]-e['py'])**2<=e['radius']**2] = 1.
            robot = SimpleNamespace(px=-4.,py=0.,vx=0.,vy=0.,gx=4.,gy=0.,radius=.3,theta=0.)
            state = SimpleNamespace(self_state=robot,policy_entities=detections,provenance='fixture-observation')
            env = SimpleNamespace(global_time=stamp, get_policy_state=lambda:state,
                occlusion_enabled=lambda:True,occlusion=SimpleNamespace(
                    visible_ids=[e['id'] for e in detections],sensor_grid=sensor,_mesh=mesh,res=.25))
            raw.append(dict(sequence=name,step=step,timestamp=stamp,detections=detections,
                            sensor_grid=sensor,mesh=mesh))
            a,b = legacy.read(env),modern.read(env)
            for field in fields(a):
                if field.name=='provenance':
                    continue  # Serialized observation hash moved out of online adapter.
                x,y = getattr(a,field.name),getattr(b,field.name)
                if isinstance(x,np.ndarray):
                    np.testing.assert_array_equal(x,y)
                else:
                    assert x==y,field.name
            assert legacy.reported_ids == modern.reported_ids
            assert list(legacy.rfs.tracks) == list(modern.belief.tracks)
            for key,x in legacy.rfs.tracks.items():
                y = modern.belief.tracks[key]
                for field,value in vars(x).items():
                    if isinstance(value,np.ndarray):
                        np.testing.assert_array_equal(value,getattr(y,field))
                    else:
                        assert value==getattr(y,field),(name,step,key,field)
            assert legacy.rfs._last_time == modern.belief._last_time
            observed_counts.append(len(legacy.rfs.tracks))
        if name in ('visible_miss_delete','long_occlusion_delete'):
            assert observed_counts[-1] == 0
        if name=='occlusion_reappearance':
            assert observed_counts[-1] == 2
        p,q = old.ReferenceTraceCEM(CFG),MPCPlanner(CFG)
        p.plan(a,619);q.plan(b,619)
        np.testing.assert_array_equal(p.last_controls,q.last_controls)
        records.append(dict(sequence=name,steps=length,track_counts=observed_counts,
                            position_std=pos_std,velocity_std=vel_std,detection_probability=detect))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'belief_sequences.pkl').open('wb') as f:
        pickle.dump(raw,f,protocol=pickle.HIGHEST_PROTOCOL)
    result = dict(status='PASS',sequences=records,steps=len(raw),
                  input_type='Six deterministic legal-observation fixtures, not navigation episodes',
                  intentionally_removed='provenance serialization hash; unused density/occupancy/buffer outputs')
    (OUT/'belief.json').write_text(json.dumps(result,indent=2))
    print('BELIEF_PASS sequences=6 steps='+str(len(raw)),flush=True)


def report():
    complete = json.loads((OUT/'complete.json').read_text())
    assert complete['status']=='PASS' and len(complete['tasks'])==30
    records = [json.loads((OUT/f'{n}_{c}.json').read_text()) for n,c in complete['tasks']]
    rng = np.random.default_rng(120926)
    summary = {}
    for n in (5,10,20):
        group = [r for r in records if r['n']==n]
        timing = [row for r in group for row in r['timing']]
        values = {a:np.array([row[a] for row in timing]) for a in ARMS[2:]}
        methods = {a:dict(mean=float(v.mean()),p50=float(np.median(v)),p95=float(np.percentile(v,95)))
                   for a,v in values.items()}
        paired = {}
        for a in ('exact','rectangle'):
            differences = np.array([np.mean([row['hermite']-row[a] for row in r['timing']]) for r in group])
            boots = differences[rng.integers(0,len(group),(20000,len(group)))].mean(axis=1)
            paired[a] = dict(episode_mean_difference=float(differences.mean()),
                             exploratory_ci95=np.percentile(boots,[2.5,97.5]).tolist(),
                             faster_episodes=int((differences<0).sum()),
                             slower_steps=int((values['hermite']>values[a]).sum()))
        summary[n] = dict(configuration='square' if n==20 else 'circle',steps=len(timing),
                          methods=methods,paired=paired)
    result = dict(configurations=summary,steps=complete['steps'],
                  ordered_elite_comparisons=complete['steps']*CFG.iterations*4*4,
                  iteration_comparisons=complete['steps']*CFG.iterations*4,
                  source_hashes=complete['source_hashes'],
                  limitations='Single paired timing pass on development replays; bootstrap clusters by episode')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def function_counts():
    import nav.risk as risk
    rows = []
    cdf = risk.ncx2.cdf
    bessel = risk.ive
    for n in (5,10,20):
        with (DATA/f'replay_{n}_23000.pkl').open('rb') as f:
            frames = pickle.load(f)
        for step in (0,len(frames)//2,len(frames)-1):
            for arm in ARMS[2:]:
                count = dict(table_cdf_values=0,candidate_cdf_values=0,bessel_values=0)
                def counted_cdf(*args,**kwargs):
                    output = cdf(*args,**kwargs)
                    name = 'table_cdf_values' if np.ndim(output)==2 else 'candidate_cdf_values'
                    count[name] += np.size(output)
                    return output
                def counted_bessel(*args,**kwargs):
                    output = bessel(*args,**kwargs)
                    count['bessel_values'] += np.size(output)
                    return output
                risk.ncx2.cdf = counted_cdf
                risk.ive = counted_bessel
                try:
                    p=make(arm,False)
                    p.plan(frames[step]['obs'],frames[step]['seed'])
                finally:
                    risk.ncx2.cdf = cdf
                    risk.ive = bessel
                rows.append(dict(n=n,step=step,arm=arm,counts=count,
                                 exact_candidate_rows=p.risk_rows,total_rows=p.total_rows))
    result=dict(scope='Nine saved posterior snapshots, cold starts; not the full-cohort call count',rows=rows)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'function_counts.json').write_text(json.dumps(result,indent=2))
    print('COUNT_PASS snapshots=9 plans=27',flush=True)


def run(tasks, timing=False):
    OUT.mkdir(parents=True, exist_ok=True)
    record_dir = OUT if len(tasks)==30 else OUT/'gate'
    record_dir.mkdir(exist_ok=True)
    verify_assets()
    records = []
    for n, case in tasks:
        with (DATA/f'replay_{n}_{case}.pkl').open('rb') as f:
            frames = pickle.load(f)
        models = {arm: make(arm, True) for arm in ARMS}
        transitions = []
        for step, frame in enumerate(frames):
            warm = {a: copy.deepcopy(p._previous_mean) for a,p in models.items()}
            try:
                actions = {a: plan(p,a,frame,True)[0] for a,p in models.items()}
                reference = models['old_exact']
                assert len(reference.audit_iterations) == CFG.iterations
                for arm in ARMS[1:]:
                    p = models[arm]
                    np.testing.assert_array_equal(reference.last_controls,p.last_controls)
                    np.testing.assert_array_equal(actions['old_exact'],actions[arm])
                    assert reference.audit_elites == p.audit_elites, ('elites',arm,step)
                    assert reference.audit_iterations == p.audit_iterations, ('iterations',arm,step)
                    for key,value in reference.last_diagnostics.items():
                        assert value == p.last_diagnostics[key], (key,arm,step)
                transitions.append(dict(step=step, class_id=reference.last_diagnostics['feasibility_class'],
                    elites=len(reference.audit_elites), iterations=reference.audit_iterations,
                    fallbacks={a:models[a].bound_fallbacks for a in ARMS[2:]}))
            except Exception:
                with (OUT/'mismatch.pkl').open('wb') as f:
                    pickle.dump(dict(n=n,case=case,step=step,frame=frame,warm=warm,
                        traces={a:dict(elites=p.audit_elites,iterations=p.audit_iterations)
                                for a,p in models.items()}),f)
                raise
        record = dict(n=n,case=case,steps=len(frames),checks=transitions)
        if timing:
            arms = ARMS[2:]
            orders = list(itertools.permutations(arms))
            models = {a:make(a,False) for a in arms}
            times = []
            for step,frame in enumerate(frames):
                row = {}
                for a in orders[(step+case)%6]:
                    row[a] = plan(models[a],a,frame,False)[1]
                times.append(row)
            record['timing'] = times
        records.append(record)
        (record_dir/f'{n}_{case}.json').write_text(json.dumps(record,indent=2))
        print(f'PASS {n}/{case} steps={len(frames)} ordered elites + distributions + winners + controls',flush=True)
    verify_assets()
    summary = dict(status='PASS',tasks=tasks,steps=sum(r['steps'] for r in records),
                   comparisons=4,config=CONFIG,timing=timing,
                   source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ROOT/'nav').glob('*.py')})
    (OUT/('complete.json' if len(tasks)==30 else 'gate.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','source_hashes')}),flush=True)


FOCUS = ROOT/'results/hermite_focused'
FARMS = ('E', 'R', 'T', 'H')
FTASKS = [(n,c) for n in (5,10,20) for c in range(23000,23010)]
_meter = None


def fsave(name, value):
    FOCUS.mkdir(exist_ok=True, parents=True)
    target = FOCUS/name
    temporary = target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)


def fmanifest(report=False):
    path = FOCUS/'protocol.json'
    if path.exists():
        result = json.loads(path.read_text())
        for source, checksum in result['files'].items():
            if report:
                checksum = result.get('reporting_source_hashes', {}).get(source, checksum)
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == checksum, source
        return result
    verify_assets()
    sources = list((ROOT/'nav').glob('*.py')) + [Path(__file__), ROOT/'experiments/controls.py',
                                               ROOT/'archive/hermite_audit/protocol.json']
    sources += [DATA/f'replay_{n}_{c}.pkl' for n,c in FTASKS]
    result = dict(baseline='a8c3d90', tasks=FTASKS, config=CONFIG, step=.125,
        arms=dict(E='reference exact', R='cached rectangle screening',
                  T='direct cubic point, no interval screening', H='original Hermite screening'),
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        layers=['fixed E candidates, exact truth only for post-audit',
                'independent search and warm starts on common exogenous posteriors'],
        repeats=3, order='24 lexicographic permutations; (3*global_frame+repeat)%24',
        timing='external plan wall time including table; no instrumentation or serialization',
        profiles='exclusive instrumented subroutine times, diagnostic only; never mixed with pure timing',
        units='paired episodes; repetitions are technical replicates',
        threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        no_navigation=True, budget_replay_available=False,
        stopping='no grid retuning regardless of T disagreements; no extra navigation or baselines',
        reason_partition='refined elites; refined exact tolerance pool excluding elites; other refined; excluded',
        enclosure_audit_tolerance=1e-10,
        masks='big-endian np.packbits hex; candidate index order, first population bits')
    fsave('protocol.json', result)
    return result


def fmodel(arm):
    from experiments.controls import DirectMPC
    if arm == 'T':
        return DirectMPC(CFG)
    return MPCPlanner(CFG, envelope={'E':None,'R':CachedRectangle,'H':CompiledDiscRisk}[arm])


class Meter:
    """Read-only audit instrumentation; never installed for pure timing."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.times = {}
        self.calls = dict(table_cdf=0, candidate_cdf=0, bessel=0, normal_cdf=0)
        self.stack = []

    def exclude(self, dt):
        if self.stack:
            self.stack[-1][1] += dt

    def wrap(self, fn, bucket):
        def wrapped(*args, **kwargs):
            frame = [bucket, 0.]
            self.stack.append(frame)
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter()-start
                self.times[bucket] = self.times.get(bucket,0.) + (elapsed-frame[1])*1000
                self.stack.pop()
                self.exclude(elapsed)
        return wrapped


def instrument(p, arm):
    m = p.meter = Meter()
    for name, bucket in (('_cost','common'),('_combined_clearance','common'),
                         ('_physical_clearance','common'),('_active_until_goal','common'),
                         ('_rollout','common'),('_human_clearance','common'),
                         ('_score_from_hazard','score'),('_evaluate_batch','screen')):
        setattr(p, name, m.wrap(getattr(p,name),bucket))
    original_hazard = m.wrap(p._belief_collision_hazard, 'point_query' if arm=='T' else 'exact')
    def hazard(*args, **kwargs):
        value = original_hazard(*args, **kwargs)
        start = time.perf_counter()
        p.audit_hazard = None if value is None else value.copy()
        m.exclude(time.perf_counter()-start)
        return value
    p._belief_collision_hazard = hazard
    original_build = m.wrap(p._build_envelope, 'build')
    def build(obs):
        value = original_build(obs)
        if value is not None and arm in ('R','H'):
            original_bounds = m.wrap(value.bounds, 'bounds_query')
            def bounds(positions):
                lo,hi = original_bounds(positions)
                start = time.perf_counter()
                p.audit_bounds = (lo.copy(),hi.copy())
                m.exclude(time.perf_counter()-start)
                return lo,hi
            value.bounds = bounds
        return value
    p._build_envelope = build
    return p


def invoke(p, fn, *args):
    global _meter
    previous, _meter = _meter, p.meter
    try:
        return fn(*args)
    finally:
        _meter = previous


def count_functions(enable):
    import nav.risk as risk
    import experiments.controls as controls
    if not enable:
        risk.ncx2.cdf, risk.ive, risk.ndtr, controls.ndtr = count_functions.originals
        return
    count_functions.originals = (risk.ncx2.cdf, risk.ive, risk.ndtr, controls.ndtr)
    def wrap(fn, kind):
        def counted(*args,**kwargs):
            value = fn(*args,**kwargs)
            if _meter is not None:
                key = kind
                if key=='cdf':
                    key = 'table_cdf' if any(f[0]=='build' for f in _meter.stack) else 'candidate_cdf'
                _meter.calls[key] += int(np.size(value))
            return value
        return counted
    risk.ncx2.cdf = wrap(risk.ncx2.cdf,'cdf')
    risk.ive = wrap(risk.ive,'bessel')
    risk.ndtr = wrap(risk.ndtr,'normal_cdf')
    controls.ndtr = wrap(controls.ndtr,'normal_cdf')


def selection(p, state, scores):
    """Independent read-only mirror, checked against the actual E loop each time."""
    costs,full,first,physical = scores
    if (full>=0).any():
        pool = np.flatnonzero(full>=0)
        winner = int(pool[np.argmin(costs[pool])])
        pool = np.array([winner])
    elif (first>=0).any():
        pool = np.flatnonzero(first>=0)
        pool = pool[full[pool]>=full[pool].max()-.02]
        winner = int(pool[np.argmin(costs[pool])])
    else:
        pool = np.flatnonzero(physical>=physical.max()-.02)
        winner = p._degraded_choice(costs,physical,state['params'],state['obs'])
    elites = []
    for i,count in enumerate(state['counts']):
        lo,hi = state['bounds'][i:i+2]
        ids = MPCPlanner._elite_indices(costs[lo:hi],full[lo:hi],first[lo:hi],
                                       max(4,round(count*CFG.elite_fraction))) + lo
        elites.append(ids.tolist())
    return elites,winner,pool


def packed(mask):
    return np.packbits(mask).tobytes().hex()


def reference_q(obs, positions):
    import nav.risk as risk
    d = np.linalg.norm(positions[:,None]-obs.human_segment_end[None],axis=3)
    var = .5*np.trace(obs.human_position_covariance,axis1=2,axis2=3)
    radius = obs.robot_radius + obs.entities[None,:,None,4]+CFG.human_margin
    far = d-radius>=8*np.sqrt(var)[None]
    q = np.full(d.shape,risk.ndtr(-8.))
    q[~far] = risk.ncx2.cdf(np.broadcast_to(radius**2/np.maximum(var[None],1e-9),d.shape)[~far],
                            2., (d**2/np.maximum(var[None],1e-9))[~far])
    return np.where(var[None]<1e-9,d<=radius,q)


def frame_trace(state):
    p = state['self']
    elites,winner,_ = selection(p,state,(state['costs'],state['full'],state['first'],state['first_physical']))
    assert winner == state['ibest']
    return dict(iteration=int(state['_']), elites=elites, winner=winner,
                params=digest(state['params']), means=digest(state['means']),
                deviations=digest(state['deviations']), best=digest(state['best_controls']))


def fixed_audit(state, helpers, table, result):
    from nav.risk import _keys
    p,obs = state['self'],state['obs']
    e_scores = (state['costs'],state['full'],state['first'],state['first_physical'])
    elites,winner,pool = selection(p,state,e_scores)
    assert winner==state['ibest']
    n = len(state['controls'])
    elite = np.zeros(n,bool); elite[np.concatenate(elites)] = True
    select_pool = np.zeros(n,bool); select_pool[pool] = True
    needed = elite|select_pool
    h = p.audit_hazard
    q = reference_q(obs,state['positions']) if h is not None else None
    row = dict(iteration=int(state['_']), population=n, elites=elites, winner=winner,
               exact_category=np.bincount(_keys(*e_scores[:3])[0],minlength=3).tolist(), methods={})
    active,_,_ = MPCPlanner._active_until_goal(state['positions'],obs)
    limits = p._belief_hazard_limits()[None]
    # Post-audit score endpoints reuse the production formula, never an online oracle.
    base = MPCPlanner._cost(p,state['controls'],obs,state['positions'],state['human_clearance'],state['occupancy'],None)
    gf,gs = MPCPlanner._combined_clearance(p,state['controls'],obs,state['positions'],state['human_clearance'],state['occupancy'],None)
    base_category = _keys(*e_scores[:3])[0]
    for arm,x in helpers.items():
        x.audit_bounds = None
        scores = invoke(x,x._evaluate_batch,state['controls'],obs,state['positions'],
                        state['human_clearance'],state['occupancy'],state['counts'],state['bounds'])
        got,win,_ = selection(x,state,scores)
        exact = x._exact_mask.copy() if arm!='T' else np.zeros(n,bool)
        v = dict(ordered_elites=got,winner=win,
                 elite_route_changes=sum(a!=b for a,b in zip(elites,got)),
                 winner_changed=win!=winner)
        if arm=='T':
            v.update(category_changes=int((_keys(*scores[:3])[0]!=base_category).sum()),
                     hazard_max_error=float(np.max(np.abs(x.audit_hazard-h))) if h is not None else 0.)
            if q is not None:
                tq = table.conditional(state['positions'])
                error = np.abs(tq-q)
                v.update(probability_max_error=float(error.max()),probability_mean_error=float(error.mean()))
        else:
            v.update(refined=int(exact.sum()), elite_refined=int((exact&elite).sum()),
                selection_only_refined=int((exact&select_pool&~elite).sum()),
                extra_refined=int((exact&~needed).sum()),excluded=int((~exact).sum()),
                exact_mask=packed(exact), extra_mask=packed(exact&~needed),
                incorrectly_excluded=int((~exact&elite).sum())+int(not exact[winner]),
                refined_category_changes=int(((_keys(*scores[:3])[0]!=base_category)&exact).sum()))
            assert v['elite_refined']+v['selection_only_refined']+v['extra_refined']+v['excluded']==n
            assert not v['elite_route_changes'] and not v['winner_changed'] and not v['incorrectly_excluded'], (arm,v)
            if h is not None and x.audit_bounds is not None:
                lo,hi = x.audit_bounds
                gap = hi-lo
                clo,flo,slo = MPCPlanner._score_from_hazard(x,base,gf,gs,active,limits,lo)
                chi,fhi,shi = MPCPlanner._score_from_hazard(x,base,gf,gs,active,limits,hi)
                catlo,cathi = _keys(clo,flo,slo)[0],_keys(chi,fhi,shi)[0]
                v.update(hazard_width_mean=float(gap.mean()),hazard_width_max=float(gap.max()),
                    category_ambiguous=int((catlo!=cathi).sum()),
                    enclosure_raw_misses=int(((h<lo)|(h>hi)).sum()),
                    enclosure_max_violation=float(max(0.,(lo-h).max(),(h-hi).max())),
                    enclosure_misses=int(((h<lo-1e-10)|(h>hi+1e-10)).sum()))
                assert v['enclosure_misses']==0,(arm,v)
                # Fixed histogram bins measure refinement versus interval width.
                width = gap.max(axis=1)
                bins = np.array([0,1e-5,1e-3,1e-1,1,10,100,1000,np.inf])
                v['width_bins'] = np.histogram(width,bins)[0].tolist()
                v['extra_width_bins'] = np.histogram(width[exact&~needed],bins)[0].tolist()
                # Save one natural, class-zero route boundary example per frame.
                ids = np.asarray(elites[0]); cut = int(ids[-1])
                if base_category[cut]==0:
                    local = np.flatnonzero(base_category[:state['counts'][0]]==0)
                    local = local[np.argsort(state['costs'][local],kind='stable')]
                    rank = int(np.flatnonzero(local==cut)[0])
                    chosen = local[max(0,rank-5):rank+7]
                    v['boundary'] = dict(ids=chosen.tolist(), cut_cost=float(state['costs'][cut]),
                        exact=state['costs'][chosen].tolist(), lower=clo[chosen].tolist(), upper=chi[chosen].tolist(),
                        ambiguous=(catlo[chosen]!=cathi[chosen]).tolist())
        row['methods'][arm] = v
    result.append(row)


def fgate():
    from experiments.controls import DirectDiscRisk, DirectMPC
    assert DirectMPC.plan is MPCPlanner.plan
    with (DATA/'replay_20_23000.pkl').open('rb') as f:
        frame = pickle.load(f)[0]
    obs = copy.deepcopy(frame['obs'])
    obs.entities = obs.entities[:1].copy()
    obs.human_segment_end = np.zeros((1,CFG.horizon,2))
    obs.human_position_covariance = np.tile(np.eye(2)*.0025,(1,CFG.horizon,1,1))
    obs.human_existence = np.ones(1)
    t,h = DirectDiscRisk(CFG,obs),CompiledDiscRisk(CFG,obs)
    np.testing.assert_array_equal(t.table,h.table)
    np.testing.assert_array_equal(t.derivative,h.derivative)
    radius = float(t.radius[0,0])
    z = np.array([-9.,-8.,-3.0625,-.0625,0.,.0625,7.9375,8.,9.])
    positions = np.zeros((len(z),CFG.horizon,2)); positions[:,:,0]=(radius+.05*z)[:,None]
    t.conditional_bounds = lambda *_: (_ for _ in ()).throw(AssertionError('T called intervals'))
    err = float(np.max(np.abs(t.conditional(positions)-reference_q(obs,positions))))
    assert err < h.interpolation_error+1e-10
    obs.human_position_covariance[:] = 0
    t = DirectDiscRisk(CFG,obs)
    np.testing.assert_array_equal(t.conditional(positions),reference_q(obs,positions))
    for mode in ('empty','no_covariance','zero_existence'):
        o = copy.deepcopy(frame['obs'])
        if mode=='empty':
            o.entities=np.empty((0,5)); o.human_position_covariance=None; o.human_existence=None
        elif mode=='no_covariance':
            o.human_position_covariance=None
        else:
            o.human_existence[:]=0
        a,b=fmodel('E'),fmodel('T')
        a.plan(o,frame['seed']);b.plan(o,frame['seed'])
        np.testing.assert_array_equal(a.last_controls,b.last_controls)
    p=fmodel('T')
    def guarded(cfg,obs):
        table=DirectDiscRisk(cfg,obs)
        table.bounds=lambda *_: (_ for _ in ()).throw(AssertionError('T planner called bounds'))
        return table
    p.envelope=guarded
    p.plan(frame['obs'],frame['seed'])
    fsave('gate.json',dict(status='PASS', interpolation_error=err,
                          tests=['identical nodes/derivatives','interior and tails','deterministic',
                                 'T never calls bounds','empty','absent covariance','zero existence']))
    fmechanism([(5,23000),(10,23000),(20,23000)],limit=3)


def fmechanism(tasks=FTASKS, limit=None):
    global _meter
    if not limit:
        fmanifest()
    path = FOCUS/('mechanism_gate.jsonl' if limit else 'mechanism.jsonl')
    if path.exists() and not limit:
        raise RuntimeError('audit already exists; do not overwrite evidence: '+str(path))
    count_functions(True)
    try:
        with path.open('w') as output:
            for n,case in tasks:
                with (DATA/f'replay_{n}_{case}.pkl').open('rb') as f:
                    frames = pickle.load(f)
                if limit:
                    frames=frames[:limit]
                models={a:instrument(fmodel(a),a) for a in FARMS}
                for step,frame in enumerate(frames):
                    obs=frame['obs']
                    helpers={a:instrument(fmodel(a),a) for a in ('R','T','H')}
                    for a,p in helpers.items():
                        p.risk_rows=p.refinement_batches=p.bound_fallbacks=0
                        p._compiled=invoke(p,p._build_envelope,obs)
                    from experiments.controls import DirectDiscRisk
                    truth_table=DirectDiscRisk(CFG,obs) if obs.entities.size and obs.human_position_covariance is not None else None
                    fixed=[]; traces={a:[] for a in FARMS}
                    def hook(state):
                        global _meter
                        previous,_meter=_meter,None
                        try:
                            traces['E'].append(frame_trace(state))
                            fixed_audit(state,helpers,truth_table,fixed)
                        finally:
                            _meter=previous
                    for a,p in models.items():
                        p.meter.reset()
                        p.trace_hook=hook if a=='E' else lambda state,a=a: traces[a].append(frame_trace(state))
                        invoke(p,p.plan,obs,frame['seed'])
                    free={}
                    for a in FARMS:
                        p=models[a]
                        differences=[i for i,(x,y) in enumerate(zip(traces['E'],traces[a])) if x!=y]
                        elite_changes=sum(x['elites']!=y['elites'] for x,y in zip(traces['E'],traces[a]))
                        winner_changes=sum(x['winner']!=y['winner'] for x,y in zip(traces['E'],traces[a]))
                        action_change=not np.array_equal(p.last_controls[0],models['E'].last_controls[0])
                        controls_change=not np.array_equal(p.last_controls,models['E'].last_controls)
                        free[a]=dict(first_different_iteration=differences[0] if differences else None,
                            elite_iterations_changed=elite_changes,winner_iterations_changed=winner_changes,
                            action_changed=action_change,controls_changed=controls_change,
                            max_action_difference=float(np.max(np.abs(p.last_controls[0]-models['E'].last_controls[0]))),
                            max_control_difference=float(np.max(np.abs(p.last_controls-models['E'].last_controls))),
                            traces=traces[a], calls=p.meter.calls, profile_ms=p.meter.times,
                            table_nodes=p.table_cdf_values,
                            curves=int(p._compiled.table.shape[0]) if p._compiled is not None and hasattr(p._compiled,'table') else 0)
                        if a in ('R','H'):
                            assert not differences and not controls_change,(n,case,step,a,free[a])
                    row=dict(n=n,case=case,step=step,seed=frame['seed'],fixed=fixed,free=free,
                        fixed_calls={a:p.meter.calls for a,p in helpers.items()})
                    output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
                print('MECHANISM',n,case,len(frames),'frames complete',flush=True)
    finally:
        count_functions(False)
    if not limit:
        fmanifest()


def ftiming():
    fmanifest()
    records=[json.loads(line) for line in (FOCUS/'mechanism.jsonl').read_text().splitlines()]
    assert len(records)==1358
    path=FOCUS/'timing.jsonl'
    if path.exists():
        raise RuntimeError('timing already exists; do not overwrite technical replicates')
    orders=list(itertools.permutations(FARMS))
    with path.open('w') as output:
        for repeat in range(3):
            global_step=0
            for n,case in FTASKS:
                with (DATA/f'replay_{n}_{case}.pkl').open('rb') as f:
                    frames=pickle.load(f)
                models={a:fmodel(a) for a in FARMS}
                for step,frame in enumerate(frames):
                    order=orders[(3*global_step+repeat)%len(orders)]
                    for position,a in enumerate(order):
                        start=time.perf_counter()
                        models[a].plan(frame['obs'],frame['seed'])
                        ms=(time.perf_counter()-start)*1000
                        output.write(json.dumps(dict(repeat=repeat,n=n,case=case,step=step,
                            arm=a,order=position,ms=ms))+'\n')
                    global_step+=1
                output.flush()
                print('TIMING',repeat,n,case,len(frames),'frames complete',flush=True)
    fmanifest()


def ftables():
    fmanifest(report=True)
    from collections import Counter
    mechanism=[json.loads(x) for x in (FOCUS/'mechanism.jsonl').read_text().splitlines()]
    timing=[json.loads(x) for x in (FOCUS/'timing.jsonl').read_text().splitlines()]
    assert len(mechanism)==1358 and len(timing)==1358*4*3
    assert len({(r['repeat'],r['n'],r['case'],r['step'],r['arm']) for r in timing})==len(timing)
    rng=np.random.default_rng(20260912)
    result=dict(frames=1358, episodes=30, repeats=3, configurations={},
                no_navigation=True, baseline='a8c3d90',
                order_counts={a:dict(Counter(r['order'] for r in timing if r['arm']==a)) for a in FARMS})
    for n in (5,10,20):
        group=[r for r in mechanism if r['n']==n]
        methods={}
        for a in FARMS:
            times=np.array([r['ms'] for r in timing if r['n']==n and r['arm']==a])
            episode_times=np.array([np.mean([r['ms'] for r in timing if r['n']==n and r['arm']==a and r['case']==c])
                                    for c in range(23000,23010)])
            profile={};calls={}
            for row in group:
                for k,v in row['free'][a]['profile_ms'].items(): profile[k]=profile.get(k,0.)+v/len(group)
                for k,v in row['free'][a]['calls'].items(): calls[k]=calls.get(k,0)+v
            v=dict(mean=float(times.mean()),p50=float(np.median(times)),p95=float(np.percentile(times,95)),
                p99=float(np.percentile(times,99)),episode_means=episode_times.tolist(),
                controls_changed=sum(r['free'][a]['controls_changed'] for r in group),
                actions_changed=sum(r['free'][a]['action_changed'] for r in group),
                max_action_difference=max(r['free'][a]['max_action_difference'] for r in group),
                max_control_difference=max(r['free'][a]['max_control_difference'] for r in group),
                elite_iterations_changed=sum(r['free'][a]['elite_iterations_changed'] for r in group),
                winner_iterations_changed=sum(r['free'][a]['winner_iterations_changed'] for r in group),
                first_free_disagreement=next((dict(case=r['case'],step=r['step'],
                    iteration=r['free'][a]['first_different_iteration']) for r in group
                    if r['free'][a]['first_different_iteration'] is not None),None),
                calls=calls,diagnostic_mean_ms=profile,
                curves_mean=float(np.mean([r['free'][a]['curves'] for r in group])))
            if a!='E':
                batches=[b['methods'][a] for r in group for b in r['fixed']]
                v['fixed']=dict(elite_route_changes=sum(b['elite_route_changes'] for b in batches),
                                winner_changes=sum(b['winner_changed'] for b in batches))
                if a=='T':
                    v['fixed'].update(category_changes=sum(b['category_changes'] for b in batches),
                        max_probability_error=max(b.get('probability_max_error',0.) for b in batches),
                        max_hazard_error=max(b['hazard_max_error'] for b in batches))
                else:
                    names=('refined','elite_refined','selection_only_refined','extra_refined','excluded',
                           'incorrectly_excluded','refined_category_changes','category_ambiguous','enclosure_misses','enclosure_raw_misses')
                    v['fixed'].update({k:sum(b.get(k,0) for b in batches) for k in names})
                    v['fixed']['total']=len(batches)*CFG.population
                    v['fixed']['hazard_width_mean']=float(np.mean([b.get('hazard_width_mean',0.) for b in batches]))
                    v['fixed']['enclosure_max_violation']=max(b.get('enclosure_max_violation',0.) for b in batches)
                    for k in ('width_bins','extra_width_bins'):
                        v['fixed'][k]=np.sum([b.get(k,[0]*8) for b in batches],axis=0).tolist()
                    # No-risk batches have an all-resolved mask but perform no CDF refinement.
                    bounded=[b for b in batches if 'hazard_width_mean' in b]
                    v['fixed']['bounded_partition']={k:sum(b[k] for b in bounded)
                        for k in ('refined','elite_refined','selection_only_refined','extra_refined','excluded')}
                    v['fixed']['bounded_partition']['total']=len(bounded)*CFG.population
                    v['fixed']['without_bounds_rows']=(len(batches)-len(bounded))*CFG.population
            methods[a]=v
        paired={}
        for a in ('E','R','T'):
            x=np.array(methods['H']['episode_means']);y=np.array(methods[a]['episode_means'])
            delta=x-y;ratio=x/y;idx=rng.integers(0,10,(20000,10))
            ht={(r['repeat'],r['case'],r['step']):r['ms'] for r in timing if r['n']==n and r['arm']=='H'}
            at={(r['repeat'],r['case'],r['step']):r['ms'] for r in timing if r['n']==n and r['arm']==a}
            slower=[list(k) for k in ht if ht[k]>at[k]]
            paired[a]=dict(mean_difference=float(delta.mean()),ci95=np.percentile(delta[idx].mean(axis=1),[2.5,97.5]).tolist(),
                mean_episode_ratio=float(ratio.mean()),ratio_ci95=np.percentile(ratio[idx].mean(axis=1),[2.5,97.5]).tolist(),
                slower_episodes=[23000+i for i,v in enumerate(delta) if v>0],slower_step_replicates=slower)
        result['configurations'][n]=dict(frames=len(group),methods=methods,paired=paired)
    fsave('summary.json',result)
    print(json.dumps({n:{a:{k:v[k] for k in ('p50','p95','controls_changed','actions_changed')}
                        for a,v in g['methods'].items()} for n,g in result['configurations'].items()},indent=2),flush=True)
    fplots(result,mechanism)


def fplots(result,mechanism):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'E':'#595959','R':'#d68922','T':'#36986b','H':'#3274ad'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for j,n in enumerate((5,10,20)):
        methods=result['configurations'][n]['methods']
        for i,a in enumerate(FARMS):
            axes[0].bar(j+(i-1.5)*.18,methods[a]['p50'],width=.17,color=colors[a],label=a if j==0 else None)
        for i,a in enumerate(('E','T')):
            p=result['configurations'][n]['paired'][a];lo,hi=p['ci95']
            axes[1].errorbar(j+(i-.5)*.15,p['mean_difference'],
                yerr=[[p['mean_difference']-lo],[hi-p['mean_difference']]],fmt='o',
                color=colors[a],label='H minus '+a if j==0 else None,capsize=3)
    for ax in axes:
        ax.set_xticks(range(3));ax.set_xticklabels(['5 circle','10 circle','20 square']);ax.legend(frameon=False)
    axes[0].set_ylabel('Median planning time (ms)')
    axes[1].set_ylabel('Paired episode mean difference (ms)');axes[1].axhline(0,color='#777777',linewidth=.7)
    fig.savefig(FOCUS/'latency.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    names=('elite_refined','selection_only_refined','extra_refined','excluded')
    labels=('Elite','Selection only','Extra refinement','Excluded')
    cs=('#3274ad','#36986b','#d68922','#dddddd')
    for j,(n,a) in enumerate((n,a) for n in (5,10,20) for a in ('R','H')):
        s=result['configurations'][n]['methods'][a]['fixed']['bounded_partition'];bottom=0
        for k,label,c in zip(names,labels,cs):
            height=s[k]/s['total']*100
            axes[0].bar(j,height,bottom=bottom,color=c,label=label if j==0 else None);bottom+=height
    axes[0].set_xticks(range(6));axes[0].set_xticklabels(['5 R','5 H','10 R','10 H','20 R','20 H'])
    axes[0].set_ylabel('Bound-evaluated candidate partition (%)');axes[0].legend(frameon=False,fontsize=8)
    for a in ('R','H'):
        s=result['configurations'][20]['methods'][a]['fixed']
        count=np.array(s['width_bins']);extra=np.array(s['extra_width_bins'])
        valid=count>0
        axes[1].plot(np.arange(8)[valid],100*extra[valid]/count[valid],'o-',color=colors[a],label=a)
    axes[1].set_xticks(range(8));axes[1].set_xticklabels(['<1e-5','1e-3','0.1','1','10','100','1000','inf'],rotation=35)
    axes[1].set_xlabel('Upper edge of max step-hazard interval-width bin')
    axes[1].set_ylabel('Extra refinement within bin (%)');axes[1].legend(frameon=False)
    fig.savefig(FOCUS/'mechanism.png',dpi=180);plt.close(fig)
    example=next((dict(case=r['case'],step=r['step'],batch=b) for r in mechanism if r['n']==20
                  for b in r['fixed'] if all('boundary' in b['methods'][a] for a in ('R','H'))),None)
    if example is not None:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained',sharey=True)
        for ax,a in zip(axes,('R','H')):
            b=example['batch']['methods'][a]['boundary'];cut=b['cut_cost'];x=np.arange(len(b['ids']))
            ax.vlines(x,np.array(b['lower'])-cut,np.array(b['upper'])-cut,color=colors[a],linewidth=2)
            ax.scatter(x,np.array(b['exact'])-cut,color='#222222',s=12,label='Reference score')
            ambiguous=np.asarray(b['ambiguous'])
            ax.scatter(x[ambiguous],(np.array(b['upper'])-cut)[ambiguous],marker='x',color='#b44b44',label='Category unresolved')
            ax.axhline(0,color='#777777',linestyle='--',linewidth=1)
            ax.set_yscale('symlog',linthresh=.01);ax.set_title(a+' bounds on identical E candidates')
            ax.set_xticks(x);ax.set_xticklabels(b['ids'],rotation=60,fontsize=7);ax.set_xlabel('Candidate index');ax.legend(fontsize=7,frameon=False)
        axes[0].set_ylabel('Cost minus exact route-elite cutoff (symlog)')
        fig.suptitle('First class-0 boundary: case %d, step %d, iteration %d' %
                     (example['case'],example['step'],example['batch']['iteration']),fontsize=10)
        fig.savefig(FOCUS/'boundary.png',dpi=180);plt.close(fig)


if __name__ == '__main__' and any(a.startswith('--focused-') for a in sys.argv):
    if '--focused-gate' in sys.argv:
        fgate()
    elif '--focused-mechanism' in sys.argv:
        fmechanism()
    elif '--focused-timing' in sys.argv:
        ftiming()
    elif '--focused-tables' in sys.argv:
        ftables()
    elif '--focused-all' in sys.argv:
        fmechanism();ftiming();ftables()
    else:
        raise SystemExit('unknown focused command')
elif __name__ == '__main__':
    tasks = [(5,23000),(10,23000),(20,23000)]
    if '--full' in sys.argv:
        tasks = [(n,c) for n in (5,10,20) for c in range(23000,23010)]
    if '--edges' in sys.argv:
        edges()
    elif '--belief' in sys.argv:
        belief_check()
    elif '--report' in sys.argv:
        report()
    elif '--counts' in sys.argv:
        function_counts()
    else:
        run(tasks, timing='--timing' in sys.argv)
