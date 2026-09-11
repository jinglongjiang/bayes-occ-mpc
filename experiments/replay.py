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
sys.path.insert(1, str(ROOT/'archive/hermite_audit'))
sys.path.append(str(AUDIT))
import continuous_mpc_gate as original
import rank_certified_cem as old
from fair_three_arm import CachedRectangle
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
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected


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


def run(tasks, timing=False):
    OUT.mkdir(parents=True, exist_ok=True)
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
        (OUT/f'{n}_{case}.json').write_text(json.dumps(record,indent=2))
        print(f'PASS {n}/{case} steps={len(frames)} ordered elites + distributions + winners + controls',flush=True)
    verify_assets()
    summary = dict(status='PASS',tasks=tasks,steps=sum(r['steps'] for r in records),
                   comparisons=4,config=CONFIG,timing=timing,
                   source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ROOT/'nav').glob('*.py')})
    (OUT/('complete.json' if len(tasks)==30 else 'gate.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','source_hashes')}),flush=True)


if __name__ == '__main__':
    tasks = [(5,23000),(10,23000),(20,23000)]
    if '--full' in sys.argv:
        tasks = [(n,c) for n in (5,10,20) for c in range(23000,23010)]
    if '--edges' in sys.argv:
        edges()
    else:
        run(tasks, timing='--timing' in sys.argv)
