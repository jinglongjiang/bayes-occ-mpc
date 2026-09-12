"""Causal continuous-goal inference; offline only, no navigation changes."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import goal_model_probe as base
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

OUT = ROOT / 'results/goal_posterior_probe'
BASELINE = 'f05a0fe'
ARMS = ('CV', 'C', 'D', 'Heading-FULL', 'Interaction-MAP', 'Interaction-FULL')
FLOOR = 1e-4  # displacement scale in metres; explicit numerical floor, not sensor noise


def encode(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=encode, allow_nan=False) + '\n')
    tmp.replace(p)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def data():
    protocol = json.loads((base.OUT / 'protocol.json').read_text())
    for path, expected in protocol['files'].items():
        if Path(path).resolve() == Path(base.__file__).resolve():
            old = subprocess.check_output(['git', 'show', BASELINE + ':experiments/goal_model_probe.py'], cwd=ROOT)
            if hashlib.sha256(old).hexdigest() != expected:
                raise RuntimeError('baseline model hash mismatch')
        elif digest(path) != expected:
            raise RuntimeError('frozen dependency changed: ' + path)
    if digest(base.OUT / 'episodes.jsonl') != protocol['data_sha256']:
        raise RuntimeError('raw records changed')
    base.legacy._load_modules(base.CROWD)
    records = [json.loads(line) for line in (base.OUT / 'episodes.jsonl').read_text().splitlines()]
    return records


def contexts(legal):
    past = {}
    for frame in legal['frames']:
        output = {}
        for e in frame['detections']:
            tid = e['id']
            past.setdefault(tid, []).append((frame['step'], np.array([e['px'], e['py']]),
                                             np.array([e['vx'], e['vy']])))
            output[tid] = base.legal_features(legal, frame, e, past[tid])
        yield frame['step'], output


def forward(features, goals, steps=1):
    goals = np.asarray(goals).reshape(-1, 2)
    unique, inverse = np.unique(goals, axis=0, return_inverse=True)
    p = np.array([base.predict(features, g, 'improved', .5, steps=steps) for g in unique])
    return p[inverse]


def heading_delta(features, goals):
    direction = goals - features['pos']
    distance = np.linalg.norm(direction, axis=1)
    out = .25 * features['speed'] * direction / np.maximum(distance[:, None], 1e-12)
    out[distance <= .35] = 0.
    return out


def fit(records):
    residuals, heading = [], []
    for r in records:
        if r['split'] != 'development':
            continue
        previous, last = {}, -2
        for step, features in contexts(r):
            if step == last + 1:
                for tid in previous.keys() & features.keys():
                    f = previous[tid]
                    goal = np.array(r['truth']['goals'][tid])
                    observed = features[tid]['pos'] - f['pos']
                    residuals.append(observed - (forward(f, [goal])[0, 0] - f['pos']))
                    heading.append(observed - heading_delta(f, goal[None])[0])
            previous, last = features, step
    e = np.array(residuals)

    def matrix(v):
        l = np.array([[np.exp(v[0]), 0.], [v[1], np.exp(v[2])]])
        return l @ l.T + np.eye(2) * FLOOR**2

    def objective(v):
        s = matrix(v)
        d = np.einsum('ni,ij,nj->n', e, np.linalg.inv(s), e)
        return .5 * np.linalg.slogdet(s)[1] + 3. * np.mean(np.log1p(d / 4.))

    scale = max(np.sqrt(np.mean(e**2)), FLOOR)
    result = minimize(objective, [np.log(scale), 0., np.log(scale)], method='L-BFGS-B',
                      bounds=[(-18., 0.), (-1., 1.), (-18., 0.)])
    if not result.success:
        raise RuntimeError('Student-t fit failed: ' + result.message)
    s = matrix(result.x)
    value = dict(student_df=4, student_scale=s, numerical_floor_m=FLOOR,
                 displacement_rms_m=float(np.sqrt(np.mean(np.sum(e*e, axis=1)))),
                 residual_count=len(e), scale_eigen_std_m=np.sqrt(np.linalg.eigvalsh(s)),
                 heading_sigma_m=max(FLOOR, float(np.sqrt(np.mean(np.array(heading)**2)))),
                 fit_objective=float(result.fun), development_cases=[r['case'] for r in records if r['split']=='development'])
    save('calibration.json', value)
    print('CALIBRATION', json.dumps(value, default=encode), flush=True)
    return value


class GoalPosterior:
    """No goals, future labels, outcome, or record object in this API."""
    def __init__(self, scene, first_step, first, calibration, seed, count=128, kind='interaction'):
        self.rng = np.random.default_rng(seed)
        self.scene, self.kind, self.count = scene, kind, count
        self.fixed = scene == 'circle_crossing' and first_step == 0
        self.side = -np.sign(first['pos'][0]) if scene == 'square_crossing' and first_step == 0 else 0.
        self.inverse = np.linalg.inv(calibration['student_scale'])
        self.sigma = calibration['heading_sigma_m']
        self.goals = (-first['pos'])[None].copy() if self.fixed else self.sample(count)
        self.logweights = np.full(len(self.goals), -np.log(len(self.goals)))
        self.density = np.zeros(len(self.goals))
        self.history = []
        self.previous = (first_step, first)
        self.resamples = self.moves = self.accepts = self.updates = 0
        self.seconds = 0.
        self.ess_before = float(len(self.goals))

    def sample(self, n):
        if self.scene == 'circle_crossing':
            angle = self.rng.uniform(-np.pi, np.pi, n)
            lo, hi = 4.-np.sqrt(.5), 4.+np.sqrt(.5)
            radius = np.sqrt(self.rng.uniform(lo*lo, hi*hi, n))
            return np.stack([radius*np.cos(angle), radius*np.sin(angle)], axis=1)
        out = self.rng.uniform(-5., 5., (n, 2))
        if self.side:
            out[:, 0] = self.side * np.abs(out[:, 0])
        return out

    def valid(self, goals):
        goals = np.asarray(goals).reshape(-1, 2)
        if self.scene == 'circle_crossing':
            radius = np.linalg.norm(goals, axis=1)
            return (radius >= 4.-np.sqrt(.5)) & (radius <= 4.+np.sqrt(.5))
        ok = np.all(np.abs(goals) <= 5., axis=1)
        return ok & (goals[:, 0]*self.side >= 0) if self.side else ok

    def likelihood(self, goals, entry):
        f, observed = entry
        if self.kind == 'heading':
            e = observed - heading_delta(f, goals)
            return -np.sum(e*e, axis=1) / (2*self.sigma**2)
        e = observed - (forward(f, goals)[:, 0] - f['pos'])
        return -3. * np.log1p(np.einsum('ni,ij,nj->n', e, self.inverse, e)/4.)

    def logposterior(self, goals):
        goals = np.asarray(goals).reshape(-1, 2)
        ok = self.valid(goals)
        out = np.full(len(goals), -np.inf)
        out[ok] = 0.
        for entry in self.history:
            out[ok] += self.likelihood(goals[ok], entry) if np.any(ok) else 0.
        return out

    def update(self, step, features):
        start = time.perf_counter()
        old_step, old = self.previous
        if step <= old_step:
            raise ValueError('duplicate or out-of-order evidence')
        self.previous = (step, features)
        if self.fixed or step != old_step + 1:
            self.seconds += time.perf_counter() - start
            return
        entry = (old, features['pos'] - old['pos'])
        ll = self.likelihood(self.goals, entry)
        self.history.append(entry)
        self.updates += 1
        self.density += ll
        self.logweights += ll
        self.logweights -= logsumexp(self.logweights)
        weights = np.exp(self.logweights)
        self.ess_before = float(1./np.sum(weights*weights))
        if self.ess_before < self.count/2:
            self.resamples += 1
            indices = np.searchsorted(np.cumsum(weights), (self.rng.random()+np.arange(self.count))/self.count)
            indices = np.minimum(indices, self.count-1)
            self.goals, self.density = self.goals[indices].copy(), self.density[indices].copy()
            self.logweights[:] = -np.log(self.count)
            # Fixed symmetric proposal: outside support is rejected, never clipped.
            proposal = self.goals + self.rng.normal(0., .35, self.goals.shape)
            proposed = self.logposterior(proposal)
            accept = np.log(self.rng.random(self.count)) < proposed - self.density
            self.goals[accept], self.density[accept] = proposal[accept], proposed[accept]
            self.moves += self.count
            self.accepts += int(accept.sum())
        self.seconds += time.perf_counter() - start

    def map_goal(self):
        if self.fixed:
            return self.goals[0].copy()
        start = time.perf_counter()
        best = self.goals[np.argmax(self.density)].copy()
        score = float(np.max(self.density))
        # Optimize density, not the uniform weights after resampling.
        for index in np.argsort(-self.density, kind='stable')[:2]:
            result = minimize(lambda g: -self.logposterior(g)[0], self.goals[index],
                              method='Nelder-Mead', options={'maxfev':48, 'xatol':1e-3, 'fatol':1e-4})
            if np.isfinite(result.fun) and -result.fun > score:
                best, score = result.x.copy(), -float(result.fun)
        self.seconds += time.perf_counter() - start
        return best

    def snapshot(self):
        return dict(goals=self.goals.copy(), weights=np.exp(self.logweights), density=self.density.copy(),
                    updates=self.updates, ess_before=self.ess_before, resamples=self.resamples,
                    moves=self.moves, accepted_moves=self.accepts, fixed=self.fixed)


def legal_record(record):
    return {key: record[key] for key in ('case', 'scene', 'frames')}


def infer(legal, calibration, count=128, query_keys=None, include_heading=True):
    tracks = {}
    for step, features in contexts(legal):
        frame_start = time.perf_counter()
        for tid, f in features.items():
            if tid not in tracks:
                tracks[tid] = {kind: GoalPosterior(legal['scene'], step, f, calibration,
                    legal['case']*1009+tid*97, count, kind) for kind in
                    (('interaction', 'heading') if include_heading else ('interaction',))}
            else:
                for model in tracks[tid].values():
                    model.update(step, f)
        update_ms = (time.perf_counter()-frame_start)*1000.
        for tid, f in features.items():
            if query_keys is not None and (step, tid) not in query_keys:
                continue
            began = time.perf_counter()
            model = tracks[tid]['interaction']
            snapshot = model.snapshot()
            map_goal = model.map_goal()
            snapshot['positions'] = forward(f, snapshot['goals'], 16)
            snapshot['map_goal'] = map_goal
            snapshot['map_positions'] = forward(f, [map_goal], 16)[0]
            heading = None
            if include_heading:
                heading = tracks[tid]['heading'].snapshot()
                heading['positions'] = forward(f, heading['goals'], 16)
            yield dict(case=legal['case'], step=step, tid=tid, conflict=f['conflict'],
                       goal_source=f['goal_source'], interaction=snapshot, heading=heading,
                       query_ms=(time.perf_counter()-began)*1000., update_frame_ms=update_ms)


def score_modes(positions, weights, truth):
    positions = positions[:, np.array(base.HORIZONS)-1]
    mean = np.einsum('n,nhd->hd', weights, positions)
    error = np.linalg.norm(mean-truth, axis=1)
    energy, spread = [], []
    for h in range(4):
        p = positions[:, h]
        energy.append(float(weights @ np.linalg.norm(p-truth[h], axis=1) - .5*weights @ cdist(p,p) @ weights))
        spread.append(float(weights @ np.sum((p-mean[h])**2, axis=1)))
    return error.tolist(), energy, spread


def evaluate(record, calibration, mode_stream):
    keys = {(s, tid) for s, tid, _ in base.queries(record)}
    baseline = {(s, tid): f for s, tid, f in base.queries(record)}
    rows = []
    for result in infer(legal_record(record), calibration, query_keys=keys):
        step, tid = result['step'], result['tid']
        f = baseline[step,tid]
        truth = np.array([record['truth']['positions'][step+h][tid] for h in base.HORIZONS])
        mode_stream.write(json.dumps(result, default=encode, allow_nan=False)+'\n')
        point, energy, spread = {}, {}, {}
        for arm, goal, mode in [('CV', f['goal'], 'cv'), ('C', f['goal'], 'improved'),
                                ('D', np.array(record['truth']['goals'][tid]), 'improved')]:
            positions = base.predict(f, goal, mode, .5)[None]
            point[arm], energy[arm], spread[arm] = score_modes(positions, np.ones(1), truth)
        im, hm = result['interaction'], result['heading']
        for arm, positions, weights in [('Interaction-MAP',im['map_positions'][None],np.ones(1)),
                                        ('Interaction-FULL',im['positions'],im['weights']),
                                        ('Heading-FULL',hm['positions'],hm['weights'])]:
            point[arm], energy[arm], spread[arm] = score_modes(positions, weights, truth)
        rows.append(dict(case=record['case'], people=record['people'], step=step, tid=tid,
            conflict=result['conflict'], fixed=im['fixed'], errors=point, energy=energy, spread=spread,
            ess_before=im['ess_before'], updates=im['updates'], query_ms=result['query_ms'],
            update_frame_ms=result['update_frame_ms'],
            confidence=float(np.sum(im['weights']**2)),
            map_goal_error=float(np.linalg.norm(im['map_goal']-record['truth']['goals'][tid]))))
    if set((r['step'],r['tid']) for r in rows) != keys:
        raise RuntimeError('query pool changed')
    return rows


def selftest(records, calibration):
    namespace = {'__file__':str(base.__file__), '__name__':'frozen_reference'}
    source = subprocess.check_output(['git','show',BASELINE+':experiments/goal_model_probe.py'],cwd=ROOT)
    exec(compile(source, 'frozen_goal_model_probe', 'exec'), namespace)
    tested = 0
    for record in records[:3]:
        for step, tid, f in list(base.queries(record))[:8]:
            g = np.array(record['truth']['goals'][tid])
            old = namespace['predict'](f,g,'improved',.5)
            assert np.array_equal(old, base.predict(f,g,'improved',.5))
            assert np.array_equal(old[0], base.predict(f,g,'improved',.5,steps=1)[0])
            tested += 1
    r = copy.deepcopy(records[0])
    short = legal_record(r)
    short['frames'] = short['frames'][:6]
    keys = {(5,e['id']) for e in short['frames'][-1]['detections']}
    first = list(infer(short,calibration,16,keys))
    r['truth']['goals'] = list(reversed(r['truth']['goals']))
    r['truth']['positions'] = []
    changed = legal_record(r); changed['frames'] = changed['frames'][:6]
    second = list(infer(changed,calibration,16,keys))
    for a,b in zip(first,second):
        assert np.array_equal(a['interaction']['goals'],b['interaction']['goals'])
        assert np.array_equal(a['interaction']['weights'],b['interaction']['weights'])
        assert np.array_equal(a['interaction']['positions'],b['interaction']['positions'])
    f = next(iter(next(contexts(short))[1].values()))
    p = GoalPosterior('square_crossing',3,f,calibration,71,16)
    before = p.snapshot()
    p.update(5,f)
    assert p.updates == 0 and np.array_equal(before['weights'],p.snapshot()['weights'])
    try:
        p.update(5,f)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate update accepted')
    p.update(6,f)
    assert p.updates == 1
    np.testing.assert_allclose(p.logposterior(p.goals),p.density,atol=1e-10,rtol=1e-12)
    ll=p.likelihood(np.repeat(p.goals[:1],2,axis=0),p.history[0])
    assert ll[0] == ll[1]
    checks=dict(bitwise_model_parity_queries=tested, first_step_parity=True,
                audit_label_permutation_invariance=True, missing_not_evidence=True,
                duplicate_evidence_rejected=True, full_history_density_consistency=True,
                identical_predictions_identical_likelihood=True)
    save('checks.json',checks)
    print('SELFTEST', checks, flush=True)


def summarize(records):
    rows = [json.loads(s) for s in (OUT/'predictions.jsonl').read_text().splitlines()]
    expected = sum(len(list(base.queries(r))) for r in records if r['split']=='holdout')
    if len(rows) != expected or len({(r['case'],r['step'],r['tid']) for r in rows})!=expected:
        raise RuntimeError('incomplete or duplicate predictions')
    tables, contrasts, calibrations = [], [], []
    for subset in ('all','ordinary','conflict','unknown_goal','unknown_conflict','birth_known'):
        selected = [r for r in rows if (subset=='all' or
            subset=='ordinary' and not r['conflict'] or subset=='conflict' and r['conflict'] or
            subset=='unknown_goal' and not r['fixed'] or subset=='unknown_conflict' and not r['fixed'] and r['conflict'] or
            subset=='birth_known' and r['fixed'])]
        for metric in ('errors','energy'):
            arrays, groups = [], []
            for case in sorted({r['case'] for r in selected}):
                q = [r for r in selected if r['case']==case]
                arrays.append([[np.mean([r[metric][arm][h] for r in q]) for h in range(4)] for arm in ARMS])
                groups.append(q[0]['people'])
            if not arrays:
                continue
            a, g = np.array(arrays), np.array(groups)
            balanced = np.mean([a[g==v].mean(axis=0) for v in sorted(set(groups))],axis=0)
            for i,arm in enumerate(ARMS):
                tables.append(dict(subset=subset,metric=metric,arm=arm,values=balanced[i],
                    configurations={str(v):a[g==v,i].mean(axis=0) for v in sorted(set(groups))},episodes=len(groups),queries=len(selected)))
            for left,right in (('Interaction-FULL','C'),('Interaction-MAP','C'),('Interaction-FULL','CV'),
                               ('Interaction-FULL','Interaction-MAP'),('Interaction-FULL','Heading-FULL')):
                delta = a[:,ARMS.index(left)]-a[:,ARMS.index(right)]
                contrasts.append(dict(subset=subset,metric=metric,contrast=left+' - '+right,
                    delta=np.mean([delta[g==v].mean(axis=0) for v in sorted(set(groups))],axis=0),
                    mean_ci=base.interval(delta.mean(axis=1),groups),
                    improved_episodes=int(np.sum(delta.mean(axis=1)<0)),worse_episodes=int(np.sum(delta.mean(axis=1)>0))))
    unknown = [r for r in rows if not r['fixed']]
    cuts = np.quantile([r['spread']['Interaction-FULL'][3] for r in unknown],[0,.25,.5,.75,1])
    for i in range(4):
        q=[r for r in unknown if cuts[i]<=r['spread']['Interaction-FULL'][3] and
           (r['spread']['Interaction-FULL'][3]<=cuts[i+1] if i==3 else r['spread']['Interaction-FULL'][3]<cuts[i+1])]
        if q:
            calibrations.append(dict(bin=i,queries=len(q),mean_spread_m2=np.mean([r['spread']['Interaction-FULL'][3] for r in q]),
                                     mean_squared_error_m2=np.mean([r['errors']['Interaction-FULL'][3]**2 for r in q])))
    timings={str(n):dict(query_ms=np.quantile([r['query_ms'] for r in rows if r['people']==n],[.5,.95,.99]),
                        note='per target prediction incl 2 full posteriors and MAP; not complete MPC runtime') for n in (5,10,20)}
    result=dict(table=tables,contrasts=contrasts,spread_error=calibrations,timing=timings,
        queries=len(rows),episodes=len({r['case'] for r in rows}),
        distribution='goal-only conditional deterministic D modes for every arm; no residual kernel; ES is not full predictive calibration',
        status='prediction evidence only; no navigation or novelty claim')
    if (OUT/'summary.json').exists():
        previous=json.loads((OUT/'summary.json').read_text())
        for key in ('posthoc_goal_arrival','episode_wall_seconds','episode_wall_note','common_kernel_audit'):
            if key in previous:
                result[key]=previous[key]
    save('summary.json',result)
    lines=['# Continuous Goal Posterior: Fixed Evaluation', '',
        '12 development / 18 previously viewed evaluation episodes. No new navigation runs.',
        'Every arm uses the frozen D forward model except CV. Goal-only Energy Score; no arm receives a residual kernel.',
        'Configurations equal weight; confidence intervals cluster whole episodes.', '',
        '| Subset | Metric | Arm | 1s | 2s | 3s | 4s |','|---|---|---|---:|---:|---:|---:|']
    for r in tables:
        if r['subset'] in ('ordinary','conflict','unknown_conflict'):
            lines.append('| '+r['subset']+' | '+r['metric']+' | '+r['arm']+' | '+' | '.join(f'{v:.4f}' for v in r['values'])+' |')
    lines += ['', 'Full paired intervals, subgroup counts, spread/error diagnostics and timings: summary.json.',
        'No navigation integration. Particle approximation and misspecified independent residual likelihood remain limitations.']
    (OUT/'verdict.md').write_text('\n'.join(lines)+'\n')
    print('SUMMARY',json.dumps(result,default=encode),flush=True)


def audit_distribution(records):
    """Common conditional-error kernel; no posterior or point forecast changes."""
    residuals = {n: [] for n in (5,10,20)}
    for record in records:
        if record['split'] != 'development':
            continue
        for step, tid, f in base.queries(record):
            target = np.array([record['truth']['positions'][step+h][tid] for h in base.HORIZONS])
            p = base.predict(f, np.array(record['truth']['goals'][tid]), 'improved', .5)
            residuals[record['people']].append(target-p[np.array(base.HORIZONS)-1])
    kernels = {}
    for n, residual in residuals.items():
        e = np.array(residual)
        kernels[n] = np.einsum('nhi,nhj->hij',e,e)/len(e)+np.eye(2)[None]*1e-8
    rows = [json.loads(s) for s in (OUT/'predictions.jsonl').read_text().splitlines()]
    row_index = {(r['case'],r['step'],r['tid']):r for r in rows}
    contexts_by_key = {(r['case'],s,i):f for r in records if r['split']=='holdout'
                       for s,i,f in base.queries(r)}
    by_case = {r['case']:r for r in records}
    with gzip.open(OUT/'modes.jsonl.gz','rt') as stream:
        for line in stream:
            saved = json.loads(line)
            key = saved['case'],saved['step'],saved['tid']
            row, f, record = row_index[key],contexts_by_key[key],by_case[key[0]]
            truth = np.array([record['truth']['positions'][key[1]+h][key[2]] for h in base.HORIZONS])
            im,hm=saved['interaction'],saved['heading']
            modes={
                'Interaction-FULL':(np.array(im['positions']),np.array(im['weights'])),
                'Heading-FULL':(np.array(hm['positions']),np.array(hm['weights'])),
                'Interaction-MAP':(np.array(im['map_positions'])[None],np.ones(1))}
            for arm,g,mode in [('CV',f['goal'],'cv'),('C',f['goal'],'improved'),
                                ('D',np.array(record['truth']['goals'][key[2]]),'improved')]:
                modes[arm]=(base.predict(f,g,mode,.5)[None],np.ones(1))
            rng=np.random.default_rng(key[0]*100003+key[1]*97+key[2]+7301)
            row['energy_common_kernel']={arm:[] for arm in ARMS}
            for h,k in enumerate(np.array(base.HORIZONS)-1):
                # Independent pairs give an unbiased ES estimate; draws are shared across arms.
                ua,ub=rng.random((2,4096))
                chol=np.linalg.cholesky(kernels[row['people']][h])
                ea,eb=rng.normal(size=(2,4096,2)) @ chol.T
                for arm,(p,w) in modes.items():
                    cdf=np.cumsum(w);cdf[-1]=1.
                    xa=p[np.searchsorted(cdf,ua),k]+ea
                    xb=p[np.searchsorted(cdf,ub),k]+eb
                    es=np.mean(np.linalg.norm(xa-truth[h],axis=1))-.5*np.mean(np.linalg.norm(xa-xb,axis=1))
                    row['energy_common_kernel'][arm].append(float(es))
    tables,contrasts=[],[]
    for sub in ('all','ordinary','conflict','unknown_goal','unknown_conflict'):
        chosen=[r for r in rows if sub=='all' or sub=='ordinary' and not r['conflict'] or
                sub=='conflict' and r['conflict'] or sub=='unknown_goal' and not r['fixed'] or
                sub=='unknown_conflict' and not r['fixed'] and r['conflict']]
        arrays=[];groups=[]
        for case in sorted({r['case'] for r in chosen}):
            q=[r for r in chosen if r['case']==case]
            arrays.append([[np.mean([r['energy_common_kernel'][a][h] for r in q]) for h in range(4)] for a in ARMS])
            groups.append(q[0]['people'])
        a,g=np.array(arrays),np.array(groups)
        mean=np.mean([a[g==n].mean(axis=0) for n in sorted(set(groups))],axis=0)
        tables.extend(dict(subset=sub,arm=arm,values=mean[i]) for i,arm in enumerate(ARMS))
        for left,right in (('Interaction-FULL','Interaction-MAP'),('Interaction-FULL','Heading-FULL'),
                           ('Interaction-FULL','CV'),('Interaction-FULL','C')):
            d=a[:,ARMS.index(left)]-a[:,ARMS.index(right)]
            contrasts.append(dict(subset=sub,contrast=left+' - '+right,
                mean_delta=np.mean([d[g==n].mean() for n in sorted(set(groups))]),
                mean_ci=base.interval(d.mean(axis=1),groups)))
    summary=json.loads((OUT/'summary.json').read_text())
    summary['common_kernel_audit']=dict(table=tables,contrasts=contrasts,
        covariance={str(n):v for n,v in kernels.items()},development_queries={str(n):len(r) for n,r in residuals.items()},
        interpretation='same zero-mean D residual second-moment Gaussian for ALL arms, fitted only on development; marginal-horizon ES, not calibrated joint process',
        mc_pairs=4096,role='supplementary fairness audit; point predictions and posterior untouched')
    save('summary.json',summary)
    temp=OUT/'predictions.jsonl.tmp'
    with temp.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,default=encode,allow_nan=False)+'\n')
    temp.replace(OUT/'predictions.jsonl')
    checks=json.loads((OUT/'checks.json').read_text())
    checks['scores_sha256_before_kernel_audit']=checks['scores_sha256']
    checks['scores_sha256']=digest(OUT/'predictions.jsonl')
    checks['common_residual_kernel_development_only']=True
    save('checks.json',checks)
    print('COMMON_KERNEL',json.dumps(dict(table=tables,contrasts=contrasts),default=encode),flush=True)


DEPTH = OUT / 'depth'
DEPTH_STATES = ((2000000,16,3), (2000010,12,0), (2000020,24,0))


def depth_save(name, value):
    DEPTH.mkdir(parents=True,exist_ok=True)
    path=DEPTH/name
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,default=encode,allow_nan=False)+'\n')
    tmp.replace(path)


def depth_setup(records):
    import ast
    previous=ast.parse(subprocess.check_output(['git','show','f77c533:experiments/goal_posterior_probe.py'],cwd=ROOT))
    current=ast.parse(Path(__file__).read_text())
    for name in ('GoalPosterior','infer','fit','forward','contexts','evaluate'):
        assert ast.dump(next(n for n in previous.body if getattr(n,'name',None)==name))==ast.dump(next(n for n in current.body if getattr(n,'name',None)==name))
    rows=[json.loads(x) for x in (OUT/'predictions.jsonl').read_text().splitlines()]
    states=[]
    for n in (5,10,20):
        choices=sorted({(r['case'],r['step']) for r in rows if r['people']==n and r['conflict'] and not r['fixed']})
        for i in (len(choices)//3,2*len(choices)//3):
            states.append(list(choices[i]))
    protocol=dict(baseline='f77c533',reference_states=DEPTH_STATES,action_states=states,
        grids=[32,64,128,256],shifted_grid=[256,.25,.75],
        reference_check=dict(log_normalizer_absolute=.02,binned_TV=.05,future_mean_max_m=.02),
        criterion='both 128-to-256 and independent shifted-256 comparisons; practical numerical check, not rigorous integration error bound',
        forecast_omitted_mass=1e-6,bin_count=32,posterior_and_model_unchanged=True,
        action_scope='CV vs MAP mean vs FULL mean vs D mean; only visible unknown-goal targets replaced; covariance fixed; NOT multimodal risk and NOT closed-loop benefit',
        action_replay='rebuild old environment with recorded controls; verify all states/detections and old MPC controls; common pre-query warm start',
        files={str(Path(__file__)):digest(__file__),str(base.__file__):digest(base.__file__),
               str(OUT/'calibration.json'):digest(OUT/'calibration.json'),
               str(OUT/'modes.jsonl.gz'):digest(OUT/'modes.jsonl.gz'),
               str(OUT/'predictions.jsonl'):digest(OUT/'predictions.jsonl')})
    if (DEPTH/'protocol.json').exists():
        saved=json.loads((DEPTH/'protocol.json').read_text())
        for path,h in saved['files'].items():
            if digest(path)!=h:raise RuntimeError('depth frozen file changed: '+path)
        return saved
    depth_save('protocol.json',protocol)
    return protocol


def posterior_at(record, step, tid, calibration, offset=0):
    model=None
    last=None
    for s,features in contexts(legal_record(record)):
        if s>step:break
        if tid not in features:continue
        last=features[tid]
        if model is None:
            model=GoalPosterior(record['scene'],s,last,calibration,record['case']*1009+tid*97+offset)
        else:model.update(s,last)
    if model is None or model.previous[0]!=step:raise RuntimeError('target not observed')
    return model,last


def posterior_hist(goals, weights):
    return np.histogram2d(goals[:,0],goals[:,1],bins=32,range=[[-5,5],[-5,5]],weights=weights)[0]


def grid_reference(model, features, size, offset=(.5,.5)):
    began=time.perf_counter()
    x=-5.+(np.arange(size)+offset[0])*10./size
    y=-5.+(np.arange(size)+offset[1])*10./size
    gx,gy=np.meshgrid(x,y,indexing='ij')
    goals=np.stack([gx.ravel(),gy.ravel()],axis=1)
    goals=goals[model.valid(goals)]
    logdensity=model.logposterior(goals)
    normalizer=logsumexp(logdensity)
    weights=np.exp(logdensity-normalizer)
    goal_mean=weights@goals
    covariance=np.einsum('n,ni,nj->ij',weights,goals-goal_mean,goals-goal_mean)
    order=np.argsort(-weights,kind='stable')
    kept=order[:min(len(order),np.searchsorted(np.cumsum(weights[order]),1.-1e-6)+1)]
    omitted=max(0.,1.-float(weights[kept].sum()))
    positions=forward(features,goals[kept],16)
    w=weights[kept]/weights[kept].sum()
    future_mean=np.einsum('n,nhd->hd',w,positions)
    hist=posterior_hist(goals,weights)
    summary=dict(size=size,offset=offset,nodes=len(goals),forecast_nodes=len(kept),
        log_normalizer=float(normalizer+2*np.log(10./size)),goal_mean=goal_mean,
        goal_covariance=covariance,future_mean=future_mean,histogram=hist,
        omitted_mass=omitted,forecast_mean_truncation_bound_m=8.*omitted,
        discrete_KL_from_uniform_nodes=float(np.sum(weights*(logdensity-normalizer+np.log(len(goals))))),
        seconds=time.perf_counter()-began)
    return summary,dict(goals=goals,weights=weights,logdensity=logdensity,
                        kept=kept,positions=positions,kept_weights=w)


def reference_difference(a,b):
    return dict(log_normalizer=abs(a['log_normalizer']-b['log_normalizer']),
        binned_TV=float(.5*np.abs(np.array(a['histogram'])-np.array(b['histogram'])).sum()),
        future_mean_max_m=float(np.linalg.norm(np.array(a['future_mean'])-np.array(b['future_mean']),axis=1).max()))


def depth_reference(records, calibration, protocol):
    results=[]
    for case,step,tid in DEPTH_STATES:
        r=next(r for r in records if r['case']==case)
        model,features=posterior_at(r,step,tid,calibration)
        summaries=[]
        retained=None
        for n,offset in [(n,(.5,.5)) for n in protocol['grids']]+[(256,(.25,.75))]:
            s,details=grid_reference(model,features,n,offset)
            summaries.append(s)
            if n==256 and offset==(.5,.5):retained=details
            print('GRID',case,n,offset,'nodes',s['nodes'],'seconds',round(s['seconds'],2),
                  'logZ',round(s['log_normalizer'],5),'mean',s['goal_mean'].tolist(),flush=True)
        differences=[reference_difference(summaries[-3],summaries[-2]),reference_difference(summaries[-2],summaries[-1])]
        passed=all(d['log_normalizer']<=.02 and d['binned_TV']<=.05 and d['future_mean_max_m']<=.02 for d in differences)
        ref=summaries[-2]
        truth_goal=np.array(r['truth']['goals'][tid])
        truth_ll=float(model.logposterior(truth_goal)[0])
        particle=[]
        for seed in (0,104729,209458):
            m,f=posterior_at(r,step,tid,calibration,seed)
            p=forward(f,m.goals,16)
            w=np.exp(m.logweights)
            mean=np.einsum('n,nhd->hd',w,p)
            particle.append(dict(seed_offset=seed,goal_mean=w@m.goals,
                binned_TV_to_reference=float(.5*np.abs(posterior_hist(m.goals,w)-np.array(ref['histogram'])).sum()),
                future_mean_difference_m=np.linalg.norm(mean-np.array(ref['future_mean']),axis=1),
                ess=float(1./np.sum(w*w)),unique_goals=len(np.unique(m.goals,axis=0))))
        rng=np.random.default_rng(case+step+tid)
        refmean=np.array(ref['future_mean'])
        iid=[]
        for _ in range(200):
            ids=rng.choice(len(retained['kept']),128,replace=True,p=retained['kept_weights'])
            sample=retained['positions'][ids].mean(axis=0)
            iid.append(np.linalg.norm(sample-refmean,axis=1))
        future_truth=np.array(r['truth']['positions'])[step+1:step+17,tid]
        item=dict(case=case,step=step,tid=tid,updates=model.updates,grids=summaries,
            refinement_differences=differences,reference_stable_under_registered_check=passed,
            particles=particle,iid128_future_difference_p95=np.quantile(iid,.95,axis=0),
            true_goal=truth_goal,true_goal_log_likelihood=truth_ll,
            reference_max_node_log_likelihood=float(retained['logdensity'].max()),
            posterior_mass_higher_density_than_true=float(retained['weights'][retained['logdensity']>truth_ll].sum()),
            reference_future_endpoint_errors=np.linalg.norm(refmean-future_truth,axis=1)[np.array(base.HORIZONS)-1])
        results.append(item)
        depth_save('reference.json',results)
        np.savez_compressed(DEPTH/f'reference_{case}.npz',**retained)
        print('REFERENCE',case,'stable',passed,'differences',differences,flush=True)


def depth_refine(records, calibration):
    results=json.loads((DEPTH/'reference.json').read_text())
    for item in results:
        if item['reference_stable_under_registered_check']:continue
        case,step,tid=item['case'],item['step'],item['tid']
        record=next(r for r in records if r['case']==case)
        model,f=posterior_at(record,step,tid,calibration)
        s,details=grid_reference(model,f,512)
        differences=[reference_difference(item['grids'][-2],s),reference_difference(item['grids'][-1],s)]
        passed=all(d['log_normalizer']<=.02 and d['binned_TV']<=.05 and d['future_mean_max_m']<=.02 for d in differences)
        particle=[]
        for seed in (0,104729,209458):
            m,features=posterior_at(record,step,tid,calibration,seed)
            w=np.exp(m.logweights);p=forward(features,m.goals,16)
            mean=np.einsum('n,nhd->hd',w,p)
            particle.append(dict(seed_offset=seed,future_mean_difference_m=np.linalg.norm(mean-s['future_mean'],axis=1)))
        rng=np.random.default_rng(case+step+tid);iid=[]
        for _ in range(200):
            ids=rng.choice(len(details['kept']),128,replace=True,p=details['kept_weights'])
            iid.append(np.linalg.norm(details['positions'][ids].mean(axis=0)-s['future_mean'],axis=1))
        true_ll=float(model.logposterior(np.array(record['truth']['goals'][tid]))[0])
        item['refinement_512']=dict(grid=s,differences_to_both_256_grids=differences,
            stable_under_extended_check=passed,particles=particle,iid128_future_difference_p95=np.quantile(iid,.95,axis=0),
            posterior_mass_higher_density_than_true=float(details['weights'][details['logdensity']>true_ll].sum()))
        np.savez_compressed(DEPTH/f'reference_{case}_refined.npz',**details)
        depth_save('reference.json',results)
        print('REFINED',case,'seconds',s['seconds'],'stable',passed,'differences',differences,flush=True)


def depth_likelihood(records, calibration):
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.state import FullState,ObservableState,JointState
    _,_,ActionXY,_,_=base.legacy._load_modules(base.CROWD)
    results=[]
    for case,step,tid in DEPTH_STATES:
        record=next(r for r in records if r['case']==case)
        feature_map={s:f for s,group in contexts(legal_record(record)) for i,f in group.items() if i==tid}
        observed={fr['step']:{e['id']:e for e in fr['detections']} for fr in record['frames']}
        env=base.environment(record);rows=[]
        for frame in record['frames']:
            s=frame['step']
            if s>=step:break
            np.testing.assert_allclose([[h.px,h.py] for h in env.humans],record['truth']['positions'][s],atol=1e-12,rtol=0)
            if s in feature_map and s+1 in feature_map:
                f=feature_map[s];target=feature_map[s+1]['pos'];human=env.humans[tid]
                goal=np.array(record['truth']['goals'][tid])
                predictions={'frozen_D':forward(f,[goal])[0,0]}
                for name,oracle_memory,full in (('no_extra_response_clip',False,False),
                        ('oracle_memory_legal_neighbors',True,False),('oracle_memory_full_neighbors',True,True)):
                    policy=ORCA()
                    policy._last_pref_vel=copy.deepcopy(human.policy._last_pref_vel) if oracle_memory else f['vel'].copy()
                    neighbors=([h.get_observable_state() for i,h in enumerate(env.humans) if i!=tid] if full else
                        [ObservableState(e['px'],e['py'],e['vx'],e['vy'],e['radius']) for e in f['neighbors']])
                    state=JointState(FullState(*f['pos'],*f['vel'],f['radius'],*goal,1.,0.),neighbors)
                    predictions[name]=f['pos']+.25*np.array(policy.predict(state))
                errors={name:float(np.linalg.norm(p-target)) for name,p in predictions.items()}
                if errors['oracle_memory_full_neighbors']>1e-6:raise RuntimeError('full behavior oracle does not reproduce next observation')
                rows.append(dict(step=s,legal_neighbors=len(f['neighbors']),errors_m=errors))
            env.step(ActionXY(*map(float,frame['action'])))
        results.append(dict(case=case,step=step,tid=tid,transitions=rows,
            rms_m={name:float(np.sqrt(np.mean([r['errors_m'][name]**2 for r in rows]))) for name in rows[0]['errors_m']},
            note='true-goal diagnostics ONLY; oracle memory/full neighbors prohibited from inference; no frozen model changed'))
        print('LIKELIHOOD_MODEL',case,results[-1]['rms_m'],flush=True)
    depth_save('likelihood_model.json',results)


def depth_profile(records, calibration):
    results=[]
    for case,step,tid in DEPTH_STATES:
        record=next(r for r in records if r['case']==case)
        models={}
        for s,features in contexts(legal_record(record)):
            if s>step:break
            current=[]
            for identifier,f in features.items():
                start=time.perf_counter()
                if identifier not in models:
                    models[identifier]={kind:GoalPosterior(record['scene'],s,f,calibration,case*1009+identifier*97,kind=kind)
                                        for kind in ('interaction','heading')}
                    im_ms=hd_ms=0.
                else:
                    models[identifier]['interaction'].update(s,f)
                    im_ms=1000*(time.perf_counter()-start)
                    start=time.perf_counter();models[identifier]['heading'].update(s,f)
                    hd_ms=1000*(time.perf_counter()-start)
                if s==step:
                    im,hd=models[identifier]['interaction'],models[identifier]['heading']
                    start=time.perf_counter();forward(f,im.goals,16);full_ms=1000*(time.perf_counter()-start)
                    start=time.perf_counter();g=im.map_goal();forward(f,[g],16);map_ms=1000*(time.perf_counter()-start)
                    start=time.perf_counter();forward(f,hd.goals,16);heading_ms=1000*(time.perf_counter()-start)
                    current.append(dict(tid=identifier,fixed=im.fixed,interaction_update_ms=im_ms,
                        full_rollout_ms=full_ms,map_search_and_rollout_ms=map_ms,
                        heading_update_ms=hd_ms,heading_rollout_ms=heading_ms))
            if s==step:
                item=dict(case=case,step=step,visible=len(current),targets=current,
                    FULL_only_ms=sum(x['interaction_update_ms']+x['full_rollout_ms'] for x in current),
                    MAP_only_ms=sum(x['interaction_update_ms']+x['map_search_and_rollout_ms'] for x in current),
                    note='one current observation update plus all visible target forecasts; excludes MPC and observation/filter processing; serial measurements, no extrapolation')
                results.append(item);print('PROFILE',case,item['FULL_only_ms'],item['MAP_only_ms'],flush=True)
    depth_save('profile.json',results)


def depth_actions(records, protocol):
    from nav.contracts import MPCConfig
    from nav.planner import MPCPlanner
    from integration.crowdnav import BayesObservationAdapter
    selected={tuple(x) for x in protocol['action_states']}
    modes={}
    with gzip.open(OUT/'modes.jsonl.gz','rt') as stream:
        for line in stream:
            m=json.loads(line)
            if (m['case'],m['step']) in selected:
                modes[m['case'],m['step'],m['tid']]=m
    cfg=MPCConfig(**json.loads((base.OUT/'protocol.json').read_text())['config'])
    _,_,ActionXY,_,_=base.legacy._load_modules(base.CROWD)
    results=[];checked_steps=0
    for record in records:
        steps={step for case,step in selected if case==record['case']}
        if not steps:continue
        env=base.environment(record);adapter=BayesObservationAdapter(cfg);planner=MPCPlanner(cfg)
        features={(s,i):f for s,i,f in base.queries(record)}
        for frame in record['frames']:
            step=frame['step']
            if step>max(steps):break
            human=np.array([[h.px,h.py] for h in env.humans])
            np.testing.assert_allclose(human,record['truth']['positions'][step],atol=1e-12,rtol=0)
            np.testing.assert_allclose([env.robot.px,env.robot.py,env.robot.vx,env.robot.vy,env.robot.radius],frame['robot'],atol=1e-12,rtol=0)
            obs=adapter.read(env)
            if adapter.detected_entities!=frame['detections']:raise RuntimeError('replayed legal detection mismatch')
            before=copy.deepcopy(planner._previous_mean)
            seed=record['case']*100003+step*97+1729
            action,_=planner.plan(obs,seed)
            if not np.array_equal(action,frame['action']):raise RuntimeError('original MPC action not reproduced')
            checked_steps+=1
            if step in steps:
                row=dict(case=record['case'],step=step,people=record['people'],arms={
                    'CV':dict(action=action,diagnostics=planner.last_diagnostics.copy())})
                for arm in ('CV-repeat','MAP','FULL-mean','D-mean'):
                    shadow=MPCPlanner(cfg);shadow._previous_mean=copy.deepcopy(before)
                    modified=copy.deepcopy(obs);changed=[]
                    for j,identifier in enumerate(adapter.reported_ids):
                        key=record['case'],step,identifier
                        if arm=='CV-repeat' or key not in modes or modes[key]['interaction']['fixed']:continue
                        m=modes[key]['interaction']
                        if arm=='MAP':p=np.array(m['map_positions'])
                        elif arm=='FULL-mean':p=np.einsum('n,nhd->hd',m['weights'],m['positions'])
                        else:p=base.predict(features[step,identifier],np.array(record['truth']['goals'][identifier]),'improved',.5)
                        modified.human_segment_end[j]=p
                        modified.human_segment_start[j]=np.concatenate([modified.entities[j:j+1,:2],p[:-1]])
                        changed.append(identifier)
                    alternative,ms=shadow.plan(modified,seed)
                    if arm=='CV-repeat' and not np.array_equal(alternative,action):raise RuntimeError('same-input planning not deterministic')
                    row['arms'][arm]=dict(action=alternative,delta_speed_m_s=float(np.linalg.norm(alternative-action)),
                        delta_one_step_position_m=float(.25*np.linalg.norm(alternative-action)),
                        changed_targets=changed,plan_ms=ms,diagnostics=shadow.last_diagnostics.copy())
                results.append(row);print('ACTION',record['case'],step,{a:v.get('delta_speed_m_s',0) for a,v in row['arms'].items()},flush=True)
            env.step(ActionXY(*map(float,frame['action'])))
    assert len(results)==len(selected)
    depth_save('actions.json',dict(states=results,original_steps_bitwise_reproduced=checked_steps,
        limitation='mean-channel sensitivity only; no mixture covariance, no posterior-risk integration, no outcome comparison; non-change cannot reject full distribution value'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=('prepare','run','summarize','stability','audit-distribution',
        'depth-prepare','depth-reference','depth-profile','depth-actions','depth-refine','depth-likelihood'))
    args=ap.parse_args()
    records=data()
    if args.command.startswith('depth-'):
        protocol=depth_setup(records)
        calibration=json.loads((OUT/'calibration.json').read_text())
        if args.command=='depth-reference':depth_reference(records,calibration,protocol)
        elif args.command=='depth-profile':depth_profile(records,calibration)
        elif args.command=='depth-actions':depth_actions(records,protocol)
        elif args.command=='depth-refine':depth_refine(records,calibration)
        elif args.command=='depth-likelihood':depth_likelihood(records,calibration)
        return
    if args.command=='prepare':
        if (OUT/'protocol.json').exists():
            raise RuntimeError('already frozen; do not overwrite protocol')
        calibration=fit(records)
        selftest(records,calibration)
        save('protocol.json',dict(baseline=BASELINE,particles=128,acceleration=.5,dt=.25,
            data_sha=digest(base.OUT/'episodes.jsonl'),code_sha=digest(__file__),model_sha=digest(base.__file__),
            prior='uniform area annulus 4 +/- sqrt(.5) for circle; square [-5,5]^2, birth side only when frame0 seen; public superset, not exact rejection-conditioned generator prior',
            mh='one symmetric Gaussian .35m move after ESS<N/2; full used history; outside rejected',
            map='top two particle densities, Nelder-Mead max48 evaluations per start; approximate MAP',
            likelihood='Student-t df4 displacement; heading isotropic Gaussian preferred-velocity residual in displacement units',
            evaluation='same 1200 queries, 18 already viewed episodes; no new confirmation claim',
            energy='exact weighted discrete goal-only mixture; deterministic D conditional for ALL arms',
            seed='case*1009+id*97',no_navigation=True))
        return
    p=json.loads((OUT/'protocol.json').read_text())
    if p['code_sha']!=digest(__file__) or p['model_sha']!=digest(base.__file__):
        raise RuntimeError('frozen experimental source changed')
    calibration=json.loads((OUT/'calibration.json').read_text())
    if args.command=='audit-distribution':
        audit_distribution(records)
    elif args.command=='run':
        if (OUT/'predictions.jsonl').exists():
            raise RuntimeError('results exist; refusing duplicate run')
        with gzip.open(OUT/'modes.jsonl.gz','wt',compresslevel=1) as modes, (OUT/'predictions.jsonl').open('w') as scores:
            for record in records:
                if record['split']!='holdout':continue
                start=time.perf_counter()
                rows=evaluate(record,calibration,modes)
                for row in rows:scores.write(json.dumps(row,default=encode,allow_nan=False)+'\n')
                scores.flush();modes.flush()
                print('EPISODE',record['case'],record['people'],'queries',len(rows),'seconds',round(time.perf_counter()-start,2),flush=True)
        summarize(records)
    elif args.command=='summarize':
        summarize(records)
    else:
        out=[]
        for record in [next(r for r in records if r['split']=='development' and r['people']==n) for n in (5,10,20)]:
            qs=list(base.queries(record))
            selected=qs[len(qs)//2]
            keys={(selected[0],selected[1])}
            legal=legal_record(record);legal['frames']=[f for f in legal['frames'] if f['step']<=selected[0]]
            for n in (64,128,256):
                start=time.perf_counter()
                result=list(infer(legal,calibration,n,keys,include_heading=False))[0]
                snapshot=result['interaction']
                truth=np.array([record['truth']['positions'][selected[0]+h][selected[1]] for h in base.HORIZONS])
                errors,es,spread=score_modes(snapshot['positions'],snapshot['weights'],truth)
                out.append(dict(case=record['case'],step=selected[0],tid=selected[1],particles=n,
                    seconds=time.perf_counter()-start,errors=errors,energy=es,updates=snapshot['updates']))
                print('STABILITY',out[-1],flush=True)
        save('stability.json',out)


if __name__=='__main__':
    main()
