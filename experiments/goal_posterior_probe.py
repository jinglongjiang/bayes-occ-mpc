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


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=('prepare','run','summarize','stability','audit-distribution'))
    args=ap.parse_args()
    records=data()
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
