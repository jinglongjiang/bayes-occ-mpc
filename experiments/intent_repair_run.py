"""Isolated end-to-end intention repair experiment; production modules unchanged."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import ncx2, multivariate_normal
from scipy.spatial.distance import cdist
from experiments import goal_posterior_probe as old
from experiments import goal_model_probe as base
from nav.contracts import MPCConfig
from nav.planner import MPCPlanner
from nav.risk import CompiledDiscRisk, exact_hazard
from integration.crowdnav import BayesObservationAdapter

OUT = ROOT / 'results/intent_repair'
H = np.array(base.HORIZONS) - 1
ARMS = ('CV', 'OldFULL', 'WideFULL', 'NewMAP', 'NewFULL', 'D')
NAV_ARMS = ('O', 'B', 'F0', 'M', 'F')


def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=old.encode, allow_nan=False) + '\n')
    tmp.replace(p)


def read(name):
    return json.loads((OUT / name).read_text())


def append(name, value):
    with (OUT / name).open('a') as f:
        f.write(json.dumps(value, default=old.encode, allow_nan=False) + '\n')


def log(*args):
    print(*args, flush=True)


def setup(records):
    if (OUT / 'protocol.json').exists():
        p = read('protocol.json')
        for path, sha in p['frozen_dependencies'].items():
            if old.digest(path) != sha:
                raise RuntimeError('frozen dependency changed: ' + path)
        return p
    dev = [r for r in records if r['split'] == 'development']
    anchors = {a[0]: a for a in old.DEPTH_STATES}
    states = []
    for r in dev:
        queries = [(r['case'], s, i) for s, i, f in base.queries(r)
                   if f['goal_source'] != 'birth_antipode' and s > 1]
        states.append(list(anchors.get(r['case'], queries[len(queries)//2])))
    fit_cases = [r['case'] for n in (5, 10, 20)
                 for r in [x for x in dev if x['people'] == n][:2]]
    paths = list((ROOT/'nav').glob('*.py')) + [ROOT/'integration/crowdnav.py',
        Path(old.__file__), Path(base.__file__), base.OUT/'episodes.jsonl',
        old.OUT/'calibration.json']
    p = dict(baseline='fecf501', states=states, fit_cases=fit_cases,
        validation_cases=[r['case'] for r in dev if r['case'] not in fit_cases],
        tau_grid=[.25, .5, 1.], fraction_grid=[.25, .5, .75],
        development_support='256 common seeded uniform prior goals per track; cached D residuals; selection approximation explicitly audited later',
        initial_particles=128, maximum_particles=256, maximum_mh=2,
        reference_tolerances=dict(binned_TV=.05, predictive_mean_m=.05),
        compression_grid=[4, 8, 16], compression_limits=dict(mean_m=.05, ES_m=.01, risk=.01),
        config=read_config(), covariance='development D residual isotropic second moment at all 16 steps; visible unknown goals only; hidden and known-birth tracks retain original covariance',
        geometry='same legal CV track for all arms; only probability channel changes',
        selection='unknown-goal conflict common-kernel Energy Score on six development-validation episodes; ties 1e-5 prefer shorter tau then smaller fraction',
        no_wall_clock_cutoff=True, max_navigation_cases=18,
        frozen_dependencies={str(p): old.digest(p) for p in paths},
        created=time.time(), source_revisions=[], stages={})
    save('protocol.json', p)
    return p


def read_config():
    return json.loads((base.OUT/'protocol.json').read_text())['config']


def gaussian_step(residual, mean, covariance, sigma, fraction, rho, gap=1):
    """Predictive evidence BEFORE conditioning; stationary AR nuisance prior."""
    v, r = fraction*sigma, (1-fraction)*sigma
    phi = rho**gap
    pm = phi*mean
    pc = phi*phi*covariance + (1-phi*phi)*v
    innovation_cov = pc+r
    inverse = np.linalg.inv(innovation_cov)
    innovation = residual-pm
    ll = -.5*(2*np.log(2*np.pi)+np.linalg.slogdet(innovation_cov)[1]
               + np.einsum('ni,ij,nj->n', innovation, inverse, innovation))
    gain = pc @ inverse
    updated = pm + innovation @ gain.T
    # Joseph form retains positive semidefiniteness after many observations.
    identity = np.eye(2)-gain
    covariance = identity @ pc @ identity.T + gain @ r @ gain.T
    return ll, updated, covariance


class RepairedPosterior(old.GoalPosterior):
    def __init__(self, scene, step, features, calibration, seed, sigma,
                 fraction=.5, tau=.5, count=128, mh=1, wide=False):
        super().__init__(scene, step, features, calibration, seed, count)
        self.sigma_matrix = np.array(sigma)
        self.fraction = 0. if wide else fraction
        self.rho = np.exp(-.25/tau)
        self.bias = np.zeros_like(self.goals)
        self.bias_covariance = self.fraction*self.sigma_matrix
        self.bias_step = step
        self.mh_steps = mh
        self._density_cache = {}

    def replay(self, goals):
        goals = np.asarray(goals).reshape(-1, 2)
        valid = self.valid(goals)
        density = np.full(len(goals), -np.inf)
        means = np.zeros_like(goals)
        covariance = self.fraction*self.sigma_matrix
        density[valid] = 0.
        if not valid.any():
            return density, means, covariance
        last = None
        for step, f, observed in self.history:
            residual = observed-(old.forward(f, goals[valid])[:, 0]-f['pos'])
            ll, m, covariance = gaussian_step(residual, means[valid], covariance,
                self.sigma_matrix, self.fraction, self.rho, 1 if last is None else step-last)
            density[valid] += ll
            means[valid] = m
            last = step
        return density, means, covariance

    def logposterior(self, goals):
        goals=np.asarray(goals).reshape(-1,2)
        # MAP's repeated identical trials need not replay the same ORCA history twice.
        if len(goals)==1:
            key=tuple(goals[0])
            if key not in self._density_cache:
                self._density_cache[key]=float(self.replay(goals)[0][0])
            return np.array([self._density_cache[key]])
        return self.replay(goals)[0]

    def update(self, step, features):
        started = time.perf_counter()
        self._density_cache.clear()
        previous_step, previous = self.previous
        if step <= previous_step:
            raise ValueError('duplicate observation')
        self.previous = (step, features)
        if self.fixed:
            return
        if step != previous_step+1:
            # Propagation is deferred until the next valid innovation; no fabricated evidence.
            return
        residual = features['pos']-previous['pos']-(old.forward(previous, self.goals)[:, 0]-previous['pos'])
        ll, self.bias, self.bias_covariance = gaussian_step(residual, self.bias,
            self.bias_covariance, self.sigma_matrix, self.fraction, self.rho, step-self.bias_step)
        self.bias_step = step
        self.history.append((step, previous, features['pos']-previous['pos']))
        self.updates += 1
        self.density += ll
        self.logweights += ll
        self.logweights -= logsumexp(self.logweights)
        if not np.isfinite(self.logweights).all():
            raise FloatingPointError('invalid goal weights')
        w = np.exp(self.logweights)
        self.ess_before = float(1/(w@w))
        if self.ess_before < self.count/2:
            indices = np.searchsorted(np.cumsum(w), (self.rng.random()+np.arange(self.count))/self.count)
            indices = np.minimum(indices, len(w)-1)
            self.goals = self.goals[indices].copy()
            self.density = self.density[indices].copy()
            self.bias = self.bias[indices].copy()
            self.logweights[:] = -np.log(self.count)
            self.resamples += 1
            for _ in range(self.mh_steps):
                proposed_goals = self.goals+self.rng.normal(0., .35, self.goals.shape)
                proposed, bias, _ = self.replay(proposed_goals)
                accept = np.log(self.rng.random(self.count)) < proposed-self.density
                self.goals[accept] = proposed_goals[accept]
                self.density[accept] = proposed[accept]
                self.bias[accept] = bias[accept]
                self.moves += self.count
                self.accepts += int(accept.sum())
        self.seconds += time.perf_counter()-started


def selftest():
    rng = np.random.default_rng(5021)
    sigma = np.array([[.012, .002], [.002, .009]])
    checks = []
    for fraction, rho in ((.5,.6), (0.,.6), (.75,.8)):
        residual = rng.normal(0, .1, (5,2))
        mean = np.zeros((1,2)); cov = fraction*sigma; total = 0.
        times = [0,1,4,5,9]
        for k, e in enumerate(residual):
            ll,mean,cov = gaussian_step(e[None],mean,cov,sigma,fraction,rho,
                                       1 if k==0 else times[k]-times[k-1])
            total += ll[0]
        joint = np.block([[fraction*sigma*rho**abs(i-j)+(1-fraction)*sigma*(i==j)
                           for j in times] for i in times])
        np.testing.assert_allclose(total,multivariate_normal.logpdf(residual.ravel(),cov=joint),atol=1e-10)
        checks.append(dict(fraction=fraction,rho=rho,sequential_batch_equal=True))
    save('kernel_checks.json',checks)


def model_at(record, step, tid, calibration, settings, seed=0):
    model = None
    for s, features in old.contexts(old.legal_record(record)):
        if s > step:
            break
        if tid not in features:
            continue
        f = features[tid]
        if model is None:
            model = RepairedPosterior(record['scene'],s,f,calibration,
                record['case']*1009+tid*97+seed,**settings)
        else:
            model.update(s,f)
    if model is None or model.previous[0] != step:
        raise RuntimeError('invalid query')
    # Check SMC ancestry and MH terminal states against independent history replay.
    d,b,c = model.replay(model.goals)
    np.testing.assert_allclose(d,model.density,atol=1e-8,rtol=0)
    np.testing.assert_allclose(b,model.bias,atol=1e-10,rtol=0)
    np.testing.assert_allclose(c,model.bias_covariance,atol=1e-10,rtol=0)
    return model,f


def factorial(records, protocol):
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.state import FullState,ObservableState,JointState
    _,_,ActionXY,_,_ = base.legacy._load_modules(base.CROWD)
    output = []
    for case,step,tid in protocol['states']:
        record = next(r for r in records if r['case']==case)
        fmap = {s:fs[tid] for s,fs in old.contexts(old.legal_record(record)) if tid in fs}
        env = base.environment(record); rows=[]
        for frame in record['frames']:
            s = frame['step']
            if s >= step:
                break
            if s in fmap and s+1 in fmap:
                f = fmap[s]; target=fmap[s+1]['pos']; human=env.humans[tid]
                values={}
                for memory in (0,1):
                    for full in (0,1):
                        policy=ORCA()
                        policy._last_pref_vel=copy.deepcopy(human.policy._last_pref_vel) if memory else f['vel'].copy()
                        neighbors=([h.get_observable_state() for i,h in enumerate(env.humans) if i!=tid] if full else
                            [ObservableState(e['px'],e['py'],e['vx'],e['vy'],e['radius']) for e in f['neighbors']])
                        state=JointState(FullState(*f['pos'],*f['vel'],f['radius'],*record['truth']['goals'][tid],1.,0.),neighbors)
                        p=f['pos']+.25*np.array(policy.predict(state))
                        values[f'E{memory}{full}']=float(np.linalg.norm(p-target))
                if values['E11']>1e-6:
                    raise RuntimeError('full oracle replay failed')
                rows.append(dict(step=s,errors=values))
            env.step(ActionXY(*map(float,frame['action'])))
        means={k:float(np.mean([r['errors'][k] for r in rows])) for k in ('E00','E01','E10','E11')}
        output.append(dict(case=case,step=step,tid=tid,transitions=rows,mean=means,
            neighbor_approx=means['E01']-means['E00'], neighbor_true=means['E11']-means['E10'],
            memory_legal=means['E10']-means['E00'], interaction=means['E11']-means['E10']-means['E01']+means['E00']))
        save('factorial.json',output)
        log('FACTORIAL',case,means)


def common_score(positions,weights,truth,variance,seed,pairs=1024):
    rng=np.random.default_rng(seed)
    means=np.einsum('n,nhd->hd',weights,positions)
    errors=np.linalg.norm(means[H]-truth,axis=1)
    energies=[]
    for j,k in enumerate(H):
        ua,ub=rng.random((2,pairs)); noise=rng.normal(size=(2,pairs,2))*np.sqrt(variance[k])
        cumulative=np.cumsum(weights);cumulative[-1]=1.
        a=positions[np.searchsorted(cumulative,ua),k]+noise[0]
        b=positions[np.searchsorted(cumulative,ub),k]+noise[1]
        energies.append(float(np.linalg.norm(a-truth[j],axis=1).mean()-.5*np.linalg.norm(a-b,axis=1).mean()))
    return errors,energies


def fit(records,protocol):
    residuals=[];future={n:[] for n in (5,10,20)}
    for record in records:
        if record['split']!='development':
            continue
        previous={};last=-2
        for s,features in old.contexts(old.legal_record(record)):
            if record['case'] in protocol['fit_cases'] and s==last+1:
                for tid in previous.keys() & features.keys():
                    f=previous[tid];g=record['truth']['goals'][tid]
                    residuals.append(features[tid]['pos']-old.forward(f,[g])[0,0])
            previous,last=features,s
        for s,tid,f in base.queries(record):
            d=base.predict(f,np.array(record['truth']['goals'][tid]),'improved',.5)
            truth=np.array(record['truth']['positions'])[s+1:s+17,tid]
            future[record['people']].append(truth-d)
    e=np.array(residuals);sigma=e.T@e/len(e)
    sigma=.9*sigma+.1*np.eye(2)*np.trace(sigma)/2+np.eye(2)*1e-8
    variance={str(n):np.maximum(np.mean(np.array(v)**2,axis=(0,2)),1e-8) for n,v in future.items()}
    save('calibration.json',dict(sigma=sigma,variance=variance,residuals=len(e),fit_cases=protocol['fit_cases'],
        future_kernel_cases=[r['case'] for r in records if r['split']=='development']))
    log('FIT',sigma.tolist())


def prior_support(model,n=256):
    u=np.random.default_rng(817).random((n,2))
    if model.scene=='circle_crossing':
        a=-np.pi+2*np.pi*u[:,0];lo,hi=4-np.sqrt(.5),4+np.sqrt(.5)
        r=np.sqrt(lo*lo+u[:,1]*(hi*hi-lo*lo))
        return np.stack([r*np.cos(a),r*np.sin(a)],axis=1)
    g=-5+10*u
    if model.side:
        g[:,0]=model.side*np.abs(g[:,0])
    return g


def select(records,protocol,calibration):
    fitted=read('calibration.json');sigma=np.array(fitted['sigma'])
    choices=[(tau,f) for tau in protocol['tau_grid'] for f in protocol['fraction_grid']]
    cachepath=OUT/'selection_cache.jsonl.gz'
    if not cachepath.exists():
        with gzip.open(str(cachepath)+'.tmp','wt') as stream:
            for record in records:
                if record['case'] not in protocol['validation_cases']:
                    continue
                keys={(s,tid) for s,tid,f in base.queries(record) if f['conflict'] and f['goal_source']!='birth_antipode'}
                tracks={}
                for s,features in old.contexts(old.legal_record(record)):
                    for tid,f in features.items():
                        if tid not in tracks:
                            m=old.GoalPosterior(record['scene'],s,f,calibration,817)
                            tracks[tid]=dict(goals=prior_support(m),previous=(s,f),history=[])
                        else:
                            t=tracks[tid];ps,pf=t['previous']
                            if s==ps+1:
                                e=f['pos']-old.forward(pf,t['goals'])[:,0]
                                t['history'].append((s,e))
                            t['previous']=(s,f)
                        if (s,tid) not in keys:
                            continue
                        t=tracks[tid];positions=old.forward(f,t['goals'],16)
                        value=dict(case=record['case'],people=record['people'],step=s,tid=tid,
                            history=t['history'],positions=positions,
                            truth=np.array(record['truth']['positions'])[s+H+1,tid])
                        stream.write(json.dumps(value,default=old.encode)+'\n')
                log('SELECT_CACHE',record['case'],len(keys))
        Path(str(cachepath)+'.tmp').replace(cachepath)
    scores=[]
    with gzip.open(cachepath,'rt') as stream:
        for line in stream:
            row=json.loads(line);p=np.array(row['positions']);n=len(p)
            for tau,fraction in choices+[(.25,0.)]:
                density=np.zeros(n);mean=np.zeros((n,2));cov=fraction*sigma;last=None
                for s,e in row['history']:
                    ll,mean,cov=gaussian_step(np.array(e),mean,cov,sigma,fraction,np.exp(-.25/tau),1 if last is None else s-last)
                    density+=ll;last=s
                w=np.exp(density-logsumexp(density))
                _,es=common_score(p,w,np.array(row['truth']),np.array(fitted['variance'][str(row['people'])]),
                    row['case']*100003+row['step']*97+row['tid'])
                scores.append(dict(case=row['case'],people=row['people'],tau=tau,fraction=fraction,ES=float(np.mean(es)),ESS=float(1/(w@w))))
    table=[]
    for tau,fraction in choices+[(.25,0.)]:
        rows=[r for r in scores if r['tau']==tau and r['fraction']==fraction]
        cases=[dict(case=c,people=next(r['people'] for r in rows if r['case']==c),
                    ES=float(np.mean([r['ES'] for r in rows if r['case']==c]))) for c in sorted({r['case'] for r in rows})]
        value=np.mean([np.mean([r['ES'] for r in cases if r['people']==n]) for n in sorted({r['people'] for r in cases})])
        table.append(dict(tau=tau,fraction=fraction,ES=float(value),cases=cases))
    best=min(r['ES'] for r in table if r['fraction']>0)
    chosen=min((r for r in table if r['fraction']>0 and r['ES']<=best+1e-5),key=lambda r:(r['tau'],r['fraction']))
    save('selection.json',dict(table=table,selected=chosen,rows=scores))
    log('SELECTED',chosen)


def settings():
    s=read('selection.json')['selected']
    return dict(sigma=np.array(read('calibration.json')['sigma']),fraction=s['fraction'],tau=s['tau'],count=128,mh=1)


def numerical(records,calibration):
    results=[];cfg=settings()
    for case,step,tid in old.DEPTH_STATES:
        r=next(r for r in records if r['case']==case)
        m,f=model_at(r,step,tid,calibration,cfg)
        summaries=[];details=None
        for size,offset in [(64,(.5,.5)),(128,(.5,.5)),(128,(.25,.75)),(256,(.5,.5)),(256,(.25,.75)),(512,(.5,.5))]:
            ref,det=old.grid_reference(m,f,size,offset)
            summaries.append(ref);details=det
            log('NEW_GRID',case,size,offset,ref['seconds'])
            if len(summaries)>=3:
                diffs=[old.reference_difference(summaries[-1],x) for x in summaries[-3:-1]]
                if all(x['log_normalizer']<=.02 and x['binned_TV']<=.05 and x['future_mean_max_m']<=.02 for x in diffs):
                    break
        stable=all(x['log_normalizer']<=.02 and x['binned_TV']<=.05 and x['future_mean_max_m']<=.02 for x in diffs)
        particles=[]
        for count,mh in ((128,1),(256,2)):
            for seed in (0,104729,209458):
                mod,feat=model_at(r,step,tid,calibration,dict(cfg,count=count,mh=mh),seed)
                w=np.exp(mod.logweights);p=old.forward(feat,mod.goals,16)
                mean=np.einsum('n,nhd->hd',w,p)
                delta=float(np.linalg.norm(mean-np.array(ref['future_mean']),axis=1).max())
                tv=float(.5*np.abs(old.posterior_hist(mod.goals,w)-np.array(ref['histogram'])).sum())
                particles.append(dict(count=count,mh=mh,seed=seed,mean_max=delta,TV=tv,pass_check=delta<=.05 and tv<=.05))
            if all(p['pass_check'] for p in particles if p['count']==count):
                break
        target=base.predict(f,np.array(r['truth']['goals'][tid]),'improved',.5)
        equivalent=np.linalg.norm(details['positions'][:,H]-target[None,H],axis=2).mean(axis=1)
        mass={str(t):float(details['kept_weights'][equivalent<=t].sum()) for t in (.05,.1,.2)}
        np.savez_compressed(OUT/f'reference_{case}.npz',**details)
        results.append(dict(case=case,step=step,tid=tid,grids=summaries,differences=diffs,stable=stable,
            particles=particles,equivalent_goal_mass=mass))
        save('numerical.json',results)
    use256=any(not p['pass_check'] for row in results for p in row['particles'] if p['count']==128)
    save('inference_config.json',dict(count=256 if use256 else 128,mh=2 if use256 else 1,
        numerical_pass=all(row['stable'] and all(p['pass_check'] for p in row['particles'] if p['count']==(256 if use256 else 128)) for row in results),
        note='particle TV to fine bins is a stringent finite-sample test; failure retained, not redefined'))


def predict(records,calibration):
    fitted=read('calibration.json');cfg=dict(settings(),**{k:read('inference_config.json')[k] for k in ('count','mh')})
    old_modes={}
    with gzip.open(old.OUT/'modes.jsonl.gz','rt') as stream:
        for line in stream:
            m=json.loads(line);old_modes[m['case'],m['step'],m['tid']]=m['interaction']
    for r in records:
        if r['split']!='holdout' or (OUT/f'prediction_{r["case"]}.jsonl.gz').exists():
            continue
        queries={(s,i):f for s,i,f in base.queries(r)};tracks={};result=[]
        last_query=max(s for s,i in queries)
        path=OUT/f'prediction_{r["case"]}.jsonl.gz'
        with gzip.open(str(path)+'.tmp','wt') as stream:
            for step,features in old.contexts(old.legal_record(r)):
                if step>last_query:break
                for tid,f in features.items():
                    if tid not in tracks:
                        tracks[tid]={name:RepairedPosterior(r['scene'],step,f,calibration,r['case']*1009+tid*97,**cfg,wide=name=='wide') for name in ('new','wide')}
                    else:
                        for m in tracks[tid].values():m.update(step,f)
                    if (step,tid) not in queries:continue
                    new,wide=tracks[tid]['new'],tracks[tid]['wide'];saved=old_modes[r['case'],step,tid]
                    pnew=old.forward(f,new.goals,16);pwide=old.forward(f,wide.goals,16)
                    pmap=old.forward(f,[new.map_goal()],16)
                    modes={'CV':(base.predict(f,f['goal'],'cv',.5)[None],np.ones(1)),
                        'D':(base.predict(f,np.array(r['truth']['goals'][tid]),'improved',.5)[None],np.ones(1)),
                        'OldFULL':(np.array(saved['positions']),np.array(saved['weights'])),
                        'WideFULL':(pwide,np.exp(wide.logweights)),
                        'NewMAP':(pmap,np.ones(1)), 'NewFULL':(pnew,np.exp(new.logweights))}
                    truth=np.array(r['truth']['positions'])[step+H+1,tid]
                    scores={a:common_score(p,w,truth,np.array(fitted['variance'][str(r['people'])]),r['case']*100003+step*97+tid) for a,(p,w) in modes.items()}
                    near=float(np.linalg.norm(np.array(r['truth']['positions'])[step:step+17,tid]-r['truth']['goals'][tid],axis=1).min())<.5
                    row=dict(case=r['case'],people=r['people'],step=step,tid=tid,fixed=new.fixed,conflict=f['conflict'],near_goal=near,
                        errors={a:v[0] for a,v in scores.items()},energy={a:v[1] for a,v in scores.items()})
                    result.append(row)
                    stream.write(json.dumps(dict(row=row,new_goals=new.goals,new_weights=np.exp(new.logweights),
                        new_positions=pnew,map_positions=pmap,wide_positions=pwide,wide_weights=np.exp(wide.logweights)),default=old.encode)+'\n')
        Path(str(path)+'.tmp').replace(path)
        log('PREDICT',r['case'],len(result))


def proxy_features(f,goals):
    """Cheap no-neighbor response proxy, used only to choose real goal medoids."""
    pos=np.broadcast_to(f['pos'],goals.shape).copy()
    velocity=np.broadcast_to(f['vel'],goals.shape).copy()
    result=[]
    for k in range(16):
        delta=goals-pos
        norm=np.linalg.norm(delta,axis=1,keepdims=True)
        desired=delta/np.maximum(1.,norm)
        vs=np.linalg.norm(velocity,axis=1);ds=np.linalg.norm(desired,axis=1)
        active=(vs>=.1)&(ds>=.1)
        angle=np.clip(np.arctan2(velocity[:,0]*desired[:,1]-velocity[:,1]*desired[:,0],
                                np.sum(velocity*desired,axis=1)),-.375,.375)
        c,s=np.cos(angle),np.sin(angle)
        rotated=np.stack([c*velocity[:,0]-s*velocity[:,1],s*velocity[:,0]+c*velocity[:,1]],axis=1)
        desired[active]=rotated[active]* (ds[active]/vs[active])[:,None]
        change=desired-velocity
        velocity+=change*np.minimum(1.,.125/np.maximum(np.linalg.norm(change,axis=1,keepdims=True),1e-12))
        pos+=.25*velocity
        if k in (1,3,7,15):result.append(pos-f['pos'])
    result.append(np.linalg.norm(goals-pos,axis=1,keepdims=True))
    return np.concatenate(result,axis=1)


def compress_goals(f,goals,weights,k):
    unique,inverse=np.unique(goals,axis=0,return_inverse=True)
    mass=np.bincount(inverse,weights=weights,minlength=len(unique))
    positive=mass>0;unique,mass=unique[positive],mass[positive]
    mass/=mass.sum()
    if len(unique)<=k:return unique,mass
    feature=proxy_features(f,unique)
    distance=cdist(feature,feature,metric='sqeuclidean')
    representatives=[int(np.argmax(mass))]
    for _ in range(k-1):
        residual=np.min(distance[:,representatives],axis=1)*mass
        if residual.max()<1e-18:break
        representatives.append(int(np.argmax(residual)))
    for _ in range(5):
        cluster=np.argmin(distance[:,representatives],axis=1)
        proposed=[]
        for j,index in enumerate(representatives):
            ids=np.flatnonzero(cluster==j)
            proposed.append(int(ids[np.argmin(distance[np.ix_(ids,ids)]@mass[ids])]) if len(ids) else index)
        if proposed==representatives:break
        representatives=proposed
    cluster=np.argmin(distance[:,representatives],axis=1)
    w=np.bincount(cluster,weights=mass,minlength=len(representatives))
    valid=w>0
    return unique[np.array(representatives)[valid]],w[valid]/w[valid].sum()


def disc_conditional(positions,centers,variance,radius):
    d=np.linalg.norm(positions[:,None]-centers[None],axis=3)
    v=np.maximum(variance,1e-9)
    scaled=np.broadcast_to(radius**2/v,d.shape)
    nc=d*d/v
    far=d-radius>=8*np.sqrt(variance)
    q=np.full(d.shape,ndtr(-8.))
    q[~far]=ncx2.cdf(scaled[~far],2.,nc[~far])
    q=np.where(variance[None]<1e-9,d<=radius,q)
    if not np.isfinite(q).all():raise FloatingPointError('invalid mixture probability')
    return q


class MixtureEnvelope:
    def __init__(self,cfg,obs,mixtures):
        flat=copy.copy(obs);entities=[];centers=[];covariance=[];weights=[];offsets=[0]
        for j in range(len(obs.entities)):
            p,w,v=mixtures.get(j,(obs.human_segment_end[j:j+1],np.ones(1),.5*np.trace(obs.human_position_covariance[j],axis1=1,axis2=2)))
            p,w,v=np.array(p),np.array(w),np.array(v)
            if (not np.isfinite(p).all() or not np.isfinite(w).all() or (w<0).any()
                    or abs(w.sum()-1)>1e-10 or not np.isfinite(v).all() or (v<0).any()):
                raise ValueError('invalid modes')
            centers.extend(p);weights.extend(w);entities.extend([obs.entities[j]]*len(w))
            covariance.extend([v[:,None,None]*np.eye(2)]*len(w));offsets.append(len(weights))
        flat.entities=np.array(entities);flat.human_segment_end=np.array(centers)
        flat.human_position_covariance=np.array(covariance)
        flat.human_existence=np.ones(len(weights))
        self.table=CompiledDiscRisk(cfg,flat)
        self.table_cdf_values=self.table.table_cdf_values
        self.offsets=np.array(offsets);self.weights=np.array(weights)
        self.obs=obs;self.cfg=cfg;self.seconds=0.

    def aggregate(self,q):
        person=np.add.reduceat(q*self.weights[None,:,None],self.offsets[:-1],axis=1)
        r=self.obs.human_existence
        if self.cfg.existence_override is not None:r=np.full_like(r,self.cfg.existence_override)
        return -np.log1p(-np.clip(person*r[None,:,None],0.,1.-1e-12)).sum(axis=1)

    def bounds(self,positions):
        start=time.perf_counter()
        # Bound temporary arrays independently of the retained mode count.
        chunk=max(1,262144//max(1,len(self.weights)*self.cfg.horizon))
        lower,upper=[],[]
        for first in range(0,len(positions),chunk):
            lo,hi=self.table.conditional_bounds(positions[first:first+chunk])
            lower.append(self.aggregate(lo));upper.append(self.aggregate(hi))
        out=np.concatenate(lower),np.concatenate(upper)
        self.seconds+=time.perf_counter()-start
        return out

    def exact(self,positions):
        t=self.table
        chunk=max(1,262144//max(1,len(self.weights)*self.cfg.horizon))
        return np.concatenate([self.aggregate(disc_conditional(positions[i:i+chunk],t.obs.human_segment_end,t.variance,t.radius))
                               for i in range(0,len(positions),chunk)])


class MixturePlanner(MPCPlanner):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.mixtures={}
        self.risk_seconds=0.
        self.geometry_counts=[0,0]
        self.audit_geometry=False

    def _build_envelope(self,obs):
        self.risk_seconds=0.;self.geometry_counts=[0,0]
        if not obs.entities.size:return None
        start=time.perf_counter()
        result=MixtureEnvelope(self.cfg,obs,self.mixtures)
        self.risk_seconds+=time.perf_counter()-start
        self.mixture_evaluator=result
        return result

    def _belief_collision_hazard(self,controls,obs,positions=None):
        self.risk_rows+=len(controls)
        if not obs.entities.size:return None
        if positions is None:positions=obs.robot_xy[None,None]+np.cumsum(controls*self.cfg.dt,axis=1)
        start=time.perf_counter();h=self.mixture_evaluator.exact(positions)
        self.risk_seconds+=time.perf_counter()-start
        return h

    def _score_from_hazard(self,base_cost,geo_full,geo_first,active,limits,h):
        if self.audit_geometry and h is not None:
            permitted=np.all(np.where(active,h<=limits,True),axis=1)
            self.geometry_counts[0]+=int(np.sum(permitted & (geo_full<0)))
            self.geometry_counts[1]+=int(permitted.sum())
        return super()._score_from_hazard(base_cost,geo_full,geo_first,active,limits,h)


def fitted_settings():
    return dict(settings(),**{k:read('inference_config.json')[k] for k in ('count','mh')})


class IntentEngine:
    """Only receives legal detections, robot observation and public scene."""
    def __init__(self,scene,case,arm,calibration,k):
        self.scene,self.case,self.arm,self.calibration,self.k=scene,case,arm,calibration,k
        self.past={};self.tracks={};self.variance=read('calibration.json')['variance']
        self.settings=fitted_settings()

    def update(self,step,detections,obs,reported_ids,people,forecast=True):
        timing=dict(goal_update=0.,compression=0.,rollout=0.)
        start=time.perf_counter()
        frame=dict(step=step,detections=detections,robot=[*obs.robot_xy,*obs.robot_velocity,obs.robot_radius])
        features={}
        for e in detections:
            tid=e['id'];p=np.array([e['px'],e['py']]);v=np.array([e['vx'],e['vy']])
            self.past.setdefault(tid,[]).append((step,p,v))
            f=base.legal_features(dict(scene=self.scene),frame,e,self.past[tid]);features[tid]=f
            if self.arm=='B' or f['goal_source']=='birth_antipode':continue
            if tid not in self.tracks:
                if self.arm=='F0':
                    self.tracks[tid]=old.GoalPosterior(self.scene,step,f,self.calibration,self.case*1009+tid*97)
                else:
                    self.tracks[tid]=RepairedPosterior(self.scene,step,f,self.calibration,self.case*1009+tid*97,**self.settings)
            else:self.tracks[tid].update(step,f)
        timing['goal_update']=(time.perf_counter()-start)*1000
        if not forecast:return {},timing
        mixtures={};variance=np.array(self.variance[str(people)])
        for j,tid in enumerate(reported_ids):
            if tid not in features or features[tid]['goal_source']=='birth_antipode':continue
            f=features[tid]
            if self.arm=='B':
                # Risk predictors share the same measured initial state; hard geometry stays untouched.
                mixtures[j]=(base.predict(f,f['goal'],'cv',.5)[None],np.ones(1),variance)
                continue
            m=self.tracks[tid]
            start=time.perf_counter()
            if self.arm=='M':goals,w=m.map_goal()[None],np.ones(1)
            else:goals,w=compress_goals(f,m.goals,np.exp(m.logweights),self.k)
            timing['goal_update' if self.arm=='M' else 'compression']+=(time.perf_counter()-start)*1000
            start=time.perf_counter();p=old.forward(f,goals,16)
            timing['rollout']+=(time.perf_counter()-start)*1000
            mixtures[j]=(p,w,variance)
        return mixtures,timing


def observation_at(record,step):
    _,_,ActionXY,_,_=base.legacy._load_modules(base.CROWD)
    env=base.environment(record);adapter=BayesObservationAdapter(MPCConfig(**read_config()))
    for frame in record['frames']:
        obs=adapter.read(env)
        if adapter.detected_entities!=frame['detections']:raise RuntimeError('record replay mismatch')
        if frame['step']==step:return env,adapter,obs
        env.step(ActionXY(*map(float,frame['action'])))
    raise ValueError('missing frame')


def compression(records,protocol,calibration):
    results=[];cfg=MPCConfig(**protocol['config']);inference=fitted_settings()
    for case,step,tid in protocol['states']:
        r=next(r for r in records if r['case']==case)
        m,f=model_at(r,step,tid,calibration,inference)
        weights=np.exp(m.logweights);start=time.perf_counter();full=old.forward(f,m.goals,16)
        full_ms=(time.perf_counter()-start)*1000
        env,adapter,obs=observation_at(r,step)
        planner=MPCPlanner(cfg);rng=np.random.default_rng(case+step)
        seeds=planner._seed_trajectories(obs)
        raw=np.repeat(planner._initial_mean(obs)[None],24,axis=0)+rng.normal(0,cfg.init_std,(24,16,2))
        raw[:len(seeds)]=seeds
        _,_,positions=planner._rollout(raw,obs)
        variance=np.array(read('calibration.json')['variance'][str(r['people'])])
        radius=obs.robot_radius+f['radius']+cfg.human_margin
        qfull=np.einsum('nmh,m->nh',disc_conditional(positions,full,variance[None],radius),weights)
        reference_mean=np.einsum('m,mhd->hd',weights,full)
        truth=np.array(r['truth']['positions'])[step+H+1,tid]
        _,esfull=common_score(full,weights,truth,variance,case+step,pairs=8192)
        modes=[]
        for k in (4,8,16):
            start=time.perf_counter();goals,w=compress_goals(f,m.goals,weights,k)
            compress_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();p=old.forward(f,goals,16);rollout_ms=(time.perf_counter()-start)*1000
            mean=np.einsum('m,mhd->hd',w,p)
            q=np.einsum('nmh,m->nh',disc_conditional(positions,p,variance[None],radius),w)
            _,es=common_score(p,w,truth,variance,case+step,pairs=8192)
            delta=float(np.linalg.norm(mean-reference_mean,axis=1).max())
            risk=float(np.abs(q-qfull).max());edelta=float(np.mean(np.array(es)-esfull))
            flips=int(np.sum((q[:,0]<=cfg.near_chance_limit)!=(qfull[:,0]<=cfg.near_chance_limit)))
            modes.append(dict(k=k,actual=len(w),mean_max=delta,risk_max=risk,ES_delta=edelta,first_boundary_flips=flips,
                compression_ms=compress_ms,rollout_ms=rollout_ms,pass_check=delta<=.05 and risk<=.01 and edelta<=.01 and flips==0))
        truth_d=base.predict(f,np.array(r['truth']['goals'][tid]),'improved',.5)
        equivalence=np.linalg.norm(full[:,H]-truth_d[None,H],axis=2).mean(axis=1)
        old_model,_=old.posterior_at(r,step,tid,calibration)
        old_weights=np.exp(old_model.logweights);old_full=old.forward(f,old_model.goals,16)
        old_mean=np.einsum('m,mhd->hd',old_weights,old_full)
        old_risk=np.einsum('nmh,m->nh',disc_conditional(positions,old_full,variance[None],radius),old_weights)
        _,old_es=common_score(old_full,old_weights,truth,variance,case+step,pairs=8192)
        old_modes=[]
        for k in (4,8,16):
            goals,w=compress_goals(f,old_model.goals,old_weights,k);p=old.forward(f,goals,16)
            mean=np.einsum('m,mhd->hd',w,p)
            risk=np.einsum('nmh,m->nh',disc_conditional(positions,p,variance[None],radius),w)
            _,es=common_score(p,w,truth,variance,case+step,pairs=8192)
            delta=float(np.linalg.norm(mean-old_mean,axis=1).max())
            rd=float(np.abs(risk-old_risk).max());ed=float(np.mean(np.array(es)-old_es))
            flips=int(np.sum((risk[:,0]<=cfg.near_chance_limit)!=(old_risk[:,0]<=cfg.near_chance_limit)))
            old_modes.append(dict(k=k,mean_max=delta,risk_max=rd,ES_delta=ed,first_boundary_flips=flips,
                                  pass_check=delta<=.05 and rd<=.01 and ed<=.01 and flips==0))
        cv=base.predict(f,f['goal'],'cv',.5)
        results.append(dict(case=case,step=step,tid=tid,full_count=len(weights),full_ms=full_ms,modes=modes,
            old_modes=old_modes,equivalent_goal_mass={str(t):float(weights[equivalence<=t].sum()) for t in (.05,.1,.2)},
            same_query_errors={name:np.linalg.norm(p[H]-truth,axis=1) for name,p in
                               [('CV',cv),('D',truth_d),('OldFULL',old_mean),('NewFULL',reference_mean)]}))
        save('compression.json',results);log('COMPRESS',case,[(x['k'],x['pass_check'],round(x['risk_max'],4)) for x in modes])
    eligible=[k for k in (4,8,16) if all(next(m for m in r[name] if m['k']==k)['pass_check']
                                       for r in results for name in ('modes','old_modes'))]
    save('compression_config.json',dict(k=min(eligible) if eligible else inference['count'],passed=bool(eligible),
        fallback='if no K passes, retain all inference particles before deduplicating identical goals; report missed runtime rather than silently lose mode accuracy'))


def interface(records,calibration):
    record=next(r for r in records if r['case']==2000020)
    _,adapter,obs=observation_at(record,24)
    cfg=MPCConfig(**read_config());rng=np.random.default_rng(1931)
    controls=MPCPlanner(cfg)._project_controls(rng.normal(0,.7,(24,16,2)),obs.robot_velocity)
    positions=obs.robot_xy[None,None]+np.cumsum(.25*controls,axis=1)
    ref=exact_hazard(cfg,controls,obs,positions)
    mix=MixtureEnvelope(cfg,obs,{})
    np.testing.assert_allclose(mix.exact(positions),ref,atol=1e-12,rtol=0)
    j=0;p=obs.human_segment_end[j:j+1];v=.5*np.trace(obs.human_position_covariance[j],axis1=1,axis2=2)
    split=MixtureEnvelope(cfg,obs,{j:(np.repeat(p,2,axis=0),np.array([.3,.7]),v)})
    np.testing.assert_allclose(split.exact(positions),ref,atol=1e-12,rtol=0)
    modes=np.repeat(p,2,axis=0);modes[0,:,0]-=.5;modes[1,:,0]+=.5
    a=MixtureEnvelope(cfg,obs,{j:(modes,np.array([.2,.8]),v)})
    b=MixtureEnvelope(cfg,obs,{j:(modes[::-1],np.array([.8,.2]),v)})
    np.testing.assert_allclose(a.exact(positions),b.exact(positions),atol=1e-12,rtol=0)
    lo,hi=a.bounds(positions);exact=a.exact(positions)
    assert np.all(lo<=exact+1e-10) and np.all(hi>=exact-1e-10)
    zero=copy.deepcopy(obs);zero.human_existence[:]=0
    assert np.all(MixtureEnvelope(cfg,zero,{j:(modes,np.array([.2,.8]),v)}).exact(positions)==0)
    np.testing.assert_array_equal(obs.human_segment_end,adapter.belief.output(16).segment_end)
    original=MPCPlanner(cfg);bridge=MixturePlanner(cfg)
    x,_=original.plan(obs,713);y,_=bridge.plan(obs,713)
    np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)
    # Deployment engine has no record/truth parameter; poisoned metadata never enters it.
    features={s:f for s,f in old.contexts(old.legal_record(record))}
    modified=copy.deepcopy(record);modified['truth']['goals']=[[999.,-999.]]*record['people']
    alternate={s:f for s,f in old.contexts(old.legal_record(modified))}
    for s in features:
        for tid in features[s]:
            np.testing.assert_array_equal(features[s][tid]['goal'],alternate[s][tid]['goal'])
    model,_=model_at(record,24,0,calibration,fitted_settings())
    cached=model.logposterior
    model.logposterior=lambda goals:model.replay(goals)[0]
    reference_goal=model.map_goal()
    model.logposterior=cached
    np.testing.assert_array_equal(model.map_goal(),reference_goal)
    save('interface.json',dict(single_mode=True,duplicate_mode=True,mode_order=True,zero_existence=True,
        mixed_before_log=True,numerical_enclosure=True,geometry_unchanged=True,cv_reference_action=True,
        truth_metadata_does_not_enter_features=True,MAP_density_cache_exact=True,
        scope='floating tolerance 1e-12; synthetic and one development observation; no safety certificate'))
    log('INTERFACE PASSED')


def register_navigation(protocol):
    if 'navigation' in protocol:return protocol
    used,hashes,audit=base.history()
    if audit['unreadable']:
        # Keep the audit explicit: corrupt files are not silently assumed unused.
        save('history_unreadable.json',audit['unreadable'])
        raise RuntimeError('historical registry contains unreadable result files; resolve before new cases')
    start=max(3000000,max(used,default=0)+1000)
    cases=[];seen=set()
    for group,(people,scene) in enumerate(((5,'circle_crossing'),(10,'circle_crossing'),(20,'square_crossing'))):
        for j in range(6):
            item=dict(case=start+group*6+j,people=people,scene=scene)
            env=base.environment(item);layout=base.layout(env)
            if item['case'] in used or layout in hashes or layout in seen:raise RuntimeError('reused layout')
            seen.add(layout);item['layout_hash']=layout;cases.append(item)
    protocol['navigation']=dict(cases=cases,history_files=audit['files'],arms=NAV_ARMS,
        seed='case*100003+step*97+1729',order='rotate arm order by case index; all five complete before next case',
        covariance_frozen=old.digest(OUT/'calibration.json'),
        inference_config=read('inference_config.json'),compression_config=read('compression_config.json'))
    save('protocol.json',protocol)
    return protocol


def episode(item,arm,calibration,k):
    cfg=MPCConfig(**read_config());env=base.environment(item)
    if base.layout(env)!=item['layout_hash']:raise RuntimeError('initial layout mismatch')
    adapter=BayesObservationAdapter(cfg)
    planner=MPCPlanner(cfg) if arm=='O' else MixturePlanner(cfg)
    engine=None if arm=='O' else IntentEngine(item['scene'],item['case'],arm,calibration,k)
    _,_,ActionXY,_,_=base.legacy._load_modules(base.CROWD)
    steps=[];done=False;overlap=0;step=0
    began=time.perf_counter()
    while not done:
        started=time.perf_counter();obs=adapter.read(env);observed=time.perf_counter()
        timing=dict(goal_update=0.,compression=0.,rollout=0.)
        if engine is not None:
            planner.mixtures,timing=engine.update(step,adapter.detected_entities,obs,adapter.reported_ids,item['people'])
        action,plan_ms=planner.plan(obs,item['case']*100003+step*97+1729)
        if (not np.isfinite(action).all() or np.linalg.norm(action)>cfg.v_max+1e-6
                or np.linalg.norm(action-obs.robot_velocity)>cfg.a_max*cfg.dt+1e-6):
            raise RuntimeError('invalid executable action')
        command=ActionXY(*map(float,action));elapsed=(time.perf_counter()-started)*1000
        timing.update(observation=(observed-started)*1000,mpc=plan_ms,total=elapsed)
        if engine is not None:
            query=planner.mixture_evaluator.seconds if obs.entities.size else 0.
            timing['risk']=(planner.risk_seconds+query)*1000
            timing['mpc_other']=max(0.,plan_ms-timing['risk'])
        else:
            timing['risk']=None;timing['mpc_other']=None
        before=np.array(env.robot.get_position());humans=np.array([h.get_position() for h in env.humans])
        radii=np.array([h.radius+env.robot.radius for h in env.humans])
        _,_,term,trunc,info=env.step(command)
        after=np.array(env.robot.get_position())
        np.testing.assert_allclose(after,before+.25*action,atol=1e-12,rtol=0)
        clearance=base.legacy.swept_min_clearance(before,after,humans,np.array([h.get_position() for h in env.humans]),radii)
        overlap+=int(clearance < -1e-9)
        steps.append(dict(step=step,action=action,robot_after=after,visible=len(adapter.detected_entities),
            timing=timing,clearance=clearance,feasibility=planner.last_diagnostics['feasibility_class']))
        done=term or trunc;step+=1
        if step>110:raise RuntimeError('nonterminating environment')
    event=str(info['event']);collision=int(event=='collision' or overlap>0)
    success=int(event=='reach_goal' and not collision);timeout=int(event=='timeout' and not collision)
    if success+collision+timeout!=1:raise RuntimeError('unclassified outcome '+event)
    return dict(**item,arm=arm,status='ok',event=event,success=success,collision=collision,timeout=timeout,
        penalty=float(env.global_time) if success else 25.,nav_time=float(env.global_time),steps=steps,
        seconds=time.perf_counter()-began)


def navigate(protocol,calibration):
    protocol=register_navigation(protocol)
    path=OUT/'episodes.jsonl'
    existing={(r['case'],r['arm']) for r in map(json.loads,path.read_text().splitlines()) if r['status']=='ok'} if path.exists() else set()
    k=read('compression_config.json')['k']
    for index,item in enumerate(protocol['navigation']['cases']):
        order=list(NAV_ARMS[index%5:]+NAV_ARMS[:index%5])
        for arm in order:
            if (item['case'],arm) in existing:continue
            try:
                row=episode(item,arm,calibration,k)
            except Exception as exc:
                append('execution_errors.jsonl',dict(case=item['case'],arm=arm,error=repr(exc),time=time.time()))
                raise
            append('episodes.jsonl',row)
            log('NAV',item['case'],item['people'],arm,row['success'],row['collision'],row['timeout'],round(row['penalty'],2),round(row['seconds'],1))


def action_diagnostic(records,calibration):
    import types
    cfg=MPCConfig(**read_config());k=read('compression_config.json')['k']
    _,_,ActionXY,_,_=base.legacy._load_modules(base.CROWD)
    results=[]
    for r in records:
        if r['split']!='holdout':continue
        steps={len(r['frames'])//3,2*len(r['frames'])//3}
        env=base.environment(r);adapter=BayesObservationAdapter(cfg);reference=MPCPlanner(cfg)
        engines={arm:IntentEngine(r['scene'],r['case'],arm,calibration,k) for arm in NAV_ARMS if arm!='O'}
        for frame in r['frames']:
            step=frame['step']
            if step>max(steps):break
            obs=adapter.read(env)
            if adapter.detected_entities!=frame['detections']:raise RuntimeError('action diagnostic legal observations differ')
            previous=copy.deepcopy(reference._previous_mean);pool=[]
            reference.trace_hook=(lambda state: pool.append(state['params'].copy())) if step in steps else None
            original,_=reference.plan(obs,r['case']*100003+step*97+1729)
            np.testing.assert_array_equal(original,frame['action'])
            mixtures={}
            for arm,engine in engines.items():
                mixtures[arm],_=engine.update(step,adapter.detected_entities,obs,adapter.reported_ids,r['people'],forecast=step in steps)
            if step in steps:
                candidates=pool[0]
                row=dict(case=r['case'],people=r['people'],step=step,arms={},candidate_count=len(candidates),
                    scope='fixed original first-iteration pool shared by all arms, same starting warm state; offline future clearance only')
                for arm in NAV_ARMS:
                    p=MPCPlanner(cfg,envelope=None) if arm=='O' else MixturePlanner(cfg)
                    if arm!='O':p.mixtures=mixtures[arm]
                    p._previous_mean=copy.deepcopy(previous)
                    def fixed_rollout(self,samples,observation):
                        positions=observation.robot_xy[None,None]+np.cumsum(candidates*.25,axis=1)
                        return candidates,candidates,positions
                    p._rollout=types.MethodType(fixed_rollout,p)
                    audited=[]
                    def audit_pool(state):
                        if audited or arm=='O':return
                        hazard=p.mixture_evaluator.exact(state['positions'])
                        active=p._active_until_goal(state['positions'],obs)[0]
                        permitted=np.all(np.where(active,hazard<=p._belief_hazard_limits()[None],True),axis=1)
                        geometric,_=p._combined_clearance(state['controls'],obs,state['positions'],
                                                         state['human_clearance'],state['occupancy'],None)
                        p.geometry_counts=[int(np.sum(permitted & (geometric<0))),int(permitted.sum())]
                        audited.append(True)
                    p.trace_hook=audit_pool
                    action,_=p.plan(obs,r['case']*100003+step*97+1729)
                    positions=obs.robot_xy+np.cumsum(.25*p.last_controls,axis=0)
                    available=min(16,len(r['truth']['positions'])-step-1)
                    truth=np.array(r['truth']['positions'])[step+1:step+1+available]
                    radius=np.array([h.radius for h in env.humans])+env.robot.radius
                    clearance=float(np.min(np.linalg.norm(positions[:available,None]-truth,axis=2)-radius[None])) if available else None
                    row['arms'][arm]=dict(action=action,diagnostics=p.last_diagnostics.copy(),future_discrete_clearance=clearance,
                        future_evaluation_steps=available,full_four_second_labels=available==16,
                        progress=float(np.linalg.norm(obs.goal_xy-obs.robot_xy)-np.linalg.norm(obs.goal_xy-positions[-1])),
                        geometry_rejected=(p.geometry_counts if arm!='O' else None))
                results.append(row);save('actions.json',results)
                log('ACTION_COMMON',r['case'],step,{a:round(float(np.linalg.norm(np.array(v['action'])-row['arms']['O']['action'])),4) for a,v in row['arms'].items()})
            env.step(ActionXY(*map(float,frame['action'])))


def prediction_rows():
    rows=[]
    for p in sorted(OUT.glob('prediction_*.jsonl.gz')):
        with gzip.open(p,'rt') as stream:
            rows.extend(json.loads(s)['row'] for s in stream)
    return rows


def summarize_predictions():
    rows=prediction_rows();table=[];contrasts=[]
    masks={'all':lambda r:True,'ordinary':lambda r:not r['conflict'],
        'known_birth':lambda r:r['fixed'], 'unknown_conflict':lambda r:r['conflict'] and not r['fixed'],
        'unknown_conflict_nonarrival':lambda r:r['conflict'] and not r['fixed'] and not r['near_goal']}
    for subset,mask in masks.items():
        selected=[r for r in rows if mask(r)]
        if not selected:continue
        cases=sorted({r['case'] for r in selected});groups=[];values=[]
        for case in cases:
            sample=[r for r in selected if r['case']==case];groups.append(sample[0]['people'])
            values.append([[np.mean([r[field][arm] for r in sample],axis=0) for arm in ARMS] for field in ('errors','energy')])
        a=np.array(values);g=np.array(groups)
        mean=np.mean([a[g==n].mean(axis=0) for n in sorted(set(groups))],axis=0)
        for j,arm in enumerate(ARMS):table.append(dict(subset=subset,arm=arm,queries=len(selected),episodes=len(cases),errors=mean[0,j],energy=mean[1,j]))
        for left,right in (('NewFULL','OldFULL'),('NewFULL','WideFULL'),('NewFULL','NewMAP'),('NewFULL','CV')):
            for field,index in (('errors',0),('energy',1)):
                d=a[:,index,ARMS.index(left)]-a[:,index,ARMS.index(right)]
                contrasts.append(dict(subset=subset,field=field,left=left,right=right,
                    delta=float(np.mean([d[g==n].mean() for n in set(groups)])),CI=base.interval(d.mean(axis=1),groups)))
    return dict(table=table,contrasts=contrasts,queries=len(rows),reference='fixed reused evaluation episodes, not fresh confirmation')


def timing_summary(values):
    a=np.array(values)
    return dict(p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),
                late_fraction=float(np.mean(a>250)),count=len(a))


def report():
    raw=prediction_rows()
    expected={(r['case'],r['step'],r['tid']) for r in map(json.loads,(old.OUT/'predictions.jsonl').read_text().splitlines())}
    actual={(r['case'],r['step'],r['tid']) for r in raw}
    if actual!=expected or len(raw)!=len(expected):raise RuntimeError('incomplete or duplicated prediction pool')
    tmp=OUT/'predictions.jsonl.tmp'
    with tmp.open('w') as stream:
        for row in raw:stream.write(json.dumps(row,default=old.encode,allow_nan=False)+'\n')
    tmp.replace(OUT/'predictions.jsonl')
    prediction=summarize_predictions();save('prediction_summary.json',prediction)
    rows=[json.loads(s) for s in (OUT/'episodes.jsonl').read_text().splitlines()] if (OUT/'episodes.jsonl').exists() else []
    table=[];timing=[];paired=[]
    for people in (5,10,20,'all'):
        for arm in NAV_ARMS:
            q=[r for r in rows if r['arm']==arm and (people=='all' or r['people']==people)]
            if not q:continue
            components={name:timing_summary([s['timing'][name] for r in q for s in r['steps'] if s['timing'].get(name) is not None])
                        for name in ('observation','goal_update','compression','rollout','mpc','total')}
            if arm!='O':
                for name in ('risk','mpc_other'):
                    components[name]=timing_summary([s['timing'][name] for r in q for s in r['steps']])
            timing.append(dict(people=people,arm=arm,components=components))
            table.append(dict(people=people,arm=arm,episodes=len(q),success=sum(r['success'] for r in q),
                collision=sum(r['collision'] for r in q),timeout=sum(r['timeout'] for r in q),
                penalty=float(np.mean([r['penalty'] for r in q])),pipeline=components['total']))
    bykey={(r['case'],r['arm']):r for r in rows}
    for left,right in (('F','M'),('F','F0'),('F','B'),('B','O')):
        cases=sorted(c for c,a in bykey if a==left and (c,right) in bykey)
        if not cases:continue
        transitions={};delta=[];groups=[]
        for c in cases:
            a,b=bykey[c,left],bykey[c,right]
            la=next(k for k in ('success','collision','timeout') if a[k]);lb=next(k for k in ('success','collision','timeout') if b[k])
            label=lb+' -> '+la;transitions[label]=transitions.get(label,0)+1
            delta.append(a['penalty']-b['penalty']);groups.append(a['people'])
        paired.append(dict(left=left,right=right,cases=len(cases),transitions=transitions,
            penalty_delta=float(np.mean(delta)),CI=base.interval(np.array(delta),groups),
            beneficial=sum(bykey[c,left]['success'] and not bykey[c,right]['success'] for c in cases),
            harmful=sum(bykey[c,right]['success'] and not bykey[c,left]['success'] for c in cases)))
    save('timing.json',timing);save('navigation_summary.json',dict(table=table,paired=paired))
    lines=['# Intent repair: end-to-end results','',
        'Frozen original D and production MPC; only experiment likelihood and risk adapter changed.',
        'Prediction evaluation reuses viewed data. Navigation uses separately registered new layouts. No threshold/search tuning.',
        '', '## Prediction: unknown-goal conflict','',
        '| Arm | 1 s | 2 s | 3 s | 4 s | Mean ES |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in prediction['table']:
        if r['subset']=='unknown_conflict':lines.append('| '+r['arm']+' | '+' | '.join('%.4f'%v for v in r['errors'])+' | %.4f |'%np.mean(r['energy']))
    lines+=['','## Navigation and complete runtime','',
        '| Arm | N | Success | Collision | Timeout | Penalty s | p50 ms | p95 ms | p99 ms | >250 ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table:
        if r['people']=='all':
            t=r['pipeline'];lines.append('| %s | %d | %d | %d | %d | %.3f | %.1f | %.1f | %.1f | %.1f%% |'%(r['arm'],r['episodes'],r['success'],r['collision'],r['timeout'],r['penalty'],t['p50'],t['p95'],t['p99'],100*t['late_fraction']))
    lines+=['','## Checks and limitations','',
        '- Numerical status: '+json.dumps(read('inference_config.json')),
        '- Compression status: '+json.dumps(read('compression_config.json')),
        '- Risk is conditional on a fitted isotropic marginal D error kernel, not a certified joint safety probability.',
        '- Current CV geometry is common to all arms and may block mode-dependent routes.',
        '- F versus F0 includes both likelihood repair and the registered particle/MH refinement; L2 versus L1 isolates the temporal likelihood at matched approximation.',
        '- Offline action clearance uses only available future labels; records shorter than four seconds are flagged, not extended.',
        '- The factorial diagnostic uses the actual ORCA/TTC generator without extra D rollout clipping in all four cells; D forecasts remain frozen.',
        '- Timing is measured serially, includes inference and MPC, excludes environment truth/audit and writing.',
        '- Risk is a subset of MPC time, not an additional summand. Component percentiles do not add.',
        '- Goal MAP optimization is included in goal-update time for M.',
        '- No claim of novelty from AR residuals, Rao-Blackwellization or medoids alone.',
        '', '## Paired outcomes','']
    for r in paired:lines.append('- '+json.dumps(r))
    lines+=['','Prediction contrasts and grouped latency/navigation tables are in the neighboring JSON reports.']
    # This file is generated exclusively from the registered numerical results.
    (OUT/'verdict.md').write_text('\n'.join(lines)+'\n')
    log('REPORT',len(rows),'navigation episodes',prediction['queries'],'prediction queries')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','factorial','fit','select','numerical','predict','compress','interface','actions','navigate','report','all'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    records=old.data();calibration=json.loads((old.OUT/'calibration.json').read_text());protocol=setup(records)
    source=old.digest(__file__)
    protocol['source_revisions'].append(dict(sha=source,command=args.command,time=time.time()))
    protocol['cpu_affinity']=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    save('protocol.json',protocol)
    started=time.perf_counter()
    if args.command=='all':
        for stage in ('prepare','factorial','fit','select','numerical','predict','compress','interface','actions','navigate','report'):
            if args.resume and read('protocol.json')['stages'].get(stage,{}).get('status')=='complete':continue
            subprocess.run([sys.executable,str(Path(__file__).resolve()),stage],cwd=ROOT,check=True)
        return
    if args.command=='prepare':selftest()
    elif args.command=='factorial':factorial(records,protocol)
    elif args.command=='fit':fit(records,protocol)
    elif args.command=='select':select(records,protocol,calibration)
    elif args.command=='numerical':numerical(records,calibration)
    elif args.command=='predict':predict(records,calibration)
    elif args.command=='compress':compression(records,protocol,calibration)
    elif args.command=='interface':interface(records,calibration)
    elif args.command=='actions':action_diagnostic(records,calibration)
    elif args.command=='navigate':navigate(protocol,calibration)
    elif args.command=='report':report()
    protocol=read('protocol.json');protocol['stages'][args.command]=dict(status='complete',seconds=time.perf_counter()-started,source_sha=source)
    save('protocol.json',protocol)


if __name__=='__main__':
    main()
