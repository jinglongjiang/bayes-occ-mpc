"""Frozen-input backend audit; no new episodes, tuning or navigation queue."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments import intent_backend as fast
from experiments import intent_repair_run as ref
from experiments import goal_posterior_probe as old
from integration.crowdnav import BayesObservationAdapter
from nav.contracts import MPCConfig
from nav.planner import MPCPlanner
from reproducibility.smoke import check_posterior

OUT=ROOT/'results/intent_backend'


def save(name,value):
    OUT.mkdir(exist_ok=True)
    temporary=OUT/(name+'.tmp')
    temporary.write_text(json.dumps(value,default=old.encode,allow_nan=False,indent=2)+'\n')
    temporary.replace(OUT/name)


def load(name,default=None):
    return json.loads((OUT/name).read_text()) if (OUT/name).exists() else default


def protocol():
    p=load('protocol.json')
    if p is None:
        p=dict(baseline='822c28bfda6fb819083a13253739936384144c7a',
            states=ref.read('protocol.json')['states'],settings=ref.fitted_settings(),
            config=ref.read_config(),mode_cap=ref.read('compression_config.json')['k'],
            repetitions=3,tolerances=dict(trajectory=1e-10,log_density=1e-8,weights=1e-10,risk=1e-10,controls=1e-9),
            scope='12 fixed development states, common canonical prefix and warm start; no new navigation outcomes',
            ordering='rotate four backends by state and repetition; single process, one CPU, one numerical thread',
            primary_cache='empty history context cache at each timed control; warm-cache repeat measured only in kernel audit',
            source_revisions=[])
    assert p['config']==ref.read_config()
    p['source_revisions'].append({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (Path(__file__).resolve(),ROOT/'experiments/intent_backend.py',ROOT/'experiments/intent_kernels.cpp')})
    save('protocol.json',p);return p


def difference(a,b,tolerance):
    a,b=np.array(a),np.array(b)
    value=float(np.max(abs(a-b))) if a.size else 0.
    np.testing.assert_allclose(a,b,rtol=0,atol=tolerance)
    return value


def timed(function):
    start=time.perf_counter();result=function();return result,1000*(time.perf_counter()-start)


def kernels(records,calibration,p):
    rows=load('kernels.json',[]);done={tuple(r['state']) for r in rows}
    reference=check_posterior()
    with fast.enabled(): repaired=check_posterior()
    save('posterior_check.json',dict(original=reference,optimized=repaired))
    for case,step,tid in p['states']:
        if (case,step,tid) in done:continue
        record=next(r for r in records if r['case']==case)
        model,f=ref.model_at(record,step,tid,calibration,ref.fitted_settings())
        behavior=fast.Behavior();metrics={};times={}
        for h in (1,16):
            a,ta=timed(lambda:old.forward(f,model.goals,h))
            b,tb=timed(lambda:behavior.forward(f,model.goals,h))
            metrics['trajectory_'+str(h)]=difference(a,b,p['tolerances']['trajectory'])
            times['D_'+str(h)]=dict(original=ta,optimized=tb)
        a,ta=timed(lambda:model.replay(model.goals))
        with fast.enabled(risk=False) as cache:
            b,tb=timed(lambda:model.replay(model.goals))
            _,tw=timed(lambda:model.replay(model.goals))
            hits=cache.hits
        metrics['history_log_density']=difference(a[0],b[0],p['tolerances']['log_density'])
        metrics['history_bias']=difference(a[1],b[1],p['tolerances']['weights'])
        times['history']=dict(original=ta,optimized_cold=tb,optimized_warm=tw,context_hits=hits)
        with fast.enabled(risk=False):newmodel,_=ref.model_at(record,step,tid,calibration,ref.fitted_settings())
        metrics['particle_goals']=difference(model.goals,newmodel.goals,p['tolerances']['trajectory'])
        metrics['particle_weights']=difference(model.logweights,newmodel.logweights,p['tolerances']['weights'])
        rows.append(dict(state=[case,step,tid],people=record['people'],goals=len(model.goals),
                         history=len(model.history),errors=metrics,milliseconds=times))
        save('kernels.json',rows);print('KERNEL',case,metrics,times,flush=True)


def prepare(record,step,p,calibration):
    env=old.base.environment(record);adapter=BayesObservationAdapter(MPCConfig(**p['config']))
    original=MPCPlanner(MPCConfig(**p['config']))
    engine=ref.IntentEngine(record['scene'],record['case'],'F',calibration,p['mode_cap'])
    _,_,ActionXY,_,_=old.base.legacy._load_modules(old.base.CROWD)
    for frame in record['frames']:
        if frame['step']==step:return env,adapter,engine,original._previous_mean
        obs=adapter.read(env)
        if adapter.detected_entities!=frame['detections']:raise RuntimeError('prefix detections')
        action,_=original.plan(obs,record['case']*100003+frame['step']*97+1729)
        np.testing.assert_array_equal(action,frame['action'])
        engine.update(frame['step'],adapter.detected_entities,obs,adapter.reported_ids,record['people'],forecast=False)
        env.step(ActionXY(*map(float,action)))
    raise ValueError(step)


def pipeline(env,adapter_before,engine_before,warm,record,step,behavior,risk,trace=False):
    adapter=copy.deepcopy(adapter_before);engine=copy.deepcopy(engine_before)
    planner=ref.MixturePlanner(MPCConfig(**ref.read_config()));planner._previous_mean=copy.deepcopy(warm)
    traces=[]
    def collect(s):
        ids=[]
        for route,count in enumerate(s['counts']):
            start,end=s['bounds'][route:route+2]
            ids.extend((planner._elite_indices(s['costs'][start:end],s['full'][start:end],s['first'][start:end],
                max(4,round(count*planner.cfg.elite_fraction)))+start).tolist())
        traces.append(dict(elites=ids,winner=s['ibest'],category=s['iclass'],means=s['means'].copy(),
                           deviations=s['deviations'].copy(),best=s['best_controls'].copy()))
    if trace:planner.trace_hook=collect
    with fast.enabled(behavior,risk):
        start=time.perf_counter();obs=adapter.read(env);observed=time.perf_counter()
        mixtures,timing=engine.update(step,adapter.detected_entities,obs,adapter.reported_ids,record['people'])
        planner.mixtures=mixtures
        action,ms=planner.plan(obs,record['case']*100003+step*97+1729)
        # Include constructing the executable command, not environment truth advancement.
        from crowd_sim.envs.utils.action import ActionXY
        ActionXY(*map(float,action))
        total=1000*(time.perf_counter()-start)
    timing.update(observation=1000*(observed-start),mpc=ms,total=total,
                  risk=1000*(planner.risk_seconds+planner.mixture_evaluator.seconds))
    timing['mpc_other']=ms-timing['risk']
    return dict(timing=timing,action=action,controls=planner.last_controls.copy(),traces=traces,
                obs=obs,mixtures=mixtures,engine=engine,diagnostics=planner.last_diagnostics)


def pipelines(records,calibration,p):
    rows=load('pipeline.json',[]);done={tuple(r['state']) for r in rows}
    variants=[('original',False,False),('behavior_only',True,False),('risk_only',False,True),('combined',True,True)]
    for index,(case,step,tid) in enumerate(p['states']):
        if (case,step,tid) in done:continue
        record=next(r for r in records if r['case']==case)
        print('PREPARE',case,step,flush=True)
        env,adapter,engine,warm=prepare(record,step,p,calibration)
        a=pipeline(env,adapter,engine,warm,record,step,False,False,True)
        b=pipeline(env,adapter,engine,warm,record,step,True,True,True)
        errors=dict(controls=difference(a['controls'],b['controls'],p['tolerances']['controls']))
        for ta,tb in zip(a['traces'],b['traces']):
            assert ta['elites']==tb['elites'] and ta['winner']==tb['winner'] and ta['category']==tb['category']
            difference(ta['means'],tb['means'],p['tolerances']['controls'])
            difference(ta['deviations'],tb['deviations'],p['tolerances']['controls'])
        for target,m in a['engine'].tracks.items():
            n=b['engine'].tracks[target]
            difference(m.goals,n.goals,p['tolerances']['trajectory'])
            difference(m.logweights,n.logweights,p['tolerances']['weights'])
        positions=a['obs'].robot_xy[None,None]+np.cumsum(np.tile(a['controls'][None],(64,1,1))*.25,axis=1)
        rng=np.random.default_rng(case+step);positions+=rng.normal(0,.3,positions.shape)
        exact=ref.MixtureEnvelope(MPCConfig(**p['config']),a['obs'],a['mixtures'])
        fused=fast.FusedEnvelope(MPCConfig(**p['config']),a['obs'],a['mixtures'])
        errors['risk_exact']=difference(exact.exact(positions),fused.exact(positions),p['tolerances']['risk'])
        for j,(lo,hi) in enumerate(zip(exact.bounds(positions),fused.bounds(positions))):
            errors['risk_bound_'+str(j)]=difference(lo,hi,p['tolerances']['risk'])
        measures=[]
        for repetition in range(p['repetitions']):
            order=(index+repetition)%len(variants)
            for name,behavior,risk in variants[order:]+variants[:order]:
                result=pipeline(env,adapter,engine,warm,record,step,behavior,risk)
                difference(a['controls'],result['controls'],p['tolerances']['controls'])
                measures.append(dict(variant=name,repetition=repetition,**result['timing']))
        row=dict(state=[case,step,tid],people=record['people'],visible=int(np.sum(a['obs'].human_visible)),
            errors=errors,ordered_elites_equal=True,action=a['action'],full_modes=sum(len(v[1]) for v in a['mixtures'].values()),
            measures=measures,diagnostics=a['diagnostics'])
        rows.append(row);save('pipeline.json',rows)
        print('PIPELINE',case,{name:round(np.median([m['total'] for m in measures if m['variant']==name]),2)
                              for name,_,_ in variants},flush=True)


def report():
    rows=load('pipeline.json');table=[]
    for people in (5,10,20):
        selected=[r for r in rows if r['people']==people]
        for variant in ('original','behavior_only','risk_only','combined'):
            samples=[m for r in selected for m in r['measures'] if m['variant']==variant]
            per_state=[np.median([m['total'] for m in r['measures'] if m['variant']==variant]) for r in selected]
            components={k:float(np.median([m[k] for m in samples])) for k in ('observation','goal_update','compression','rollout','risk','mpc_other','mpc','total')}
            table.append(dict(people=people,variant=variant,states=len(selected),components=components,
                p95=float(np.percentile([m['total'] for m in samples],95)),
                late_fraction=float(np.mean([m['total']>250 for m in samples])),state_medians=per_state))
    save('summary.json',table)
    lines=['# Frozen-model computational backend audit','',
        'Same 12 development states and parameters; three technical repeats per backend. No navigation experiment.',
        'The original RVO LP and constraint code is compiled through access-only friend declarations in generated headers.',
        'History contexts cache only state-dependent constraints/TTC. Every proposed goal still replays its own complete likelihood.',
        'Future modes have separate constraints after their states diverge. Neighbor CV predictions and physical constraints are unchanged.',
        'Risk fusion retains SciPy CDF, Hermite nodes/bounds, per-person mode mixture, existence and clipping.',
        '', '| People | Backend | Goal update ms | D rollout ms | Risk ms | MPC other ms | Total p50 ms | p95 ms | >250 ms |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table:
        c=r['components'];lines.append('| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f%% |'%(
            r['people'],r['variant'],c['goal_update'],c['rollout'],c['risk'],c['mpc_other'],c['total'],r['p95'],100*r['late_fraction']))
    lines+=['','Component medians do not add. Compilation, prefix restoration, deepcopy and assertions are outside timing.',
            'Timing includes a fresh legal observation/filter update, goal update, unchanged compression, prediction, risk and complete MPC.',
            'Primary timings start with empty historical context caches; only reuse within the current update is counted.',
            'These are development-state latency samples, not all-step closed-loop real-time certification.',
            'Conclusion: all four computational changes are implemented, but the unchanged full-mode method does not meet the 250 ms budget.',
            'At 20 people the remaining dominant cost is mixture risk, not goal inference or behavior rollout. No representation redesign is included.',
            'Previous negative scientific results and failed posterior/compression precision gates remain unchanged.']
    (OUT/'verdict.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',len(rows),'states',flush=True)


def edges():
    from types import SimpleNamespace
    fixture=json.loads((ROOT/'reproducibility/posterior_fixture.json').read_text())
    old.base.legacy._load_modules(old.base.CROWD)
    results={}
    goals=np.array([[0.,0.],[.1,0.],[3.,1.],[-3.,-1.],[3.,1.]])
    for count in (0,1,12):
        for speed in (0.,.7):
            f=copy.deepcopy(fixture['history'][0]['features'])
            f.update(pos=np.zeros(2),vel=np.array([speed,0.]),radius=.3)
            f['neighbors']=[dict(px=.4*np.cos(i),py=.4*np.sin(i),vx=-.2,vy=.1,radius=.3)
                            for i in range(count)]
            a=old.forward(f,goals,16);b=fast.Behavior().forward(f,goals,16)
            results['behavior_%s_%s'%(count,speed)]=difference(a,b,1e-10)
    for variance in (0.,.04,4.):
        for override in (None,.7):
            print('EDGE_RISK',variance,override,flush=True)
            cfg=MPCConfig(existence_override=override);h=cfg.horizon
            centers=np.zeros((1,h,2));v=np.full(h,variance)
            obs=SimpleNamespace(entities=np.array([[0.,0.,0.,0.,.3]]),robot_radius=.3,
                human_segment_end=centers,human_position_covariance=v[None,:,None,None]*np.eye(2),
                human_existence=np.array([.4]))
            positions=np.zeros((5,h,2));positions[:,:,0]=np.array([0.,.3,1.,8.,100.])[:,None]
            mixture={0:(centers,np.ones(1),v)}
            a=ref.MixtureEnvelope(cfg,obs,mixture);b=fast.FusedEnvelope(cfg,obs,mixture)
            difference(obs.human_existence,[.4],0.)
            try:a.exact(positions)
            except FloatingPointError:
                # Keep the installed reference CDF's exceptional-input behavior.
                try:b.exact(positions)
                except FloatingPointError:pass
                else:raise AssertionError('reference failure not preserved')
                results['reference_cdf_rejection_%s_%s'%(variance,override)]=True
                positions[0,:,0]=.01
            errors=[difference(a.exact(positions),b.exact(positions),1e-10)]
            errors += [difference(x,y,1e-10) for x,y in zip(a.bounds(positions),b.bounds(positions))]
            split={0:(np.repeat(centers,2,axis=0),np.array([.25,.75]),v)}
            c=fast.FusedEnvelope(cfg,obs,split)
            errors.append(difference(b.exact(positions),c.exact(positions),1e-10))
            results['risk_%s_%s'%(variance,override)]=max(errors)
    save('edge_checks.json',results)
    print('EDGES',results,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['kernels','pipelines','report','edges','all'])
    args=parser.parse_args()
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    fast.library()
    if args.command=='edges':edges();return
    records=old.data();calibration=json.loads((old.OUT/'calibration.json').read_text());p=protocol()
    if args.command in ('kernels','all'):kernels(records,calibration,p)
    if args.command in ('pipelines','all'):pipelines(records,calibration,p)
    if args.command in ('report','all'):report()


if __name__=='__main__':main()
