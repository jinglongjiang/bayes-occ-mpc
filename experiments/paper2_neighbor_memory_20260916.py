"""Oracle-goal conditional-model screen using only past-observed neighbors."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

ROOT=Path('/home/abc/workspace/bayes_occ_mpc_hermite')
sys.path.insert(0,str(ROOT))
from experiments import goal_posterior_probe as p
OUT=Path('/home/abc/temp/paper2_neighbor_memory_20260916')


def native(f,goal,neighbors,policy=None):
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
    if policy is None:
        policy=ORCA()
        policy._last_pref_vel=f['vel'].copy()
    state=JointState(FullState(*f['pos'],*f['vel'],f['radius'],*goal,1.,0.),
        [ObservableState(e['px'],e['py'],e['vx'],e['vy'],e['radius']) for e in neighbors])
    return f['pos']+.25*np.asarray(policy.predict(state))


def remembered_contexts(record):
    legal=p.legal_record(record)
    contexts=dict(p.contexts(legal)); past={}
    for frame in legal['frames']:
        step=frame['step']
        for e in frame['detections']:
            past[e['id']]=(step,dict(e))
        current={e['id'] for e in frame['detections']}
        for tid,f in contexts[step].items():
            missing=[]
            for other,(last,e) in past.items():
                age=.25*(step-last)
                if other==tid or other in current or age>4.:
                    continue
                h=dict(e); h['px']+=age*h['vx']; h['py']+=age*h['vy']; h['age_seconds']=age; missing.append(h)
            f['remembered_neighbors']=f['neighbors']+missing
        yield step,contexts[step]


def native_path(f,goal,neighbors,steps=16,pref=None,dense=False):
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
    policy=ORCA(); policy._last_pref_vel=f['vel'].copy() if pref is None else pref.copy()
    pos=f['pos'].copy();vel=f['vel'].copy(); path=[]
    for k in range(steps):
        state=JointState(FullState(*pos,*vel,f['radius'],*goal,1.,0.),
            [ObservableState(e['px']+.25*k*e['vx'],e['py']+.25*k*e['vy'],e['vx'],e['vy'],e['radius']) for e in neighbors])
        vel=np.asarray(policy.predict(state));pos=pos+.25*vel;path.append(pos.copy())
    return np.array(path) if dense else np.array(path)[np.array(p.base.HORIZONS)-1]


def swept_collision(robot,humans,radius):
    # Exact minimum for each pair of piecewise-linear executed segments.
    rel=humans[None]-robot[:,None]
    delta=np.diff(rel,axis=2);start=rel[:,:,:-1]
    alpha=np.clip(-np.sum(start*delta,axis=-1)/np.maximum(np.sum(delta**2,axis=-1),1e-15),0,1)
    distance=np.linalg.norm(start+alpha[:,:,:,None]*delta,axis=-1).min(axis=2)
    return distance<=radius


def decision_main():
    from types import SimpleNamespace
    from nav.planner import MPCPlanner
    from nav.contracts import MPCConfig
    out=OUT/'decision';out.mkdir(exist_ok=False)
    records=p.data();calibration=json.loads((p.OUT/'calibration.json').read_text())
    chosen=json.loads((OUT/'posterior/summary.json').read_text())['128']['selected_scales']['memory']
    (out/'protocol.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='historical fixed query local decision diagnostic; NOT full navigation or new data',
        posterior='frozen 128-node memory likelihood; scale from existing development selection',
        scale=chosen,horizon_seconds=2,collision_cost=10,
        objective='terminal goal distance + 10 * probability(any swept collision in 2s)',
        candidates='same original holonomic MPC projected speed/direction lattice for every arm',
        arms=['CV','MAP','Mean','ScalarCV','ScalarMean','Moment','FULL','OracleFocal','OracleAll'],
        scalar_grid=[0,.1,.2,.4,.8],moment='256 full joint-time Gaussian samples, same mean/cov as FULL',
        gate='FULL must beat MAP, Moment and development-tuned scalar on actual local cost; fresh confirmation still required',
        caveats=['one focal target belief; remaining people use same legal CV memory',
                 'human futures independent of robot in original environment',
                 'no PPO, no claim about unicycle closed loop']),indent=2))
    selected_pairs={c['case']:{(r['step'],r['tid']) for r in c['rows']} for c in json.loads((OUT/'cases.json').read_text())}
    errors=[]
    for record in records:
        if record['split']!='development':continue
        fs=dict(remembered_contexts(record))
        for step,tid in selected_pairs[record['case']]:
            f=fs[step][tid]
            errors.append(fs[step+1][tid]['pos']-native(f,np.array(record['truth']['goals'][tid]),f['remembered_neighbors']))
    inverse=np.linalg.inv(np.cov(np.array(errors).T)+np.eye(2)*1e-6)
    cfg=dict(json.loads((p.base.OUT/'protocol.json').read_text())['config']);cfg['horizon']=8
    planner=MPCPlanner(MPCConfig(**cfg));all_cases=[]
    for record in records:
        pool=[(s,i) for s,i,f in p.base.queries(record) if f['goal_source']!='birth_antipode' and f['conflict']]
        ix=np.unique(np.linspace(0,len(pool)-1,min(4,len(pool)),dtype=int)) if pool else []
        wanted={pool[j] for j in ix};targets={i for _,i in wanted};tracks={};rows=[]
        for step,fs in remembered_contexts(record):
            for tid in targets & fs.keys():
                f=fs[tid]
                if tid not in tracks:
                    sampler=p.GoalPosterior(record['scene'],step,f,calibration,record['case']*1009+tid*97,count=128)
                    tracks[tid]=dict(goals=sampler.goals,ll=np.zeros(len(sampler.goals)),old=(step,f),updates=0)
                else:
                    tr=tracks[tid];last,old=tr['old']
                    if step==last+1:
                        e=np.array([f['pos']-native(old,g,old['remembered_neighbors']) for g in tr['goals']])
                        d=np.einsum('ni,ij,nj->n',e,inverse,e)
                        tr['ll']+=-3*np.log1p(d/(4*chosen**2));tr['updates']+=1
                    tr['old']=(step,f)
                tr=tracks[tid]
                if (step,tid) not in wanted or not tr['updates']:continue
                w=np.exp(tr['ll']-logsumexp(tr['ll']))
                paths=np.array([native_path(f,g,f['remembered_neighbors'],steps=8,dense=True) for g in tr['goals']])
                mean=np.einsum('n,nhd->hd',w,paths)
                flat=paths.reshape(len(paths),-1);center=flat-mean.ravel()
                cov=(center.T*w)@center
                eig,U=np.linalg.eigh(cov);rng=np.random.default_rng(record['case']*997+step*101+tid)
                z=rng.normal(size=(128,16));z=np.r_[z,-z]
                gaussian=(mean.ravel()+z@(U*np.sqrt(np.maximum(eig,0))).T).reshape(256,8,2)
                frame=record['frames'][step];r=np.array(frame['robot']);goal=np.array([0.,4.])
                obs=SimpleNamespace(robot_xy=r[:2],robot_velocity=r[2:4],goal_xy=goal)
                angle=np.arctan2(*(goal-r[:2])[::-1]);directions=angle+np.array([0.,-.392699,.392699,-.785398,.785398,-1.570796,1.570796])
                velocities=np.vstack([np.zeros(2),np.concatenate([v*np.c_[np.cos(directions),np.sin(directions)] for v in (.25,.5,.75,1.)])])
                controls=np.repeat(velocities[:,None],8,axis=1)
                _,_,rp=planner._rollout(controls,obs);rp=np.concatenate([np.repeat(r[None,None,:2],len(rp),axis=0),rp],axis=1)
                terminal=np.linalg.norm(rp[:,-1]-goal,axis=1)
                others=f['remembered_neighbors'];times=.25*np.arange(9)
                hp=np.array([np.array([e['px'],e['py']])+times[:,None]*np.array([e['vx'],e['vy']]) for e in others]).reshape(-1,9,2)
                other_risk=swept_collision(rp,hp,np.array([r[4]+e['radius'] for e in others])[None]).any(axis=1) if len(hp) else np.zeros(len(rp),bool)
                def risk(future,weights=None,margin=0.):
                    future=np.asarray(future).reshape(-1,8,2)
                    h=np.concatenate([np.repeat(f['pos'][None,None],len(future),axis=0),future],axis=1)
                    hit=swept_collision(rp,h,r[4]+f['radius']+margin)
                    value=hit.mean(axis=1) if weights is None else hit@weights
                    return np.where(other_risk,1.,value)
                truth=np.asarray(record['truth']['positions'][step:step+9]).transpose(1,0,2)
                actual=swept_collision(rp,truth,r[4]+.3).any(axis=1)
                actual_cost=terminal+10*actual
                cv=f['pos']+times[1:,None]*f['vel']
                risks=dict(CV=risk(cv),MAP=risk(paths[w.argmax()]),Mean=risk(mean),
                    FULL=risk(paths,w),Moment=risk(gaussian),OracleFocal=risk(truth[tid,1:]),OracleAll=actual.astype(float))
                for margin in (0,.1,.2,.4,.8):
                    risks['ScalarCV_'+str(margin)]=risk(cv,margin=margin)
                    risks['ScalarMean_'+str(margin)]=risk(mean,margin=margin)
                choices={k:int(np.argmin(terminal+10*v)) for k,v in risks.items()}
                rows.append(dict(step=step,tid=tid,choices=choices,
                    cost={k:float(actual_cost[j]) for k,j in choices.items()},
                    collision={k:bool(actual[j]) for k,j in choices.items()},best=float(actual_cost.min()),
                    full_moment_risk_max_difference=float(np.max(abs(risks['FULL']-risks['Moment'])))))
        all_cases.append(dict(case=record['case'],split=record['split'],rows=rows))
        (out/f"case_{record['case']}.json").write_text(json.dumps(all_cases[-1]));print('DECISION',record['case'],len(rows),flush=True)
    development=[c for c in all_cases if c['split']=='development' and c['rows']]
    holdout=[c for c in all_cases if c['split']=='holdout' and c['rows']]
    selected={}
    for family in ('ScalarCV','ScalarMean'):
        keys=[family+'_'+str(m) for m in (0,.1,.2,.4,.8)]
        selected[family]=min(keys,key=lambda k:np.mean([np.mean([r['cost'][k] for r in c['rows']]) for c in development]))
    arms=['CV','MAP','Mean','FULL','Moment','OracleFocal','OracleAll']+list(selected.values())
    vals={k:np.array([np.mean([r['cost'][k] for r in c['rows']]) for c in holdout]) for k in arms}
    stats={k:dict(cost=float(v.mean()),collision=float(np.mean([np.mean([r['collision'][k] for r in c['rows']]) for c in holdout])),
        regret=float(np.mean([np.mean([r['cost'][k]-r['best'] for r in c['rows']]) for c in holdout]))) for k,v in vals.items()}
    rng=np.random.default_rng(2407);comparisons={}
    for k in arms:
        diff=vals['FULL']-vals[k]
        comparisons[k]=dict(mean=float(diff.mean()),ci95=np.quantile(rng.choice(diff,(10000,len(diff))).mean(1),[.025,.975]).tolist())
    result=dict(selected=selected,holdout_cases=len(holdout),holdout_queries=sum(len(c['rows']) for c in holdout),
        results=stats,full_minus=comparisons)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


def uncertain_neighbors(f,seed,count=8):
    rng=np.random.default_rng(seed)
    worlds=[[dict(e) for e in f['remembered_neighbors']] for _ in range(count)]
    for j,e in enumerate(f['remembered_neighbors']):
        age=e.get('age_seconds',0.)
        if age==0:continue
        m=age/.25; q=.55**2
        pp=q*.25**4*m*(4*m*m-1)/12
        vv=q*.25**2*m; pv=q*.25**3*m*m/2
        cov=np.array([[pp,0,pv,0],[0,pp,0,pv],[pv,0,vv,0],[0,pv,0,vv]])
        z=rng.normal(size=(count//2,4));z=np.r_[z,-z]@np.linalg.cholesky(cov+np.eye(4)*1e-12).T
        for k in range(count):
            for d,key in enumerate(('px','py','vx','vy')):worlds[k][j][key]+=z[k,d]
    return worlds


def posterior_main(marginal=False):
    out=OUT/('marginal' if marginal else 'posterior');out.mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),particles=[64,128],
        scales=[1,2,4,8],queries_per_case=4,ttl=4,
        prediction='same native ORCA + remembered neighbors for all goal distributions',
        inference='visible versus remembered likelihood, plus uncertain-neighbor integral' if marginal else 'visible likelihood versus remembered-neighbor likelihood',
        nuisance_samples=8 if marginal else 0,
        approximation='per-step hidden-neighbor CV Gaussian marginal; not exact joint filtering' if marginal else 'point memory',
        selection='development mean energy; test layouts excluded; historical data not fresh',
        gate='memory FULL must beat visible FULL and memory MAP on point/energy, >=2% and CI upper<0',
        oracle='true goals used ONLY to fit training residual scale and separate diagnostic')))
    records=p.data(); calibration=json.loads((p.OUT/'calibration.json').read_text())
    # Calibrate each forward model only on training layouts, using the same selected pairs.
    pairs=json.loads((OUT/'cases.json').read_text()); selection={c['case']:{(r['step'],r['tid']) for r in c['rows']} for c in pairs}
    errors={k:[] for k in ('visible','memory')}
    for record in records:
        if record['split']!='development':continue
        fs=dict(remembered_contexts(record))
        for step,tid in sorted(selection[record['case']]):
            f=fs[step][tid]; goal=np.array(record['truth']['goals'][tid]); target=fs[step+1][tid]['pos']
            for name,key in [('visible','neighbors'),('memory','remembered_neighbors')]:
                errors[name].append(target-native(f,goal,f[key]))
    inverses={k:np.linalg.inv(np.cov(np.array(v).T)+np.eye(2)*1e-6) for k,v in errors.items()}
    if marginal:inverses['marginal']=inverses['memory'].copy()
    scales=np.array([1.,2.,4.,8.]); results=[]
    for record in records:
        pool=[(s,i) for s,i,f in p.base.queries(record) if f['goal_source']!='birth_antipode' and f['conflict']]
        ids=np.unique(np.linspace(0,len(pool)-1,min(4,len(pool)),dtype=int)) if pool else []
        wanted={pool[i] for i in ids};targets={i for _,i in wanted};tracks={};rows=[]
        for step,fs in remembered_contexts(record):
            for tid in targets & fs.keys():
                f=fs[tid]
                if tid not in tracks:
                    sampler=p.GoalPosterior(record['scene'],step,f,calibration,record['case']*1009+tid*97,count=128)
                    tracks[tid]=dict(goals=sampler.goals,previous=(step,f),scores={k:np.zeros((4,len(sampler.goals))) for k in inverses},updates=0)
                else:
                    tr=tracks[tid];last,old=tr['previous']
                    if step==last+1:
                        for name,key in [('visible','neighbors'),('memory','remembered_neighbors')]:
                            e=np.array([f['pos']-native(old,g,old[key]) for g in tr['goals']])
                            d=np.einsum('ni,ij,nj->n',e,inverses[name],e)
                            tr['scores'][name] += -3*np.log1p(d[None]/(4*scales[:,None]**2))
                        if marginal:
                            worlds=uncertain_neighbors(old,record['case']*100003+last*97+tid)
                            ll=[]
                            for world in worlds:
                                e=np.array([f['pos']-native(old,g,world) for g in tr['goals']])
                                d=np.einsum('ni,ij,nj->n',e,inverses['marginal'],e)
                                ll.append(-3*np.log1p(d[None]/(4*scales[:,None]**2)))
                            tr['scores']['marginal']+=logsumexp(ll,axis=0)-np.log(len(ll))
                        tr['updates']+=1
                    tr['previous']=(step,f)
                if (step,tid) not in wanted or tracks[tid]['updates']==0:continue
                tr=tracks[tid];positions=np.array([native_path(f,g,f['remembered_neighbors']) for g in tr['goals']])
                truth=np.array([record['truth']['positions'][step+h][tid] for h in p.base.HORIZONS])
                pointdist=np.linalg.norm(positions-truth,axis=2)
                pair=np.array([cdist(positions[:,h],positions[:,h]) for h in range(4)])
                output={}
                for n in (64,128):
                    arms={}
                    for name in inverses:
                        ll=tr['scores'][name][:,:n];w=np.exp(ll-logsumexp(ll,axis=1)[:,None])
                        mean=np.einsum('kn,nhd->khd',w,positions[:n])
                        energy=(w@pointdist[:n]-.5*np.einsum('kn,hnm,km->kh',w,pair[:,:n,:n],w)).mean(1)
                        arms[name]=dict(point=np.linalg.norm(mean-truth,axis=2).mean(1).tolist(),energy=energy.tolist(),
                            map=pointdist[np.argmax(w,axis=1)].mean(1).tolist())
                    output[str(n)]=arms
                cv=f['pos']+np.array(p.base.HORIZONS)[:,None]*.25*f['vel']
                rows.append(dict(step=step,tid=tid,updates=tr['updates'],results=output,
                    cv=float(np.linalg.norm(cv-truth,axis=1).mean())))
        item=dict(case=record['case'],split=record['split'],rows=rows);results.append(item)
        (out/f"case_{record['case']}.json").write_text(json.dumps(item))
        print('posterior',record['case'],len(rows),flush=True)
    summary={}
    for n in ('64','128'):
        dev=[c for c in results if c['split']=='development' and c['rows']]
        test=[c for c in results if c['split']=='holdout' and c['rows']]
        selected={name:int(np.argmin(np.mean([np.mean([r['results'][n][name]['energy'] for r in c['rows']],axis=0) for c in dev],axis=0))) for name in inverses}
        vals={}
        for name,idx in selected.items():
            for metric in ('point','energy','map'):
                vals[name+'_'+metric]=np.array([np.mean([r['results'][n][name][metric][idx] for r in c['rows']]) for c in test])
        rng=np.random.default_rng(2407); comparisons={}
        for a,b in [('memory_point','visible_point'),('memory_energy','visible_energy'),('memory_point','memory_map')]:
            diff=vals[a]-vals[b];ci=np.quantile(rng.choice(diff,(10000,len(diff))).mean(1),[.025,.975])
            comparisons[a+'-'+b]=dict(mean=float(diff.mean()),ci95=ci.tolist(),relative_improvement=float(-diff.mean()/vals[b].mean()))
        if marginal:
            for a,b in [('marginal_point','memory_point'),('marginal_energy','memory_energy'),('marginal_point','marginal_map')]:
                diff=vals[a]-vals[b];ci=np.quantile(rng.choice(diff,(10000,len(diff))).mean(1),[.025,.975])
                comparisons[a+'-'+b]=dict(mean=float(diff.mean()),ci95=ci.tolist(),relative_improvement=float(-diff.mean()/vals[b].mean()))
        summary[n]=dict(selected_scales={k:float(scales[v]) for k,v in selected.items()},
            holdout_layouts=len(test),holdout_queries=sum(len(c['rows']) for c in test),
            means={k:float(v.mean()) for k,v in vals.items()},comparisons=comparisons,
            cv=float(np.mean([np.mean([r['cv'] for r in c['rows']]) for c in test])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


def stateful_main():
    from crowd_sim.envs.policy.orca import ORCA
    out=OUT/'stateful';out.mkdir(exist_ok=False)
    records=p.data();calibration=json.loads((p.OUT/'calibration.json').read_text())
    selection={c['case']:{(r['step'],r['tid']) for r in c['rows']} for c in json.loads((OUT/'cases.json').read_text())}
    errors=[]
    for r in records:
        if r['split']!='development':continue
        tracks={}
        for step,fs in remembered_contexts(r):
            for tid,f in fs.items():
                old=tracks.get(tid)
                if old is not None and old[0]==step-1:
                    if (step-1,tid) in selection[r['case']]:errors.append(f['pos']-old[2])
                    policy=old[1]
                else:policy=ORCA();policy._last_pref_vel=f['vel'].copy()
                pred=native(f,np.array(r['truth']['goals'][tid]),f['remembered_neighbors'],policy)
                tracks[tid]=(step,policy,pred)
    inverse=np.linalg.inv(np.cov(np.array(errors).T)+np.eye(2)*1e-6)
    (out/'protocol.json').write_text(json.dumps(dict(scales=[1,2,4,8],particles=[64,128],
        update='one ORCA internal preferred-velocity state per goal; observation gaps reset; no private state read',
        covariance=np.linalg.inv(inverse).tolist(),calibration_count=len(errors),
        future='all posterior arms share native stateful ORCA forward; neighbors legal CV memory',
        scope='diagnostic model consistency, not novelty claim')))
    scales=np.array([1.,2.,4.,8.]);all_cases=[]
    for r in records:
        pool=[(s,i) for s,i,f in p.base.queries(r) if f['goal_source']!='birth_antipode' and f['conflict']]
        ids=np.unique(np.linspace(0,len(pool)-1,min(4,len(pool)),dtype=int)) if pool else []
        wanted={pool[i] for i in ids};targets={i for _,i in wanted};tracks={};rows=[]
        for step,fs in remembered_contexts(r):
            for tid in targets & fs.keys():
                f=fs[tid]
                if tid not in tracks:
                    sampler=p.GoalPosterior(r['scene'],step,f,calibration,r['case']*1009+tid*97,count=128)
                    policies=[ORCA() for _ in sampler.goals]
                    for model in policies:model._last_pref_vel=f['vel'].copy()
                    tr=dict(goals=sampler.goals,policies=policies,step=step,updates=0,ll=np.zeros((4,len(policies))))
                    tracks[tid]=tr
                else:
                    tr=tracks[tid]
                    if step==tr['step']+1:
                        e=f['pos']-tr['pred'];d=np.einsum('ni,ij,nj->n',e,inverse,e)
                        tr['ll']+=-3*np.log1p(d[None]/(4*scales[:,None]**2));tr['updates']+=1
                    else:
                        for model in tr['policies']:model._last_pref_vel=f['vel'].copy()
                pref=[model._last_pref_vel.copy() for model in tr['policies']]
                tr['pred']=np.array([native(f,g,f['remembered_neighbors'],model) for g,model in zip(tr['goals'],tr['policies'])]);tr['step']=step
                if (step,tid) not in wanted or tr['updates']==0:continue
                paths=np.array([native_path(f,g,f['remembered_neighbors'],pref=v) for g,v in zip(tr['goals'],pref)])
                truth=np.array([r['truth']['positions'][step+h][tid] for h in p.base.HORIZONS])
                pdist=np.linalg.norm(paths-truth,axis=2);pair=np.array([cdist(paths[:,h],paths[:,h]) for h in range(4)])
                arms={}
                for n in (64,128):
                    ll=tr['ll'][:,:n];w=np.exp(ll-logsumexp(ll,axis=1)[:,None])
                    mean=np.einsum('kn,nhd->khd',w,paths[:n]);energy=(w@pdist[:n]-.5*np.einsum('kn,hnm,km->kh',w,pair[:,:n,:n],w)).mean(1)
                    arms[str(n)]=dict(point=np.linalg.norm(mean-truth,axis=2).mean(1).tolist(),energy=energy.tolist(),map=pdist[np.argmax(w,axis=1)].mean(1).tolist())
                rows.append(dict(step=step,tid=tid,results=arms))
        case=dict(case=r['case'],split=r['split'],rows=rows);all_cases.append(case)
        (out/f"case_{r['case']}.json").write_text(json.dumps(case));print('stateful',r['case'],len(rows),flush=True)
    summary={};rng=np.random.default_rng(2407)
    for n in ('64','128'):
        dev=[c for c in all_cases if c['split']=='development' and c['rows']];test=[c for c in all_cases if c['split']=='holdout' and c['rows']]
        idx=int(np.argmin(np.mean([np.mean([r['results'][n]['energy'] for r in c['rows']],axis=0) for c in dev],axis=0)))
        vals={k:np.array([np.mean([r['results'][n][k][idx] for r in c['rows']]) for c in test]) for k in ('point','energy','map')}
        diff=vals['point']-vals['map']
        summary[n]=dict(scale=float(scales[idx]),layouts=len(test),queries=sum(len(c['rows']) for c in test),
            means={k:float(v.mean()) for k,v in vals.items()},full_minus_map=float(diff.mean()),
            ci95=np.quantile(rng.choice(diff,(10000,len(diff))).mean(1),[.025,.975]).tolist())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'protocol.json').write_text(json.dumps(dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ttl_seconds=4,max_pairs_per_layout=32,uses_true_goal='oracle conditional-model diagnosis ONLY',
        inputs='legal detections and history only for neighbors; no private response memory',
        selection='evenly spaced consecutive visible unknown-birth-goal transitions'),indent=2))
    records=p.data(); result=[]
    for record in records:
        legal=p.legal_record(record)
        features=dict(p.contexts(legal)); candidates=[]
        for step, fs in features.items():
            for tid,f in fs.items():
                if tid in features.get(step+1,{}) and f['goal_source']!='birth_antipode':
                    candidates.append((step,tid))
        indices=np.unique(np.linspace(0,len(candidates)-1,min(32,len(candidates)),dtype=int)) if candidates else []
        chosen={candidates[i] for i in indices}; past={}; rows=[]
        for frame in legal['frames']:
            step=frame['step']
            for e in frame['detections']:
                past[e['id']]=(step,dict(e))
            current={e['id'] for e in frame['detections']}
            for tid,f in features[step].items():
                if (step,tid) not in chosen:
                    continue
                memory=[]
                for other,(last,e) in past.items():
                    age=.25*(step-last)
                    if other==tid or other in current or age>4.:
                        continue
                    h=dict(e);h['px']+=age*h['vx'];h['py']+=age*h['vy'];memory.append(h)
                # Truth is supplied only to this diagnostic, never to neighbor reconstruction.
                goal=np.asarray(record['truth']['goals'][tid])
                target=features[step+1][tid]['pos']
                prediction=dict(old_D=p.forward(f,[goal])[0,0],
                    visible=native(f,goal,f['neighbors']),
                    memory=native(f,goal,f['neighbors']+memory))
                rows.append(dict(step=step,tid=tid,remembered=len(memory),
                    errors={k:float(np.linalg.norm(v-target)) for k,v in prediction.items()}))
        result.append(dict(case=record['case'],split=record['split'],rows=rows))
        print('done',record['case'],len(rows),flush=True)
    (OUT/'cases.json').write_text(json.dumps(result))
    summary={}
    for split in ('development','holdout'):
        cases=[c for c in result if c['split']==split and c['rows']]
        means={arm:np.array([np.mean([r['errors'][arm] for r in c['rows']]) for c in cases]) for arm in ('old_D','visible','memory')}
        delta=means['memory']-means['visible'];rng=np.random.default_rng(20260916)
        ci=np.quantile(rng.choice(delta,(10000,len(delta))).mean(1),[.025,.975])
        summary[split]=dict(layouts=len(cases),pairs=sum(len(c['rows']) for c in cases),
            means={k:float(v.mean()) for k,v in means.items()},memory_minus_visible=float(delta.mean()),ci95=ci.tolist(),
            pairs_with_memory=sum(r['remembered']>0 for c in cases for r in c['rows']))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    if '--decision' in sys.argv:
        p.data()
        decision_main()
    elif '--stateful' in sys.argv:
        p.data()
        stateful_main()
    elif '--marginal' in sys.argv:posterior_main(marginal=True)
    elif '--posterior' in sys.argv:posterior_main()
    else:main()
