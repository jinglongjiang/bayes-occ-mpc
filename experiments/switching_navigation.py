"""Independent ORCA closed loop with frozen source-data predictors; not human trials."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import joblib
import numpy as np
from experiments import switching_validation as source
from experiments.switching_decision import OUT,ROOT,H,old,save
from experiments import goal_model_probe as envbase
from experiments import intent_repair_run as mixture
from experiments.intent_risk_query import Direct
from integration.crowdnav import BayesObservationAdapter
from nav.contracts import MPCConfig
from nav.planner import MPCPlanner

ARMS=('O','CV','TREE','MAP','IID','FULL')


class DirectPlanner(mixture.MixturePlanner):
    def _build_envelope(self,obs):
        self.risk_seconds=0.;self.geometry_counts=[0,0]
        if not obs.entities.size:return None
        start=time.perf_counter()
        self.mixture_evaluator=Direct(self.cfg,obs,self.mixtures)
        self.risk_seconds=time.perf_counter()-start
        return None


class Online:
    def __init__(self,model,arm):
        self.model=model;self.arm=arm;self.past={};self.first={}

    def update(self,t,detections,obs,ids):
        old.H=H
        features={}
        for entity in detections:
            tid=entity['id'];pos=np.array([entity['px'],entity['py']])
            self.first.setdefault(tid,(t,pos.copy()))
            hist=self.past.setdefault(tid,[])
            if hist and t-hist[-1][0]>.3:hist.clear()
            hist.append((t,pos,obs.robot_xy.copy()))
            if t-hist[0][0]<1.-1e-8:continue
            times=np.array([h[0] for h in hist]);people=np.array([h[1] for h in hist]);robots=np.array([h[2] for h in hist])
            f,rot,cv,ca,_,_=old.prefix_features(times,people,robots)
            ages=np.minimum(np.array([4.,3.,2.,1.,0.]),t-times[0])
            longp=(old.interpolate(times,people,t-ages)-pos)@rot
            longr=(old.interpolate(times,robots,t-ages)-pos)@rot
            feature=np.r_[f,longp.ravel(),longr.ravel(),ages,
                min(t-self.first[tid][0],30.),(pos-self.first[tid][1])@rot,np.zeros(3)]
            seq=source.sequence(times,people,t)
            if len(seq['e']):
                posterior,iid,_=source.filtered(self.model['bank'],seq)
                p,ind=posterior[-1],iid[-1]
            else:p=ind=self.model['bank'].prior
            features[tid]=(feature,rot,cv,p,ind,pos)
        result={};intervals=np.arange(1,17)*.25
        variance=np.interp(intervals,np.r_[0,H],np.r_[0,self.model['variance']])
        for j,tid in enumerate(ids):
            if tid not in features:continue
            f,rot,cv,p,ind,pos=features[tid]
            if self.arm=='CV':paths=cv[None];weights=np.ones(1)
            elif self.arm=='TREE':paths=self.model['pooled'].predict(f[None]).reshape(1,4,2)+cv;weights=np.ones(1)
            else:
                paths=source.expert_predict(self.model['experts'],f[None],cv[None])[0]
                if self.arm=='MAP':paths=paths[p.argmax():p.argmax()+1];weights=np.ones(1)
                else:weights=ind if self.arm=='IID' else p
            expanded=np.array([np.column_stack([np.interp(intervals,np.r_[0,H],np.r_[0,path[:,d]]) for d in range(2)]) for path in paths])
            result[j]=(expanded@rot.T+pos,weights,variance)
        return result,len(features)


def register():
    path=OUT/'navigation_protocol.json'
    if path.exists():return json.loads(path.read_text())
    used,hashes,audit=envbase.history()
    # Unreadable legacy files remain disclosed; new IDs are still screened against parsed registries.
    start=max(8000000,max(used,default=0)+1000)
    cases=[];seen=set()
    for group,(people,scene) in enumerate((n,s) for n in (5,10,20) for s in ('circle_crossing','square_crossing')):
        for i in range(10):
            item=dict(case=start+group*10+i,people=people,scene=scene,index=len(cases))
            env=envbase.environment(item);digest=envbase.layout(env)
            if item['case'] in used or digest in hashes or digest in seen:raise RuntimeError('layout overlap')
            seen.add(digest);item['layout_hash']=digest;cases.append(item)
    cfg=json.loads((ROOT/'archive/hermite_audit/protocol.json').read_text())['config']
    protocol=dict(cases=cases,arms=ARMS,config=cfg,source_commit='0c8b814',
        inference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((OUT/'deployment.joblib').read_bytes()).hexdigest(),
        history_files=audit['files'],history_unreadable=audit['unreadable'],
        cpu_affinity=sorted(os.sched_getaffinity(0)),parallel_workers=6,
        scope='Synchronous independent ORCA/TTC simulator; nonresponsive humans. Not real interactive validation.',
        geometry='All six arms retain the same CV hard geometry; five new arms share source-calibrated variance when eligible.',
        timing='Concurrent exploratory timings, not deployment certification',
        seeds='case*100003+step*97+1729; arm order rotated by layout index')
    save('navigation_protocol.json',protocol)
    return protocol


def episode(item,arm,config,model):
    cfg=MPCConfig(**config);env=envbase.environment(item)
    assert envbase.layout(env)==item['layout_hash']
    adapter=BayesObservationAdapter(cfg)
    planner=MPCPlanner(cfg) if arm=='O' else DirectPlanner(cfg)
    if arm!='O':planner.audit_geometry=True
    engine=Online(model,arm)
    _,_,ActionXY,_,_=envbase.legacy._load_modules(envbase.CROWD)
    steps=[];done=False;overlap=0;step=0
    while not done:
        start=time.perf_counter();obs=adapter.read(env)
        inferred=0;it=0.
        if arm!='O':
            tt=time.perf_counter();planner.mixtures,inferred=engine.update(env.global_time,adapter.detected_entities,obs,adapter.reported_ids)
            it=(time.perf_counter()-tt)*1000
        action,plan_ms=planner.plan(obs,item['case']*100003+step*97+1729)
        speed_excess=max(0.,float(np.linalg.norm(action))-cfg.v_max)
        acceleration_excess=max(0.,float(np.linalg.norm(action-obs.robot_velocity))-cfg.a_max*cfg.dt)
        if not np.isfinite(action).all() or max(speed_excess,acceleration_excess)>1e-6:raise RuntimeError('invalid action')
        command=ActionXY(*map(float,action));elapsed=(time.perf_counter()-start)*1000
        before=np.array(env.robot.get_position());humans=np.array([h.get_position() for h in env.humans])
        radii=np.array([h.radius+env.robot.radius for h in env.humans])
        _,_,term,trunc,info=env.step(command)
        after=np.array(env.robot.get_position());human_after=np.array([h.get_position() for h in env.humans])
        np.testing.assert_allclose(after,before+cfg.dt*action,atol=1e-12,rtol=0)
        clearance=envbase.legacy.swept_min_clearance(before,after,humans,human_after,radii)
        overlap+=int(clearance < -1e-9)
        extent=4. if item['scene']=='circle_crossing' else 5.
        # Spawn-region excursion is not the actuator bound_violations field of legacy reports.
        workspace_excess=max(0.,float(np.max(np.abs(after)))-extent)
        geo=planner.geometry_counts if arm!='O' else [0,0]
        steps.append(dict(step=step,action=action.tolist(),robot_after=after.tolist(),
            human_audit_hash=hashlib.sha256(human_after.tobytes()).hexdigest()[:16],
            visible=len(adapter.detected_entities),eligible=inferred,tracks=len(adapter.reported_ids),
            inference_ms=it,total_ms=elapsed,plan_ms=plan_ms,clearance=clearance,
            speed_excess=speed_excess,acceleration_excess=acceleration_excess,
            workspace_excess=workspace_excess,geometry_rejected=int(geo[0]),risk_permitted=int(geo[1]),
            feasibility=planner.last_diagnostics['feasibility_class']))
        done=term or trunc;step+=1
        if step>110:raise RuntimeError('nonterminating environment')
    event=str(info['event']);collision=int(event=='collision' or overlap>0)
    success=int(event=='reach_goal' and not collision);timeout=int(event=='timeout' and not collision)
    if success+collision+timeout!=1:raise RuntimeError('unclassified outcome')
    return dict(**item,arm=arm,status='ok',event=event,success=success,collision=collision,timeout=timeout,
        penalty=float(env.global_time) if success else 25.,nav_time=float(env.global_time),steps=steps)


def block(item,config):
    model=joblib.load(OUT/'deployment.joblib')
    rotate=item['index']%len(ARMS);order=ARMS[rotate:]+ARMS[:rotate]
    return [episode(item,arm,config,model) for arm in order]


def navigate():
    protocol=register();path=OUT/'navigation.jsonl'
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=protocol['inference_source_sha256']:raise RuntimeError('navigation code changed')
    if hashlib.sha256((OUT/'deployment.joblib').read_bytes()).hexdigest()!=protocol['model_sha256']:raise RuntimeError('model changed')
    existing=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    counts={case:sum(r['case']==case for r in existing) for case in set(r['case'] for r in existing)}
    if any(c!=6 for c in counts.values()):raise RuntimeError('incomplete block in saved data')
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(block,item,protocol['config']):item for item in protocol['cases'] if item['case'] not in counts}
        for future in as_completed(futures):
            item=futures[future]
            try:rows=future.result()
            except Exception as exc:
                save('execution_error.json',dict(item=item,error=repr(exc)));raise
            common=min(len(r['steps']) for r in rows)
            for s in range(common):
                if len({r['steps'][s]['human_audit_hash'] for r in rows})!=1:raise RuntimeError('human paths differ despite invisible robot')
            with path.open('a') as f:
                for r in rows:f.write(json.dumps(r)+'\n')
            print('BLOCK',item['case'],item['people'],item['scene'],[(r['arm'],r['event'],r['penalty']) for r in rows],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','run','smoke'])
    args=parser.parse_args()
    if args.command=='register':register()
    elif args.command=='run':navigate()
    else:
        item=dict(case=0,people=5,scene='circle_crossing',index=0)
        item['layout_hash']=envbase.layout(envbase.environment(item))
        cfg=json.loads((ROOT/'archive/hermite_audit/protocol.json').read_text())['config']
        model=joblib.load(OUT/'deployment.joblib')
        result=episode(item,'FULL',cfg,model)
        save('smoke.json',result);print(result['event'],len(result['steps']),flush=True)
