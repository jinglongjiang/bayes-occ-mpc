"""Frozen PPO density confirmation using tensor-only deterministic inference."""
import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import sys
import time
import types
import zipfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path('/home/abc/workspace/bayes_set_continuous/CrowdNav')
sys.path.insert(0,str(REPO))
from crowd_nav.bayes_continuous.environment import BeliefEnv

SOURCE = REPO/'repair_results/final_belief_ppo_qualification_20260915'
PARAMS = REPO/'repair_results/params'
OUT = Path('/home/abc/temp/paper2_ppo_density_20260916')
SCENARIOS = ['baseline_circle','baseline_square','dense_circle','dense_square','large_circle','large_square']


class FeatureBase(nn.Module):
    def __init__(self, observation_space, features_dim):
        super().__init__()
        self._features_dim=features_dim


def production_class(file, name, namespace):
    tree=ast.parse(file.read_text())
    selected=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==name]
    assert len(selected)==1
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(file),'exec'),namespace)
    return namespace[name]


ActionHistory = production_class(REPO/'crowd_nav/bayes_continuous/train_smoke.py','ActionHistory',
                                 dict(gym=gym,np=np))
SetEncoder = production_class(REPO/'crowd_nav/bayes_continuous/network.py','SetEncoder',
                              dict(torch=torch,nn=nn,BaseFeaturesExtractor=FeatureBase))


class Actor:
    def __init__(self, checkpoint, space):
        with zipfile.ZipFile(checkpoint) as archive:
            self.weights=torch.load(archive.open('policy.pth'),map_location='cpu',weights_only=True)
        w=self.weights
        self.encoder=SetEncoder(space,192).eval()
        self.encoder.load_state_dict({k[len('pi_features_extractor.'):]:v for k,v in w.items() if k.startswith('pi_features_extractor.')},strict=True)

    def linear(self,x,prefix):
        return F.linear(x,self.weights[prefix+'.weight'],self.weights[prefix+'.bias'])

    def predict(self,obs,check=False):
        with torch.no_grad():
            batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}
            state=self.encoder(batch)
            if check:
                p='pi_features_extractor.'
                h=F.relu(self.linear(F.relu(self.linear(batch['humans'],p+'human.0')),p+'human.2'))
                r=F.relu(self.linear(batch['robot'],p+'robot.0'))
                m=batch['mask']>.5
                att=((h*self.linear(r,p+'query')[:,None]).sum(-1)/8).masked_fill(~m,-1e9).softmax(-1)*m
                att=att/att.sum(-1,keepdim=True).clamp_min(1e-8)
                maximum=h.masked_fill(~m[...,None],-torch.inf).amax(1)
                maximum=torch.where(m.sum(1,keepdim=True)>0,maximum,torch.zeros_like(maximum))
                alternate=F.relu(self.linear(torch.cat([r,(h*att[...,None]).sum(1),maximum],dim=1),p+'fuse.0'))
                torch.testing.assert_close(state,alternate,rtol=0,atol=1e-6)
            state=F.relu(self.linear(state,'mlp_extractor.policy_net.0'))
            state=F.relu(self.linear(state,'mlp_extractor.policy_net.2'))
            action=self.weights['action_net.1.center']+self.weights['action_net.1.scale']*torch.tanh(self.linear(state,'action_net.0'))
            return np.clip(action[0].numpy(),[0,-1.2],[1,1.2]).astype(np.float32)


def episode(env,actor,layout,case,profile,check=False,geometry_check=False):
    obs,_=env.reset(options=dict(layout_seed=layout,test_case=case,profile=profile))
    actual=len(env.unwrapped.world.env.humans)
    assert int(obs['mask'].sum())==actual
    geometry={}
    if geometry_check:
        world=env.unwrapped.world.env
        r=world.robot
        humans=world.humans
        robot_clearance=min(np.hypot(h.px-r.px,h.py-r.py)-h.radius-r.radius for h in humans)
        human_clearance=min(np.hypot(a.px-b.px,a.py-b.py)-a.radius-b.radius
                            for i,a in enumerate(humans) for b in humans[i+1:])
        assert min(robot_clearance,human_clearance)>=world.discomfort_dist-1e-9
        geometry=dict(initial_robot_clearance=float(robot_clearance),initial_human_clearance=float(human_clearance))
    fingerprint=hashlib.sha256(np.asarray([[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref]
        for h in env.unwrapped.world.env.humans],np.float64).tobytes()).hexdigest()
    for _ in range(140):
        action=actor.predict(obs,check=check)
        obs,_,done,truncated,info=env.step(action)
        if done or truncated:
            return dict(info['episode_result'],layout_sha256=fingerprint,humans=actual,**geometry)
    raise AssertionError('140-step terminal contract')


def preflight():
    torch.set_num_threads(1)
    env=ActionHistory(BeliefEnv(PARAMS,arm='full',training=False),route=True)
    checkpoint=SOURCE/'2407_full/attempt0_20480.zip'
    actor=Actor(checkpoint,env.observation_space)
    receipt=[]
    for profile,file in [('nominal','attempt0_20480.json'),('train_nonstationary','attempt0_20480_nonstationary.json')]:
        reference=json.loads((checkpoint.parent/file).read_text())['records'][:5]
        for expected in reference:
            got=episode(env,actor,expected['layout_seed'],expected['test_case'],profile,check=True)
            for key in ['layout_sha256','outcome','steps','bound_violations']:
                assert got[key]==expected[key],(profile,key,got,expected)
            assert abs(got['reward']-expected['reward'])<2e-5,(got,expected)
            receipt.append(dict(profile=profile,layout=got['layout_seed'],reward_error=got['reward']-expected['reward']))
    env.close()
    return dict(passed=True,reference_episodes=receipt,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                notes='Production AST encoder/history; tensor-only PPO mean; no SB3 pickle or training')


def sequential_spawn(self,human_num,rule):
    assert rule in ('circle_crossing','square_crossing')
    self.humans=[]
    generate=self.generate_circle_crossing_human if rule=='circle_crossing' else self.generate_square_crossing_human
    for _ in range(human_num):
        self.humans.append(generate())


def worker(task):
    seed,arm,scenario,count,index=task[:5]
    corrected=len(task)>5 and task[5]
    torch.set_num_threads(1)
    destination=OUT/'sequential_spawn' if corrected else OUT
    dest=destination/('%s_%s_%s.json'%(seed,arm,scenario))
    if dest.exists():
        cached=json.loads(dest.read_text())
        assert len(cached['records'])==count
        return cached['summary']
    env=ActionHistory(BeliefEnv(PARAMS,arm=arm,training=False,scenario=scenario,seed=2407),route=True)
    if corrected:
        world=env.unwrapped.world.env
        world.generate_random_human_position=types.MethodType(sequential_spawn,world)
    checkpoint=SOURCE/('%s_%s'%(seed,arm))/'attempt0_20480.zip'
    actor=Actor(checkpoint,env.observation_space)
    start=time.monotonic()
    records=[]
    for i in range(count):
        records.append(episode(env,actor,120000000+index*10000+i,400000+index*1000+i,
                               'heldout_nonstationary',geometry_check=corrected))
    env.close()
    summary=dict(seed=seed,arm=arm,scenario=scenario,cases=count,
                success=sum(r['outcome']=='success' for r in records),collision=sum(r['outcome']=='collision' for r in records),
                timeout=sum(r['outcome']=='timeout' for r in records),mean_return=float(np.mean([r['reward'] for r in records])),
                elapsed_seconds=time.monotonic()-start)
    dest.write_text(json.dumps(dict(summary=summary,records=records,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),indent=2))
    print(json.dumps(summary),flush=True)
    return summary


def forecast_case(task):
    from crowd_nav.bayes_continuous.teacher import PlannerObservation, UnicycleConfig, UnicycleCEMMPC
    scenario, case = task
    torch.set_num_threads(1)
    dest=OUT/'dense_forecast'/('%s_%s.json'%(scenario,case))
    if dest.exists():
        return json.loads(dest.read_text())
    env=ActionHistory(BeliefEnv(PARAMS,arm='no_belief',training=False,scenario=scenario,seed=2407),route=True)
    world=env.unwrapped.world
    world.env.generate_random_human_position=types.MethodType(sequential_spawn,world.env)
    checkpoint=SOURCE/'2407_no_belief/attempt0_20480.zip'
    actor=Actor(checkpoint,env.observation_space)
    planner=UnicycleCEMMPC(UnicycleConfig(horizon=8,omega_max=1.2,human_margin=.5))
    def reset():
        o,_=env.reset(options=dict(layout_seed=case,test_case=case,profile='heldout_nonstationary'))
        assert not world.robot.visible
        return o
    def humans():
        return np.asarray([[h.px,h.py,h.vx,h.vy,h.radius] for h in world.env.humans],np.float64)
    def state_hash(o):
        r=world.robot
        values=[np.asarray([r.px,r.py,r.vx,r.vy,r.theta,world.env.global_time],np.float64).tobytes(),humans().tobytes()]
        values.extend(np.asarray(o[k]).tobytes() for k in sorted(o))
        return hashlib.sha256(b''.join(values)).hexdigest()
    def observation(current):
        r=world.robot
        return PlannerObservation(np.array([r.px,r.py]),np.array([r.vx,r.vy]),r.radius,
            np.array([r.gx,r.gy]),current,None,None,None,None,None,None,None,None,'dense_forecast_diagnostic',r.theta)
    obs=reset(); prefix=[];frames=[];anchor=None
    layout_sha=hashlib.sha256(np.asarray([[h.px,h.py,h.gx,h.gy,h.radius,h.v_pref] for h in world.env.humans],np.float64).tobytes()).hexdigest()
    times=np.arange(1,9)*.25
    for t in range(140):
        current=humans();frames.append(current.copy());a=actor.predict(obs)
        if 3<=t<=131:
            po=observation(current)
            sample=np.repeat(a[None,None,:],8,axis=1).astype(float);sample[:,:,1]*=.25
            _,vel,pos=planner._rollout(sample,po)
            cv=current[:,None,:2]+current[:,None,2:4]*times[None,:,None]
            po.human_segment_start=np.concatenate([current[:,None,:2],cv[:,:-1]],axis=1);po.human_segment_end=cv
            clearance=planner._human_clearance(vel,po)
            if float(clearance.min())<.5:
                anchor=dict(step=t,state_hash=state_hash(obs),trigger_clearance=float(clearance.min()))
                break
        prefix.append(a.copy());obs,_,d,tr,info=env.step(a)
        if d or tr:break
    if anchor is None:
        result=dict(case=case,scenario=scenario,layout_sha=layout_sha,eligible=False,outcome=info['outcome'])
        dest.write_text(json.dumps(result,indent=2));env.close();return result
    t=anchor['step']
    def replay():
        o=reset()
        for a in prefix:
            o,_,d,tr,_=env.step(a);assert not(d or tr)
        assert state_hash(o)==anchor['state_hash']
        return o
    initial=actor.predict(obs);current=humans();po=observation(current)
    bank=[np.clip(initial+np.array([dv,dw]),[0,-1.2],[1,1.2]) for dv in (-.2,0,.2) for dw in (-.4,0,.4)]
    bank.append(np.array([0.,0.]))
    samples=np.repeat(np.asarray(bank)[:,None,:],8,axis=1);samples[:,:,1]*=.25
    controls,velocities,positions=planner._rollout(samples,po)
    executable=controls.copy();executable[:,:,1]/=.25
    cv=current[:,None,:2]+current[:,None,2:4]*times[None,:,None]
    acc=(frames[t][:,2:4]-frames[t-3][:,2:4])/.75
    acc*=np.minimum(1.,1./np.maximum(np.linalg.norm(acc,axis=1,keepdims=True),1e-12))
    history=cv+acc[:,None,:]*(times-1+np.exp(-times))[None,:,None]
    future=[]
    for k in range(8):
        world.scheduler.advance(world.policies)
        people=world.env.humans
        acts=[h.act([other.get_observable_state() for other in people if other is not h]) for h in people]
        for h,a in zip(people,acts):h.step(a)
        world.env.global_time+=.25
        future.append(humans()[:,:2].copy())
    oracle=np.stack(future,axis=1);selections={};forecasts={}
    for name,pred in [('current',cv),('history',history),('oracle',oracle)]:
        po.human_segment_start=np.concatenate([current[:,None,:2],pred[:,:-1]],axis=1);po.human_segment_end=pred
        clearance=planner._human_clearance(velocities,po)
        cost=planner._cost(velocities,po,positions,clearance,None,None)
        assert np.isfinite(cost).all()
        selections[name]=int(np.argmin(cost))
        forecasts[name]=dict(costs=cost.tolist(),min_clearance=clearance.min(axis=(1,2)).tolist(),
            endpoint_error=float(np.linalg.norm(pred[:,-1]-oracle[:,-1],axis=1).mean()))
    branches=[]
    for j in range(11):
        o=replay();ret=0.;minimum=100.
        for k in range(140-t):
            a=executable[j,k].astype(np.float32) if j<10 and k<8 else actor.predict(o)
            o,r,d,tr,info=env.step(a);ret+=(.99**k)*r
            minimum=min(minimum,float(info['actual_clearance']))
            if k<8:np.testing.assert_allclose(humans()[:,:2],oracle[:,k],atol=2e-6,rtol=0)
            if d or tr:break
        assert d or tr
        branches.append(dict(candidate=j,return_value=float(ret),outcome=info['outcome'],steps=k+1,min_clearance=minimum))
    result=dict(case=case,scenario=scenario,layout_sha=layout_sha,eligible=True,anchor=anchor,
        selections=selections,forecasts=forecasts,branches=branches,
        history= [f.tolist() for f in frames[-4:]],oracle_future=oracle.tolist(),controls=executable.tolist(),
        prefix_actions=[a.tolist() for a in prefix],replay_exact=True,human_future_verified=True)
    dest.write_text(json.dumps(result,indent=2));env.close()
    print('DENSE_FORECAST',scenario,case,t,[branches[selections[n]]['outcome'] for n in ['current','history','oracle']],flush=True)
    return result


def dense_forecast(workers):
    destination=OUT/'dense_forecast';destination.mkdir(exist_ok=True)
    scenarios=['dense_circle','large_circle','dense_square','large_square']
    protocol=dict(cases_per_scene=20,scenarios=scenarios,case_start=984000,
        profile='heldout_nonstationary',spawn='sequential append, initial overlaps fixed',
        actor='2407_no_belief/attempt0_20480.zip frozen',
        checkpoint_sha=hashlib.sha256((SOURCE/'2407_no_belief/attempt0_20480.zip').read_bytes()).hexdigest(),
        trigger='first t in [3,131] with current actor action held 2s CV swept clearance <0.5m; no future/outcome selection',
        candidates='3x3 dv +/-.2 dw +/-.4 plus stop; existing acceleration projection; execute2s then frozen actor; baseline feedback branch',
        selectors='identical existing unicycle cost, margin .5; CV vs four-frame decayed acceleration vs exact2s human future',
        gate='Oracle-Current mean return >=.02 and paired scene-stratified episode bootstrap CI lower>0; no collision increase. Same separate gate for History-Current.',
        limitation='privileged forecast with fixed cost is not optimal information upper bound, legal inference, or Bayesian necessity; heldout here used as research data, not final untouched paper test',
        production_changed=False,training_updates=0,script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=destination/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    tasks=[(s,984000+100*i+j) for i,s in enumerate(scenarios) for j in range(20)]
    with mp.get_context('spawn').Pool(workers) as pool:
        rows=list(pool.imap_unordered(forecast_case,tasks))
    assert len({r['layout_sha'] for r in rows})==len(rows)
    valid=[r for r in rows if r['eligible']];summary=dict(cases=len(rows),eligible=len(valid),arms={})
    for arm in ['current','history','oracle','actor','best']:
        picked=[r['branches'][10 if arm=='actor' else int(np.argmax([b['return_value'] for b in r['branches']])) if arm=='best' else r['selections'][arm]] for r in valid]
        summary['arms'][arm]=dict(success=sum(b['outcome']=='success' for b in picked),collision=sum(b['outcome']=='collision' for b in picked),
            timeout=sum(b['outcome']=='timeout' for b in picked),mean_return=float(np.mean([b['return_value'] for b in picked])))
    for arm in ['history','oracle']:
        groups=[];excess=0
        for s in scenarios:
            ds=[]
            for r in valid:
                if r['scenario']!=s:continue
                a=r['branches'][r['selections'][arm]];c=r['branches'][r['selections']['current']]
                ds.append(a['return_value']-c['return_value']);excess+=int(a['outcome']=='collision')-int(c['outcome']=='collision')
            if ds:groups.append(np.asarray(ds))
        rng=np.random.default_rng(2407)
        boots=[np.mean([rng.choice(g,len(g),replace=True).mean() for g in groups]) for _ in range(10000)]
        mean=float(np.mean([g.mean() for g in groups]));ci=np.quantile(boots,[.025,.975]).tolist()
        summary[arm+'_vs_current']=dict(mean_return_gain=mean,ci95=ci,excess_collisions=excess,
            passed=bool(mean>=.02 and ci[0]>0 and excess<=0))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--spawn-audit',action='store_true')
    parser.add_argument('--dense-forecast',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.dense_forecast:
        dense_forecast(args.workers)
        return
    if args.spawn_audit:
        destination=OUT/'sequential_spawn'
        destination.mkdir(exist_ok=True)
        protocol=dict(intervention='Only replace list comprehension spawn by sequential append in diagnostic process',
                      production_changed=False,retraining=False,scenarios=SCENARIOS,count=args.count,
                      seeds=[2407,4807,7207],arms=['no_belief','map','full'],
                      profile='heldout_nonstationary',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      note='Same random case seeds; physical layouts necessarily change after rejection sampling fix. Not paired same-layout treatment effect.')
        path=destination/'protocol.json'
        if path.exists():
            assert json.loads(path.read_text())==protocol
        else:
            path.write_text(json.dumps(protocol,indent=2))
        tasks=[(seed,arm,scenario,args.count,index,True) for index,scenario in enumerate(SCENARIOS)
               for seed in [2407,4807,7207] for arm in ['no_belief','map','full']]
        with mp.get_context('spawn').Pool(args.workers) as pool:
            summaries=list(pool.imap_unordered(worker,tasks,chunksize=1))
        (destination/'completed.json').write_text(json.dumps(summaries,indent=2))
        return
    original=json.loads((SOURCE/'protocol.json').read_text())
    assert hashlib.sha256((REPO/'crowd_nav/bayes_continuous/train_smoke.py').read_bytes()).hexdigest()==original['code_sha256']
    for name,digest in original['params_sha256'].items():
        assert hashlib.sha256((PARAMS/name).read_bytes()).hexdigest()==digest
    start=time.monotonic()
    receipt=preflight()
    receipt['elapsed_seconds']=time.monotonic()-start
    (OUT/'preflight.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)
    if not args.run:
        return
    protocol=dict(source='fixed final 20480-step PPO checkpoints, no selection',seeds=[2407,4807,7207],
                arms=['no_belief','map','full'],scenarios=SCENARIOS,cases_per_cell=args.count,
                profile='heldout_nonstationary',policy='deterministic mean',
                training=False,checkpoint_selection=False,primary='equal-weight five high-density held-out scenarios',
                gate='FULL success >= each baseline+3pp, no collision increase, return positive in >=2/3 seeds and paired CI lower>0',
                limitation='three RL seeds share an IL initialization; positive pilot needs new IL seeds and independent confirmation',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())==protocol
    else:
        protocol_path.write_text(json.dumps(protocol,indent=2))
    tasks=[(seed,arm,scenario,args.count,index) for index,scenario in enumerate(SCENARIOS)
                for seed in [2407,4807,7207] for arm in ['no_belief','map','full']]
    with mp.get_context('spawn').Pool(args.workers) as pool:
        summaries=list(pool.imap_unordered(worker,tasks,chunksize=1))
    (OUT/'completed.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
