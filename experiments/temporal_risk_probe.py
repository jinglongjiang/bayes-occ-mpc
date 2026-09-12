"""Frozen temporal-risk decision gate. No training or production changes."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import hashlib
import json
import pickle
import subprocess
import time
import numpy as np
from scipy.stats import ncx2, norm
from numpy.polynomial.legendre import leggauss
from experiments import replay as replay
from nav.belief import BayesianBelief, BeliefConfig
from nav.planner import MPCPlanner

ROOT = replay.ROOT
OUT = ROOT/'results/temporal_risk_probe'
CFG = replay.CFG
F, Q = BayesianBelief(BeliefConfig(dt=CFG.dt, acceleration_std=CFG.acceleration_std))._transition()
G = CFG.acceleration_std*np.array([[CFG.dt**2/2,0],[0,CFG.dt**2/2],[CFG.dt,0],[0,CFG.dt]])
EPS = (.25,.10,.50)
BASE = '12c20dc'


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def register():
    sources = list((ROOT/'nav').glob('*.py')) + [Path(__file__), ROOT/'experiments/replay.py']
    sources += [replay.DATA/f'replay_{n}_{c}.pkl' for n,c in replay.FTASKS]
    sources += [replay.DATA/'source/three_arm.py',replay.DATA/'source/bayesian_rfs.py']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    path = OUT/'protocol.json'
    if path.exists():
        prior = json.loads(path.read_text()); assert prior['hashes']==hashes
        return
    save('protocol.json',dict(baseline=BASE, config=replay.CONFIG, hashes=hashes,
        state_rule='floor(length/3), floor(2*length/3); closest prior nonterminal if needed',
        candidates='four routes x (three uniform first-iteration + three lowest J0 last-iteration); deduplicate, fill same route in stable order; include O replacing last uniform if absent',
        conditional_model='one initial 4D Gaussian draw, sequential independent low-rank acceleration increments; existence applied once per person',
        P0='clean exact-reset model only; unique missed-age in 0..40 matching ALL 16 future covariance matrices; no generic inversion',
        samples_select=4096,samples_eval=65536,chunk=2048,epsilon=list(EPS),
        C_quadrature='standard-normal Cartesian disk quadrature 16x16 then 32x32; 64x64 if pair diff>0.001 or crowd diff>0.002; numerical convergence not formal bound',
        C_tail='truncate each standard normal coordinate at +/-9; skip pairs with min(q_prev,q_next)<1e-12; record no safety certification',
    reference_pool='union of O and I/C/J winners for all eps, plus two J candidates closest to each eps',
        risk_CI='paired delta-method SE from independent per-human Bernoulli covariance, evaluated on fresh samples; risk upper bound aggregates per-person Bonferroni Wilson intervals',
        gate='main eps .25; I-J change >=.05m/s; >=6 states in >=4 episodes; risk gain>=.01 with paired 95% CI, J0<=1.02 I, OR J upper95<=eps, terminal distance gain>=.10m and J0<=I',
        source_IDS='replay lacks persistent IDs; independent-person propagation uses stable local entity row, never matches identities between saved states',
        no_training=True,no_new_navigation=True,branch_validation='only after gate passes; no partial snapshot restoration',
        seed=20260912,threads=1, created=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def recover(obs):
    chain=[np.zeros((4,4))]
    for _ in range(40+CFG.horizon): chain.append(F@chain[-1]@F.T+Q)
    ps=[]; ages=[]; residual=0.
    for i in range(len(obs.entities)):
        errors=[np.max(np.abs(np.array(chain[a+1:a+1+CFG.horizon])[:,:2,:2]-obs.human_position_covariance[i])) for a in range(41)]
        ids=np.flatnonzero(np.array(errors)<1e-11)
        if len(ids)!=1: raise ValueError('DATA_INCOMPLETE: nonunique clean P0 reconstruction')
        a=int(ids[0]);ps.append(chain[a]);ages.append(a);residual=max(residual,errors[a])
        if bool(obs.human_visible[i]): assert a==0
        means=[];x=obs.entities[i,:4].copy()
        for _ in range(CFG.horizon): x=F@x;means.append(x[:2].copy())
        np.testing.assert_allclose(means,obs.human_segment_end[i],rtol=0,atol=1e-10)
    return np.asarray(ps).reshape(-1,4,4),ages,float(residual)


def pool_from_batches(batches, output, obs, seed):
    rng=np.random.default_rng(seed); first,last=batches[0],batches[-1]
    assert len(first['counts'])==4, 'unexpected route allocation'
    planner=MPCPlanner(CFG,envelope=None)
    last_cost=planner._cost(last['controls'],obs,last['positions'],last['human_clearance'],last['occupancy'],None)
    chosen=[];labels=[];seen=set()
    def add(u,label):
        key=u.tobytes()
        if key in seen:return False
        seen.add(key);chosen.append(u.copy());labels.append(label);return True
    for route in range(4):
        lo,hi=first['bounds'][route:route+2]
        before=len(chosen)
        for i in rng.choice(np.arange(lo,hi),3,replace=False): add(first['controls'][i],dict(route=route,stage='first',index=int(i)))
        order=np.argsort(last_cost[lo:hi],kind='stable')+lo
        for i in order:
            if len(chosen)>=before+6:break
            add(last['controls'][i],dict(route=route,stage='last',index=int(i)))
        for i in range(lo,hi):
            if len(chosen)>=before+6:break
            add(first['controls'][i],dict(route=route,stage='fill',index=i))
    if output.tobytes() not in seen:
        at=next(i for i in range(len(labels)-1,-1,-1) if labels[i]['stage']=='first')
        chosen[at]=output.copy();labels[at]=dict(stage='original_output',route=labels[at]['route'])
    u=np.asarray(chosen);assert len(u)<=24
    oi=next(i for i,x in enumerate(u) if np.array_equal(x,output))
    return u,labels,oi


def geometry(obs,u):
    p=MPCPlanner(CFG,envelope=None)
    pos=obs.robot_xy+np.cumsum(u*CFG.dt,axis=1)
    active,_,dist=p._active_until_goal(pos,obs)
    hc=p._human_clearance(u,obs) if obs.entities.size else None
    occ=obs.occupancy_probability.sample(pos) if obs.occupancy_probability is not None else None
    j0=p._cost(u,obs,pos,hc,occ,None)
    gf,gs=p._combined_clearance(u,obs,pos,hc,occ,None)
    q=replay.reference_q(obs,pos) if obs.entities.size else np.zeros((len(u),0,CFG.horizon))
    r=obs.human_existence if obs.entities.size else np.array([])
    first_risk=1-np.prod(1-q[:,:,0]*r[None],axis=1)
    velocity=np.linalg.norm(u,axis=2).max(axis=1)<=CFG.v_max+1e-10
    acceleration=np.linalg.norm(np.diff(np.concatenate((np.broadcast_to(obs.robot_velocity,(len(u),1,2)),u),axis=1),axis=1),axis=2).max(axis=1)<=CFG.a_max*CFG.dt+1e-10
    safe_first=first_risk<=p._belief_risk_limits()[0]+1e-12
    relaxed=velocity&acceleration&safe_first&(gs>=0)
    full=relaxed&(gf>=0)
    endpoint=dist[np.arange(len(u)),active.sum(axis=1)-1]
    return dict(pos=pos,active=active,j0=j0,q=q,r=r,G=full,relaxed=relaxed,
                geo_full=gf,geo_first=gs,first_risk=first_risk,terminal=endpoint,
                blocked=dict(kinematics=int((~(velocity&acceleration)).sum()),first_probability=int((~safe_first).sum()),first_geometry=int((gs<0).sum()),full_geometry=int((gf<0).sum())))


def aggregate(fi,r): return 1-np.prod(1-fi*np.asarray(r)[None],axis=1)


def sample(obs,p0,pos,active,S,seed,noise_root=None):
    t=time.perf_counter();rng=np.random.default_rng(seed)
    noise_root=G if noise_root is None else noise_root
    c,h=pos.shape[:2];m=len(obs.entities)
    hits=np.zeros((c,m,S),bool);segments=np.zeros((c,m));twice=np.zeros((c,m))
    marginal=np.zeros((c,m,h));adjacent=np.zeros((c,m,h-1))
    for i in range(m):
        eig,vec=np.linalg.eigh(p0[i]);assert eig.min()>=-1e-12
        root=vec*np.sqrt(np.maximum(eig,0))[None]
        radius=obs.robot_radius+obs.entities[i,4]+CFG.human_margin
        for start in range(0,S,2048):
            size=min(2048,S-start)
            x=obs.entities[i,:4]+rng.standard_normal((size,4))@root.T
            path=np.empty((size,h,2))
            for k in range(h):
                x=x@F.T+rng.standard_normal((size,2))@noise_root.T;path[:,k]=x[:,:2]
            z=((pos[:,None]-path[None])**2).sum(axis=3)<=radius**2
            z &= active[:,None]
            runs=z[:,:,0].astype(np.int16)+((~z[:,:,:-1])&z[:,:,1:]).sum(axis=2)
            hits[:,i,start:start+size]=runs>0
            segments[:,i]+=runs.sum(axis=1);twice[:,i]+=(runs>=2).sum(axis=1)
            marginal[:,i]+=z.sum(axis=1);adjacent[:,i]+=(z[:,:,:-1]&z[:,:,1:]).sum(axis=1)
    fi=hits.mean(axis=2) if m else np.zeros((c,0))
    B=segments/S;qh=marginal/S;sh=adjacent/S
    np.testing.assert_allclose(qh.sum(axis=2)-sh.sum(axis=2),B,rtol=0,atol=1e-14)
    return dict(hits=hits,fi=fi,rho=aggregate(fi,obs.human_existence if m else []),
                B=B,extra=B-fi,multiple=twice/S,qhat=qh,shat=sh,ms=(time.perf_counter()-t)*1000)


def pair_integrals(obs,p0,pos,q,nodes):
    """Conditional Gaussian integral over the previous disk in standardized coordinates."""
    t=time.perf_counter();z,w=leggauss(nodes);out=np.zeros(q.shape[:2]+(CFG.horizon-1,))
    pseq=[]
    for initial in p0:
        p=initial.copy();seq=[]
        for _ in range(CFG.horizon):p=F@p@F.T+Q;seq.append(p.copy())
        pseq.append(seq)
    evaluations=0
    for i in range(len(obs.entities)):
        R=obs.robot_radius+obs.entities[i,4]+CFG.human_margin
        for k in range(1,CFG.horizon):
            prev,nextp=pseq[i][k-1],pseq[i][k]
            var=prev[0,0];cross=(prev@F.T)[0,0];gain=cross/var
            cv=nextp[0,0]-cross**2/var
            assert var>0 and cv>0
            sigma=np.sqrt(var);mu=obs.human_segment_end[i,k-1];nmu=obs.human_segment_end[i,k]
            for j in np.flatnonzero(np.minimum(q[:,i,k-1],q[:,i,k])>=1e-12):
                center=pos[j,k-1]
                a=max(-9.,(center[0]-R-mu[0])/sigma);b=min(9.,(center[0]+R-mu[0])/sigma)
                if a>=b:continue
                tx=(a+b)/2+(b-a)/2*z;xs=mu[0]+sigma*tx
                dy=np.sqrt(np.maximum(0,R*R-(xs-center[0])**2))
                low=np.maximum(-9.,(center[1]-dy-mu[1])/sigma);high=np.minimum(9.,(center[1]+dy-mu[1])/sigma)
                length=np.maximum(0.,high-low)
                ty=(low+high)[:,None]/2+length[:,None]/2*z[None]
                px=np.broadcast_to(xs[:,None],ty.shape);py=mu[1]+sigma*ty
                conditional=nmu+gain*(np.stack((px,py),axis=2)-mu)
                d2=((conditional-pos[j,k])**2).sum(axis=2)
                inner=ncx2.cdf(R*R/cv,2,d2/cv)
                weights=((b-a)/2*w*norm.pdf(tx))[:,None]*(length[:,None]/2*w[None]*norm.pdf(ty))
                out[j,i,k-1]=(weights*inner).sum();evaluations+=nodes*nodes
    return out,dict(nodes=nodes,ms=(time.perf_counter()-t)*1000,cdf_values=evaluations)


def classical(obs,p0,g):
    q=g['q'];active=g['active'];a16,t16=pair_integrals(obs,p0,g['pos'],q,16)
    a32,t32=pair_integrals(obs,p0,g['pos'],q,32)
    def value(a):
        B=(q*active[:,None]).sum(axis=2)-(a*active[:,None,1:]).sum(axis=2)
        return B,aggregate(np.clip(B,0,1),g['r'])
    _,r16=value(a16);B,rho=value(a32)
    gap=float(np.max(np.abs(a16-a32),initial=0));delta=float(np.max(np.abs(r16-rho),initial=0))
    times=[t16,t32];prev=a16;best=a32
    if gap>.001 or delta>.002:
        best,t64=pair_integrals(obs,p0,g['pos'],q,64);times.append(t64)
        B,new=value(best);gap=float(np.max(np.abs(best-a32),initial=0));delta=float(np.max(np.abs(new-rho),initial=0));rho=new;prev=a32
    return rho,dict(B=B.tolist(),pair_max_resolution_difference=gap,crowd_max_resolution_difference=delta,
        converged=gap<=.001 and delta<=.002,timings=times,
        pair_frechet_violation=float(max(0,np.max(best-np.minimum(q[:,:,:-1],q[:,:,1:]),initial=0),np.max(np.maximum(0,q[:,:,:-1]+q[:,:,1:]-1)-best,initial=0))))


def choose(risk,g,eps,relaxed=False):
    ids=np.flatnonzero((g['relaxed'] if relaxed else g['G'])&(risk<=eps))
    return int(ids[np.argmin(g['j0'][ids])]) if len(ids) else None


def compare(evaldata,indices,a,b,g,u,eps):
    if a is None or b is None:return dict(status='NO_FEASIBLE',qualifies=False)
    ia,ib=indices.index(a),indices.index(b);fi=evaldata['fi'];r=g['r'];S=evaldata['hits'].shape[2]
    ga=np.array([r[i]*np.prod(np.delete(1-r*fi[ia],i)) for i in range(len(r))])
    gb=np.array([r[i]*np.prod(np.delete(1-r*fi[ib],i)) for i in range(len(r))])
    var=0.
    for i in range(len(r)):
        ha=evaldata['hits'][ia,i];hb=evaldata['hits'][ib,i]
        cov=np.mean(ha&hb)-fi[ia,i]*fi[ib,i]
        var+=ga[i]**2*fi[ia,i]*(1-fi[ia,i])+gb[i]**2*fi[ib,i]*(1-fi[ib,i])-2*ga[i]*gb[i]*cov
    gain=float(evaldata['rho'][ia]-evaldata['rho'][ib]);se=np.sqrt(max(0.,var)/S)
    z=norm.ppf(1-.05/(2*max(len(r),1)))
    f=fi[ib];upper=(f+z*z/(2*S)+z*np.sqrt(f*(1-f)/S+z*z/(4*S*S)))/(1+z*z/S)
    risk_upper=float(aggregate(upper[None],r)[0])
    du=float(np.linalg.norm(u[a,0]-u[b,0]));progress=float(g['terminal'][a]-g['terminal'][b])
    goodA=gain>=.01 and gain-1.96*se>0 and g['j0'][b]<=1.02*g['j0'][a]
    goodB=risk_upper<=eps and progress>=.10 and g['j0'][b]<=g['j0'][a]
    valid=bool(g['G'][a] and g['G'][b] and du>=.05)
    status='RESOLVED' if goodA or goodB else ('UNRESOLVED' if abs(gain)<=1.96*se else 'NO_SUPPORTED_GAIN')
    return dict(status=status,qualifies=valid and bool(goodA or goodB),
        action_delta=du,first_displacement_delta=du*CFG.dt,
        one_second_position_delta=float(np.linalg.norm(g['pos'][a,3]-g['pos'][b,3])),
        risk_gain=gain,risk_gain_ci=[float(gain-1.96*se),float(gain+1.96*se)],
        J_risk_upper95=risk_upper,terminal_progress=progress,J0_difference=float(g['j0'][b]-g['j0'][a]),
        both_in_G=bool(g['G'][a] and g['G'][b]))


def evaluate(n,case,step,frame,u,labels,oi):
    obs=frame['obs'];p0,ages,residual=recover(obs);g=geometry(obs,u)
    seed=[20260912,n,case,step]
    selection=sample(obs,p0,g['pos'],g['active'],4096,seed+[0])
    independent=aggregate(1-np.prod(1-g['q']*g['active'][:,None],axis=2),g['r'])
    crisk,cdiag=classical(obs,p0,g)
    risks={'I':independent,'C':crisk,'J':selection['rho']}
    choices={str(e):{a:choose(v,g,e) for a,v in risks.items()} for e in EPS}
    relaxed={str(e):{a:choose(v,g,e,True) for a,v in risks.items()} for e in EPS}
    ids={oi}
    for e in EPS:
        ids.update(i for i in choices[str(e)].values() if i is not None)
        ids.update(int(i) for i in np.argsort(abs(selection['rho']-e),kind='stable')[:2])
    indices=sorted(ids)
    verified=sample(obs,p0,g['pos'][indices],g['active'][indices],65536,seed+[1])
    comparisons={}
    for e in EPS:
        c=choices[str(e)];comparisons[str(e)]={a+'-J':compare(verified,indices,c[a] if a!='O' else oi,c['J'],g,u,e) for a in ('I','C','O')}
    qtarget=g['q']*g['active'][:,None]
    tolerance=6*np.sqrt(qtarget*(1-qtarget)/4096)+6/4096
    assert np.all(np.abs(selection['qhat']-qtarget)<=tolerance), 'MC marginal contract failed'
    # Re-evaluation is restricted to selected/critical competitors, not a new full argmin claim.
    stable={}
    for e in EPS:
        feasible=[i for j,i in enumerate(indices) if g['G'][i] and verified['rho'][j]<=e]
        w=min(feasible,key=lambda i:g['j0'][i]) if feasible else None
        stable[str(e)]=dict(selection=choices[str(e)]['J'],verification_pool_winner=w,changed=w!=choices[str(e)]['J'])
    dI=independent[:,None]-independent[None];dJ=selection['rho'][:,None]-selection['rho'][None]
    flips=np.triu((dI*dJ<0)&(np.abs(dJ)>=.01),1)
    old_h=-np.log1p(-np.clip(g['q']*g['r'][None,:,None],0,1-1e-12)).sum(axis=1)
    planner=MPCPlanner(CFG,envelope=None)
    old_cost,old_full,old_first=planner._score_from_hazard(g['j0'],g['geo_full'],g['geo_first'],g['active'],planner._belief_hazard_limits()[None],old_h)
    row=dict(n=n,case=case,step=step,age=ages,P0=p0.tolist(),P0_residual=residual,
        labels=labels,controls=u.tolist(),O=oi,J0=g['j0'].tolist(),terminal=g['terminal'].tolist(),
        old_cost=old_cost.tolist(),old_categories=np.where(old_full>=0,0,np.where(old_first>=0,1,2)).tolist(),
        selection_stream_ranking_flips_ge_01=int(flips.sum()),
        active_steps=g['active'].sum(axis=1).tolist(),G=g['G'].tolist(),relaxed_G=g['relaxed'].tolist(),blocked=g['blocked'],
        risks={a:v.tolist() for a,v in risks.items()},choices=choices,without_full_geometry=relaxed,
        C=cdiag,select_ms=selection['ms'],eval_ms=verified['ms'],comparisons=comparisons,reference_stability=stable,
        MC_marginal_max_error=float(np.max(abs(selection['qhat']-qtarget),initial=0)),
        ideal_adjacent_rho=aggregate(np.minimum(1,selection['B']),g['r']).tolist(),
        extra_segments=selection['extra'].tolist(),multiple_segment_frequency=selection['multiple'].tolist(),
        reference_candidates=indices,reference_rho=verified['rho'].tolist(),reference_F=verified['fi'].tolist())
    return row


def gates():
    np.testing.assert_allclose(G@G.T,Q,atol=1e-15)
    for bits in ([0,0,0],[0,1,1],[1,0,1],[1,1,1]):
        z=np.array(bits,bool);segments=int(z[0])+int(((~z[:-1])&z[1:]).sum())
        assert z.sum()-(z[:-1]&z[1:]).sum()==segments
        assert segments-int(z.any())==max(segments-1,0)
    o=pickle.load(open(replay.DATA/'replay_5_23000.pkl','rb'))[10]['obs']
    p0,_,_=recover(o);u=np.zeros((1,CFG.horizon,2));g=geometry(o,u)
    a=np.zeros_like(g['active']);a[:,0]=True
    sampled=sample(o,p0,g['pos'],a,4096,[20260912,999])
    np.testing.assert_array_equal(sampled['fi'],sampled['qhat'][:,:,0])
    np.testing.assert_array_equal(aggregate(sampled['fi'],np.zeros(len(p0))),[0])
    np.testing.assert_allclose(aggregate(sampled['fi'],np.ones(len(p0))),1-np.prod(1-sampled['fi'],axis=1))
    empty=copy.deepcopy(o);empty.entities=np.empty((0,5));empty.human_existence=np.array([])
    assert sample(empty,np.empty((0,4,4)),g['pos'],a,32,[7])['rho'][0]==0
    positions=np.repeat(o.goal_xy[None,None],CFG.horizon,axis=1)
    active,_,_=MPCPlanner._active_until_goal(positions,o)
    assert active.sum()==1 and active[0,0]
    deterministic=sample(o,np.zeros_like(p0),g['pos'],g['active'],32,[8],noise_root=np.zeros_like(G))
    radius=o.robot_radius+o.entities[:,4]+CFG.human_margin
    expected=((((g['pos'][:,None]-o.human_segment_end[None])**2).sum(axis=3)<=radius[None,:,None]**2)&g['active'][:,None]).any(axis=2)
    np.testing.assert_array_equal(deterministic['fi'],expected)
    save('checks.json',dict(status='PASS',checks=['low-rank Q','run-count identity','unique clean covariance reconstruction','H=1','r=0/1','no entities','deterministic trajectory','goal entry retained']))


def summarize(rows):
    result={}
    for eps in EPS:
        groups={}
        for arm in ('I','C','O'):
            cs=[r['comparisons'][str(eps)][arm+'-J'] for r in rows]
            good=[r for r,c in zip(rows,cs) if c['qualifies']]
            groups[arm+'-J']=dict(qualifying_states=len(good),episodes=len({(r['n'],r['case']) for r in good}),
                actionable_changes=sum(c.get('both_in_G',False) and c.get('action_delta',0)>=.05 for c in cs),
                no_feasible=sum(c['status']=='NO_FEASIBLE' for c in cs))
        result[str(eps)]=groups
    main=result['0.25']['I-J'];passed=main['qualifying_states']>=6 and main['episodes']>=4
    verdict='GATE_PASSED_EXECUTION_PENDING' if passed else 'STOP_TEMPORAL'
    save('summary.json',dict(verdict=verdict,states=len(rows),comparisons=result,
        C_converged_states=sum(r['C']['converged'] for r in rows),
        C_24_candidates_ms=[sum(t['ms'] for t in r['C']['timings']) for r in rows],
        G_candidates=sum(sum(r['G']) for r in rows),total_candidates=sum(len(r['G']) for r in rows),
        J_available_before_geometry=sum(r['without_full_geometry']['0.25']['J'] is not None for r in rows),
        J_available_after_geometry=sum(r['choices']['0.25']['J'] is not None for r in rows),
        reference_selection_changed=sum(r['reference_stability']['0.25']['changed'] for r in rows),
        no_training=True,no_new_navigation=True))
    lines=['# Temporal First-Hit Gate', '', 'Verdict: '+verdict, '',
           '| epsilon | contrast | executable changes >=0.05 m/s | independently supported states | episodes | no feasible |',
           '|---|---|---:|---:|---:|---:|']
    for e,g in result.items():
        for a,c in g.items():lines.append(f"| {e} | {a} | {c['actionable_changes']} | {c['qualifying_states']} | {c['episodes']} | {c['no_feasible']} |")
    lines+=['','No neural training. No new navigation episodes. C timings are real quadrature costs on the small pool, not a full-workload real-time claim.',
            'I-J isolates temporal dependence; O-J also changes risk objective and interface. J is a finite-sample model reference, not environment truth.',
            'Protocol and all candidate risks, controls, constraints, convergence and paired MC intervals are in this directory.']
    (OUT/'verdict.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verdict=verdict,comparisons=result)),flush=True)


def run():
    register();gates()
    path=OUT/'states.jsonl'
    if path.exists():raise RuntimeError('refusing to overwrite existing state results')
    rows=[];pending=[]
    for n,case in replay.FTASKS:
        frames=pickle.load(open(replay.DATA/f'replay_{n}_{case}.pkl','rb'))
        target={len(frames)//3,2*len(frames)//3}
        model=MPCPlanner(CFG,envelope=None)
        for step,frame in enumerate(frames):
            batches=[]
            if step in target:
                def capture(s):
                    batches.append({k:copy.deepcopy(s[k]) for k in ('controls','positions','human_clearance','occupancy','counts','bounds')})
                model.trace_hook=capture
            else:model.trace_hook=None
            model.plan(frame['obs'],frame['seed'])
            if step not in target:continue
            assert np.linalg.norm(frame['obs'].robot_xy-frame['obs'].goal_xy)>=frame['obs'].robot_radius
            u,labels,oi=pool_from_batches(batches,model.last_controls,frame['obs'],20260912+n+case+step)
            pending.append((n,case,step,frame,u,labels,oi))
            if step==max(target):break
        print('CANDIDATES',n,case,flush=True)
    probes=[next(x for x in pending if x[0]==n) for n in (5,10,20)]
    probe_keys={x[:3] for x in probes}
    pending=probes+[x for x in pending if x[:3] not in probe_keys]
    with path.open('w') as out:
        for n,case,step,frame,u,labels,oi in pending:
            row=evaluate(n,case,step,frame,u,labels,oi);rows.append(row)
            out.write(json.dumps(row,allow_nan=False)+'\n');out.flush()
            print('STATE',len(rows),n,case,step,'G',sum(row['G']),'C_ms',round(sum(t['ms'] for t in row['C']['timings'])),
                  'I-J',row['comparisons']['0.25']['I-J'],flush=True)
    assert len(rows)==60
    register();summarize(rows)


if __name__=='__main__':
    if '--summary' in sys.argv:summarize([json.loads(x) for x in (OUT/'states.jsonl').read_text().splitlines()])
    else:run()
