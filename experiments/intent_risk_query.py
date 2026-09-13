"""Frozen-input direct-table and adaptive mixture-tree experiment, not production."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import ctypes as ct
import json
import time
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments import intent_backend as fast
from experiments import intent_backend_audit as audit
from experiments import intent_repair_run as ref

OUT=audit.ROOT/'results/intent_risk_query'
EPSILON=1e-4  # Absolute width budget on the per-step any-person collision probability.


def save(name,value):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(value,default=audit.old.encode,allow_nan=False,indent=2)+'\n')


def library():
    lib=fast.library();p=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    ints=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.risk_tree_build.argtypes=[p,p,ints,ct.c_int,ct.c_int];lib.risk_tree_build.restype=ct.c_void_p
    lib.risk_tree_free.argtypes=[ct.c_void_p];lib.risk_tree_free.restype=None
    lib.approximate_risk.argtypes=[ct.c_int,ct.c_void_p,p,ct.c_int,ct.c_int,p,p,p,p,ints,p,p,ct.c_int,
        ct.c_double,ct.c_double,p,ints,p,ct.c_int,ct.c_double,p,p,ints]
    return lib


class Direct(fast.FusedEnvelope):
    mode=0
    def __init__(self,*args,**kwargs):
        started=time.perf_counter();super().__init__(*args,**kwargs)
        self.build_ms=1000*(time.perf_counter()-started);self.tree_ms=0.;self.query_ms=0.
        self.cdf_ms=0.;self.tree=None;self.lib=library();self.counts=np.zeros(3,dtype=np.int64)
    def interval(self,positions):
        p=fast.doubles(positions);lo=np.empty(p.shape[:2]);hi=np.empty_like(lo);t=self.table
        if p.ndim!=3 or p.shape[1:]!=(self.cfg.horizon,2) or not np.isfinite(p).all():
            raise ValueError('invalid candidate positions')
        counts=np.zeros(3,dtype=np.int64);started=time.perf_counter()
        self.lib.approximate_risk(self.mode,self.tree,p,len(p),p.shape[1],self.centers,self.variance,self.sigma,
            self.radius,self.component,t.table,t.derivative,len(t.grid),t.step,t.interpolation_error,
            self.weights,self.offsets,self.r,len(self.r),EPSILON/max(1,len(self.r)),lo,hi,counts)
        self.query_ms+=1000*(time.perf_counter()-started);self.counts+=counts
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo>hi).any():
            raise FloatingPointError('invalid approximate risk')
        return lo,hi
    def exact(self,positions):
        # This is the common evaluator API name, not a claim of CDF precision.
        return self.interval(positions)[1]


class Hierarchy(Direct):
    mode=1
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if EPSILON/max(1,len(self.r))<2*(self.table.interpolation_error+1e-12):
            raise ValueError('requested group tolerance is below the leaf interpolation floor')
        for a,b in zip(self.offsets[:-1],self.offsets[1:]):
            if not np.all(self.variance[a:b]==self.variance[a]) or not np.all(self.radius[a:b]==self.radius[a]):
                raise ValueError('a tree requires common per-person conditional variance and radius')
        started=time.perf_counter()
        self.tree=self.lib.risk_tree_build(self.centers,self.weights,self.offsets,len(self.r),self.cfg.horizon)
        self.tree_ms=1000*(time.perf_counter()-started)
    def __del__(self):
        if getattr(self,'tree',None):self.lib.risk_tree_free(self.tree);self.tree=None


class ProfileReference(fast.FusedEnvelope):
    def __init__(self,*args,**kwargs):
        started=time.perf_counter();super().__init__(*args,**kwargs)
        self.build_ms=1000*(time.perf_counter()-started);self.tree_ms=0.;self.query_ms=0.;self.cdf_ms=0.
        self.audit_positions=None
    def bounds(self,positions):
        if self.audit_positions is not None:self.audit_positions.append(positions.copy())
        started=time.perf_counter();out=super().bounds(positions)
        self.query_ms+=1000*(time.perf_counter()-started);return out
    def exact(self,positions):
        started=time.perf_counter();out=super().exact(positions)
        self.cdf_ms+=1000*(time.perf_counter()-started);return out


class ApproxPlanner(ref.MixturePlanner):
    def _build_envelope(self,obs):
        super()._build_envelope(obs)
        # Use the existing full-evaluation path and unchanged scoring/search loop.
        return None


def run_pipeline(env,adapter,engine,warm,record,step,variant,trace=False):
    previous_e,previous_p=ref.MixtureEnvelope,ref.MixturePlanner
    created=[]
    def make(*args):
        value={'reference':ProfileReference,'direct':Direct,'hierarchy':Hierarchy}[variant](*args)
        if variant=='reference' and trace:value.audit_positions=[]
        created.append(value);return value
    ref.MixtureEnvelope=make
    if variant!='reference':ref.MixturePlanner=ApproxPlanner
    try:
        result=audit.pipeline(env,adapter,engine,warm,record,step,True,False,trace)
        evaluator=created[-1]
        result['timing'].update(table_build=evaluator.build_ms,tree_build=evaluator.tree_ms,
            bound_or_point_query=evaluator.query_ms,remaining_exact=evaluator.cdf_ms)
        if hasattr(evaluator,'counts'):result['timing']['query_counts']=evaluator.counts.copy()
        if variant=='reference' and trace:result['candidate_positions']=np.concatenate(evaluator.audit_positions)
        return result
    finally:ref.MixtureEnvelope,ref.MixturePlanner=previous_e,previous_p


def self_test():
    from types import SimpleNamespace
    rng=np.random.default_rng(8371);cfg=audit.MPCConfig();h=cfg.horizon;results=[]
    for variance in (0.,.09,2.):
        centers=rng.normal(0,1,(32,h,2));w=rng.random(32);w/=w.sum();v=np.full(h,variance)
        obs=SimpleNamespace(entities=np.array([[0.,0.,0.,0.,.3]]),robot_radius=.3,
            human_segment_end=centers[:1],human_position_covariance=v[None,:,None,None]*np.eye(2),
            human_existence=np.array([.7]))
        mix={0:(centers,w,v)};positions=rng.normal(0,2,(64,h,2))
        reference=fast.FusedEnvelope(cfg,obs,mix).exact(positions)
        tree=Hierarchy(cfg,obs,mix);lo,hi=tree.interval(positions)
        assert np.all(reference>=lo-1e-10) and np.all(reference<=hi+1e-10)
        width=float(np.max(np.expm1(-lo)-np.expm1(-hi)));assert width<=EPSILON+1e-9
        direct=Direct(cfg,obs,mix)
        split={0:(np.repeat(centers,2,axis=0),np.repeat(w*.5,2),v)}
        np.testing.assert_allclose(direct.exact(positions),Direct(cfg,obs,split).exact(positions),atol=1e-10,rtol=0)
        order=rng.permutation(32);permuted=Hierarchy(cfg,obs,{0:(centers[order],w[order],v)})
        l,u=permuted.interval(positions)
        assert np.all(reference>=l-1e-10) and np.all(reference<=u+1e-10)
        if variance>0:
            for evaluator in (tree,direct):
                evaluator.table.table[:]=np.nan
                try:evaluator.exact(positions)
                except FloatingPointError:pass
                else:raise AssertionError('nonfinite interpolation must not become low risk')
        results.append(dict(variance=variance,width=width,containment=True,split=True,permutation=True,
                            nan_rejected=variance>0))
    save('self_test.json',results);print(results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=12)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))});library()
    if args.self_test:self_test();return
    protocol=json.loads((audit.OUT/'protocol.json').read_text())
    save('protocol.json',dict(base='c711be6c1f4fd60c0f83a8a303bcd9b2d158551e',epsilon=EPSILON,
        states=protocol['states'],repetitions=3,config=protocol['config'],settings=protocol['settings'],
        scope='fixed development inputs, no navigation; common canonical warm start at each state',
        hierarchy_score='upper bound, no elite certification; conditional on valid floating probability envelopes',
        sources={str(p.relative_to(audit.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (Path(__file__).resolve(),audit.ROOT/'experiments/intent_kernels.cpp')}))
    records={r['case']:r for r in audit.old.data()}
    calibration=json.loads((audit.old.OUT/'calibration.json').read_text())
    path=OUT/'states.json';rows=json.loads(path.read_text()) if path.exists() else []
    for index,(case,step,tid) in enumerate(protocol['states'][:args.limit]):
        if any(r['state']==[case,step,tid] for r in rows):continue
        record=records[case];print('PREPARE',case,flush=True)
        with fast.enabled(behavior=True,risk=False):env,adapter,engine,warm=audit.prepare(record,step,protocol,calibration)
        results={v:run_pipeline(env,adapter,engine,warm,record,step,v,True) for v in ('reference','direct','hierarchy')}
        a=results['reference'];cfg=audit.MPCConfig(**protocol['config'])
        positions=a['candidate_positions']
        evaluators={v:c(cfg,a['obs'],a['mixtures']) for v,c in
                    [('reference',ProfileReference),('direct',Direct),('hierarchy',Hierarchy)]}
        exact=evaluators['reference'].exact(positions);limits=audit.MPCPlanner(cfg)._belief_hazard_limits()[None]
        errors={}
        for v in ('direct','hierarchy'):
            lo,hi=evaluators[v].interval(positions);q=-np.expm1(-hi);truth=-np.expm1(-exact)
            errors[v]=dict(max_probability_error=float(np.max(abs(q-truth))),
                threshold_false_safe=int(np.sum((hi<=limits)&(exact>limits))),
                threshold_false_unsafe=int(np.sum((hi>limits)&(exact<=limits))),
                max_interval_width=float(np.max(np.expm1(-lo)-np.expm1(-hi))),
                interval_violations=int(np.sum((exact<lo-1e-10)|(exact>hi+1e-10))) if v=='hierarchy' else None,
                action_difference=float(np.linalg.norm(results[v]['action']-a['action'])),
                controls_difference=float(np.max(abs(results[v]['controls']-a['controls']))),
                ordered_elite_changes=sum(x['elites']!=y['elites'] for x,y in zip(a['traces'],results[v]['traces'])))
            if v=='hierarchy':
                assert errors[v]['interval_violations']==0,errors[v]
                assert errors[v]['max_interval_width']<=EPSILON+1e-9,errors[v]
        measures=[];variants=['reference','direct','hierarchy']
        for repeat in range(3):
            order=(index+repeat)%3
            for v in variants[order:]+variants[:order]:
                r=run_pipeline(env,adapter,engine,warm,record,step,v)
                measures.append(dict(variant=v,repetition=repeat,**r['timing']))
        rows.append(dict(state=[case,step,tid],people=record['people'],common_candidates=len(positions),errors=errors,measures=measures))
        save('states.json',rows)
        print('DONE',case,{v:round(np.median([m['total'] for m in measures if m['variant']==v]),2) for v in variants},errors,flush=True)
    table=[]
    for people in (5,10,20):
        for v in ('reference','direct','hierarchy'):
            values=[m['total'] for r in rows if r['people']==people for m in r['measures'] if m['variant']==v]
            if values:table.append(dict(people=people,variant=v,p50=float(np.median(values)),p95=float(np.percentile(values,95)),late=float(np.mean(np.array(values)>250))))
    save('summary.json',table);print(table,flush=True)


if __name__=='__main__':main()
