"""Packed T deployment-resource audit; no model or navigation changes."""
import os
from pathlib import Path
ALLOWED=sorted(os.sched_getaffinity(0));PHYSICAL=[];seen=set()
for cpu in reversed(ALLOWED):
    topology=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
    key=tuple((topology/f).read_text().strip() for f in ('physical_package_id','core_id'))
    if key not in seen:PHYSICAL.append(cpu);seen.add(key)
PHYSICAL=PHYSICAL[:8]
os.environ.update(OMP_PLACES=','.join('{%d}'%c for c in PHYSICAL),OMP_PROC_BIND='close',
    OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import intent_risk_query as q
from experiments import intent_backend_audit as audit
from experiments import intent_backend as fast
from experiments import intent_repair_run as ref
from experiments.packed_direct_execution import packed_direct as packed
OUT=ROOT/'results/intent_direct_parallel'
THREADS=[t for t in (1,2,4,8) if t<=len(PHYSICAL)]


def save(name,value):
    OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(value,default=audit.old.encode,allow_nan=False,indent=2)+'\n')


def placement(evaluator,threads):
    actual=evaluator.query_context.last_cpus()
    assert len(actual)==threads and len(set(actual))==threads,(actual,threads)
    assert set(actual)==set(PHYSICAL[:threads]),(actual,PHYSICAL[:threads])
    return actual


def pipeline(env,adapter,engine,warm,record,step,threads,trace=False):
    os.sched_setaffinity(0,{PHYSICAL[0]})
    old_e,old_p=ref.MixtureEnvelope,ref.MixturePlanner;created=[];batches=[]
    def factory(*args):
        value=q.Direct(*args) if threads==0 else CLASSES[threads](*args)
        if trace and threads==0:
            original=value.exact
            def capture(positions):
                batches.append(positions.copy());return original(positions)
            value.exact=capture
        created.append(value);return value
    ref.MixtureEnvelope=factory;ref.MixturePlanner=q.ApproxPlanner
    try:result=audit.pipeline(env,adapter,engine,warm,record,step,True,False,trace)
    finally:ref.MixtureEnvelope,ref.MixturePlanner=old_e,old_p
    e=created[-1]
    result['timing'].update(build_pack=e.build_ms,query=e.query_ms,
        actual_cpus=placement(e,threads) if threads else [os.sched_getcpu()] if hasattr(os,'sched_getcpu') else [PHYSICAL[0]])
    if batches:result['candidate_positions']=np.concatenate(batches)
    return result


def self_test():
    from types import SimpleNamespace
    rng=np.random.default_rng(7714);cfg=audit.MPCConfig();h=cfg.horizon;rows=[]
    for variance in (0.,.09,2.):
        centers=rng.normal(0,1,(16,h,2));v=np.full(h,variance);w=rng.random(16);w/=w.sum()
        obs=SimpleNamespace(entities=np.array([[0.,0.,0.,0.,.3]]),robot_radius=.3,
            human_segment_end=centers[:1],human_position_covariance=v[None,:,None,None]*np.eye(2),human_existence=np.array([.7]))
        modes={0:(centers,w,v)};positions=rng.normal(0,2,(32,h,2));reference=q.Direct(cfg,obs,modes).exact(positions)
        for t in THREADS:
            e=CLASSES[t](cfg,obs,modes);started=time.perf_counter();value=e.exact(positions)
            first_query_ms=1000*(time.perf_counter()-started)
            np.testing.assert_array_equal(value,reference)
            cpus=placement(e,t)
            split={0:(np.repeat(centers,2,axis=0),np.repeat(w*.5,2),v)}
            np.testing.assert_allclose(CLASSES[t](cfg,obs,split).exact(positions),reference,atol=1e-10,rtol=0)
            order=rng.permutation(16)
            np.testing.assert_allclose(CLASSES[t](cfg,obs,{0:(centers[order],w[order],v)}).exact(positions),reference,atol=1e-10,rtol=0)
            try:e.exact(np.full_like(positions,np.nan))
            except ValueError:pass
            else:raise AssertionError('invalid positions accepted')
            obs_zero=copy.copy(obs);obs_zero.human_existence=np.zeros(1)
            np.testing.assert_array_equal(CLASSES[t](cfg,obs_zero,modes).exact(positions),np.zeros((32,h)))
            e.query_context.close()
            try:e.exact(positions)
            except RuntimeError:pass
            else:raise AssertionError('closed context accepted')
            rows.append(dict(variance=variance,threads=t,cpus=cpus,bitwise_equal=True,first_query_ms=first_query_ms))
    # Exercise the packed kernel's non-common-parameter path within one person.
    cfg=audit.MPCConfig();h=cfg.horizon
    obs=SimpleNamespace(entities=np.array([[0.,0.,0.,0.,.3],[0.,0.,0.,0.,.4]]),robot_radius=.3,
        human_segment_end=np.zeros((2,h,2)),human_position_covariance=np.zeros((2,h,2,2)),
        human_existence=np.array([.7,.2]))
    modes={i:(rng.normal(0,1,(8,h,2)),np.ones(8)/8,np.full(h,v)) for i,v in enumerate((.09,.36))}
    e=q.Direct(cfg,obs,modes)
    for key in ('variance','sigma','radius','component'):
        array=getattr(e,key);array[1]=array[8]
    positions=rng.normal(0,2,(16,h,2));expected=e.exact(positions);context=packed.QueryContext(e)
    for t in THREADS:np.testing.assert_array_equal(context.query(positions,t),expected)
    context.close()
    empty=SimpleNamespace(cfg=cfg,weights=np.empty(0),offsets=np.zeros(1,dtype=np.int64),r=np.empty(0))
    np.testing.assert_array_equal(packed.QueryContext(empty).query(positions),np.zeros((16,h)))
    e.variance[0,0]=np.nan
    try:packed.QueryContext(e)
    except ValueError:pass
    else:raise AssertionError('nonfinite context accepted')
    save('self_test.json',dict(common_parameter_cases=rows,heterogeneous_modes_bitwise=True,
        empty_crowd=True,nonfinite_context_rejected=True));print('SELF_TEST',len(rows),flush=True)


def main():
    global CLASSES
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=12)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    started=time.perf_counter();packed.library();compile_ms=1000*(time.perf_counter()-started)
    q.library();CLASSES={t:packed.make_direct_class(q.Direct,t) for t in THREADS}
    if args.self_test:self_test();return
    p=json.loads((audit.OUT/'protocol.json').read_text())
    files=[Path(__file__).resolve(),ROOT/'experiments/packed_direct_execution/packed_direct.py',ROOT/'experiments/packed_direct_execution/risk_execution.cpp']
    hardware=dict(allowed=ALLOWED,physical_cpus=PHYSICAL,threads=THREADS,
        omp={k:os.environ[k] for k in ('OMP_PLACES','OMP_PROC_BIND','OMP_DYNAMIC','OMP_WAIT_POLICY')},
        lscpu=subprocess.check_output(['lscpu'],text=True),
        quota={str(f):f.read_text().strip() for f in [Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),Path('/sys/fs/cgroup/cpu/user.slice/cpu.cfs_quota_us')] if f.exists()})
    save('protocol.json',dict(base='82857b3af7bf1c319f3a695531ee9f624fa48f38',hardware=hardware,
        states=p['states'],settings=p['settings'],config=p['config'],repetitions=3,
        gate='all 36 full-pipeline measurements <=250 ms for at least one packed thread configuration',
        initial_library_load_or_build_ms=compile_ms,
        package_sha256='5ec34194c0c365bab18deb28b6d9f6f3feefe9dff2c500955bb04a2ef1444550',
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
    self_test()
    records={r['case']:r for r in audit.old.data()};calibration=json.loads((audit.old.OUT/'calibration.json').read_text())
    path=OUT/'states.json';rows=json.loads(path.read_text()) if path.exists() else []
    variants=[0]+THREADS
    for index,(case,step,tid) in enumerate(p['states'][:args.limit]):
        if any(r['state']==[case,step,tid] for r in rows):continue
        print('PREPARE',case,flush=True);record=records[case]
        with fast.enabled(behavior=True,risk=False):env,adapter,engine,warm=audit.prepare(record,step,p,calibration)
        a=pipeline(env,adapter,engine,warm,record,step,0,True)
        positions=a['candidate_positions'];cfg=audit.MPCConfig(**p['config'])
        expected=q.Direct(cfg,a['obs'],a['mixtures']).exact(positions)
        errors=[]
        for t in THREADS:
            e=CLASSES[t](cfg,a['obs'],a['mixtures']);value=e.exact(positions)
            np.testing.assert_array_equal(value,expected)
            b=pipeline(env,adapter,engine,warm,record,step,t,True)
            np.testing.assert_array_equal(a['controls'],b['controls'])
            assert len(a['traces'])==len(b['traces'])
            assert all(x['elites']==y['elites'] and x['winner']==y['winner'] for x,y in zip(a['traces'],b['traces']))
            errors.append(dict(threads=t,maximum_hazard_error=float(np.max(abs(value-expected))),cpus=placement(e,t),controls_bitwise=True,ordered_elites_equal=True))
        measures=[]
        for repeat in range(3):
            shift=(index+repeat)%len(variants)
            for t in variants[shift:]+variants[:shift]:
                b=pipeline(env,adapter,engine,warm,record,step,t)
                np.testing.assert_array_equal(a['controls'],b['controls'])
                measures.append(dict(threads=t,repetition=repeat,**b['timing']))
        rows.append(dict(state=[case,step,tid],people=record['people'],candidates=len(positions),errors=errors,measures=measures))
        save('states.json',rows)
        print('DONE',case,{t:round(np.median([m['total'] for m in measures if m['threads']==t]),2) for t in variants},flush=True)
    table=[]
    for people in (5,10,20):
        for t in variants:
            values=[m['total'] for r in rows if r['people']==people for m in r['measures'] if m['threads']==t]
            if values:table.append(dict(people=people,threads=t,p50=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=max(values),late=float(np.mean(np.array(values)>250))))
    save('summary.json',table)
    accepted=[t for t in THREADS if len(rows)==12 and all(m['total']<=250 for r in rows for m in r['measures'] if m['threads']==t)]
    save('gate.json',dict(accepted_threads=accepted,selected=min(accepted) if accepted else None,
        next_stage='existing 18-record shadow replay' if accepted else 'stop: full-state budget not passed'))
    print('SUMMARY',table,'ACCEPTED',accepted,flush=True)


if __name__=='__main__':main()
