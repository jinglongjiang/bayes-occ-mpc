#!/usr/bin/env python3
"""Frozen, paired baseline cohort. The dispatcher is separate from the algorithm."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/abc/miniconda3/envs/crowdnav/bin/python'
SSH = ['ssh', '-S', '/run/user/1000/bayes-4090-control.sock', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=15', 'root@45.126.120.6']
REMOTE_PYTHON = '/root/venvs/bayes_mpc_eval_20260906/bin/python'
TORCH_PATH = '/root/miniconda3/envs/mamba/lib/python3.10/site-packages'
BASELINES = ('orca', 'cadrl', 'lstm', 'sarl', 'dsrnn', 'attngraph')
CHECKPOINTS = {'cadrl':'rl_model-cadrl.pth', 'lstm':'rl_model_lstm.pth',
               'sarl':'rl_model_sarl.pth', 'dsrnn':'dsrnn_27776.pt',
               'attngraph':'attngraph/trained_models/GST_predictor_rand/checkpoints/41665.pt'}
SAVE_LOCK = threading.RLock()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with SAVE_LOCK:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(out):
    crowdnav = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
    source = out / 'source'
    source.mkdir(parents=True)
    for name in ('continuous_mpc_gate.py', 'bayesian_rfs.py', 'evaluate_matched_safety.py',
                 'external_baseline_eval.py', 'test_bayesian_rfs.py', 'run_external_cohort.py',
                 'evaluate_belief_quality.py', 'analyze_paired.py', 'reobservation_mpc.py'):
        shutil.copy2(ROOT / name, source / name)
    for directory in ('crowd_sim', 'crowd_nav/policy', 'crowd_nav/configs'):
        for path in (crowdnav / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.config', '.yaml', '.yml', '.json'):
                target = source / 'environment' / path.relative_to(crowdnav)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in ('crowd_nav/__init__.py', 'crowd_nav/contracts.py'):
        shutil.copy2(crowdnav/name, source/'environment'/name)
    assets = source / 'assets'
    assets.mkdir()
    import baselines
    shutil.copytree(Path(baselines.__file__).parent, assets/'baselines',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for method in ('cadrl', 'sarl', 'lstm', 'dsrnn'):
        directory = 'dsrnn' if method == 'dsrnn' else 'mamba_vl'
        shutil.copy2(crowdnav/'crowd_nav/runs'/directory/CHECKPOINTS[method], assets/CHECKPOINTS[method])
    ag = Path('/home/abc/temp/CrowdNav_Prediction_AttnGraph')
    for directory in ('rl', 'gst_updated', 'trained_models/GST_predictor_rand'):
        shutil.copytree(ag/directory, assets/'attngraph'/directory,
                        ignore=shutil.ignore_patterns('__pycache__', '.git', '*.log'))
    for name in ('arguments.py',):
        shutil.copy2(ag/name, assets/'attngraph'/name)
    prior = ROOT/'results/matched_safety_20260906_115150'
    for name in ('calibration.json', 'frozen_selection.json'):
        shutil.copy2(prior/name, out/name)
    import numpy as np
    calibration=json.loads((out/'calibration.json').read_text())
    rows=np.concatenate([np.load(prior/'calibration_records'/f'case_{c}.npz')['rows']
                         for c in calibration['cases_fit']])
    quantiles=(.50,.68,.90,.95)
    radii=np.empty((2,16,4))
    for visible in (0,1):
        for h in range(1,17):
            errors=np.sort(np.linalg.norm(rows[(rows[:,3]==visible)&(rows[:,2]==h),4:6],axis=1))
            if len(errors)<20:
                raise RuntimeError('Insufficient original conformal fit stratum')
            for j,q in enumerate(quantiles):
                rank=min(len(errors),int(np.ceil((len(errors)+1)*q)))
                radii[visible,h-1,j]=errors[rank-1]
    calibration['audit_conformal_quantile_radii']=radii.tolist()
    calibration['audit_quantiles']=list(quantiles)
    calibration['audit_fit_source']='Original clean 5-person FIT episodes only; no noise/test outcomes'
    save(out/'calibration.json',calibration)
    return source


def layout_signature(env, case):
    import numpy as np
    env.reset(options={'test_case': case})
    rows = [[env.robot.px, env.robot.py, env.robot.gx, env.robot.gy, env.robot.radius]]
    rows += [[h.px, h.py, h.gx, h.gy, h.radius] for h in env.humans]
    return hashlib.sha256(np.asarray(rows, dtype=np.float64).tobytes()).hexdigest()


def worker(out, spec_path, device):
    import numpy as np
    from functools import partial
    from dataclasses import asdict
    source = out/'source'
    sys.path.insert(0, str(source/'environment'))
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source/'assets'))
    from continuous_mpc_gate import build_env, _load_modules, run_episode, ObservationAdapter
    from evaluate_matched_safety import SCENES, config, MatchedAdapter, CVMemoryAdapter, summarize
    from external_baseline_eval import load_controller, run_episode as external_episode
    spec = json.loads(spec_path.read_text())
    root = source/'environment'
    scene = SCENES[spec['scene']]
    _, n, sim, radius, width = scene
    calibration = json.loads((out/'calibration.json').read_text())
    cfg = config(spec['point'])
    if spec.get('smoke'):
        from dataclasses import replace
        cfg = replace(cfg, population=64, iterations=1)
    deadline = spec.get('deadline', 1 if spec.get('smoke') else 25)
    env, env_config, _ = build_env(root, 'sensor', n, sim, radius, width,
                                   time_limit=deadline, occlusion=spec['occlusion'])
    env.robot.v_pref = 1.
    env_config.set('robot', 'v_pref', '1.0')
    if spec['method'] in BASELINES:
        import torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        torch.manual_seed(91627)
        if spec['method'] == 'dsrnn' and spec['occlusion']:
            raise RuntimeError('No audited partial-observation DSRNN adapter')
        checkpoint = source/'assets'/CHECKPOINTS.get(spec['method'], 'unused')
        policy, used_device = load_controller(spec['method'], root, env_config,
            checkpoint, device, source/'assets/attngraph')
    else:
        used_device = 'cpu'
    _, _, action_cls, _ = _load_modules(root)
    rows, layouts, audits = [], {}, []
    begin = time.time()
    for case in range(spec['offset'], spec['offset'] + spec['count']):
        layouts[str(case)] = layout_signature(env, case)
        if spec['method'] in BASELINES:
            row = asdict(external_episode(env, policy, action_cls, spec['method'], case))
        else:
            from evaluate_belief_quality import PairedCalibrationAudit
            from continuous_mpc_gate import ContinuousCEMMPC
            method = spec['method']
            planner_instances=[]
            planner_type=ContinuousCEMMPC
            if method == 'sensor':
                arm, adapter = 'sensor', ObservationAdapter
            elif method == 'cv_memory':
                arm, adapter = 'sensor', partial(CVMemoryAdapter, max_age=spec['max_age'])
            else:
                arm = 'bayes'
                adapter = partial(MatchedAdapter, method=method, calibration=calibration,
                                  point=spec['point'])
            if method in ('branch','blind_branch'):
                from reobservation_mpc import (ReobservationAdapter, ReobservationCEMMPC,
                                               BlindContingencyCEMMPC)
                adapter=partial(ReobservationAdapter,method='bayes',calibration=calibration,point=spec['point'])
                cls=ReobservationCEMMPC if method=='branch' else BlindContingencyCEMMPC
                def planner_type(cfg):
                    instance=cls(cfg);planner_instances.append(instance);return instance
            noise=spec.get('noise',[0.,0.,1.])
            audit=(PairedCalibrationAudit(cfg,calibration,noise,case,spec['point'])
                   if spec.get('calibration_audit') else None)
            row = asdict(run_episode(root, arm, n, sim, case, cfg, radius, width,
                position_noise_std=noise[0],velocity_noise_std=noise[1],
                detection_probability=noise[2], time_limit=deadline,
                adapter_factory=adapter, planner_type=planner_type,
                occlusion=spec['occlusion'],step_observer=audit))
            if planner_instances:
                row['branch_steps']=getattr(planner_instances[0],'branch_steps',0)
            if audit is not None:
                audits.append({'case_id':case,**audit.result()})
        if not np.isfinite([row['nav_time'],row['path_length'],row['min_clearance']]).all():
            raise RuntimeError('invalid result')
        rows.append(row)
        print(f"{spec['name']} {case} {row['event']} {row['nav_time']:.2f}", flush=True)
        save(out/(spec['name']+'.progress.json'), {'completed':len(rows),'expected':spec['count']})
    summary = summarize(rows)
    payload = {'protocol':spec, 'device':used_device, 'seconds':time.time()-begin,
               'source_sha256':{str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')},
               'layout_hashes':layouts, 'episodes':rows, 'summary':summary,
               'calibration_audits':audits}
    save(out/(spec['name']+'.json'), payload)


def report(out):
    import numpy as np
    from evaluate_matched_safety import SCENES, summarize
    groups, layouts = {}, {}
    for path in out.glob('formal_*.json'):
        if path.name.endswith('.progress.json'):
            continue
        data = json.loads(path.read_text())
        spec = data['protocol']
        for case, signature in data['layout_hashes'].items():
            pair = (spec['scene'], case)
            if pair in layouts and layouts[pair] != signature:
                raise RuntimeError('Initial layout differs between paired arms: '+str(pair))
            layouts[pair] = signature
        key = ('occluded' if spec['occlusion'] else 'full') + '/' + spec['method']
        groups.setdefault(key, {}).setdefault(spec['scene'], []).extend(data['episodes'])
    result = {}
    lines = ['# First-Cohort Results', '', 'Development-frozen settings; no score-based environment gate.', '',
             '| Observation / Method | N | Audited SR | Audited CR | TR | Failure-penalized Time |',
             '|---|---:|---:|---:|---:|---:|']
    for key, scenes in sorted(groups.items()):
        all_rows = [row for rows in scenes.values() for row in rows]
        stats = summarize(all_rows)
        result[key] = {'summary':stats,'scenes':{str(s):summarize(r) for s,r in scenes.items()}}
        lines.append(f"| {key} | {len(all_rows)} | {stats['success_without_overlap']:.2%} | "
                     f"{stats['collision_union']:.2%} | {stats['timeout']:.2%} | {stats['audited_penalized_time']:.3f}s |")
    comparisons = {}
    if 'occluded/bayes' in groups:
        for key, scenes in groups.items():
            if not key.startswith('occluded/') or key == 'occluded/bayes':
                continue
            deltas = {k:[] for k in ('collision_union','success_without_overlap','time')}
            for s in range(6):
                b = {r['case_id']:r for r in groups['occluded/bayes'].get(s,[])}
                a = {r['case_id']:r for r in scenes.get(s,[])}
                if set(a) != set(b) or len(a) != 100:
                    break
                for metric in deltas:
                    if metric == 'time':
                        values = [(b[c]['nav_time'] if b[c]['success_without_overlap'] else 25.) -
                                  (a[c]['nav_time'] if a[c]['success_without_overlap'] else 25.) for c in sorted(b)]
                    else:
                        values = [b[c][metric]-a[c][metric] for c in sorted(b)]
                    deltas[metric].append(values)
            else:
                for metric, values in deltas.items():
                    cluster = np.asarray(values).mean(axis=0)
                    rng = np.random.default_rng(182)
                    boot = cluster[rng.integers(0,100,(20000,100))].mean(axis=1)
                    comparisons.setdefault(key,{})[metric] = {'delta':float(cluster.mean()),
                        'exploratory_ci95':np.quantile(boot,[.025,.975]).tolist()}
    save(out/'analysis.json', {'groups':result,'paired_case_cluster_intervals':comparisons,
                             'verified_initial_layouts':len(layouts)})
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def noise_report(out):
    import numpy as np
    from evaluate_matched_safety import summarize
    groups, layouts, calibration = {}, {}, {}
    for path in sorted(out.glob('formal_*.json')):
        if path.name.endswith('.progress.json'):
            continue
        data=json.loads(path.read_text()); spec=data['protocol']
        key=spec['noise_label']+'/'+spec['method']
        groups.setdefault(key,[]).extend(data['episodes'])
        for case,h in data['layout_hashes'].items():
            pair=(spec['scene'],case)
            if pair in layouts and layouts[pair]!=h:
                raise RuntimeError('Noise cohort layout mismatch')
            layouts[pair]=h
    for path in sorted(out.glob('audit_*.json')):
        if path.name.endswith('.progress.json'):
            continue
        data=json.loads(path.read_text()); label=data['protocol']['noise_label']
        for episode in data['calibration_audits']:
            for row in episode['strata']:
                key=(label,row['method'],row['detected_at_origin'],row['missed_bin'],row['horizon'])
                bucket=calibration.setdefault(key,{'n':0,'hits':np.zeros(4),'areas':np.zeros(4),'nll':0.})
                bucket['n']+=row['n'];bucket['hits']+=row['coverage_hits'];bucket['areas']+=row['area_sum']
                if row['conditional_nll_sum'] is not None:bucket['nll']+=row['conditional_nll_sum']
    summary={k:summarize(v) for k,v in groups.items()}
    comparisons={}
    from analyze_paired import binary_pair
    for label in ('moderate','severe'):
        b={r['case_id']:r for r in groups.get(label+'/bayes',[])}
        for name in ('static_cov','ewma','conformal'):
            a={r['case_id']:r for r in groups.get(label+'/'+name,[])}
            if len(a)!=600 or set(a)!=set(b):continue
            comparison={}
            for metric in ('collision_union','success_without_overlap','timeout'):
                x=np.array([b[c][metric] for c in sorted(b)]);y=np.array([a[c][metric] for c in sorted(b)])
                comparison[metric]=binary_pair(x,y)
            delta=np.array([(b[c]['nav_time'] if b[c]['success_without_overlap'] else 25.)-
                            (a[c]['nav_time'] if a[c]['success_without_overlap'] else 25.) for c in sorted(b)])
            rng=np.random.default_rng(182)
            bootstrap=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            comparison['audited_time']={'delta':float(delta.mean()),'exploratory_ci95':np.quantile(bootstrap,[.025,.975]).tolist()}
            comparisons[label+'/'+name]=comparison
    calibration_rows=[]
    for key,value in sorted(calibration.items()):
        label,method,visible,age,h=key;n=value['n']
        calibration_rows.append({'noise':label,'method':method,'detected_at_origin':visible,
            'missed_bin':age,'horizon':h,'n':n,'coverage':(value['hits']/n).tolist(),
            'ellipse_or_disk_area':(value['areas']/n).tolist(),
            'conditional_nll':value['nll']/n if method!='conformal' else None})
    save(out/'analysis.json',{'navigation':summary,'paired_comparisons':comparisons,
         'calibration':calibration_rows,'calibration_intervals':'Per-episode sufficient statistics retained in audit files; pooled coverage is descriptive.',
         'scope':'Fixed risk point p2, 20-person dense square; not six-scene macro or an updated filter.'})
    lines=['# Matched Noise Cohort','','| Noise / Method | N | Audited SR | Audited CR | TR | Failure-penalized Time |',
           '|---|---:|---:|---:|---:|---:|']
    for k,v in sorted(summary.items()):
        lines.append(f"| {k} | {v['n']} | {v['success_without_overlap']:.2%} | {v['collision_union']:.2%} | {v['timeout']:.2%} | {v['audited_penalized_time']:.3f}s |")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def dispatch(out, remote_root, prepare_only=False, cohort='external'):
    source = out/'source'
    sys.path.insert(0,str(source))
    from evaluate_matched_safety import SCENES
    manifest = json.loads((out/'manifest.json').read_text())
    completed, blocked, unavailable = [], [], []
    stage_status = {}
    remote_slots = threading.BoundedSemaphore(5)
    prior_status = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    prior_completed = {r['name']:r for r in prior_status.get('completed',[])}
    started, last_note = time.time(), time.time()
    def note(stage):
        nonlocal last_note
        path = Path('/home/abc/temp/answer.md')
        previous = path.read_text(encoding='utf-8')
        save(out/'handoff_snapshot.json',{'time':dt.datetime.now().isoformat(),'recent_text':previous[-14000:]})
        with path.open('a',encoding='utf-8') as stream:
            stream.write('\n\n## '+dt.datetime.now().strftime('%Y-%m-%d %H:%M KST')+'：首批对照自动进度\n\n')
            stream.write(f'阶段：{stage}；完成{len(completed)}个批任务，失败{len(blocked)}个。结果：`{out}`。\n')
            stream.write(('本次做单事件重新检测分支与同搜索预算消融；不是SH-MPC。' if cohort=='branch' else
                         '本次做同风险预算噪声对照与同观测校准；不自动修改滤波器。' if cohort=='noise' else
                         '本次只做协议验收、不遮挡外部基准、遮挡sensor/均值/简单记忆/完整Bayes；未实现主动观测。')+
                         '自动记录不代表已阅读并答复CC新意见。\n')
        last_note = time.time()

    def validate_result(spec):
        path=out/(spec['name']+'.json')
        if not path.exists():
            return None
        data=json.loads(path.read_text())
        if data['protocol'] != spec:
            raise RuntimeError('cached job protocol mismatch: '+spec['name'])
        if [r['case_id'] for r in data['episodes']] != list(range(spec['offset'],spec['offset']+spec['count'])):
            raise RuntimeError('incomplete or mispaired worker result')
        for path, sha in data['source_sha256'].items():
            if manifest['source_sha256'][path] != sha:
                raise RuntimeError('source drift: '+path)
        return data

    def execute_once(spec, host, device='cpu', attempt=0):
        save(out/'jobs'/(spec['name']+'.json'),spec)
        name = spec['name']
        if host == 'remote':
            subprocess.run(['rsync','-a','-e',shlex.join(SSH[:-1]),
                str(out/'jobs'/(name+'.json')), SSH[-1]+':'+remote_root+'/jobs/'],check=True)
            bootstrap = "import sys,runpy;sys.path.append('"+TORCH_PATH+"');runpy.run_path(sys.argv.pop(1),run_name='__main__')"
            cmd = [*SSH, shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
                *(['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1'] if spec['method']=='attngraph' else []),
                'timeout','--kill-after=15s','5400',REMOTE_PYTHON,'-c',bootstrap,
                remote_root+'/source/run_external_cohort.py','worker','--out',remote_root,
                '--spec',remote_root+'/jobs/'+name+'.json','--device',device])]
        else:
            cmd = [sys.executable,str(source/'run_external_cohort.py'),'worker','--out',str(out),
                   '--spec',str(out/'jobs'/(name+'.json')),'--device',device]
        begin = time.time()
        log_path=out/(name+'.log')
        if log_path.exists():
            archive=out/'attempt_logs'
            archive.mkdir(exist_ok=True)
            shutil.copy2(log_path,archive/(name+f'.{time.time_ns()}.log'))
        with log_path.open('w') as log:
            process = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=5500)
        if process.returncode:
            return {'name':name,'host':host,'device':device,'failed':True,'code':process.returncode}
        if host == 'remote':
            subprocess.run(['rsync','-a','-e',shlex.join(SSH[:-1]),
                SSH[-1]+':'+remote_root+'/'+name+'.json',str(out)+'/' ],check=True)
        validate_result(spec)
        return {'name':name,'host':host,'device':device,'seconds':time.time()-begin,'failed':False}

    def execute(spec, host, device='cpu'):
        cached=validate_result(spec)
        if cached is not None:
            previous=prior_completed.get(spec['name'],{})
            return {'name':spec['name'],'host':previous.get('host',host),
                    'device':cached['device'],'seconds':previous.get('seconds',cached['seconds']),
                    'failed':False,'reused':True}
        if host!='remote':
            return execute_once(spec,host,device)
        for attempt in range(3):
            with remote_slots:
                try:
                    result=execute_once(spec,host,device,attempt)
                except subprocess.CalledProcessError as error:
                    result={'name':spec['name'],'host':host,'device':device,
                            'failed':True,'code':error.returncode,'transport':True}
            if not result['failed'] or result['code'] not in (12,23,30,35,255):
                return result
            time.sleep(2*(attempt+1))
        return result

    def stage(label, specs, slots):
        pending_specs = list(specs)
        def take(slot):
            for i,s in enumerate(pending_specs):
                if s.get('preferred_host',slot[0])==slot[0]:
                    return pending_specs.pop(i)
            return None
        with futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
            running = {}
            for slot in slots:
                spec = take(slot)
                if spec is not None:
                    running[pool.submit(execute,spec,*slot)] = slot
            while running:
                done,_ = futures.wait(running,timeout=20,return_when=futures.FIRST_COMPLETED)
                for future in done:
                    slot=running.pop(future)
                    row=future.result()
                    (blocked if row['failed'] else completed).append(row)
                    print(label,row,flush=True)
                    spec=take(slot)
                    if spec is not None:
                        running[pool.submit(execute,spec,*slot)] = slot
                with SAVE_LOCK:
                    stage_status[label] = {'active_count':len(running), 'total_jobs':len(specs)}
                    save(out/'status.json',{'stages':stage_status,'completed':completed,'failed':blocked,
                         'active_count':sum(v['active_count'] for v in stage_status.values()),
                         'elapsed_seconds':time.time()-started})
                if time.time()-last_note >= 5400:
                    note(label)
        note(label+' finished')

    def spec(method,scene,offset,count,occlusion,point=2,age=2.,name=None,smoke=False):
        return {'name':name or f"formal_{'occ' if occlusion else 'full'}_{method}_{scene}_{offset}",
                'method':method,'scene':scene,'offset':offset,'count':count,'occlusion':occlusion,
                'point':point,'max_age':age,'smoke':smoke,'robot_visible':False}

    cpu_slots=[('local','cpu')]*6+[('remote','cpu')]*6
    if cohort=='branch':
        import numpy as np
        # Two additional CPU jobs per host can coexist with the four noise jobs.
        cpu_slots=[('local','cpu')]*2+[('remote','cpu')]*2
        smokes=[]
        for host in ('local','remote'):
            for method in ('bayes','branch','blind_branch'):
                s=spec(method,3,4998,1,True,name=f'smoke_branch_{method}_{host}',smoke=True)
                s.update(preferred_host=host,deadline=3,noise=[.1,.2,.8])
                smokes.append(s)
        stage('branch_preflight',smokes,cpu_slots[:2]+cpu_slots[-2:])
        if blocked:raise RuntimeError('Branch smoke failed; no development/formal jobs launched')
        for method in ('bayes','branch','blind_branch'):
            a,b=[json.loads((out/f'smoke_branch_{method}_{h}.json').read_text()) for h in ('local','remote')]
            if a['layout_hashes']!=b['layout_hashes']:
                raise RuntimeError('Branch cross-host initial layout mismatch')
            x,y=a['episodes'][0],b['episodes'][0]
            for name in ('event','solver_steps','actual_overlap_steps'):
                if x[name]!=y[name]:raise RuntimeError('Branch cross-host event mismatch')
            for name in ('nav_time','path_length','actual_min_clearance'):
                np.testing.assert_allclose(x[name],y[name],rtol=1e-8,atol=1e-8)
            if method!='bayes' and not x['branch_steps']:
                raise RuntimeError('Smoke did not exercise re-observation; expand functional fixture')
        save(out/'cross_host_parity.json',{'passed':True,'methods':3,'dynamics_tolerance':1e-8})
        if prepare_only:return
        development=[];formal=[]
        for scene in range(6):
            for method in ('bayes','branch','blind_branch'):
                for offset in (4800,4805):
                    s=spec(method,scene,offset,5,True,name=f'pilot_{method}_{scene}_{offset}')
                    s['preferred_host']='local' if offset==4800 else 'remote'
                    development.append(s)
                for offset in range(10000,10100,10):
                    s=spec(method,scene,offset,10,True)
                    s['preferred_host']='local' if offset<10050 else 'remote'
                    formal.append(s)
        stage('branch_development_functional',development,cpu_slots)
        if blocked:raise RuntimeError('Branch development infrastructure failed')
        stage('branch_frozen_six_config_validation',formal,cpu_slots)
        report(out)
        save(out/'complete.json',{'seconds':time.time()-started,'jobs_completed':len(completed),
            'failed':blocked,'no_training':True,'development_episodes':180,'formal_episodes':1800,
            'cohort':'branch','scope':'Single tracked target, one binary event, approximate visibility. Not SH-MPC or full observation tree.'})
        note('分支首版六配置评测完成；现代MPC对照仍需接口验收')
        return
    if cohort=='noise':
        cpu_slots=[('local','cpu')]*4+[('remote','cpu')]*4
        levels={'clean':[0.,0.,1.],'moderate':[.05,.1,.9],'severe':[.1,.2,.8]}
        def noise_spec(method,label,offset,count,audit=False,smoke=False,host=None):
            name=f"{'smoke_noise' if smoke else ('audit' if audit else 'formal_noise')}_{label}_{method}_{offset}"
            if smoke:name+='_'+host
            s=spec(method,3,offset,count,True,point=2,name=name,smoke=smoke)
            s.update(noise=levels[label],noise_label=label,calibration_audit=audit)
            if host:s['preferred_host']=host
            return s
        note('同预算噪声预检')
        smokes=[noise_spec(m,'severe',4998,1,audit=(m=='bayes'),smoke=True,host=h)
                for h in ('local','remote') for m in ('bayes','static_cov','ewma','conformal')]
        stage('noise_preflight',smokes,cpu_slots[:2]+cpu_slots[-2:])
        if blocked:raise RuntimeError('Noise preflight failed; no full cohort launched')
        for method in ('bayes','static_cov','ewma','conformal'):
            rows=[json.loads((out/f'smoke_noise_severe_{method}_4998_{host}.json').read_text()) for host in ('local','remote')]
            if rows[0]['layout_hashes']!=rows[1]['layout_hashes'] or rows[0]['episodes'][0]['observation_hash']!=rows[1]['episodes'][0]['observation_hash']:
                raise RuntimeError('Cross-host noise parity failed')
            import numpy as np
            for a,b in zip(rows[0]['calibration_audits'],rows[1]['calibration_audits']):
                if a['scored_forecasts']!=b['scored_forecasts'] or len(a['strata'])!=len(b['strata']):
                    raise RuntimeError('Cross-host calibration counts differ')
                for x,y in zip(a['strata'],b['strata']):
                    for field in ('method','horizon','detected_at_origin','missed_bin','n','coverage_hits'):
                        if x[field]!=y[field]:raise RuntimeError('Cross-host calibration stratum differs')
                    for field in ('area_sum','error_sum','conditional_nll_sum'):
                        if x[field] is not None:
                            np.testing.assert_allclose(x[field],y[field],rtol=1e-9,atol=1e-9)
        if prepare_only:return
        audits=[noise_spec('bayes',label,c,10,audit=True,host='local' if c<4250 else 'remote')
                for label in levels for c in range(4200,4300,10)]
        jobs=[noise_spec(method,label,c,20,host='local' if c<6300 else 'remote')
              for label in ('moderate','severe') for c in range(6000,6600,20)
              for method in ('bayes','static_cov','ewma','conformal')]
        stage('same_observation_calibration',audits,cpu_slots)
        if blocked:raise RuntimeError('Calibration infrastructure failed')
        stage('matched_budget_noise_validation',jobs,cpu_slots)
        noise_report(out)
        save(out/'complete.json',{'seconds':time.time()-started,'jobs_completed':len(completed),
             'failed':blocked,'no_training':True,'calibration_episodes':300,'formal_episodes':4800,
             'cohort':'noise','algorithm_changes':False})
        note('噪声批次完成；SH-MPC和新分支不是本批完成项')
        return
    note('双机接口预检')
    smokes=[]
    for host in ('local','remote'):
        for method in BASELINES:
            for device in ('cpu','cuda') if method != 'orca' else ('cpu',):
                s=spec(method,3,4997,1,False,name=f'smoke_{host}_{device}_{method}',smoke=True)
                smokes.append((s,host,device))
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        smoke_results=list(pool.map(lambda item:execute(*item),smokes))
    save(out/'preflight.json',smoke_results)
    usable={}
    for method in BASELINES:
        for host in ('local','remote'):
            options=[r for r in smoke_results if r['name'].endswith('_'+method) and r['host']==host and not r['failed']]
            if options:
                def steady_latency(row):
                    import numpy as np
                    data=json.loads((out/(row['name']+'.json')).read_text())
                    times=data['episodes'][0]['plan_step_ms'][2:]
                    return float(np.median(times))
                usable[(method,host)]=min(options,key=steady_latency)['device']
    save(out/'baseline_devices.json',{m+'@'+h:d for (m,h),d in usable.items()})
    # Verify both hosts can preserve the existing MPC policy before launching it.
    mpc_smokes=[(spec(m,3,4997,1,o,name=f'smoke_mpc_{m}_{o}_{host}',smoke=True),host,'cpu')
                for host in ('local','remote') for o in (True,False)
                for m in ('bayes','sensor','posterior_mean','cv_memory')]
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        mpc_results=list(pool.map(lambda item:execute(*item),mpc_smokes))
    save(out/'mpc_preflight.json',mpc_results)
    if any(r['failed'] for r in mpc_results):
        raise RuntimeError('MPC preflight failed; no formal navigation jobs launched')
    import numpy as np
    for m in ('bayes','sensor','posterior_mean','cv_memory'):
        for o in (True,False):
            a,b=[json.loads((out/f'smoke_mpc_{m}_{o}_{h}.json').read_text()) for h in ('local','remote')]
            if a['layout_hashes'] != b['layout_hashes']:
                raise RuntimeError('Cross-host layout mismatch')
            for field in ('event','observation_hash','actual_overlap_steps','solver_steps'):
                if a['episodes'][0][field] != b['episodes'][0][field]:
                    raise RuntimeError('Cross-host MPC mismatch: '+m+'/'+field)
            for field in ('nav_time','path_length','actual_min_clearance'):
                if not np.isclose(a['episodes'][0][field],b['episodes'][0][field],rtol=1e-10,atol=1e-10):
                    raise RuntimeError('Cross-host MPC numerical mismatch: '+m+'/'+field)
    save(out/'cross_host_parity.json',{'passed':True,'matched_mpc_conditions':8})
    if prepare_only:
        note('预检完成，尚未开始正式评测')
        return
    # Memory baseline receives its own development search; test outcomes cannot select its lifetime.
    points=json.loads((out/'frozen_selection.json').read_text())['points']
    dev=[spec('cv_memory',0,4800,100,True,point=points['bayes'],age=age,
              name=f'dev_memory_{age:g}') for age in (.5,1.,2.,4.)]
    stage('memory_development',dev,cpu_slots[:2]+cpu_slots[-2:])
    values={s['max_age']:json.loads((out/(s['name']+'.json')).read_text())['summary'] for s in dev}
    feasible=[age for age,v in values.items() if v['collision_union'] <= .01]
    age=min(feasible,key=lambda a:values[a]['audited_penalized_time']) if feasible else min(values,key=lambda a:(values[a]['collision_union'],values[a]['audited_penalized_time']))
    save(out/'memory_selection.json',{'max_age':age,'development':values,'test_not_started':True})
    cpu_jobs=[]
    for scene in range(6):
        for offset in range(5000,5100,20):
            for method,o in [('bayes',False),('static_cov',False),('bayes',True),
                             ('sensor',True),('posterior_mean',True),('cv_memory',True),('orca',False)]:
                point=points['static_cov'] if method=='static_cov' else points['bayes']
                cpu_jobs.append(spec(method,scene,offset,20,o,point,age))
    neural=[(m,spec(m,scene,offset,20,False)) for m in BASELINES if m!='orca'
            for scene in range(6) for offset in range(5000,5100,20)]
    # Separate pools prevent the lightweight GPU baselines from queueing behind CPU MPC.
    def neural_stage():
        for method in BASELINES[1:]:
            slots=[(h,usable[(method,h)]) for h in ('local','remote') if (method,h) in usable]
            if not slots:
                unavailable.append(method)
                save(out/(method+'_blocked.json'),{'reason':'Neither host passed the native adapter smoke; no substituted baseline'})
                continue
            stage('baseline_'+method,[s for m,s in neural if m==method],slots)
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(stage,'mpc_and_orca_formal',cpu_jobs,cpu_slots)
        b=pool.submit(neural_stage)
        a.result(); b.result()
    report(out)
    save(out/'complete.json',{'seconds':time.time()-started,'jobs_completed':len(completed),
         'failed':blocked,'unavailable_baselines':unavailable,
         'no_training':True,'closed_loop_actuation':False})
    note('首批评测结束；失败任务如有已单列，不冒称全部通过')


def prepare_modern_solver(out):
    """Build an isolated open-source solver, not a renamed SH-MPC benchmark."""
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    status = {'stage':'starting','commands_completed':[],
              'formal_sh_mpc_evaluation':False,
              'remaining':['Scenario module integration', 'Matched holonomic dynamics',
                           'Observation and control contract tests']}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(label, command, cwd=None):
        status['stage'] = label
        save(out/'status.json', status)
        print(label + ': ' + shlex.join([str(c) for c in command]), flush=True)
        subprocess.run([str(c) for c in command], cwd=cwd, env=env,
                       check=True, timeout=2700)
        status['commands_completed'].append(label)
        save(out/'status.json', status)

    try:
        acados = out/'acados'
        if not acados.exists():
            run('clone_pinned_acados', ['git','clone','--depth','1','--branch','v0.4.0',
                'https://github.com/acados/acados.git',acados])
        run('acados_submodules', ['git','submodule','update','--init','--recursive','--depth','1'], acados)
        status['acados_commit'] = subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=acados,text=True).strip()
        python = out/'venv/bin/python'
        if not python.exists():
            run('isolated_python', [sys.executable,'-m','venv',out/'venv'])
        run('python_dependencies', [python,'-m','pip','install','numpy==1.23.5',
            'scipy==1.10.1','casadi==3.6.7','setuptools<76','wheel',
            '-e',acados/'interfaces/acados_template'])
        run('configure_acados', ['cmake','-S',acados,'-B',acados/'build',
            '-DCMAKE_BUILD_TYPE=Release','-DACADOS_WITH_QPOASES=OFF',
            '-DACADOS_WITH_OPENMP=OFF','-DBLASFEO_TARGET=GENERIC',
            '-DHPIPM_TARGET=GENERIC',f'-DCMAKE_INSTALL_PREFIX={acados}'])
        run('compile_acados', ['cmake','--build',acados/'build','--target','install','-j','2'])
        env['ACADOS_SOURCE_DIR'] = str(acados)
        env['LD_LIBRARY_PATH'] = str(acados/'lib') + ':' + env.get('LD_LIBRARY_PATH','')
        run('library_import_smoke', [python,'-c',
            "import ctypes,os; from acados_template import AcadosOcp; "
            "ctypes.CDLL(os.path.join(os.environ['ACADOS_SOURCE_DIR'],'lib/libacados.so')); "
            "assert AcadosOcp() is not None; print('PASS: acados import and shared library')"])
        status.update(stage='dependencies_ready', seconds=time.time()-start)
        save(out/'status.json',status)
        save(out/'complete.json',status)
    except Exception as error:
        status.update(stage='blocked',error=repr(error),seconds=time.time()-start)
        save(out/'status.json',status)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','run','worker','report','prepare-modern'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--remote-root',default='/root/bayes_external_cohort_20260906')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--wait-for',type=Path)
    parser.add_argument('--cohort',choices=('external','noise','branch'),default='external')
    args=parser.parse_args()
    if args.command=='prepare-modern':
        prepare_modern_solver(args.out)
    elif args.command=='prepare':
        source=snapshot(args.out)
        protocol=({'formal_cases':[10000,10099],'development_cases':[4800,4809],
                   'development_cases_already_seen':True,'smoke_cases':[4998,4998],
                   'risk_point':2,'cohort':'branch','methods':['bayes','branch','blind_branch'],
                   'allocation':'Each machine: 990 episodes; same 6 configurations and budgets.',
                   'mechanism':'One binary re-detection event at step 2 for one missed track; common control prefix.',
                   'approximation':'128 moment-matched joint CV samples; angular disc visibility; only binary observation, not future measured position.',
                   'claim_limits':['No novelty guarantee from contingency concept alone.',
                                   'Development is a functional pilot; no outcome-based retuning before frozen validation.']}
                  if args.cohort=='branch' else
                  {'formal_cases':[6000,6599],'diagnostic_cases':[4200,4299],
                   'diagnostic_cases_already_seen':True,'smoke_cases':[4998,4998],
                   'human_count':20,'scenario':'square_crossing','square_width':10.,
                   'risk_point':2,'position_velocity_detection_noise':{'clean':[0.,0.,1.],
                        'moderate':[.05,.1,.9],'severe':[.1,.2,.8]},
                   'allocation':'Each machine gets 300/600 cases per method/noise; 50/100 diagnostic cases per noise',
                   'cohort':'noise'} if args.cohort=='noise' else
                  {'formal_cases':[5000,5099],'memory_dev_cases':[4800,4899],'smoke_cases':[4997,4997],
                   'cohort':'external'})
        save(args.out/'manifest.json',{'created':dt.datetime.now().isoformat(),
            'source_sha256':{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()},
            'protocol':{'robot_visible':False,'v_max':1.,'dt':.25,'time_limit':25.,
                        **protocol},
            'claim_limits':['Existing model design has seen higher-density data; this is not pristine 5-person-only algorithm design.',
                            'Baseline checkpoint training populations require provenance audit; architecture defaults are not proof.',
                            'Full observation disables both geometric occlusion and the limited FOV.',
                            'No novelty or publication claim follows solely from baseline wins.'],
            'stop_policy':'Stop for data/interface failures, never for missing a predicted success-rate band.'})
        (args.out/'jobs').mkdir()
    elif args.command=='worker':
        worker(args.out,args.spec,args.device)
    elif args.command=='run':
        if args.wait_for is not None and not args.prepare_only:
            save(args.out/'waiting.json',{'prerequisite':str(args.wait_for),
                                        'waiting_since':dt.datetime.now().isoformat()})
            waiting_since=time.monotonic()
            while not args.wait_for.exists():
                if time.monotonic()-waiting_since > 86400:
                    raise RuntimeError('Prerequisite did not complete within one day')
                time.sleep(30)
            if json.loads(args.wait_for.read_text()).get('failed'):
                raise RuntimeError('Upstream batch failed; dependent experiments not started')
            (args.out/'waiting.json').unlink(missing_ok=True)
        dispatch(args.out,args.remote_root,args.prepare_only,args.cohort)
    else:
        (noise_report if args.cohort=='noise' else report)(args.out)


if __name__=='__main__':
    main()
