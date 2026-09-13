"""Read-only analysis and figures for the frozen closeout experiment."""
import copy
import hashlib
import io
import json
import pickle
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'archive/legacy')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.occlusion_confirmation import OUT, verify, legacy, matched, build, CROWD
from experiments.occlusion_closeout import save


def mechanisms(p):
    calibration=json.loads((OUT/'calibration.json').read_text())
    entry=calibration['age_margin'][f'{p["age_target"]:.2f}']
    results=[]
    pictures=[]
    for path in sorted((OUT/'mechanism').glob('*.pkl'),
                       key=lambda f:tuple(int(v) for v in f.stem.split('_'))):
        snapshot=pickle.loads(path.read_bytes())
        obs=snapshot['observation']
        planner=legacy.ContinuousCEMMPC(legacy.MPCConfig(**p['configs']['bayes']))
        seed=snapshot['case']*100003+snapshot['step']*97+1729
        rng=np.random.default_rng(seed)
        bases=np.concatenate((planner._initial_mean(obs)[None],planner._route_seeds(obs)))
        noise=rng.standard_normal((512,16,2))
        noise[:,1:]=.68*noise[:,:-1]+.32*noise[:,1:]
        raw=bases[np.arange(512)%len(bases)]+planner.cfg.init_std*noise
        seeds=planner._seed_trajectories(obs)
        raw[:len(seeds)]=seeds
        params,controls,positions=planner._rollout(raw,obs)
        state=dict(scene=snapshot['scene'],case=snapshot['case'],step=snapshot['step'],
                   design='common finite candidate pool, cold-start diagnostic; not actual warm-start CEM replay',arms={})
        plot=[]
        for arm in ('posterior_mean','bayes','age_margin'):
            o=copy.deepcopy(obs)
            if arm=='posterior_mean':
                o.human_position_covariance=None
                o.human_existence=None
            elif arm=='age_margin':
                tracks=[t for t in snapshot['tracks'].values() if t.visible or t.existence>=.15]
                assert len(tracks)==len(o.entities)
                ages=np.array([t.missed_steps for t in tracks])
                o.human_uncertainty_buffer=entry['scale']*((ages[:,None]+np.arange(1,17)[None])*.25)**entry['power']
                o.human_position_covariance=None
                o.human_existence=None
            hc=planner._human_clearance(controls,o) if o.entities.size else None
            occupancy=o.occupancy_probability.sample(positions) if o.occupancy_probability is not None else None
            hazard=planner._belief_collision_hazard(controls,o,positions)
            cost=planner._cost(controls,o,positions,hc,occupancy,hazard)
            full,first=planner._combined_clearance(controls,o,positions,hc,occupancy,hazard)
            _,physical=planner._physical_clearance(controls,o,positions,hc)
            if (full>=0).any():
                pool=np.flatnonzero(full>=0);winner=int(pool[np.argmin(cost[pool])]);level=0
            elif (first>=0).any():
                pool=np.flatnonzero(first>=0);pool=pool[full[pool]>=full[pool].max()-.02]
                winner=int(pool[np.argmin(cost[pool])]);level=1
            else:
                winner=int(planner._degraded_choice(cost,physical,params,o));level=2
            state['arms'][arm]=dict(winner=winner,action=controls[winner,0].tolist(),
                feasibility_class=level,full_feasible=int((full>=0).sum()),first_feasible=int((first>=0).sum()),
                cost=float(cost[winner]))
            plot.append((arm,positions[winner].copy(),full>=0))
        state['action_delta_bayes_mean']=float(np.linalg.norm(np.array(state['arms']['bayes']['action'])-
                                                            state['arms']['posterior_mean']['action']))
        results.append(state);pictures.append((snapshot,positions,plot))
    save(OUT/'mechanism_summary.json',dict(states=results,
        caution='Action changes alone do not establish better navigation; no truth enters candidate scoring.'))
    if pictures:
        columns=2
        fig,axes=plt.subplots((len(pictures)+1)//2,columns,figsize=(11,3.3*((len(pictures)+1)//2)),squeeze=False)
        colors={'posterior_mean':'#666666','bayes':'#0072b2','age_margin':'#d55e00'}
        for ax,(snapshot,positions,plot) in zip(axes.ravel(),pictures):
            o=snapshot['observation']
            for trajectory in positions[::16]:ax.plot(*trajectory.T,color='#dddddd',linewidth=.5)
            for arm,trajectory,_ in plot:ax.plot(*trajectory.T,label=arm,color=colors[arm],linewidth=2)
            ax.scatter(*o.robot_xy,color='black',marker='s');ax.scatter(*o.goal_xy,color='green',marker='*')
            for i,e in enumerate(o.entities):
                ax.plot(*o.human_segment_end[i].T,color='#999999',linestyle=':')
                ax.add_patch(plt.Circle(e[:2],e[4],color='#bbbbbb',alpha=.6))
            ax.set_title(f"Scene {snapshot['scene']}, step {snapshot['step']}: common candidate evaluation")
            ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8)
        for ax in axes.ravel()[len(pictures):]:ax.set_visible(False)
        fig.tight_layout();fig.savefig(OUT/'decision_mechanisms.pdf');plt.close(fig)
        # The main-text panels are selected by configuration and time, not outcome.
        chosen=[item for item in pictures if item[0]['scene']==3]
        if chosen:
            fig,axes=plt.subplots(1,len(chosen),figsize=(6*len(chosen),4),squeeze=False)
            for ax,(snapshot,positions,plot) in zip(axes.ravel(),chosen):
                o=snapshot['observation']
                for trajectory in positions[::16]:ax.plot(*trajectory.T,color='#dddddd',linewidth=.5)
                for arm,trajectory,_ in plot:ax.plot(*trajectory.T,label=arm,color=colors[arm],linewidth=2)
                ax.scatter(*o.robot_xy,color='black',marker='s')
                for i,e in enumerate(o.entities):
                    ax.plot(*o.human_segment_end[i].T,color='#999999',linestyle=':')
                    ax.add_patch(plt.Circle(e[:2],e[4],color='#bbbbbb',alpha=.6))
                ax.set_title(f"Dense square, step {snapshot['step']}")
                ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(OUT/'decision_main.pdf');plt.close(fig)


def uncertainty_trace(p, rows):
    """Replay recorded actions, not a new navigation trial; truth only scores tracks."""
    calibration=json.loads((OUT/'calibration.json').read_text())
    cfg=legacy.MPCConfig(**p['configs']['bayes'])
    _,_,ActionXY,_,_=legacy._load_modules(CROWD)
    traces=[]
    fig,axes=plt.subplots(3,2,figsize=(11,9))
    for scene,ax in enumerate(axes.ravel()):
        row=next(r for r in rows if r['scene']==scene and r['case_id']==p['cases'][0]
                 and r['method']=='bayes' and r['sensor']=='range_and_occlusion')
        env=build(scene,row['case_id'])
        adapter=matched.MatchedAdapter('bayes',cfg.horizon,cfg.human_margin,cfg.dt,
            cfg.chance_limit,cfg.fixed_uncertainty_radius,cfg.acceleration_std,
            method='bayes',calibration=calibration,point=p['points']['bayes'])
        histories={}
        for step in row['steps']:
            adapter.read(env)
            for identifier in adapter.reported_ids:
                track=adapter.rfs.tracks[identifier]
                human=env.humans[identifier]
                error=float(np.linalg.norm(track.mean[:2]-np.array([human.px,human.py])))
                histories.setdefault(identifier,[]).append(dict(t=float(env.global_time),
                    error=error,variance=float(track.covariance[0,0]),missed=track.missed_steps))
            env.step(ActionXY(step['action_a'],step['action_b']))
            if not np.allclose([env.robot.px,env.robot.py],[step['x'],step['y']],atol=1e-9,rtol=0):
                raise RuntimeError('Recorded-action replay does not recover robot state')
        identifier=max(histories,key=lambda i:(sum(s['missed']>0 for s in histories[i]),-i))
        h=histories[identifier]
        traces.append(dict(scene=scene,case=row['case_id'],track=identifier,history=h,
                           selection='first registered Bayes layout; most missing track-steps, smallest ID breaks ties'))
        t=[s['t'] for s in h]
        ax.fill_between(t,0,1,where=[s['missed']>0 for s in h],
                        transform=ax.get_xaxis_transform(),color='black',alpha=.05,
                        label='Missing observation')
        ax.plot(t,[s['error'] for s in h],label='Actual position error',color='#d55e00')
        ax.plot(t,[np.sqrt(5.991464547107979*s['variance']) for s in h],label='Model 95% radius',color='#0072b2')
        ax.set_title(f'{p["scenes"][scene][0]}, track {identifier}')
        ax.set_xlabel('Time (s)');ax.set_ylabel('Distance (m)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'uncertainty_trace.pdf');plt.close(fig)
    save(OUT/'uncertainty_trace.json',traces)


def report():
    p=verify()
    summary=json.loads((OUT/'confirmation_summary.json').read_text())
    if not summary['complete']:
        raise RuntimeError('Cannot generate final report from incomplete confirmation')
    rows=[json.loads(line) for line in (OUT/'episodes.jsonl').read_text().splitlines()]
    failures=[]
    opportunities=[]
    for scene in range(6):
        subset=[r for r in rows if r['method']=='bayes' and r['sensor']=='range_and_occlusion' and r['scene']==scene]
        steps=[s for r in subset for s in r['observation_stats']]
        hidden=sum(s['hidden'] for s in steps)
        count=sum(s['opportunity'] for s in steps)
        opportunities.append(dict(scene=scene,steps=len(steps),hidden_track_steps=hidden,
            eligible_track_steps=count,eligible_fraction=count/hidden if hidden else None,
            states_with_opportunity=sum(s['opportunity']>0 for s in steps),
            caveat='7x7 Gaussian quadrature diagnostic; unobserved out-of-grid space remains unknown; not a bound or a decision-value test'))
    save(OUT/'opportunity_summary.json',opportunities)
    keyed={(r['scene'],r['case_id'],r['method'],r['sensor']):r for r in rows}
    rng=np.random.default_rng(20260914)
    indices=rng.integers(0,100,(10000,100))
    attribution={}
    for field in ('collision_union','success_without_overlap','penalty'):
        blocks=np.array([[keyed[s,c,'covariance','range_and_occlusion'][field]-
                          keyed[s,c,'posterior_mean','range_and_occlusion'][field]
                          for s in range(6)] for c in p['cases']]).mean(axis=1)
        attribution[field]=dict(delta=float(blocks.mean()),
            ci95=np.quantile(blocks[indices].mean(axis=1),[.025,.975]).tolist())
    save(OUT/'covariance_attribution.json',attribution)
    for arm in p['arms']:
        subset=[r for r in rows if r['method']==arm and r['sensor']=='range_and_occlusion']
        collisions=[r for r in subset if r['collision_union']]
        hidden_collision=degraded_collision=stalled_timeout=0
        for r in subset:
            if r['collision_union']:
                t=r['first_actual_overlap_time'] or r['nav_time']
                index=max(0,min(len(r['steps'])-1,int(round(t/.25))-1))
                hidden_collision+=int(r['observation_stats'][index]['hidden']>0)
                degraded_collision+=int(r['steps'][index].get('feasibility_class')==2)
            if not r['collision_union'] and not r['success_without_overlap']:
                # At least a two-second interval with less than 0.10 m goal progress.
                distances=np.array([s['goal_distance'] for s in r['steps']])
                stalled_timeout+=int(len(distances)>8 and np.any(distances[:-8]-distances[8:]<.10))
        failures.append(dict(method=arm,collisions=len(collisions),
            any_hidden_track_at_collision=hidden_collision,degraded_at_collision=degraded_collision,
            timeout_with_low_progress_interval=stalled_timeout))
    save(OUT/'failure_analysis.json',failures)
    mechanisms(p)
    uncertainty_trace(p, rows)
    table=summary['table']
    names=[s[0] for s in p['scenes']]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm in p['arms']:
        rr=[r for r in table if r['sensor']=='range_and_occlusion' and r['method']==arm]
        axes[0].plot(range(6),[100*r['collision']/r['n'] for r in rr],marker='o',label=arm)
        axes[1].plot(range(6),[r['penalty'] for r in rr],marker='o',label=arm)
    for ax in axes:
        ax.set_xticks(range(6));ax.set_xticklabels(names,rotation=35,ha='right');ax.grid(alpha=.2)
    axes[0].set_ylabel('Audited collision (%)');axes[1].set_ylabel('Penalized time (s)')
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'confirmation_by_configuration.pdf');plt.close(fig)
    lines=['# Occlusion Closeout Results','',
        '4080 registered confirmation episodes completed; 100 additional development episodes are not test data.',
        'No model, risk parameter, or sample-count adjustment based on confirmation outcomes.','',
        '## Decision','',
        '- The position-uncertainty contribution survives independent testing: Bayes has 10 collisions versus 46 for the shared-tracker posterior mean; the paired reduction is 6.0 percentage points, with a 95% seed-block interval of 4.0 to 8.0 points.',
        '- The former freezing explanation does not survive the stronger age-margin calibration: both methods have 586 successes, and Age has three timeouts versus four for Bayes. Bayes has a shorter penalized time, not established equal safety.',
        '- Planning existence weighting has no demonstrated additional benefit: covariance-one has seven collisions and 589 successes. The full method is not uniformly best across configurations.',
        '- Bayes does not establish a collision advantage over EWMA, Conformal, or Age after the registered secondary comparisons. These are competitive alternatives, not failed baselines.',
        '- The visibility subset does not establish an extra human-occlusion-specific effect beyond range restriction; the interaction intervals include zero.',
        '- The defensible paper is a fully specified navigation system and controlled mechanism study. These experiments do not create a new Bayesian theorem or guarantee journal acceptance. No additional algorithm branch or outcome-seeking queue is launched.','',
        '## Main Table','',
        '| Configuration | Method | n | Success | Collision | Timeout | Penalty (s) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        if r['sensor']=='range_and_occlusion':
            lines.append(f"| {names[r['scene']]} | {r['method']} | {r['n']} | {r['success']} | {r['collision']} | {r['timeout']} | {r['penalty']:.3f} |")
    lines+=['','## Paired Comparisons','',
            'Intervals resample the 100 seed blocks, each containing six configurations. Secondary collision p-values use Holm correction.']
    for r in summary['comparisons']:
        lines.append('\nBayes minus '+r['b']+': '+json.dumps(r,ensure_ascii=False))
    lines+=['','## Visibility Attribution','',
        'Each row contains 120 layouts. Combined observations reuse the registered main comparison.',
        '| Visibility | Method | Success | Collision | Timeout | Penalty (s) |',
        '|---|---|---:|---:|---:|---:|']
    visibility=[]
    for sensor in ('range_and_occlusion','range_only','full'):
        for arm in ('posterior_mean','covariance'):
            subset=[r for r in rows if r['sensor']==sensor and r['method']==arm
                    and r['case_id'] in p['visibility_cases']]
            result=dict(sensor=sensor,method=arm,n=len(subset),
                success=sum(r['success_without_overlap'] for r in subset),
                collision=sum(r['collision_union'] for r in subset),
                timeout=sum(not r['success_without_overlap'] and not r['collision_union'] for r in subset),
                penalty=float(np.mean([r['penalty'] for r in subset])))
            visibility.append(result)
            lines.append(f"| {sensor} | {arm} | {result['success']} | {result['collision']} | {result['timeout']} | {result['penalty']:.3f} |")
    save(OUT/'visibility_summary.json',dict(table=visibility,effects=summary['visibility_effects']))
    for effect in summary['visibility_effects']:lines.append('\n'+json.dumps(effect))
    lines+=['','## Complete Outcome Transitions','',
        'Rows are Bayes outcomes and columns are comparator outcomes; all transitions are retained.']
    transitions={}
    def event(r):
        return 'collision' if r['collision_union'] else ('success' if r['success_without_overlap'] else 'timeout')
    for other in p['arms']:
        if other=='bayes':continue
        matrix={a:{b:0 for b in ('success','collision','timeout')} for a in ('success','collision','timeout')}
        for case in p['cases']:
            for scene in range(6):
                matrix[event(keyed[scene,case,'bayes','range_and_occlusion'])][
                    event(keyed[scene,case,other,'range_and_occlusion'])]+=1
        transitions[other]=matrix
        lines.append('\n'+other+': '+json.dumps(matrix))
    save(OUT/'outcome_transitions.json',transitions)
    lines+=['','## Data and Code Map','',
        '| Evidence | Data | Code |', '|---|---|---|',
        '| Frozen allocation and loaded parameters | confirmation_protocol.json; calibration.json | experiments/occlusion_confirmation.py |',
        '| Added age-margin development points | development.jsonl; development_summary.json | experiments/occlusion_closeout.py |',
        '| New main and visibility results | episodes.jsonl; confirmation_summary.json | experiments/occlusion_confirmation.py summarize |',
        '| Covariance attribution and full event transitions | covariance_attribution.json; outcome_transitions.json | experiments/occlusion_report.py |',
        '| Same-state candidate mechanisms | mechanism/*.pkl; mechanism_summary.json | experiments/occlusion_report.py |',
        '| Recorded-action uncertainty trace | uncertainty_trace.json | experiments/occlusion_report.py |',
        '| Historical manuscript and numerical tables | manuscript_before.txt; historical archive | Original versions; not new confirmation |']
    lines+=['','## Boundaries','',
        '- No safety-equivalence inference from nonsignificance.',
        '- Opportunity counts are Gaussian quadrature diagnostics, not measured benefits of a negative-evidence algorithm.',
        '- Same-controller controls still differ in risk channels and calibration; they are not immune to fairness criticism.',
        '- Configuration differences combine geometry and pedestrian count, not density alone.',
        '- Concurrent timings do not establish a real-time deployment guarantee.',
        '- The unchanged simulator records its 25 timeout endpoints at 25.25 s on the control grid; all receive the fixed 25 s failure penalty. No success or collision terminates after 25 s.',
        '- Scientific outcomes do not establish novelty or guarantee journal acceptance.']
    lines+=['','## Reproduction','',
        'The compressed bundle contains original episode records, calibration inputs, snapshots, figures, and the manuscript with its local assets. Its manifest records each content hash and original path.',
        'The original protocol is not rewritten to pretend that historical absolute paths were portable. The portable analysis command resolves archived inputs explicitly, verifies their original hashes and the executing core source, and requires the recorded RVO2 binary.',
        '```bash',
        'mkdir -p /tmp/occlusion-closeout-check',
        'tar -xzf results/occlusion_closeout/reproduction_bundle.tar.gz -C /tmp/occlusion-closeout-check',
        'env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/occlusion_report.py portable /tmp/occlusion-closeout-check',
        '# In /tmp/occlusion-closeout-check/manuscript:',
        'tectonic latex_FCS.txt --keep-logs',
        '```',
        'Navigation can be rerun on the original registered paths with `python experiments/occlusion_confirmation.py run --workers 6`; existing episode keys are skipped. Use an explicitly separate result copy for a fresh rerun, not a mixture of old and new timing records.',
        'Portable reanalysis is not an independent simulator installation or a new scientific replication. See portable_analysis_check.json and the existing repository reproducibility documentation for the tested scope.']
    (OUT/'verdict.md').write_text('\n'.join(lines)+'\n')


def package():
    """Archive data and exact inputs without changing the frozen protocol paths."""
    p=verify()
    if not json.loads((OUT/'confirmation_summary.json').read_text())['complete']:
        raise RuntimeError('Refusing to package incomplete confirmation')
    entries={}
    for path in OUT.iterdir():
        if path.is_file() and path.suffix in ('.json','.jsonl','.pdf','.md','.txt') and path.name!='bundle_manifest.json':
            entries['run/'+path.name]=path
    for path in (OUT/'mechanism').glob('*.pkl'):
        entries['run/mechanism/'+path.name]=path
    for path in (OUT/'binding').glob('*'):
        if path.is_file():entries['run/binding/'+path.name]=path
    for name in ('occlusion_report.py','occlusion_binding.py'):
        entries['analysis/'+name]=ROOT/'experiments'/name
    extra=json.loads((OUT/'development_source_manifest.json').read_text())['files']
    for source in set(p['files'])|set(extra):
        path=Path(source)
        if not path.is_absolute():path=ROOT/path
        entries['inputs/'+str(path).lstrip('/')]=path
    cohort=Path('/home/abc/workspace/bayes_occ_mpc/results/first_cohort_20260906_192723')
    historical={}
    for condition,arms in (('occ',('sensor','cv_memory','posterior_mean','bayes')),('full',('bayes','static_cov'))):
        for arm in arms:
            historical[f'{condition}_{arm}']=[f for f in cohort.glob(f'formal_{condition}_{arm}_*.json')
                                               if '.progress.' not in f.name]
    historical['existence']=list(Path('/home/abc/temp/exist_ablation').glob('*.json'))
    historical['age_comparison']=list(Path('/home/abc/temp/formal/D').glob('*.json'))
    historical['severe_comparison']=list(Path('/home/abc/temp/formal/E').glob('*.json'))
    historical['planner_repeat']=list(Path('/home/abc/temp/queueA/out').glob('*_s1_sc*.json'))
    historical['latency']=list(Path('/home/abc/temp/latency/out').glob('*.json'))
    historical['provenance']=[cohort/n for n in ('manifest.json','analysis.json','cross_host_parity.json','calibration.json')]
    historical['source']=list((cohort/'source').rglob('*.py'))+list((cohort/'source').rglob('*.config'))
    historical['source'] += [Path('/home/abc/temp/queueA/run_seed_stability.py'),
                             Path('/home/abc/temp/latency/run_latency_audit_holonomic.py'),
                             Path('/home/abc/temp/formal/run_formal.py')]
    inventory={}
    for group,paths in historical.items():
        inventory[group]=[]
        for path in sorted(paths):
            name='historical/'+str(path).lstrip('/')
            entries[name]=path
            inventory[group].append(dict(path=str(path),bundle_path=name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(OUT/'historical_manifest.json',dict(scope='Historical records, not additional confirmation episodes',groups=inventory))
    entries['run/historical_manifest.json']=OUT/'historical_manifest.json'
    manuscript=Path('/home/abc/workspace/nav_data/mamba/camrl/mamba_log')
    for name in ('latex_FCS.txt','latex_FCS.pdf','fcs.cls','logo.pdf','occlusion_figure.png',
                 'confirmation_by_configuration.pdf','decision_main.pdf','uncertainty_trace.pdf'):
        path=manuscript/name
        if not path.exists():raise FileNotFoundError(path)
        entries['manuscript/'+name]=path
    manifest=dict(scope='Frozen input paths are retained for provenance, not rewritten in the historical protocol.',
        files={name:dict(original_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
               for name,path in sorted(entries.items())})
    save(OUT/'bundle_manifest.json',manifest)
    raw=json.dumps(manifest,indent=2).encode()
    with tarfile.open(OUT/'reproduction_bundle.tar.gz','w:gz',compresslevel=6) as archive:
        for name,path in sorted(entries.items()):archive.add(path,arcname=name,recursive=False)
        info=tarfile.TarInfo('bundle_manifest.json');info.size=len(raw)
        archive.addfile(info,io.BytesIO(raw))
    print('Packaged',len(entries),'files:',OUT/'reproduction_bundle.tar.gz')


def portable_analysis(directory):
    """Reanalyse an extracted bundle without reading its original absolute paths."""
    global OUT,verify
    from experiments import occlusion_confirmation as confirmation
    directory=Path(directory).resolve()
    OUT=directory/'run'
    manifest=json.loads((directory/'bundle_manifest.json').read_text())['files']
    for name,entry in manifest.items():
        if name in ('run/episodes.jsonl','run/confirmation_protocol.json','run/calibration.json') or name.startswith('run/mechanism/'):
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError('Recorded data hash mismatch: '+name)
    p=json.loads((OUT/'confirmation_protocol.json').read_text())
    original_root=next(s.split('/archive/legacy/')[0] for s in p['files']
                       if s.endswith('/archive/legacy/continuous_mpc_gate.py'))
    checked=[]
    for source,expected in p['files'].items():
        absolute=source if source.startswith('/') else original_root+'/'+source
        archived=directory/'inputs'/absolute.lstrip('/')
        if hashlib.sha256(archived.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Archived input hash mismatch: '+source)
        relative=absolute[len(original_root)+1:] if absolute.startswith(original_root+'/') else None
        if relative and relative.startswith(('archive/','vendor/','experiments/')):
            actual=ROOT/relative
            if hashlib.sha256(actual.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Executing code differs from frozen source: '+str(actual))
        checked.append(str(archived))
    import rvo2
    runtime=json.loads((OUT/'runtime.json').read_text())
    if hashlib.sha256(Path(rvo2.__file__).read_bytes()).hexdigest()!=runtime['rvo2_sha256']:
        raise RuntimeError('RVO2 binary differs; cross-build numerical reproduction requires separate validation')
    def checked_protocol():return p
    verify=checked_protocol
    confirmation.verify=checked_protocol
    confirmation.OUT=OUT
    confirmation.summarize()
    report()
    save(OUT/'portable_analysis_check.json',dict(passed=True,checked_files=len(checked),
        loaded_modules=dict(legacy=legacy.__file__,matched=matched.__file__,rvo2=rvo2.__file__),
        data_directory=str(OUT),scope='Relocated data analysis and recorded-action replay on the same installed environment; not a cross-environment installation test'))


def external_scope():
    """Extract repaired adapter diagnostics; never pool with holonomic confirmation."""
    directory=Path('/home/abc/temp/modern/repaired_main')
    source=directory/'test_results.json'
    original=json.loads(source.read_text())
    compact=[]
    for row in original:
        task,r=row['task'],row['result']
        collision=bool(r['collision_union'])
        native=r.get('native_steps',[])
        compact.append(dict(scene=task['scene'][0],humans=task['scene'][1],case=task['case'],
            family=task['candidate']['family'],occluded=task['occluded'],
            success=int(r['success_without_overlap']),collision=int(collision),
            timeout=int(not collision and not r['success_without_overlap']),
            penalty=r['nav_time'] if r['success_without_overlap'] else 25.,
            solver_steps=len(native),solver_successes=sum(bool(s.get('success')) for s in native),
            low_speed_collision=int(collision and abs(r['steps'][-1]['speed'])<.1),
            solved_collision=int(collision and bool(native) and bool(native[-1].get('success'))),
            slack_collision=int(collision and bool(native) and native[-1].get('slack1',0)>1e-9),
            requested_saturations=r['bound_violations'],
            runner_sha256=r['runner_sha256'],binary_sha256=r['binary_sha256'],
            settings_sha256=r['settings_sha256'],layout_sha256=r['layout_sha256']))
    assert len(compact)==3600
    assert len({(r['scene'],r['case'],r['family'],r['occluded']) for r in compact})==3600
    summary=[]
    for occluded in (True,False):
        for scene in [None]+[s[0] for s in matched.SCENES]:
            for family in ('bayes','tmpc','shmpc'):
                rows=[r for r in compact if r['family']==family and r['occluded']==occluded
                      and (scene is None or r['scene']==scene)]
                counts={k:sum(r[k] for r in rows) for k in ('success','collision','timeout','solver_steps',
                    'solver_successes','low_speed_collision','solved_collision','slack_collision','requested_saturations')}
                summary.append(dict(scene=scene,family=family,occluded=occluded,n=len(rows),**counts,
                    penalty=float(np.mean([r['penalty'] for r in rows])),
                    solve_rate=counts['solver_successes']/counts['solver_steps'] if counts['solver_steps'] else None))
    evidence=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Historical repaired continuous-unicycle adapters; distinct from new holonomic confirmation',
        stationary_definition='Speed below 0.1 m/s at END of collision interval; not speed at first contact or causal attribution',
        records=compact,summary=summary)
    save(OUT/'external_scope.json',evidence)
    for row in summary:
        if row['occluded']:
            print(row)


def binding_report():
    directory=OUT/'binding'
    summary=json.loads((directory/'summary.json').read_text())
    if not summary['complete']:
        raise RuntimeError('Binding audit incomplete')
    rows=[json.loads(path.read_text()) for path in directory.glob('[0-5]_*.json')]
    protocol=verify()
    assert {(r['scene'],r['case']) for r in rows}=={
        (s,c) for s in range(6) for c in protocol['cases']}
    cases=protocol['cases']
    rng=np.random.default_rng(2026091301)
    draws=rng.integers(0,len(cases),(10000,len(cases)))
    intervals=[]
    for order in (7,15):
        blocks=[]
        for case in cases:
            steps=[s for r in rows if r['case']==case for s in r['steps']]
            blocks.append([sum(s[str(order)]['max_probability_removal']>=.01 for s in steps),len(steps)])
        blocks=np.asarray(blocks)
        sampled=blocks[draws].sum(axis=1)
        intervals.append(dict(quadrature=order,fraction=float(blocks[:,0].sum()/blocks[:,1].sum()),
            ci=np.quantile(sampled[:,0]/sampled[:,1],[.025,.975]).tolist(),
            unit='100 seed blocks, six configurations per block; descriptive, not a new efficacy test'))
    steps=[s for r in rows for s in r['steps']]
    disagree=sum((s['7']['max_probability_removal']>=.01)!=(s['15']['max_probability_removal']>=.01) for s in steps)
    main=next(r for r in summary['configurations'] if r['scene'] is None and r['quadrature']==15 and r['threshold']==.01)
    decision=('Do not implement negative-evidence mixture in this round: registered potential-binding rate is below 10%.'
              if main['fraction_all']<.1 else
              'Potential relevance threshold passed; this is not a decision-benefit or novelty result.')
    save(directory/'interpretation.json',dict(intervals=intervals,classification_disagreements=disagree,
        states=len(steps),decision=decision,
        limitation='Only risk-reduction potential on selected plans is screened; alternative candidate effects and risk increases are not bounded. '
        'Finite quadrature checks are not certified cell-mass integration. These data have now been inspected and are developmental.',
        mirror_correction='Reflecting all geometry and coupled random draws can reflect any equivariant policy. '
        'This does not isolate posterior shape. An informative contrast holds kinematics, candidates and constraints fixed '
        'and changes only legal visibility evidence; a Gaussian projection control is also needed.'))
    text=['# Selected-plan risk contribution audit','',decision,'',
          'All 600 recorded Bayes layouts were replayed with original seeds and separate warm starts. '
          'Every first command and executed robot position matched within 1e-9. No new navigation outcomes were generated.',
          '', '| Configuration | Control states | Geometric opportunity states | Potential binding states | Fraction of all states |',
          '|---|---:|---:|---:|---:|']
    for row in summary['configurations']:
        if row['quadrature']==15 and row['threshold']==.01:
            name='All' if row['scene'] is None else protocol['scenes'][row['scene']][0]
            text.append(f"| {name} | {row['states']} | {row['opportunity_states']} | {row['potential_binding_states']} | {100*row['fraction_all']:.2f}% |")
    text += ['', 'Binding here means >=1 percentage point maximum per-step collision-probability reduction '
        'after removing ALL geometrically eligible tracks from the selected plan. This is an optimistic diagnostic, '
        'not a valid posterior update, an actual constraint switch, or demonstrated navigation benefit.',
        '',f'7 versus 15 point quadrature classification disagreements: {disagree}/{len(steps)}.',
        '', 'Sensitivity thresholds 0.5 and 2 percentage points, all denominators, and descriptive seed-block intervals '
        'are retained in summary.json and interpretation.json. No efficacy significance test or additional sample selection is performed.']
    (directory/'verdict.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))


def cell_screen_one(row):
    from scipy.special import ndtr
    from experiments.occlusion_binding import eligible
    p=verify()
    cfg=legacy.MPCConfig(**p['configs']['bayes'])
    calibration=json.loads((OUT/'calibration.json').read_text())
    _,_,ActionXY,_,_=legacy._load_modules(CROWD)
    env=build(row['scene'],row['case_id'])
    adapter=matched.MatchedAdapter('bayes',cfg.horizon,cfg.human_margin,cfg.dt,
        cfg.chance_limit,cfg.fixed_uncertainty_radius,cfg.acceleration_std,
        method='bayes',calibration=calibration,point=p['points']['bayes'])
    previous=json.loads((OUT/'binding'/f"{row['scene']}_{row['case_id']}.json").read_text())['steps']
    assert len(previous)==len(row['steps'])
    records=[]
    for step,old in zip(row['steps'],previous):
        adapter.read(env)
        mx,my=adapter.sensor_mesh
        grid=adapter.sensor_grid
        dx,dy=mx[0,1]-mx[0,0],my[1,0]-my[0,0]
        xedges=np.r_[mx[0]-dx/2,mx[0,-1]+dx/2]
        yedges=np.r_[my[:,0]-dy/2,my[-1,0]+dy/2]
        exact=set()
        for j,identifier in enumerate(adapter.reported_ids):
            track=adapter.rfs.tracks[identifier]
            if track.visible or track.covariance[0,0]<=0:continue
            sd=np.sqrt(track.covariance[0,0])
            px=np.diff(ndtr((xedges-track.mean[0])/sd))
            py=np.diff(ndtr((yedges-track.mean[1])/sd))
            mass=py[:,None]*px[None,:]
            free=mass[grid==0].sum()
            unknown=mass[grid==.5].sum()+max(0.,1-mass.sum())
            if free>=.01 and unknown>=.01:exact.add(j)
        coarse=set(eligible(adapter,15))
        assert len(coarse)==old['15']['eligible_tracks']
        bound=old['15']['max_probability_removal']>=.01
        # Track removal has nonnegative additive hazard, giving set-inclusion bounds.
        if not exact:
            low=high=False
        elif exact==coarse:
            low=high=bound
        elif exact.issubset(coarse) and not bound:
            low=high=False
        elif exact.issuperset(coarse) and bound:
            low=high=True
        else:
            low,high=False,True
        records.append(dict(step=old['step'],eligible_tracks=len(exact),
            same_set=exact==coarse,binding_lower=low,binding_upper=high))
        env.step(ActionXY(step['action_a'],step['action_b']))
        if not np.allclose([env.robot.px,env.robot.py],[step['x'],step['y']],atol=1e-9,rtol=0):
            raise RuntimeError('Cell-screen replay mismatch')
    result=dict(scene=row['scene'],case=row['case_id'],steps=records)
    save(OUT/'binding'/f"cell_{row['scene']}_{row['case_id']}.json",result)
    return row['scene'],row['case_id']


def cell_screen():
    from concurrent.futures import ProcessPoolExecutor
    directory=OUT/'binding'
    protocol=dict(scope='Numerical robustness check; does not change primary 15-point diagnostic',
        integration='Separable Gaussian rectangular-cell mass using ndtr; outside grid remains unknown',
        threshold=.01,eligible_mass=.01,population='all 600 frozen Bayes layouts',
        decision='Report set-inclusion lower/upper counts. Do not replace unresolved states with favorable guesses.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=directory/'cell_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Cell protocol changed')
    save(path,protocol)
    assert json.loads((directory/'summary.json').read_text())['complete']
    rows=[json.loads(line) for line in (OUT/'episodes.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['method']=='bayes' and r['sensor']=='range_and_occlusion']
    pending=[r for r in rows if not (directory/f"cell_{r['scene']}_{r['case_id']}.json").exists()]
    with ProcessPoolExecutor(6) as pool:
        for item in pool.map(cell_screen_one,pending):print('Cell screen',*item,flush=True)
    records=[json.loads(path.read_text()) for path in directory.glob('cell_[0-5]_*.json')]
    assert len(records)==600
    result=[]
    for scene in [None]+list(range(6)):
        steps=[s for r in records if scene is None or r['scene']==scene for s in r['steps']]
        result.append(dict(scene=scene,states=len(steps),
            opportunity_states=sum(s['eligible_tracks']>0 for s in steps),
            different_sets=sum(not s['same_set'] for s in steps),
            binding_lower=sum(s['binding_lower'] for s in steps),binding_upper=sum(s['binding_upper'] for s in steps)))
    save(directory/'cell_summary.json',dict(complete=True,results=result,
        caution='Numerical cell-mass integration, not directed-rounding certification. Bounds concern whole-track removal on selected plans only.'))
    print(result)


if __name__=='__main__':
    if sys.argv[1:]==['package']:package()
    elif sys.argv[1:]==['external']:external_scope()
    elif sys.argv[1:]==['binding']:binding_report()
    elif sys.argv[1:]==['cells']:cell_screen()
    elif len(sys.argv)==3 and sys.argv[1]=='portable':portable_analysis(sys.argv[2])
    else:report()
