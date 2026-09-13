"""Read-only analysis and figures for the frozen closeout experiment."""
import copy
import json
import pickle
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'archive/legacy')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.occlusion_confirmation import OUT, verify, legacy, matched
from experiments.occlusion_closeout import save


def mechanisms(p):
    calibration=json.loads((OUT/'calibration.json').read_text())
    entry=calibration['age_margin'][f'{p["age_target"]:.2f}']
    results=[]
    pictures=[]
    for path in sorted((OUT/'mechanism').glob('*.pkl')):
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
        fig,axes=plt.subplots(len(pictures),1,figsize=(7,4*len(pictures)),squeeze=False)
        colors={'posterior_mean':'#666666','bayes':'#0072b2','age_margin':'#d55e00'}
        for ax,(snapshot,positions,plot) in zip(axes[:,0],pictures):
            o=snapshot['observation']
            for trajectory in positions[::16]:ax.plot(*trajectory.T,color='#dddddd',linewidth=.5)
            for arm,trajectory,_ in plot:ax.plot(*trajectory.T,label=arm,color=colors[arm],linewidth=2)
            ax.scatter(*o.robot_xy,color='black',marker='s');ax.scatter(*o.goal_xy,color='green',marker='*')
            for i,e in enumerate(o.entities):
                ax.plot(*o.human_segment_end[i].T,color='#999999',linestyle=':')
                ax.add_patch(plt.Circle(e[:2],e[4],color='#bbbbbb',alpha=.6))
            ax.set_title(f"Scene {snapshot['scene']}, step {snapshot['step']}: common candidate evaluation")
            ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(OUT/'decision_mechanisms.pdf');plt.close(fig)


def report():
    p=verify()
    summary=json.loads((OUT/'confirmation_summary.json').read_text())
    if not summary['complete']:
        raise RuntimeError('Cannot generate final report from incomplete confirmation')
    rows=[json.loads(line) for line in (OUT/'episodes.jsonl').read_text().splitlines()]
    failures=[]
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
        '| Configuration | Method | n | Success | Collision | Timeout | Penalty (s) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        if r['sensor']=='range_and_occlusion':
            lines.append(f"| {names[r['scene']]} | {r['method']} | {r['n']} | {r['success']} | {r['collision']} | {r['timeout']} | {r['penalty']:.3f} |")
    lines+=['','## Paired Comparisons','',
            'Intervals resample the 100 seed blocks, each containing six configurations. Secondary collision p-values use Holm correction.']
    for r in summary['comparisons']:
        lines.append('\nBayes minus '+r['b']+': '+json.dumps(r,ensure_ascii=False))
    lines+=['','## Boundaries','',
        '- No safety-equivalence inference from nonsignificance.',
        '- Opportunity counts are Gaussian quadrature diagnostics, not measured benefits of a negative-evidence algorithm.',
        '- Same-controller controls still differ in risk channels and calibration; they are not immune to fairness criticism.',
        '- Configuration differences combine geometry and pedestrian count, not density alone.',
        '- Concurrent timings do not establish a real-time deployment guarantee.',
        '- Scientific outcomes do not establish novelty or guarantee journal acceptance.']
    (OUT/'verdict.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    report()
