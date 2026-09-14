"""Paired reporting for factual alarms and completed independent navigation blocks."""
import json
import numpy as np
import pandas as pd
from experiments.switching_decision import OUT,save


def interval(v):
    v=np.asarray(v,float)
    if len(v)<2:return None
    rng=np.random.default_rng(7709)
    return np.quantile(rng.choice(v,(5000,len(v)),replace=True).mean(1),[.025,.975]).tolist()


def factual():
    f=pd.read_csv(OUT/'factual_queries.csv')
    metrics=pd.DataFrame(json.loads((OUT/'factual_metrics.json').read_text()))
    rows=[];pairs=[]
    for radius in (.7,1.):
        for subset in ('all_loso','moving_loso','moving_temporal'):
            g=f[f.radius==radius];m=metrics[metrics.radius==radius]
            if subset=='all_loso':g=g[g.fold!='moving_temporal'];m=m[m.fold!='moving_temporal']
            elif subset=='moving_loso':g=g[(g.fold!='moving_temporal')&g.moving];m=m[m.fold=='1737635531(dog)']
            else:g=g[g.fold=='moving_temporal'];m=m[m.fold=='moving_temporal']
            # A seed is not a new encounter. Average its scores before cluster summaries.
            averaged=g.groupby(['fold','track','time','method'],as_index=False).agg(
                brier=('brier','mean'),event=('event','first'),alarmed=('alarmed','mean'))
            for method,h in averaged.groupby('method'):
                one=m[m.method==method]
                rows.append(dict(radius=radius,subset=subset,method=method,queries=len(h),tracks=h.track.nunique(),
                    event_queries=int(h.event.sum()),event_tracks=int(h[h.event].track.nunique()),
                    brier=float(h.groupby('track').brier.mean().mean()),
                    auc_record_mean=float(one.groupby('fold').auc.mean().mean()),
                    ap_record_mean=float(one.groupby('fold').ap.mean().mean()),
                    fpr_record_mean=float(one.groupby('fold').fpr.mean().mean()),
                    recall_record_mean=float(one.groupby('fold').recall.mean().mean()),
                    missed_seed_mean=float(one.groupby('seed').missed.sum().mean()),
                    false_alarm_seed_mean=float(one.groupby('seed').false_alarms.sum().mean())))
            pivot=averaged.pivot(index=['fold','track','time'],columns='method',values='brier')
            for b in ('CV','TREE','MAP','IID'):
                d=(pivot.FULL-pivot[b]).groupby(['fold','track']).mean()
                pairs.append(dict(radius=radius,subset=subset,comparison='FULL-'+b,
                    track_difference=float(d.mean()),track_ci95=interval(d.values),
                    record_difference=float(d.groupby('fold').mean().mean()),
                    record_ci95=interval(d.groupby('fold').mean().values)))
    save('factual_summary.json',dict(rows=rows,paired_brier=pairs))
    print(pd.DataFrame([r for r in rows if r['radius']==.7]).to_string(index=False))


def navigation():
    records=[json.loads(l) for l in (OUT/'navigation.jsonl').read_text().splitlines()]
    assert len(records)==360
    episodes=[];steps=[]
    for r in records:
        e={k:v for k,v in r.items() if k!='steps'}
        s=r['steps']
        e.update(min_clearance=min(x['clearance'] for x in s),steps=len(s),
            stopped_steps=sum(np.linalg.norm(x['action'])<.05 for x in s),
            eligible_person_steps=sum(x['eligible'] for x in s),visible_person_steps=sum(x['visible'] for x in s),
            total_person_steps=sum(x['tracks'] for x in s),
            geometry_rejected=sum(x['geometry_rejected'] for x in s),risk_permitted=sum(x['risk_permitted'] for x in s),
            actuator_violations=sum(max(x['speed_excess'],x['acceleration_excess'])>1e-6 for x in s),
            workspace_excursion_steps=sum(x['workspace_excess']>1e-6 for x in s))
        episodes.append(e)
        for x in s:steps.append(dict(arm=r['arm'],case=r['case'],**x))
    ep=pd.DataFrame(episodes);st=pd.DataFrame(steps)
    ep.to_csv(OUT/'navigation_episodes.csv',index=False)
    totals=[];density=[];timing=[]
    for arm,g in ep.groupby('arm'):
        h=st[st.arm==arm]
        totals.append(dict(arm=arm,n=len(g),success=int(g.success.sum()),collision=int(g.collision.sum()),
            timeout=int(g.timeout.sum()),penalty=float(g.penalty.mean()),
            mean_navigation_time_success=float(g[g.success==1].nav_time.mean()),
            stopped_step_fraction=float(g.stopped_steps.sum()/g.steps.sum()),
            eligible_visible_fraction=float(g.eligible_person_steps.sum()/max(1,g.visible_person_steps.sum())),
            eligible_all_fraction=float(g.eligible_person_steps.sum()/max(1,g.total_person_steps.sum())),
            geometry_reject_fraction=float(g.geometry_rejected.sum()/max(1,g.risk_permitted.sum())),
            actuator_violations=int(g.actuator_violations.sum()),workspace_excursion_steps=int(g.workspace_excursion_steps.sum())))
        timing.append(dict(arm=arm,steps=len(h),p50_ms=float(h.total_ms.quantile(.5)),
            p95_ms=float(h.total_ms.quantile(.95)),p99_ms=float(h.total_ms.quantile(.99)),
            inference_p50_ms=float(h.inference_ms.quantile(.5)),exceed_250_fraction=float((h.total_ms>250).mean()),
            concurrent=True))
    for (people,scene,arm),g in ep.groupby(['people','scene','arm']):
        density.append(dict(people=int(people),scene=scene,arm=arm,n=len(g),
            success=int(g.success.sum()),collision=int(g.collision.sum()),timeout=int(g.timeout.sum()),
            penalty=float(g.penalty.mean())))
    pairs=[]
    for a,b in [('FULL',x) for x in ('O','CV','TREE','MAP','IID')]+[('CV','O')]:
        aa=ep[ep.arm==a].set_index('case').sort_index();bb=ep[ep.arm==b].set_index('case').sort_index()
        pairs.append(dict(comparison=a+'-'+b,
            success_gained=int(((aa.success==1)&(bb.success==0)).sum()),
            success_lost=int(((aa.success==0)&(bb.success==1)).sum()),
            collision_avoided=int(((aa.collision==0)&(bb.collision==1)).sum()),
            collision_added=int(((aa.collision==1)&(bb.collision==0)).sum()),
            penalty_difference=float((aa.penalty-bb.penalty).mean()),penalty_ci95=interval(aa.penalty-bb.penalty),
            collision_difference=float((aa.collision-bb.collision).mean()),
            collision_ci95=interval(aa.collision-bb.collision) if np.any(aa.collision!=bb.collision) else None,
            zero_discordance_one_sided_upper95=float(1-.05**(1/len(aa))) if not np.any(aa.collision!=bb.collision) else None))
    action=[]
    indexed={(r['case'],r['arm']):r for r in records}
    for b in ('CV','TREE','MAP','IID'):
        changed=[];shared_pose_violations=0
        for case in ep.case.unique():
            a=indexed[case,'FULL']['steps'];bb=indexed[case,b]['steps']
            first=next((i for i,(x,y) in enumerate(zip(a,bb)) if np.linalg.norm(np.array(x['action'])-y['action'])>1e-6),None)
            if first is not None:
                changed.append(dict(case=int(case),step=first))
                if first>0 and not np.allclose(a[first-1]['robot_after'],bb[first-1]['robot_after'],atol=1e-5):shared_pose_violations+=1
        action.append(dict(comparison='FULL-'+b,layouts_with_action_divergence=len(changed),
                           first_divergence=changed,preceding_pose_mismatches=shared_pose_violations))
    save('navigation_summary.json',dict(totals=totals,density=density,paired=pairs,timing=timing,action=action))
    print(pd.DataFrame(totals).to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['factual','navigation']);a=p.parse_args()
    factual() if a.command=='factual' else navigation()
