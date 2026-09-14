"""Replay every collision using recorded commands; truth is used only by the auditor."""
import json
import joblib
import numpy as np
from experiments.switching_navigation import Online,envbase,OUT,ROOT
from experiments.switching_decision import save
from integration.crowdnav import BayesObservationAdapter
from nav.contracts import MPCConfig


def run():
    records=[json.loads(l) for l in (OUT/'navigation.jsonl').read_text().splitlines()]
    cfg=MPCConfig(**json.loads((OUT/'navigation_protocol.json').read_text())['config'])
    model=joblib.load(OUT/'deployment.joblib');out=[]
    _,_,ActionXY,_,_=envbase.legacy._load_modules(envbase.CROWD)
    for record in records:
        if not record['collision']:continue
        env=envbase.environment(record);adapter=BayesObservationAdapter(cfg);engine=Online(model,record['arm'])
        for step in record['steps']:
            obs=adapter.read(env)
            mix,_=engine.update(env.global_time,adapter.detected_entities,obs,adapter.reported_ids)
            before=np.array(env.robot.get_position());humans=np.array([h.get_position() for h in env.humans])
            radii=np.array([h.radius+env.robot.radius for h in env.humans])
            env.step(ActionXY(*step['action']))
            after=np.array(env.robot.get_position());human_after=np.array([h.get_position() for h in env.humans])
            np.testing.assert_allclose(after,step['robot_after'],atol=1e-12)
            if step['clearance']>=0:continue
            gaps=[envbase.legacy.swept_min_clearance(before,after,humans[j:j+1],human_after[j:j+1],radii[j:j+1]) for j in range(len(humans))]
            tid=int(np.argmin(gaps));known=tid in adapter.reported_ids
            row=dict(case=record['case'],arm=record['arm'],step=step['step'],collider_id=tid,
                visible=tid in {e['id'] for e in adapter.detected_entities},tracked=known,
                clearance=float(gaps[tid]),feasibility=step['feasibility'])
            if known:
                j=adapter.reported_ids.index(tid)
                row.update(learned_forecast_enabled=j in mix,
                    cv_next_error=float(np.linalg.norm(obs.human_segment_end[j,0]-human_after[tid])),
                    existence=float(obs.human_existence[j]))
                if j in mix:
                    paths,w,v=mix[j]
                    row.update(mode_weights=w.tolist(),mode_next_errors=np.linalg.norm(paths[:,0]-human_after[tid],axis=1).tolist(),
                        variance_first=float(v[0]))
            out.append(row)
    save('collision_audit.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':run()
