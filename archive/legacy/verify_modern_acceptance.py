"""Synthetic goal and physical-contact checks for frozen native controllers."""
import copy
import json
import math
from pathlib import Path

import numpy as np

import modern_main_table as run
import modern_repair as repair
from modern_dynamics import ramp_displacement
from continuous_mpc_gate import swept_min_clearance


def case(candidate, name, position, heading, speed, obstacle):
    root = run.OUT / "acceptance"
    path, native = repair.settings(candidate["arm"], candidate["profile"], out=root,
                                   risk=candidate["risk"])
    controller = repair.JointBridgeController(candidate["arm"], candidate["horizon"], .25,
        settings=str(path), human_margin=.1, path_overshoot=0., guidance_seed=2026)
    trace=[]
    try:
        controller._reference_path(repair.empty_observation((0.,-4.),(0.,4.),math.pi/2))
        obs=repair.empty_observation(position,(0.,4.),heading,speed)
        if obstacle:
            obs.entities=np.array([[0.,0.,0.,0.,.3]])
            obs.human_state_covariance=np.diag([.0001,.0001,0.,0.])[None]
            obs.human_position_covariance=np.tile(np.eye(2)[None,None]*.0001,(1,candidate["horizon"],1,1))
            obs.acceleration_std=0.
            obs.human_segment_start=np.zeros((1,candidate["horizon"],2))
            obs.human_segment_end=np.zeros((1,candidate["horizon"],2))
        for step in range(100):
            controller.sample_seed=2026000+step
            command, elapsed=controller.act(obs)
            before=obs.robot_xy.copy()
            v0=float(np.linalg.norm(obs.robot_velocity))
            v1=float(np.clip(command[0],max(0.,v0-.5),min(1.,v0+.5)))
            turn=float(np.clip(command[1],-.2,.2))
            obs.robot_xy+=ramp_displacement(obs.robot_heading,v0,v1,turn,.25)
            clearance=None
            if obstacle:
                ts=np.linspace(0.,1.,33)
                curve=before+ramp_displacement(obs.robot_heading,v0,v0+ts*(v1-v0),ts*turn,ts*.25)
                clearance=min(swept_min_clearance(curve[k],curve[k+1],np.zeros((1,2)),
                    np.zeros((1,2)),np.array([.6])) for k in range(32))
                clearance-=math.hypot(abs(v1-v0)/.25,max(v0,v1)*abs(turn)/.25)*(.25/32)**2/8
            obs.robot_heading=(obs.robot_heading+turn)%(2*math.pi)
            obs.robot_velocity=v1*np.array([math.cos(obs.robot_heading),math.sin(obs.robot_heading)])
            distance=float(np.linalg.norm(obs.robot_xy-obs.goal_xy))
            trace.append(dict(xy=obs.robot_xy.tolist(),distance=distance,speed=v1,turn=turn,
                clearance=clearance,solved=controller.last.success,native=copy.deepcopy(controller.last_native),
                plan_ms=elapsed))
            if distance<.3 or (clearance is not None and clearance < -1e-9):
                break
        result=dict(candidate=candidate,case=name,success=distance<.3,
                    collision=any(s["clearance"] is not None and s["clearance"]< -1e-9 for s in trace),
                    steps=trace,nav_time=len(trace)*.25)
        repair.write_result(root/f"{candidate['family']}_{name}.json",result)
        print(candidate["family"],name,"success",result["success"],"collision",result["collision"],
              "time",result["nav_time"],flush=True)
        return result
    finally:
        controller.close()


def main():
    frozen=json.loads((run.OUT/"frozen.json").read_text())
    run.verify_sources(frozen["sources"])
    repair.runtime_environment()
    results=[]
    for c in frozen["selected"]:
        if c["family"]=="bayes":
            continue
        for values in (("aligned",(0.,-4.),math.pi/2,0.,False),
                       ("near_lateral",(.6,3.),math.pi/2,1.,False),
                       ("lateral",(1.,0.),math.pi/2,.5,False),
                       ("away",(0.,0.),-math.pi/2,0.,False),
                       ("stationary_obstacle",(0.,-4.),math.pi/2,0.,True)):
            results.append(case(c,*values))
    run.verify_sources(frozen["sources"])
    repair.write_result(run.OUT/"acceptance.json",results)
    if any(not r["success"] or r["collision"] for r in results):
        raise SystemExit("synthetic acceptance incomplete; inspect traces before main-table release")


if __name__=="__main__":
    main()
