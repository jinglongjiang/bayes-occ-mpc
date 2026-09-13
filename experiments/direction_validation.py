"""Local association diagnostics, not a recursive tracker or navigation trial."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'archive/legacy')]
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from experiments.occlusion_confirmation import OUT, verify, legacy, matched, build, CROWD, layout
from experiments.occlusion_closeout import save
from experiments.occlusion_shape_probe import circle, selection

DEST = ROOT / 'results/direction_validation'


def two_assignments(cost):
    """Exact two best injective assignments, conditional on the supplied detections."""
    rows, cols = linear_sum_assignment(cost)
    if len(rows) != len(cost) or np.any(cost[rows, cols] >= 1e90):
        return None
    best = float(cost[rows, cols].sum())
    second, alt = np.inf, None
    for row, col in zip(rows, cols):
        changed = cost.copy()
        changed[row, col] = 1e100
        r, c = linear_sum_assignment(changed)
        value = float(changed[r, c].sum())
        if value < second and np.all(changed[r, c] < 1e90):
            second, alt = value, c
    return cols, alt, best, second


def synthetic():
    rng = np.random.default_rng(14092026)
    checks = 0
    for n in range(2, 6):
        for m in range(1, n + 1):
            cost = rng.normal(size=(m, n))
            values = sorted(sum(cost[j, p[j]] for j in range(m))
                            for p in itertools.permutations(range(n), m))
            result = two_assignments(cost)
            assert np.allclose(result[2:], values[:2])
            checks += 1
    xor = np.array([[1., 0.], [0., 1.]])
    positive = np.array([[1., 1.], [0., 0.]])
    def risks(q):
        return dict(joint=float(np.mean(1-np.prod(1-q, axis=1))),
                    independent=float(1-np.prod(1-q.mean(axis=0))),
                    map=float(1-np.prod(1-q[0])))
    assert risks(xor) == dict(joint=1., independent=.75, map=1.)
    assert risks(positive) == dict(joint=.5, independent=.75, map=1.)
    permutation_error = 0.
    for _ in range(100):
        q = rng.random((10, 32))
        a = 1-np.prod(1-q, axis=0)
        b = 1-np.prod(1-q[rng.permutation(10)], axis=0)
        permutation_error = max(permutation_error, float(abs(a-b).max()))
    assert permutation_error < 1e-14
    # Same posterior mean of sensor health, different shared-event probability.
    health = np.array([.1, .9])
    shared_miss = float(np.mean((1-health)**2))
    mean_miss = float((1-health.mean())**2)
    assert np.isclose(shared_miss, .41) and np.isclose(mean_miss, .25)
    # A different example where MAP really does change the preferred risk.
    worlds = np.array([[0., .3], [1., .3]])
    full = np.array([.6, .4]) @ worlds
    assert worlds[0].argmin() == 0 and full.argmin() == 1
    return dict(assignment_bruteforce_cases=checks, xor=risks(xor),
        positive_correlation=risks(positive), permutation_tests=100,
        max_permutation_error=permutation_error,
        sensor_health=dict(shared_two_misses=shared_miss, mean_plugin=mean_miss),
        genuine_map_counterexample=dict(weights=[.6,.4], conditional_risks=worlds.tolist(),
                                         mixture_risks=full.tolist()),
        meaning='Algebra and solver checks only. No measured navigation benefit or novelty.')


def updated(priors, measurements, assignment, position_only):
    states = [(m.copy(), p.copy(), radius) for m, p, radius in priors]
    for j, i in enumerate(assignment):
        mean, cov, radius = states[i]
        z = measurements[j]
        if position_only:
            gain = np.linalg.solve(cov[:2,:2], cov[:2,:]).T
            mean = mean + gain @ (z[:2]-mean[:2])
            transform = np.eye(4)
            transform[:,:2] -= gain
            cov = transform @ cov @ transform.T
            cov = (cov + cov.T)/2
        else:
            mean = z[:4].copy()
            cov = np.zeros((4,4))
        states[i] = mean, cov, float(z[4])
    return states


def component_probabilities(states, positions, obs, planner, F, Q):
    result = np.empty((len(states), len(positions), planner.cfg.horizon))
    for i, (mean, cov, radius) in enumerate(states):
        mean, cov = mean.copy(), cov.copy()
        for k in range(planner.cfg.horizon):
            mean = F @ mean
            cov = F @ cov @ F.T + Q
            v = float(np.trace(cov[:2,:2])/2)
            if not np.allclose(cov[:2,:2], v*np.eye(2), atol=1e-10, rtol=0):
                raise RuntimeError('Anisotropic conditional state requires a different integrator')
            result[i,:,k] = circle(np.linalg.norm(positions[:,k]-mean[:2],axis=1),
                                  max(v,0.), obs.robot_radius+radius+planner.cfg.human_margin)
    return result


def candidate_pool(obs, cfg, seed):
    planner = legacy.ContinuousCEMMPC(cfg)
    rng = np.random.default_rng(seed)
    bases = np.concatenate((planner._initial_mean(obs)[None], planner._route_seeds(obs)))
    noise = rng.standard_normal((cfg.population, cfg.horizon, 2))
    noise[:,1:] = .68*noise[:,:-1] + .32*noise[:,1:]
    raw = bases[np.arange(cfg.population)%len(bases)] + cfg.init_std*noise
    seeds = planner._seed_trajectories(obs)
    raw[:len(seeds)] = seeds
    _, controls, positions = planner._rollout(raw, obs)
    return planner, controls, positions


def diagnostic(prior_tracks, detections, obs, F, Q, cfg, seed):
    # Truth IDs only define the known-target diagnostic population and score it.
    # The assignment function receives numeric measurements, never identifiers.
    ids = list(prior_tracks)
    known = [e for e in detections if int(e['id']) in prior_tracks]
    known.sort(key=lambda e:(e['px'],e['py'],e['vx'],e['vy']))
    entry = dict(priors=len(ids), known_detections=len(known), ambiguity=False)
    if len(ids)<2 or not known:
        return entry
    states = [(F@t.mean, F@t.covariance@F.T+Q, t.radius) for t in prior_tracks.values()]
    z = np.array([[e[k] for k in ('px','py','vx','vy','radius')] for e in known])
    cost = np.empty((len(z),len(ids)))
    for i,(m,p,_) in enumerate(states):
        residual = z[:,:2]-m[:2]
        mahal = np.einsum('ji,ij->j', residual, np.linalg.solve(p[:2,:2],residual.T))
        cost[:,i] = .5*(mahal + np.linalg.slogdet(p[:2,:2])[1] + 2*np.log(2*np.pi))
        cost[mahal>9.21034037197618,i] = 1e100
    answer = two_assignments(cost)
    if answer is None:
        entry['status']='no_feasible_global_assignment_under_gate'
        return entry
    a,b,ca,cb = answer
    correct = [ids[i]==int(e['id']) for i,e in zip(a,known)]
    entry.update(status='valid', map_wrong_links=int(len(correct)-sum(correct)),
                 scored_links=len(correct), second_weight=float(expit(ca-cb)))
    if b is None or entry['second_weight'] < .2:
        return entry
    entry['ambiguity'] = True
    entry['selected_track_set_changes'] = set(a)!=set(b)
    entry['assigned_prior_had_miss'] = any(prior_tracks[ids[i]].missed_steps>0 for i in set(a)|set(b))
    weight = entry['second_weight']
    # Keep genuinely new detections as identical background in every world.
    background=[(np.array([e[k] for k in ('px','py','vx','vy')]),np.zeros((4,4)),e['radius'])
                for e in detections if int(e['id']) not in prior_tracks]
    planner,controls,positions = candidate_pool(obs,cfg,seed)
    entry['arms'] = {}
    for position_only in (False,True):
        worlds=[updated(states,z,c,position_only)+background for c in (a,b)]
        q=[component_probabilities(s,positions,obs,planner,F,Q) for s in worlds]
        risk=[-np.expm1(np.log1p(-np.clip(x,0,1-1e-12)).sum(axis=0)) for x in q]
        mixture=(1-weight)*risk[0]+weight*risk[1]
        independent=-np.expm1(np.log1p(-np.clip((1-weight)*q[0]+weight*q[1],0,1-1e-12)).sum(axis=0))
        choices={name:selection(planner,controls,positions,obs,-np.log1p(-np.clip(r,0,1-1e-12)))
                 for name,r in [('map',risk[0]),('second',risk[1]),('full',mixture),('independent',independent)]}
        risks=[x.max(axis=1) for x in risk]
        first=risks[0][:,None]-risks[0][None,:]
        second=risks[1][:,None]-risks[1][None,:]
        reversal=bool(np.any(((first>=.01)&(second<=-.01))|((first<=-.01)&(second>=.01))))
        delta=float(np.linalg.norm(np.array(choices['full']['action'])-choices['map']['action']))
        entry['arms']['position_only_update' if position_only else 'native_exact_state_update'] = dict(
            hypothesis_max_risk_difference=float(abs(risk[0]-risk[1]).max()),
            independent_max_risk_error=float(abs(mixture-independent).max()),
            ordering_reversal_over_1pp=reversal, full_map_action_delta=delta,
            feasible_full_map_change=delta>=.05 and choices['full']['level']==0,
            choices=choices, candidates=len(controls))
    return entry


def audit(row):
    p=json.loads((OUT/'confirmation_protocol.json').read_text())
    cfg=legacy.MPCConfig(**p['configs']['bayes'])
    calibration=json.loads((OUT/'calibration.json').read_text())
    _,_,ActionXY,_,_=legacy._load_modules(CROWD)
    env=build(row['scene'],row['case_id'])
    if layout(env)!=p['layout_hashes'][f"{row['scene']}:{row['case_id']}"]:
        raise RuntimeError('Layout mismatch')
    adapter=matched.MatchedAdapter('bayes',cfg.horizon,cfg.human_margin,cfg.dt,
        cfg.chance_limit,cfg.fixed_uncertainty_radius,cfg.acceleration_std,
        method='bayes',calibration=calibration,point=p['points']['bayes'])
    records=[]
    for step,record in enumerate(row['steps']):
        priors={i:copy.deepcopy(t) for i,t in adapter.rfs.tracks.items()
                if t.visible or t.existence>=adapter.rfs.cfg.report_existence}
        obs=adapter.read(env)
        F,Q=adapter.rfs._transition()
        item=diagnostic(priors,adapter.detected_entities,obs,F,Q,cfg,
                        row['case_id']*100003+step*97+1729)
        item['step']=step
        records.append(item)
        env.step(ActionXY(record['action_a'],record['action_b']))
        if not np.allclose([env.robot.px,env.robot.py],[record['x'],record['y']],atol=1e-9,rtol=0):
            raise RuntimeError('Recorded-action state replay mismatch')
    result=dict(scene=row['scene'],case=row['case_id'],states=records)
    save(DEST/f"{row['scene']}_{row['case_id']}.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    p=verify()
    DEST.mkdir(parents=True,exist_ok=True)
    tests=synthetic()
    save(DEST/'selftests.json',tests)
    cases=p['cases'][:10]
    protocol=dict(scope='Exploratory conditional one-frame association diagnostic, not an anonymous recursive tracker',
        population='First 10 frozen cases in each of six configurations, all steps; no outcome selection',
        cases=cases, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((OUT/'episodes.jsonl').read_bytes()).hexdigest(),
        privileged_condition='Prior tracks come from existing identity-associated legal history. Current ID only partitions known/new targets and scores assignments, not likelihood or update.',
        association='Exact top two injective assignments from predicted position Gaussian density; 99% 2D gate; equal assignment prior; conditional on known target detections; no clutter or birth ambiguity.',
        ambiguity='Second normalized weight within TOP TWO >= .2, not mass in full posterior',
        risk='Original per-step disc event, original Q/radius/horizon; all included persons exist with probability one to isolate association, same across worlds; across-person conditional independence.',
        comparisons=['native exact position and velocity update','position-only current update with same privileged past prior'],
        controls='Same 512 initial candidate pool and original common geometry/cost; no independent CEM or new navigation outcomes',
        relevance='Two conditional max-over-horizon candidate risk orderings reverse with >=.01 difference in both; also report full-vs-MAP first speed delta >=.05 and full feasibility.',
        stop='Complete fixed 60 replays regardless of sign; fail on state mismatch; no likelihood tuning on results')
    path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise RuntimeError('Frozen diagnostic protocol changed')
    save(path,protocol)
    rows=[]
    with (OUT/'episodes.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['method']=='bayes' and r['sensor']=='range_and_occlusion' and r['case_id'] in cases:
                rows.append(r)
    assert len(rows)==60
    pending=[r for r in rows if not (DEST/f"{r['scene']}_{r['case_id']}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(audit,pending):
            print('Completed',result['scene'],result['case'],'ambiguous',sum(s['ambiguity'] for s in result['states']),flush=True)
    rows=[json.loads((DEST/f"{r['scene']}_{r['case_id']}.json").read_text()) for r in rows]
    output=[]
    for scene in [None]+list(range(6)):
        states=[s for r in rows if scene is None or r['scene']==scene for s in r['states']]
        ambiguous=[s for s in states if s['ambiguity']]
        record=dict(scene=scene,states=len(states),ambiguous_states=len(ambiguous),
            valid_assignment_states=sum(s.get('status')=='valid' for s in states),
            no_feasible_assignment_states=sum(s.get('status')=='no_feasible_global_assignment_under_gate' for s in states),
            map_wrong_links=sum(s.get('map_wrong_links',0) for s in states),scored_links=sum(s.get('scored_links',0) for s in states),
            changed_track_set_states=sum(s['selected_track_set_changes'] for s in ambiguous),arms={})
        for arm in ('native_exact_state_update','position_only_update'):
            values=[s['arms'][arm] for s in ambiguous]
            record['arms'][arm]=dict(order_reversals=sum(v['ordering_reversal_over_1pp'] for v in values),
                feasible_full_map_changes=sum(v['feasible_full_map_change'] for v in values),
                max_action_delta=max((v['full_map_action_delta'] for v in values),default=0),
                max_hypothesis_risk_difference=max((v['hypothesis_max_risk_difference'] for v in values),default=0),
                max_independence_error=max((v['independent_max_risk_error'] for v in values),default=0))
        output.append(record)
    save(DEST/'summary.json',dict(episodes=60,rows=output,limitation=protocol['scope']))
    print(json.dumps(output[0],indent=2),flush=True)


if __name__=='__main__':
    main()
