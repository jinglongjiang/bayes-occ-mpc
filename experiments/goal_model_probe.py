"""Offline goal/model factorial gate. No learned predictor or planner changes."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'archive/legacy'))
import continuous_mpc_gate as legacy
from integration.crowdnav import BayesObservationAdapter
from nav.contracts import MPCConfig
from nav.planner import MPCPlanner
from experiments.budget_navigation import history, layout

CROWD = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
OUT = ROOT / 'results/goal_model_probe'
DT = .25
ARMS = ('CV', 'TURN', 'A', 'B', 'C', 'D')
HORIZONS = (4, 8, 12, 16)


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(item):
    env, _, _ = legacy.build_env(CROWD, 'bayes', item['people'], item['scene'],
        circle_radius=4., square_width=10., time_limit=25, occlusion=True)
    env.reset(options={'test_case': item['case']})
    assert env.time_step == DT and not env.robot.visible
    return env


def initialize():
    if (OUT / 'protocol.json').exists():
        return load_protocol()
    print('Historical case/hash audit', flush=True)
    used, hashes, historical = history()
    start = max(2000000, max(used, default=0) + 1000)
    items, seen = [], set()
    for group, (people, scene) in enumerate(((5, 'circle_crossing'),
                                            (10, 'circle_crossing'),
                                            (20, 'square_crossing'))):
        for j in range(10):
            case = start + 10 * group + j
            item = dict(case=case, people=people, scene=scene,
                        split='development' if j < 4 else 'holdout')
            env = environment(item)
            digest = layout(env)
            if case in used or digest in hashes or digest in seen:
                raise RuntimeError('historical/within-run layout collision')
            seen.add(digest)
            item['layout_hash'] = digest
            items.append(item)
    cfg = json.loads((ROOT / 'archive/hermite_audit/protocol.json').read_text())['config']
    paths = list((ROOT / 'nav').glob('*.py')) + [ROOT / 'integration/crowdnav.py',
        Path(__file__), CROWD / 'crowd_sim/envs/crowd_sim.py',
        CROWD / 'crowd_sim/envs/policy/orca.py']
    protocol = dict(baseline='b6cc089', items=items, config=cfg,
        files={str(p): sha(p) for p in paths}, history_files=historical['files'],
        history_unreadable=historical['unreadable'],
        scope='30 collection episodes maximum; offline predictions only; no MPC integration',
        observation='clean legal detections with original occlusion; currently detected targets only',
        queries='every fourth frame; only queries with all 16 future frames; no outcome selection',
        conflict='current constant-velocity robot/target minimum gap over 4s < 0.5m',
        splits='4 development + 6 holdout per configuration; no frame-level split',
        horizons_seconds=[1, 2, 3, 4],
        source_goal='antipode only if first detected at frame zero; otherwise heading/scene intersection',
        old_model='constant estimated speed immediately points at supplied goal; speed from last 8 legal detections',
        improved_model='local ORCA+TTC, own smoothing initialized to observed velocity; legal neighbors CV; bounded angular and velocity response',
        improved_acceleration_grid=[.5, 1., 2.], turn_decay_grid=[.5, 1., 2.],
        selection='C acceleration and TURN decay independently minimize episode/config-balanced dev mean endpoint error over four horizons',
        continuity=dict(max_turn_rate=1.5, max_speed=1.),
        bootstrap='5000 stratified episode-cluster samples; configurations equal weight',
        practical_gate='D versus C >=10% AND >=0.10m mean error reduction in conflict subset averaged over 1..4s; >=2 configs improve; ordinary degradation <=0.05m; screening only',
        missing='no imputed position treated as evidence; neighbor extrapolation uses current legal detections only',
        truth='separate goals/positions for scoring; only B/D receive target goal; no hidden neighbors or internal simulator state')
    save('protocol.json', protocol)
    return protocol


def load_protocol():
    p = json.loads((OUT / 'protocol.json').read_text())
    for path, digest in p['files'].items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: ' + path)
    if 'data_sha256' in p and sha(OUT/'episodes.jsonl') != p['data_sha256']:
        raise RuntimeError('frozen observation/label data changed')
    return p


def collect(protocol):
    existing = {}
    path = OUT / 'episodes.jsonl'
    if path.exists():
        existing = {r['case']: r for r in map(json.loads, path.read_text().splitlines())}
    _, _, ActionXY, _, _ = legacy._load_modules(CROWD)
    for item in protocol['items']:
        if item['case'] in existing:
            continue
        env = environment(item)
        assert layout(env) == item['layout_hash']
        cfg = MPCConfig(**protocol['config'])
        adapter = BayesObservationAdapter(cfg)
        planner = MPCPlanner(cfg)
        record = dict(item, frames=[], truth=dict(goals=[[h.gx, h.gy] for h in env.humans],
            positions=[]), outcomes=None)
        done, step = False, 0
        started = time.perf_counter()
        while True:
            record['truth']['positions'].append([[h.px, h.py] for h in env.humans])
            if done:
                break
            obs = adapter.read(env)
            detected = [dict(x) for x in adapter.detected_entities]
            action, _ = planner.plan(obs, item['case'] * 100003 + step * 97 + 1729)
            record['frames'].append(dict(step=step, timestamp=float(env.global_time),
                detections=detected,
                robot=[env.robot.px, env.robot.py, env.robot.vx, env.robot.vy, env.robot.radius],
                action=np.asarray(action).tolist()))
            before = np.array(env.robot.get_position())
            _, _, term, trunc, info = env.step(ActionXY(*map(float, action)))
            if not np.allclose(env.robot.get_position(), before + DT * action, atol=1e-12, rtol=0):
                raise RuntimeError('execution mismatch')
            done, step = term or trunc, step + 1
            if step > 110:
                raise RuntimeError('nonterminating environment')
            record['outcomes'] = info['event']
        # Truth is serialized separately from the legal frame payload.
        with path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print('COLLECT', item['people'], item['case'], item['split'], step,
              record['outcomes'], round(time.perf_counter() - started, 2), flush=True)


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-10 else np.zeros(2)


def legal_features(item, frame, entity, past):
    pos = np.array([entity['px'], entity['py']], dtype=np.float64)
    vel = np.array([entity['vx'], entity['vy']], dtype=np.float64)
    observations = past[-8:]
    # Match the vendored old tracker's public speed prior/EMA and gap discipline.
    speed = 1.
    for previous, current in zip(past[:-1], past[1:]):
        if current[0] - previous[0] == 1:
            observed = np.linalg.norm(current[1] - previous[1]) / DT
            speed = float(np.clip(.7 * speed + .3 * observed, .2, 2.5))
    direction = unit(np.mean([v for _, _, v in observations], axis=0))
    first_step, first_pos, _ = past[0]
    if item['scene'] == 'circle_crossing' and first_step == 0:
        goal, source = -first_pos, 'birth_antipode'
    else:
        if np.linalg.norm(direction) < 1e-8:
            direction = unit(-pos)
        if item['scene'] == 'circle_crossing':
            b = float(pos @ direction)
            t = max(0., -b + math.sqrt(max(0., b*b + 16. - float(pos @ pos))))
            goal = pos + max(t, 1.) * direction
        else:
            x = -first_pos[0]
            t = (x - pos[0]) / direction[0] if abs(direction[0]) > .05 else -1.
            y = pos[1] + t * direction[1] if t > 0 else pos[1]
            goal = np.array([x, np.clip(y, -5., 5.)])
        source = 'heading_geometry'
    omega = 0.
    if len(past) >= 2 and past[-1][0] - past[-2][0] == 1:
        previous = past[-2][2]
        if min(np.linalg.norm(previous), np.linalg.norm(vel)) >= .1:
            omega = float(np.clip(math.atan2(np.cross(previous, vel), previous @ vel) / DT, -1.5, 1.5))
    robot = np.array(frame['robot'])
    relative, velocity = pos - robot[:2], vel - robot[2:4]
    t = np.clip(-relative @ velocity / max(velocity @ velocity, 1e-12), 0, 4)
    conflict = np.linalg.norm(relative + t * velocity) - entity['radius'] - robot[4] < .5
    neighbors = [dict(e) for e in frame['detections'] if e['id'] != entity['id']]
    return dict(pos=pos, vel=vel, radius=entity['radius'], speed=speed,
                goal=goal, goal_source=source, omega=omega, conflict=bool(conflict), neighbors=neighbors)


def predict(features, goal, mode, parameter=1.):
    pos = np.asarray(features['pos'], dtype=np.float64).copy()
    velocity = np.asarray(features['vel'], dtype=np.float64).copy()
    if mode == 'improved':
        from crowd_sim.envs.policy.orca import ORCA
        from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
        policy = ORCA()
        policy._last_pref_vel = velocity.copy()
    result = []
    omega = features['omega']
    for k in range(16):
        if mode == 'old':
            velocity = (features['speed'] * unit(goal - pos)
                        if np.linalg.norm(goal - pos) > .35 else np.zeros(2))
        elif mode == 'turn':
            angle = omega * DT
            c, s = math.cos(angle), math.sin(angle)
            velocity = np.array([c*velocity[0]-s*velocity[1], s*velocity[0]+c*velocity[1]])
            omega *= math.exp(-DT / parameter)
        elif mode == 'improved':
            people = [ObservableState(e['px']+k*DT*e['vx'], e['py']+k*DT*e['vy'],
                                      e['vx'], e['vy'], e['radius']) for e in features['neighbors']]
            state = JointState(FullState(*pos, *velocity, features['radius'], *goal, 1., 0.), people)
            desired = np.array(policy.predict(state))
            if np.linalg.norm(velocity) >= .1 and np.linalg.norm(desired) >= .1:
                angle = math.atan2(np.cross(velocity, desired), velocity @ desired)
                angle = np.clip(angle, -1.5*DT, 1.5*DT)
                c, s = math.cos(angle), math.sin(angle)
                desired = np.linalg.norm(desired) * np.array([[c, -s], [s, c]]) @ unit(velocity)
            delta = desired - velocity
            velocity += delta * min(1., parameter*DT/max(np.linalg.norm(delta), 1e-12))
        pos = pos + DT * velocity
        result.append(pos.copy())
    return np.array(result)


def queries(record):
    past = {}
    last = len(record['truth']['positions']) - 1
    for frame in record['frames']:
        step = frame['step']
        for e in frame['detections']:
            tid = e['id']
            past.setdefault(tid, []).append((step, np.array([e['px'], e['py']]), np.array([e['vx'], e['vy']])))
        if step % 4 or step + 16 > last:
            continue
        for e in frame['detections']:
            yield step, e['id'], legal_features(record, frame, e, past[e['id']])


def evaluate(record, acceleration, decay):
    rows = []
    for step, tid, features in queries(record):
        truth_goal = np.asarray(record['truth']['goals'][tid])
        simple = features['goal']
        predictions = dict(CV=predict(features, simple, 'cv'),
            TURN=predict(features, simple, 'turn', decay),
            A=predict(features, simple, 'old'), B=predict(features, truth_goal, 'old'),
            C=predict(features, simple, 'improved', acceleration),
            D=predict(features, truth_goal, 'improved', acceleration))
        error = {arm: [float(np.linalg.norm(pred[h-1] - record['truth']['positions'][step+h][tid]))
                       for h in HORIZONS] for arm, pred in predictions.items()}
        rows.append(dict(case=record['case'], people=record['people'], step=step, track=tid,
            conflict=features['conflict'], errors=error, goal_source=features['goal_source'],
            goal_error=float(np.linalg.norm(simple - truth_goal)),
            neighbor_count=len(features['neighbors'])))
    return rows


def mean_error(rows, arm):
    return float(np.mean([r['errors'][arm] for r in rows]))


def run(protocol):
    legacy._load_modules(CROWD)
    records = list(map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines()))
    assert len(records) == 30 and len({r['case'] for r in records}) == 30
    selected_file = OUT / 'selected.json'
    if not selected_file.exists():
        dev = [r for r in records if r['split'] == 'development']
        scores = []
        for parameter in protocol['improved_acceleration_grid']:
            c, turn = [], []
            for record in dev:
                rows = evaluate(record, parameter, parameter)
                if not rows:
                    raise RuntimeError('empty development episode')
                c.append(mean_error(rows, 'C'))
                turn.append(mean_error(rows, 'TURN'))
            scores.append(dict(parameter=parameter, C=float(np.mean(c)), TURN=float(np.mean(turn))))
            print('DEVELOPMENT', scores[-1], flush=True)
        selected = dict(acceleration=min(scores, key=lambda s: (s['C'], s['parameter']))['parameter'],
                        turn_decay=min(scores, key=lambda s: (s['TURN'], s['parameter']))['parameter'], scores=scores)
        save('selected.json', selected)
    selected = json.loads(selected_file.read_text())
    rows = []
    for record in records:
        if record['split'] != 'holdout':
            continue
        r = evaluate(record, selected['acceleration'], selected['turn_decay'])
        rows.extend(r)
        print('HOLDOUT', record['people'], record['case'], len(r), flush=True)
    save('predictions.json', rows)
    summarize(protocol, records, rows, selected)


def interval(values, groups):
    values = np.asarray(values)
    rng = np.random.default_rng(20260912)
    boot = []
    indices = [np.flatnonzero(np.asarray(groups) == n) for n in sorted(set(groups))]
    for _ in range(5000):
        boot.append(np.mean([values[rng.choice(ids, len(ids), replace=True)].mean(axis=0) for ids in indices], axis=0))
    return np.quantile(boot, [.025, .975], axis=0).tolist()


def summarize(protocol, records, rows, selected):
    tables, contrasts, counts = [], [], []
    for subset in ('all', 'ordinary', 'conflict'):
        per_episode, groups = [], []
        for record in records:
            if record['split'] != 'holdout':
                continue
            q = [r for r in rows if r['case'] == record['case'] and
                 (subset == 'all' or r['conflict'] == (subset == 'conflict'))]
            counts.append(dict(subset=subset, case=record['case'], queries=len(q)))
            if q:
                per_episode.append([[np.mean([r['errors'][a][h] for r in q]) for h in range(4)] for a in ARMS])
                groups.append(record['people'])
        if len(set(groups)) != 3:
            raise RuntimeError('missing configuration for subset ' + subset)
        values = np.array(per_episode)
        balanced = np.mean([values[np.asarray(groups)==n].mean(axis=0) for n in (5,10,20)], axis=0)
        ci = interval(values, groups)
        for a, arm in enumerate(ARMS):
            tables.append(dict(subset=subset, arm=arm, endpoint_m=balanced[a].tolist(),
                               ci_low=ci[0][a], ci_high=ci[1][a], episodes=len(groups)))
        for a,b in (('C','A'),('D','C'),('B','A'),('C','CV'),('D','CV'),('C','TURN'),('D','TURN')):
            delta = values[:,ARMS.index(a)]-values[:,ARMS.index(b)]
            means = {str(n):delta[np.asarray(groups)==n].mean(axis=0).tolist() for n in (5,10,20)}
            contrasts.append(dict(subset=subset, contrast=a+'-'+b,
                delta_m=np.mean(list(means.values()),axis=0).tolist(),
                ci=interval(delta,groups), configuration_delta=means,
                episodes_improved=int(np.sum(delta.mean(axis=1)<0)), episodes_worse=int(np.sum(delta.mean(axis=1)>0))))
    target = next(r for r in contrasts if r['subset']=='conflict' and r['contrast']=='D-C')
    ordinary = next(r for r in contrasts if r['subset']=='ordinary' and r['contrast']=='D-C')
    c = next(r for r in tables if r['subset']=='conflict' and r['arm']=='C')
    gain = -float(np.mean(target['delta_m']))
    passed = (gain >= .1 and gain >= .1*np.mean(c['endpoint_m']) and
        sum(np.mean(v)<0 for v in target['configuration_delta'].values())>=2 and
        np.mean(ordinary['delta_m'])<=.05)
    coverage = dict(queries=len(rows), birth_antipode=sum(r['goal_source']=='birth_antipode' for r in rows),
        goal_error_m=float(np.mean([r['goal_error'] for r in rows])),
        missing_neighbors_mean=float(np.mean([r['people']-1-r['neighbor_count'] for r in rows])))
    save('summary.json', dict(selected=selected, table=tables, contrasts=contrasts,
        counts=counts, coverage=coverage, target_information_gate=bool(passed)))
    lines=['# Goal / Motion Model Factorial Gate', '',
        '30 collection episodes, 12 development / 18 holdout. No training or new navigation arm.',
        'All predictions are deterministic endpoint forecasts, not posterior calibration.',
        'Configurations equally weighted; intervals use stratified episode bootstrap.', '',
        '| Subset | Arm | 1s error m | 2s | 3s | 4s |', '|---|---|---:|---:|---:|---:|']
    for r in tables:
        lines.append('| '+r['subset']+' | '+r['arm']+' | '+' | '.join(f'{v:.3f}' for v in r['endpoint_m'])+' |')
    lines += ['', '| Subset | Contrast (negative better) | 1s | 2s | 3s | 4s | Improved/worse episodes |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in contrasts:
        lines.append('| '+r['subset']+' | '+r['contrast']+' | '+' | '.join(f'{v:.3f}' for v in r['delta_m'])+
                     f" | {r['episodes_improved']}/{r['episodes_worse']} |")
    lines += ['', 'Target-information resource gate: '+('PASS' if passed else 'STOP'),
        'This does not authorize posterior training or establish closed-loop benefit.',
        'No comparison has access to hidden neighbor truth. D/B alone receive the target goal.',
        'Stopping at episode termination censors the last 4s. Only currently detected targets are queried.',
        'Full confidence intervals, configuration differences and sample counts: summary.json.',
        'The improved model is one bounded-response ORCA/TTC approximation, not a certified human model.']
    (OUT/'verdict.md').write_text('\n'.join(lines)+'\n')


def selftest():
    legacy._load_modules(CROWD)
    f=dict(pos=np.zeros(2), vel=np.array([.5,0.]), speed=.5, omega=0.,radius=.3,neighbors=[])
    g=np.array([10.,0.])
    assert np.array_equal(predict(f,g,'old'),predict(f,g,'cv'))
    assert np.array_equal(predict(f,g,'turn'),predict(f,g,'cv'))
    a=predict(f,g,'improved',1.)
    velocities=np.diff(np.vstack([f['pos'],a]),axis=0)/DT
    assert np.max(np.linalg.norm(np.diff(np.vstack([f['vel'],velocities]),axis=0),axis=1))<=DT+1e-10
    assert np.array_equal(a,predict(f,g,'improved',1.))
    f2=dict(f, neighbors=[dict(id=7,px=2.,py=0.,vx=-.5,vy=0.,radius=.3)])
    assert np.isfinite(predict(f2,g,'improved',1.)).all()
    assert np.isfinite(predict(dict(f, vel=np.array([0,0])),g,'improved',1.)).all()
    print('SELFTEST_PASS', flush=True)


def audit():
    """Read-only prediction checks and descriptive decompositions; no reselection."""
    records = list(map(json.loads, (OUT/'episodes.jsonl').read_text().splitlines()))
    rows = json.loads((OUT/'predictions.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text())
    means = []
    for subset in ('all', 'ordinary', 'conflict'):
        for a,b in (('C','A'),('D','C'),('B','A'),('C','CV'),('D','CV'),('D','TURN')):
            values, groups = [], []
            for rec in records:
                q=[r for r in rows if r['case']==rec['case'] and
                   (subset=='all' or r['conflict']==(subset=='conflict'))]
                if q:
                    values.append(float(np.mean([np.mean(r['errors'][a])-np.mean(r['errors'][b]) for r in q])))
                    groups.append(rec['people'])
            balanced=float(np.mean([np.mean(np.asarray(values)[np.array(groups)==n]) for n in (5,10,20)]))
            means.append(dict(subset=subset,contrast=a+'-'+b,delta=balanced,ci=interval(values,groups)))
    coverage, censor = {}, []
    for n in (5,10,20):
        anchors=(np.array([[4*np.cos(2*np.pi*k/8),4*np.sin(2*np.pi*k/8)] for k in range(8)])
                 if n!=20 else np.array([[x,y] for x in (-2.5,2.5) for y in (-3.75,-1.25,1.25,3.75)]))
        errors=[]
        for rec in records:
            if rec['people']==n and rec['split']=='holdout':
                goals=np.array(rec['truth']['goals'])
                errors.extend(np.min(np.linalg.norm(goals[:,None]-anchors[None],axis=2),axis=1).tolist())
        coverage[str(n)]=dict(mean_nearest_anchor_m=float(np.mean(errors)),
            p95=float(np.quantile(errors,.95)),exact_matches=int(np.sum(np.array(errors)<1e-9)),targets=len(errors))
    record_map={r['case']:r for r in records}
    for rec in records:
        if rec['split']=='holdout':
            total=sum(len(f['detections']) for f in rec['frames'] if f['step']%4==0)
            used=sum(r['case']==rec['case'] for r in rows)
            censor.append(dict(case=rec['case'],potential_queries=total,used=used,tail_censored=total-used))
    born=[r for r in rows if r['goal_source']=='birth_antipode']
    assert all(r['goal_error']<1e-12 and r['errors']['C']==r['errors']['D'] for r in born)
    for row in rows:
        rec=record_map[row['case']]; t=row['track']; s=row['step']
        future=np.asarray(rec['truth']['positions'])[s:s+17,t]
        row['near_goal_window']=bool(np.min(np.linalg.norm(future-rec['truth']['goals'][t],axis=1))<.5)
    arrival=[]
    for subset in ('all','conflict'):
        for near in (False,True):
            q=[r for r in rows if r['near_goal_window']==near and (subset=='all' or r['conflict'])]
            values,groups,errors=[],[],[]
            for case,rec in record_map.items():
                selected=[r for r in q if r['case']==case]
                if selected:
                    errors.append([np.mean([r['errors'][arm] for r in selected],axis=0) for arm in ('CV','C','D')])
                    values.append(float(np.mean([np.mean(r['errors']['D'])-np.mean(r['errors']['C']) for r in selected])))
                    groups.append(rec['people'])
            e=np.array(errors)
            table=np.mean([e[np.array(groups)==n].mean(axis=0) for n in sorted(set(groups))],axis=0)
            arrival.append(dict(subset=subset,near_goal_window=near,queries=len(q),episodes=len(values),
                configurations=sorted(set(groups)),CV_C_D=table.tolist(),D_minus_C_CI=interval(values,groups),
                note='Post-hoc truth-based descriptive label; not a new gate, main subset, or deployable selector'))
    summary.update(mean_horizon_contrasts=means,goal_dictionary_coverage=coverage,censoring=censor,
        posthoc_goal_arrival_check=arrival,
        checks=dict(birth_antipode_exact_queries=len(born),holdout_queries=len(rows),
                    conflict_queries=sum(r['conflict'] for r in rows),episodes_sha256=sha(OUT/'episodes.jsonl'),
                    predictions_sha256=sha(OUT/'predictions.json')))
    save('summary.json',summary)
    print('AUDIT_PASS',summary['checks'],flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('command', choices=['selftest','init','collect','evaluate','audit'])
    args=ap.parse_args()
    if args.command=='selftest':
        selftest()
    elif args.command=='init':
        initialize()
    elif args.command=='collect':
        collect(load_protocol())
    elif args.command=='audit':
        load_protocol()
        audit()
    else:
        run(load_protocol())
