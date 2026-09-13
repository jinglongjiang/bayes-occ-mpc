"""Independent closed-loop confirmation; no production algorithm changes."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import pickle
import subprocess
import sys
import traceback

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'archive/legacy')]
from experiments import occlusion_closeout as prep
import continuous_mpc_gate as legacy
import evaluate_matched_safety as matched
from reproducibility.runtime import CROWD

OUT = prep.OUT
ARMS = ('posterior_mean', 'covariance', 'bayes', 'age_margin', 'ewma', 'conformal')


def layout(env):
    import hashlib
    rows = [[a.px, a.py, a.gx, a.gy, a.v_pref, a.radius] for a in [env.robot] + env.humans]
    return hashlib.sha256(np.asarray(rows, dtype=np.float64).round(9).tobytes()).hexdigest()


def build(scene, case):
    _, count, scenario, radius, width = matched.SCENES[scene]
    env, _, _ = legacy.build_env(CROWD, 'bayes', count, scenario, radius, width, 25, True)
    env.reset(options={'test_case': case})
    return env


def register():
    prep.verify()
    path = OUT / 'confirmation_protocol.json'
    if path.exists():
        verify()
        return
    development = json.loads((OUT / 'development_summary.json').read_text())
    if not development['complete']:
        raise RuntimeError('Development not finished')
    from experiments.budget_navigation import history
    used, _, history_audit = history()
    if history_audit['unreadable']:
        raise RuntimeError('Unreadable historical records; resolve before claiming new layouts')
    start = max(3000000, max(used, default=0) + 1000)
    if start + 100 > 2**32 - 10000:
        raise RuntimeError('Seed range invalid')
    cases = list(range(start, start + 100))
    assert not set(cases).intersection(used)
    hashes = {}
    for scene in range(6):
        env = build(scene, cases[0])
        old = set()
        for case in sorted(used):
            env.reset(options={'test_case': case})
            old.add(layout(env))
        for case in cases:
            env.reset(options={'test_case': case})
            h = layout(env)
            if h in old or h in hashes.values():
                raise RuntimeError('Layout hash overlap')
            hashes[f'{scene}:{case}'] = h
    base = json.loads((OUT / 'protocol.json').read_text())
    points = base['historical_points']
    configs = {a: asdict(matched.config(points.get(a, points['bayes']))) for a in ARMS}
    configs['covariance']['existence_override'] = 1.
    sources = dict(base['files'])
    for p in (Path(__file__), OUT / 'development_summary.json',
              *sorted(Path('/home/abc/temp/devsweep/clean').glob('clean_*_p*.json'))):
        sources[str(p)] = prep.digest(p)
    protocol = dict(stage='independent confirmation', baseline_commit=base['baseline_commit'],
        cases=cases, visibility_cases=cases[:20], scenes=matched.SCENES, arms=ARMS,
        configs=configs, age_target=development['selected_age_target'], points=points,
        layout_hashes=hashes, history=history_audit, files=sources,
        expected_episodes=4080, primary='bayes - posterior_mean audited collision',
        statistical_unit='case/seed block containing all six scenes', bootstrap_repeats=10000,
        scope='clean original holonomic protocol; no real-time claim from concurrent timings',
        opportunity='descriptive current hidden-track Gaussian quadrature: observed-free and unknown masses each >= .01; not a novelty gate',
        sensors='range_only removes shadows, preserves grid/FOV/occupancy conventions; full exposes current detections only',
        stopping='run all registered tasks; stop only on execution/protocol error; no outcome-dependent expansion')
    prep.save(path, protocol)
    print('Registered 600 new layouts and 4080 episodes', flush=True)


def verify():
    p = json.loads((OUT / 'confirmation_protocol.json').read_text())
    for source, h in p['files'].items():
        if prep.digest(source) != h:
            raise RuntimeError('Frozen confirmation source changed: ' + source)
    return p


def range_sensor(self, robot_xy, humans):
    mx, my = self._build_mesh(robot_xy)
    occ = np.zeros(mx.shape, np.float32)
    ids = np.full(mx.shape, -1, np.int32)
    for h in humans:
        mask = (mx-h['px'])**2 + (my-h['py'])**2 <= h['radius']**2
        occ[mask], ids[mask] = 1., h['id']
    sensor = np.zeros(mx.shape, np.float32)
    sensor[(mx-robot_xy[0])**2 + (my-robot_xy[1])**2 > self.fov_radius**2] = .5
    visible, hidden = [], []
    for h in humans:
        mask = ids == h['id']
        if not mask.any() or np.all(sensor[mask] == .5):
            hidden.append(h['id'])
        else:
            sensor[mask] = 1.
            visible.append(h['id'])
    self._mesh = (mx, my)
    return occ, ids, sensor, visible, hidden


def episode(task):
    scene, case, arm, sensor = task
    p = json.loads((OUT / 'confirmation_protocol.json').read_text())
    calibration = json.loads((OUT / 'calibration.json').read_text())
    cfg = legacy.MPCConfig(**p['configs'][arm])
    matched.AGE_TARGETS = prep.TARGETS
    point = (prep.TARGETS.index(p['age_target']) if arm == 'age_margin'
             else p['points'].get(arm, p['points']['bayes']))
    factory = (partial(matched.AgeMarginAdapter, calibration=calibration, point=point)
               if arm == 'age_margin' else partial(matched.MatchedAdapter,
                   calibration=calibration, point=point, method='bayes' if arm == 'covariance' else arm))
    legacy._load_modules(CROWD)
    from crowd_sim.envs.occlusion_belief import OcclusionBelief
    original = OcclusionBelief._label_and_sensor
    if sensor == 'range_only':
        OcclusionBelief._label_and_sensor = range_sensor
    observation_stats = []
    nodes, weights = np.polynomial.hermite.hermgauss(7)
    xy = np.array(np.meshgrid(nodes, nodes)).reshape(2, -1).T * np.sqrt(2.)
    weights = np.outer(weights, weights).ravel() / np.pi

    def observe(env, obs, adapter, step):
        if step == 0 and layout(env) != p['layout_hashes'][f'{scene}:{case}']:
            raise RuntimeError('Initial layout mismatch')
        hidden = [t for t in adapter.rfs.tracks.values() if not t.visible]
        opportunity = 0
        mesh = adapter.sensor_mesh
        grid = adapter.sensor_grid
        for track in hidden:
            var = float(track.covariance[0, 0])
            if var <= 0:
                continue
            samples = track.mean[:2] + np.sqrt(var) * xy
            col = np.floor((samples[:, 0]-mesh[0][0, 0])/(mesh[0][0, 1]-mesh[0][0, 0])+.5).astype(int)
            row = np.floor((samples[:, 1]-mesh[1][0, 0])/(mesh[1][1, 0]-mesh[1][0, 0])+.5).astype(int)
            valid = (row >= 0) & (col >= 0) & (row < grid.shape[0]) & (col < grid.shape[1])
            labels = np.full(len(samples), .5)
            labels[valid] = grid[row[valid], col[valid]]
            opportunity += int(weights[labels == 0].sum() >= .01 and weights[labels == .5].sum() >= .01)
        observation_stats.append(dict(step=step, hidden=len(hidden), opportunity=opportunity,
                                      max_missed=max((t.missed_steps for t in hidden), default=0)))
        if arm == 'bayes' and sensor == 'range_and_occlusion' and case == p['cases'][0] and step in (8, 24):
            snapshot = dict(observation=obs, tracks=adapter.rfs.tracks,
                            detections=adapter.detected_entities, mesh=mesh, sensor=grid,
                            step=step, scene=scene, case=case)
            path = OUT / 'mechanism' / f'{scene}_{case}_{step}.pkl'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(pickle.dumps(snapshot, protocol=5))

    try:
        _, count, scenario, radius, width = matched.SCENES[scene]
        result = legacy.run_episode(CROWD, 'bayes', count, scenario, case, cfg, radius, width,
            time_limit=25, adapter_factory=factory, step_observer=observe,
            occlusion=sensor != 'full')
    finally:
        OcclusionBelief._label_and_sensor = original
    row = prep.clean(asdict(result))
    row.update(method=arm, scene=scene, sensor=sensor, observation_stats=observation_stats,
               penalty=row['nav_time'] if row['success_without_overlap'] else 25.)
    if row['bound_violations']:
        raise RuntimeError('Unexpected actuator violation')
    return row


def tasks(p):
    # Rotate arm order by layout; outcomes never affect scheduling.
    for scene in range(6):
        for j, case in enumerate(p['cases']):
            order = ARMS[j % len(ARMS):] + ARMS[:j % len(ARMS)]
            for arm in order:
                yield scene, case, arm, 'range_and_occlusion'
    for sensor in ('range_only', 'full'):
        for scene in range(6):
            for case in p['visibility_cases']:
                for arm in ('posterior_mean', 'covariance'):
                    yield scene, case, arm, sensor


def run(workers, limit):
    p = verify()
    path = OUT / 'episodes.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r['scene'], r['case_id'], r['method'], r['sensor']) for r in rows}
    if len(done) != len(rows):
        raise RuntimeError('Duplicate result')
    pending = [t for t in tasks(p) if t not in done]
    if limit:
        pending = pending[:limit]
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open('a') as stream:
        for i, row in enumerate(pool.map(episode, pending, chunksize=1), len(done)+1):
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            if i % 12 == 0:
                print(f'Completed {i}/{p["expected_episodes"]}', flush=True)


def summarize():
    p = verify()
    path = OUT / 'episodes.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    keyed = {(r['scene'], r['case_id'], r['method'], r['sensor']): r for r in rows}
    expected = set(tasks(p))
    if len(keyed) != len(rows) or set(keyed) - expected:
        raise RuntimeError('Duplicate or unexpected result')
    complete = set(keyed) == expected
    table = []
    for sensor in ('range_and_occlusion', 'range_only', 'full'):
        for scene in range(6):
            for arm in ARMS:
                subset = [r for r in rows if (r['sensor'],r['scene'],r['method']) == (sensor,scene,arm)]
                if not subset:
                    continue
                table.append(dict(sensor=sensor,scene=scene,method=arm,n=len(subset),
                    success=sum(r['success_without_overlap'] for r in subset),
                    collision=sum(r['collision_union'] for r in subset),
                    timeout=sum(not r['collision_union'] and not r['success_without_overlap'] for r in subset),
                    penalty=float(np.mean([r['penalty'] for r in subset]))))
    comparisons = []
    visibility_effects = []
    if complete:
        rng = np.random.default_rng(20260913)
        indices = rng.integers(0,100,(10000,100))
        signs = rng.choice((-1.,1.),size=(20000,100))
        for other in ('posterior_mean','covariance','age_margin','ewma','conformal'):
            a = [keyed[s,c,'bayes','range_and_occlusion'] for c in p['cases'] for s in range(6)]
            b = [keyed[s,c,other,'range_and_occlusion'] for c in p['cases'] for s in range(6)]
            comparison = dict(a='bayes', b=other)
            for field in ('collision_union','success_without_overlap','penalty'):
                delta = np.array([x[field]-y[field] for x,y in zip(a,b)]).reshape(100,6).mean(axis=1)
                samples = delta[indices].mean(axis=1)
                comparison[field] = dict(delta=float(delta.mean()), ci95=np.quantile(samples,[.025,.975]).tolist())
                if field == 'collision_union':
                    null = (signs*delta).mean(axis=1)
                    comparison['seed_block_randomization_p'] = float((1+np.count_nonzero(
                        np.abs(null) >= abs(delta.mean())-1e-15))/(len(null)+1))
            comparison['A_fail_B_success'] = sum(not x['success_without_overlap'] and y['success_without_overlap'] for x,y in zip(a,b))
            comparison['A_success_B_fail'] = sum(x['success_without_overlap'] and not y['success_without_overlap'] for x,y in zip(a,b))
            # Descriptive episode-level McNemar; clustered intervals are primary.
            ab = sum(x['collision_union'] and not y['collision_union'] for x,y in zip(a,b))
            ba = sum(y['collision_union'] and not x['collision_union'] for x,y in zip(a,b))
            comparison['descriptive_mcnemar_p'] = float(min(1.,2*binom.cdf(min(ab,ba),ab+ba,.5))) if ab+ba else 1.
            comparisons.append(comparison)
        secondary = [r for r in comparisons if r['b'] != 'posterior_mean']
        order = sorted(secondary,key=lambda r:r['seed_block_randomization_p'])
        running = 0.
        for rank, r in enumerate(order):
            running = max(running,min(1.,(len(order)-rank)*r['seed_block_randomization_p']))
            r['secondary_holm_p'] = running
        small_indices = rng.integers(0,20,(10000,20))
        for sensor in ('range_only','full'):
            for field in ('collision_union','success_without_overlap','penalty'):
                blocks = []
                for case in p['visibility_cases']:
                    delta = []
                    for scene in range(6):
                        def effect(condition):
                            return (keyed[scene,case,'covariance',condition][field]
                                    - keyed[scene,case,'posterior_mean',condition][field])
                        delta.append(effect('range_and_occlusion')-effect(sensor))
                    blocks.append(np.mean(delta))
                blocks = np.array(blocks)
                visibility_effects.append(dict(comparison='combined minus '+sensor,field=field,
                    delta=float(blocks.mean()),ci95=np.quantile(blocks[small_indices].mean(axis=1),[.025,.975]).tolist()))
    prep.save(OUT / 'confirmation_summary.json', dict(complete=complete,n=len(rows),expected=4080,
        table=table,comparisons=comparisons,visibility_effects=visibility_effects,
        note='No safety-equivalence or deployment timing claim'))
    print(json.dumps(dict(complete=complete,n=len(rows),table=table,comparisons=comparisons),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('register','run','summarize'))
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.command == 'register': register()
    elif args.command == 'run': run(args.workers,args.limit)
    else: summarize()
