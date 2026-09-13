"""Read-only selected-plan risk contribution audit, separate from confirmation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.occlusion_confirmation import OUT, verify, legacy, matched, build, CROWD
from experiments.occlusion_closeout import save
import numpy as np

DEST = OUT / 'binding'


def eligible(adapter, order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    xy = np.array(np.meshgrid(nodes, nodes)).reshape(2, -1).T * np.sqrt(2.)
    weights = np.outer(weights, weights).ravel() / np.pi
    mx, my = adapter.sensor_mesh
    grid = adapter.sensor_grid
    result = []
    for j, identifier in enumerate(adapter.reported_ids):
        track = adapter.rfs.tracks[identifier]
        if track.visible or track.covariance[0, 0] <= 0:
            continue
        points = track.mean[:2] + np.sqrt(track.covariance[0, 0]) * xy
        col = np.floor((points[:, 0] - mx[0, 0]) / (mx[0, 1] - mx[0, 0]) + .5).astype(int)
        row = np.floor((points[:, 1] - my[0, 0]) / (my[1, 0] - my[0, 0]) + .5).astype(int)
        valid = (row >= 0) & (col >= 0) & (row < grid.shape[0]) & (col < grid.shape[1])
        labels = np.full(len(points), .5)
        labels[valid] = grid[row[valid], col[valid]]
        if weights[labels == 0].sum() >= .01 and weights[labels == .5].sum() >= .01:
            result.append(j)
    return result


def audit(row):
    p = verify()
    cfg = legacy.MPCConfig(**p['configs']['bayes'])
    calibration = json.loads((OUT / 'calibration.json').read_text())
    _, _, ActionXY, _, _ = legacy._load_modules(CROWD)
    env = build(row['scene'], row['case_id'])
    adapter = matched.MatchedAdapter('bayes', cfg.horizon, cfg.human_margin, cfg.dt,
        cfg.chance_limit, cfg.fixed_uncertainty_radius, cfg.acceleration_std,
        method='bayes', calibration=calibration, point=p['points']['bayes'])
    planner = legacy.ContinuousCEMMPC(cfg)
    records = []
    start = time.perf_counter()
    for step in row['steps']:
        obs = adapter.read(env)
        index = int(round(env.global_time / cfg.dt))
        action, _ = planner.plan(obs, row['case_id'] * 100003 + index * 97 + 1729)
        expected = np.array([step['action_a'], step['action_b']])
        if not np.allclose(action, expected, atol=1e-9, rtol=0):
            raise RuntimeError(('Action replay mismatch', row['scene'], row['case_id'], index))
        controls = planner.last_controls[None]
        positions = obs.robot_xy[None, None] + np.cumsum(controls * cfg.dt, axis=1)
        hazard = planner._belief_collision_hazard(controls, obs, positions)
        entry = dict(step=index, hidden=sum(not t.visible for t in adapter.rfs.tracks.values()),
                     reported_hidden=sum(not adapter.rfs.tracks[i].visible for i in adapter.reported_ids))
        for order in (7, 15):
            indices = eligible(adapter, order)
            delta = np.zeros(cfg.horizon)
            if indices and hazard is not None:
                reduced = copy.copy(obs)
                reduced.human_existence = obs.human_existence.copy()
                reduced.human_existence[indices] = 0.
                without = planner._belief_collision_hazard(controls, reduced, positions)
                delta = (-np.expm1(-hazard) + np.expm1(-without))[0]
                if delta.min() < -1e-12:
                    raise RuntimeError('Non-monotone removal')
            entry[str(order)] = dict(eligible_tracks=len(indices),
                max_probability_removal=float(delta.max()), first_probability_removal=float(delta[0]))
        records.append(entry)
        env.step(ActionXY(*expected))
        if not np.allclose([env.robot.px, env.robot.py], [step['x'], step['y']], atol=1e-9, rtol=0):
            raise RuntimeError('State replay mismatch')
    result = dict(scene=row['scene'], case=row['case_id'], steps=records,
                  replay_seconds=time.perf_counter()-start)
    save(DEST / ('%s_%s.json' % (row['scene'], row['case_id'])), result)
    return row['scene'], row['case_id']


def summarize():
    rows = [json.loads(f.read_text()) for f in DEST.glob('[0-5]_*.json')]
    output = dict(complete=len(rows) == 600, episodes=len(rows), configurations=[])
    for scene in [None] + list(range(6)):
        steps = [s for r in rows if scene is None or r['scene'] == scene for s in r['steps']]
        for order in (7, 15):
            hidden = sum(s['hidden'] > 0 for s in steps)
            opportunities = sum(s[str(order)]['eligible_tracks'] > 0 for s in steps)
            for threshold in (.005, .01, .02):
                bound = sum(s[str(order)]['max_probability_removal'] >= threshold for s in steps)
                output['configurations'].append(dict(scene=scene, quadrature=order, threshold=threshold,
                    states=len(steps), hidden_states=hidden, opportunity_states=opportunities,
                    potential_binding_states=bound, fraction_all=bound/len(steps) if steps else None,
                    fraction_hidden=bound/hidden if hidden else None,
                    fraction_opportunity=bound/opportunities if opportunities else None))
    output['interpretation'] = ('Removing eligible tracks is an optimistic risk-reduction diagnostic, not a legal '
        'posterior update, not a bound on risk increases or alternative-candidate effects, and not evidence of '
        'decision improvement. Primary descriptive threshold .01, quadrature 15; 7 checks prior coarse screening.')
    save(DEST / 'summary.json', output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    verify()
    DEST.mkdir(exist_ok=True)
    protocol = dict(baseline='6bf42a163344e9d624ea6b0d4502c55e2fc4ac1f',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((OUT/'episodes.jsonl').read_bytes()).hexdigest(),
        scope='Post-confirmation read-only developmental diagnostic; no new navigation outcomes',
        population='All 600 full-Bayes range-and-occlusion layouts, all recorded control steps',
        primary_threshold=.01, sensitivity_thresholds=[.005, .02], quadrature_orders=[7, 15],
        definition='Max per-step collision probability decrease on actual selected plan after removal of ALL eligible tracks',
        denominators=['all control states', 'states with hidden tracks', 'states with eligible reported tracks'],
        decision='Below 10% of all control states: do not implement negative-evidence mixture in this round. '
        'Above 10%: only establishes potential relevance, not benefit; do not automatically claim novelty.',
        replay_tolerance=1e-9, stopping='Finish registered population; stop on source or action/state mismatch')
    path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Diagnostic protocol changed')
    save(path, protocol)
    rows = [json.loads(line) for line in (OUT/'episodes.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['method']=='bayes' and r['sensor']=='range_and_occlusion']
    assert len(rows) == 600
    pending = [r for r in rows if not (DEST/('%s_%s.json' % (r['scene'], r['case_id']))).exists()]
    if args.limit is not None:
        pending = pending[:args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(audit, pending, chunksize=1):
            print('Replayed', *result, flush=True)
    summarize()


if __name__ == '__main__':
    main()
