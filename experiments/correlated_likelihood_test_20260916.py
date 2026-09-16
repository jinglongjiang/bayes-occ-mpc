"""Bounded historical holdout test. No actor, environment or forward-model edits."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

ROOT = Path('/home/abc/workspace/bayes_occ_mpc_hermite')
sys.path.insert(0, str(ROOT))
from experiments import goal_posterior_probe as p
OUT = Path('/home/abc/temp/correlated_likelihood_test_20260916')
CONFIGS = [(0., s, e) for s in (1., 2., 4., 8.) for e in (1., .5, .25, .1)]
CONFIGS += [(r, s, 1.) for r in (.3, .6, .9, .98) for s in (1., 2., 4., 8.)]


def calculate(history, inverse, config):
    rho, scale, eta = config
    score = np.zeros(history[0][1].shape[0])
    previous = None
    for step, residual in history:
        linked = previous is not None and previous[0] + 1 == step
        innovation = residual - rho * previous[1] if linked else residual
        variance = (1 - rho ** 2) if linked else 1.
        d = np.einsum('ni,ij,nj->n', innovation, inverse, innovation)
        score += -3. * np.log1p(d / (4 * scale ** 2 * variance))
        previous = (step, residual)
    return eta * score


def worker(record):
    path = OUT / ('case_%s.json' % record['case'])
    p.base.legacy._load_modules(p.base.CROWD)
    calibration = json.loads((p.OUT / 'calibration.json').read_text())
    legal = p.legal_record(record)
    pool = [(s, tid, f) for s, tid, f in p.base.queries(record)
            if f['goal_source'] != 'birth_antipode' and f['conflict']]
    ids = np.unique(np.linspace(0, len(pool)-1, min(8, len(pool)), dtype=int)) if pool else []
    selected = {(pool[i][0], pool[i][1]) for i in ids}
    targets = {tid for _, tid in selected}
    tracks, rows = {}, []
    for step, features in p.contexts(legal):
        for tid in targets & features.keys():
            f = features[tid]
            if tid not in tracks:
                model = p.GoalPosterior(record['scene'], step, f, calibration,
                                        record['case'] * 1009 + tid * 97, count=256)
                tracks[tid] = dict(goals=model.goals, previous=(step, f), history=[], inverse=model.inverse)
            else:
                track = tracks[tid]
                previous_step, previous = track['previous']
                if step == previous_step + 1:
                    residual = f['pos'] - p.forward(previous, track['goals'])[:, 0]
                    track['history'].append((previous_step, residual))
                track['previous'] = (step, f)
            if (step, tid) not in selected:
                continue
            track = tracks[tid]
            if not track['history']:
                continue
            positions = p.forward(f, track['goals'], 16)[:, np.array(p.base.HORIZONS)-1]
            # Truth is introduced only after legal predictions have been computed.
            truth = np.array([record['truth']['positions'][step+h][tid] for h in p.base.HORIZONS])
            truth_goal = np.array(record['truth']['goals'][tid])
            distances = np.linalg.norm(positions-truth, axis=2)
            pair = np.array([cdist(positions[:, h], positions[:, h]) for h in range(4)])
            scores = np.array([calculate(track['history'], track['inverse'], c) for c in CONFIGS])
            results = {}
            for n in (128, 256):
                w = np.exp(scores[:, :n] - logsumexp(scores[:, :n], axis=1)[:, None])
                mean = np.einsum('kn,nhd->khd', w, positions[:n])
                point = np.linalg.norm(mean-truth, axis=2).mean(axis=1)
                energy = (w @ distances[:n] - .5*np.einsum('kn,hnm,km->kh', w, pair[:, :n, :n], w)).mean(axis=1)
                goal_dist = np.linalg.norm(track['goals'][:n]-truth_goal, axis=1)
                # Local true-goal mass is descriptive, not continuous-goal classification accuracy.
                mass = w @ (goal_dist < .5)
                results[str(n)] = dict(point=point.tolist(), energy=energy.tolist(), true_region_mass=mass.tolist())
            cv = f['pos'] + np.array(p.base.HORIZONS)[:, None] * .25 * f['vel']
            rows.append(dict(step=step, tid=tid, results=results,
                             cv=float(np.linalg.norm(cv-truth, axis=1).mean())))
    result = dict(case=record['case'], split=record['split'], pool=len(pool), queries=len(rows), rows=rows)
    path.write_text(json.dumps(result, allow_nan=False))
    print('DONE', record['case'], len(rows), flush=True)
    return result


def summarize(cases):
    result = {}
    for n in ('128', '256'):
        means = {}
        for split in ('development', 'holdout'):
            use = [c for c in cases if c['split'] == split and c['rows']]
            means[split] = {metric: np.array([np.mean([r['results'][n][metric] for r in c['rows']], axis=0)
                                             for c in use]) for metric in ('point', 'energy', 'true_region_mass')}
        dev = means['development']['energy'].mean(axis=0)
        simple = int(np.argmin(dev[:16]))
        correlated = int(16 + np.argmin(dev[16:]))
        table = {}
        for label, idx in [('original', 0), ('simple', simple), ('correlated', correlated)]:
            table[label] = dict(index=idx, config=CONFIGS[idx], **{
                metric: float(means['holdout'][metric][:, idx].mean()) for metric in means['holdout']})
        comparisons = {}
        rng = np.random.default_rng(2407)
        for metric in ('point', 'energy'):
            diff = means['holdout'][metric][:, correlated] - means['holdout'][metric][:, simple]
            ids = rng.integers(0, len(diff), (10000, len(diff)))
            ci = np.quantile(diff[ids].mean(axis=1), [.025, .975])
            comparisons[metric] = dict(correlated_minus_simple=float(diff.mean()), ci95=ci.tolist(),
                                       relative_improvement=float(-diff.mean()/table['simple'][metric]))
        passed = all(c['ci95'][1] < 0 and c['relative_improvement'] >= .02 for c in comparisons.values())
        result[n] = dict(table=table, comparisons=comparisons, prediction_gate=passed)
    result['decision_gate_run'] = False
    result['approve_navigation'] = all(result[n]['prediction_gate'] for n in ('128', '256'))
    result['case_counts'] = {s: sum(c['split']==s and bool(c['rows']) for c in cases) for s in ('development','holdout')}
    result['query_counts'] = {s: sum(c['queries'] for c in cases if c['split']==s) for s in ('development','holdout')}
    (OUT/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


def main():
    records = p.data()
    OUT.mkdir(exist_ok=False)
    assert not ({r['case'] for r in records if r['split']=='development'} &
                {r['case'] for r in records if r['split']=='holdout'})
    # rho=0 must exactly recover the old independent Student-t score.
    e = np.array([[.1, .2], [.3, -.1]])
    history = [(0, e), (1, e*.8)]
    expected = sum(-3*np.log1p(np.sum(x*x, axis=1)/4) for _, x in history)
    np.testing.assert_allclose(calculate(history, np.eye(2), (0,1,1)), expected)
    protocol = dict(configs=CONFIGS, max_queries_per_case=8, candidates=[128,256],
        selection='evenly spaced eligible legal unknown-goal conflict queries; no arrival truth filter',
        tuning='development episode-mean Energy Score only; all holdout episodes excluded',
        gate='>=2% point and Energy Score gain versus selected simple baseline; paired episode CI upper<0, both candidate counts',
        scope='historical holdout, previously inspected by other studies; not fresh confirmation',
        unchanged='IL/PPO, forward motion model, calibrated Student scale, observations, reward',
        candidate='Student-t AR(1) residual likelihood; standard correlated-residual baseline, not claimed novel',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calibration_sha256=p.digest(p.OUT/'calibration.json'),data_sha256=p.digest(p.base.OUT/'episodes.jsonl'))
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        cases = list(pool.map(worker, records))
    summarize(cases)


if __name__ == '__main__':
    main()
