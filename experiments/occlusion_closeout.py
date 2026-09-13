"""Frozen-paper closeout: development-only age-margin extension and provenance.

This entry deliberately separates preparation from confirmation. No test result
can select a working point, and preparation never launches navigation episodes.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'archive/legacy'))
import evaluate_matched_safety as matched
from continuous_mpc_gate import run_episode
from reproducibility.runtime import CROWD

OUT = ROOT / 'results/occlusion_closeout'
FORMAL = Path('/home/abc/temp/formal')
CALIBRATION = Path('/home/abc/temp/calibration/calibration_clean.json')
RECORDS = CALIBRATION.parent / 'calibration_records'
TARGETS = (.50, .60, .75, .85, .90, .95, .975)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        verify()
        return
    calibration = json.loads(CALIBRATION.read_text())
    fit_ids, validation_ids = calibration['cases_fit'], calibration['cases_validation']
    paths = [RECORDS / f'case_{c}.npz' for c in fit_ids + validation_ids]
    fit = np.concatenate([np.load(RECORDS / f'case_{c}.npz')['rows'] for c in fit_ids])
    validation = [np.load(RECORDS / f'case_{c}.npz')['rows'] for c in validation_ids]
    expanded = matched.fit_age_margin(fit, validation, .25, 16, targets=TARGETS)
    if len(expanded) != len(TARGETS):
        raise RuntimeError('Incomplete age-margin fit')
    # Preserve all original fitted parameters; only add the missing low points.
    for target in TARGETS[2:]:
        keys = (f'{target:.2f}', f'{target:.3f}'.rstrip('0'))
        old = next((calibration['age_margin'][k] for k in keys if k in calibration['age_margin']), None)
        if old is None:
            raise RuntimeError('Missing historical age-margin fit: ' + str(target))
        expanded[f'{target:.2f}'] = old
    calibration['age_margin'] = expanded
    frozen = json.loads((FORMAL / 'frozen_clean.json').read_text())
    sources = [CALIBRATION, FORMAL / 'frozen_clean.json', *paths,
               Path(__file__), ROOT / 'archive/legacy/continuous_mpc_gate.py',
               ROOT / 'archive/legacy/evaluate_matched_safety.py',
               ROOT / 'archive/legacy/bayesian_rfs.py']
    sources += list((CROWD / 'crowd_sim').rglob('*.py'))
    sources += [CROWD / 'crowd_nav/configs/env.config']
    protocol = dict(
        baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        status='development_only; confirmation not registered or started',
        algorithm='original holonomic exact Bayes-MPC; no intent, trees, or new belief model',
        scenes=matched.SCENES,
        primary='paired audited collision: full Bayes versus same-tracker posterior mean',
        secondary=['covariance existence=1 versus posterior mean', 'age-margin', 'EWMA', 'Conformal'],
        inference='layout/seed-block paired intervals; Holm for registered secondary collision tests; no safety equivalence from nonsignificance',
        development_cases=validation_ids,
        age_targets=TARGETS,
        age_navigation_new_targets=TARGETS[:2],
        working_point_rule=frozen['rule'],
        historical_points=frozen['points'],
        fixed_config=asdict(matched.config(frozen['points']['bayes'])),
        confirmation_plan=dict(A=3600, B=480, total=4080,
            registration='after development selection and historical ID/hash exclusion',
            visibility=['range_and_occlusion', 'range_only', 'full_current_observation']),
        scope=dict(no_success_seeking_retuning=True, no_new_negative_evidence_algorithm=True,
                   no_mirror_claim_of_bayes_necessity=True, deadline_is_reported_not_gate=True),
        files={str(p): digest(p) for p in sources})
    save(OUT / 'calibration.json', calibration)
    protocol['files'][str(OUT / 'calibration.json')] = digest(OUT / 'calibration.json')
    save(path, protocol)
    print(json.dumps(dict(fit_rows=len(fit), validation_rows=sum(map(len, validation)),
                         age_margin=expanded, status=protocol['status']), indent=2), flush=True)


def verify():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, expected in protocol['files'].items():
        if digest(path) != expected:
            raise RuntimeError('Frozen source changed: ' + path)
    return protocol


def development_episode(task):
    target, case = task
    calibration = json.loads((OUT / 'calibration.json').read_text())
    # The geometric-only arm does not consume the probability thresholds.
    matched.AGE_TARGETS = TARGETS
    point = TARGETS.index(target)
    factory = partial(matched.AgeMarginAdapter, calibration=calibration, point=point)
    result = run_episode(CROWD, 'bayes', 5, 'circle_crossing', case,
                         matched.config(0), 4., None, time_limit=25,
                         adapter_factory=factory)
    row = clean(asdict(result))
    row.update(target=target, stage='development', audited_penalty=(
        row['nav_time'] if row['success_without_overlap'] else 25.))
    return row


def development(workers, limit):
    protocol = verify()
    path = OUT / 'development.jsonl'
    old = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r['target'], r['case_id']) for r in old}
    if len(done) != len(old):
        raise RuntimeError('Duplicate development result')
    tasks = [(t, c) for t in TARGETS[:2] for c in protocol['development_cases']
             if (t, c) not in done]
    if limit:
        tasks = tasks[:limit]
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open('a') as stream:
        for row in pool.map(development_episode, tasks, chunksize=1):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({k: row[k] for k in ('target', 'case_id', 'success_without_overlap',
                                                 'collision_union', 'audited_penalty')}), flush=True)


def summarize():
    protocol = verify()
    path = OUT / 'development.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    table = []
    for target in TARGETS:
        if target in TARGETS[:2]:
            subset = [r for r in rows if r['target'] == target]
        else:
            point = TARGETS.index(target) - 2
            p = Path('/home/abc/temp/devsweep/clean') / f'clean_age_margin_p{point}.json'
            subset = json.loads(p.read_text())['episodes']
        table.append(dict(target=target, n=len(subset),
            collisions=sum(r['collision_union'] for r in subset),
            successes=sum(r['success_without_overlap'] for r in subset),
            penalty=float(np.mean([r['nav_time'] if r['success_without_overlap'] else 25.
                                   for r in subset])) if subset else None))
    complete = all(r['n'] == len(protocol['development_cases']) for r in table)
    selection = None
    if complete:
        floor = min(r['collisions']/r['n'] for r in table)
        eligible = [r for r in table if r['collisions']/r['n'] <= floor + .02 + 1e-12]
        selection = min(eligible, key=lambda r: (r['penalty'], r['target']))['target']
    result = dict(complete=complete, age_frontier=table, selected_age_target=selection,
                  confirmation_episodes_completed=0,
                  note='Development only. No new navigation claim or safety-equivalence conclusion.')
    save(OUT / 'development_summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'development', 'summarize'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'development':
        development(args.workers, args.limit)
    else:
        summarize()
