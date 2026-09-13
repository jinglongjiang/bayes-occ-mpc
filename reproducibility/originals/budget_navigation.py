"""Population-only closed-loop budget experiment, isolated from the navigation core."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import dataclasses
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'archive/legacy'))
import continuous_mpc_gate as legacy
from integration.crowdnav import BayesObservationAdapter
from nav.belief import BeliefConfig
from nav.contracts import MPCConfig
from nav.planner import MPCPlanner
from nav.risk import CompiledDiscRisk

OUT = ROOT / 'results/hermite_budget'
CROWD = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
ARMS = ('exact', 'hermite')
CONDITIONS = ('clean', 'severe')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


def layout(env):
    rows = [[a.px, a.py, a.gx, a.gy, a.v_pref, a.radius]
            for a in [env.robot] + env.humans]
    return hashlib.sha256(np.asarray(rows, dtype=np.float64).round(9).tobytes()).hexdigest()[:16]


def environment(case):
    env, _, _ = legacy.build_env(CROWD, 'bayes', 20, 'square_crossing',
                                 square_width=10., time_limit=25, occlusion=True)
    env.reset(options={'test_case': case})
    assert env.time_step == .25 and env.time_limit == 25
    assert not env.robot.visible and env.robot.kinematics == 'holonomic'
    return env


def history():
    roots = [Path('/home/abc/temp'), Path('/home/abc/workspace/bayes_occ_mpc/results'),
             ROOT / 'results']
    cases, hashes, failures = set(), set(), []
    count = 0

    def walk(v, key=''):
        if isinstance(v, dict):
            if key == 'layout_hashes':
                hashes.update(x for x in v.values() if isinstance(x, str))
                cases.update(int(x) for x in v if str(x).isdigit())
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            for x in v:
                walk(x, key)
        elif isinstance(v, int) and not isinstance(v, bool):
            if key in ('case', 'case_id', 'test_case', 'cases', 'case_ids', 'test_cases'):
                if 0 <= v < 2**31:
                    cases.add(v)
        elif isinstance(v, str) and key in ('layout_hash', 'initial_layout_hash'):
            hashes.add(v)

    for root in roots:
        for path in root.rglob('*'):
            if OUT == path or OUT in path.parents or not path.is_file():
                continue
            match = re.search(r'replay_\d+_(\d+)\.pkl$', path.name)
            if match:
                cases.add(int(match[1]))
            if path.suffix not in ('.json', '.jsonl'):
                continue
            try:
                with path.open() as stream:
                    if path.suffix == '.jsonl':
                        for line in stream:
                            if line.strip():
                                walk(json.loads(line))
                    else:
                        walk(json.load(stream))
                count += 1
            except (ValueError, OSError) as exc:
                failures.append({'path': str(path), 'error': str(exc)})
    return cases, hashes, dict(roots=list(map(str, roots)), files=count,
                               unreadable=failures, historical_cases=sorted(cases))


def initialize():
    if (OUT / 'protocol.json').exists():
        return load_protocol()
    base = json.loads((ROOT / 'archive/hermite_audit/protocol.json').read_text())['config']
    configs, sources = {}, {}
    scales = (.5, .75, 1., 1.5, 2.)
    for condition in CONDITIONS:
        source = Path('/home/abc/temp/formal') / ('frozen_' + condition + '.json')
        point = json.loads(source.read_text())['points']['bayes']
        config = dict(base)
        config.update(chance_limit=min(.9, .5 * scales[point]),
                      near_chance_limit=.15 * scales[point])
        configs[condition] = config
        sources[condition] = dict(path=str(source), sha256=sha(source), point=point)
    n0 = base['population']
    assert n0 == configs['clean']['population'] == configs['severe']['population']
    print('Scanning historical result files before registering layouts', flush=True)
    used, hashes, audit = history()
    start = max(1000000, max(used, default=0) + 1000)
    if start + 80 >= 2**32 - 10000:
        raise RuntimeError('case seed outside generator range')
    # Reconstruct dense-square layouts for all historical IDs under this generator.
    env = environment(0)
    for case in sorted(used):
        env.reset(options={'test_case': case})
        hashes.add(layout(env))
    splits = {'development': {}, 'pilot': {}}
    new_hashes = set()
    case = start
    for phase, n in (('development', 10), ('pilot', 30)):
        for condition in CONDITIONS:
            splits[phase][condition] = []
            for _ in range(n):
                env.reset(options={'test_case': case})
                digest = layout(env)
                if case in used or digest in hashes or digest in new_hashes:
                    raise RuntimeError('layout collision; do not silently replace selected cases')
                new_hashes.add(digest)
                splits[phase][condition].append(dict(case=case, layout_hash=digest))
                case += 1
    files = list((ROOT / 'nav').glob('*.py')) + [ROOT / 'integration/crowdnav.py',
             Path(__file__), ROOT / 'archive/legacy/continuous_mpc_gate.py',
             ROOT / 'archive/hermite_audit/protocol.json']
    files += list((CROWD / 'crowd_sim').rglob('*.py'))
    files += [CROWD / 'crowd_nav/configs/env.config']
    protocol = dict(baseline='f4fbb18', branch='hermite-budget-20260912',
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        config=configs, frozen_sources=sources, N0=n0,
        grid=[int(n0 * x) for x in (1, 1.25, 1.5, 1.75, 2)], splits=splits,
        files={str(p): sha(p) for p in files}, history=audit,
        layout_check=dict(registered=80, internal_duplicates=0, historical_overlap=0,
                          historical_hash_count=len(hashes)),
        observation_noise=dict(clean=[0., 0., 1.], severe=[.1, .2, .8]),
        seeds=dict(observation='case*1000003+7919', planner='case*100003+step*97+1729',
                   environment='CrowdSim test offset + case; no ID modulo before generation'),
        budget=dict(ms=250, quantile=99, eligibility='both condition p99 <=250',
                    mode='soft, synchronous fixed simulation dt; no deadline action change',
                    timing='adapter.read through checked ActionXY construction; env.step and audit excluded'),
        selection='min collisions, max audited success, min mean penalty, min N among eligible',
        gate='both arms meet budget; B collisions <= A; net successes >=3 OR penalty gain >=0.5s with success not reduced',
        scope='steps 1-3 only; no automatic additional baselines or confirm',
        cpu=platform.processor(), affinity=sorted(os.sched_getaffinity(0)),
        software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        threads={k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    save('protocol.json', protocol)
    save('selected_configs.json', {'status': 'not_calibrated'})
    (OUT / 'episodes.jsonl').touch()
    (OUT / 'verdict.md').write_text('Pending full-pipeline development calibration. No navigation conclusion.\n')
    return protocol


def load_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, digest in protocol['files'].items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: ' + path)
    return protocol


def episode(protocol, phase, arm, n, condition, item):
    case = item['case']
    cfg = MPCConfig(**dict(protocol['config'][condition], population=n))
    env = environment(case)
    assert layout(env) == item['layout_hash']
    _, _, ActionXY, _, _ = legacy._load_modules(CROWD)
    pos, vel, detection = protocol['observation_noise'][condition]
    bc = BeliefConfig(dt=cfg.dt, acceleration_std=cfg.acceleration_std,
                     detection_probability=min(.98, detection),
                     position_measurement_std=pos, velocity_measurement_std=vel)
    adapter = BayesObservationAdapter(cfg, bc, detection, case * 1000003 + 7919)
    planner = MPCPlanner(cfg, envelope=None if arm == 'exact' else CompiledDiscRisk)
    timings, actions, positions, clearances, planning = [], [], [], [], []
    event, overlap, terminated, truncated = 'nothing', 0, False, False
    while not (terminated or truncated):
        t0 = time.perf_counter()
        obs = adapter.read(env)
        step = int(round(env.global_time / cfg.dt))
        action, plan_ms = planner.plan(obs, case * 100003 + step * 97 + 1729)
        if not np.isfinite(action).all() or np.linalg.norm(action) > cfg.v_max + 1e-6:
            raise RuntimeError('invalid output action')
        command = ActionXY(float(action[0]), float(action[1]))
        elapsed = (time.perf_counter() - t0) * 1000
        before = np.array(env.robot.get_position())
        humans = np.asarray([h.get_position() for h in env.humans])
        radii = np.asarray([h.radius + env.robot.radius for h in env.humans])
        _, _, terminated, truncated, info = env.step(command)
        after = np.array(env.robot.get_position())
        # Outside timed control: independently verify the actual execution model.
        if not np.allclose(after, before + np.asarray(command) * cfg.dt, atol=1e-12, rtol=0):
            raise RuntimeError('holonomic prediction/execution mismatch')
        clearance = legacy.swept_min_clearance(before, after, humans,
                        np.asarray([h.get_position() for h in env.humans]), radii)
        overlap += int(clearance < -1e-9)
        timings.append(elapsed)
        planning.append(float(plan_ms))
        actions.append(list(command))
        positions.append(after.tolist())
        clearances.append(clearance)
        event = str(info.get('event', 'nothing'))
        if len(timings) > 110:
            raise RuntimeError('environment did not terminate')
    collision = int(event == 'collision' or overlap > 0)
    success = int(event == 'reach_goal' and not collision)
    timeout = int(event == 'timeout' and not collision)
    if success + collision + timeout != 1:
        raise RuntimeError('unclassified outcome: ' + event)
    return dict(phase=phase, arm=arm, population=n, condition=condition, case=case,
        layout_hash=item['layout_hash'], status='ok', event=event,
        success=success, collision=collision, timeout=timeout,
        raw_success=int(event == 'reach_goal'), raw_timeout=int(event == 'timeout'),
        actual_overlap_steps=overlap, nav_time=float(env.global_time),
        penalty=float(env.global_time) if success else 25.,
        pipeline_step_ms=timings, plan_step_ms=planning,
        actions=actions, positions=positions, physical_clearance=clearances)


def rows():
    return [json.loads(line) for line in (OUT / 'episodes.jsonl').read_text().splitlines()]


def key(row):
    return tuple(row[k] for k in ('phase', 'arm', 'population', 'condition', 'case'))


def run_jobs(protocol, phase, populations):
    previous = rows()
    if any(r['status'] != 'ok' for r in previous):
        raise RuntimeError('recorded code error: resolve explicitly, do not treat as timeout')
    done = {key(r) for r in previous}
    if len(done) != len(previous):
        raise RuntimeError('duplicate episode keys')
    jobs = []
    for condition in CONDITIONS:
        for index, item in enumerate(protocol['splits'][phase][condition]):
            # Alternate arm order to reduce systematic thermal/time-order bias.
            for arm in ARMS[::1 if index % 2 == 0 else -1]:
                for n in populations[arm]:
                    jobs.append((phase, arm, n, condition, item))
    for job in jobs:
        phase, arm, n, condition, item = job
        identity = (phase, arm, n, condition, item['case'])
        if identity in done:
            continue
        start = time.perf_counter()
        try:
            row = episode(protocol, *job)
        except Exception:
            row = dict(zip(('phase','arm','population','condition','case'), identity),
                       status='error', traceback=traceback.format_exc())
            with (OUT / 'episodes.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            raise
        row['wall_seconds'] = time.perf_counter() - start
        with (OUT / 'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        done.add(identity)
        print('%s %s N=%s %s case=%s S/C/T=%d/%d/%d p99=%.1f wall=%.1fs' %
              (phase, arm, n, condition, item['case'], row['success'], row['collision'],
               row['timeout'], np.percentile(row['pipeline_step_ms'],99), row['wall_seconds']), flush=True)


def stats(group):
    times = np.concatenate([r['pipeline_step_ms'] for r in group])
    return dict(n=len(group), success=sum(r['success'] for r in group),
        collision=sum(r['collision'] for r in group), timeout=sum(r['timeout'] for r in group),
        penalty=float(np.mean([r['penalty'] for r in group])),
        p50=float(np.percentile(times, 50)), p95=float(np.percentile(times, 95)),
        p99=float(np.percentile(times, 99)), deadline_fraction=float(np.mean(times > 250)),
        steps=len(times))


def select(protocol):
    data = [r for r in rows() if r['phase'] == 'development' and r['status'] == 'ok']
    if len(data) != 200:
        raise RuntimeError('development incomplete: %d/200' % len(data))
    results, selected = {}, {}
    for arm in ARMS:
        choices = []
        results[arm] = {}
        for n in protocol['grid']:
            group = [r for r in data if r['arm'] == arm and r['population'] == n]
            by_condition = {c: stats([r for r in group if r['condition'] == c]) for c in CONDITIONS}
            assert all(s['n'] == 10 for s in by_condition.values())
            eligible = all(s['p99'] <= 250 for s in by_condition.values())
            total = stats(group)
            results[arm][str(n)] = dict(eligible=eligible, combined=total, conditions=by_condition)
            if eligible:
                choices.append((total['collision'], -total['success'], total['penalty'], n))
        selected[arm] = min(choices)[-1] if choices else None
    value = dict(status='frozen' if all(selected.values()) else 'budget_failed', selected=selected,
                 development=results, frozen=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 protocol_sha256=sha(OUT / 'protocol.json'))
    save('selected_configs.json', value)
    print('SELECTED', selected, flush=True)
    return value


def summarize(protocol):
    selected = json.loads((OUT / 'selected_configs.json').read_text())
    data = rows()
    if selected['status'] != 'frozen':
        (OUT / 'verdict.md').write_text('Verdict 3: budget qualification incomplete or failed.\n'
            'No independent navigation conclusion. See selected_configs.json for all development candidates.\n')
        return
    pilot = [r for r in data if r['phase'] == 'pilot' and r['status'] == 'ok']
    if len(pilot) != 120 or any(r['status'] != 'ok' for r in data):
        (OUT / 'verdict.md').write_text('Verdict 3: independent pilot incomplete or contains code errors.\n')
        return
    lines = ['# Population-only budget pilot', '',
             '| Condition | Arm | N | Success/Collision/Timeout | Penalty s | Pipeline p50/p95/p99 ms | Overrun % |',
             '|---|---|---:|---|---:|---|---:|']
    budget = True
    totals = {}
    for condition in ('combined',) + CONDITIONS:
        for arm in ARMS:
            group = [r for r in pilot if r['arm'] == arm and
                     (condition == 'combined' or r['condition'] == condition)]
            s = stats(group)
            budget &= s['p99'] <= 250
            if condition == 'combined':
                totals[arm] = s
            lines.append('| %s | %s | %d | %d/%d/%d | %.3f | %.2f/%.2f/%.2f | %.3f |' %
                (condition, arm, selected['selected'][arm], s['success'], s['collision'],
                 s['timeout'], s['penalty'], s['p50'], s['p95'], s['p99'], s['deadline_fraction']*100))
    paired = {a: {r['case']: r for r in pilot if r['arm'] == a} for a in ARMS}
    assert paired['exact'].keys() == paired['hermite'].keys()
    rescued = harmed = 0
    changes = {}
    diffs = []
    for case, a in paired['exact'].items():
        b = paired['hermite'][case]
        assert a['layout_hash'] == b['layout_hash']
        rescued += int(not a['success'] and b['success'])
        harmed += int(a['success'] and not b['success'])
        label = next(k for k in ('success','collision','timeout') if a[k]) + ' -> ' + next(
            k for k in ('success','collision','timeout') if b[k])
        changes[label] = changes.get(label, 0) + 1
        diffs.append([b['success']-a['success'], b['penalty']-a['penalty']])
    a, b = totals['exact'], totals['hermite']
    gate = budget and b['collision'] <= a['collision'] and (
        b['success']-a['success'] >= 3 or (a['penalty']-b['penalty'] >= .5 and b['success'] >= a['success']))
    verdict = 1 if gate else (2 if budget else 3)
    lines += ['', 'A failed / B succeeded: %d; A succeeded / B failed: %d.' % (rescued, harmed),
              'Complete transitions: ' + json.dumps(changes, sort_keys=True),
              'Budget qualified in both conditions: %s. Engineering continuation gate: %s.' % (budget, gate),
              '', 'Verdict %d: %s' % (verdict, {1:'independent navigation benefit within the registered budget; further controls still required.',
              2:'retain computation acceleration only; do not expand this navigation-benefit experiment.',
              3:'budget qualification failed; no within-budget navigation claim.'}[verdict]),
              '', 'This is a 60-layout engineering pilot, not a significance or safety-equivalence claim.',
              'Only clean/severe 20-person dense_square; synchronous soft budget, not hard real-time deployment.',
              'No iCEM/EWMA/480-episode confirmation was started.']
    (OUT / 'verdict.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('initialize','budget_calibrate','pilot','summarize','run'))
    args = parser.parse_args()
    protocol = initialize()
    if args.command in ('budget_calibrate', 'run'):
        run_jobs(protocol, 'development', {a: protocol['grid'] for a in ARMS})
        select(protocol)
    if args.command in ('pilot', 'run'):
        selected = json.loads((OUT / 'selected_configs.json').read_text())
        if selected['status'] == 'frozen':
            if selected['protocol_sha256'] != sha(OUT / 'protocol.json'):
                raise RuntimeError('protocol changed after selection')
            run_jobs(protocol, 'pilot', {a: [selected['selected'][a]] for a in ARMS})
    if args.command != 'initialize':
        summarize(protocol)


if __name__ == '__main__':
    main()
