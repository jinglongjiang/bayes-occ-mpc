"""Small deterministic replay; never launches a navigation experiment queue."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments import goal_model_probe as base
from experiments import goal_posterior_probe as old
from experiments import intent_repair_run as repair
from integration.crowdnav import BayesObservationAdapter
from nav.contracts import MPCConfig
from nav.planner import MPCPlanner


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def check_posterior():
    fixture=json.loads((ROOT/'reproducibility/posterior_fixture.json').read_text())
    actual=repair.fitted_settings()
    assert json.loads(json.dumps(actual,default=plain))==fixture['settings']
    base.legacy._load_modules(base.CROWD)
    model=None;max_error=0.;previous=None;gaps=0
    for row in fixture['history']:
        f=row['features']
        for key in ('pos','vel','goal'):f[key]=np.array(f[key])
        step=row['step']
        if model is None:
            model=repair.RepairedPosterior(fixture['scene'],step,f,fixture['calibration'],fixture['seed'],**actual)
            assert not model.fixed
        else:
            gaps+=int(step>previous+1);model.update(step,f)
        for key in ('goals','logweights','density','bias','bias_covariance'):
            a=np.array(getattr(model,key));b=np.array(row['reference'][key])
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
            max_error=max(max_error,float(np.max(abs(a-b))))
        previous=step
    assert gaps>0 and model.updates>0
    forecast=old.forward(f,model.goals,16)
    np.testing.assert_allclose(forecast,fixture['forecast'],atol=1e-12,rtol=0)
    return dict(settings=fixture['settings'],history_rows=len(fixture['history']),gaps=gaps,
                updates=model.updates,max_posterior_error=max_error,
                max_forecast_error=float(np.max(abs(forecast-fixture['forecast']))))


def run(fixture):
    results = []
    _, _, ActionXY, _, _ = base.legacy._load_modules(base.CROWD)
    for item in fixture['episodes']:
        env = base.environment(item)
        adapter = BayesObservationAdapter(MPCConfig(**fixture['mpc_config']))
        planner = MPCPlanner(MPCConfig(**fixture['mpc_config']))
        _, effective, _ = base.legacy.build_env(base.CROWD, 'bayes', item['people'],
            item['scene'], circle_radius=4., square_width=10., time_limit=25, occlusion=True)
        row = dict(case=item['case'], layout_hash=base.layout(env),
                   effective_config={s: dict(effective.items(s)) for s in effective.sections()}, frames=[])
        for frame in item['frames']:
            obs = adapter.read(env)
            if adapter.detected_entities != frame['detections']:
                raise RuntimeError('recorded legal detections mismatch')
            action, _ = planner.plan(obs, item['case']*100003+frame['step']*97+1729)
            np.testing.assert_array_equal(action, frame['action'])
            env.step(ActionXY(*map(float, action)))
            row['frames'].append(dict(action=action, robot=list(env.robot.get_position()),
                humans=[list(h.get_position()) for h in env.humans]))
        f = item['features']
        for name in ('pos', 'vel', 'goal'): f[name] = np.asarray(f[name])
        # Freeze D outputs and the actual multimodal evaluator on a tiny candidate batch.
        forecasts = old.forward(f, np.array(item['goals']), 16)
        row['D'] = forecasts
        modes = {}
        if len(obs.entities):
            modes[0] = (forecasts, np.array([.2, .3, .5]), np.full(16, .02))
        mixed = repair.MixtureEnvelope(planner.cfg, obs, modes)
        controls = np.zeros((2, 16, 2)); controls[1, :, 0] = .1
        positions = obs.robot_xy[None, None] + np.cumsum(controls*.25, axis=1)
        row['mixture_risk'] = mixed.exact(positions)
        results.append(row)
    return json.loads(json.dumps(results, default=plain))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--check-entry', action='store_true', help='Also read/validate existing experiment records; no writes or simulation queue')
    parser.add_argument('--posterior-only',action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = ROOT / 'reproducibility/fixture.json'
    if args.capture:
        # Capture only before path edits, using the already recorded experiment inputs.
        records = old.data()
        fixture = dict(mpc_config=repair.read_config(), episodes=[])
        for case in (2000000, 2000010, 2000020):
            record = next(r for r in records if r['case'] == case)
            query = next(base.queries(record))
            f = query[2]
            fixture['episodes'].append(dict(case=case, people=record['people'], scene=record['scene'],
                frames=record['frames'][:12], features=f,
                goals=[f['goal'], np.asarray(f['goal'])+[.3, -.2], np.asarray(f['goal'])+[-.2, .3]]))
        fixture = json.loads(json.dumps(fixture, default=plain))
        fixture['reference'] = run(fixture)
        path.write_text(json.dumps(fixture, indent=2, default=plain)+'\n')
        args.output.write_text(json.dumps(dict(status='captured', source='6ee89d7', steps=36))+'\n')
        return
    from reproducibility.runtime import verify_frozen, sha
    forbidden = ['/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/']
    if ROOT != Path('/home/abc/workspace/bayes_occ_mpc_hermite'):
        forbidden.append('/home/abc/workspace/bayes_occ_mpc_hermite/')
    def deny_original_source(event, arguments):
        if event == 'open' and arguments and isinstance(arguments[0], (str, bytes)):
            path = os.fsdecode(arguments[0])
            if any(path.startswith(prefix) for prefix in forbidden):
                raise RuntimeError('attempted original-machine file access: '+path)
    sys.addaudithook(deny_original_source)
    manifest = json.loads((ROOT/'reproducibility/manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if sha(ROOT/relative) != expected: raise RuntimeError('manifest mismatch: '+relative)
    for rel, key in (('results/goal_model_probe/protocol.json', 'files'),
                     ('results/intent_repair/protocol.json', 'frozen_dependencies')):
        for p, h in json.loads((ROOT/rel).read_text())[key].items():
            # The old model protocol pins an earlier goal_model_probe revision.
            if p.endswith('/experiments/goal_model_probe.py') and rel.startswith('results/goal_model'):
                snap=ROOT/'reproducibility/originals/goal_model_protocol.py'
                if sha(snap)!=h: raise RuntimeError('original model protocol hash')
                verify_frozen(p, manifest['path_changes']['experiments/goal_model_probe.py']['original_sha256'])
            else: verify_frozen(p,h)
    try:
        verify_frozen(ROOT/'nav/planner.py', '0'*64)
    except RuntimeError:
        pass
    else:
        raise AssertionError('hash mismatch was accepted')
    fixture = json.loads(path.read_text())
    if args.check_entry:
        records = old.data()
        repair.setup(records)
    posterior=check_posterior()
    result = [] if args.posterior_only else run(fixture)
    errors = {}
    for expected, actual in zip(fixture['reference'], result):
        assert actual['layout_hash'] == expected['layout_hash']
        assert actual['effective_config'] == expected['effective_config']
        for field in ('D', 'mixture_risk'):
            a,b=np.array(actual[field]),np.array(expected[field])
            errors[str(actual['case'])+'/'+field] = float(np.max(np.abs(a-b)))
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        for a,b in zip(actual['frames'], expected['frames']):
            np.testing.assert_array_equal(a['action'], b['action'])
            for field in ('robot','humans'):
                errors[str(actual['case'])+'/'+field] = max(errors.get(str(actual['case'])+'/'+field,0.),
                    float(np.max(np.abs(np.array(a[field])-b[field]))))
                np.testing.assert_allclose(a[field],b[field],rtol=0,atol=1e-12)
    modules={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
             if getattr(m,'__file__',None) and (n.startswith(('crowd_sim','crowd_nav')) or n=='rvo2')}
    for name,p in modules.items():
        if name!='rvo2' and base.CROWD.resolve() not in Path(p).parents:
            raise RuntimeError('nonlocal module '+name)
    report=dict(status='pass', scope='same-host independent-directory replay; not a new navigation result',
        root=str(ROOT), python=sys.version, platform=platform.platform(), modules=modules,
        PYTHONPATH=os.environ.get('PYTHONPATH'), max_absolute_errors=errors,
        layout_count=0 if args.posterior_only else 3, replay_steps=0 if args.posterior_only else 36,
        action_tolerance='bitwise', numeric_atol=1e-12, numeric_rtol=0,posterior=posterior)
    report['experiment_entry_records_checked'] = len(records) if args.check_entry else None
    report['forbidden_source_roots'] = forbidden
    report['wrong_hash_rejected'] = True
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
