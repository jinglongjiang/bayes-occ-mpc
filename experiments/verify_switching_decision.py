"""Check the complete frozen decision study and hash its delivery artifacts."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/switching_decision'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((OUT / 'navigation_protocol.json').read_text())
    assert digest(ROOT / 'experiments/switching_navigation.py') == protocol['inference_source_sha256']
    assert digest(OUT / 'deployment.joblib') == protocol['model_sha256']
    records = [json.loads(x) for x in (OUT / 'navigation.jsonl').read_text().splitlines()]
    arms = {'O', 'CV', 'TREE', 'MAP', 'IID', 'FULL'}
    assert len(records) == len({(r['case'], r['arm']) for r in records}) == 360
    cases = sorted({r['case'] for r in records})
    assert cases == list(range(8000000, 8000060))
    for case in cases:
        block = [r for r in records if r['case'] == case]
        assert {r['arm'] for r in block} == arms
        assert len({r['layout_hash'] for r in block}) == 1
        for step in range(min(len(r['steps']) for r in block)):
            assert len({r['steps'][step]['human_audit_hash'] for r in block}) == 1
    for r in records:
        assert r['status'] == 'ok'
        assert r['success'] + r['collision'] + r['timeout'] == 1
        for s in r['steps']:
            assert max(s['speed_excess'], s['acceleration_excess']) <= 1e-6
    reproduction = json.loads((OUT / 'forecast_reproduction.json').read_text())
    assert len(reproduction) == 21
    assert max(r['max_error'] for r in reproduction) < 1e-14
    tests = subprocess.run([sys.executable, '-m', 'unittest',
        'experiments.test_response_validation', 'experiments.test_switching_validation',
        'experiments.test_switching_decision'], cwd=str(ROOT), capture_output=True, text=True)
    assert tests.returncode == 0, tests.stderr
    paths = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'verification.json')
    result = dict(passed=True, episodes=len(records), layouts=len(cases),
        steps=sum(len(r['steps']) for r in records), python=platform.python_version(),
        human_paths_equal_on_paired_prefix=True, frozen_code_and_model_match=True,
        forecast_reproduction_max=max(r['max_error'] for r in reproduction),
        test_output=tests.stderr, artifacts={str(p.relative_to(ROOT)): digest(p) for p in paths})
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('artifacts', 'test_output')}, indent=2))


if __name__ == '__main__':
    main()
