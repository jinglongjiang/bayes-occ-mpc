"""Read-only portable verification of frozen-transfer evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
mapping = json.loads((root / 'source_mapping.json').read_text())
protocols = [json.loads((root / c / 'protocol.json').read_text()) for c in ['clean', 'severe']]
assert protocols[0]['source_hashes'] == protocols[1]['source_hashes']
for original, digest in protocols[0]['source_hashes'].items():
    assert hashlib.sha256((root / mapping[original]).read_bytes()).hexdigest() == digest, original
result = json.loads((root / 'summary.json').read_text())
values = {}
for condition in ['clean', 'severe']:
    for arm, expected in result['summary'][condition].items():
        block = json.loads((root / condition / (arm + '_sc3.json')).read_text())
        rows = sorted(block['episodes'], key=lambda row: row['case_id'])
        assert [r['case_id'] for r in rows] == list(range(5100, 5200))
        assert block['frozen_sensor'] and block['point'] == protocols[0]['points'][arm]
        assert block['noise'] == ([0, 0, 1] if condition == 'clean' else [.1, .2, .8])
        a = np.array([[r['success_without_overlap'], r['collision_union'],
                       r['nav_time'] if r['success_without_overlap'] else 25.] for r in rows])
        values[condition, arm] = a
        assert int(a[:, 0].sum()) == expected['successes']
        assert int(a[:, 1].sum()) == expected['collisions']
        assert 100 - int(a[:, :2].sum()) == expected['other_failures']
        assert np.isclose(a[:, 2].mean(), expected['penalized_time'])
indices = np.random.default_rng(2407).integers(0, 100, size=(10000, 100))
for arm, comparison in result['paired_change_comparisons'].items():
    delta = (values['severe', 'bayes'] - values['clean', 'bayes']) - (values['severe', arm] - values['clean', arm])
    level = comparison['confidence']
    interval = np.quantile(delta[indices].mean(axis=1), [(1-level)/2, 1-(1-level)/2], axis=0)
    for i, name in enumerate(['success_change', 'collision_change', 'time_change']):
        expected = comparison['metrics'][name]
        assert np.isclose(delta[:, i].mean(), expected['mean'])
        assert np.allclose(interval[:, i], expected['ci'])
print('PASS: source hashes, 1000 paired episodes, summaries and bootstrap intervals')
