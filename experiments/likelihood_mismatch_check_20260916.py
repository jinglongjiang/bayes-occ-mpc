"""Fixed historical-state sensitivity check; no policy or production edits."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/abc/workspace/bayes_occ_mpc_hermite')
sys.path.insert(0, str(ROOT))
from experiments import goal_posterior_probe as probe


def main():
    records = probe.data()
    calibration = json.loads((probe.OUT / 'calibration.json').read_text())
    reference = json.loads((probe.DEPTH / 'reference.json').read_text())
    output = []
    for case, step, tid in probe.DEPTH_STATES:
        record = next(r for r in records if r['case'] == case)
        model, features = probe.posterior_at(record, step, tid, calibration)
        ref = next(r for r in reference if r['case'] == case)
        path = probe.DEPTH / ('reference_%s_refined.npz' % case)
        if not path.exists():
            path = probe.DEPTH / ('reference_%s.npz' % case)
        grid = np.load(path)
        rng = np.random.default_rng(case)
        sampled = rng.choice(len(grid['goals']), 256, replace=False)
        # Truth and old MAP are diagnostic challenge points, not prior samples.
        goals = np.vstack([ref['true_goal'], grid['goals'][np.argmax(grid['logdensity'])],
                           grid['goals'][sampled]])
        residuals = []
        for f, observed in model.history:
            residuals.append(observed - (probe.forward(f, goals)[:, 0] - f['pos']))
        residuals = np.asarray(residuals)
        mahal = np.einsum('tni,ij,tnj->tn', residuals, model.inverse, residuals)
        original = -3 * np.log1p(mahal / 4).sum(axis=0)
        np.testing.assert_allclose(original, model.logposterior(goals), atol=1e-10)
        np.testing.assert_allclose(original[0], ref['true_goal_log_likelihood'], atol=1e-8)
        rows = []
        for scale in (1., 2., 4., 8.):
            ll = -3 * np.log1p(mahal / (4 * scale ** 2)).sum(axis=0)
            for eta in (1., .5, .25, .1):
                score = eta * ll
                rows.append(dict(scale=scale, eta=eta,
                    true_minus_best_loglik=float(score[0] - score.max()),
                    candidates_above_truth=int((score > score[0] + 1e-12).sum()),
                    best_index=int(score.argmax())))
            assert all(rows[-i]['best_index'] == rows[-1]['best_index'] for i in range(1, 5))
        output.append(dict(case=case, step=step, transitions=len(model.history),
                           candidates=len(goals), rows=rows))
    old = json.loads((probe.DEPTH / 'likelihood_model.json').read_text())
    for case in old:
        for key, saved in case['rms_m'].items():
            actual = np.sqrt(np.mean([r['errors_m'][key] ** 2 for r in case['transitions']]))
            assert abs(actual - saved) < 1e-12
        assert all(r['errors_m']['oracle_memory_full_neighbors'] == 0
                   for r in case['transitions'])
    result = dict(scope='three historical development states; not heldout or navigation',
                  scales=[1, 2, 4, 8], temperatures=[1, .5, .25, .1],
                  rows=output, oracle_transition_count=sum(len(x['transitions']) for x in old),
                  oracle_rms= [{k:v for k,v in x.items() if k in ('case','rms_m')} for x in old],
                  checks='PASS: original likelihood reproduction, RMS, tempering MAP invariance',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = Path('/home/abc/temp/likelihood_mismatch_check_20260916.json')
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
