"""Dataset integrity and representation checks; no navigation claims."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def structural_checks():
    rng = np.random.default_rng(20260914)
    # Enumerate every partition of four people, not arbitrary inconsistent pairs.
    partitions = sorted(set(tuple(next(j for j in range(4) if a[j] == a[i])
                                  for i in range(4))
                            for a in itertools.product(range(4), repeat=4)))
    pairs = list(itertools.combinations(range(4), 2))
    indicators = np.array([[p[i] == p[j] for i, j in pairs] for p in partitions])
    error = 0.
    for _ in range(1000):
        w = rng.dirichlet(np.ones(len(partitions)))
        costs = rng.uniform(0, 5, (len(pairs), 12))
        full = w @ (indicators @ costs)
        marginal = (w @ indicators) @ costs
        error = max(error, float(abs(full-marginal).max()))
        assert np.allclose(full, marginal, atol=1e-12)
        assert full.argmin() == marginal.argmin()
    # Unknown object persistence and unknown detection can be observationally confounded.
    # Fresh independent existence with r=.5,pD=.8 and r=.8,pD=.5 has the same count law.
    from scipy.stats import binom
    count_a = binom.pmf(np.arange(11), 10, .5*.8)
    count_b = binom.pmf(np.arange(11), 10, .8*.5)
    assert np.array_equal(count_a, count_b)
    return dict(group_partitions=len(partitions), group_trials=1000,
                group_additive_expected_cost_max_error=error,
                group_scope='Fixed candidate additive pairwise social cost only; not non-additive group dynamics or information gathering',
                sensor_count_confounding_max_error=float(abs(count_a-count_b).max()),
                sensor_scope='Independent fresh existence example, not general persistent-track identifiability',
                shared_health_two_misses=.5*.9**2+.5*.1**2,
                mean_health_two_misses=.5**2)


def audit_file(path):
    d = pd.read_csv(path)
    out = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               rows=len(d), columns=list(d.columns), missing=d.isna().sum().to_dict())
    for c in ('Pedestrian_ID', 'Robot_Presence', 'Robot_Type', 'Robot_Influence'):
        if c in d:
            if c == 'Pedestrian_ID':
                out['tracks'] = int(d[c].nunique())
            else:
                out[c] = {str(k): int(v) for k, v in d[c].value_counts(dropna=False).items()}
    if 'Pedestrian_ID' in d and 'Frame_Number' in d:
        out['duplicate_track_timestamps'] = int(d.duplicated(['Pedestrian_ID', 'Frame_Number']).sum())
        groups = d.groupby('Pedestrian_ID')
        out['track_label_counts'] = (groups['Robot_Influence'].first().value_counts().to_dict()
                                     if 'Robot_Influence' in d else {})
        out['label_changes_within_track'] = int((groups['Robot_Influence'].nunique() > 1).sum()) if 'Robot_Influence' in d else None
        out['time_range'] = [float(d.Frame_Number.min()), float(d.Frame_Number.max())]
        delta = groups.Frame_Number.diff().dropna().values
        out['timestamp_delta_quantiles'] = np.quantile(delta, [0,.01,.5,.99,1]).tolist() if len(delta) else []
        out['nonpositive_deltas'] = int((delta <= 0).sum())
    if {'X_Robot', 'Y_Robot'} <= set(d):
        out['robot_unique_positions'] = len(d[['X_Robot', 'Y_Robot']].drop_duplicates())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.root.rglob('*.csv'))
    result = dict(files=[audit_file(p) for p in files], structural_checks=structural_checks())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=int)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False, default=int))


if __name__ == '__main__':
    main()
