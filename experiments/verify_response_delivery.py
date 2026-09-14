"""Check archived protocols, splits, numerical outputs and unchanged formal inputs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from experiments.occlusion_confirmation import verify
from experiments.response_data_audit import structural_checks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/response_validation'


def main():
    verify()
    result = dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn']},
        formal_hashes='PASS', structural=structural_checks(),outputs={})
    for version in ['stable','long_context']:
        p=json.loads((OUT/version/'prediction_protocol.json').read_text())
        source=(subprocess.check_output(['git','show','38f60df:experiments/peroi_response_validation.py'],cwd=ROOT)
                if version=='stable' else (ROOT/'experiments/peroi_response_validation.py').read_bytes())
        assert hashlib.sha256(source).hexdigest()==p['source_sha256']
        for split in p['splits']:
            sets=[set(split[f'{name}_tracks']) for name in ['train','val','test']]
            assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        f=pd.read_csv(OUT/version/'prediction_queries.csv')
        columns=['cv','ca','pooled','true_label','map','full_mean','prior_mean','shuffled_label']
        assert np.isfinite(f[columns].values).all()
        assert (f[columns].values>=0).all() and (f[columns].values<1000).all()
        sessions=f[(f.family=='trees')&(f.fold!='moving_temporal')]
        result['outputs'][version]=dict(rows=len(f),queries=len(sessions),
            tracks=int(sessions.track.nunique()),sessions=int(sessions.session.nunique()),
            source_hash='PASS',split_disjointness='PASS',finite_predictions='PASS')
    data=Path('/home/abc/temp/dataset-PeRoI.zip')
    result['dataset_md5']=hashlib.md5(data.read_bytes()).hexdigest()
    assert result['dataset_md5']=='69a095a086cbf12631eb4c359b2d9477'
    result['dataset_sha256']=hashlib.sha256(data.read_bytes()).hexdigest()
    (OUT/'delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
