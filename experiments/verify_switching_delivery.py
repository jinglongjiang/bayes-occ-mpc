"""Delivery invariants, including source/data hashes and aggregate reconstruction."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import platform
import scipy
import sklearn
import numpy as np
import pandas as pd
from experiments.switching_validation import ROOT,OUT


def run(data):
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((ROOT/'experiments/switching_validation.py').read_bytes()).hexdigest()==manifest['source_sha256']
    assert hashlib.sha256((OUT/'PROTOCOL_ZH.txt').read_bytes()).hexdigest()==manifest['protocol_sha256']
    for filename,digest in manifest['data_files'].items():
        assert hashlib.sha256((data/filename).read_bytes()).hexdigest()==digest
    protocol=json.loads((ROOT/'results/response_validation/long_context/prediction_protocol.json').read_text())
    checks=[]
    for directory in (OUT,OUT/'nonlinear'):
        path=directory/'queries.csv'
        frame=pd.read_csv(path if path.exists() else directory/'queries.csv.gz')
        assert set(frame.seed)=={11,29,47}
        assert frame.groupby(['fold','track','time']).size().eq(3).all()
        assert len(frame)==18240
        assert np.isfinite(frame.select_dtypes(include='number').values).all()
        for fold in protocol['splits']:
            actual=set(frame.loc[frame.fold==fold['name'],'track'])
            assert actual==set(fold['test_tracks'])
            assert actual.isdisjoint(fold['train_tracks'])
            assert actual.isdisjoint(fold['val_tracks'])
        summary=json.loads((directory/'summary.json').read_text())
        main=frame[frame.fold!='moving_temporal']
        mean=main.groupby(['track','time']).learned_full.mean().groupby('track').mean().mean()
        assert abs(mean-summary[0]['means']['learned_full'])<1e-12
        for col in [c for c in frame if c.endswith('_brier')]:
            assert frame[col].between(0,1).all()
        # Deterministic compressed archive is small enough to publish without raw trajectories.
        if path.exists():
            with (directory/'queries.csv.gz').open('wb') as f:
                with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as zipped:
                    zipped.write(path.read_bytes())
            with gzip.open(directory/'queries.csv.gz','rb') as f:
                assert f.read()==path.read_bytes()
        checks.append(dict(directory=str(directory.relative_to(ROOT)),rows=len(frame),
            distinct_query_keys=int(frame.groupby(['fold','track','time']).ngroups),
            forecast_mean_reconstructed=float(mean),finite=True,split_disjoint=True))
    nonlinear=json.loads((OUT/'nonlinear/manifest.json').read_text())
    assert hashlib.sha256((ROOT/'experiments/switching_nonlinear.py').read_bytes()).hexdigest()==nonlinear['source_sha256']
    tests=subprocess.run([sys.executable,'-m','unittest',
        'experiments.test_switching_validation','experiments.test_response_validation'],cwd=str(ROOT),capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    result=dict(checks=checks,source_and_data_hashes_match=True,tests=tests.stderr,
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,
                         scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform()),
        scope='Same-machine deterministic artifact checks, not cross-environment replication or closed-loop navigation')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('/home/abc/temp/peroi-full/data'))
    run(parser.parse_args().data)
