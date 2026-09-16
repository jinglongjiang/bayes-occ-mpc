"""Exploratory forecasting audit; future robot positions are privileged, not causal."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.covariance import LedoitWolf

ROOT = Path('/home/abc/temp/paper2_navware_20260916/selected_tracks')
OUT = Path('/home/abc/temp/paper2_navware_probe_20260916')
VALID = {3, 10, 17, 23}


def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(scope='observational prediction, not action counterfactual or navigation',
                    split='group1 train, blind-corner group1 validation, group2 test',
                    history_seconds=1.0, future_seconds=2.0, query_spacing_seconds=1.0,
                    maximum_interpolation_gap_seconds=0.25,
                    predictor='HistGradientBoosting max_iter=150 max_leaf_nodes=15 min_samples_leaf=30 l2=1',
                    seeds=[1, 2, 3], limitations=['manual selected traversals',
                    'only focal human and robot consistently present in both CSV formats',
                    'future robot is endogenous privileged information',
                    'group2 has only one participant group, not independent population replication'])
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    pieces = {}
    for f in sorted(ROOT.rglob('*.csv')):
        scene = int(f.parent.name.split('_')[0])
        d = pd.read_csv(f)
        if 'column' in d:
            for person, p in d.groupby('column'):
                pieces.setdefault((scene, int(person)), []).append(p[['timestamp', 'x', 'y', 'robot_x', 'robot_y']])
        else:
            for person in range(1, 6):
                p = d[['timestamp', 'x%d' % person, 'y%d' % person, 'robot_x', 'robot_y']].copy()
                p.columns = ['timestamp', 'x', 'y', 'robot_x', 'robot_y']
                pieces.setdefault((scene, person), []).append(p)
    rows, features, targets = [], [], []
    duplicates = 0
    conflicts = 0
    for (scene, person), ps in sorted(pieces.items()):
        d = pd.concat(ps).dropna().sort_values('timestamp')
        duplicates += int(d.duplicated('timestamp').sum())
        duplicate_rows = d[d.duplicated('timestamp', keep=False)]
        if len(duplicate_rows):
            conflicts += int((duplicate_rows.groupby('timestamp')[['x', 'y']].nunique().max(axis=1) > 1).sum())
        d = d.drop_duplicates('timestamp')
        ts = d.timestamp.to_numpy(np.int64)
        if len(ts) < 31:
            continue
        time = (ts - ts[0]) / 1e9
        xy = d[['x', 'y', 'robot_x', 'robot_y']].to_numpy(float)
        previous = -np.inf
        for j in range(len(time)):
            t = time[j]
            if t < 1 or t + 2 > time[-1] or t - previous < 1:
                continue
            lo, hi = np.searchsorted(time, [t - 1, t + 2])
            lo = max(0, lo - 1)
            hi = min(len(time) - 1, hi)
            if np.max(np.diff(time[lo:hi+1])) > .25:
                continue
            offsets = np.array([-1, -.5, 0, 1, 2])
            z = np.stack([np.interp(t + offsets, time, xy[:, k]) for k in range(4)], axis=1)
            human, robot = z[:, :2], z[:, 2:]
            hv = (human[2] - human[1]) / .5
            rv = (robot[2] - robot[1]) / .5
            current = np.r_[robot[2] - human[2], hv, rv]
            history = np.r_[(human[:2] - human[2]).ravel(), (robot[:2] - robot[2]).ravel()]
            future = (robot[3:] - robot[2]).ravel()
            features.append(np.r_[current, history, future])
            targets.append((human[3:] - human[2]).ravel())
            rows.append(dict(scene=scene, person=person, timestamp=int(ts[j]),
                             split='test' if scene >= 27 else 'validation' if scene in VALID else 'train'))
            previous = t
    if conflicts:
        raise RuntimeError('Conflicting repeated manual trajectories: %s' % conflicts)
    meta = pd.DataFrame(rows)
    x, y = np.asarray(features), np.asarray(targets)
    train = meta.split.to_numpy() == 'train'
    test = meta.split.to_numpy() == 'test'
    summary = dict(files=541, deduplicated_rows=duplicates, conflicting_timestamps=conflicts,
                   queries=meta.groupby('split').size().to_dict(),
                   scenes=meta.groupby('split').scene.nunique().to_dict(), results={},
                   caveat='Oracle robot future advantage is not causal or Bayesian evidence')
    predictions = {}
    for name, keep in [('Current', 6), ('History', 14), ('OracleRobotFuture', 18)]:
        xx = x.copy()
        xx[:, keep:] = 0
        predictions[name] = []
        for seed in [1, 2, 3]:
            model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                    min_samples_leaf=30, l2_regularization=1, random_state=seed))
            model.fit(xx[train], y[train])
            pred = model.predict(xx[test])
            predictions[name].append(pred)
        predictions[name] = np.mean(predictions[name], axis=0)
    predictions['CV'] = np.concatenate([x[test, 2:4], 2*x[test, 2:4]], axis=1)
    result_rows = meta[test].reset_index(drop=True)
    for name, pred in predictions.items():
        errors = np.linalg.norm((pred-y[test]).reshape(-1, 2, 2), axis=2)
        result_rows[name] = errors[:, 1]
        summary['results'][name] = dict(ade=float(errors.mean()), fde2=float(errors[:, 1].mean()),
                scene_equal_fde2=float(result_rows.groupby('scene')[name].mean().mean()))
    rng = np.random.RandomState(2407)
    for first, second in [('History', 'Current'), ('OracleRobotFuture', 'History')]:
        diff = result_rows.groupby('scene')[first].mean()-result_rows.groupby('scene')[second].mean()
        bs = rng.choice(diff.to_numpy(), (10000, len(diff)), replace=True).mean(axis=1)
        summary[first+'-'+second] = dict(scene_mean=float(diff.mean()),
                scene_bootstrap_ci95=np.quantile(bs, [.025, .975]).tolist())
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta.to_csv(OUT/'queries.csv', index=False)
    result_rows.to_csv(OUT/'test_errors.csv', index=False)
    np.savez_compressed(OUT/'features.npz', x=x, y=y, **predictions)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def joint_screen():
    out = OUT / 'joint'
    out.mkdir(exist_ok=True)
    protocol = dict(train_scenes=[6, 20, 26], calibration_scenes=[13], test_scenes=[32, 39, 46, 52],
                    means='same shared per-human HistGB', covariance='LedoitWolf on calibration residuals',
                    controls='same means and exact same per-human covariance blocks; remove off-diagonal human blocks only',
                    horizon=[.5, 1, 1.5, 2], query_spacing=1, thresholds=[.6, 1.0], samples=1024,
                    scope='proximity to recorded robot at discrete horizons; NOT counterfactual collision/navigation')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    feats, target, human0, robotfuture, metas = [], [], [], [], []
    for f in sorted(ROOT.rglob('*.csv')):
        d = pd.read_csv(f)
        if 'x1' not in d:
            continue
        scene = int(f.parent.name.split('_')[0])
        ts = d.timestamp.to_numpy(np.int64)
        time = (ts-ts[0])/1e9
        arr = d[['robot_x', 'robot_y']+sum((['x%d'%i, 'y%d'%i] for i in range(1, 6)), [])].to_numpy(float)
        previous = -np.inf
        for j, t in enumerate(time):
            if t < 1 or t+2 > time[-1] or t-previous < 1:
                continue
            lo, hi = np.searchsorted(time, [t-1, t+2])
            if np.max(np.diff(time[max(0,lo-1):min(len(time),hi+1)])) > .25:
                continue
            offsets = np.array([-1, -.5, 0, .5, 1, 1.5, 2])
            z = np.stack([np.interp(t+offsets, time, arr[:, k]) for k in range(12)], axis=1).reshape(7, 6, 2)
            if not np.isfinite(z).all():
                continue
            theta = float(d.robot_yaw_rad.iloc[j])
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            z = (z-z[2, 0]) @ rot
            h, r = z[:, 1:], z[:, 0]
            order = np.argsort(np.linalg.norm(h[2], axis=1))
            h = h[:, order]
            v = (h[2]-h[1])/.5
            oldv = (h[1]-h[0])/.5
            rv = np.tile((r[2]-r[1])/.5, (5, 1))
            feats.append(np.concatenate([h[2], v, oldv, rv], axis=1))
            target.append((h[3:]-h[2]).transpose(1,0,2).reshape(5,8))
            human0.append(h[2])
            robotfuture.append(r[3:])
            metas.append(dict(scene=scene, file=f.name, timestamp=int(ts[j])))
            previous = t
    x, y = np.asarray(feats), np.asarray(target)
    h0, rf = np.asarray(human0), np.asarray(robotfuture)
    meta = pd.DataFrame(metas)
    train = meta.scene.isin([6,20,26]).to_numpy()
    calib = (meta.scene==13).to_numpy()
    test = meta.scene.isin([32,39,46,52]).to_numpy()
    model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                        min_samples_leaf=30, l2_regularization=1, random_state=2407))
    model.fit(x[train].reshape(-1,8), y[train].reshape(-1,8))
    mean = model.predict(x.reshape(-1,8)).reshape(-1,5,8)
    residual = (y[calib]-mean[calib]).reshape(-1,40)
    estimator = LedoitWolf().fit(residual)
    covariance = estimator.covariance_ + 1e-8*np.eye(40)
    independent = covariance.copy()
    for i in range(5):
        for j in range(5):
            if i != j:
                independent[i*8:(i+1)*8,j*8:(j+1)*8] = 0
    for i in range(5):
        assert np.array_equal(covariance[i*8:(i+1)*8,i*8:(i+1)*8], independent[i*8:(i+1)*8,i*8:(i+1)*8])
    rng = np.random.RandomState(2407)
    base = rng.normal(size=(512,40))
    base = np.concatenate([base,-base])
    noises = {name: (base @ np.linalg.cholesky(cov).T + estimator.location_).reshape(1024,5,4,2)
              for name,cov in [('Joint',covariance),('Independent',independent)]}
    result = meta[test].reset_index(drop=True)
    yt = y[test].reshape(-1,5,4,2)+h0[test,:,None,:]
    mt = mean[test].reshape(-1,5,4,2)+h0[test,:,None,:]
    rt = rf[test]
    true_distance = np.linalg.norm(yt-rt[:,None,:,:], axis=-1).min(axis=(1,2))
    for threshold in [.6, 1.0]:
        label = true_distance < threshold
        result['event_%.1f'%threshold] = label.astype(int)
        for name, noise in noises.items():
            probability = []
            for i in range(len(mt)):
                distances = np.linalg.norm(mt[i][None]+noise-rt[i][None,None], axis=-1).min(axis=(1,2))
                probability.append(float(np.mean(distances < threshold)))
            result['%s_%.1f'%(name,threshold)] = probability
    summary = dict(queries=dict(train=int(train.sum()), calibration=int(calib.sum()), test=int(test.sum())),
                   covariance_shrinkage=float(estimator.shrinkage_), comparisons={})
    for threshold in [.6,1.0]:
        label = result['event_%.1f'%threshold].to_numpy()
        stat = dict(events=int(label.sum()), total=len(label))
        for name in noises:
            p = result['%s_%.1f'%(name,threshold)].to_numpy()
            result['%s_brier_%.1f'%(name,threshold)] = (p-label)**2
            stat[name] = dict(brier=float(np.mean((p-label)**2)), mean_probability=float(p.mean()),
                             scene_equal_brier=float(result.groupby('scene')['%s_brier_%.1f'%(name,threshold)].mean().mean()))
        differences = result.groupby('scene')['Joint_brier_%.1f'%threshold].mean()-result.groupby('scene')['Independent_brier_%.1f'%threshold].mean()
        stat['scene_differences_joint_minus_independent'] = differences.to_dict()
        summary['comparisons'][str(threshold)] = stat
    summary['limitations'] = ['only four test recordings, one group of participants',
                'recorded robot future used equally in scoring, not a legal planned counterfactual',
                'Gaussian residual approximation; failure does not exclude all dependence models',
                'no Bayesian posterior, RL, or navigation claim']
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result.to_csv(out/'test_queries.csv', index=False)
    np.savez_compressed(out/'covariance.npz', covariance=covariance, independent=independent, mean=estimator.location_)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    joint_screen() if '--joint' in sys.argv else main()
