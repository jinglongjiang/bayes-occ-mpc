"""Prefix-only processed-trajectory diagnostics, not a causal response simulator."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, log_loss, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/response_validation'
LABELS = ['attract', 'avoid', 'neutral']
H = np.array([.5, 1., 2.])


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, default=lambda x:x.tolist())+'\n')


def interpolate(t, p, q):
    return np.column_stack([np.interp(q, t, p[:,j]) for j in range(p.shape[1])])


def prefix_features(t, p, r):
    """All arguments end at the query; inaccessible future cannot affect features."""
    q = t[-1]+np.arange(-1., .001, .25)
    hist = interpolate(t, p, q)
    robot = interpolate(t, r, q)
    velocity = (hist[-1]-hist[-3])/.5
    old_velocity = (hist[-3]-hist[0])/.5
    speed = np.linalg.norm(velocity)
    direction = velocity/max(speed, 1e-9) if speed > .05 else np.array([1.,0.])
    rot = np.array([direction, [-direction[1],direction[0]]]).T
    human_local = (hist-hist[-1]) @ rot
    relative_robot = (robot-hist[-1]) @ rot
    rv = (robot[-1]-robot[-3])/.5
    acc = (velocity-old_velocity)/.5
    acc *= min(1., 2./max(np.linalg.norm(acc), 1e-9))
    rel = hist[-1]-robot[-1]
    distance = np.linalg.norm(rel)
    closing = -np.dot(rel,velocity-rv)/max(distance,1e-9)
    feature = np.r_[human_local.ravel(), relative_robot.ravel(),
                    velocity@rot, old_velocity@rot, rv@rot,
                    distance, closing]
    cv = H[:,None]*(velocity@rot)
    ca = cv+.5*H[:,None]**2*(acc@rot)
    return feature, rot, cv, ca, distance, closing


def build(root):
    records, X, Y, CV, CA = [], [], [], [], []
    counts = {}
    files = sorted(p for p in root.rglob('filtered*.csv') if 'no_robot' not in str(p))
    for path in files:
        d = pd.read_csv(path)
        session = path.parent.name
        info = dict(tracks=int(d.Pedestrian_ID.nunique()), eligible_tracks=0,
                    queries=0, rejected_gaps=0, rejected_nonfinite=0,
                    rejected_kinematics=0)
        for ident,g in d.groupby('Pedestrian_ID', sort=True):
            g = g.sort_values('Frame_Number')
            t = g.Frame_Number.values.astype(float)/1000
            p = g[['X_Position','Y_Position']].values
            r = g[['X_Robot','Y_Robot']].values
            if len(t)<5 or np.any(np.diff(t)<=0):
                continue
            labels = g.Robot_Influence.unique()
            if len(labels)!=1 or labels[0] not in LABELS:
                continue
            present = np.isfinite(r).all(axis=1)
            distances = np.where(present, np.linalg.norm(p-r,axis=1), np.inf)
            closest = t[np.argmin(distances)]
            query_times = np.arange(t[0]+1., t[-1]-2.+1e-6, 1.)
            used = 0
            for desired in query_times:
                j = np.searchsorted(t, desired)
                if j>=len(t) or t[j]+2>t[-1]:
                    continue
                lo = max(0, np.searchsorted(t,t[j]-1.,side='right')-1)
                hi = np.searchsorted(t,t[j]+2.,side='left')
                if np.max(np.diff(t[lo:hi+1])) > .3:
                    info['rejected_gaps'] += 1
                    continue
                if not np.isfinite(r[lo:j+1]).all() or not np.isfinite(p[lo:hi+1]).all():
                    info['rejected_nonfinite'] += 1
                    continue
                f,rot,cv,ca,dist,closing = prefix_features(t[lo:j+1],p[lo:j+1],r[lo:j+1])
                if dist>6:
                    continue
                if np.linalg.norm(cv[1])>4 or np.linalg.norm(f[24:26])>4:
                    info['rejected_kinematics'] += 1
                    continue
                # No final goal, actual closest time, session ID or label in features.
                kind = str(g.Robot_Type.iloc[0]).lower()
                platform = [float(k in kind) for k in ('go','hsr','mpo')]
                f = np.r_[f,platform]
                y = (interpolate(t,p,t[j]+H)-p[j])@rot
                X.append(f); Y.append(y); CV.append(cv); CA.append(ca)
                records.append(dict(session=session,track=f'{session}/{ident}',
                    label=LABELS.index(labels[0]),time=float(t[j]),
                    track_start=float(t[0]),track_end=float(t[-1]),
                    distance=float(dist),closing=float(closing),
                    lead_to_closest=float(closest-t[j]),
                    moving='dynamic_robot' in str(path)))
                used += 1
                if used>=10:
                    break
            info['eligible_tracks'] += int(used>0)
            info['queries'] += used
        counts[session] = info
    print('DATA',json.dumps(counts),flush=True)
    save('query_population.json',counts)
    return np.array(X),np.array(Y),np.array(CV),np.array(CA),pd.DataFrame(records)


def conditional_features(X, labels):
    onehot = np.eye(3)[labels]
    return np.c_[X,onehot,(X[:,:,None]*onehot[:,None,:]).reshape(len(X),-1)]


def regressor(family, parameter):
    if family=='ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=parameter))
    return ExtraTreesRegressor(n_estimators=96, min_samples_leaf=parameter,
                               max_depth=16,random_state=20260914,n_jobs=4)


def choose_reg(X,Y,train,val,family):
    grid = [1.,10.,100.] if family=='ridge' else [5,15]
    best = None
    for param in grid:
        m = regressor(family,param).fit(X[train],Y[train].reshape(len(train),-1))
        pred = m.predict(X[val]).reshape(-1,3,2)
        score = float(np.linalg.norm(pred-Y[val],axis=2).mean())
        if best is None or score<best[0]:
            best = score,param,m
    return best[2],dict(parameter=best[1],validation_error=best[0])


def calibrated_prob(model, X, temperature):
    p = np.full((len(X),3),1e-5)
    p[:,model.classes_.astype(int)] = model.predict_proba(X)
    return softmax(np.log(np.maximum(p,1e-5))/temperature,axis=1)


def choose_classifier(X,labels,train,val):
    models = [make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000)),
              ExtraTreesClassifier(n_estimators=128,min_samples_leaf=5,max_depth=16,
                                   random_state=20260914,n_jobs=4)]
    best = None
    for model in models:
        model.fit(X[train],labels[train])
        for temp in [.5,1.,2.,4.]:
            p = calibrated_prob(model,X[val],temp)
            score = log_loss(labels[val],p,labels=[0,1,2])
            if best is None or score<best[0]:
                best = score,temp,model
    return best[2],best[1],dict(validation_nll=best[0],temperature=best[1],
                              family=type(best[2]).__name__)


def energy(modes,weights,target):
    # Exact energy score of this finite predictive distribution, no best-of-K.
    first = np.sum(weights[:,:,None]*np.linalg.norm(modes-target[:,None,:,:],axis=-1),axis=1)
    pair = np.linalg.norm(modes[:,:,None,:,:]-modes[:,None,:,:,:],axis=-1)
    second = .5*np.sum(weights[:,:,None,None]*weights[:,None,:,None]*pair,axis=(1,2))
    return first-second


def metrics(y,p):
    return dict(n=len(y),counts=np.bincount(y,minlength=3).tolist(),
                accuracy=float((p.argmax(1)==y).mean()),
                balanced_accuracy=float(balanced_accuracy_score(y,p.argmax(1))),
                nll=float(log_loss(y,p,labels=[0,1,2])),
                brier=float(np.square(p-np.eye(3)[y]).sum(1).mean()),
                ambiguous_max_below_08=float((p.max(1)<.8).mean()),
                confusion=confusion_matrix(y,p.argmax(1),labels=[0,1,2]).tolist())


def bootstrap_difference(frame,a,b,cluster):
    means = frame.assign(diff=frame[a]-frame[b]).groupby(cluster)['diff'].mean().values
    rng = np.random.default_rng(20260914)
    draws = np.mean(rng.choice(means,(2000,len(means)),replace=True),axis=1)
    return dict(clusters=len(means),mean=float(means.mean()),
                interval95=np.quantile(draws,[.025,.975]).tolist())


def run(root):
    X,Y,CV,CA,meta = build(root)
    assert len(X)>0 and np.isfinite(X).all() and np.isfinite(Y).all()
    labels = meta.label.values
    sessions = sorted(meta.session.unique())
    splits = []
    for i,session in enumerate(sessions):
        test = np.flatnonzero(meta.session==session)
        val = np.flatnonzero(meta.session==sessions[(i+1)%len(sessions)])
        train = np.flatnonzero(~meta.session.isin([session,sessions[(i+1)%len(sessions)]]))
        splits.append((session,train,val,test))
    # Supplementary same-moving-record temporal split, purged by whole track intervals.
    moving = meta[meta.moving]
    start,end = moving.track_start.min(),moving.track_end.max()
    cut1,cut2 = start+.6*(end-start),start+.8*(end-start)
    train = np.flatnonzero(meta.moving & (meta.track_end<cut1-4))
    val = np.flatnonzero(meta.moving & (meta.track_start>cut1+4) & (meta.track_end<cut2-4))
    test = np.flatnonzero(meta.moving & (meta.track_start>cut2+4))
    if min(len(train),len(val),len(test))>=15 and len(np.unique(labels[train]))==3:
        splits.append(('moving_temporal',train,val,test))
    protocol = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Processed positions, prefix-only features; not raw-sensor online validation or causal robot-action response identification',
        horizons=H.tolist(), sample='Every 1s after 1s history, 2s future available, within 6m, first at most 10 per track; gaps<=.3s',
        splits=[dict(name=s,train_tracks=sorted(meta.iloc[tr].track.unique()),
                      val_tracks=sorted(meta.iloc[va].track.unique()),
                      test_tracks=sorted(meta.iloc[te].track.unique())) for s,tr,va,te in splits],
        selection='Validation only; ridge alpha1/10/100, ET leaves5/15; classifier LR/ET temperature .5/1/2/4',
        limitations='Modes learned from factual motion; changing candidate robot actions is unsupported. Mixture ES is a finite point distribution, not calibrated residual uncertainty.')
    path = OUT/'prediction_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise RuntimeError('Frozen prediction protocol changed')
    save('prediction_protocol.json',protocol)
    all_rows, details, classifications = [], [], []
    for name,tr,va,te in splits:
        assert not (set(meta.iloc[tr].track)&set(meta.iloc[te].track))
        classifier,temp,cfg = choose_classifier(X,labels,tr,va)
        probs = calibrated_prob(classifier,X[te],temp)
        prior = np.bincount(labels[tr],minlength=3)+1
        prior = prior/prior.sum()
        class_result = dict(fold=name,selected=cfg,test=metrics(labels[te],probs),
                            prior=metrics(labels[te],np.tile(prior,(len(te),1))))
        approaching = meta.iloc[te].closing.values>0
        if approaching.any():
            class_result['approaching'] = metrics(labels[te][approaching],probs[approaching])
        classifications.append(class_result)
        print('CLASS',name,json.dumps(class_result),flush=True)
        for family in ['ridge','trees']:
            # Same residual target; true-label arm gets explicit extra diagnostic information.
            base,base_cfg = choose_reg(X,Y-CV,tr,va,family)
            CX = conditional_features(X,labels)
            conditional,cond_cfg = choose_reg(CX,Y-CV,tr,va,family)
            modes = np.stack([conditional.predict(conditional_features(X[te],np.full(len(te),z))).reshape(-1,3,2)+CV[te]
                              for z in range(3)],axis=1)
            prediction = dict(cv=CV[te],ca=CA[te],pooled=base.predict(X[te]).reshape(-1,3,2)+CV[te],
                              true_label=modes[np.arange(len(te)),labels[te]],
                              map=modes[np.arange(len(te)),probs.argmax(1)],
                              full_mean=np.sum(probs[:,:,None,None]*modes,axis=1),
                              prior_mean=np.sum(prior[None,:,None,None]*modes,axis=1))
            score = energy(modes,probs,Y[te])
            # Shuffled whole-track labels: negative control of the oracle-label channel.
            rng = np.random.default_rng(20260914)
            tracks = meta.track.unique()
            original = meta.groupby('track').label.first()
            shuffled = dict(zip(tracks,rng.permutation(original.reindex(tracks).values)))
            sl = meta.track.map(shuffled).values
            shuffle_model = regressor(family,cond_cfg['parameter']).fit(
                conditional_features(X[tr],sl[tr]),(Y-CV)[tr].reshape(len(tr),-1))
            prediction['shuffled_label'] = shuffle_model.predict(conditional_features(X[te],sl[te])).reshape(-1,3,2)+CV[te]
            errors = {k:np.linalg.norm(v-Y[te],axis=-1) for k,v in prediction.items()}
            for local,idx in enumerate(te):
                row = meta.iloc[idx].to_dict()
                row.update(fold=name,family=family)
                row.update({f'{k}_{h}':float(v[local,j]) for k,v in errors.items() for j,h in enumerate(H)})
                row.update({k:float(v[local].mean()) for k,v in errors.items()})
                row['full_finite_energy'] = float(score[local].mean())
                row['mode_spread_2s'] = float(np.max(np.linalg.norm(modes[local,:,None,-1]-modes[local,None,:,-1],axis=-1)))
                row['max_probability'] = float(probs[local].max())
                all_rows.append(row)
            details.append(dict(fold=name,family=family,base=base_cfg,conditional=cond_cfg,
                test_queries=len(te),mean_errors={k:v.mean(0).tolist() for k,v in errors.items()},
                finite_energy=float(score.mean())))
            print('PRED',name,family,json.dumps(details[-1]),flush=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(OUT/'prediction_queries.csv',index=False)
    save('classification.json',classifications)
    save('prediction_folds.json',details)
    results = []
    for family in ['ridge','trees']:
        for subset in ['all_loso','moving_loso','moving_temporal','approaching_loso']:
            f = frame[frame.family==family]
            if subset=='all_loso': f=f[f.fold!='moving_temporal']
            elif subset=='moving_loso': f=f[(f.fold!='moving_temporal')&f.moving]
            elif subset=='moving_temporal': f=f[f.fold=='moving_temporal']
            else: f=f[(f.fold!='moving_temporal')&(f.closing>0)]
            if not len(f): continue
            methods = ['cv','ca','pooled','true_label','map','full_mean','prior_mean','shuffled_label','full_finite_energy']
            results.append(dict(family=family,subset=subset,queries=len(f),tracks=int(f.track.nunique()),
                sessions=int(f.session.nunique()),
                track_equal_mean=f.groupby('track')[methods].mean().mean().to_dict(),
                paired={f'{a}-minus-{b}':bootstrap_difference(f,a,b,'track') for a,b in
                        [('true_label','pooled'),('true_label','cv'),('full_mean','map'),('full_mean','pooled')]},
                session_paired=bootstrap_difference(f,'true_label','pooled','session')))
    save('prediction_summary.json',results)
    print('COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    run(args.root)
