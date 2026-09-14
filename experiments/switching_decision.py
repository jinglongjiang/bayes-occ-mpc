"""Frozen forecast risk on factual robot paths; no counterfactual human claims."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ncx2
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments import switching_validation as source
from experiments.switching_validation import ROOT,H,old,Bank

OUT=ROOT/'results/switching_decision'
PREVIOUS=ROOT/'results/switching_validation'
METHODS=('CV','TREE','MAP','IID','FULL')


def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist())+'\n')


def population(root):
    old.H=H;old.LONG_CONTEXT=True;old.OUT=OUT/'data'
    X,Y,CV,CA,meta=old.build(root)
    seqs=source.load_sequences(root,meta)
    return X,Y,CV,CA,meta,seqs


def trees(X,Y,indices,weights,seed,k,leaf):
    return [ExtraTreesRegressor(n_estimators=96//k,min_samples_leaf=leaf,max_depth=16,
        random_state=seed+j,n_jobs=1).fit(X[indices],Y[indices].reshape(len(indices),-1),
        sample_weight=weights[indices,j]) for j in range(k)]


def bankfit(seqs,meta,tr,k,seed):
    bank=Bank(k,seed).fit(seqs,meta.iloc[tr].track.unique())
    e=np.concatenate([seqs[t]['e'] for t in meta.iloc[tr].track.unique()])
    bank.residual_reference=float(np.quantile(np.linalg.norm(e,axis=1),.75))
    p,iid,_=source.query_states(bank,seqs,meta)
    return bank,p,iid


def fit_deployment(root):
    X,Y,CV,CA,meta,seqs=population(root)
    idx=np.arange(len(meta));bank,p,iid=bankfit(seqs,meta,idx,6,11)
    experts=trees(X,Y-CV,idx,p,11,6,15)
    pooled=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=15,max_depth=16,
        random_state=11,n_jobs=1).fit(X,(Y-CV).reshape(len(meta),-1))
    configs=json.loads((PREVIOUS/'selected.json').read_text())
    variance=np.median([c['sigma'] for c in configs if c['seed']==11 and c['fold']!='moving_temporal'],axis=0)
    artifact=dict(bank=bank,experts=experts,pooled=pooled,variance=variance)
    joblib.dump(artifact,OUT/'deployment.joblib',compress=3)
    save('deployment.json',dict(k=6,seed=11,total_trees=96,leaf=15,training_queries=len(meta),
        training_tracks=meta.track.nunique(),variance=variance,
        sha256=hashlib.sha256((OUT/'deployment.joblib').read_bytes()).hexdigest(),
        no_simulator_training=True,platform_feature='zeros: unseen robot platform'))


def actual_robot_paths(root,meta,X):
    robot=np.full((len(meta),4,2),np.nan)
    grouped={t:g for t,g in meta.groupby('track')}
    for path in sorted(root.rglob('filtered*.csv')):
        if 'no_robot' in str(path):continue
        d=pd.read_csv(path)
        for tid,track in d.groupby('Pedestrian_ID'):
            key=f'{path.parent.name}/{tid}'
            if key not in grouped:continue
            track=track.sort_values('Frame_Number')
            t=track.Frame_Number.values/1000.;p=track[['X_Position','Y_Position']].values
            r=track[['X_Robot','Y_Robot']].values
            for idx,row in grouped[key].iterrows():
                j=np.searchsorted(t,row.time-1e-8)
                hi=np.searchsorted(t,row.time+4.)
                if hi>=len(t) or not np.isfinite(r[j:hi+1]).all():continue
                lo=max(0,np.searchsorted(t,t[j]-1.,side='right')-1)
                _,rot,_,_,_,_=old.prefix_features(t[lo:j+1],p[lo:j+1],r[lo:j+1])
                np.testing.assert_allclose((r[j]-p[j])@rot,X[idx,18:20],atol=1e-8)
                robot[idx]=(old.interpolate(t,r,row.time+H)-p[j])@rot
    return robot


def occupancy(modes,w,robot,sigma,radius):
    d2=np.sum((modes-robot[:,None])**2,axis=-1)
    q=ncx2.cdf(radius**2/sigma[None,None,:],2,d2/sigma[None,None,:])
    return np.sum(w[:,:,None]*q,axis=1)


def alarm(y,q,threshold):
    return dict(queries=len(y),events=int(y.sum()),
        auc=float(roc_auc_score(y,q)) if len(np.unique(y))==2 else None,
        ap=float(average_precision_score(y,q)) if y.any() else None,
        fpr=float((q[~y]>threshold).mean()) if (~y).any() else None,
        recall=float((q[y]>threshold).mean()) if y.any() else None,
        threshold=float(threshold),false_alarms=int(((q>threshold)&~y).sum()),
        missed=int(((q<=threshold)&y).sum()))


def factual(root):
    X,Y,CV,CA,meta,seqs=population(root)
    robot=actual_robot_paths(root,meta,X)
    valid=np.isfinite(robot).all(axis=(1,2))
    configs=json.loads((PREVIOUS/'selected.json').read_text())
    nonlinear=json.loads((PREVIOUS/'nonlinear/selected.json').read_text())
    protocol=json.loads((ROOT/'results/response_validation/long_context/prediction_protocol.json').read_text())
    previous=pd.read_csv(PREVIOUS/'nonlinear/queries.csv.gz')
    rows=[];metrics=[];checks=[]
    for fold in protocol['splits']:
        tr,va,allte=[np.flatnonzero(meta.track.isin(fold[k+'_tracks'])) for k in ('train','val','test')]
        te=allte[valid[allte]];va=va[valid[va]]
        for seed in source.SEEDS:
            cfg=next(c for c in configs if c['fold']==fold['name'] and c['seed']==seed)
            nn=next(c for c in nonlinear if c['fold']==fold['name'] and c['seed']==seed)
            bank,p,iid=bankfit(seqs,meta,tr,cfg['k'],seed)
            np.testing.assert_allclose(bank.model.means_,cfg['means'],atol=1e-12)
            models=trees(X,Y-CV,tr,p,seed,cfg['k'],nn['leaf'])
            base=old.regressor('trees',cfg['universal']['trees']['alpha_or_leaf']).fit(X[tr],(Y-CV)[tr].reshape(len(tr),-1))
            allm=source.expert_predict(models,X[allte],CV[allte])
            err=np.linalg.norm((allm*p[allte,:,None,None]).sum(1)-Y[allte],axis=-1).mean(1)
            saved=previous[(previous.fold==fold['name'])&(previous.seed==seed)]
            np.testing.assert_allclose(err,saved.learned_full,rtol=1e-10,atol=1e-12)
            checks.append(dict(fold=fold['name'],seed=seed,max_error=float(np.max(np.abs(err-saved.learned_full.values)))))
            inputs={}
            for label,idx in [('val',va),('test',te)]:
                modes=source.expert_predict(models,X[idx],CV[idx])
                pooled=base.predict(X[idx]).reshape(-1,4,2)+CV[idx]
                inputs[label]=dict(CV=(CV[idx,None],np.ones((len(idx),1))),
                    TREE=(pooled[:,None],np.ones((len(idx),1))),
                    MAP=(modes[np.arange(len(idx)),p[idx].argmax(1)][:,None],np.ones((len(idx),1))),
                    IID=(modes,iid[idx]),FULL=(modes,p[idx]))
            for radius in (.7,1.):
                truth=np.linalg.norm(Y[te]-robot[te],axis=-1)<radius
                valtruth=np.linalg.norm(Y[va]-robot[va],axis=-1)<radius
                for method in METHODS:
                    q=occupancy(*inputs['test'][method],robot[te],np.array(cfg['sigma']),radius)
                    qv=occupancy(*inputs['val'][method],robot[va],np.array(cfg['sigma']),radius)
                    # Maximum marginal probability is an alarm score, not a union probability.
                    neg=~valtruth.any(1)
                    threshold=float(np.quantile(qv.max(1)[neg],.95))
                    metrics.append(dict(fold=fold['name'],seed=seed,method=method,radius=radius,
                        **alarm(truth.any(1),q.max(1),threshold)))
                    for j,idx in enumerate(te):
                        row=dict(fold=fold['name'],seed=seed,method=method,radius=radius,
                            track=meta.iloc[idx].track,time=float(meta.iloc[idx].time),
                            moving=bool(meta.iloc[idx].moving),event=bool(truth[j].any()),
                            alarm_score=float(q[j].max()),alarmed=bool(q[j].max()>threshold),
                            brier=float(np.mean((q[j]-truth[j])**2)))
                        for k,h in enumerate(H):row[f'q_{h}']=float(q[j,k]);row[f'y_{h}']=int(truth[j,k])
                        rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/'factual_queries.csv',index=False)
            save('factual_metrics.json',metrics);save('forecast_reproduction.json',checks)
            print('FACTUAL',fold['name'],seed,'queries',len(te),flush=True)
    save('factual_population.json',dict(original=len(meta),valid=int(valid.sum()),excluded=int((~valid).sum()),
        valid_moving=int((valid&meta.moving.values).sum()),test_seed_runs=len(checks)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['factual','fit'])
    parser.add_argument('root',type=Path);args=parser.parse_args()
    factual(args.root) if args.command=='factual' else fit_deployment(args.root)
