"""Prequential real-trajectory screening; no navigation or novelty claim."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import gammaln
from scipy.stats import chi2, f as f_distribution
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/abc/temp/peroi-full/data')
OUT = Path('/home/abc/temp/paper2_scene_adaptation_20260916')
H = np.array([1., 2.])


def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist())+'\n')


def load():
    features, targets, cv, meta = [], [], [], []
    manifest = {}
    for path in sorted(ROOT.rglob('filtered*.csv')):
        session = str(path.parent.relative_to(ROOT))
        fold = int(hashlib.sha256(path.parent.name.encode()).hexdigest(), 16) % 5
        part = 'test' if fold == 0 else ('val' if fold == 1 else 'train')
        d = pd.read_csv(path)
        manifest[session] = dict(split=part, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for ident, g in d.groupby('Pedestrian_ID', sort=True):
            g = g.sort_values('Frame_Number')
            t = g.Frame_Number.to_numpy(float)/1000.
            p = g[['X_Position', 'Y_Position']].to_numpy(float)
            if len(t)<5 or np.any(np.diff(t)<=0):
                continue
            used = 0
            for desired in np.arange(t[0]+1., t[-1]-2.+1e-6, 1.):
                j = int(np.searchsorted(t, desired))
                if j>=len(t) or t[j]+2>t[-1]:
                    continue
                lo = max(0, np.searchsorted(t, t[j]-1., side='right')-1)
                hi = np.searchsorted(t, t[j]+2.)
                if np.max(np.diff(t[lo:hi+1]))>.3 or not np.isfinite(p[lo:hi+1]).all():
                    continue
                hist = np.column_stack([np.interp(t[j]+np.array([-1., -.5, 0.]), t[:j+1], p[:j+1,k]) for k in range(2)])
                v = (hist[2]-hist[1])/.5
                old = (hist[1]-hist[0])/.5
                speed = np.linalg.norm(v)
                if speed>4. or np.linalg.norm(old)>4.:
                    continue
                u = v/speed if speed>.05 else np.array([1.,0.])
                rot = np.array([u, [-u[1],u[0]]]).T
                vel, prev = v@rot, old@rot
                acc = (vel-prev)/.5
                f = np.r_[vel,prev,acc,np.linalg.norm(prev),speed**2,acc**2,vel*acc]
                future = np.column_stack([np.interp(t[j]+H,t,p[:,k]) for k in range(2)])
                target = ((future-p[j])@rot).reshape(-1)
                baseline = (H[:,None]*vel).reshape(-1)
                features.append(f); targets.append(target); cv.append(baseline)
                meta.append(dict(session=session, track=f'{session}/{ident}', split=part,
                                 time=float(t[j]), label_time=float(t[j]+2.), first=used==0))
                used += 1
                if used>=6:
                    break
    return np.array(features),np.array(targets),np.array(cv),pd.DataFrame(meta),manifest


def adapt(meta, x, residual, base, sigma2, method, parameter):
    pred = base.copy()
    var = np.broadcast_to(sigma2, base.shape).copy()
    audit = []
    for session, group in meta.groupby('session'):
        order = group.sort_values(['time','track']).index.to_numpy()
        # One evidence item per person avoids treating six overlapping futures as iid.
        labels = group[group['first']].sort_values(['label_time','track']).index.to_numpy()
        mu = np.zeros(4); count = 0; latest = -np.inf
        P = np.eye(x.shape[1])/parameter if method=='bayes' else None
        W = np.zeros((x.shape[1],4))
        for i in order:
            while count<len(labels) and meta.loc[labels[count],'label_time'] <= meta.loc[i,'time']:
                k = labels[count]
                if method == 'ewma':
                    mu = (1-parameter)*mu+parameter*residual[k]
                elif method == 'bayes':
                    z = x[k]; px = P@z; gain = px/(1+z@px)
                    W += np.outer(gain, residual[k]-z@W)
                    P -= np.outer(gain, px)
                    P = (P+P.T)/2
                count += 1
                latest = meta.loc[k,'label_time']
            assert latest <= meta.loc[i,'time']
            if method == 'ewma':
                pred[i] += mu
            elif method == 'bayes':
                pred[i] += x[i]@W
                var[i] *= 1+max(0.,x[i]@P@x[i])
        audit.append(dict(session=session,unique_evidence=count,queries=len(order)))
    return pred,var,audit


def metrics(meta, y, pred, var):
    err = y-pred
    d = pd.DataFrame(dict(session=meta.session,track=meta.track,
        ade2=np.linalg.norm(err[:,2:],axis=1),
        nll=.5*np.sum(np.log(2*np.pi*var)+err**2/var,axis=1),
        coverage90=np.mean(np.abs(err)<=norm.ppf(.95)*np.sqrt(var),axis=1),
        width90=np.mean(2*norm.ppf(.95)*np.sqrt(var),axis=1)))
    return d.groupby(['session','track']).mean(numeric_only=True).groupby('session').mean()


def main():
    OUT.mkdir(exist_ok=False)
    dump('protocol.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base='global Ridge residual to CV, train only', horizons=H,
        global_alpha=[1.,10.,100.], ewma_alpha=[.01,.05,.2], bayes_precision=[10.,100.,1000.],
        selection='validation recording-equal Gaussian NLL for adaptive models; ADE2 for global ridge',
        evidence='first eligible query per track, admitted only after 2s outcome is observable',
        limitation='processed tracks; exploratory reused public data; no robot counterfactual'))
    start=time.time()
    X,Y,CV,meta,manifest=load()
    dump('data_manifest.json',manifest)
    print('population',meta.groupby('split').size().to_dict(),flush=True)
    print('sessions',meta.groupby('split').session.nunique().to_dict(),flush=True)
    masks={k:meta.split.to_numpy()==k for k in ('train','val','test')}
    assert all(v.any() for v in masks.values())
    scaler=StandardScaler().fit(X[masks['train']]); SX=scaler.transform(X)
    SX=np.clip(SX,-10,10)
    best=None
    for alpha in (1.,10.,100.):
        reg=Ridge(alpha=alpha).fit(SX[masks['train']], (Y-CV)[masks['train']])
        base=CV+reg.predict(SX)
        score=metrics(meta,Y,base,np.ones_like(Y)).loc[meta[meta.split=='val'].session.unique()].ade2.mean()
        if best is None or score<best[0]: best=(score,alpha,base)
    base=best[2]; residual=Y-base
    sigma2=np.maximum(np.mean(residual[masks['train']]**2,axis=0),.05**2)
    # Small shared linear adapter: intercept, speed and recent longitudinal/lateral acceleration.
    x=np.c_[np.ones(len(X)),SX[:,[0,4,5]]]
    arrays={'global':(base,np.broadcast_to(sigma2,Y.shape)), 'cv':(CV,np.broadcast_to(sigma2,Y.shape))}
    selected={}; audits={}
    for method,grid in [('ewma',(.01,.05,.2)),('bayes',(10.,100.,1000.))]:
        best_adapt=None
        for par in grid:
            pred,var,audit=adapt(meta,x,residual,base,sigma2,method,par)
            m=metrics(meta,Y,pred,var)
            value=m.loc[meta[meta.split=='val'].session.unique()].nll.mean()
            if best_adapt is None or value<best_adapt[0]: best_adapt=(value,par,pred,var,audit)
        selected[method]=dict(parameter=best_adapt[1],validation_nll=float(best_adapt[0]))
        arrays[method]=best_adapt[2:4]; audits[method]=best_adapt[4]
    # Exact same Bayesian mean with point-estimate predictive variance is a necessary control.
    arrays['bayes_point']=(arrays['bayes'][0],np.broadcast_to(sigma2,Y.shape))
    result={}; tables={}
    te=meta[meta.split=='test'].session.unique()
    for name,(pred,var) in arrays.items():
        table=metrics(meta,Y,pred,var).loc[te]; tables[name]=table
        result[name]=dict(mean=table.mean().to_dict(),recordings=table.to_dict('index'))
    rng=np.random.default_rng(20260916); comparisons={}
    for other in ('global','ewma','bayes_point'):
        comparisons[other]={}
        for field in ('ade2','nll'):
            delta=(tables['bayes'][field]-tables[other][field]).to_numpy()
            draws=rng.choice(delta,(10000,len(delta))).mean(axis=1)
            comparisons[other][field]=dict(mean=float(delta.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
    dump('summary.json',dict(population=meta.groupby('split').size().to_dict(),
        recordings=meta.groupby('split').session.nunique().to_dict(),global_alpha=best[1],
        selected=selected,results=result,comparisons=comparisons,audits=audits,seconds=time.time()-start))
    meta.to_csv(OUT/'queries.csv',index=False)
    np.savez_compressed(OUT/'predictions.npz',target=Y,**{f'{k}_{s}':v[j] for k,v in arrays.items() for j,s in enumerate(('mean','variance'))})
    print(json.dumps(dict(selected=selected,results={k:v['mean'] for k,v in result.items()},comparisons=comparisons),indent=2),flush=True)


def covariance_main():
    directory=OUT/'covariance'; directory.mkdir(exist_ok=False)
    def save(name,value):
        (directory/name).write_text(json.dumps(value,indent=2,default=lambda x:x.tolist())+'\n')
    save('protocol.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        population='same fixed PeRoI queries and recording splits as scene adaptation',
        mean='frozen global ridge; 2 second error in past velocity frame',
        grids=dict(ewma=[.01,.05,.2],iw=[5,20,100],global_t=[3,5,10,30]),
        observation='first query per track, evidence available only after label_time',
        metrics=['NLL','90% ellipse coverage','ellipse area'],cold_start='at most 20 matured tracks',
        gate='Bayes must reliably beat EWMA and fixed Student-t before decision tests',
        limitation='zero-mean residual and shared stationary covariance working model, not sensor truth'))
    meta=pd.read_csv(OUT/'queries.csv'); data=np.load(OUT/'predictions.npz')
    err=data['target'][:,2:]-data['global_mean'][:,2:]
    train=meta.split.to_numpy()=='train'; C=err[train].T@err[train]/train.sum()+np.eye(2)*1e-6
    total=len(err); evidence=np.zeros(total,int); counts={}
    def distributions(kind,parameter):
        cov=np.broadcast_to(C,(total,2,2)).copy(); dfs=np.full(total,np.inf)
        for session,group in meta.groupby('session'):
            order=group.sort_values(['time','track']).index.to_numpy()
            labels=group[group['first']].sort_values(['label_time','track']).index.to_numpy()
            at=0; scatter=parameter*C.copy(); state=C.copy(); latest=-np.inf
            for i in order:
                while at<len(labels) and meta.loc[labels[at],'label_time']<=meta.loc[i,'time']:
                    k=labels[at]; outer=np.outer(err[k],err[k])
                    if kind=='ewma': state=(1-parameter)*state+parameter*outer
                    else: scatter+=outer
                    latest=meta.loc[k,'label_time']; at+=1
                assert latest<=meta.loc[i,'time']; evidence[i]=at
                if kind=='ewma':cov[i]=state+np.eye(2)*1e-6
                else:
                    # IW(nu=parameter+3+n, Psi), zero-mean 2D predictive Student-t.
                    cov[i]=scatter/(parameter+at)
                    dfs[i]=parameter+at+2
            counts[session]=at
        return cov,dfs
    def table(cov,dfs,mask=None):
        inv=np.linalg.inv(cov); logdet=np.linalg.slogdet(cov)[1]
        mahal=np.einsum('ni,nij,nj->n',err,inv,err)
        nll=np.log(2*np.pi)+.5*logdet+.5*mahal
        threshold=np.full(total,chi2.ppf(.9,2))
        finite=np.isfinite(dfs)
        if finite.any():
            df=dfs[finite]; ratio=(df-2)/df; m=mahal[finite]/ratio
            nll[finite]=(-gammaln((df+2)/2)+gammaln(df/2)+np.log(df*np.pi)
                +.5*logdet[finite]+np.log(ratio)+(df+2)/2*np.log1p(m/df))
            threshold[finite]=2*f_distribution.ppf(.9,2,df)*ratio
        d=pd.DataFrame(dict(session=meta.session,track=meta.track,nll=nll,
            coverage90=(mahal<=threshold).astype(float),area90=np.pi*threshold*np.exp(.5*logdet)))
        if mask is not None:d=d[mask]
        return d.groupby(['session','track']).mean(numeric_only=True).groupby('session').mean()
    arrays={'global_full':(np.broadcast_to(C,(total,2,2)),np.full(total,np.inf)),
            'global_isotropic':(np.broadcast_to(np.eye(2)*np.trace(C)/2,(total,2,2)),np.full(total,np.inf))}
    selected={}; val=meta[meta.split=='val'].session.unique()
    for kind,grid in [('global_t',[3,5,10,30]),('ewma',[.01,.05,.2]),('iw',[5,20,100])]:
        best=None
        for parameter in grid:
            if kind=='global_t':arr=(np.broadcast_to(C,(total,2,2)),np.full(total,float(parameter)))
            else:arr=distributions(kind,parameter)
            score=float(table(*arr).loc[val].nll.mean())
            if best is None or score<best[0]:best=(score,parameter,arr)
        selected[kind]=dict(parameter=best[1],val_nll=best[0]);arrays[kind]=best[2]
    arrays['iw_point']=(arrays['iw'][0],np.full(total,np.inf))
    te=meta[meta.split=='test'].session.unique(); tables={}; summary={}
    for subset,mask in [('all',None),('cold',evidence<=20)]:
        ts={name:table(*arr,mask).reindex(te).dropna() for name,arr in arrays.items()}
        tables[subset]=ts; rng=np.random.default_rng(2407); comparisons={}
        for other in ('global_isotropic','global_full','global_t','ewma','iw_point'):
            delta=(ts['iw'].nll-ts[other].nll).dropna().to_numpy()
            comparisons[other]=dict(mean=float(delta.mean()),ci95=np.quantile(rng.choice(delta,(10000,len(delta))).mean(1),[.025,.975]))
        summary[subset]=dict(means={k:v.mean().to_dict() for k,v in ts.items()},comparisons=comparisons,
            recordings={k:v.to_dict('index') for k,v in ts.items()})
    save('summary.json',dict(selected=selected,training_covariance=C,
        eigenvalue_ratio=float(np.linalg.eigvalsh(C)[-1]/np.linalg.eigvalsh(C)[0]),
        results=summary,evidence=counts,cold_test_queries=int(((evidence<=20)&(meta.split.to_numpy()=='test')).sum())))
    np.savez_compressed(directory/'distributions.npz',**{name+'_cov':v[0] for name,v in arrays.items()},
        iw_df=arrays['iw'][1],evidence=evidence)
    print(json.dumps(dict(selected=selected,results={k:dict(means=v['means'],comparisons=v['comparisons']) for k,v in summary.items()}),indent=2,default=lambda x:x.tolist()),flush=True)


if __name__=='__main__':
    import sys
    covariance_main() if '--covariance' in sys.argv else main()
