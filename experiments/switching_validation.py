"""Causal finite-regime diagnostics on frozen PeRoI queries, not a navigation claim."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.stats import ncx2
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from experiments import peroi_response_validation as old

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/switching_validation'
H = np.array([.5, 1., 2., 4.])
SEEDS = (11, 29, 47)


def save(name, value):
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT/name).write_text(json.dumps(value, indent=2, default=lambda x:x.tolist())+'\n')


def normalize_update(prior, loglike):
    logjoint = np.log(np.maximum(prior, 1e-300))+loglike
    evidence = logsumexp(logjoint)
    return np.exp(logjoint-evidence), float(evidence)


def ct_path(v, earlier_v, horizons):
    speed = np.linalg.norm(v)
    angle = np.arctan2(np.cross(earlier_v, v), np.dot(earlier_v, v))
    omega = np.clip(angle/.5, -1.5, 1.5) if min(speed,np.linalg.norm(earlier_v))>.05 else 0.
    if abs(omega)<1e-7:
        return horizons[:,None]*v
    turn = np.array([-v[1],v[0]])
    return (np.sin(omega*horizons)[:,None]*v+
            (1-np.cos(omega*horizons))[:,None]*turn)/omega


def sequence(t, p, until):
    """Only interpolate inside an already observed prefix; each displacement used once."""
    keep = t<=until+1e-8
    t,p = t[keep],p[keep]
    items = []
    indices=np.unique(np.searchsorted(t,np.arange(t[0]+1.,t[-1]+1e-8,.25)))
    for left,right in zip(indices[:-1],indices[1:]):
        if right>=len(t): continue
        start,end=t[left],t[right]
        lo = max(0,np.searchsorted(t,start-1.,side='right')-1)
        if np.max(np.diff(t[lo:right+1]))>.3:
            continue
        pts = old.interpolate(t[:left+1],p[:left+1],start+np.array([-1.,-.5,0.]))
        v = (pts[2]-pts[1])/.5
        v_old = (pts[1]-pts[0])/.5
        dt=end-start
        actual = (p[right]-p[left])/dt
        direction = v/max(np.linalg.norm(v),1e-9) if np.linalg.norm(v)>.05 else np.array([1.,0.])
        rot = np.array([direction,[-direction[1],direction[0]]]).T
        innovation = (actual-v)@rot
        acc = (v-v_old)/.5
        acc *= min(1.,2./max(np.linalg.norm(acc),1e-9))
        forecasts = np.array([v, v+.5*acc*dt, ct_path(v,v_old,np.array([dt]))[0]/dt])
        residuals = (actual-forecasts)@rot
        items.append((end,innovation,residuals))
    if not items:
        return dict(t=np.empty(0),e=np.empty((0,2)),r=np.empty((0,3,2)))
    return dict(t=np.array([a[0] for a in items]),e=np.array([a[1] for a in items]),
                r=np.array([a[2] for a in items]))


def load_sequences(root,meta):
    wanted = meta.groupby('track').time.max().to_dict()
    result = {}
    for path in sorted(root.rglob('filtered*.csv')):
        if 'no_robot' in str(path): continue
        d = pd.read_csv(path)
        for ident,g in d.groupby('Pedestrian_ID',sort=True):
            key = f'{path.parent.name}/{ident}'
            if key not in wanted: continue
            g = g.sort_values('Frame_Number')
            result[key] = sequence(g.Frame_Number.values/1000.,g[['X_Position','Y_Position']].values,wanted[key])
    assert set(result)==set(wanted)
    return result


class Bank:
    def __init__(self,k,seed):
        self.k=k
        self.model=GaussianMixture(k,covariance_type='full',reg_covar=1e-4,
                                   n_init=2,max_iter=200,random_state=seed)

    def fit(self,seqs,tracks):
        observations=np.concatenate([seqs[t]['e'] for t in tracks])
        self.model.fit(observations)
        self.prior=self.model.weights_
        count=np.ones((self.k,self.k))*.1
        for t in tracks:
            s=seqs[t]
            if len(s['e'])<2: continue
            resp=self.model.predict_proba(s['e'])
            valid=np.diff(s['t'])<.351
            count += resp[:-1][valid].T@resp[1:][valid]
        self.transition=count/count.sum(1,keepdims=True)
        return self

    def likelihood(self,e):
        return self.model._estimate_log_prob(e)


def filtered(bank,s):
    likelihood=bank.likelihood(s['e'])
    posterior=[]; iid=[]; diagnostics=[]
    p=bank.prior.copy()
    ewma=0.; cusum=0.
    for j,ll in enumerate(likelihood):
        reset=j==0 or s['t'][j]-s['t'][j-1]>.351
        prior=bank.prior if reset else p@bank.transition
        p,logev=normalize_update(prior,ll)
        ind,_=normalize_update(bank.prior,ll)
        kl=np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(prior,1e-300)))
        residual=np.linalg.norm(s['e'][j])
        if reset: ewma=0.; cusum=0.
        ewma=.8*ewma+.2*residual
        cusum=max(0.,cusum+residual-bank.residual_reference)
        posterior.append(p.copy()); iid.append(ind)
        diagnostics.append([residual,ewma,cusum,-logev,kl])
    return np.array(posterior),np.array(iid),np.array(diagnostics)


def query_states(bank,seqs,meta):
    posterior=np.zeros((len(meta),bank.k)); iid=posterior.copy()
    diagnostics=np.zeros((len(meta),5))
    for track,group in meta.groupby('track',sort=False):
        s=seqs[track]
        if len(s['e'])==0:
            posterior[group.index]=bank.prior; iid[group.index]=bank.prior
            continue
        p,i,d=filtered(bank,s)
        j=np.searchsorted(s['t'],group.time.values+1e-8,side='right')-1
        for idx,n in zip(group.index,j):
            if n<0: posterior[idx]=iid[idx]=bank.prior
            else: posterior[idx],iid[idx],diagnostics[idx]=p[n],i[n],d[n]
    return posterior,iid,diagnostics


def mean_error(meta,error,idx):
    return float(pd.DataFrame({'track':meta.iloc[idx].track.values,
                              'e':error}).groupby('track').e.mean().mean())


def experts_fit(X,Y,tr,p,alpha):
    return [Ridge(alpha=alpha).fit(X[tr],Y[tr].reshape(len(tr),-1),sample_weight=p[tr,k])
            for k in range(p.shape[1])]


def expert_predict(models,X,CV):
    return np.stack([m.predict(X).reshape(-1,4,2)+CV for m in models],axis=1)


def simple_bank(seqs,meta,tr,X,CV,CA):
    tracks=meta.iloc[tr].track.unique()
    residual=np.concatenate([seqs[t]['r'] for t in tracks])
    scale=max(float(np.mean(residual[:,0]**2)),1e-4)
    output=np.zeros((len(meta),3))
    transition=np.full((3,3),.025); np.fill_diagonal(transition,.95)
    for track,g in meta.groupby('track',sort=False):
        s=seqs[track]; p=np.ones(3)/3; history=[]
        for j,r in enumerate(s['r']):
            prior=np.ones(3)/3 if j==0 or s['t'][j]-s['t'][j-1]>.351 else p@transition
            p,_=normalize_update(prior,-.5*np.sum(r*r,axis=1)/scale)
            history.append(p.copy())
        for idx,t in zip(g.index,g.time):
            j=np.searchsorted(s['t'],t+1e-8,side='right')-1
            output[idx]=history[j] if j>=0 else np.ones(3)/3
    ct=np.array([ct_path(x[20:22],x[22:24],H) for x in X])
    modes=np.stack([CV,CA,ct],axis=1)
    return modes,output


def event_metrics(y,score,threshold):
    neg=~y; pos=y
    return dict(events=int(pos.sum()),n=len(y),
        auc=float(roc_auc_score(y,score)) if len(np.unique(y))>1 else None,
        ap=float(average_precision_score(y,score)) if pos.any() else None,
        fpr=float((score[neg]>threshold).mean()) if neg.any() else None,
        tpr=float((score[pos]>threshold).mean()) if pos.any() else None,
        threshold=float(threshold))


def probability_scores(modes,weights,Y,robot,sigma):
    d2=np.sum((modes-robot[:,None])**2,axis=-1)
    q=ncx2.cdf(.7**2/sigma[None,None,:],2,d2/sigma[None,None,:])
    q=np.sum(weights[:,:,None]*q,axis=1)
    truth=np.linalg.norm(Y-robot,axis=-1)<.7
    component=-np.sum((Y[:,None]-modes)**2,axis=-1)/(2*sigma)-np.log(2*np.pi*sigma)
    nll=-logsumexp(component+np.log(np.maximum(weights,1e-300))[:,:,None],axis=1)
    return np.mean((q-truth)**2,axis=1),nll.mean(1),q,truth


def run(root):
    started=time.time()
    old.H=H; old.LONG_CONTEXT=True; old.OUT=OUT/'data'
    X,Y,CV,CA,meta=old.build(root)
    seqs=load_sequences(root,meta)
    protocol=json.loads((ROOT/'results/response_validation/long_context/prediction_protocol.json').read_text())
    save('manifest.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((OUT/'PROTOCOL_ZH.txt').read_bytes()).hexdigest(),
        queries=len(meta),tracks=meta.track.nunique(),transitions=sum(len(s['t']) for s in seqs.values()),
        data_files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.rglob('filtered*.csv')) if 'no_robot' not in str(p)},
        prior_protocol_sha256=hashlib.sha256((ROOT/'results/response_validation/long_context/prediction_protocol.json').read_bytes()).hexdigest()))
    rows=[]; selected=[]; detection=[]
    for fold in protocol['splits']:
        tr,va,te=[np.flatnonzero(meta.track.isin(fold[k+'_tracks'])) for k in ('train','val','test')]
        assert set(meta.iloc[tr].track).isdisjoint(meta.iloc[te].track)
        assert set(meta.iloc[va].track).isdisjoint(meta.iloc[te].track)
        scaler=old.PhysicalScaler().fit(X[tr]); SX=scaler.transform(X)
        base={}; base_val={}; base_cfg={}
        for family in ('ridge','trees'):
            # Freeze choice by track-equal validation error, never test error.
            best=None
            for param in ([1.,10.,100.] if family=='ridge' else [5,15]):
                model=old.regressor(family,param).fit(X[tr],(Y-CV)[tr].reshape(len(tr),-1))
                pv=model.predict(X[va]).reshape(-1,4,2)+CV[va]
                score=mean_error(meta,np.linalg.norm(pv-Y[va],axis=-1).mean(1),va)
                if best is None or score<best[0]: best=score,param,model,pv
            base[family]=best[2].predict(X[te]).reshape(-1,4,2)+CV[te]
            base_val[family]=best[3]; base_cfg[family]=dict(alpha_or_leaf=best[1],validation_error=best[0])
        sigma=np.maximum(np.mean((base_val['trees']-Y[va])**2,axis=(0,2)),.05**2)
        simple_modes,simple_p=simple_bank(seqs,meta,tr,X,CV,CA)
        train_e=np.concatenate([seqs[t]['e'] for t in meta.iloc[tr].track.unique()])
        residual_reference=float(np.quantile(np.linalg.norm(train_e,axis=1),.75))
        failure_threshold=float(np.quantile(np.linalg.norm(Y[tr,2]-CV[tr,2],axis=1),.9))
        gaussian=Bank(1,11).fit(seqs,meta.iloc[tr].track.unique()); gaussian.residual_reference=residual_reference
        _,_,gd=query_states(gaussian,seqs,meta)
        for seed in SEEDS:
            best=None
            for k in (3,6):
                bank=Bank(k,seed).fit(seqs,meta.iloc[tr].track.unique()); bank.residual_reference=residual_reference
                p,iid,diag=query_states(bank,seqs,meta)
                for alpha in (1.,10.,100.):
                    models=experts_fit(SX,Y-CV,tr,p,alpha)
                    modes=expert_predict(models,SX[va],CV[va])
                    pred=np.sum(p[va,:,None,None]*modes,axis=1)
                    score=mean_error(meta,np.linalg.norm(pred-Y[va],axis=-1).mean(1),va)
                    if best is None or score<best[0]: best=score,alpha,bank,p,iid,diag,models
            score,alpha,bank,p,iid,diag,models=best
            modes=expert_predict(models,SX[te],CV[te])
            weights=p[te]
            predictions=dict(cv=CV[te],ca=CA[te],ct=simple_modes[te,2],
                simple_full=np.sum(simple_p[te,:,None,None]*simple_modes[te],axis=1),
                simple_map=simple_modes[te,np.argmax(simple_p[te],axis=1)],
                universal_ridge=base['ridge'],universal_trees=base['trees'],
                learned_full=np.sum(weights[:,:,None,None]*modes,axis=1),
                learned_map=modes[np.arange(len(te)),weights.argmax(1)],
                learned_iid=np.sum(iid[te,:,None,None]*modes,axis=1),
                learned_prior=np.sum(bank.prior[None,:,None,None]*modes,axis=1))
            errors={k:np.linalg.norm(v-Y[te],axis=-1) for k,v in predictions.items()}
            robot=X[te,18:20,None].transpose(0,2,1)+H[None,:,None]*X[te,24:26,None].transpose(0,2,1)
            probabilistic={}
            for key in ('learned_full','learned_map','learned_prior','learned_iid','universal_trees','universal_ridge','cv','simple_full'):
                if key=='learned_full': mm,ww=modes,weights
                elif key=='learned_prior': mm,ww=modes,np.tile(bank.prior,(len(te),1))
                elif key=='learned_iid': mm,ww=modes,iid[te]
                elif key=='simple_full': mm,ww=simple_modes[te],simple_p[te]
                else: mm,ww=predictions[key][:,None],np.ones((len(te),1))
                bs,nll,qr,truth=probability_scores(mm,ww,Y[te],robot,sigma)
                probabilistic[key+'_brier']=bs; probabilistic[key+'_nll']=nll
            for local,idx in enumerate(te):
                row=meta.iloc[idx].to_dict(); row.update(fold=fold['name'],seed=seed,k=bank.k,
                    prefix_change=bool(diag[idx,0]>residual_reference),max_probability=float(weights[local].max()))
                for key,value in errors.items():
                    row[key]=float(value[local].mean())
                    for j,h in enumerate(H): row[f'{key}_{h}']=float(value[local,j])
                for key,value in probabilistic.items(): row[key]=float(value[local])
                row['probe_event_count']=int(truth[local].sum())
                rows.append(row)
            config=dict(fold=fold['name'],seed=seed,k=bank.k,alpha=alpha,validation_error=score,
                means=bank.model.means_,covariances=bank.model.covariances_,transition=bank.transition,
                converged=bool(bank.model.converged_),universal=base_cfg,sigma=sigma,
                test_track_error={key:mean_error(meta,value.mean(1),te) for key,value in errors.items()})
            selected.append(config)
            future_event=np.linalg.norm(Y[:,2]-CV[:,2],axis=1)>failure_threshold
            for col,key in enumerate(('residual','ewma','cusum','bank_nll','update_kl')):
                score_all=diag[:,col]
                normal=va[~future_event[va]]
                threshold=np.quantile(score_all[normal],.95)
                detection.append(dict(fold=fold['name'],seed=seed,method=key,
                    event_definition='future 2s CV error above training q90; not true switch labels',
                    failure_threshold=failure_threshold,**event_metrics(future_event[te],score_all[te],threshold)))
            threshold=np.quantile(gd[va[~future_event[va]],3],.95)
            detection.append(dict(fold=fold['name'],seed=seed,method='single_gaussian_nll',
                                  **event_metrics(future_event[te],gd[te,3],threshold)))
            pd.DataFrame(rows).to_csv(OUT/'queries.csv',index=False)
            save('selected.json',selected); save('detection.json',detection)
            print('FINISHED',fold['name'],seed,json.dumps(config['test_track_error']),flush=True)
    save('runtime.json',dict(seconds=time.time()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
