"""Observed stop-duration screening with explicit right censoring."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax

ROOT=Path('/home/abc/temp/peroi-full/data')
OUT=Path('/home/abc/temp/paper2_stop_duration_20260916')


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=lambda x:x.tolist())+'\n')


def collect():
    rows=[]
    for path in sorted(ROOT.rglob('filtered*.csv')):
        session=str(path.parent.relative_to(ROOT))
        fold=int(hashlib.sha256(path.parent.name.encode()).hexdigest(),16)%5
        split='test' if fold==0 else 'val' if fold==1 else 'train'
        frame=pd.read_csv(path)
        for tid,g in frame.groupby('Pedestrian_ID',sort=True):
            g=g.sort_values('Frame_Number');t=g.Frame_Number.to_numpy(float)/1000
            xy=g[['X_Position','Y_Position']].to_numpy(float)
            if len(t)<6 or np.any(np.diff(t)<=0) or not np.isfinite(xy).all():continue
            chunks=np.split(np.arange(len(t)),np.flatnonzero(np.diff(t)>.3)+1)
            found=False
            for ix in chunks:
                if len(ix)<6 or t[ix[-1]]-t[ix[0]]<2:continue
                grid=np.arange(t[ix[0]],t[ix[-1]]+1e-8,.25)
                p=np.column_stack([np.interp(grid,t[ix],xy[ix,j]) for j in (0,1)])
                speed=np.linalg.norm(p[2:]-p[:-2],axis=1)/.5;tt=grid[2:]
                moving=False;start=None;high=0
                for j,s in enumerate(speed):
                    if start is None:
                        if s>=.25:moving=True
                        elif s<=.15 and moving:start=j;high=0
                    else:
                        high=high+1 if s>=.25 else 0
                        if high>=2:
                            duration=tt[j]-tt[start]
                            if duration>=1:
                                rows.append(dict(session=session,track=f'{session}/{tid}',split=split,
                                    duration=float(duration),event=1,start=float(tt[start])))
                                found=True;break
                            start=None;moving=True
                if not found and start is not None and tt[-1]-tt[start]>=1:
                    rows.append(dict(session=session,track=f'{session}/{tid}',split=split,
                        duration=float(tt[-1]-tt[start]),event=0,start=float(tt[start])))
                    found=True
                if found:break
    return pd.DataFrame(rows)


def collect_diamor():
    sys.path.insert(0,'/home/abc/temp/paper2_groups_20260916')
    from probe import annotations
    root=Path('/home/abc/temp/paper2_groups_20260916');rows=[]
    for day in (1,2):
        with zipfile.ZipFile(root/f'DIAMOR-{day}.zip') as z:
            links,aliases,bad=annotations(z.read(f'groups_DIAMOR-{day}.dat').decode())
            blocks=[];last_bin=None;selected_stamp=None
            for chunk in pd.read_csv(z.open(f'person_DIAMOR-{day}_all.csv'),header=None,chunksize=200000):
                a=chunk.to_numpy();eligible=[]
                for stamp in np.unique(a[:,0]):
                    key=int(stamp*4)
                    if key!=last_bin:last_bin=key;selected_stamp=stamp
                    if stamp==selected_stamp:eligible.append(stamp)
                a=a[np.isin(a[:,0],eligible)][:,:4];blocks.append(a)
            values=np.concatenate(blocks);del blocks
        base=values[:,0].min();end=values[:,0].max();values[:,1]=[aliases.get(int(i),int(i)) for i in values[:,1]]
        for tid,g in pd.DataFrame(values).groupby(1,sort=True):
            tid=int(tid)
            if tid<=0 or tid in bad:continue
            g=g.drop_duplicates(0).sort_values(0);a=g.to_numpy();t=a[:,0];xy=a[:,2:4]/1000
            found=False
            for ix in np.split(np.arange(len(t)),np.flatnonzero(np.diff(t)>.4)+1):
                if len(ix)<6 or t[ix[-1]]-t[ix[0]]<2:continue
                grid=np.arange(t[ix[0]],t[ix[-1]]+1e-8,.25)
                pos=np.column_stack([np.interp(grid,t[ix],xy[ix,j]) for j in (0,1)])
                speed=np.linalg.norm(pos[2:]-pos[:-2],axis=1)/.5;tt=grid[2:]
                moving=False;start=None;high=0
                for j,s in enumerate(speed):
                    if start is None:
                        if s>=.25:moving=True
                        elif s<=.15 and moving:start=j;high=0
                    else:
                        high=high+1 if s>=.25 else 0
                        if high>=2:
                            duration=tt[j]-tt[start]
                            if duration>=1:
                                at=tt[start];event=1;found=True;break
                            start=None;moving=True
                if not found and start is not None and tt[-1]-tt[start]>=1:
                    duration=tt[-1]-tt[start];at=tt[start];event=0;found=True
                if found:
                    split='test' if day==2 else 'train' if at<base+.8*(end-base) else 'val'
                    rows.append(dict(session=f'day{day}_block{int((at-base)//600)}',track=f'{day}/{tid}',
                        split=split,duration=float(duration),event=event,start=float(at),day=day))
                    break
        print('DIAMOR stop collection',day,len(rows),flush=True)
        del values
    return pd.DataFrame(rows)


def fit_models(train):
    t=np.maximum(train.duration.to_numpy()-1,.125);e=train.event.to_numpy()
    rate=max(e.sum()/t.sum(),1e-5)
    models={'Exponential':dict(rates=np.array([rate]),weights=np.ones(1))}
    for k in (2,3):
        def loss(v):
            rates=np.exp(v[:k]);logw=v[k:]-logsumexp(v[k:])
            return -np.mean(logsumexp(logw[None]-t[:,None]*rates[None]+e[:,None]*np.log(rates)[None],axis=1))
        best=None
        for scale in (.5,1.,2.):
            initial=np.r_[np.log(np.geomspace(max(rate*.2,1e-4),rate*5,k)*scale),np.zeros(k)]
            r=minimize(loss,initial,method='L-BFGS-B',bounds=[(-9,4)]*k+[(-8,8)]*k)
            if best is None or r.fun<best.fun:best=r
        if not best.success:raise RuntimeError(best.message)
        models[f'FULL{k}']=dict(rates=np.exp(best.x[:k]),weights=softmax(best.x[k:]))
    def weibull(v):
        shape,scale=np.exp(v);power=(t/scale)**shape
        logdensity=np.log(shape/scale)+(shape-1)*np.log(t/scale)-power
        return -np.mean(e*logdensity+(1-e)*(-power))
    fits=[minimize(weibull,[np.log(s),np.log(1/rate)],method='L-BFGS-B',bounds=[(-3,3),(-4,9)]) for s in (.5,1.,2.)]
    best=min(fits,key=lambda r:r.fun)
    if not best.success:raise RuntimeError(best.message)
    models['Weibull']=dict(shape=np.exp(best.x[0]),scale=np.exp(best.x[1]))
    times=np.unique(t[e==1]);logsurv=[];value=0.
    for time in times:
        risk=(t>=time).sum();events=((t==time)&(e==1)).sum()
        # Nelson-Aalen exponential survival avoids a terminal zero from finite data.
        value-=events/risk;logsurv.append(value)
    models['EmpiricalSurvival']=dict(times=times,logs=np.array(logsurv),tail_rate=rate)
    return models


def log_survival(model,t):
    t=np.asarray(t)
    if 'rates' in model:
        return logsumexp(np.log(model['weights'])-t[...,None]*model['rates'],axis=-1)
    if 'shape' in model:return -(t/model['scale'])**model['shape']
    index=np.searchsorted(model['times'],t,side='right')-1
    result=np.where(index>=0,model['logs'][np.maximum(index,0)],0.)
    return result-model['tail_rate']*np.maximum(t-model['times'][-1],0)


def evaluate(data,models):
    rows=[]
    for row in data.itertuples():
        for age in (1.,2.,4.,8.):
            if row.duration<=age:continue
            for horizon in (1.,2.,4.,8.):
                residual=row.duration-age
                if not row.event and residual<horizon:continue
                label=float(row.event and residual<=horizon);values={}
                for name,model in models.items():
                    u=age-1
                    prob=1-np.exp(log_survival(model,u+horizon)-log_survival(model,u))
                    values[name]=float(np.clip(prob,1e-6,1-1e-6))
                full=models['FULL'];post=softmax(np.log(full['weights'])-(age-1)*full['rates'])
                values['MAP']=float(1-np.exp(-horizon*full['rates'][post.argmax()]))
                # Same conditional mean waiting time, collapsed to one exponential.
                mean=float(post@(1/full['rates']))
                values['MeanWait']=float(1-np.exp(-horizon/mean))
                entry=dict(session=row.session,track=row.track,split=row.split,age=age,horizon=horizon,label=label)
                for name,prob in values.items():
                    entry[name+'_nll']=-(label*np.log(prob)+(1-label)*np.log1p(-prob))
                    entry[name+'_brier']=(prob-label)**2
                rows.append(entry)
    return pd.DataFrame(rows)


def main(diamor=False):
    OUT.mkdir(exist_ok=False)
    save('protocol.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grids=dict(mixture_components=[2,3],query_ages=[1,2,4,8],horizons=[1,2,4,8]),
        onset='observed moving >=.25 then speed <=.15; .5s backward velocity, .25s grid',
        event='second successive >=.25m/s sample; gaps>.3s and track end right censored',
        condition='only stops observed >=1s; fit residual beyond first second to avoid left-truncation bias',
        inference='Bayesian persistent exponential-component type posterior given observed survival age',
        baselines=['Exponential','Weibull','EmpiricalSurvival','MAP','MeanWait'],
        selection='training right-censored likelihood; K selected by recording-equal validation event NLL',
        warning='evaluated horizons omit unresolved right-censored labels; report this selection limitation',
        gate='FULL reliably better than Weibull and EmpiricalSurvival before any robot decision experiment'))
    if diamor:
        save('dataset_addendum.json',dict(dataset='DIAMOR two full days, 4Hz complete-frame subsampling',
            split='first 80% of day1 train, remaining 20% val, entire day2 test',
            gaps='subsample spacing <=.4s; same .5s backward velocity and speed thresholds',
            clusters='ten-minute blocks within one venue, NOT independent recording generalization',
            reason='PeRoI had only 24 stops, insufficient evidence; no hypothesis or threshold changed'))
    data=collect_diamor() if diamor else collect();data.to_csv(OUT/'stops.csv',index=False)
    print(data.groupby('split').agg(stops=('event','size'),events=('event','sum'),recordings=('session','nunique')),flush=True)
    if any(not (data.split==p).any() for p in ('train','val','test')):
        save('summary.json',dict(verdict='insufficient split coverage'));return
    models=fit_models(data[data.split=='train']);scores={}
    for k in (2,3):
        trial=dict(models,FULL=models[f'FULL{k}']);d=evaluate(data[data.split=='val'],trial)
        scores[k]=float(d.groupby(['session','track']).mean(numeric_only=True).groupby('session').mean().FULL_nll.mean())
    selected=min(scores,key=scores.get);models['FULL']=models[f'FULL{selected}'];save('models.json',dict(models=models,selected=selected,scores=scores))
    d=evaluate(data,models);d.to_csv(OUT/'queries.csv',index=False)
    test=d[d.split=='test'];table=test.groupby(['session','track']).mean(numeric_only=True).groupby('session').mean()
    rng=np.random.default_rng(2407);comparisons={}
    for other in ('Exponential','Weibull','EmpiricalSurvival','MAP','MeanWait'):
        delta=(table.FULL_nll-table[other+'_nll']).to_numpy()
        comparisons[other]=dict(mean=float(delta.mean()),ci95=np.quantile(rng.choice(delta,(10000,len(delta))).mean(1),[.025,.975]).tolist())
    result=dict(population=data.groupby('split').agg(stops=('event','size'),events=('event','sum'),recordings=('session','nunique')).to_dict('index'),
        selected=selected,means=table.filter(regex='_(nll|brier)$').mean().to_dict(),comparisons=comparisons,
        evaluable_test_queries=len(test),test_recordings=len(table))
    save('summary.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    diamor='--diamor' in sys.argv
    if diamor:OUT=OUT/'diamor'
    main(diamor)
