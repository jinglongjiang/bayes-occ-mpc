"""Exploratory cross-session information screen, not navigation validation."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path('/home/abc/temp/paper2_bi3_data')


def build(root=ROOT, site='um'):
    records = []
    assert site in ('um', 'laas')
    for path in sorted((root / 'Bi3/jsons' / site).glob('*/cv.json')):
        raw = json.loads(path.read_text())
        time = np.array([r['time'] for r in raw])
        pose = np.array([[r['robot_state']] + r['agent_states'] for r in raw])
        goals = np.array([r['robot_goal'] for r in raw])
        turning = np.array([r['turning'] for r in raw], dtype=float)
        assert np.all(np.diff(time) > 0)
        query = np.arange(time[0] + 2.5, time[-1] - 2.1, .2)
        # All feature lookups are as-of; targets alone use future samples.
        offsets = np.array([0., -.2, -.4, -.8, -1.2, -2.])
        idx = np.searchsorted(time, query[:, None] + offsets, side='right') - 1
        assert np.all(time[idx] <= query[:, None] + offsets)
        past = pose[idx]
        for human in [1, 2]:
            order = [human, 3-human, 0]
            p = past[:, :, order]
            origin = p[:, 0, 0, :2]
            relative = p[..., :2] - origin[:, None, None, :]
            velocity = (p[:, 0, :, :2] - p[:, 1, :, :2]) / (time[idx[:, 0]] - time[idx[:, 1]])[:, None, None]
            proprio = np.column_stack([np.sin(p[:, 0, 2, 2]), np.cos(p[:, 0, 2, 2]),
                                       goals[idx[:, 0]] - origin, turning[idx[:, 0]]])
            current = np.concatenate([relative[:, 0].reshape(-1, 6), velocity.reshape(-1, 6), proprio], 1)
            history = np.concatenate([current, relative[:, 1:].reshape(-1, 30)], 1)
            heading = np.concatenate([current, np.sin(p[:, 0, :2, 2]), np.cos(p[:, 0, :2, 2])], 1)
            target_idx = np.searchsorted(time, query + 2., side='right') - 1
            target = pose[target_idx, human, :2] - origin
            near = np.linalg.norm(relative[:, 0, 2], axis=1) < 1.
            rel = relative[:, 0, 2]
            rel_vel = velocity[:, 2] - velocity[:, 0]
            closest_time = np.clip(-np.sum(rel*rel_vel, axis=1) /
                                   np.maximum(np.sum(rel_vel**2, axis=1), 1e-8), 0, 2)
            cv_distance = np.linalg.norm(rel + closest_time[:, None]*rel_vel, axis=1)
            risk = (cv_distance < .6) & (closest_time > .1)
            records.append(dict(session=int(path.parent.name[len(site):]), current=current,
                                history=history, heading=heading, target=target, near=near,
                                query=query, target_time=time[target_idx], human=human, risk=risk))
    return records


def main(combined=False, external=False, external_site='um'):
    data = build()
    external_records = build(ROOT / 'external', site=external_site) if external else []
    if external:
        assert external_records
        if external_site == 'um':
            assert all(r['session'] >= 10 for r in external_records)
    if combined:
        for r in data + external_records:
            r['history_heading'] = np.concatenate([r['history'], r['heading'][:, -4:]], axis=1)
    train = [r for r in data if r['session'] <= 5]
    valid = [r for r in data if r['session'] == 6]
    test = external_records if external else [r for r in data if r['session'] >= 7]
    result = {'scope': 'exploratory only; all nine sessions belong to official training domain',
              'shared_proprioception': 'robot heading, current robot goal, current controller turning flag; as-of only',
              'heading_arm': 'adds only the two human orientations; robot orientation shared',
              'split': {'train': [1,2,3,4,5], 'validation': [6], 'test': [7,8,9]}, 'arms': {}}
    y = np.concatenate([r['target'] for r in train])
    for arm in (['history', 'history_heading'] if combined else ['current', 'history', 'heading']):
        x = np.concatenate([r[arm] for r in train])
        choices = []
        for leaf in ([15] if external else [7, 15]):
            models = [HistGradientBoostingRegressor(max_leaf_nodes=leaf, max_iter=150,
                      min_samples_leaf=30, learning_rate=.05, l2_regularization=1,
                      early_stopping=False, random_state=2407).fit(x, y[:, d]) for d in range(2)]
            score = np.mean([np.linalg.norm(np.column_stack([m.predict(r[arm]) for m in models])-r['target'], axis=1).mean() for r in valid])
            choices.append((score, models, leaf))
        score, models, leaf = min(choices, key=lambda z: z[0])
        rows = []
        for r in test:
            error = np.linalg.norm(np.column_stack([m.predict(r[arm]) for m in models])-r['target'], axis=1)
            rows.append({'session': r['session'], 'error': float(error.mean()),
                         'near_error': float(error[r['near']].mean()), 'n': len(error), 'near_n': int(r['near'].sum()),
                         'risk_n': int(r['risk'].sum()),
                         'risk_error': float(error[r['risk']].mean()) if r['risk'].any() else None})
        result['arms'][arm] = {'validation': float(score), 'leaves': leaf, 'rows': rows,
                               'mean_error': float(np.mean([r['error'] for r in rows])),
                               'mean_near_error': float(np.mean([r['near_error'] for r in rows]))}
        print(arm, result['arms'][arm]['mean_error'], result['arms'][arm]['mean_near_error'], flush=True)
    result['combined_heading_test'] = combined
    if external:
        result['scope'] = 'new-record check of frozen history vs history+heading; not Bayesian or navigation validation'
        result['split']['test'] = sorted(set(r['session'] for r in test))
        result['selection'] = '15 leaves fixed from original UM6 selection, no retuning on new records'
        result['external_site'] = external_site
    (ROOT / ('laas_history_heading.json' if external and external_site == 'laas' else 'external_history_heading.json' if external else 'exploratory_history_heading.json' if combined else 'exploratory_information.json')).write_text(json.dumps(result, indent=2))


def adaptation(heading_external=False, external_site='um', drift_audit=False):
    from scipy.special import gammaln
    data = build()
    train = [r for r in data if r['session'] <= 5]
    valid = [r for r in data if r['session'] == 6]
    test = build(ROOT / 'external', site=external_site) if heading_external else [r for r in data if r['session'] >= 7]
    assert test
    if heading_external:
        data += test
        for r in data:
            r['history'] = np.concatenate([r['history'], r['heading'][:, -4:]], axis=1)
    x = np.concatenate([r['history'] for r in train])
    y = np.concatenate([r['target'] for r in train])
    selected = json.loads((ROOT / 'exploratory_information.json').read_text())['arms']['history']['leaves']
    models = [HistGradientBoostingRegressor(max_leaf_nodes=selected, max_iter=150,
              min_samples_leaf=30, learning_rate=.05, l2_regularization=1,
              early_stopping=False, random_state=2407).fit(x, y[:, d]) for d in range(2)]
    for r in data:
        r['residual'] = r['target'] - np.column_stack([m.predict(r['history']) for m in models])
    residual = np.concatenate([r['residual'] for r in (valid if heading_external else train)])
    variance = np.maximum(np.mean(residual**2, axis=0) if heading_external else residual.var(0), .01)

    if drift_audit:
        rng = np.random.RandomState(2407)
        rows = []
        for r in test:
            errors = r['residual'][::11]
            centered = errors - errors.mean(axis=0)
            denominator = np.mean(centered**2)
            lag = float(np.mean(centered[:-1]*centered[1:]) / max(denominator, 1e-12))
            null = []
            for _ in range(200):
                shuffled = centered[rng.permutation(len(centered))]
                null.append(np.mean(shuffled[:-1]*shuffled[1:]) / max(denominator, 1e-12))
            blocks = np.array([x.mean(axis=0) for x in np.array_split(errors, 4)])
            rows.append(dict(session=r['session'], human=r['human'], n=len(errors), lag1=lag,
                             permutation95=np.percentile(null, 95),
                             first_last_bias_distance=float(np.linalg.norm(blocks[-1]-blocks[0])),
                             block_means=blocks.tolist()))
        result = dict(scope='descriptive residual dependence audit; full-record centering is diagnostic only, not an online feature',
                      protocol='frozen UM predictor; nonoverlapping 2.2s query stride; 200 within-record permutations; no multiplicity claim',
                      rows=rows, median_lag1=float(np.median([r['lag1'] for r in rows])),
                      median_first_last_bias_distance=float(np.median([r['first_last_bias_distance'] for r in rows])),
                      rows_above_permutation95=int(sum(r['lag1'] > r['permutation95'] for r in rows)))
        (ROOT / 'laas_residual_dependence.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
        return

    def evaluate(records, mode, parameter):
        output = []
        for r in records:
            mu = np.zeros(2)
            cov = np.ones(2) * parameter ** 2 if mode == 'bayes' else np.zeros(2)
            errors, nlls, plugin_nlls = [], [], []
            previous = 0
            for i, now in enumerate(r['query']):
                # Nonoverlapping evidence windows; completed observations only.
                while previous < i and r['target_time'][previous] <= now:
                    observation = r['residual'][previous]
                    if mode == 'bayes':
                        gain = cov / (cov + variance)
                        mu += gain * (observation - mu)
                        cov *= 1 - gain
                    elif mode == 'ewma':
                        mu = (1 - parameter) * mu + parameter * observation
                    previous += 11
                error = r['residual'][i] - mu
                predictive_var = variance + cov
                errors.append(np.linalg.norm(error))
                if mode == 'student':
                    shape = variance * (parameter - 2) / parameter
                    ll = gammaln((parameter+2)/2)-gammaln(parameter/2)-np.log(parameter*np.pi)
                    ll -= .5*np.log(shape).sum() + (parameter+2)/2*np.log1p(np.sum(error**2/shape)/parameter)
                    nlls.append(-ll)
                else:
                    nlls.append(.5 * np.sum(np.log(2*np.pi*predictive_var) + error**2/predictive_var))
                plugin_nlls.append(.5 * np.sum(np.log(2*np.pi*variance) + error**2/variance))
            output.append(dict(session=r['session'], human=r['human'],
                               error=float(np.mean(errors)), nll=float(np.mean(nlls)),
                               plugin_nll=float(np.mean(plugin_nlls))))
        return output

    result = {'scope': 'exploratory constant person-specific residual bias, Gaussian likelihood approximation',
              'evidence': '2 second targets observed before update; evidence stride 2.2 seconds',
              'variance': variance.tolist(), 'arms': {}}
    grids = [('fixed', [0]), ('ewma', [0, .01, .03, .1, .3]), ('bayes', [.01, .1, .3, 1.])]
    if heading_external:
        grids.append(('student', [3, 5, 10, 30]))
        result['scope'] = 'frozen history+heading predictor; UM6 residual second moment calibration; UM10-17 external residual adaptation'
        result['external_site'] = external_site
        if external_site == 'laas':
            result['scope'] = 'UM1-5 training, UM6 frozen selection, LAAS-only causal adaptation; no LAAS training or tuning'
    for mode, grid in grids:
        scored = [(np.mean([r['nll'] for r in evaluate(valid, mode, p)]), p) for p in grid]
        score, parameter = min(scored)
        rows = evaluate(test, mode, parameter)
        result['arms'][mode] = dict(parameter=parameter, validation_nll=float(score), rows=rows,
                                   error=float(np.mean([r['error'] for r in rows])),
                                   nll=float(np.mean([r['nll'] for r in rows])),
                                   plugin_nll=float(np.mean([r['plugin_nll'] for r in rows])))
        print(mode, result['arms'][mode], flush=True)
    (ROOT / ('laas_heading_adaptation.json' if heading_external and external_site == 'laas' else 'external_heading_adaptation.json' if heading_external else 'exploratory_adaptation.json')).write_text(json.dumps(result, indent=2))


def model_bank(density=False):
    from scipy.special import logsumexp
    data=build();train=[r for r in data if r['session']<=5]
    valid=[r for r in data if r['session']==6];test=[r for r in data if r['session']>=7]
    def fit(rows):
        x=np.concatenate([r['history'] for r in rows]);y=np.concatenate([r['target'] for r in rows])
        return [HistGradientBoostingRegressor(max_leaf_nodes=15,max_iter=150,min_samples_leaf=30,
                learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=2407).fit(x,y[:,d]) for d in range(2)]
    experts=[fit([r for r in train if r['session']==s]) for s in range(1,6)]
    universal=fit(train)
    for r in data:
        r['bank']=np.stack([np.column_stack([m.predict(r['history']) for m in expert]) for expert in experts],axis=1)
        r['universal']=np.column_stack([m.predict(r['history']) for m in universal])
    # Each expert's noise estimate excludes its own training session.
    variances=np.stack([np.maximum(np.concatenate([r['target']-r['bank'][:,k] for r in train if r['session']!=k+1]).var(0),.01) for k in range(5)])
    if density:
        variances=np.stack([np.maximum(np.mean(np.concatenate([r['target']-r['bank'][:,k] for r in valid])**2,axis=0),.01) for k in range(5)])
    def evaluate(records,mode,rate,power):
        rows=[]
        for r in records:
            weights=np.ones(5)/5;previous=0;errors=[];nll=[];ent=[];moment_nll=[]
            for i,now in enumerate(r['query']):
                while previous<i and r['target_time'][previous]<=now:
                    assert r['target_time'][previous]<=now
                    residual=r['target'][previous]-r['bank'][previous]
                    ll=-.5*np.sum(np.log(2*np.pi*variances)+residual**2/variances,axis=1)
                    prior=(1-rate)*weights+rate/5
                    logw=np.log(np.maximum(prior,1e-300))+power*ll
                    weights=np.exp(logw-logsumexp(logw));previous+=11
                w=weights.copy()
                if mode=='uniform':w[:]=.2
                if mode=='map':w=np.eye(5)[int(np.argmax(w))]
                mean=np.sum(w[:,None]*r['bank'][i],axis=0)
                residual=r['target'][i]-r['bank'][i]
                ll=-.5*np.sum(np.log(2*np.pi*variances)+residual**2/variances,axis=1)
                nll.append(-logsumexp(np.log(np.maximum(w,1e-300))+ll))
                if density:
                    delta=r['bank'][i]-mean
                    cov=np.diag(np.sum(w[:,None]*variances,axis=0))+(delta*w[:,None]).T@delta
                    error=r['target'][i]-mean
                    sign,logdet=np.linalg.slogdet(cov);assert sign>0
                    moment_nll.append(np.log(2*np.pi)+.5*(logdet+error@np.linalg.solve(cov,error)))
                errors.append(np.linalg.norm(r['target'][i]-mean))
                ent.append(-np.sum(w*np.log(np.maximum(w,1e-300))))
            rows.append(dict(session=r['session'],human=r['human'],n=len(errors),
                             error=float(np.mean(errors)),nll=float(np.mean(nll)),entropy=float(np.mean(ent)),
                             moment_nll=float(np.mean(moment_nll)) if density else None))
        return rows
    result=dict(scope='exploratory delayed Bayesian expert selection, not human type inference or navigation approval',
        split=dict(train=[1,2,3,4,5],validation=[6],test=[7,8,9]),
        evidence='completed2s targets, evidence windows2.2s apart; history features as-of',
        caveats=['all sessions are official training-domain data, repeatedly explored',
                 'experts correspond to training recordings, not known human behavioral types',
                 'bank has5x the trees of universal; bank comparisons share identical experts',
                 'Gaussian likelihood approximate; future robot actions not given, no causal action model',
                 'rate is prior reset per evidence update, power!=1 is generalized Bayes'],
        variance=variances.tolist(),arms={})
    result['arms']['universal']=dict(error=float(np.mean([np.linalg.norm(r['target']-r['universal'],axis=1).mean() for r in test])))
    for mode in ['uniform','map','full']:
        grid=[(0,1)] if mode=='uniform' else [(rate,power) for rate in [0,.05,.2,.5,1.] for power in [.1,.3,1.]]
        scores=[]
        for rate,power in grid:
            rows=evaluate(valid,mode,rate,power);scores.append((np.mean([r['nll'] for r in rows]),rate,power))
        score,rate,power=min(scores);rows=evaluate(test,mode,rate,power)
        result['arms'][mode]=dict(rate=rate,power=power,validation_nll=float(score),rows=rows,
                                  error=float(np.mean([r['error'] for r in rows])),nll=float(np.mean([r['nll'] for r in rows])))
        if density:result['arms'][mode]['moment_nll']=float(np.mean([r['moment_nll'] for r in rows]))
        print(mode,result['arms'][mode]['error'],result['arms'][mode]['nll'],flush=True)
    if density:
        from scipy.stats import multivariate_t
        variance=np.maximum(np.mean(np.concatenate([r['target']-r['universal'] for r in valid])**2,axis=0),.01)
        def score_universal(records,df):
            rows=[]
            for r in records:
                e=r['target']-r['universal']
                ll=-.5*np.sum(np.log(2*np.pi*variance)+e**2/variance,axis=1) if df is None else multivariate_t.logpdf(e,loc=np.zeros(2),shape=np.diag(variance)*(df-2)/df,df=df)
                rows.append(dict(session=r['session'],human=r['human'],nll=float(-np.mean(ll))))
            return rows
        choices=[(np.mean([r['nll'] for r in score_universal(valid,df)]),df) for df in [3,5,10,30]]
        _,df=min(choices)
        for name,d in [('universal_gaussian',None),('universal_student',df)]:
            rows=score_universal(test,d);result['arms'][name]=dict(df=d,rows=rows,nll=float(np.mean([r['nll'] for r in rows])))
        result['density_protocol']='all zero-mean diagonal residual variances calibrated by MSE onUM6; df and bank hyperparameters selected there; no test refit'
        result['universal_variance']=variance.tolist()
    (ROOT/('exploratory_model_bank_density.json' if density else 'exploratory_model_bank.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def joint_error(geometry=False,occupancy=False,external=False,plugin=False):
    from scipy.stats import multivariate_normal,multivariate_t
    from sklearn.covariance import LedoitWolf
    data=build();sessions=[]
    for s in range(1,10):
        pair=sorted([r for r in data if r['session']==s],key=lambda r:r['human'])
        assert len(pair)==2 and np.array_equal(pair[0]['query'],pair[1]['query'])
        sessions.append(dict(session=s,x=pair[0]['history'],y=np.concatenate([r['target'] for r in pair],axis=1),
                             query=pair[0]['query'],target_time=pair[0]['target_time']))
    train=[r for r in sessions if r['session']<=5];valid=sessions[5];test=sessions[6:]
    if external:
        external_data=build(ROOT/'external',site='laas');test=[]
        for s in sorted(set(r['session'] for r in external_data)):
            pair=sorted([r for r in external_data if r['session']==s],key=lambda r:r['human'])
            assert len(pair)==2 and np.array_equal(pair[0]['query'],pair[1]['query'])
            test.append(dict(session=s,x=pair[0]['history'],y=np.concatenate([r['target'] for r in pair],axis=1),
                             query=pair[0]['query'],target_time=pair[0]['target_time']))
        assert len(test)==20
        sessions+=test
    x=np.concatenate([r['x'] for r in train]);y=np.concatenate([r['y'] for r in train])
    models=[HistGradientBoostingRegressor(max_leaf_nodes=15,max_iter=150,min_samples_leaf=30,
            learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=2407).fit(x,y[:,j]) for j in range(4)]
    for r in sessions:r['error']=r['y']-np.column_stack([m.predict(r['x']) for m in models])
    if geometry:
        for r in sessions:
            direction=r['x'][:,2:4]
            angle=np.arctan2(direction[:,1],direction[:,0])
            c,s=np.cos(angle),np.sin(angle)
            rot=np.stack([np.stack([c,s],axis=1),np.stack([-s,c],axis=1)],axis=1)
            r['error']=np.einsum('nij,nkj->nki',rot,r['error'].reshape(-1,2,2)).reshape(-1,4)
    bias=valid['error'].mean(0)
    cov=LedoitWolf().fit(valid['error']).covariance_+np.eye(4)*1e-8
    block=cov.copy();block[:2,2:]=0;block[2:,:2]=0
    rows=[]
    for r in test:
        e=r['error']-bias
        values=dict(gaussian_full=-multivariate_normal.logpdf(e,cov=cov),
                    gaussian_block=-multivariate_normal.logpdf(e,cov=block),
                    student_full=-multivariate_t.logpdf(e,shape=cov*3/5,df=5),
                    student_block_shared_scale=-multivariate_t.logpdf(e,shape=block*3/5,df=5),
                    student_independent=-multivariate_t.logpdf(e[:,:2],shape=cov[:2,:2]*3/5,df=5)-multivariate_t.logpdf(e[:,2:],shape=cov[2:,2:]*3/5,df=5))
        rows.append(dict(session=r['session'],n=len(e),nll={k:float(v.mean()) for k,v in values.items()}))
    result=dict(scope='two-human joint 2s forecast error; not decision or online Bayesian approval',
                protocol='same four-output nonlinear mean; UM1-5train,UM6bias/covcalibration,UM7-9test;matched2D marginals',
                covariance=cov.tolist(),correlation=(cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))).tolist(),
                rows=rows,mean={k:float(np.mean([r['nll'][k] for r in rows])) for k in rows[0]['nll']},
                caveats=['all officialtrain domain, repeated exploratory use, only3test sessions',
                         'world-frame human displacements, actual logged controller, not candidate-action counterfactual',
                         'block multivariateStudent retains shared scale dependence; independentStudent removes it',
                         'static covariance fit is not recursive Bayesian posterior'])
    def online(record,kind,parameter,independent=False,predictive='posterior'):
        e=record['error']-bias;previous=0;matrix=cov.copy();nu=parameter
        if kind=='iw':matrix=matrix*(nu-5)
        scores=[]
        for i,now in enumerate(record['query']):
            while previous<i and record['target_time'][previous]<=now:
                outer=np.outer(e[previous],e[previous])
                if kind=='iw':matrix+=outer;nu+=1
                else:matrix=(1-parameter)*matrix+parameter*outer
                previous+=11
            df=nu-3 if kind=='iw' else 5
            scale=matrix/df if kind=='iw' else matrix*3/5
            if independent:
                scale=scale.copy();scale[:2,2:]=0;scale[2:,:2]=0
            if predictive == 'gaussian':
                scores.append(-multivariate_normal.logpdf(e[i],cov=scale*df/(df-2)+np.eye(4)*1e-9))
            elif predictive == 'posterior':
                scores.append(-multivariate_t.logpdf(e[i],shape=scale+np.eye(4)*1e-9,df=df))
            else:
                fixed_df=float(predictive)
                scores.append(-multivariate_t.logpdf(e[i],shape=scale*df/(df-2)*(fixed_df-2)/fixed_df+np.eye(4)*1e-9,df=fixed_df))
        return float(np.mean(scores))
    adaptive={}
    for kind,grid in [('iw',[6,12,24,64]),('ewma',[.01,.05,.1,.2])]:
        for independent in [False,True]:
            scores=[(online(valid,kind,p,independent),p) for p in grid]
            score,p=min(scores);values=[online(r,kind,p,independent) for r in test]
            adaptive[kind+('_block' if independent else '_full')]=dict(parameter=p,validation_nll=score,
                    test_nll=values,mean_nll=float(np.mean(values)))
    result['adaptive']=adaptive
    if plugin:
        parameter=adaptive['iw_full']['parameter']
        controls={}
        choices=[(online(valid,'iw',parameter,predictive=df),df) for df in [3,5,10,30]]
        _,fixed_df=min(choices)
        for name,predictive in [('posterior','posterior'),('gaussian','gaussian'),('fixed_student',fixed_df)]:
            values=[online(r,'iw',parameter,predictive=predictive) for r in test]
            controls[name]=dict(predictive=predictive,mean_nll=float(np.mean(values)),test_nll=values)
        result['matched_covariance_controls']=controls
    result['adaptive_protocol']='known fixed residual mean; IW prior E[cov]=UM6cov;df=nu-3; completed2s residuals update every2.2s; all block versions retain shared Student scale;UM6 selects parameters'
    if occupancy:
        from scipy.stats import qmc,norm,chi2
        from sklearn.metrics import roc_auc_score
        sample=qmc.Sobol(6,scramble=True,seed=2407).random_base2(12)
        standard=norm.ppf(np.clip(sample[:,:4],1e-10,1-1e-10))/np.sqrt(chi2.ppf(sample[:,4],5)/5)[:,None]
        independent=norm.ppf(np.clip(sample[:,:4],1e-10,1-1e-10))
        independent[:,:2]/=np.sqrt(chi2.ppf(sample[:,4],5)/5)[:,None]
        independent[:,2:]/=np.sqrt(chi2.ppf(sample[:,5],5)/5)[:,None]
        occupancy_rows=[]
        for r in test:
            prediction=r['y']-r['error']+bias
            origin=np.concatenate([np.zeros((len(r['x']),2)),r['x'][:,2:4]],axis=1)
            reference=r['x'][:,4:6]+2*r['x'][:,10:12]
            truth=(r['y']+origin).reshape(-1,2,2)
            labels=(np.linalg.norm(truth-reference[:,None,:],axis=2)<.6).any(1)
            metrics={};probabilities={}
            for arm,matrix in [('full',cov),('block',block),('independent',block)]:
                draws=independent if arm=='independent' else standard
                noise_samples=draws@np.linalg.cholesky(matrix*3/5).T
                values=[]
                for start in range(0,len(prediction),64):
                    points=(prediction[start:start+64,None,:]+origin[start:start+64,None,:]+noise_samples[None,:,:]).reshape(-1,4096,2,2)
                    hit=(np.linalg.norm(points-reference[start:start+64,None,None,:],axis=3)<.6).any(2)
                    values.extend(hit.mean(1))
                p=np.asarray(values);probabilities[arm]=p
                metrics[arm]=dict(brier=float(np.mean((p-labels)**2)),
                    auroc=float(roc_auc_score(labels,p)) if labels.any() and not labels.all() else None)
            occupancy_rows.append(dict(session=r['session'],n=len(labels),positives=int(labels.sum()),metrics=metrics,
                mean_abs_probability_delta=float(np.mean(np.abs(probabilities['full']-probabilities['block'])))))
        result['occupancy']=dict(protocol='radius0.6m around current robot CV position at2s; real future human locations;4096sharedSobolStudent samples;full vs block same marginals',
             limitations='endpoint occupancy forecast, not swept collision; hypothetical reference point not a counterfactual robot action; static calibratedStudent, not adaptive posterior',
             rows=occupancy_rows,mean_brier={arm:float(np.mean([r['metrics'][arm]['brier'] for r in occupancy_rows])) for arm in ['full','block','independent']})
    if geometry:
        result['geometry_control']='rotate both human errors into current human1-to-human2 axis; current geometry only; determinant1 preserves density units; means unchanged before calibration'
    if external:
        result['protocol']='same frozen UM1-5 mean and UM6 calibration/selection; LAAS1-20 cross-site test; no LAAS tuning'
        result['caveats'][0]='20 LAAS records; earlier other-model exploration of these records; not a new untouched confirmatory dataset'
    (ROOT/('laas_joint_plugin.json' if plugin else 'laas_joint_occupancy.json' if external else 'joint_occupancy_independent.json' if occupancy else 'joint_error_geometry.json' if geometry else 'joint_error.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__ == '__main__':
    if '--joint-plugin-laas' in sys.argv:joint_error(external=True,plugin=True)
    elif '--joint-occupancy-laas' in sys.argv:joint_error(occupancy=True,external=True)
    elif '--residual-dependence-laas' in sys.argv:adaptation(heading_external=True, external_site='laas', drift_audit=True)
    elif '--heading-adaptation-laas' in sys.argv:adaptation(heading_external=True, external_site='laas')
    elif '--history-heading-laas' in sys.argv:main(combined=True, external=True, external_site='laas')
    elif '--heading-adaptation-external' in sys.argv:adaptation(heading_external=True)
    elif '--history-heading-external' in sys.argv:main(combined=True, external=True)
    elif '--history-heading' in sys.argv:main(combined=True)
    elif '--joint-occupancy' in sys.argv:joint_error(occupancy=True)
    elif '--joint-error-axis' in sys.argv:joint_error(geometry=True)
    elif '--joint-error' in sys.argv:joint_error()
    elif '--bank-density' in sys.argv:model_bank(density=True)
    elif '--model-bank' in sys.argv:model_bank()
    elif '--adaptation' in sys.argv:adaptation()
    else:main()
