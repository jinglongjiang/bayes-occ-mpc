"""Frozen-representation Bayesian last-layer audit, not a new RL algorithm."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
import torch
from torch.nn import functional as F

BASE=Path('/home/abc/workspace/bayes_set_continuous/CrowdNav/repair_results')
OUT=Path('/home/abc/temp/paper2_q_posterior_20260916')


def features(sd,observations,actions):
    result=[];qs=[]
    with torch.no_grad():
        for lo in range(0,len(actions),256):
            obs={k:torch.as_tensor(np.stack([o[k] for o in observations[lo:lo+256]]),dtype=torch.float32)
                 for k in observations[0]}
            def linear(x,p):return F.linear(x,sd[p+'.weight'],sd[p+'.bias'])
            h=F.relu(linear(F.relu(linear(obs['humans'],'features_extractor.human.0')),'features_extractor.human.2'))
            r=F.relu(linear(obs['robot'],'features_extractor.robot.0'))
            mask=obs['mask']>.5
            logits=(h*linear(r,'features_extractor.query')[:,None]).sum(-1)/8
            w=logits.masked_fill(~mask,-1e9).softmax(-1)*mask
            w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
            avg=(h*w[...,None]).sum(1)
            maximum=h.masked_fill(~mask[...,None],-torch.inf).amax(1)
            maximum=torch.where(mask.sum(1,keepdim=True)>0,maximum,torch.zeros_like(maximum))
            enc=F.relu(linear(torch.cat([r,avg,maximum],1),'features_extractor.fuse.0'))
            a=torch.as_tensor(np.asarray(actions[lo:lo+256]),dtype=torch.float32)
            a=(a-torch.tensor([.5,0.]))/torch.tensor([.5,1.2])
            x=torch.cat([enc,a],1)
            z=F.relu(linear(F.relu(linear(x,'qf0.0')),'qf0.2'))
            q1=linear(z,'qf0.4')[:,0]
            q2=linear(F.relu(linear(F.relu(linear(x,'qf1.0')),'qf1.2')),'qf1.4')[:,0]
            result.append(torch.cat([z,torch.ones(len(z),1)],1).numpy())
            qs.append(torch.stack([q1,q2],1).numpy())
    return np.concatenate(result).astype(float),np.concatenate(qs).astype(float)


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    files=dict(critic=BASE/'critic_truth_tree/C4_wide_mc5000_td1000_critic.pt',
        transitions=BASE/'no_belief_local_finetune/frozen_transitions.pt',
        counterfactuals=BASE/'critic_truth_tree/counterfactuals.pt')
    protocol=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in files.items()},
        fit='fixed Q1 penultimate features, Gaussian last-layer correction to MC return',
        precision=[.001,.01,.1,1.,10.,100.],
        calibration='first 10 counterfactual layouts; remaining 40 layouts test, both states stay together',
        uncertainty='posterior variance of candidate-minus-baseline, including their covariance',
        controls=['original Q1','original minQ','mean correction','constant calibrated error bound','posterior scaled bound'],
        limitation='historically reused audit; behavior MC differs from deterministic continuation; not prospective RL')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    sd=torch.load(files['critic'],map_location='cpu',weights_only=False)
    raw=torch.load(files['transitions'],map_location='cpu',weights_only=False)
    cf=torch.load(files['counterfactuals'],map_location='cpu',weights_only=False)
    complete=[];episode=[];eid=0
    for row in raw:
        episode.append(row)
        if row['done']:
            g=0
            for item in reversed(episode):
                g=float(item['reward'])+.99*g
                complete.append((item,g,eid))
            episode=[];eid+=1
    x,q=features(sd,[v[0]['observation'] for v in complete],[v[0]['action'] for v in complete])
    y=np.array([v[1] for v in complete]);r=y-q[:,0]
    xc=[];qc=[]
    for node in cf:
        xx,qq=features(sd,[node['observation']]*len(node['candidates']),node['candidates'])
        xc.append(xx);qc.append(qq)
    xc=np.array(xc);qc=np.array(qc)
    audit=json.loads((BASE/'critic_truth_tree/C4_wide_mc5000_td1000.json').read_text())
    np.testing.assert_allclose(qc.min(2),np.array([v['q'] for v in audit['per_state']]),atol=2e-6)
    truth=np.array([v['returns'] for v in cf]);cases=np.array([v['case'] for v in cf])
    cal=np.isin(cases,sorted(set(cases))[:10]);test=~cal
    assert not set(cases[cal])&set(cases[test])
    dx=xc-xc[:,4:5,:];adv_true=truth-truth[:,4:5]
    # Center corrections at the trained head. Same mean is used by all uncertainty ablations.
    choices=[]
    for alpha in protocol['precision']:
        factor=cho_factor(x.T@x+alpha*np.eye(x.shape[1]))
        w=cho_solve(factor,x.T@r)
        pred=qc[:,:,0]+xc@w;adv=pred-pred[:,4:5]
        loss=float(np.mean((adv[cal]-adv_true[cal])**2))
        choices.append((loss,alpha))
    alpha=min(choices)[1];factor=cho_factor(x.T@x+alpha*np.eye(x.shape[1]))
    w=cho_solve(factor,x.T@r);mean=qc[:,:,0]+xc@w;adv=mean-mean[:,4:5]
    sigma2=max(float(np.mean((r-x@w)**2)),1e-8)
    variance=sigma2*np.sum(dx*cho_solve(factor,dx.reshape(-1,x.shape[1]).T).T.reshape(dx.shape),axis=2)
    std=np.sqrt(np.maximum(variance,0))
    # The multiplier is deliberately calibrated against actual counterfactual errors.
    # An uncalibrated iid Gaussian posterior is also reported, not silently trusted.
    delta=adv-adv_true;nonzero=np.linalg.norm(dx,axis=2)>1e-10
    calibration=cal[:,None]&nonzero
    constant=max(0.,float(np.quantile(delta[calibration],.95)))
    multiplier=max(0.,float(np.quantile(delta[calibration]/np.maximum(std[calibration],1e-8),.95)))
    scores=dict(original_q1=qc[:,:,0]-qc[:,4:5,0],original_min=qc.min(2)-qc.min(2)[:,4:5],
        posterior_mean=adv,constant_bound=adv-constant,calibrated_posterior=adv-multiplier*std,
        nominal_posterior=adv-1.645*std,baseline=np.zeros_like(adv))
    for v in scores.values():v[:,4]=0
    selections={k:(np.argmax(v,axis=1) if k!='baseline' else np.full(len(cf),4)) for k,v in scores.items()}
    out={};per=[]
    for name,pick in selections.items():
        ret=truth[np.arange(len(cf)),pick];gain=ret-truth[:,4]
        for i in np.flatnonzero(test):
            per.append(dict(arm=name,case=int(cases[i]),step=cf[i]['step'],gain=float(gain[i]),
                regret=float(truth[i].max()-ret[i]),changed=bool(pick[i]!=4),
                degraded=bool(gain[i]<-.01),pick=int(pick[i])))
        selected=test & (pick!=4)
        out[name]=dict(n=int(test.sum()),changed=int(selected.sum()),mean_gain=float(gain[test].mean()),
            mean_regret=float((truth.max(1)-ret)[test].mean()),harm_over_01=int((test&(gain<-.01)).sum()),
            benefit_over_01=int((test&(gain>.01)).sum()))
    frame=pd.DataFrame(per);frame.to_csv(OUT/'per_state.csv',index=False)
    rng=np.random.default_rng(2407);paired={}
    table=frame.groupby(['case','arm']).gain.mean().unstack()
    for control in ('baseline','posterior_mean','constant_bound'):
        d=(table.calibrated_posterior-table[control]).to_numpy()
        ci=np.quantile(rng.choice(d,(10000,len(d)),replace=True).mean(1),[.025,.975])
        paired[control]=dict(mean=float(d.mean()),ci95=ci.tolist())
    summary=dict(complete_episodes=eid,training_transitions=len(complete),partial_excluded=len(episode),
        calibration_layouts=int(len(set(cases[cal]))),test_layouts=int(len(set(cases[test]))),
        precision=alpha,choices=choices,likelihood_variance=sigma2,constant_bound=constant,
        posterior_multiplier=multiplier,uncalibrated_claim='Not assumed calibrated or Bayes-optimal',
        test_positive_error_coverage_nominal=float((delta[test]<=1.645*std[test]+1e-10).mean()),
        test_positive_error_coverage_calibrated=float((delta[test]<=multiplier*std[test]+1e-10).mean()),
        arms=out,paired=paired,status='screen_only_not_approved')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
