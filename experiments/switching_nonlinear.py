"""Equal total tree budget sensitivity, frozen after linear diagnostics."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from experiments import switching_validation as main
from experiments.switching_validation import old,ROOT,H,OUT,SEEDS


def run(root):
    dest=OUT/'nonlinear';dest.mkdir(exist_ok=True)
    old.H=H;old.LONG_CONTEXT=True;old.OUT=dest/'data'
    X,Y,CV,CA,meta=old.build(root)
    seqs=main.load_sequences(root,meta)
    configs=json.loads((OUT/'selected.json').read_text())
    protocol=json.loads((ROOT/'results/response_validation/long_context/prediction_protocol.json').read_text())
    source=pd.read_csv(OUT/'queries.csv')
    rows=[];details=[]
    (dest/'manifest.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((OUT/'NONLINEAR_PROTOCOL_ZH.txt').read_bytes()).hexdigest()),indent=2))
    for fold in protocol['splits']:
        tr,va,te=[np.flatnonzero(meta.track.isin(fold[k+'_tracks'])) for k in ('train','val','test')]
        for seed in SEEDS:
            cfg=next(c for c in configs if c['fold']==fold['name'] and c['seed']==seed)
            k=cfg['k']; bank=main.Bank(k,seed).fit(seqs,meta.iloc[tr].track.unique())
            train_e=np.concatenate([seqs[t]['e'] for t in meta.iloc[tr].track.unique()])
            bank.residual_reference=float(np.quantile(np.linalg.norm(train_e,axis=1),.75))
            np.testing.assert_allclose(bank.model.means_,cfg['means'],rtol=1e-10,atol=1e-12)
            p,iid,_=main.query_states(bank,seqs,meta)
            best=None
            for leaf in (5,15):
                models=[ExtraTreesRegressor(n_estimators=96//k,min_samples_leaf=leaf,max_depth=16,
                    random_state=seed+j,n_jobs=4).fit(X[tr],(Y-CV)[tr].reshape(len(tr),-1),sample_weight=p[tr,j]) for j in range(k)]
                mv=main.expert_predict(models,X[va],CV[va])
                pv=(mv*p[va,:,None,None]).sum(1)
                score=main.mean_error(meta,np.linalg.norm(pv-Y[va],axis=-1).mean(1),va)
                if best is None or score<best[0]:best=score,leaf,models
            modes=main.expert_predict(best[2],X[te],CV[te]);weights=p[te]
            predictions=dict(learned_full=(modes*weights[:,:,None,None]).sum(1),
                learned_map=modes[np.arange(len(te)),weights.argmax(1)],
                learned_iid=(modes*iid[te,:,None,None]).sum(1),
                learned_prior=(modes*bank.prior[None,:,None,None]).sum(1))
            errors={key:np.linalg.norm(v-Y[te],axis=-1) for key,v in predictions.items()}
            robot=X[te,18:20][:,None]+H[None,:,None]*X[te,24:26][:,None]
            frame=source[(source.fold==fold['name'])&(source.seed==seed)].copy().reset_index(drop=True)
            assert np.all(frame.track.values==meta.iloc[te].track.values)
            np.testing.assert_allclose(frame.time,meta.iloc[te].time)
            for key,value in errors.items():
                frame[key]=value.mean(1)
                for j,h in enumerate(H):frame[f'{key}_{h}']=value[:,j]
                ww=weights if key=='learned_full' else iid[te] if key=='learned_iid' else np.tile(bank.prior,(len(te),1))
                mm=modes
                if key=='learned_map':mm=predictions[key][:,None];ww=np.ones((len(te),1))
                bs,nll,_,_=main.probability_scores(mm,ww,Y[te],robot,np.array(cfg['sigma']))
                frame[key+'_brier']=bs;frame[key+'_nll']=nll
            rows.append(frame)
            detail=dict(fold=fold['name'],seed=seed,k=k,leaf=best[1],validation_error=best[0],
                test_track_error={key:main.mean_error(meta,v.mean(1),te) for key,v in errors.items()})
            details.append(detail)
            pd.concat(rows,ignore_index=True).to_csv(dest/'queries.csv',index=False)
            (dest/'selected.json').write_text(json.dumps(details,indent=2))
            print('NONLINEAR',json.dumps(detail),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
