"""Aggregate repeated seeds before resampling independent tracks or records."""
import numpy as np
import pandas as pd
from experiments.switching_validation import OUT,save,H


def difference(f,a,b,cluster):
    v=(f[a]-f[b]).groupby(f[cluster]).mean().values
    if len(v)<2: return dict(mean=float(np.mean(v)),n=len(v),ci95=None)
    rng=np.random.default_rng(4231)
    boot=rng.choice(v,(4000,len(v)),replace=True).mean(1)
    return dict(mean=float(v.mean()),n=len(v),ci95=np.quantile(boot,[.025,.975]))


def run():
    raw=pd.read_csv(OUT/'queries.csv')
    methods=['cv','ca','ct','simple_full','simple_map','universal_ridge','universal_trees',
             'learned_full','learned_map','learned_iid','learned_prior']
    # Seeds do not create independent observations.
    numeric=[k for k in raw.select_dtypes(include='number').columns if k not in ['seed','label','k','time']]
    f=raw.groupby(['fold','track','time','session','moving','prefix_change'],as_index=False)[numeric].mean()
    results=[]
    for name,mask in [('all_loso',f.fold!='moving_temporal'),
        ('moving_loso',(f.fold!='moving_temporal')&f.moving),
        ('moving_temporal',f.fold=='moving_temporal'),
        ('prefix_change_loso',(f.fold!='moving_temporal')&f.prefix_change),
        ('ordinary_loso',(f.fold!='moving_temporal')&~f.prefix_change)]:
        g=f[mask]
        if not len(g): continue
        columns=methods+[k for k in g if k.endswith('_brier') or k.endswith('_nll')]
        result=dict(subset=name,queries=len(g),tracks=g.track.nunique(),sessions=g.session.nunique(),
            means=g.groupby('track')[columns].mean().mean().to_dict(),
            per_horizon={m:g.groupby('track')[[f'{m}_{h}' for h in H]].mean().mean().values for m in methods},
            paired_track={},paired_session={},probe_events=int(g.probe_event_count.sum()))
        pairs=[('learned_full',b) for b in ['cv','universal_ridge','universal_trees','learned_map','learned_iid','learned_prior']]
        pairs += [('learned_full_'+s,b+'_'+s) for s in ('brier','nll') for b in ('learned_map','universal_trees')]
        for a,b in pairs:
            result['paired_track'][a+'-minus-'+b]=difference(g,a,b,'track')
            # First equalize tracks within each session, not densely queried long tracks.
            sg=g.groupby(['session','track'],as_index=False)[[a,b]].mean()
            result['paired_session'][a+'-minus-'+b]=difference(sg,a,b,'session')
        results.append(result)
    seeds=[]
    for (fold,seed),g in raw.groupby(['fold','seed']):
        seeds.append(dict(fold=fold,seed=int(seed),**g.groupby('track')[methods].mean().mean().to_dict()))
    save('summary.json',results);save('seed_summary.json',seeds)
    print(pd.DataFrame([dict(subset=r['subset'],tracks=r['tracks'],**{k:r['means'][k] for k in methods}) for r in results]).to_string(index=False))


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from experiments import switching_validation as main
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=OUT)
    OUT=parser.parse_args().directory
    main.OUT=OUT
    run()
