"""Read-only summaries of frozen prediction outputs with explicit stratification."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.peroi_response_validation import bootstrap_difference

OUT = Path(__file__).resolve().parents[1]/'results/response_validation'


def main():
    tables = []
    methods = ['cv','pooled','true_label','map','full_mean','prior_mean','shuffled_label']
    for version in ['stable','long_context']:
        frame = pd.read_csv(OUT/version/'prediction_queries.csv')
        for family in ['ridge','trees']:
            for subset in ['all','moving_transfer','moving_temporal','approaching']:
                f = frame[frame.family==family]
                if subset=='all': f=f[f.fold!='moving_temporal']
                elif subset=='moving_transfer': f=f[(f.fold!='moving_temporal')&f.moving]
                elif subset=='moving_temporal': f=f[f.fold=='moving_temporal']
                else: f=f[(f.fold!='moving_temporal')&(f.closing>0)]
                if not len(f): continue
                horizons = [.5,1.,2.] if version=='stable' else [.5,1.,2.,4.]
                for h in horizons:
                    cols = [f'{m}_{h}' for m in methods]
                    tables.append(dict(version=version,family=family,subset=subset,horizon=h,
                        queries=len(f),tracks=int(f.track.nunique()),
                        means=f.groupby('track')[cols].mean().mean().to_dict(),
                        true_label_vs_pooled=bootstrap_difference(f,f'true_label_{h}',f'pooled_{h}','track')))
        # This uses realized closest time only for retrospective reporting, never model inputs.
        lead_rows = []
        f = frame[(frame.family=='trees')&(frame.fold!='moving_temporal')]
        for lead in [.5,1.,1.5]:
            s = f.assign(lead_error=abs(f.lead_to_closest-lead))
            s = s[s.lead_error<=.5].sort_values(['track','lead_error','time']).drop_duplicates('track')
            for moving in [False,True]:
                z=s[s.moving==moving]
                if len(z):
                    lead_rows.append(dict(lead=lead,moving=moving,n=len(z),
                        label_counts=z.label.value_counts().to_dict(),
                        mean_max_probability=float(z.max_probability.mean()),
                        below_08=int((z.max_probability<.8).sum()),
                        errors=z[methods].mean().to_dict()))
        (OUT/version/'retrospective_lead_summary.json').write_text(json.dumps(lead_rows,indent=2)+'\n')
    (OUT/'horizon_summary.json').write_text(json.dumps(tables,indent=2)+'\n')
    print('Tables',len(tables))
    for t in tables:
        if t['version']=='long_context' and t['family']=='trees' and t['horizon']==4.:
            print(json.dumps(t))


if __name__=='__main__':
    main()
