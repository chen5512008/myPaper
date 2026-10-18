from pathlib import Path
import json,sys
import numpy as np,pandas as pd
V=Path(__file__).resolve().parent;D=V.parent;T=D/'transformer_revision'
sys.path.insert(0,str(D/'fusion_revision_20261017'))
from common import pin
from scoring import block_ci
SITES=json.loads((D/'protocol.json').read_text(encoding='utf-8'))['sites']
variants={'full':'empirical_reference','shared_scale':'empirical_affine','no_query':'empirical_no_distribution_query','no_availability':'empirical_no_availability'}
rows=[];preds={};anchor=None
for name,variant in variants.items():
    for seed in [17,29,43]:
        folder=(T/'runs' if name=='full' else T/'reviewer_revision/runs' if seed==17 else V/'runs')/f'pooled_{variant}_{seed}'
        assert (folder/'run.json').exists(),folder
        z=dict(np.load(folder/'complete.npz'));p=z['prediction'];y=z['target'];loss=pin(p,y)
        if anchor is None:anchor=z
        assert np.array_equal(z['site'],anchor['site']) and np.array_equal(z['origin'],anchor['origin']) and np.array_equal(y,anchor['target'])
        cells=[]
        for s in SITES:
            meta=json.loads((D/'prepared'/f'{s}.json').read_text(encoding='utf-8'));threshold=meta['training_high_wave_threshold_m']
            ix=z['site']==s
            for j in range(4):
                high=y[ix,j]>=threshold
                cells.append(float(np.abs(p[ix,j,4]-y[ix,j])[high].mean()))
        rows.append({'variant':name,'seed':seed,'pinball':float(loss.mean()),'highwave_MAE_macro':float(np.mean(cells)),'best_validation':json.loads((folder/'run.json').read_text(encoding='utf-8'))['best_validation']})
        preds[name,seed]=loss.mean((1,2))
frame=pd.DataFrame(rows);frame.to_csv(V/'ablation_by_seed.csv',index=False)
summary=frame.groupby('variant')[['pinball','highwave_MAE_macro']].agg(['mean','std']).reset_index();summary.columns=['_'.join(x).strip('_') for x in summary.columns.to_flat_index()];summary.to_csv(V/'ablation_mean_sd.csv',index=False)
effects=[];ci=[]
for name in ['shared_scale','no_query','no_availability']:
    for seed in [17,29,43]:effects.append({'variant':name,'seed':seed,'ablation_minus_full':float((preds[name,seed]-preds['full',seed]).mean())})
    diff=np.mean([preds[name,seed]-preds['full',seed] for seed in [17,29,43]],axis=0)
    for days in [7,28]:ci.append({'variant':name,**block_ci(diff,anchor['origin'],days)})
pd.DataFrame(effects).to_csv(V/'paired_seed_effects.csv',index=False);pd.DataFrame(ci).to_csv(V/'ablation_time_intervals.csv',index=False)
print(summary.to_string(index=False));print(pd.DataFrame(effects).to_string(index=False))
