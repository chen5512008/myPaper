"""One-pass scores and synchronized time-block intervals for frozen models."""
import pandas as pd
from common import *
from bank import apply_controls
from fusion import pool
from gate import apply_gate

def metrics(p,y,high):
    losses=pin(p,y);lo=p[...,0];up=p[...,-1];width=up-lo
    interval=width+10*np.maximum(lo-y,0)+10*np.maximum(y-up,0)
    return {'pinball':float(losses.mean()),'median_MAE':float(np.abs(p[...,4]-y).mean()),
        'coverage80':float(((y>=lo)&(y<=up)).mean()),'width80':float(width.mean()),'interval_score80':float(interval.mean()),
        'highwave_count':int(high.sum()),'highwave_MAE':float(np.abs(p[...,4]-y)[high].mean()) if high.any() else None,
        'highwave_q90_loss':float(losses[...,-1][high].mean()) if high.any() else None}

def block_ci(diff,origin,days):
    # Aggregate simultaneous errors over all sites before resampling time.
    f=pd.DataFrame({'origin':origin.astype('datetime64[ns]'),'diff':diff});v=f.groupby('origin',sort=True)['diff'].mean().to_numpy()
    length=days*4;n=len(v);nb=int(np.ceil(n/length));rng=np.random.default_rng(20261017+days);values=[]
    for _ in range(20):
        starts=rng.integers(0,n,size=(100,nb));ix=(starts[:,:,None]+np.arange(length))%n
        values.extend(v[ix.reshape(100,-1)[:,:n]].mean(1))
    low,high=np.quantile(values,[.025,.975])
    return {'difference':float(v.mean()),'ci_low':float(low),'ci_high':float(high),'block_days':days,'replicates':2000,'distinct_origins':n,'method':'circular moving time blocks; all sites synchronized'}

selection=json.loads((R/'selection.json').read_text());names=selection['members'];obj=selection['object'];ctrl=load(R/'selection/controls.npz')
threshold={}
for s in SITES:
    z=load(D/'prepared'/f'{s}.npz');threshold[s]=np.quantile(z['raw'][z['time'].astype('datetime64[h]')<np.datetime64('2020-01-01'),0],.95)
out=R/'results';out.mkdir(exist_ok=True);rows=[];details=[];intervals=[];gates=[]
for role in ['test','h1','q3']:
    d=load(R/'prepared'/f'{role}.npz');anchor={'target':d['y'],'site':d['site'],'origin':d['origin']}
    bank={n:align(load(R/'predictions'/role/f'{n}.npz'),anchor)['prediction'] for n in names};p=np.stack(list(bank.values()),-1)
    if 'weights' in obj:
        sid=np.array([SITES.index(str(s)) for s in d['site']]);fused=pool(p,np.array(obj['weights']),sid)
    else:
        fused,w=apply_gate(p,d,obj['path']);gates.append({'period':role,'mean_weights':w.mean(0).tolist()});np.savez_compressed(out/f'{role}_gate_weights.npz',weight=w,site=d['site'],origin=d['origin'])
    bank.update(apply_controls(d,bank['LightGBM'],ctrl));bank['Uniform_pool']=p.mean(-1);bank['Selected_fusion']=fused
    high=d['y']>np.array([threshold[str(s)] for s in d['site']])[:,None]
    for name,forecast in bank.items():
        assert np.isfinite(forecast).all() and (np.diff(forecast,axis=-1)>=-1e-6).all()
        save_predictions(out/f'{role}_{name}.npz',forecast,d)
        row={'period':role,'model':name,**metrics(forecast,d['y'],high)};rows.append(row)
        for j,h in enumerate(H):
            for s in SITES:
                ix=d['site']==s;details.append({'period':role,'model':name,'site':s,'horizon':int(h),**metrics(forecast[ix,j:j+1],d['y'][ix,j:j+1],high[ix,j:j+1])})
        if name!='LightGBM':
            diff=(pin(forecast,d['y'])-pin(bank['LightGBM'],d['y'])).mean((1,2))
            for days in [7,28]:intervals.append({'period':role,'model':name,'reference':'LightGBM',**block_ci(diff,d['origin'],days)})
    for name in names+['LightGBM_calibrated','Uniform_pool']:
        diff=(pin(fused,d['y'])-pin(bank[name],d['y'])).mean((1,2))
        for days in [7,28]:intervals.append({'period':role,'model':'Selected_fusion','reference':name,**block_ci(diff,d['origin'],days)})
    print('SCORED',role,{r['model']:round(r['pinball'],8) for r in rows if r['period']==role},flush=True)
pd.DataFrame(rows).to_csv(out/'scores.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(details).to_csv(out/'scores_by_site_horizon.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(intervals).drop_duplicates(['period','model','reference','block_days']).to_csv(out/'paired_block_intervals.csv',index=False,encoding='utf-8-sig')
dump(out/'evaluation.json',{'timestamp_utc':now(),'selection_sha256':hashlib.sha256((R/'selection.json').read_bytes()).hexdigest(),'roles':{'test':'2022-2025 previously inspected, exploratory','h1':'2026 H1 previously inspected, exploratory','q3':'2026 Q3 held out from this fusion selection, one-pass confirmation'},'source_finality':'Source metadata does not identify expver. Recent2026 values are a fixed downloaded snapshot; final ERA5 release status is unconfirmed.','highwave_threshold_by_site':{s:float(v) for s,v in threshold.items()},'gates':gates,'no_test_driven_retuning':True})
