import pandas as pd
from common import *

def metrics(p,y,high):
    loss=pin(p,y);lo=p[...,0];up=p[...,-1];width=up-lo
    return {'pinball':float(loss.mean()),'median_MAE':float(np.abs(p[...,4]-y).mean()),'coverage80':float(((y>=lo)&(y<=up)).mean()),'width80':float(width.mean()),'interval_score80':float((width+10*np.maximum(lo-y,0)+10*np.maximum(y-up,0)).mean()),'highwave_count':int(high.sum()),'highwave_MAE':float(np.abs(p[...,4]-y)[high].mean()) if high.any() else np.nan,'highwave_q90_loss':float(loss[...,-1][high].mean()) if high.any() else np.nan}

def block_ci(diff,origin,days):
    v=pd.DataFrame({'origin':origin.astype('datetime64[ns]'),'diff':diff}).groupby('origin',sort=True)['diff'].mean().to_numpy();length=days*4;n=len(v);nb=int(np.ceil(n/length));rng=np.random.default_rng(20261017+days);values=[]
    for _ in range(20):
        starts=rng.integers(0,n,size=(100,nb));ix=(starts[:,:,None]+np.arange(length))%n;values.extend(v[ix.reshape(100,-1)[:,:n]].mean(1))
    lo,hi=np.quantile(values,[.025,.975]);return {'difference':float(v.mean()),'ci_low':float(lo),'ci_high':float(hi),'block_days':days,'replicates':2000,'distinct_origins':n,'method':'circular moving time blocks; all sites synchronized'}
