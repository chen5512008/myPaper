from pathlib import Path
import json,math
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
V=Path(__file__).parent;R=V.parent;D=R.parent;O=V/'results';O.mkdir(exist_ok=True)
SITES=json.loads((D/'protocol.json').read_text())['sites'];H=[6,12,24,48];Q=np.arange(1,10)/10
ckpt=torch.load(R/'runs/pooled_empirical_reference_17/model.pt',map_location='cpu',weights_only=False)
residual=np.array(ckpt['empirical_reference'],dtype='float32')
raw=[]
for site in SITES:
    f=dict(np.load(D/'forecasts'/f'{site}_test_base.npz'));z=np.load(D/'prepared'/f'{site}.npz')
    last=z['raw'][f['indices'],0]
    raw.append({'prediction':np.maximum(0,last[:,None,None]+residual[None]),'target':f['target'],'origin':f['origin'],'site':f['site']})
unadapted={k:np.concatenate([a[k] for a in raw]) for k in raw[0]}
np.savez_compressed(O/'empirical_unadapted.npz',**unadapted)
models={'Empirical unadapted':unadapted}
for label,path in [('Chronos adapter',R/'runs/pooled_full_17/complete.npz'),('Empirical adapter',R/'runs/pooled_empirical_reference_17/complete.npz'),('LCA',D/'adapters/pooled_full_17/complete.npz'),('LightGBM',D/'tabular/test.npz')]:
    models[label]=dict(np.load(path))
for label,folder in [('Chronos-2 native','forecasts'),('TimesFM 3.0','timesfm3')]:
    a=[dict(np.load(D/folder/f'{s}_test_native.npz')) for s in SITES]
    models[label]={k:np.concatenate([z[k] for z in a]) for k in ['prediction','target','origin','site']}
for p in (V/'runs').glob('pooled_*'):
    if (p/'run.json').exists():models[p.name.removeprefix('pooled_')]=dict(np.load(p/'complete.npz'))
# Fixed ordering and explicit target/origin alignment for all paired analyses.
anchor=models['Empirical adapter']
for name,d in models.items():
    lookup={(str(s),str(o)):i for i,(s,o) in enumerate(zip(d['site'],d['origin']))}
    ids=np.array([lookup[(str(s),str(o))] for s,o in zip(anchor['site'],anchor['origin'])])
    models[name]={k:np.asarray(d[k])[ids] for k in ['prediction','target','origin','site']}
    assert np.allclose(models[name]['target'],anchor['target'])

def pin(p,y):
    e=y[...,None]-p
    return np.maximum(Q*e,(Q-1)*e)

def values(p,y,lo=None,hi=None):
    lo=p[:,0] if lo is None else lo;hi=p[:,-1] if hi is None else hi
    return {'pinball':float(pin(p,y).mean()),'mae':float(abs(p[:,4]-y).mean()),'bias':float((p[:,4]-y).mean()),
      'q90_loss':float(pin(p,y)[:,-1].mean()),'coverage':float(((y>=lo)&(y<=hi)).mean()),
      'width':float((hi-lo).mean()),'upper_miss':float((y>hi).mean()),
      'interval_score':float((hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)).mean())}
rows=[]
for name,d in models.items():
    for site in SITES:
        threshold=json.loads((D/'prepared'/f'{site}.json').read_text())['training_high_wave_threshold_m']
        s=d['site']==site
        for j,h in enumerate(H):
            p=d['prediction'][s,j];y=d['target'][s,j]
            for group,ix in [('all',np.ones(len(y),dtype=bool)),('high',y>=threshold)]:
                rows.append({'model':name,'site':site,'horizon':h,'subset':group,'n':int(ix.sum()),**values(p[ix],y[ix])})
scores=pd.DataFrame(rows);scores.to_csv(O/'scores_by_site.csv',index=False)
summary=scores.groupby(['model','subset'])[['pinball','mae','bias','q90_loss','coverage','width','upper_miss','interval_score']].mean()
summary.to_csv(O/'scores.csv')
ci=[]
for core,base in [('Empirical adapter','Empirical unadapted'),('Empirical adapter','LightGBM'),('Chronos adapter','Empirical unadapted')]:
    diff=[]
    for site in SITES:
        ix=anchor['site']==site
        diff.append(pin(models[core]['prediction'][ix],anchor['target'][ix]).mean(-1)-pin(models[base]['prediction'][ix],anchor['target'][ix]).mean(-1))
    delta=np.mean(diff,axis=0);n=len(delta);rng=np.random.default_rng(20261017);samples=[]
    for _ in range(2000):
        start=rng.integers(n,size=math.ceil(n/28));ix=((start[:,None]+np.arange(28))%n).ravel()[:n]
        samples.append(delta[ix].mean(0))
    for j,h in enumerate(H):
        ci.append({'core':core,'baseline':base,'horizon':h,'delta':float(delta[:,j].mean()),'low':float(np.quantile(samples,.025,axis=0)[j]),'high':float(np.quantile(samples,.975,axis=0)[j])})
pd.DataFrame(ci).to_csv(O/'paired_ci.csv',index=False)

calrows=[];calparams=[]
for name in ['Chronos-2 native','Chronos adapter','Empirical adapter','LightGBM']:
    d=models[name];year=pd.DatetimeIndex(d['origin']).year
    # Use target timestamps, not origins, to keep all calibration and evaluation outcomes disjoint.
    origins=pd.DatetimeIndex(d['origin'])
    for j,h in enumerate(H):
        target_times=origins+pd.Timedelta(hours=h)
        fit=(target_times>='2022-01-01')&(target_times<'2023-01-01')
        test=(target_times>='2023-01-01')&(target_times<'2026-01-01')
        p=d['prediction'][:,j];y=d['target'][:,j]
        score=np.maximum(p[fit,0]-y[fit],y[fit]-p[fit,-1])
        rank=min(len(score),math.ceil((len(score)+1)*.8))
        margin=max(0.,float(np.partition(score,rank-1)[rank-1]))
        calparams.append({'model':name,'horizon':h,'n':int(fit.sum()),'margin_m':margin,'fit_target_period':'2022','score_target_period':'2023-2025'})
        for site in SITES:
            ids=np.where(test&(d['site']==site))[0]
            threshold=json.loads((D/'prepared'/f'{site}.json').read_text())['training_high_wave_threshold_m']
            for state in ['raw','calibrated']:
                lo=np.maximum(0,p[ids,0]-(margin if state=='calibrated' else 0))
                hi=p[ids,-1]+(margin if state=='calibrated' else 0)
                for group,ix in [('all',np.ones(len(ids),dtype=bool)),('high',y[ids]>=threshold)]:
                    vals=values(p[ids][ix],y[ids][ix],lo[ix],hi[ix])
                    for k in ['pinball','mae','bias','q90_loss']:vals.pop(k)
                    calrows.append({'model':name,'horizon':h,'site':site,'state':state,'subset':group,'n':int(ix.sum()),**vals})
pd.DataFrame(calparams).to_csv(O/'calibration_parameters.csv',index=False)
cal=pd.DataFrame(calrows);cal.to_csv(O/'calibration_by_site.csv',index=False)
cal.groupby(['model','state','subset'])[['coverage','width','upper_miss','interval_score']].mean().to_csv(O/'calibration.csv')
result={'scores':summary.reset_index().to_dict('records'),'calibration':cal.groupby(['model','state','subset'])[['coverage','width','upper_miss','interval_score']].mean().reset_index().to_dict('records'),'completed_empirical_ablations':len(list((V/'runs').glob('*/run.json')))}
(O/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.unicode_minus':False})
for lang in ['en','zh']:
    zh=lang=='zh'
    names=['Empirical unadapted','Empirical adapter','Chronos adapter','LightGBM']
    labels=['未适配经验参考','经验参考加适配器','Chronos-2 加适配器','LightGBM'] if zh else names
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax,subset,key,title in [(axs[0],'all','pinball','总体分位数损失（米）' if zh else 'Aggregate pinball loss (m)'),(axs[1],'high','q90_loss','高波浪 0.9 分位数损失（米）' if zh else 'High wave 0.9 quantile loss (m)')]:
        ax.barh(range(4),[summary.loc[(n,subset),key] for n in names],color=['#9eafbc','#8e6aae','#147d77','#d18d56'])
        ax.set(yticks=range(4),yticklabels=labels,xlabel=title);ax.invert_yaxis()
    for ext in ['png','svg','pdf']:fig.savefig(O/f'reference_utility_{lang}.{ext}',dpi=230,bbox_inches='tight')
    plt.close(fig)
    c=cal.groupby(['model','state','subset'])[['coverage','width']].mean()
    calibration_names=['Chronos-2 native','Chronos adapter','Empirical adapter','LightGBM']
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained');xx=np.arange(4)
    for state,offset,col,label in [('raw',-.17,'#829eae','原始区间' if zh else 'Raw intervals'),('calibrated',.17,'#147d77','校准区间' if zh else 'Calibrated intervals')]:
        for ax,key in zip(axs,['coverage','width']):
            ax.bar(xx+offset,[c.loc[(n,state,'all'),key]*(100 if key=='coverage' else 1) for n in calibration_names],.34,color=col,label=label)
            ax.set_xticks(xx,['原生 Chronos-2','Chronos 适配','经验参考适配','LightGBM'] if zh else ['Native Chronos-2','Chronos adapter','Empirical adapter','LightGBM'],rotation=15,ha='right')
    axs[0].axhline(80,color='#b24c57',ls='--',lw=1);axs[0].set_ylabel('覆盖率（%）' if zh else 'Coverage (%)');axs[1].set_ylabel('区间宽度（米）' if zh else 'Interval width (m)');axs[0].legend(fontsize=8)
    for ext in ['png','svg','pdf']:fig.savefig(O/f'calibration_followup_{lang}.{ext}',dpi=230,bbox_inches='tight')
    plt.close(fig)
print(summary.to_string())
print('Calibration 2023-2025',cal.groupby(['model','state','subset'])[['coverage','width','upper_miss','interval_score']].mean().to_string())
