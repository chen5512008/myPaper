from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent;D=R.parent;O=R/'results';O.mkdir(exist_ok=True)
SITES=json.loads((D/'protocol.json').read_text())['sites'];H=[6,12,24,48];Q=np.arange(1,10)/10
def loss(p,y):
    e=y[...,None]-p;return np.maximum(e*Q,e*(Q-1)).mean(-1)
def readparts(folder,kind):
    ds=[dict(np.load(D/folder/f'{s}_test_{kind}.npz')) for s in SITES]
    return {k:np.concatenate([d[k] for d in ds]) for k in ['prediction','target','site','origin']}
models={'Chronos-2 native':readparts('forecasts','native'),'TimesFM 3.0':readparts('timesfm3','native'),'LightGBM':dict(np.load(D/'tabular/test.npz')),'Old adapter':dict(np.load(D/'adapters/pooled_full_17/complete.npz')),'Old masked adapter':dict(np.load(D/'adapters/pooled_masked_core_17/complete.npz'))}
for folder in sorted((R/'runs').glob('pooled_*')):
    if (folder/'run.json').exists():models[folder.name.removeprefix('pooled_')]=dict(np.load(folder/'complete.npz'))
rows=[];hirows=[]
for name,d in models.items():
    for s in SITES:
        z=d['site']==s;p=d['prediction'][z];y=d['target'][z]
        threshold=json.loads((D/'prepared'/f'{s}.json').read_text())['training_high_wave_threshold_m']
        for j,h in enumerate(H):
            rows.append(dict(model=name,site=s,horizon=h,pinball=float(loss(p,y)[:,j].mean()),mae=float(abs(p[:,j,4]-y[:,j]).mean()),coverage=float(((y[:,j]>=p[:,j,0])&(y[:,j]<=p[:,j,-1])).mean()),width=float((p[:,j,-1]-p[:,j,0]).mean()),crossing=float((np.diff(p[:,j],axis=-1)<-1e-7).mean())))
            ix=y[:,j]>=threshold
            hirows.append(dict(model=name,site=s,horizon=h,n=int(ix.sum()),mae=float(abs(p[ix,j,4]-y[ix,j]).mean()),bias=float((p[ix,j,4]-y[ix,j]).mean())))
metric=pd.DataFrame(rows);metric.to_csv(O/'metrics.csv',index=False)
overall=metric.groupby('model')[['pinball','mae','coverage','width','crossing']].mean();overall.to_csv(O/'overall.csv')
high=pd.DataFrame(hirows);high.to_csv(O/'high_wave_by_site.csv',index=False);high.groupby('model')[['mae','bias']].mean().to_csv(O/'high_wave.csv')
stress=[]
for key,folder in [('Old adapter',D/'adapters/pooled_full_17'),('Old masked adapter',D/'adapters/pooled_masked_core_17')]+[(p.name.removeprefix('pooled_'),p) for p in (R/'runs').glob('pooled_*') if (p/'run.json').exists()]:
    for scenario in ['complete','random30','last24','all_missing']:
        d=np.load(folder/f'{scenario}.npz');stress.append({'model':key,'scenario':scenario,'pinball':float(loss(d['prediction'],d['target']).mean())})
pd.DataFrame(stress).to_csv(O/'stress.csv',index=False)
transfer=[]
for s in SITES:
    path=R/'runs'/f'holdout_{s}_full_17/complete.npz'
    if not (path.parent/'run.json').exists():continue
    d=np.load(path);old=np.load(D/'adapters'/f'holdout_{s}_full_17/complete.npz')
    for j,h in enumerate(H):transfer.append(dict(site=s,horizon=h,new=float(loss(d['prediction'],d['target'])[:,j].mean()),old=float(loss(old['prediction'],old['target'])[:,j].mean()),native=float(metric.query('model=="Chronos-2 native" and site==@s and horizon==@h').iloc[0].pinball)))
pd.DataFrame(transfer).to_csv(O/'transfer.csv',index=False)
if 'full_17' in models:
    paired=[]
    for comparator in ['Old adapter','Chronos-2 native','LightGBM','affine_17','no_availability_17','no_distribution_query_17','empirical_reference_17']:
        if comparator not in models:continue
        core=models['full_17'];base=models[comparator];diff=[]
        for s in SITES:
            inds=np.where(core['site']==s)[0];mapping={str(o):i for i,o in enumerate(base['origin']) if base['site'][i]==s}
            ix=np.array([mapping[str(core['origin'][i])] for i in inds])
            if not np.allclose(core['target'][inds],base['target'][ix]):raise ValueError('Origin alignment mismatch')
            diff.append(loss(core['prediction'][inds],core['target'][inds])-loss(base['prediction'][ix],base['target'][ix]))
        delta=np.mean(diff,axis=0);rng=np.random.default_rng(20261017);n=len(delta);bs=[]
        for _ in range(2000):
            starts=rng.integers(n,size=int(np.ceil(n/28)));ii=((starts[:,None]+np.arange(28))%n).ravel()[:n];bs.append(delta[ii].mean(0))
        for j,h in enumerate(H):paired.append(dict(comparator=comparator,horizon=h,delta=float(delta[:,j].mean()),low=float(np.quantile(bs,.025,axis=0)[j]),high=float(np.quantile(bs,.975,axis=0)[j])))
    pd.DataFrame(paired).to_csv(O/'paired_ci.csv',index=False)

plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.unicode_minus':False})
labels={'full_17':'Proposed','Old adapter':'Previous adapter','Old masked adapter':'Previous masked','Chronos-2 native':'Chronos-2','TimesFM 3.0':'TimesFM 3.0','LightGBM':'LightGBM','affine_17':'Shared scale','no_availability_17':'No availability','no_distribution_query_17':'No distribution query','empirical_reference_17':'Empirical reference','legacy_matched_17':'Previous matched training'}
def save(fig,name,lang):
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(O/f'{name}_{lang}.{ext}',dpi=220)
    plt.close(fig)
for lang in ['en','zh']:
    zh=lang=='zh'
    names=[n for n in ['Chronos-2 native','TimesFM 3.0','Old adapter','full_17','empirical_reference_17','LightGBM'] if n in overall.index]
    fig,axs=plt.subplots(1,3,figsize=(11,3.5))
    hi=high.groupby('model').mae.mean()
    for ax,vals,title in zip(axs,[overall.pinball,overall.mae,hi],['分位数损失（米）' if zh else 'Pinball loss (m)','中位数误差（米）' if zh else 'Median MAE (m)','高波浪误差（米）' if zh else 'High wave MAE (m)']):
        ax.barh(range(len(names)),[vals[n] for n in names],color=['#a9bbc8','#719ca9','#ac9c80','#187f78','#9a83b5','#df9968'][:len(names)]);ax.set_yticks(range(len(names)),[labels.get(n,n) for n in names]);ax.invert_yaxis();ax.set_xlabel(title)
    save(fig,'performance',lang)
    names=[n for n in ['full_17','affine_17','no_availability_17','no_distribution_query_17','empirical_reference_17','legacy_matched_17'] if n in overall.index]
    fig,ax=plt.subplots(figsize=(9,3.3));ax.barh(range(len(names)),[overall.loc[n,'pinball'] for n in names],color=['#187f78']+['#829eae']*(len(names)-1));ax.set_yticks(range(len(names)),[labels[n] for n in names]);ax.invert_yaxis();ax.set_xlabel('分位数损失（米），越低越好' if zh else 'Pinball loss (m), lower is better');save(fig,'ablation',lang)
    fig,ax=plt.subplots(figsize=(9,3.6));st=pd.DataFrame(stress)
    for name in ['Old adapter','Old masked adapter','full_17','no_availability_17']:
        z=st[st.model==name].set_index('scenario')
        if len(z):ax.plot(range(4),z.loc[['complete','random30','last24','all_missing'],'pinball'],'o-',label=labels[name])
    ax.set_xticks(range(4),['完整','随机缺失30%','末端24小时','全部气象缺失'] if zh else ['Complete','Random 30%','Last 24 h','All weather missing']);ax.set_ylabel('Pinball (m)');ax.legend(fontsize=8);save(fig,'missingness',lang)
    fig,axs=plt.subplots(2,3,figsize=(10,5.8))
    for s,ax in zip(SITES,axs.flat):
        ax.plot(Q,Q,':',color='gray')
        for name in ['Chronos-2 native','Old adapter','full_17','LightGBM']:
            if name not in models:continue
            d=models[name];ix=d['site']==s;ax.plot(Q,(d['target'][ix,...,None]<=d['prediction'][ix]).mean((0,1)),label=labels[name])
        ax.set_title(s);ax.set(xlabel='名义概率' if zh else 'Nominal probability',ylabel='经验概率' if zh else 'Empirical probability')
    axs[0,0].legend(fontsize=6);save(fig,'calibration',lang)
    if transfer:
        tab=pd.DataFrame(transfer).groupby('site')[['native','old','new']].mean();names=[s for s in SITES if s in tab.index];fig,ax=plt.subplots(figsize=(9,3.6));xx=np.arange(len(names))
        for k,off,color,label in [('native',-.25,'#a9bbc8','Chronos-2'),('old',0,'#ac9c80','Previous adapter'),('new',.25,'#187f78','Proposed')]:ax.bar(xx+off,tab.loc[names,k],.25,color=color,label=label)
        ax.set_xticks(xx,names);ax.set_ylabel('Pinball (m)');ax.legend(ncol=3);save(fig,'transfer',lang)
summary={'overall':overall.reset_index().to_dict('records'),'high_wave':high.groupby('model')[['mae','bias']].mean().reset_index().to_dict('records'),'stress':stress,'spatial_folds_complete':len(transfer)//4}
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(overall.to_string())
