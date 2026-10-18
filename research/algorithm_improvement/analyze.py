"""All configurations retained; paired time-block intervals, no model selection."""
from pathlib import Path
import json,math,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent;D=R.parent;O=R/'results';O.mkdir(exist_ok=True)
SITES=json.loads((D/'protocol.json').read_text())['sites'];H=[6,12,24,48];Q=np.arange(1,10)/10
selection=json.loads((R/'selection.json').read_text());mode=selection['variant']
def load(path):return dict(np.load(path))
old={}
for s in [17,29,43]:old[f'Original empirical seed{s}']=load(D/f'transformer_revision/runs/pooled_empirical_reference_{s}/complete.npz')
old['Original Chronos']=load(D/'transformer_revision/runs/pooled_full_17/complete.npz')
old['LightGBM original']=load(D/'tabular/test.npz');old['LightGBM extended']=load(R/'strong_lightgbm/test.npz')
for label,folder in [('Chronos native','forecasts'),('TimesFM 3.0','timesfm3')]:
    pieces=[load(D/folder/f'{s}_test_native.npz') for s in SITES]
    old[label]={k:np.concatenate([a[k] for a in pieces]) for k in ['prediction','target','site','origin']}
for run in sorted((R/'runs').glob('*')):
    if (run/'complete.npz').exists():old[run.name]=load(run/'complete.npz')
new={}
mapping={'selected_17':f'{mode}_17','selected_29':f'{mode}_29','selected_43':f'{mode}_43','LightGBM_original':'LightGBM original','LightGBM_extended':'LightGBM extended','original_empirical':'Original empirical seed17','original_chronos':'Original Chronos','Chronos_native':'Chronos native'}
mapping.update({f'original_empirical_{s}':f'Original empirical seed{s}' for s in [29,43]})
confirmation_done=(R/'confirmation_2026/completed.json').exists()
if confirmation_done:
    for stem,label in mapping.items():new[label]=load(R/f'confirmation_2026/predictions/{stem}.npz')
period_models=[('2022-2025',old)]+([('2026-H1',new)] if confirmation_done else [])
def align(models):
    anchor=next(iter(models.values()))
    for name,z in models.items():
        lookup={(str(s),str(t)):i for i,(s,t) in enumerate(zip(z['site'],z['origin']))};assert len(lookup)==len(z['site'])
        ids=[lookup[(str(s),str(t))] for s,t in zip(anchor['site'],anchor['origin'])]
        models[name]={k:np.asarray(z[k])[ids] for k in ['prediction','target','site','origin']}
        assert np.allclose(models[name]['target'],anchor['target'])
        p=models[name]['prediction'];assert np.isfinite(p).all() and (p>=0).all() and (np.diff(p,axis=-1)>=-1e-6).all()
    return anchor
def loss(p,y):
    e=y[...,None]-p;return np.maximum(Q*e,(Q-1)*e)
def metrics(p,y):
    lo=p[:,0];hi=p[:,-1];med=p[:,4];l=loss(p,y)
    return {'pinball':float(l.mean()),'mae':float(abs(med-y).mean()),'bias':float((med-y).mean()),'coverage':float(((lo<=y)&(y<=hi)).mean()),'width':float((hi-lo).mean()),'interval_score':float((hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)).mean()),'q90_loss':float(l[:,-1].mean())}
rows=[];cis=[];audits=[]
for period,models in period_models:
    anchor=align(models)
    for name,z in models.items():
        for s in SITES:
            threshold=json.loads((D/f'prepared/{s}.json').read_text())['training_high_wave_threshold_m'];ix=z['site']==s
            for j,h in enumerate(H):
                p=z['prediction'][ix,j];y=z['target'][ix,j]
                for group,valid in [('all',np.ones(len(y),dtype=bool)),('high',y>=threshold)]:
                    rows.append({'period':period,'model':name,'site':s,'horizon':h,'subset':group,'n':int(valid.sum()),**metrics(p[valid],y[valid])})
        audits.append({'period':period,'model':name,'n':len(z['target']),'minimum_prediction':float(z['prediction'].min()),'minimum_quantile_gap':float(np.diff(z['prediction'],axis=-1).min())})
    for baseline in ['LightGBM original','LightGBM extended','Original empirical seed17','Original Chronos']:
        delta=[]
        for s in SITES:
            ix=anchor['site']==s
            ns=np.stack([loss(models[f'{mode}_{seed}']['prediction'][ix],anchor['target'][ix]).mean(-1) for seed in [17,29,43]])
            delta.append(ns.mean(0)-loss(models[baseline]['prediction'][ix],anchor['target'][ix]).mean(-1))
        d=np.mean(delta,0);d=np.column_stack([d,d.mean(-1)]);n=len(d)
        for block in [28,112]:
            rng=np.random.default_rng(20261017);stats=[]
            for _ in range(2000):
                starts=rng.integers(n,size=math.ceil(n/block));ix=((starts[:,None]+np.arange(block))%n).ravel()[:n];stats.append(d[ix].mean(0))
            low,high=np.quantile(stats,[.025,.975],axis=0)
            for j,h in enumerate(H+['all']):cis.append({'period':period,'baseline':baseline,'horizon':h,'block_origins':block,'delta_new_minus_baseline':float(d[:,j].mean()),'low':float(low[j]),'high':float(high[j])})
scores=pd.DataFrame(rows);scores.to_csv(O/'scores_by_site.csv',index=False)
keys=['pinball','mae','bias','coverage','width','interval_score','q90_loss'];summary=scores.groupby(['period','model','subset'])[keys].mean().reset_index();summary.to_csv(O/'scores.csv',index=False)
pd.DataFrame(cis).to_csv(O/'paired_block_intervals.csv',index=False)
pd.DataFrame(audits).to_csv(O/'prediction_checks.csv',index=False)
robust=[]
for run in sorted((R/'runs').glob('*')):
    if not (run/'test_scores.json').exists():continue
    for scenario,value in json.loads((run/'test_scores.json').read_text()).items():robust.append({'model':run.name,'scenario':scenario,'pinball':value})
pd.DataFrame(robust).to_csv(O/'missingness.csv',index=False)
# Report training variability explicitly; mean scores are not an ensemble forecast.
aggregate=[]
for period,_ in period_models:
    for subset in ['all','high']:
        part=summary[(summary.period==period)&(summary.subset==subset)&summary.model.isin([f'{mode}_{s}' for s in [17,29,43]])]
        aggregate.append({'period':period,'subset':subset,'seeds':[17,29,43],'mean':part[keys].mean().to_dict(),'sample_sd':part[keys].std(ddof=1).to_dict(),'values':part.to_dict('records')})
(O/'summary.json').write_text(json.dumps({'selection':selection,'seed_summary':aggregate,'scores':summary.to_dict('records'),'all_configurations_included':True,'not_ensemble':True},indent=2))
chosen=json.loads((R/'strong_lightgbm/selection.json').read_text())['chosen'];chosen='LightGBM '+chosen
checks=[]
for period,_ in period_models:
    z=summary[(summary.period==period)&(summary.subset=='all')].set_index('model')
    vals=np.array([z.loc[f'{mode}_{s}','pinball'] for s in [17,29,43]])
    for baseline in ['LightGBM original','LightGBM extended','Original empirical seed17']:
        checks.append({'period':period,'check':f'mean beats {baseline}','passed':bool(vals.mean()<z.loc[baseline,'pinball'])})
    checks.append({'period':period,'check':'each seed beats validation-selected LightGBM','passed':bool((vals<z.loc[chosen,'pinball']).all())})
    checks.append({'period':period,'check':'at least 1 percent mean improvement','passed':bool((1-vals.mean()/z.loc[chosen,'pinball'])>=.01)})
    for block in [28,112]:
        c=next(c for c in cis if c['period']==period and c['baseline']==chosen and c['horizon']=='all' and c['block_origins']==block)
        checks.append({'period':period,'check':f'CI below zero, block {block}','passed':c['high']<0})
checks.append({'period':'2026-H1','check':'reserved chronological evaluation completed','passed':confirmation_done})
(O/'decision.json').write_text(json.dumps({'rule':json.loads((R/'decision_rule.json').read_text()),'selected_baseline':chosen,'checks':checks,'performance_gate_passed':all(c['passed'] for c in checks),'attention_selected':'_only' not in mode,'structural_improvement_selected':('_control' not in mode or '_region_' in mode),'paper_update_allowed_by_performance_gate':all(c['passed'] for c in checks)},indent=2))
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.unicode_minus':False})
fig,axs=plt.subplots(1,len(period_models),figsize=(5*len(period_models),4),layout='constrained',squeeze=False);axs=axs.ravel()
labels=['Original Chronos','Original empirical seed17','LightGBM original','LightGBM extended'];short=['原 Chronos 适配','原经验参考适配','原 LightGBM','加强 LightGBM']
for ax,(period,_) in zip(axs,period_models):
    z=summary[(summary.period==period)&(summary.subset=='all')].set_index('model');vals=[z.loc[k,'pinball'] for k in labels];seeds=[z.loc[f'{mode}_{s}','pinball'] for s in [17,29,43]]
    ax.barh(np.arange(5),vals+[np.mean(seeds)],color=['#a8b5c5','#9eaeba','#d9a85d','#bc7d3f','#147d77']);ax.errorbar(np.mean(seeds),4,xerr=np.std(seeds,ddof=1),fmt='none',ecolor='black',capsize=4)
    for j,v in enumerate(vals+[np.mean(seeds)]):ax.text(v+.001,j,f'{v:.5f}',ha='left',va='center',fontsize=9)
    ax.set_xlim(0,max(vals+[np.mean(seeds)])*1.2)
    ax.set(yticks=range(5),yticklabels=short+['验证集选中模型（三种子）'],xlabel='平均分位数损失（米，越小越好）',title=period);ax.invert_yaxis()
fig.text(.5,-.025,'误差线：三个训练种子指标的标准差。各时间段按各自样本评价。',ha='center',va='top',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(O/f'baseline_comparison.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
gain_ranges=[]
for period,_ in period_models:
    zz=scores[(scores.period==period)&(scores.subset=='all')]
    bb=zz[zz.model=='LightGBM extended'].pivot(index='site',columns='horizon',values='pinball').loc[SITES,H]
    mm=zz[zz.model.isin([f'{mode}_{s}' for s in [17,29,43]])].groupby(['site','horizon']).pinball.mean().unstack().loc[SITES,H]
    gain_ranges.append(float(abs(100*(bb-mm)/bb).to_numpy().max()))
shared_limit=max(5,math.ceil(max(gain_ranges)))
fig,axs=plt.subplots(1,len(period_models),figsize=(5*len(period_models),3.7),layout='constrained',squeeze=False);axs=axs.ravel()
for ax,(period,_) in zip(axs,period_models):
    z=scores[(scores.period==period)&(scores.subset=='all')];base=z[z.model=='LightGBM extended'].pivot(index='site',columns='horizon',values='pinball').loc[SITES,H]
    model=z[z.model.isin([f'{mode}_{s}' for s in [17,29,43]])].groupby(['site','horizon']).pinball.mean().unstack().loc[SITES,H]
    gain=100*(base-model)/base;im=ax.imshow(gain,cmap='RdYlGn',vmin=-shared_limit,vmax=shared_limit)
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{gain.iloc[i,j]:.1f}',ha='center',va='center',fontsize=9)
    ax.set(xticks=range(4),xticklabels=H,yticks=range(6),yticklabels=SITES,xlabel='预测提前量（小时）',title=period);fig.colorbar(im,ax=ax,label='相对加强 LightGBM 的改进（%）')
for ext in ['png','svg','pdf']:fig.savefig(O/f'site_horizon_gain.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
print(summary[summary.subset=='all'][['period','model','pinball','mae']].to_string(index=False))
print('SEED_SUMMARY',json.dumps(aggregate,ensure_ascii=False))
print(pd.DataFrame(cis).query("horizon == 'all'").to_string(index=False))
