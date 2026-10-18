"""All figures regenerate from exported scores, weights and predictions."""
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from common import *
O=R/'results';figdir=O/'figures';figdir.mkdir(exist_ok=True)
df=pd.read_csv(O/'scores_mean_std.csv');detail=pd.read_csv(O/'site_horizon_by_seed.csv');selection=json.loads((R/'selection.json').read_text())
periods=['test','h1','q3'];titles=['2022–2025 (exploratory)','2026 H1 (exploratory)','2026 Q3 (confirmation)']
methods=['LightGBM','LightGBM_calibrated','Neural3','Chronos_adapter','Chronos_native','XGBoost','CatBoost_RMSEcal','TabM','Uniform_pool','Selected_fusion']
labels={'LightGBM':'LightGBM','LightGBM_calibrated':'Calibrated LightGBM','Neural3':'Multiscale neural ensemble','Chronos_adapter':'Chronos adapter','Chronos_native':'Chronos-2 native','XGBoost':'XGBoost quantile','CatBoost_RMSEcal':'CatBoost + residual quantiles','TabM':'TabM quantile','Uniform_pool':'Uniform aggregation','Selected_fusion':'Selected aggregation'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'era5-fusion-20261017'})
def save(fig,name):
    for ext in ['png','pdf','svg']:
        meta={'Date':None} if ext=='svg' else ({'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
        fig.savefig(figdir/f'{name}.{ext}',dpi=300,metadata=meta)
    plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,5.2),sharey=True,layout='constrained');allgains=[]
for ax,role,title in zip(axes,periods,titles):
    z=df[df.period==role].set_index('model');base=z.loc['LightGBM','pinball_mean']
    gains=np.array([100*(base-z.loc[n,'pinball_mean'])/base for n in methods]);allgains.extend(gains);sd=np.array([100*z.loc[n,'pinball_std']/base for n in methods]);yy=np.arange(len(methods))
    ax.axvline(0,color='.4',lw=1);ax.barh(yy,gains,color=['#be5c45' if g<0 else '#347b8b' for g in gains],height=.58,alpha=.85)
    ax.errorbar(gains,yy,xerr=sd,fmt='none',ecolor='#172d3b',capsize=2);ax.set_title(title,fontsize=11);ax.set_xlabel('Pinball reduction relative to LightGBM (%)');ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
for ax in axes:ax.set_xlim(np.floor(min(allgains))-1,np.ceil(max(allgains))+1)
axes[0].set_yticks(np.arange(len(methods)),[labels[n] for n in methods]);axes[0].invert_yaxis();fig.text(.5,-.02,'Error bars show sample SD over three new expert training seeds.',ha='center',fontsize=8);save(fig,'model_comparison')
matrices=[]
for role in periods:
    z=detail[detail.period==role].groupby(['model','site','horizon']).pinball.mean();a=np.array([[100*(z['LightGBM',s,h]-z['Selected_fusion',s,h])/z['LightGBM',s,h] for h in H] for s in SITES]);matrices.append(a)
bound=max(1,float(np.ceil(max(np.abs(a).max() for a in matrices))));fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True,layout='constrained')
for ax,a,title in zip(axes,matrices,titles):
    im=ax.imshow(a,cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto');ax.set_xticks(range(4),[str(h) for h in H]);ax.set_yticks(range(6),SITES);ax.set_xlabel('Forecast lead (h)');ax.set_title(title,fontsize=11)
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{a[i,j]:+.1f}',ha='center',va='center',color='white' if abs(a[i,j])>bound*.55 else '#18242d',fontsize=10)
fig.colorbar(im,ax=axes,label='Pinball reduction relative to LightGBM (%)',shrink=.85);save(fig,'site_horizon_gain')
obj=selection['object']
if 'weights' in obj:
    w=np.array(obj['weights']);w=np.repeat(w[None],4,axis=0) if w.ndim==1 else w
    if w.ndim==3:w=w.mean(0)
else:
    z=np.load(O/'test_gate_weights.npz');w=z['weight'].mean(0)
fig,ax=plt.subplots(figsize=(11,3.2),layout='constrained');im=ax.imshow(w,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(4),[f'{h} h' for h in H]);ax.set_xticks(range(len(selection['members'])),[labels[n] for n in selection['members']],rotation=22,ha='right');ax.set_title('Frozen aggregation weights' if 'weights' in obj else 'Mean state gate weights on 2022–2025 forecasts (seed 17)')
for i in range(4):
    for j in range(w.shape[1]):ax.text(j,i,'<0.01' if 0<w[i,j]<.01 else f'{w[i,j]:.2f}',ha='center',va='center',color='white' if w[i,j]>.55 else '#1a2c35')
fig.colorbar(im,ax=ax,label='Weight',shrink=.8);save(fig,'expert_weights')
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');colors={'LightGBM':'#6d7480','TabM':'#d6903c','Selected_fusion':'#286f87'}
for ax,role,title in zip(axes,periods,titles):
    z=df[df.period==role].set_index('model');ax.axvline(80,color='.55',ls=':',lw=1.2)
    for name,color in colors.items():
        x=100*z.loc[name,'coverage80_mean'];y=z.loc[name,'width80_mean'];ax.scatter(x,y,s=65,color=color,label=labels[name])
    ax.set_title(title,fontsize=11);ax.set_xlabel('Empirical 80% interval coverage (%)');ax.set_ylabel('Mean interval width (m)');ax.grid(alpha=.15)
handles,legendlabels=axes[0].get_legend_handles_labels();fig.legend(handles,legendlabels,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=3,frameon=False);save(fig,'coverage_width')
dump(figdir/'figure_provenance.json',{'timestamp_utc':now(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_scores_sha256':hashlib.sha256((O/'scores_mean_std.csv').read_bytes()).hexdigest(),'error_bars':'Sample SD over three new expert training seeds; not a temporal confidence interval','heatmap':'Single shared symmetric color normalization across all three periods','period_status':dict(zip(periods,titles))})
print('FIGURES',figdir,flush=True)
