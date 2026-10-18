"""Recreate the SCI revision figures entirely from exported CSV tables."""
from pathlib import Path
import os
os.environ['SOURCE_DATE_EPOCH']='1792281600'
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;out=R/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':10.5,'axes.titlesize':12,'svg.hashsalt':'era5-sci4-20261018','pdf.fonttype':42,'savefig.dpi':300})
colors={'lightgbm':'#4477AA','tabm':'#AA4499','tabm_flags':'#CC6677','adapter':'#228833'}
def save(fig,name):
    for suffix in ['png','pdf','svg']:
        metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else {'Date':None} if suffix=='svg' else None
        fig.savefig(out/f'{name}.{suffix}',bbox_inches='tight',metadata=metadata)
    plt.close(fig)
for lang in ['en','zh']:
    plt.rcParams['font.family']='Microsoft YaHei' if lang=='zh' else 'DejaVu Sans'
    if (R/'evaluation/summary.csv').exists():
        df=pd.read_csv(R/'evaluation/summary.csv');byseed=pd.read_csv(R/'evaluation/scores_by_seed.csv');scenarios=['complete','random30','last24','all_missing']
        labels=['Complete','Random\n30%','Last\n24 h','All weather\nabsent'] if lang=='en' else ['完整输入','随机缺测\n30%','末端缺测\n24小时','气象\n全部缺测']
        names={'lightgbm':'LightGBM (1 run)','tabm_flags':'TabM + missing flags (3 seeds)','adapter':'Adapter (3 seeds)'} if lang=='en' else {'lightgbm':'LightGBM（1次）','tabm_flags':'显式缺测TabM（3种子）','adapter':'适配器（3种子）'}
        fig,axs=plt.subplots(1,2,figsize=(8.2,3.5),gridspec_kw={'wspace':.32})
        for name in names:
            sub=df[df.model==name].set_index('scenario').loc[scenarios];v=sub.pinball_mean.to_numpy();sd=sub.pinball_sd.fillna(0).to_numpy()
            paired=byseed[byseed.model==name].pivot(index='seed',columns='scenario',values='pinball')[scenarios];increments=paired.subtract(paired['complete'],axis=0)
            for j in range(2):
                y=v if j==0 else increments.mean(0).to_numpy();err=sd if j==0 else increments.std(0).fillna(0).to_numpy()
                axs[j].errorbar(np.arange(4),y,yerr=err,marker='o',ms=4,lw=1.5,c=colors[name],capsize=3,label=names[name])
        for ax in axs:
            ax.set_xticks(range(4),labels);ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',length=0)
        axs[0].set_ylabel('Pinball loss (m)' if lang=='en' else '分位数损失（米）');axs[1].set_ylabel('Increase over complete input (m)' if lang=='en' else '相对完整输入的损失增量（米）')
        axs[0].set_title('(a) Identical training masks' if lang=='en' else '（a）共同训练掩码');axs[1].set_title('(b) Response to interruption' if lang=='en' else '（b）输入中断响应');handles,labels_legend=axs[0].get_legend_handles_labels();fig.legend(handles,labels_legend,ncol=3,loc='upper center',bbox_to_anchor=(.52,1.13),frameon=False,fontsize=9)
        save(fig,'fair_missingness_'+lang)
    ci=pd.read_csv(R/'combinations/paired_intervals.csv')
    contrasts=[('TabM_LightGBM_ChronosAdapter','TabM_LightGBM'),('TabM_LightGBM_BothAdapters','TabM_LightGBM'),('All','All_without_adapters')]
    labels=['Add Chronos adapter\nto TabM + LightGBM','Add both adapters\nto TabM + LightGBM','Restore both adapters\nto the five other experts'] if lang=='en' else ['TabM＋LightGBM\n加入Chronos适配器','TabM＋LightGBM\n加入两个适配专家','其余五专家\n恢复两个适配专家']
    fig,axs=plt.subplots(1,3,figsize=(8.2,3.45),sharey=True,gridspec_kw={'wspace':.18})
    lows=[];highs=[]
    for ax,period,title in zip(axs,['test','h1','q3'],['2022–2025','2026 H1','2026 Q3']):
        for row,(model,ref) in enumerate(contrasts):
            for days,offset,color in [(7,-.07,'#4477AA'),(28,.07,'#228833')]:
                v=ci[(ci.period==period)&(ci.model==model)&(ci.reference==ref)&(ci.block_days==days)].iloc[0]
                x=v.difference*1e4;lo=v.ci_low*1e4;hi=v.ci_high*1e4
                ax.errorbar(x,row+offset,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',ms=3.5,lw=1.2,capsize=2.5,color=color,label=('7 day blocks' if days==7 else '28 day blocks') if lang=='en' else ('7天块' if days==7 else '28天块'))
                lows.append(lo);highs.append(hi)
        ax.axvline(0,c='#666666',ls='--',lw=.8);ax.grid(axis='x',alpha=.15);ax.set_title(title);ax.set_xlabel('Loss difference ($10^{-4}$ m)' if lang=='en' else '损失差值（$10^{-4}$ 米）');ax.set_yticks(range(3),labels)
    for ax in axs:ax.set_xlim(min(lows)-.3,max(highs)+.3)
    axs[0].invert_yaxis();handles,leglabels=axs[0].get_legend_handles_labels();fig.legend(handles[:2],leglabels[:2],frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.57,1.1),ncol=2)
    save(fig,'component_attribution_'+lang)
print('FIGURES RECREATED',flush=True)
