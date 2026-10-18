from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
V=Path(__file__).resolve().parent;D=V.parent;O=V/'figures';O.mkdir(exist_ok=True)
df=pd.read_csv(V/'ablation_by_seed.csv');sc=pd.read_csv(V/'diagnostics/overall.csv')
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'era5-review-followup-20261018','pdf.fonttype':42})
def save(fig,name):
    for ext in ['png','pdf','svg']:
        meta={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {}
        fig.savefig(O/f'{name}.{ext}',dpi=250,bbox_inches='tight',metadata=meta)
    plt.close(fig)
for lang in ['en','zh']:
    en=lang=='en';fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    labs=['Full','Shared scale','No reference query','No availability'] if en else ['完整','共享尺度','移除参考查询','移除可用性']
    axes[0].barh(range(4),[.12634,.12734,.13282,.12600],color=['#216e79','#a5b6bd','#a5b6bd','#a5b6bd'],height=.55)
    axes[0].set_yticks(range(4),labs);axes[0].invert_yaxis();axes[0].set_xlim(.12,.135);axes[0].set_title('Chronos reference, seed 17' if en else 'Chronos 参考，种子17');axes[0].set_xlabel('Pinball loss (m)' if en else '分位数损失（米）')
    for j,name in enumerate(['full','shared_scale','no_query','no_availability']):
        z=df[df.variant==name].pinball.to_numpy();axes[1].errorbar(z.mean(),j,xerr=z.std(ddof=1),fmt='s',color='#216e79',capsize=4,ms=7)
        axes[1].scatter(z,[j-.14,j,j+.14],s=19,color='#be7b3e',zorder=3)
    axes[1].set_yticks(range(4),labs);axes[1].invert_yaxis();axes[1].set_xlabel('Pinball loss (m), mean and sample SD' if en else '分位数损失（米），均值与样本标准差');axes[1].set_title('Empirical reference, seeds 17, 29, 43' if en else '经验参考，种子17、29、43')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    save(fig,f'ablation_{lang}')
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained');xx=np.arange(4)
    labels=['Complete','Random 30%','Last 24 h','All absent'] if en else ['完整输入','随机30%','末端24小时','全部气象缺失']
    names=['Empirical_adapter','Empirical_no_availability','Chronos_adapter','Chronos_no_availability','LightGBM_native_nan','LightGBM_train_mean']
    legends=['Empirical adapter','Empirical without availability','Chronos adapter','Chronos without availability','LightGBM native NaN','LightGBM training mean'] if en else ['经验适配器','经验参考移除可用性','Chronos适配器','Chronos移除可用性','LightGBM原生缺失值','LightGBM训练均值填补']
    for name,label,col,marker in zip(names,legends,['#1b7780','#70b0ac','#8d65a3','#ba9ec7','#c07936','#ddac78'],['o','o','s','s','^','^']):
        z=sc[sc.model==name].set_index('scenario').loc[['complete','random30','last24','all_missing'],'pinball'].to_numpy()
        axes[0].plot(xx,z,marker=marker,label=label,color=col,lw=1.5,ms=4)
        axes[1].plot(xx,z-z[0],marker=marker,color=col,lw=1.5,ms=4)
    for ax in axes:ax.set_xticks(xx,labels,rotation=15);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Pinball loss (m)' if en else '分位数损失（米）');axes[1].set_ylabel('Increase from complete input (m)' if en else '相对完整输入的损失增加（米）')
    handles,ls=axes[0].get_legend_handles_labels();fig.legend(handles,ls,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False,fontsize=8)
    save(fig,f'interruption_{lang}')
print(O)
