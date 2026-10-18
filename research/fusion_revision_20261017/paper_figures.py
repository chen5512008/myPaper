import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
P=D/'paper/fusion_revision_20261017';P.mkdir(exist_ok=True);d=pd.read_csv(R/'results/site_horizon_by_seed.csv');z=d[d.period=='q3'].groupby(['model','site','horizon']).pinball.mean()
a=np.array([[100*(z['LightGBM',s,h]-z['Selected_fusion',s,h])/z['LightGBM',s,h] for h in H] for s in SITES]);w=np.array(json.loads((R/'selection_diagnostics.json').read_text())['mean2021_weights_by_horizon'])
for lang in ['en','zh']:
    en=lang=='en';plt.rcParams.update({'font.family':'DejaVu Sans' if en else 'Microsoft YaHei','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'era5-paper-fusion','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.5,3.2),layout='constrained');im=ax.imshow(a,cmap='Blues',vmin=0,vmax=15,aspect='auto');ax.set_xticks(range(4),[str(h) for h in H]);ax.set_yticks(range(6),SITES);ax.set_xlabel('Forecast lead (h)' if en else '预测提前量（小时）')
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{a[i,j]:+.1f}',ha='center',va='center',fontsize=10,color='white' if a[i,j]>8 else '#172c3c')
    fig.colorbar(im,ax=ax,label='Pinball reduction (%)' if en else '分位数损失降幅（%）',shrink=.92)
    fig.savefig(P/f'paper_q3_gain_{lang}.png',dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,2.6),layout='constrained');im=ax.imshow(w,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(7),['LightGBM','Neural3','Chronos\nadapter','Chronos\nnative','XGBoost','CatBoost','TabM'],fontsize=8);ax.set_yticks(range(4),[f'{h} h' for h in H])
    for i in range(4):
        for j in range(7):ax.text(j,i,'<.01' if w[i,j]<.01 else f'{w[i,j]:.2f}',ha='center',va='center',color='white' if w[i,j]>.55 else '#172c3c',fontsize=8.5)
    fig.colorbar(im,ax=ax,label='Weight' if en else '权重',shrink=.8);fig.savefig(P/f'paper_gate_weights_{lang}.png',dpi=300);plt.close(fig)
print('PAPER FIGURES',P)
