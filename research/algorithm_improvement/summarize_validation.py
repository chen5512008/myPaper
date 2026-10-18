"""Validation audit and readable report; never reads test predictions."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent;O=R/'results';O.mkdir(exist_ok=True)
rows=[];Q=np.arange(1,10)/10
for f in sorted((R/'runs').glob('*_17/run.json')):
    j=json.loads(f.read_text());d=np.load(f.parent/'validation.npz');p=d['prediction'];y=d['target'];e=y[...,None]-p
    loss=float(np.maximum(Q*e,(Q-1)*e).mean())
    assert abs(loss-j['best_validation'])<1e-6
    assert np.isfinite(p).all() and (p>=0).all() and (np.diff(p,axis=-1)>=-1e-6).all()
    rows.append({'model':j['variant'],'validation_pinball':loss,'stored_parameters':j['parameters'],'seed':17,'epochs':len(j['trace']),'selected_weights':j.get('selected_weights','raw'),'seconds':j['seconds'],'site_identity':'_region_' in j['variant'],'checkpoint_sha256':hashlib.sha256((f.parent/'model.pt').read_bytes()).hexdigest()})
baseline=json.loads((R/'strong_lightgbm/selection.json').read_text())
for name,key,identity in [('LightGBM original','original_validation',True),('LightGBM extended','extended_validation',True)]:
    rows.append({'model':name,'validation_pinball':baseline[key],'stored_parameters':None,'seed':17,'epochs':None,'selected_weights':'early-stopped trees','seconds':None,'site_identity':identity,'checkpoint_sha256':''})
table=pd.DataFrame(rows).sort_values('validation_pinball');table.to_csv(O/'validation_ranking.csv',index=False)
selected=json.loads((R/'selection.json').read_text());name=selected['variant'];val=float(table.loc[table.model==name,'validation_pinball'].iloc[0]);best_base=min(baseline['original_validation'],baseline['extended_validation'])
passed=val<best_base
report=['算法改进实验记录',f'生成时间：{datetime.datetime.now().isoformat(timespec="seconds")}',
        '本记录为实验结果，不是改写后的论文。原论文正文尚未因验证集成绩改写。','',
        f'按验证集选中的结构：{name}',f'选中结构验证损失：{val:.8f}',f'加强版 LightGBM 验证损失：{baseline["extended_validation"]:.8f}',
        f'原 LightGBM 验证损失：{baseline["original_validation"]:.8f}',f'相对最强简单基线的验证改进：{100*(1-val/best_base):.3f}%（正数为改进）','',
        '阶段和公平性：',
        '第一阶段七组预设结构；第二阶段五组平滑训练及门控结构；额外三组位置条件化对照。每次扩展均在验证集阶段记录，原协议和各阶段选择记录完整保留。',
        '训练仍用 2000—2019 年，选型用 2020—2021 年；没有因为测试结果删掉不利配置。',
        'LightGBM 原本含点位编号，位置条件化对照给神经模型加入同一信息。位置版本仅支持这些已训练点位的时间预测，不由此宣称空间零样本迁移。',
        '加强版 LightGBM 增加相同历史变化量，并将树数上限由 300 增至 600；继续按验证集早停，使用相同训练标签。',
        '第二阶段同时改变优化器、学习率调度、批量和权重平滑，属于训练方案整体对照；不能将其变化归因于其中某一项。',
        '结构筛选各用种子 17；只对最终候选进行额外种子复评，完整报告不以最佳测试种子代替平均表现。','',
        '本轮改变：',
        '1. 分位数间隔使用可补偿的 softplus 参数化，保留非负、非交叉输出。',
        '2. 加入滞后值、窗口均值/标准差和变化量组成的历史统计分支。',
        '3. 给统计分支增加分布与趋势条件门控，检验直接相加的局限。',
        '4. 位置条件化只增加 6×32=192 个参数，点位信息来自已知元数据。','',
        '完整验证集排序：']
for _,r in table.iterrows():report.append(f'{r["model"]}: {r["validation_pinball"]:.8f}')
report+=['','当前判断：']
if passed:report+=['候选通过验证集筛选门槛。验证集领先不是测试结论；仍须完成重复种子、原测试期和预留 2026 年上半年评价，才能决定更新论文。']
else:report+=['本轮候选尚未超过加强后的简单基线，不支持将论文改写为“所提方法优于简单基线”。预留的 2026 年数据应继续保留，不为追逐更好成绩反复查看。','这份记录保留所有已完成的配置、权重、训练轨迹和验证预测。需要继续研究方法或增加有物理预测价值的输入，不能靠筛选点位、删除基线、挑选测试种子或改换主指标解决。']
report+=['','复现入口：REPRODUCE.txt、run.py、run_stage2.py、run_region.py。','若未产生新的测试与确认预测文件，不得将验证损失作为测试损失引用。']
(O/'实验结果与论文更新判断.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(12,8),layout='constrained');ordered=table.sort_values('validation_pinball',ascending=False).reset_index(drop=True)
for i,r in ordered.iterrows():
    color='#b34b43' if r['model'].startswith('LightGBM') else ('#147d77' if r['model']==name else '#8092a8')
    ax.scatter(r['validation_pinball'],i,color=color,s=48,zorder=3);ax.annotate(f'{r["validation_pinball"]:.5f}',(r['validation_pinball'],i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
ax.set(yticks=range(len(ordered)),yticklabels=ordered.model,xlabel='2020—2021 验证集平均分位数损失（米，越小越好）',title='完整候选筛选结果；验证成绩不能代替测试成绩')
ax.axvline(best_base,color='#b34b43',ls='--',lw=1,label='验证集选中的 LightGBM');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right');ax.set_xlim(float(table.validation_pinball.min())-.0004,float(table.validation_pinball.max())+.0015)
for ext in ['png','svg','pdf']:fig.savefig(O/f'validation_comparison.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
manifest={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in R.glob('*.py')}
(O/'code_sha256.json').write_text(json.dumps(manifest,indent=2));print(table[['model','validation_pinball']].to_string(index=False));print('VALIDATION_GATE',passed)
