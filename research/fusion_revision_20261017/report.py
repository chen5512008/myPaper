"""Evidence-based Chinese outcome, no automatic claim of novelty or acceptance."""
import pandas as pd
from common import *
out=R/'results';df=pd.read_csv(out/'scores_mean_std.csv');ci=pd.read_csv(out/'fusion_repeated_block_intervals.csv');selection=json.loads((R/'selection.json').read_text());decision={}
lines=['融合算法实验结果与论文更新判断','作者：陈玉杰 Yujie Chen','',
    '一 结论依据','LightGBM 保留为主要强基线，采用已有161维特征和完整训练结果，没有替换为较弱模型。另列校准LightGBM、简单残差分位数持久性模型和所有融合组成模型。',
    '主指标为全部六点、6/12/24/48小时、0.1至0.9共九个分位数的平均pinball loss，越低越好。下表对三个新专家训练种子的独立指标取平均，不是先平均预测再计算误差。',
    '旧2022至2025年测试集和2026年上半年在之前研究中已被查看，本轮作为探索性对比。2026年第三季度在本轮融合选择结束后才进行预测与评分。',
    '','二 主结果（单位m）','时段 | 强LightGBM | 校准LightGBM | TabM | 选定融合 | 融合相对强LightGBM降幅 | 融合相对TabM降幅']
for period,title in [('test','2022至2025年'),('h1','2026年上半年'),('q3','2026年第三季度')]:
    z=df[df.period==period].set_index('model');base=z.loc['LightGBM','pinball_mean'];fusion=z.loc['Selected_fusion','pinball_mean'];tab=z.loc['TabM','pinball_mean'];cal=z.loc['LightGBM_calibrated','pinball_mean'];gain=100*(base-fusion)/base;gt=100*(tab-fusion)/tab
    lines.append(f'{title} | {base:.8f} | {cal:.8f} | {tab:.8f} | {fusion:.8f} | {gain:+.3f}% | {gt:+.3f}%')
    decisions=ci[(ci.period==period)&(ci.reference=='LightGBM')]
    decision[period]={'LightGBM':float(base),'TabM_mean':float(tab),'fusion_mean':float(fusion),'gain_vs_LightGBM_pct':float(gain),'gain_vs_TabM_pct':float(gt),'both_block_intervals_below_zero':bool((decisions.ci_high<0).all())}
lines+=['','三 时间块置信区间','下列差值为融合损失减参考模型损失；区间完全小于0才支持融合损失更低。所有点位同时按时间块重采样，避免把空间和时间相关记录误当作独立样本。']
for _,r in ci[ci.reference.isin(['LightGBM','LightGBM_calibrated','TabM'])].iterrows():lines.append(f"{r.period} 对 {r.reference}，{int(r.block_days)}天块：差值 {r.difference:+.8f}，95%区间 [{r.ci_low:+.8f}, {r.ci_high:+.8f}]")
lines+=['','四 选定算法',f"2021年验证集选定：{selection['selected']['method']}；验证损失 {selection['selected']['selection_loss']:.8f}。",'专家：'+', '.join(selection['members']),
    'TabM采用ICLR2025公开架构、16个共享参数成员、160个数值特征与点位类别编码；训练分别计算成员分位数损失，预测时平均成员输出。四个预测时长同时输出九个分位数。',
    '融合权重拟合于2020年，融合结构和正则强度选择于2021年。17、29、43三个种子只改变新增专家训练；融合权重和融合网络保持固定。已有Neural3明确为三个网络预测的平均集成。',
    '若选中ordered_gate，权重依赖历史状态、模型预测和预测时长，同时有界调整中位数与八个相邻分位数间距；若未选中，不把该候选写作获胜方法。',
    'CatBoost原生CPU MultiQuantile在本机出现bad allocation，两个并行任务因内存压力被停止。正式完成的替代版本为GPU RMSE残差模型加2020年残差分位数校准，并以CatBoost_RMSEcal单独命名；没有把它描述为原生MultiQuantile结果。',
    '','五 论文可写到什么程度']
beats_base=all(v['gain_vs_LightGBM_pct']>0 for v in decision.values());confirmed=decision['q3']['both_block_intervals_below_zero'];beats_tab=all(v['gain_vs_TabM_pct']>0 for v in decision.values())
decision.update({'beats_LightGBM_point_estimate_all_periods':beats_base,'Q3_confirms_vs_LightGBM_under_both_block_sizes':confirmed,'beats_TabM_point_estimate_all_periods':beats_tab})
if beats_base and confirmed:lines.append('本轮结果支持更新论文中的强基线对照：融合在三个时段的点估计都优于保留的LightGBM，且本轮预留季度的两个时间块区间均支持该差异。')
elif beats_base:lines.append('融合在三个时段的点估计都优于LightGBM，但预留季度的置信区间仍包含0；可以报告数值提升，不宜写成已证实的稳定优势。')
else:lines.append('融合尚未在三个时段都优于LightGBM，应完整报告表现差异，不能将不利时段删除后写成普遍提升。')
if not beats_tab:lines.append('融合没有在全部时段优于现代TabM组成模型。论文必须区分“新专家增强了预测能力”和“融合本身有额外贡献”，不能把TabM带来的收益全部归因于融合创新。')
else:lines.append('融合在三个时段的点估计均优于TabM；额外贡献仍须结合对TabM的时间块区间和去除专家消融判断。')
lines.append('相对TabM，2026H1和Q3的7天块置信区间包含0，28天块区间小于0；额外收益对区间设定敏感，不能写成相对TabM已经取得一致显著优势。')
lines.append('融合相对LightGBM的高波浪MAE在三个时段均更低，但相对TabM的高波浪MAE在三个时段均略高；融合80%区间也略宽于TabM。这些权衡需要在正文和讨论中保留。')
lines+=['达到较低预测误差不等于算法创新性已充分，也不等于保证SCI四区或CCF-C录用。六个ERA5网格点仍缺独立浮标外部验证；2026年数据为下载快照，源元数据未确认最终ERA5发布状态。',
    '','六 可复现文件','scores_by_seed.csv：全部模型、种子、时段的原始评分。','scores_mean_std.csv：三种子均值与样本标准差。','fusion_repeated_block_intervals.csv：同步时间块区间。','site_horizon_by_seed.csv：海域与预测时长分解。','ablation_scores.csv：在查看测试结果前拟合的逐专家删除消融。','figures/：由CSV和权重直接生成的PNG、PDF、SVG。','selection.json、frozen_before_evaluation.json：选择记录和查看测试数据前的模型哈希。','resource_amendment.json：CatBoost失败与替代执行记录。',str(R)]
(out/'实验结果与论文更新判断.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(out/'decision.json',{'timestamp_utc':now(),**decision})
hashes={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in R.glob('*.py')};dump(out/'code_sha256.json',hashes)
print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)
