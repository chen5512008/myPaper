from pathlib import Path
import json,shutil,subprocess
import pandas as pd
R=Path(__file__).parent;D=R.parent;OUT=D/'paper/algorithm_revision';OUT.mkdir(exist_ok=True)
RES=R/'results'
overall=pd.read_csv(RES/'overall.csv').set_index('model')
high=pd.read_csv(RES/'high_wave.csv').set_index('model')
stress=pd.read_csv(RES/'stress.csv');transfer=pd.read_csv(RES/'transfer.csv')
runs={p.parent.name:json.loads(p.read_text()) for p in (R/'runs').glob('*/run.json')}
required=['full_17','full_29','full_43','affine_17','no_availability_17','no_distribution_query_17','empirical_reference_17','legacy_matched_17']
if not all(k in overall.index for k in required) or transfer.site.nunique()!=6:raise RuntimeError('Report requires all prespecified pooled and six spatial runs.')
v=overall.loc['full_17'];old=overall.loc['Old adapter'];native=overall.loc['Chronos-2 native']
improve=lambda a,b:100*(1-a/b)
folds=transfer.groupby('site')[['native','old','new']].mean()
labels_en={'Chronos-2 native':'Chronos-2 covariates','TimesFM 3.0':'TimesFM 3.0 covariates','LightGBM':'LightGBM','Old adapter':'Previous adapter','Old masked adapter':'Previous masked adapter','full_17':'Distribution query adapter (17)','full_29':'Distribution query adapter (29)','full_43':'Distribution query adapter (43)','affine_17':'Shared scale ablation','no_availability_17':'Without explicit availability','no_distribution_query_17':'Without distribution query','empirical_reference_17':'Empirical reference instead of Chronos-2','legacy_matched_17':'Previous architecture with matched training'}
labels_zh={'Chronos-2 native':'Chronos-2 历史协变量','TimesFM 3.0':'TimesFM 3.0 历史协变量','LightGBM':'LightGBM','Old adapter':'旧版适配器','Old masked adapter':'旧版缺测训练','full_17':'分布查询适配器（17）','full_29':'分布查询适配器（29）','full_43':'分布查询适配器（43）','affine_17':'共享尺度消融','no_availability_17':'移除显式可用性','no_distribution_query_17':'移除分布查询','empirical_reference_17':'经验参考替代 Chronos-2','legacy_matched_17':'旧结构采用相同训练协议'}
for seed in [17,29,43]:
    labels_en[f'empirical_reference_{seed}']=f'Empirical reference ({seed})'
    labels_zh[f'empirical_reference_{seed}']=f'经验参考版本（{seed}）'
def tab(headers,rows):return '| '+' | '.join(headers)+' |\n|'+' --- |'*len(headers)+'\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)+'\n'

for lang in ['zh','en']:
    zh=lang=='zh';labels=labels_zh if zh else labels_en
    s=('---\ntitle: 波高概率预测的分布查询 Transformer 适配与实验验证\n---\n\n陈玉杰（Yujie Chen）\n\n' if zh else '---\ntitle: Distribution Query Transformer Adaptation for Probabilistic Wave Forecasting\n---\n\nYujie Chen\n\n')
    s+=('本报告记录六点位论文在原有冻结 Chronos-2 适配框架上的算法改进、实际对照实验和可支持的贡献，供后续修改主文使用。\n\n' if zh else 'This report documents an algorithm revision of the six site study, its executed comparisons and the contribution supported by those results. It is intended to guide the next manuscript revision.\n\n')
    s+=('## 1 问题与改进思路\n\n' if zh else '## 1 Motivation and design\n\n')
    s+=(r'''原稿通过历史气象修正冻结基础模型的波高分位数，已有实验表明这一方向有效。进一步改进需要回答更具体的问题：基础模型本次预测的分布形状，是否应该影响历史信息的选择？气象记录缺失时，又如何让修正过程明确知道哪些输入仍可使用？

旧版只对全部分位数进行同一次平移和缩放。因此，同一提前量内各相邻分位数间隔的比例保持不变，无法分别修正上下尾部。旧版查询向量只包含当前状态与提前量，不读取基础预测本身；气象缺测则填为标准化零值，与正常的均值水平混在一起。这些是现有实现的具体限制。

Transformer 通过注意力让输入片段交换信息。在本任务中，过去 168 小时被切成 28 个六小时分块，模型可以根据本次预测的分布与提前量，为不同历史片段分配权重。大型预训练部分仍是 Chronos-2；新增 Transformer 是 36,114 参数的适配器，承担有监督修正。

本文将基础模型的九个分位数加入查询，并以海浪与气象两个分支读取历史。输出端分别调整八个相邻分位数间隔，保留预测顺序。气象掩码不仅用于训练增强，还直接进入输入表示与注意力权重；全部气象缺失时，气象修正项严格归零。

实验分别检验分布查询、共享尺度限制和显式可用性的作用，并以经验残差分布替代 Chronos-2 检验预训练预测的价值。主要评价仍是分位数损失，辅以区间覆盖率、宽度、高波浪误差及空间迁移。这样可以分清哪些改进得到支持，以及更灵活的修正会付出什么代价。

''' if zh else r'''The previous manuscript uses historical weather to correct frozen wave height quantiles. Its results motivate a more specific question: should the shape of the foundation forecast influence which history is retrieved, and how should that correction respond when weather observations are absent?

The previous adapter shifts and scales all quantiles together. Ratios between adjacent quantile gaps therefore remain fixed within each horizon, preventing separate adjustment of the lower and upper tails. Its query uses the current state and forecast lead but not the foundation forecast itself. Missing weather is replaced by standardized zero, which is also a valid mean level observation.

A Transformer allows historical patches to exchange information through attention. We divide 168 hours into 28 patches of six hours. A query formed from the predicted distribution and lead then retrieves information from these patches. Chronos-2 remains the large pretrained component; the added Transformer is a supervised adapter with 36,114 parameters.

The revision supplies all nine base quantiles to the query and separates wave and weather history into two branches. Eight adjacent quantile gaps are adjusted independently while preserving order. Weather masks enter the embeddings and attention weights directly. When all weather observations are absent, the weather correction is exactly zero.

Experiments isolate the distribution query, the shared scale restriction and explicit availability. Replacing Chronos-2 with a reference derived from training residuals tests the value of pretrained predictions. Pinball loss remains the primary metric, with interval coverage, width, high wave errors and spatial transfer used to characterize the trade-offs.

''')
    s+=('![图 1 分布查询与气象可用性共同控制有序分位数修正。](architecture_zh.png){width=16cm}\n\n' if zh else '![Figure 1 Distribution queries and weather availability control ordered quantile correction.](architecture_en.png){width=16cm}\n\n')
    s+=('## 2 方法与结构性质\n\n' if zh else '## 2 Method and structural properties\n\n')
    s+=(r'''沿用原稿八维历史输入：波高、平均周期、波向正弦与余弦，以及 10 米风的两个分量、阵风和海平面气压。前四维为海浪分支，后四维为气象分支。所有标准化量仅由源位置训练期估计，起报时刻后的气象值不进入模型。

令 $M_{t,j,c}$ 表示第 j 个气象分块内各小时及通道是否观测。分块可用率为 $r_{t,j}=\operatorname{mean}(M_{t,j,:})$，整体可用率为 $\bar r_t=\operatorname{mean}_j r_{t,j}$。气象值先与掩码相乘，再与掩码一起嵌入，因此模型能区分缺测和观测到的零值。每个分支使用一个 Transformer 自注意力块，隐藏维度 32、四个头、两层前馈网络；位置嵌入保留分块顺序。

基础分位数 $q^0_{t,h,1:9}$、标准化后的最新海浪状态 $x^w_t$ 与提前量组成查询；$s_y$ 为源训练期波高标准差：

''' if zh else r'''The eight historical features are wave height, mean period, the sine and cosine of wave direction, two 10 m wind components, gust and mean sea level pressure. The first four form the wave branch and the remaining four form the weather branch. Normalization uses source training observations only. No weather value after the forecast origin enters the model.

Let $M_{t,j,c}$ mark observed weather entries in patch j. Patch availability is $r_{t,j}=\operatorname{mean}(M_{t,j,:})$, and overall availability is $\bar r_t=\operatorname{mean}_j r_{t,j}$. Weather values are multiplied by the mask and embedded together with it, distinguishing missing entries from observed zero values. Each branch has one Transformer self attention block with width 32, four heads and a two layer feedforward network. Learned positional embeddings retain patch order.

The query combines base quantiles $q^0_{t,h,1:9}$, the standardized latest wave state $x^w_t$ and the lead; $s_y$ is the source training wave height standard deviation:

''')
    s+=r'$$u_{t,h}=f_q([q^0_{t,h,1:9}/s_y;\,x^w_t;\,h/48]). \qquad (1)$$'+'\n\n'
    s+=('海浪交叉注意力先生成摘要 $z^w_{t,h}$，气象查询使用 $u_{t,h}+z^w_{t,h}$。气象自注意力和交叉注意力均在键位置加入可用率的对数：\n\n' if zh else 'Wave cross attention first produces $z^w_{t,h}$; weather retrieval uses $u_{t,h}+z^w_{t,h}$. Weather self attention and cross attention add log availability at key positions:\n\n')
    s+=r'$$A^m_{t,h,j}=\operatorname{softmax}_j\left(\frac{Q_{t,h}^{\top}K_{t,j}}{\sqrt {d_k}}+\log\{\max(r_{t,j},10^{-8})\}\right),\quad d_k=8. \qquad (2)$$'+'\n\n'
    s+=('两个输出头生成九维修正系数，合成为\n\n' if zh else 'Two heads produce nine correction coefficients, combined as\n\n')
    s+=r'$$\theta_{t,h}=\theta^w_{t,h}+\bar r_t\,\theta^m_{t,h}. \qquad (3)$$'+'\n\n'
    s+=('第一维调整中位数，其余八维调整相邻分位数间隔：\n\n' if zh else 'The first coefficient shifts the median, and the remaining eight scale adjacent quantile gaps:\n\n')
    s+=r'$$\widehat m_{t,h}=q^0_{t,h,5}+s_y\theta_{t,h,0},\quad \widehat d_{t,h,j}=(q^0_{t,h,j+1}-q^0_{t,h,j})\exp\{2\tanh(\theta_{t,h,j})\}. \qquad (4)$$'+'\n\n'
    s+=r'$$\widetilde q_{t,h,k}=\begin{cases}\widehat m_{t,h}-\sum_{j=k}^{4}\widehat d_{t,h,j},&k<5,\\\widehat m_{t,h},&k=5,\\\widehat m_{t,h}+\sum_{j=5}^{k-1}\widehat d_{t,h,j},&k>5,\end{cases}\quad\widehat q_{t,h,k}=\max\{0,\widetilde q_{t,h,k}\}. \qquad (5)$$'+'\n\n'
    s+=(r'''原始分位数先排序。由于原始间隔非负、指数因子为正，累积重建保持顺序；最后的非负截断也是单调操作。输出头末层初始化为零，此时模型等于原始参考预测的非负投影。完全没有气象观测时，式 (3) 的气象项为零，海浪分支及其查询又不依赖气象，从而得到明确的海浪分支回退。这些性质由结构保证，实际误差仍需实验评价。

训练仍使用九个分位数上的平均 pinball loss。它以非对称方式惩罚高估与低估，与预测分位数对应。训练不对事后高波浪标签额外加权。

''' if zh else r'''Base quantiles are sorted first. Nonnegative base gaps and positive multipliers preserve order under cumulative reconstruction; clipping at zero is also monotone. The final head layers are initialized to zero, giving the nonnegative reference forecast at initialization. With no observed weather, the weather term in (3) vanishes. The wave branch and its query are independent of weather, giving an explicit wave branch fallback. These are structural properties; forecast error remains an empirical question.

Training minimizes mean pinball loss over nine quantiles. Its asymmetric penalties correspond to the quantile being estimated. No additional weight is assigned using future high wave labels.

''')
    s+=('## 3 实验协议与对照\n\n' if zh else '## 3 Protocol and controls\n\n')
    s+=(r'''六点及时间划分沿用原稿：渤海、黄海、东海、台湾海峡、南海北部和海南岛东侧；2000 至 2019 年训练，2020 至 2021 年验证，2022 至 2025 年评价。历史长度 168 小时，起报间隔 6 小时，预测提前量为 6、12、24、48 小时。完整模型固定主种子 17，并报告 29、43；所有消融及空间折使用 17。

训练采用 Adam，学习率 0.001、批量 512、最多 30 轮，完整源验证集损失连续五轮没有改善即停止，梯度范数裁剪为 1。每批以 50% 概率保留完整输入，25% 概率随机移除 30% 气象条目，25% 概率移除末端 3、6、12、24 或 168 小时气象。所有版本采用相同协议，测试随机缺测掩码固定种子 777。旧结构重训参照也采用这一协议，以区分结构与训练设置变化。

''' if zh else r'''The sites and partitions follow the existing study: Bohai Sea, Yellow Sea, East China Sea, Taiwan Strait, northern South China Sea and east of Hainan; training uses 2000 to 2019, validation 2020 to 2021 and evaluation 2022 to 2025. Context is 168 hours, origins are six hours apart and leads are 6, 12, 24 and 48 hours. Seed 17 is the prespecified primary run, with 29 and 43 also reported. Ablations and spatial folds use 17.

Adam uses learning rate 0.001, batch size 512, at most 30 epochs, patience five on complete source validation loss and gradient norm clipping at one. Half of training batches retain complete input; one quarter randomly removes 30% of weather entries, and one quarter removes a terminal interval of 3, 6, 12, 24 or 168 hours. All revised variants share this protocol. Evaluation masks use seed 777. The previous architecture is also retrained under the same protocol to separate structural changes from training changes.

''')
    controls=[['affine','八个间隔系数取均值，其余网络与参数量不变','Average eight gap coefficients; identical network and parameter count'],['no_availability','隐藏掩码、取消可用率加权和气象门控','Hide masks and disable availability weighting and gating'],['no_distribution_query','查询中的九个基础分位数置零，输出仍修正基础预测','Zero base quantiles in the query; retain them as the output reference'],['empirical_reference','最后观测值加源训练残差分位数，替代 Chronos-2','Replace Chronos-2 with the last observation plus source training residual quantiles'],['legacy_matched','旧结构使用相同掩码、裁剪与训练协议','Train the previous architecture with matching masking and optimization']]
    s+=tab(['版本','操作'] if zh else ['Variant','Intervention'],[[a,b if zh else c] for a,b,c in controls])
    s+=('## 4 已完成结果\n\n' if zh else '## 4 Completed experiments\n\n')
    s+=('以下指标先在各位置与提前量内计算，再对六个位置和四个提前量等权平均。MAE 使用预测中位数；覆盖率和宽度对应第 0.1 至第 0.9 分位数形成的名义 80% 区间。高波浪指目标波高不低于该位置训练期的第 95 百分位阈值，阈值不由测试期估计。\n\n' if zh else 'Metrics are computed within each site and lead, then averaged equally over six sites and four leads. MAE uses the predicted median. Coverage and width refer to the nominal 80% interval between the 0.1 and 0.9 quantiles. High waves are targets at or above the site specific 95th percentile estimated from the training period; test targets do not determine the threshold.\n\n')
    names=['Chronos-2 native','TimesFM 3.0','Old adapter','Old masked adapter','legacy_matched_17','full_17','full_29','full_43','LightGBM']
    s+=tab(['方法','损失 (m)','MAE (m)','覆盖率','宽度 (m)'] if zh else ['Method','Pinball (m)','MAE (m)','Coverage','Width (m)'],[[labels[n],f'{overall.loc[n,"pinball"]:.5f}',f'{overall.loc[n,"mae"]:.4f}',f'{overall.loc[n,"coverage"]:.2%}',f'{overall.loc[n,"width"]:.4f}'] for n in names])
    s+=(f'主种子新版损失为 {v.pinball:.5f} 米，相对旧版降低 {improve(v.pinball,old.pinball):.2f}%，相对原生 Chronos-2 降低 {improve(v.pinball,native.pinball):.2f}%。LightGBM 的损失为 {overall.loc["LightGBM","pinball"]:.5f} 米。\n\n' if zh else f'The primary revised model obtains {v.pinball:.5f} m loss, a reduction of {improve(v.pinball,old.pinball):.2f}% relative to the previous adapter and {improve(v.pinball,native.pinball):.2f}% relative to native Chronos-2. LightGBM obtains {overall.loc["LightGBM","pinball"]:.5f} m.\n\n')
    s+=f'![{"图 2 总体误差与高波浪误差共同评价。" if zh else "Figure 2 Aggregate errors and high wave errors."}](performance_{lang}.png){{width=16cm}}\n\n'
    names=['full_17','affine_17','no_availability_17','no_distribution_query_17','empirical_reference_17']
    s+=tab(['消融','损失 (m)','高波浪 MAE (m)'] if zh else ['Ablation','Pinball (m)','High wave MAE (m)'],[[labels[n],f'{overall.loc[n,"pinball"]:.5f}',f'{high.loc[n,"mae"]:.4f}'] for n in names])
    s+=(f'共享尺度消融损失为 {overall.loc["affine_17","pinball"]:.5f} 米；移除分布查询后为 {overall.loc["no_distribution_query_17","pinball"]:.5f} 米。这两个对照与完整模型保持相同参数量。显式可用性并未在完整输入下取得最低误差：移除该信息的损失为 {overall.loc["no_availability_17","pinball"]:.5f} 米。其作用需要结合缺测结果解释。\n\n' if zh else f'The shared scale ablation obtains {overall.loc["affine_17","pinball"]:.5f} m, and removing the distribution query gives {overall.loc["no_distribution_query_17","pinball"]:.5f} m. Both retain the full model parameter count. Explicit availability does not give the lowest error with complete inputs: removing it obtains {overall.loc["no_availability_17","pinball"]:.5f} m. Its role must be read with the missingness results.\n\n')
    ci=pd.read_csv(RES/'paired_ci.csv')
    old_ci=ci[ci.comparator=='Old adapter']
    s+=('下表给出固定主种子下新版减旧版的损失差及时间块 95% 区间，负值有利于新版。分位数形状修正与共享尺度的差异，在 6、12、24 小时的区间低于零，48 小时区间包含零；完整配对结果保存在 paired_ci.csv。\n\n' if zh else 'The table gives revised minus previous loss and 95% temporal block intervals for the primary seed; negative values favour the revision. Compared with shared scaling, intervals for gap correction lie below zero at 6, 12 and 24 hours, but include zero at 48 hours. Full paired results are in paired_ci.csv.\n\n')
    s+=tab(['提前量','差值 (m)','区间下界','区间上界'] if zh else ['Lead','Difference (m)','Lower','Upper'],[[int(row.horizon),f'{row.delta:.5f}',f'{row.low:.5f}',f'{row.high:.5f}'] for row in old_ci.itertuples()])
    s+=(f'经验参考版本的损失为 {overall.loc["empirical_reference_17","pinball"]:.5f} 米，低于使用 Chronos-2 的主版本。这一对照没有支持预训练模型在当前数据上的必要性，应将大模型作用与 Transformer 适配结构的作用分别讨论。高波浪子集上，新版 MAE 为 {high.loc["full_17","mae"]:.4f} 米，旧版为 {high.loc["Old adapter","mae"]:.4f} 米。高波浪目标与总体损失呈现不同排序。\n\n' if zh else f'The empirical reference variant obtains {overall.loc["empirical_reference_17","pinball"]:.5f} m, below the primary model using Chronos-2. This control does not support a need for the pretrained model on the present data, so its role must be distinguished from that of the Transformer adapter. On the high wave subset, revised MAE is {high.loc["full_17","mae"]:.4f} m versus {high.loc["Old adapter","mae"]:.4f} m previously. High wave and aggregate loss rankings differ.\n\n')
    extra=[f'empirical_reference_{seed}' for seed in [17,29,43] if f'empirical_reference_{seed}' in overall.index]
    s+=('经验参考在验证集表现更好后，额外复核两个种子；这一扩展发生在已查看首轮测试结果之后，属于探索性验证。结构、超参数与主种子 17 均未更改。\n\n' if zh else 'Two further empirical reference seeds were added after its stronger validation result. This extension followed inspection of the first test results and is exploratory. Architecture, hyperparameters and primary seed 17 were unchanged.\n\n')
    s+=tab(['经验参考复核','损失 (m)','高波浪 MAE (m)'] if zh else ['Empirical replication','Pinball (m)','High wave MAE (m)'],[[labels[n],f'{overall.loc[n,"pinball"]:.5f}',f'{high.loc[n,"mae"]:.4f}'] for n in extra])
    st=stress.pivot(index='model',columns='scenario',values='pinball')
    s+=tab(['方法','完整','随机30%','末端24h','全缺失'] if zh else ['Method','Complete','Random 30%','Last 24 h','All absent'],[[labels[n]]+[f'{st.loc[n,c]:.5f}' for c in ['complete','random30','last24','all_missing']] for n in ['Old adapter','Old masked adapter','full_17','no_availability_17']])
    s+=f'![{"图 3 完整与气象中断情景的分位数损失。" if zh else "Figure 3 Pinball loss with complete and interrupted weather."}](missingness_{lang}.png){{width=16cm}}\n\n'
    s+=('## 5 空间留出验证\n\n' if zh else '## 5 Spatial holdout\n\n')
    s+=('每次留出一个位置，其训练标签、归一化统计和验证标签均不参与适配；只提供起报时的历史及事后评分目标。各折采用源位置验证集选轮次。\n\n' if zh else 'Each fold excludes one site from adapter training, normalization and validation. The target site supplies only history at issuance and subsequent scoring targets. Epoch selection uses source validation data.\n\n')
    s+=tab(['留出位置','原生 Chronos-2','旧版','新版'] if zh else ['Held out site','Native Chronos-2','Previous','Revised'],[[n]+[f'{folds.loc[n,k]:.5f}' for k in ['native','old','new']] for n in ['BH01','YS01','ECS01','TS01','SCS01','HN01']])
    f=folds.mean();s+=(f'六折等权损失为 {f["new"]:.5f} 米，相对旧版变化为 {improve(f["new"],f["old"]):.2f}% 改善，相对原生 Chronos-2 为 {improve(f["new"],f["native"]):.2f}% 改善。各点差异在图中保留。\n\n' if zh else f'Equal weighting over six folds gives {f["new"]:.5f} m loss, a {improve(f["new"],f["old"]):.2f}% reduction relative to the previous adapter and {improve(f["new"],f["native"]):.2f}% relative to native Chronos-2. Site differences remain visible in the figure.\n\n')
    s+=f'![{"图 4 六次空间留出的模型比较。" if zh else "Figure 4 Model comparison across six spatial holdouts."}](transfer_{lang}.png){{width=16cm}}\n\n'
    s+=('## 6 贡献解释与证据边界\n\n' if zh else '## 6 Contribution and scope\n\n')
    s+=(r'''最有依据的候选方法贡献，是让基础预测分布参与历史查询，并将整体缩放扩展为有序间隔修正。它对应原稿两个可指出的结构限制，也有同参数量消融可检验。Transformer、跨注意力和有序分位数参数化已有研究基础，不逐项作为首创。显式可用性提供可证明的完全缺测回退，但当前结果不支持它在所有输入情景都提升精度。

相关研究需要准确区分。TimeXer [1] 使用分块与跨变量注意力；CoRA [2] 在冻结模型中通过自适应归一化注入协变量；Time-PEFT [3] 设计频率和通道适配；AdaPTS [4] 研究概率多变量适配。关于基础模型校准的研究 [5] 也表明，评价需要覆盖区间可靠性。以上工作界定研究背景，本轮没有复现它们的完整官方方法，不能用内部消融替代这些直接对照。

新版的高波浪 MAE 和区间覆盖率必须与总体损失一起报告。灵活调整间隔并不自动改善高波浪预测，较低 pinball loss 也不保证名义 80% 区间达到 80% 覆盖。LightGBM 仍是重要的监督参照。新模块有 36,114 个参数，旧版为 2,083；整体提升可能包含容量、分块表示和训练变化的共同作用，不能全部归于某个单一机制。

旧版六点测试结果已用于前期开发，本轮属于该基准上的后续研究。完整联合模型和经验参考版本各使用三个种子，其余消融及空间折为单种子；配对区间使用同步七天循环块、2,000 次重采样，没有多重比较校正，也不包含模型重训不确定性。乘法间隔修正无法重新展开原始零间隔。ERA5 是再分析目标，真正上线仍需处理发布延迟并进行独立浮标和风暴过程验证。

主文修改应围绕分布查询与有序形状适配展开，明确说明缺测设计的收益和代价。现阶段可以将其作为经过首轮实验验证的算法改进，文献首创性和投稿竞争力仍需要上述直接对照与独立验证支撑。

''' if zh else r'''The best supported candidate contribution is to condition history retrieval on the foundation predictive distribution and replace shared scaling with ordered gap correction. These changes address explicit restrictions in the previous architecture and permit controls with identical parameter counts. Transformers, cross attention and monotone quantile parameterizations have established precedents. Explicit availability provides a provable fallback when weather is entirely absent, but current results do not establish an accuracy advantage in every input scenario.

Relevant methods should be distinguished carefully. TimeXer [1] uses patch and cross variate attention; CoRA [2] injects covariates into frozen models through adaptive normalization; Time-PEFT [3] develops frequency and channel adapters; AdaPTS [4] studies probabilistic multivariate adaptation. Calibration analysis of foundation models [5] further motivates reporting interval reliability. These works define the research context. Their complete official methods were not reproduced in this experiment, and internal ablations cannot replace those direct comparisons.

High wave MAE and interval coverage must accompany aggregate loss. Flexible gap correction does not automatically improve high wave prediction, and lower pinball loss does not guarantee nominal 80% coverage. LightGBM remains an important supervised reference. The new module has 36,114 parameters versus 2,083 previously. Overall gains can reflect capacity, patch representation and training changes together; they cannot all be assigned to one mechanism.

The older six site test results were already used in development, so this is a follow-up study on the same benchmark. The full pooled model and empirical reference variant each use three seeds; other ablations and spatial folds use one. Paired intervals use synchronous circular seven day blocks and 2,000 resamples, without adjustment for multiple comparisons or uncertainty from retraining. Multiplicative correction cannot reopen zero reference gaps. ERA5 supplies reanalysis targets; deployment also requires accounting for publication latency and validation against independent buoy observations and storm events.

The next manuscript revision should focus on distribution queries and ordered shape adaptation, with the benefits and costs of availability handling stated explicitly. The present results support an initial algorithm improvement. Literature novelty and publication competitiveness still require the direct comparisons and independent validation described above.

''')
    s+=('## 7 复现材料\n\n' if zh else '## 7 Reproducibility\n\n')
    s+=('新实现、固定协议、逐轮日志、权重、逐起报预测及绘图脚本位于 transformer_revision 目录。主要入口为 model.py、run.py、analyze.py、method_figure.py 和 REPRODUCE.md。环境沿用原稿 Python 3.12.14、PyTorch 2.9.1 与固定 Chronos-2 权重版本。原稿和旧模型均保留。\n\n' if zh else 'The transformer_revision directory contains the implementation, fixed protocol, epoch logs, weights, predictions at each origin and figure scripts. Entry points are model.py, run.py, analyze.py, method_figure.py and REPRODUCE.md. The environment follows the original Python 3.12.14 and PyTorch 2.9.1 setup with the pinned Chronos-2 revision. Previous manuscripts and models are retained.\n\n')
    s+=('## 参考文献\n\n' if zh else '## References\n\n')
    s+='''[1] Y. Wang et al., “TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables,” Advances in Neural Information Processing Systems, 2024. https://papers.nips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html.

[2] G. Qin et al., “CoRA: Covariate-Aware Adaptation of Time Series Foundation Models,” arXiv:2510.12681, 2025. https://arxiv.org/abs/2510.12681.

[3] J. Na, P. Trirat, C. Park, and J.-G. Lee, “Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models,” Proceedings of Machine Learning Research, vol. 306, pp. 91489–91508, 2026. https://proceedings.mlr.press/v306/na26b.html.

[4] A. Benechehab, V. Feofanov, G. Paolo, A. Thomas, M. Filippone, and B. Kégl, “AdaPTS: Adapting Univariate Foundation Models to Probabilistic Multivariate Time Series Forecasting,” Proceedings of the 42nd International Conference on Machine Learning, 2025. https://openreview.net/forum?id=yeICCRy3lE.

[5] C. Adler, Y. Chang, F. Draxler, S. Abdi, and P. Smyth, “Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated?,” International Conference on Learning Representations, 2026. https://arxiv.org/abs/2510.16060.
'''
    (OUT/f'manuscript_{lang}.md').write_text(s,encoding='utf-8')
    for name in ['architecture','performance','missingness','transfer']:shutil.copy2(RES/f'{name}_{lang}.png',OUT/f'{name}_{lang}.png')

builder=(D/'paper/style_revision/build_docx.py').read_text(encoding='utf-8').replace('ERA5_Paper_English_20261017_Polished.docx','ERA5_Algorithm_Transformer_English.docx').replace('ERA5论文_中文_20261017_Polished.docx','ERA5_算法改进_Transformer_中文.docx')
builder=builder.replace('Research draft; Six-site forecasting and spatial holdout evaluation','Algorithm revision with completed comparative experiments')
(OUT/'build_docx.py').write_text(builder,encoding='utf-8')
print('Algorithm reports prepared',OUT)
