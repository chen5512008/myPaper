from pathlib import Path
import json
V=Path(__file__).parent;R=V.parent;D=R.parent;OUT=D/'paper/reviewer_revision'
s=(R/'method_figure.py').read_text(encoding='utf-8')
s=s.replace("R=Path(__file__).parent/'results';R.mkdir(exist_ok=True)","R=Path(__file__).resolve().parents[2]/'paper/reviewer_revision';R.mkdir(exist_ok=True)")
s=s.replace("'冻结基础模型' if zh else 'Frozen Chronos-2'","'参考分布' if zh else 'Reference distribution'")
s=s.replace("'历史波高 → 原始分位数' if zh else 'Wave history → base quantiles'","'Chronos-2 或训练残差\\n两种来源分别实验' if zh else 'Chronos-2 or training residuals\\nEvaluated separately'")
s=s.replace("'可训练参数：36,114；骨干权重保持冻结' if zh else '36,114 trainable parameters; backbone weights frozen'","'可训练参数：36,114；预训练参考的骨干保持冻结' if zh else '36,114 trainable parameters; pretrained reference remains frozen'")
# Escaped source newlines belong inside string literals.
s=s.replace("'Chronos-2 或训练残差\n两种来源分别实验'","'Chronos-2 或训练残差\\n两种来源分别实验'")
s=s.replace("'Chronos-2 or training residuals\nEvaluated separately'","'Chronos-2 or training residuals\\nEvaluated separately'")
(V/'method_figure_review.py').write_text(s,encoding='utf-8')
response='''评语逐项落实与新增数据规格

对应论文：ERA5论文_中文_评语修订稿.docx / ERA5_Paper_English_Review_Revised.docx

P0-1 经验参考是否使研究立论失效
已补测“未适配经验参考”，平均分位数损失为 0.169885；同一经验参考接入 36,114 参数适配器后为 0.122921，降低 27.65%。原文的 0.12292 一直是经验参考加完整神经适配器，不能解释为简单残差预测单独达到该分数。
标题、摘要、引言和结论现在围绕历史预测分布适配，分别评价预训练与经验两种参考。三个经验参考种子的平均损失约 0.12319，与 LightGBM 的 0.12331 接近；与 LightGBM 的区间比较只有 24 小时低于零，其余三个提前量包含零，因此没有主张稳定领先。
对应修改：摘要，第 4.1、5.3、6.1、7.2 节，表 2。

P0-2 高海况误差变差
没有删除已有 MAE 或把问题描述成已解决。补充了高波浪 0.9 分位数损失、区间覆盖率、宽度和区间评分。
Chronos-2 参考适配器的高波浪中位数 MAE 仍为 1.2087 米，高于 LCA 的 1.1442 米；其上尾分位数损失虽低于原生 Chronos-2，但覆盖率变差，不能据此声称更安全。
经验参考适配器的主种子高波浪区间评分为 5.5369 米，未适配经验参考为 7.3258 米，但高波浪覆盖率仍仅 51.05%。高海况安全适用性仍未验证，已从任务动机中减弱安全决策的推断。
对应修改：引言，第 5.4、6.5、6.6、7.4 节，表 8、9。

P1 总体覆盖率不足
新增 CQR 式非负区间扩展，模型权重保持固定，以 2022 年目标确定各提前量校准量，只用 2023 至 2025 年评分。该经典后处理单独归因，不作为本文新算法。
Chronos-2 参考适配器总体覆盖率由 76.03% 升至 80.53%，宽度由 0.8642 至 0.9321 米。经验参考适配器由 78.41% 升至 81.17%，宽度由 0.9444 至 0.9890 米。
校准不改变中位数，因此不能修复中位数高波浪 MAE。高波浪条件覆盖仍远低于 80%；以未来真实波高挑选子集，也不能直接要求它达到总体名义覆盖率。
2022 年和 2023 至 2025 年的目标严格分开，但整个基准此前已查看，复评不能被称为完全未见的确认实验。时间依赖下没有宣称标准共形有限样本保证。
对应修改：第 4.6、5.4、6.6、7.4 节，公式 9 至 11，表 10、图 8。

P1 空间验证与实测数据不足
仍保留六点、六次空间留出的既有实测结果，未虚构浮标验证或增加未运行点位。当前仍是 ERA5 再分析基准；新增外部观测需要按下方规格收集。
未把 ERA5 小时记录数解释为同等数量的独立样本，也未因增加数据量而声称独立验证已完成。

P2 组合式创新
补跑经验参考下的共享尺度、移除分布查询、移除可用性三个相同参数量对照，固定主种子 17 与原训练协议。
经验参考共享尺度损失为 0.12500，移除分布查询为 0.12312，完整版本为 0.12292。经验参考是当前波高与提前量的确定性函数，查询已包含这两项，因此参考特征在信息上冗余。这个条件解释了查询为何在两类参考下作用不同。
保留 Transformer、注意力和有序参数化的已有研究归属；没有把已有模块逐一包装成首创。方法的跨骨干和跨实测数据集泛化性仍未证实。

参考文献与格式
新增校准方法的原始出处及 2026 年时间序列共形预测综述，按正文首次引用顺序重新编号。保留五段式引言和完整研究论文结构，回应及操作信息仅写在本文件，不混入论文主线。
期刊分区、会议等级与具体刊物适配没有从评语照搬为论文事实；投稿前需按所选刊物核实。

最值得补充的数据
优先级 1：真实浮标或测站的有效波高，不是另一份 ERA5、模式再分析或网站截图。
位置：优先中国近海，与已有六个海域有关的站点即可，不必与六个网格点完全重合。1 至 3 个站点可先开展外部检验；跨两个海域更有价值。
时间：至少一个完整年，优先 2022 至 2025 年中的连续区间；两年以上更好。要有常态及完整高海况过程，不只提供挑选出的极端几天。
必须字段：站点编号、时间与时区、纬度、经度、有效波高 SWH/Hs（米）、质量控制标记或缺测说明。必须说明数据机构与测量方式。
可选字段：平均波周期、平均波向、风速风向、站点水深、设备和采样说明。即使只有实测波高也可先做目标验证。
时间分辨率：小时最佳；10 分钟、30 分钟原始观测也可，需保留原采样。不要预先平滑、填补大缺口或筛除高值。
文件格式：NetCDF、CSV、Excel 均可，保留原始文件和元数据。
若站点不在现有网格位置：还需该站坐标附近同一 ERA5 产品的波高、平均周期、波向、u10、v10、阵风、海平面气压。对 2022 至 2025 年外部验证，可提供 2021-12-24 至 2025-12-31 的逐小时输入；不需要未来气象输入。
候选来源：Copernicus Marine 的 Global Ocean Delayed Mode Wave 产品 INSITU_GLO_WAV_DISCRETE_MY_013_045，包含质量控制的波浪实测资料。必须先核查中国近海是否有目标站及连续记录，“全球产品”不保证所需海域有站。产品地址：
https://data.marine.copernicus.eu/product/INSITU_GLO_WAV_DISCRETE_MY_013_045/description
说明：外部实测不自动等于独立于 ERA5 同化；应核查站点是否进入 ERA5 的观测系统。未知时只称“外部观测验证”，不声称完全独立。

优先级 2：确认期数据
可用六个既有点位的 2026 年完整月份作为后续固定方案的时间确认，优先已完成最终发布的月份，七个变量和逐小时时间对齐。下载前应检查本地新增四点文件是否已覆盖，避免重复。它可补充时间确认，但不能代替实测浮标。
无需先增加大量同源 ERA5 点位才能完成当前修订。最关键的缺口是观测来源和独立评价，而不只是记录条数。

复现入口
transformer_revision/reviewer_revision/protocol.json：探索性补充协议。
run.py：三个经验参考消融。
analyze_review.py：未适配参考、配对区间、高波浪诊断及时间分离校准。
results：完整 CSV、参数、原始参考预测和生成图。
paper/revise_from_review.py：基于既有完整论文生成本轮中英文修订。
全部新增运行在原有设置下完成，未根据测试分数调整学习率、种子或网络结构。
'''
(OUT/'评语逐项回应与数据需求.txt').write_text(response.replace('27.65','27.64'),encoding='utf-8')
(V/'literature_calibration.json').write_text(json.dumps([
 {'title':'Conformalized Quantile Regression','year':2019,'url':'https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html','role':'Attribution for calibration score; not a new algorithm claim'},
 {'title':'Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking','year':2026,'url':'https://arxiv.org/abs/2601.18509','role':'Time dependence and calibration context'},
 {'product':'INSITU_GLO_WAV_DISCRETE_MY_013_045','url':'https://data.marine.copernicus.eu/product/INSITU_GLO_WAV_DISCRETE_MY_013_045/description','role':'Candidate external observations; station coverage unverified'}
],ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared reference diagram script and itemized review response')
