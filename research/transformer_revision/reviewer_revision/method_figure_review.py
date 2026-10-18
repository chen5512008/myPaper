from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=Path(__file__).resolve().parents[2]/'paper/reviewer_revision';R.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'svg.fonttype':'none'})
for lang in ['en','zh']:
    zh=lang=='zh';fig,ax=plt.subplots(figsize=(12,5.5));ax.set_xlim(0,12);ax.set_ylim(0,5.5);ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=.1',linewidth=.8,edgecolor='#adc0c7',facecolor=color))
        ax.text(x+.12,y+h-.22,title,fontsize=10,weight='bold',color='#163d4a',va='top')
        ax.text(x+.12,y+h-.58,body,fontsize=8.5,color='#395761',va='top',linespacing=1.5)
    def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=12,lw=1,color='#6f929f',connectionstyle='arc3'))
    box(.15,3.9,2.1,1.05,'参考分布' if zh else 'Reference distribution','Chronos-2 或训练残差\n两种来源分别实验' if zh else 'Chronos-2 or training residuals\nEvaluated separately','#e5edf5')
    box(3,3.9,3,1.05,'分布与提前量查询' if zh else 'Distribution and lead query','9 个分位数 + 当前海浪状态\n+ 6 / 12 / 24 / 48 小时' if zh else '9 quantiles + current wave state\n+ 6 / 12 / 24 / 48 h','#e5edf5')
    box(.15,2,2.1,1.15,'海浪历史' if zh else 'Wave history','168 小时 × 4 特征\n每 6 小时一个分块' if zh else '168 hours × 4 features\n6 hours per patch','#e7f1ed')
    box(3,2,3,1.15,'海浪 Transformer' if zh else 'Wave Transformer','分块自注意力 + 查询交叉注意力' if zh else 'Patch self attention\n+ query cross attention','#e7f1ed')
    box(.15,.2,2.1,1.2,'气象历史与掩码' if zh else 'Weather and observation mask','历史值 + 是否观测\n缺测值先置零' if zh else 'History + observed / missing\nMissing entries set to zero','#edf2e3')
    box(3,.2,3,1.2,'可用性加权 Transformer' if zh else 'Availability weighted Transformer','分块可用率进入注意力\n完整缺失时气象贡献为零' if zh else 'Availability weights attention\nZero weather contribution if absent','#edf2e3')
    box(7,1.6,2.1,1.8,'有序分布修正' if zh else 'Ordered distribution correction','中位数偏移\n8 个相邻间隔分别缩放\n共享正尺度为消融版本' if zh else 'Median shift\n8 positive gap multipliers\nShared scale ablation','#def0ec')
    box(9.85,1.6,1.85,1.8,'预测输出' if zh else 'Wave forecast','非负有序分位数\n中位数与 80% 区间' if zh else 'Nonnegative quantiles\nMedian and 80% interval','#f7ecd9')
    for a in [(2.33,4.4,2.9,4.4),(2.33,2.6,2.9,2.6),(2.33,.8,2.9,.8),(4.5,3.82,4.5,3.23),(6.08,4.4,7.3,3.48),(6.08,2.6,6.9,2.6),(6.08,.8,7.5,1.52),(9.18,2.5,9.77,2.5),(4.5,1.92,4.5,1.48)]:arrow(*a)
    ax.text(.15,5.27,'历史信息驱动的分布查询与缺测感知适配' if zh else 'Distribution queries and availability aware adaptation from historical inputs',fontsize=13,weight='bold',color='#163d4a')
    ax.text(6.35,.35,'可训练参数：36,114；预训练参考的骨干保持冻结' if zh else '36,114 trainable parameters; pretrained reference remains frozen',fontsize=8,color='#52666e')
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(R/f'architecture_{lang}.{ext}',dpi=240)
    plt.close(fig)
