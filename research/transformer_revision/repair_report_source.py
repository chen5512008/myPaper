from pathlib import Path
p=Path(__file__).parent/'build_report.py'
s=p.read_text(encoding='utf-8')
a=s.index("    s+=r'$$u_");b=s.index("    s+=('''原始分位数",a)
fixed=r'''    s+=r'$$u_{t,h}=f_q([q^0_{t,h,1:9}/s_y;\,x^w_t;\,h/48]). \qquad (1)$$'+'\n\n'
    s+=('海浪交叉注意力先生成摘要 $z^w_{t,h}$，气象查询使用 $u_{t,h}+z^w_{t,h}$。气象自注意力和交叉注意力均在键位置加入可用率的对数：\n\n' if zh else 'Wave cross attention first produces $z^w_{t,h}$; weather retrieval uses $u_{t,h}+z^w_{t,h}$. Weather self attention and cross attention add log availability at key positions:\n\n')
    s+=r'$$A^m_{t,h,j}=\operatorname{softmax}_j\left(\frac{Q_{t,h}^{\top}K_{t,j}}{\sqrt d}+\log\{\max(r_{t,j},10^{-8})\}\right). \qquad (2)$$'+'\n\n'
    s+=('两个输出头生成九维修正系数，合成为\n\n' if zh else 'Two heads produce nine correction coefficients, combined as\n\n')
    s+=r'$$\theta_{t,h}=\theta^w_{t,h}+\bar r_t\,\theta^m_{t,h}. \qquad (3)$$'+'\n\n'
    s+=('第一维调整中位数，其余八维调整相邻分位数间隔：\n\n' if zh else 'The first coefficient shifts the median, and the remaining eight scale adjacent quantile gaps:\n\n')
    s+=r'$$\widehat m_{t,h}=q^0_{t,h,5}+s_y\theta_{t,h,0},\quad \widehat d_{t,h,j}=(q^0_{t,h,j+1}-q^0_{t,h,j})\exp\{2\tanh(\theta_{t,h,j})\}. \qquad (4)$$'+'\n\n'
    s+=r'$$\widehat q_{t,h,k}=\max\left\{0,\begin{cases}\widehat m_{t,h}-\sum_{j=k}^{4}\widehat d_{t,h,j},&k<5,\\\widehat m_{t,h},&k=5,\\\widehat m_{t,h}+\sum_{j=5}^{k-1}\widehat d_{t,h,j},&k>5.\end{cases}\right. \qquad (5)$$'+'\n\n'
'''
s=s[:a]+fixed+s[b:]
s=s.replace("s+=('''","s+=(r'''").replace("''' if zh else '''","''' if zh else r'''")
p.write_text(s,encoding='utf-8')
