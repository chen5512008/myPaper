"""All three seeds, fixed fusion recipe; preserve rather than select repetitions."""
import pandas as pd
from common import *
from bank import apply_controls
from fusion import pool
from gate import apply_gate
from scoring import metrics,block_ci
choice=json.loads((R/'selection.json').read_text());names=choice['members'];obj=choice['object'];ctrl=load(R/'selection/controls.npz');out=R/'results';out.mkdir(exist_ok=True)
threshold={}
for s in SITES:
    z=load(D/'prepared'/f'{s}.npz');threshold[s]=np.quantile(z['raw'][z['time'].astype('datetime64[h]')<np.datetime64('2020-01-01'),0],.95)
rows=[];details=[];cis=[];ablations=[]
for role in ['test','h1','q3']:
    d=load(R/'prepared'/f'{role}.npz');anchor={'target':d['y'],'site':d['site'],'origin':d['origin']};sid=np.array([SITES.index(str(s)) for s in d['site']]);high=d['y']>np.array([threshold[str(s)] for s in d['site']])[:,None];losses={}
    for seed in [17,29,43]:
        folder=R/'predictions'/role if seed==17 else R/'predictions'/f'seed{seed}'/role
        bank={n:align(load(folder/f'{n}.npz'),anchor)['prediction'] for n in names};pp=np.stack(list(bank.values()),-1)
        fused=pool(pp,np.array(obj['weights']),sid) if 'weights' in obj else apply_gate(pp,d,obj['path'])[0]
        bank.update(apply_controls(d,bank['LightGBM'],ctrl));bank['Uniform_pool']=pp.mean(-1);bank['Selected_fusion']=fused
        for name,p in bank.items():
            losses.setdefault(name,[]).append(pin(p,d['y']).mean((1,2)))
            rows.append({'period':role,'seed':seed,'model':name,**metrics(p,d['y'],high)})
            if name in ['Selected_fusion','TabM','LightGBM','LightGBM_calibrated','Uniform_pool']:
                for s in SITES:
                    ix=d['site']==s
                    for j,h in enumerate(H):details.append({'period':role,'seed':seed,'model':name,'site':s,'horizon':int(h),**metrics(p[ix,j:j+1],d['y'][ix,j:j+1],high[ix,j:j+1])})
            save_predictions(out/f'{role}_seed{seed}_{name}.npz',p,d)
        if seed==17:
            specs=json.loads((R/'ablation_specs.json').read_text())['variants']
            for a in specs:
                pred=pool(pp[..., [names.index(n) for n in a['members']]],np.array(a['weights']))
                ablations.append({'period':role,'variant':a['name'],'validation2021':a['validation2021'],**metrics(pred,d['y'],high)})
    fl=np.mean(losses['Selected_fusion'],axis=0)
    for name in names+['LightGBM_calibrated','Uniform_pool']:
        diff=fl-np.mean(losses[name],axis=0)
        for days in [7,28]:cis.append({'period':role,'model':'Selected_fusion_mean_loss_over_seeds','reference':name,**block_ci(diff,d['origin'],days)})
    print('REPEATS',role,{n:float(np.mean(v)) for n,v in losses.items()},flush=True)
frame=pd.DataFrame(rows);frame.to_csv(out/'scores_by_seed.csv',index=False,encoding='utf-8-sig')
numeric=['pinball','median_MAE','coverage80','width80','interval_score80','highwave_MAE','highwave_q90_loss']
summary=frame.groupby(['period','model'])[numeric].agg(['mean','std']).reset_index();summary.columns=['_'.join(t).strip('_') for t in summary.columns.to_flat_index()];summary.to_csv(out/'scores_mean_std.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(details).to_csv(out/'site_horizon_by_seed.csv',index=False,encoding='utf-8-sig');pd.DataFrame(cis).to_csv(out/'fusion_repeated_block_intervals.csv',index=False,encoding='utf-8-sig');pd.DataFrame(ablations).to_csv(out/'ablation_scores.csv',index=False,encoding='utf-8-sig')
dump(out/'repeat_evaluation.json',{'timestamp_utc':now(),'seeds':[17,29,43],'fusion_selection':'Fit2020 and select2021 at seed17; hold recipe and weights fixed for all new expert repetitions','average_definition':'Mean of three independently evaluated losses and metrics, not loss of averaged predictions','uncertainty':'Sample SD over3 new expert seeds; synchronized7/28day moving time blocks on seed-mean per-origin loss differences','no_post_confirmation_tuning':True})
