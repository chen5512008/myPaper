"""Pre-test frozen leave-one-expert-out horizon-pool diagnostics."""
from common import *
from bank import validation_bank
from fusion import fit_weights,pool,year_masks
d,bank=validation_bank();names=list(bank);fit,sel=year_masks(d);p=np.stack(list(bank.values()),-1)
configs=[('all',names)]+[(f'without_{n}',[a for a in names if a!=n]) for n in names]+[('LightGBM_TabM',['LightGBM','TabM']),('LightGBM_TabM_adapter',['LightGBM','TabM','Chronos_adapter'])]
rows=[]
for label,subset in configs:
    pp=p[..., [names.index(n) for n in subset]];wh=np.stack([fit_weights(pp[fit,h],d['y'][fit,h],penalty=.001)[0] for h in range(4)])
    value=score(pool(pp,wh)[sel],d['y'][sel]);rows.append({'name':label,'members':subset,'weights':wh.tolist(),'validation2021':value})
    print('ABLATION',label,value,flush=True)
dump(R/'ablation_specs.json',{'timestamp_utc':now(),'fitting':'2020','selection_period_diagnostic':'2021','recipe':'horizon convex weights with fixed .001 penalty; all variants retained, not used to choose final fusion recipe','tests_read':False,'variants':rows})
