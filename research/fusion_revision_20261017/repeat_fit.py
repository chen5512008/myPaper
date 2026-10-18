"""Repeat every new expert at prespecified seeds, then freeze validation choice."""
import subprocess,sys
from common import *
jobs=[]
for seed in [29,43]:
    jobs.extend([(f'tabm_{seed}',[str(R/'train_tabm.py'),'--seed',str(seed),'--checkpoint-rule','global']),
        (f'xgb_{seed}',[str(R/'train_trees.py'),'--model','xgb','--seed',str(seed)]),
        (f'catpoint_{seed}',[str(R/'train_cat_point.py'),'--seed',str(seed)])])
for name,argv in jobs:
    if (R/'experts'/name/'run.json').exists():continue
    dump(R/'pipeline_state.json',{'timestamp_utc':now(),'stage':name,'status':'running','test_read':False})
    print('PIPELINE_START',name,flush=True)
    r=subprocess.run([sys.executable,'-u']+argv,cwd=R)
    if r.returncode:
        dump(R/'pipeline_state.json',{'timestamp_utc':now(),'stage':name,'status':'failed','returncode':r.returncode});raise SystemExit(r.returncode)
if not (R/'selection.json').exists():
    dump(R/'pipeline_state.json',{'timestamp_utc':now(),'stage':'fusion_selection','status':'running','test_read':False})
    r=subprocess.run([sys.executable,'-u',str(R/'select_fusion.py')],cwd=R)
    if r.returncode:raise SystemExit(r.returncode)
dump(R/'pipeline_state.json',{'timestamp_utc':now(),'stage':'training_and_selection','status':'completed','test_read':False})
