"""Validation-selected stronger simple baseline, fixed before new test access."""
from pathlib import Path
import json,time,datetime,hashlib
import numpy as np
import lightgbm as lgb
R=Path(__file__).parent;D=R.parent;O=R/'strong_lightgbm';O.mkdir(exist_ok=True)
H=np.array([6,12,24,48]);Q=np.arange(1,10)/10
protocol={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Check that any adapter gain survives a stronger simple baseline','config':'Original 137 features plus 24 lag changes; 600 trees maximum, 31 leaves, lr .05, patience40, other original settings. Select old or new entire model by 2020-2021 validation aggregate pinball.','test':'Evaluated only after baseline choice recorded. New exploratory experiment on previously inspected benchmark.'}
if not (O/'protocol.json').exists():(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
data={r:[] for r in ['train','validation','test']}
for si,site in enumerate(json.loads((D/'protocol.json').read_text())['sites']):
 z=np.load(D/'prepared'/f'{site}.npz');raw=z['raw']
 for role in data:
  ids=z[f'origins_{role}'];pieces=[raw[ids-l] for l in [0,1,3,6,12,24,48,72,167]]
  for w in [6,24,72,168]:
   hist=np.stack([raw[ids-l] for l in range(w)],axis=1);pieces += [hist.mean(1),hist.std(1)]
  feat=np.concatenate(pieces+[np.full((len(ids),1),si)]+[raw[ids]-raw[ids-l] for l in [6,24,72]],axis=1)
  data[role].append({'X':feat,'y':raw[ids[:,None]+H,0],'origin':z['time'][ids],'site':np.repeat(site,len(ids))})
data={r:{k:np.concatenate([s[k] for s in sites]) for k in sites[0]} for r,sites in data.items()}
pred=np.zeros((len(data['validation']['y']),4,9),dtype=np.float32);logs=[];start=time.perf_counter()
for hi,h in enumerate(H):
 for qi,q in enumerate(Q):
  path=O/f'h{h}_q{qi+1}.txt'
  if path.exists():booster=lgb.Booster(model_file=str(path));iteration=booster.current_iteration()
  else:
   model=lgb.LGBMRegressor(objective='quantile',alpha=float(q),n_estimators=600,num_leaves=31,learning_rate=.05,min_child_samples=50,colsample_bytree=1.,random_state=17,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
   model.fit(data['train']['X'],data['train']['y'][:,hi],eval_set=[(data['validation']['X'],data['validation']['y'][:,hi])],callbacks=[lgb.early_stopping(40,verbose=False)])
   booster=model.booster_;iteration=model.best_iteration_;booster.save_model(str(path))
  pred[:,hi,qi]=booster.predict(data['validation']['X']);logs.append({'horizon':int(h),'quantile':float(q),'best_iteration':iteration})
  print('STRONG_LGBM',h,round(q,1),iteration,'seconds',round(time.perf_counter()-start,1),flush=True)
pred=np.maximum(0,np.sort(pred,axis=-1));d=data['validation']
np.savez_compressed(O/'validation.npz',prediction=pred,target=d['y'],site=d['site'],origin=d['origin'])
def score(p,y):
 e=y[...,None]-p;return float(np.maximum(Q*e,(Q-1)*e).mean())
old=dict(np.load(D/'tabular/validation.npz'));newscore=score(pred,d['y']);oldscore=score(old['prediction'],old['target'])
chosen='extended' if newscore<oldscore else 'original'
(O/'selection.json').write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'chosen':chosen,'original_validation':oldscore,'extended_validation':newscore,'features':d['X'].shape[1],'logs':logs,'seconds':time.perf_counter()-start},indent=2))
print('BASELINE_FROZEN',chosen,newscore,oldscore,flush=True)
d=data['test'];pred=np.zeros((len(d['y']),4,9),dtype=np.float32)
for hi,h in enumerate(H):
 for qi,q in enumerate(Q):pred[:,hi,qi]=lgb.Booster(model_file=str(O/f'h{h}_q{qi+1}.txt')).predict(d['X'])
pred=np.maximum(0,np.sort(pred,axis=-1))
np.savez_compressed(O/'test.npz',prediction=pred,target=d['y'],site=d['site'],origin=d['origin'])
(O/'test_score.json').write_text(json.dumps({'extended_test_pinball':score(pred,d['y'])},indent=2));print('BASELINE_COMPLETE',flush=True)
