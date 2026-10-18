from common import *
from fusion import *
d=load(R/'prepared/validation.npz');anchor={'target':d['y'],'site':d['site'],'origin':d['origin']}
members={}
members['LightGBM']=align(load(A/'strong_lightgbm/validation.npz'),anchor)['prediction']
neural=[align(load(A/f'runs/empirical_summary_flex_only_region_stable_{s}/validation.npz'),anchor)['prediction'] for s in [17,29,43]]
members['Neural3']=np.mean(neural,axis=0)
members['Chronos_adapter']=align(load(A/'runs/chronos_summary_flex_gated_stable_17/validation.npz'),anchor)['prediction']
parts=[load(D/'forecasts'/f'{s}_validation_native.npz') for s in SITES]
members['Chronos_native']=align({k:np.concatenate([z[k] for z in parts]) for k in ['prediction','target','site','origin']},anchor)['prediction']
names=list(members);p=np.stack(list(members.values()),-1);rows,objects,best=linear_candidates(p,d,SITES)
fit,select=year_masks(d)
result={'timestamp_utc':now(),'purpose':'Preliminary validation-only complementarity diagnostic; not final fusion selection','members':names,'single_2021':{n:score(a[select],d['y'][select]) for n,a in members.items()},'candidates':rows,'winner':rows[best],'weights':objects[best]['weights'].tolist(),'no_test_metrics_read':True}
dump(R/'preliminary_fusion.json',result)
print(json.dumps(result,indent=2))
