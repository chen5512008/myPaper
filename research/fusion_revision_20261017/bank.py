"""Validation loading and post-freeze prediction bank."""
from common import *

def validation_bank():
    d=load(R/'prepared/validation.npz');anchor={'target':d['y'],'site':d['site'],'origin':d['origin']};members={}
    members['LightGBM']=align(load(A/'strong_lightgbm/validation.npz'),anchor)['prediction']
    members['Neural3']=np.mean([align(load(A/f'runs/empirical_summary_flex_only_region_stable_{s}/validation.npz'),anchor)['prediction'] for s in [17,29,43]],axis=0)
    members['Chronos_adapter']=align(load(A/'runs/chronos_summary_flex_gated_stable_17/validation.npz'),anchor)['prediction']
    parts=[load(D/'forecasts'/f'{s}_validation_native.npz') for s in SITES]
    members['Chronos_native']=align({k:np.concatenate([z[k] for z in parts]) for k in ['prediction','target','site','origin']},anchor)['prediction']
    for name,folder in [('XGBoost','xgb_17'),('CatBoost_RMSEcal','catpoint_17'),('TabM','tabm_17')]:
        members[name]=align(load(R/'experts'/folder/'validation.npz'),anchor)['prediction']
    return d,members

def controls(d,p,fit):
    # A post-training calibration control with access to the same 2020 labels.
    off=np.zeros((4,9))
    for h in range(4):
        for q,tau in enumerate(Q):off[h,q]=np.quantile(d['y'][fit,h]-p[fit,h,q],tau)
    tr=load(R/'prepared/train.npz');persistence=np.stack([np.quantile((tr['y']-tr['last'][:,None])[tr['site']==s],Q,axis=0).T for s in SITES])
    return {'calibration_offset':off,'persistence_residual':persistence}

def apply_controls(d,p,c):
    sid=np.array([SITES.index(str(s)) for s in d['site']])
    return {'LightGBM_calibrated':np.maximum(0,np.sort(p+c['calibration_offset'],axis=-1)),
        'Persistence_quantile':np.maximum(0,d['last'][:,None,None]+c['persistence_residual'][sid])}
