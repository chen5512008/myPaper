"""Prepare identical causal features and reserve a new quarter before fusion fitting."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
from common import *
sys.path.insert(0,str(D.parent/'revision_20261014'))
from prepare_sites import read_source,inspect_source
O=R/'prepared';O.mkdir(exist_ok=True)
protocol={'timestamp_utc':now(),'trigger':'User requested stronger algorithms and fusion after prior models failed stable superiority','known_evaluation_periods':['2022-2025','2026 H1; both previously inspected and exploratory in this turn'],'new_reserved_confirmation':'2026-07-01 through 2026-09-30; predictions generated only after fusion choice frozen','train_base':'2000-2019','base_early_stopping':'2020 for new models; previously cached models selected on2020-2021 retain their original provenance','fusion_fit':'2020 out-of-training predictions','fusion_selection':'2021; fixed before any new fusion test scores','no_extra_label_control':'A horizon/quantile bias-calibrated LightGBM fitted on2020 is included alongside raw LightGBM','new_models':['XGBoost3.4.1 native multi-quantile on future-minus-current wave height','CatBoost1.2.10 MultiQuantile residual model','TabM0.0.3 ICLR2025 architecture with independent member quantile loss and global/per-horizon checkpoints'],'fusion_candidates':['global convex pooling','horizon convex pooling','site/horizon convex pooling with shrinkage','small state-dependent gate with a convex-pool prior'],'baseline_policy':'Compare each new component, raw strong LightGBM, calibrated LightGBM, uniform pool and selected fusion. Do not label a fusion as necessary if it cannot improve its own strongest component.','selection_metric':'Nine-quantile mean pinball over all4leads and6sites; no dropping six-hour horizon','repetitions':'Selected stochastic new components repeated at17,29,43, with fusion recipe locked; existing neural3-seed prediction ensemble is explicitly an ensemble','failure_policy':'Retain all attempted models and scores. No score-driven revisions after new Q3 outcomes. No guarantee of superiority or acceptance.','numerical_order':'Shared nonnegative weights across nine quantiles; monotone expert predictions ensure noncrossing pools','citations':{'TabM':'https://github.com/yandex-research/tabm','TabM_paper':'https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf','CatBoost_MultiQuantile':'https://catboost.ai/docs/en/concepts/loss-functions-regression','regression_2026_preprint':'https://arxiv.org/abs/2610.13909'}}
if not (R/'protocol.json').exists():dump(R/'protocol.json',protocol)
for role in ['train','validation','test']:
    dest=O/f'{role}.npz'
    if dest.exists():continue
    parts=[]
    for si,s in enumerate(SITES):
        z=load(D/'prepared'/f'{s}.npz');ids=z[f'origins_{role}']
        parts.append({'X':features(z['raw'],ids,si),'y':z['raw'][ids[:,None]+H,0],'last':z['raw'][ids,0],'origin':z['time'][ids],'site':np.repeat(s,len(ids))})
    d={k:np.concatenate([a[k] for a in parts]) for k in parts[0]};np.savez_compressed(dest,**d);print('FEATURES',role,len(d['y']),flush=True)
if not (O/'h1.npz').exists():
    parts=[]
    for si,s in enumerate(SITES):
        z=load(A/'confirmation_2026'/f'{s}.npz');ids=z['indices']
        parts.append({'X':features(z['raw'],ids,si),'y':z['raw'][ids[:,None]+H,0],'last':z['raw'][ids,0],'origin':z['time'][ids],'site':np.repeat(s,len(ids))})
    np.savez_compressed(O/'h1.npz',**{k:np.concatenate([a[k] for a in parts]) for k in parts[0]})
sites=json.loads((D/'source_sites.json').read_text(encoding='utf-8-sig'))['sites'];byid={s['id']:s for s in sites}
qdir=R/'reserved_q3';qdir.mkdir(exist_ok=True)
for s in SITES:
    dest=qdir/f'{s}.npz'
    if dest.exists():continue
    site=byid[s];frames=[];sources=[]
    for key in ['atmosphere','waves']:
        ds,sha=read_source(site[key]);frame,coords=inspect_source(ds,site);frames.append(frame)
        ver={}
        if 'expver' in ds:
            da=ds['expver']
            if 'valid_time' in da.dims:da=da.sel(valid_time=slice('2026-07-01','2026-09-30T23:00:00'))
            ver={str(k):int(v) for k,v in zip(*np.unique(da.values,return_counts=True))}
        sources.append({'path':site[key],'sha256':sha,'kind':key,'expver_values':ver,'source_end':str(frame.index.max())})
    frame=frames[0].join(frames[1],how='outer').reindex(pd.date_range('2026-06-24','2026-10-01',freq='h',inclusive='left'))
    angle=np.deg2rad(frame.mwd);raw=np.column_stack([frame.swh,frame.mwp,np.sin(angle),np.cos(angle),frame.u10,frame.v10,frame.fg10,frame.msl]).astype('float32');times=frame.index.values.astype('datetime64[h]')
    ids=np.arange(167,len(times)-48);ids=ids[(times[ids]>=np.datetime64('2026-07-01'))&(times[ids].astype('int64')%6==0)]
    good=np.array([np.isfinite(raw[i-167:i+1]).all() and np.isfinite(raw[i+H,0]).all() for i in ids]);ids=ids[good]
    assert len(ids)>300,'Insufficient complete Q3 coverage; do not relabel a shorter period.'
    np.savez_compressed(dest,raw=raw,time=times.astype(str),indices=ids);dump(qdir/f'{s}.json',{'site':s,'sources':sources,'n':len(ids),'excluded_incomplete':int((~good).sum())});print('RESERVED',s,len(ids),flush=True)
zs={s:load(qdir/f'{s}.npz') for s in SITES};common=zs[SITES[0]]['time'][zs[SITES[0]]['indices']]
for z in zs.values():common=np.intersect1d(common,z['time'][z['indices']])
parts=[]
for si,s in enumerate(SITES):
    z=zs[s];ids=z['indices'][np.isin(z['time'][z['indices']],common)];z['indices']=ids;np.savez_compressed(qdir/f'{s}.npz',**z)
    parts.append({'X':features(z['raw'],ids,si),'y':z['raw'][ids[:,None]+H,0],'last':z['raw'][ids,0],'origin':z['time'][ids],'site':np.repeat(s,len(ids))})
np.savez_compressed(O/'q3.npz',**{k:np.concatenate([a[k] for a in parts]) for k in parts[0]});dump(qdir/'alignment.json',{'n_per_site':len(common),'first_origin':str(common[0]),'last_origin':str(common[-1]),'provisional_note':'Source expver recorded if present; fixed-download ERA5 snapshot comparison, not independent buoy truth.'})
print('READY',len(common),'per site',flush=True)
