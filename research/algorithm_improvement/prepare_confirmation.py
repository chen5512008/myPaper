"""Reserve 2026 H1 before new candidate test scoring; no model selection here."""
from pathlib import Path
import sys,json,datetime
import numpy as np,pandas as pd
R=Path(__file__).parent;D=R.parent
sys.path.insert(0,str(D.parent/'revision_20261014'))
from prepare_sites import read_source,inspect_source
O=R/'confirmation_2026';O.mkdir(exist_ok=True)
protocol={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'role':'Additional chronological evaluation reserved before new candidate test scoring. No parameter fitting, hyperparameter tuning, calibration or selection on these outcomes.','targets':'2026-01-01 through 2026-06-30 inclusive; all four targets must be in interval','context_hours':168,'stride_hours':6,'horizons':[6,12,24,48],'source':'Existing downloaded ERA5 files; not external buoy validation. Different season span from original full-year benchmark.','freeze':'Run once after selecting model using 2020-2021 validation and completing seeds17,29,43. No iteration in response to confirmation metrics.'}
if not (O/'protocol.json').exists():(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
sites=json.loads((D/'source_sites.json').read_text(encoding='utf-8-sig'))['sites'];out=[]
for site in sites:
    dest=O/f"{site['id']}.npz"
    if dest.exists():continue
    frames=[];sources=[]
    for key in ['atmosphere','waves']:
        ds,sha=read_source(site[key]);frame,coords=inspect_source(ds,site);frames.append(frame);sources.append({'path':site[key],'sha256':sha,'kind':key})
    frame=frames[0].join(frames[1],how='outer').reindex(pd.date_range('2025-12-24','2026-07-01',freq='h',inclusive='left'))
    angle=np.deg2rad(frame['mwd'].to_numpy());raw=np.column_stack([frame.swh,frame.mwp,np.sin(angle),np.cos(angle),frame.u10,frame.v10,frame.fg10,frame.msl]).astype('float32')
    time=frame.index.values.astype('datetime64[h]');ids=np.arange(167,len(time)-48);ids=ids[(time[ids]>=np.datetime64('2026-01-01'))&((time[ids].astype('int64')%6)==0)]
    valid=np.array([np.isfinite(raw[i-167:i+1]).all() and np.isfinite(raw[i+np.array([6,12,24,48]),0]).all() for i in ids]);ids=ids[valid]
    np.savez_compressed(dest,raw=raw,time=time.astype(str),indices=ids)
    (O/f"{site['id']}.json").write_text(json.dumps({'site':site['id'],'sources':sources,'n':len(ids),'first_origin':str(time[ids[0]]) if len(ids) else None,'last_origin':str(time[ids[-1]]) if len(ids) else None,'excluded_missing_windows':int((~valid).sum())},indent=2))
    print('RESERVED',site['id'],len(ids),flush=True)
# Align origins before any prediction generation.
files=[O/f"{s['id']}.npz" for s in sites];loaded=[dict(np.load(f)) for f in files]
common=loaded[0]['time'][loaded[0]['indices']]
for a in loaded[1:]:common=np.intersect1d(common,a['time'][a['indices']])
assert len(common)>0
for f,a in zip(files,loaded):a['indices']=a['indices'][np.isin(a['time'][a['indices']],common)];np.savez_compressed(f,**a)
(O/'alignment.json').write_text(json.dumps({'common_origins_per_site':len(common),'first_origin':str(common[0]),'last_origin':str(common[-1])},indent=2))
print('COMMON',len(common),flush=True)
