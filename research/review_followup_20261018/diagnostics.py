"""Fixed reviewer diagnostics. Masks and thresholds are shared, not selected on scores."""
from pathlib import Path
import argparse,json,hashlib,sys,gc,warnings
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
V=Path(__file__).resolve().parent;D=V.parent;F=D/'fusion_revision_20261017';T=D/'transformer_revision'
sys.path.insert(0,str(F))
from common import features,align,pin
from scoring import metrics,block_ci
from model import DistributionQueryAdapter
SITES=json.loads((D/'protocol.json').read_text(encoding='utf-8'))['sites'];H=[6,12,24,48]
O=V/'diagnostics';O.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['trees','adapters','analyze'],required=True);parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');args=parser.parse_args()
torch.set_num_threads(2)
anchor=dict(np.load(F/'prepared/test.npz'));N=len(anchor['y'])
train=np.concatenate([z['raw'][z['time']<'2020'] for z in [dict(np.load(D/'prepared'/f'{s}.npz')) for s in SITES]])
mean=train.astype('float64').mean(0);sd=train.astype('float64').std(0);del train
threshold={s:float(json.loads((D/'prepared'/f'{s}.json').read_text(encoding='utf-8'))['training_high_wave_threshold_m']) for s in SITES}
(O/'thresholds.json').write_text(json.dumps(threshold,indent=2),encoding='utf-8')
def mask_for(scenario):
    dest=O/f'mask_{scenario}.npz'
    if dest.exists():return np.load(dest)['mask']
    m=np.ones((N,168,4),dtype=bool)
    if scenario=='random30':
        g=torch.Generator().manual_seed(777)
        for a in range(0,N,512):m[a:a+512]=(torch.rand((min(512,N-a),168,4),generator=g)>.3).numpy()
    elif scenario=='last24':m[:,-24:]=False
    elif scenario=='all_missing':m[:]=False
    np.savez_compressed(dest,mask=m)
    return m
def feature_windows(x,si):
    pieces=[x[:,-l-1] for l in [0,1,3,6,12,24,48,72,167]]
    for w in [6,24,72,168]:
        a=x[:,-w:][:,::-1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            pieces.extend([np.nanmean(a,1),np.nanstd(a,1)])
    return np.concatenate(pieces+[np.full((len(x),1),si)]+[x[:,-1]-x[:,-l-1] for l in [6,24,72]],axis=1).astype('float32')
def histories():
    for si,s in enumerate(SITES):
        z=dict(np.load(D/'prepared'/f'{s}.npz'));ids=z['origins_test'];ix=np.flatnonzero(anchor['site']==s)
        assert np.array_equal(z['time'][ids],anchor['origin'][ix])
        yield si,s,z['raw'][ids[:,None]-np.arange(167,-1,-1)],ix
def save(name,scenario,p):
    assert np.isfinite(p).all()
    p=np.maximum(0,np.sort(p,axis=-1)).astype('float32')
    np.savez_compressed(O/f'{name}__{scenario}.npz',prediction=p,target=anchor['y'],site=anchor['site'],origin=anchor['origin'])
if args.phase=='trees':
    models=[[lgb.Booster(model_file=str(D/'algorithm_improvement/strong_lightgbm'/f'h{h}_q{q}.txt')) for q in range(1,10)] for h in H]
    for scenario in ['complete','random30','last24','all_missing']:
        m=mask_for(scenario)
        for policy in ['native_nan','train_mean']:
            name='LightGBM_'+policy;dest=O/f'{name}__{scenario}.npz'
            if dest.exists():continue
            X=np.empty_like(anchor['X'])
            for si,s,x,ix in histories():
                x=x.copy();x[...,4:]=np.where(m[ix],x[...,4:],np.nan)
                if policy=='train_mean':x=np.where(np.isfinite(x),x,mean).astype('float32')
                X[ix]=feature_windows(x,si)
            unchanged=[i for i in range(136) if i%8<4]+[136]+[i for i in range(137,161) if (i-137)%8<4]
            X[:,unchanged]=anchor['X'][:,unchanged]
            if scenario=='complete':X=anchor['X'].copy()
            p=np.stack([np.stack([b.predict(X,num_threads=2) for b in ms],-1) for ms in models],1)
            save(name,scenario,p);print(name,scenario,float(pin(p,anchor['y']).mean()),flush=True)
    print('TREES COMPLETE',flush=True)
elif args.phase=='adapters':
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    configs=[('Empirical_adapter',T/'runs/pooled_empirical_reference_17/model.pt'),('Chronos_adapter',T/'runs/pooled_full_17/model.pt'),('Empirical_no_availability',T/'reviewer_revision/runs/pooled_empirical_no_availability_17/model.pt'),('Chronos_no_availability',T/'runs/pooled_no_availability_17/model.pt')]
    bases=np.concatenate([np.load(D/'forecasts'/f'{s}_test_base.npz')['prediction'] for s in SITES])
    for name,path in configs:
        ck=torch.load(path,map_location='cpu',weights_only=False);model=DistributionQueryAdapter(ck['sd'][0],ck['mean'][0],ck['variant'],empirical_reference=ck.get('empirical_reference')).to(device)
        model.load_state_dict(ck['state_dict']);model.eval()
        for scenario in ['complete','random30','last24','all_missing']:
            if (O/f'{name}__{scenario}.npz').exists():continue
            if scenario!='random30' and (path.parent/f'{scenario}.npz').exists():
                cached=dict(np.load(path.parent/f'{scenario}.npz'));cached=align(cached,anchor)
                mask_for(scenario)
                save(name,scenario,cached['prediction'])
                note=O/'reused_predictions.json';notes=json.loads(note.read_text(encoding='utf-8')) if note.exists() else []
                notes.append({'model':name,'scenario':scenario,'source':str(path.parent/f'{scenario}.npz'),'sha256':hashlib.sha256((path.parent/f'{scenario}.npz').read_bytes()).hexdigest(),'reason':'Deterministic complete/terminal24/all-absent mask equals the original evaluation mask; no random stream involved. Origin/site/target alignment verified.'})
                note.write_text(json.dumps(notes,indent=2),encoding='utf-8')
                continue
            masks=mask_for(scenario);p=np.empty((N,4,9),dtype='float32')
            for si,s,x,ix in histories():
                x=((x-np.array(ck['mean']))/np.array(ck['sd'])).astype('float32')
                for a in range(0,len(ix),512):
                    jj=ix[a:a+512]
                    with torch.no_grad():
                        pp=model(torch.tensor(x[a:a+512],device=device),torch.tensor(bases[jj],device=device),torch.tensor(masks[jj],device=device))[0]
                    p[jj]=pp.cpu().numpy()
            save(name,scenario,p);print(name,scenario,float(pin(p,anchor['y']).mean()),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
else:
    rows=[];cellrows=[];preds={}
    for path in sorted(O.glob('*__*.npz')):
        name,scenario=path.stem.split('__');z=dict(np.load(path));p=z['prediction'];y=anchor['y'];high=np.zeros_like(y,dtype=bool)
        for s in SITES:high[anchor['site']==s]=y[anchor['site']==s]>=threshold[s]
        rows.append({'model':name,'scenario':scenario,**metrics(p,y,high)})
        preds[(name,scenario)]=p
        for s in SITES:
            ix=anchor['site']==s
            for j,h in enumerate(H):
                yy=y[ix,j:j+1];pp=p[ix,j:j+1]
                highcell=yy>=threshold[s]
                cellrows.append({'model':name,'scenario':scenario,'site':s,'horizon':h,**metrics(pp,yy,highcell),'highwave_coverage80':float(((yy>=pp[...,0])&(yy<=pp[...,-1]))[highcell].mean()),'highwave_width80':float((pp[...,-1]-pp[...,0])[highcell].mean()),'highwave_IS80':float((pp[...,-1]-pp[...,0]+10*np.maximum(pp[...,0]-yy,0)+10*np.maximum(yy-pp[...,-1],0))[highcell].mean()),'highwave_bias':float((pp[...,4]-yy)[highcell].mean())})
    # Strong complete-input contemporaries, using the same thresholds and macro aggregation.
    for name,paths in [(name,[F/'results'/f'test_seed{seed}_{key}.npz' for seed in [17,29,43]]) for name,key in [('TabM','TabM'),('Fusion','Selected_fusion')]]:
        for seed,path in zip([17,29,43],paths):
            if not path.exists():continue
            p=align(dict(np.load(path)),anchor)['prediction'];y=anchor['y']
            for s in SITES:
                ix=anchor['site']==s
                for j,h in enumerate(H):
                    yy=y[ix,j:j+1];pp=p[ix,j:j+1]
                    highcell=yy>=threshold[s]
                    cellrows.append({'model':name,'scenario':'complete','site':s,'horizon':h,'seed':seed,**metrics(pp,yy,highcell),'highwave_coverage80':float(((yy>=pp[...,0])&(yy<=pp[...,-1]))[highcell].mean()),'highwave_width80':float((pp[...,-1]-pp[...,0])[highcell].mean()),'highwave_IS80':float((pp[...,-1]-pp[...,0]+10*np.maximum(pp[...,0]-yy,0)+10*np.maximum(yy-pp[...,-1],0))[highcell].mean()),'highwave_bias':float((pp[...,4]-yy)[highcell].mean())})
    pd.DataFrame(rows).to_csv(O/'overall.csv',index=False);cells=pd.DataFrame(cellrows);cells.to_csv(O/'by_site_lead.csv',index=False)
    cells.groupby(['model','scenario'])[['pinball','highwave_MAE','highwave_q90_loss','coverage80','interval_score80','highwave_coverage80','highwave_width80','highwave_IS80','highwave_bias']].mean().to_csv(O/'macro.csv')
    cis=[]
    if ('Empirical_adapter','complete') in preds:
        for scenario in ['complete','random30','last24','all_missing']:
            for policy in ['native_nan','train_mean']:
                diff=(pin(preds['Empirical_adapter',scenario],anchor['y'])-pin(preds['LightGBM_'+policy,scenario],anchor['y'])).mean((1,2))
                for days in [7,28]:cis.append({'scenario':scenario,'reference':'LightGBM_'+policy,**block_ci(diff,anchor['origin'],days)})
    pd.DataFrame(cis).to_csv(O/'paired_intervals.csv',index=False)
    audit={'masks':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('mask_*.npz')},'checkpoint_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'algorithm_improvement/strong_lightgbm').glob('h*_q*.txt')},'scope':'Frozen complete-trained LightGBM with two fixed missing-input policies, not a retrained mask-augmented baseline. Diagnostics concern base adapters, not fusion.'}
    (O/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows)[['model','scenario','pinball','highwave_MAE']].to_string(index=False),flush=True)
