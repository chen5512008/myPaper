"""Shared fixed masked histories for an exploratory training-fairness audit."""
from pathlib import Path
import datetime,hashlib,json,sys,warnings
import numpy as np

R=Path(__file__).resolve().parent;D=R.parent;F=D/'fusion_revision_20261017';V=D/'review_followup_20261018'
sys.path.insert(0,str(F))
from common import align,pin,features
from fusion import year_masks
from scoring import metrics,block_ci
SITES=json.loads((D/'protocol.json').read_text())['sites'];H=[6,12,24,48];Q=np.arange(1,10)/10
def load(p):return dict(np.load(p))
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False),encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def anchor(role):return load(F/'prepared'/f'{role}.npz')
def histories(d,batch=1024):
    for si,s in enumerate(SITES):
        z=load(D/'prepared'/f'{s}.npz');ix=np.flatnonzero(d['site']==s)
        ids=np.searchsorted(z['time'],d['origin'][ix]);assert np.array_equal(z['time'][ids],d['origin'][ix])
        assert np.allclose(z['raw'][ids[:,None]+np.array(H),0],d['y'][ix])
        for a in range(0,len(ix),batch):
            jj=ix[a:a+batch];ii=ids[a:a+batch]
            yield si,jj,z['raw'][ii[:,None]-np.arange(167,-1,-1)]
def window_features(x,mask,si,base):
    x=x.copy();x[...,4:]=np.where(mask,x[...,4:],np.nan)
    parts=[x[:,-l-1] for l in [0,1,3,6,12,24,48,72,167]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for w in [6,24,72,168]:
            a=x[:,-w:][:,::-1];parts.extend([np.nanmean(a,1),np.nanstd(a,1)])
    f=np.concatenate(parts+[np.full((len(x),1),si)]+[x[:,-1]-x[:,-l-1] for l in [6,24,72]],axis=1).astype('float32')
    fixed=[i for i in range(136) if i%8<4]+[136]+[i for i in range(137,161) if (i-137)%8<4]
    f[:,fixed]=base[:,fixed]
    # Both tabular models receive the same explicit missingness information.
    return np.concatenate([f,mask.mean(1),mask[:,-24:].mean(1)],1).astype('float32')
def training_mask(n):
    rng=np.random.default_rng(20261018);kind=rng.integers(0,4,n);m=np.ones((n,168,4),bool)
    random=np.flatnonzero(kind==2);m[random]=rng.random((len(random),168,4))>.3
    terminal=np.flatnonzero(kind==3);length=rng.choice([3,6,12,24,168],len(terminal))
    for size in [3,6,12,24,168]:m[terminal[length==size],-size:]=False
    return m,{'complete':int((kind<2).sum()),'random30':len(random),'terminal':{str(k):int((length==k).sum()) for k in [3,6,12,24,168]}}
def prepare():
    out=R/'prepared';out.mkdir(exist_ok=True)
    protocol={'created_utc':now(),'purpose':'Equalize a fixed set of weather masks across retrained LightGBM, TabM and empirical adapter','status':'Exploratory analysis after all 2022-2026Q3 results were inspected; no new confirmatory holdout','training':'2000-2019 targets','early_stopping':'2020 complete-input targets for every model','2021':'report only, not a model search','model_seeds':{'LightGBM':[17],'TabM':[17,29,43],'Empirical_adapter':[17,29,43]},'mask_seed':20261018,'mask_distribution':'50% complete, 25% random30, 25% terminal length uniformly 3/6/12/24/168','views':'One fixed mask per training origin, shared across all models and neural seeds; not dynamic augmentation per epoch','tabular_features':'161 original causal features plus 4 full-history and 4 last24 weather availability fractions. LightGBM native NaNs and available-only summaries. TabM training-mean imputes feature NaNs and gets the same 8 availability fractions.','models':'Original architectures and optimizer budgets. LightGBM 600 trees/patience40. TabM60 epochs/patience8, global checkpoint only. Adapter30/patience5. No test-driven changes.','test_scenarios':['complete','random30','last24','all_missing'],'evaluation_masks':'Reuse the exact saved masks from reviewer follow-up, seed777'}
    if not (R/'protocol.json').exists():dump(R/'protocol.json',protocol)
    tr=anchor('train');va=anchor('validation');mask,counts=training_mask(len(tr['y']))
    maskfile=out/'shared_training_mask.npz'
    if not maskfile.exists():np.savez_compressed(maskfile,mask=mask,site=tr['site'],origin=tr['origin'])
    else:assert np.array_equal(np.load(maskfile)['mask'],mask)
    for role,d in [('train',tr),('validation',va)]:
        dest=out/f'{role}.npz'
        if dest.exists():continue
        X=np.empty((len(d['y']),169),np.float32)
        for si,ix,x in histories(d):
            mm=mask[ix] if role=='train' else np.ones((len(ix),168,4),bool)
            X[ix]=window_features(x,mm,si,d['X'][ix])
        np.savez_compressed(dest,X=X,y=d['y'],last=d['last'],site=d['site'],origin=d['origin'])
        print('PREPARED',role,X.shape,flush=True)
    raw=np.concatenate([z['raw'][z['time']<'2020'] for z in [load(D/'prepared'/f'{s}.npz') for s in SITES]]).astype('float64')
    mean=raw.mean(0);sd=np.maximum(raw.std(0),1e-8)
    empirical=np.quantile(tr['y']-tr['last'][:,None],Q,axis=0).T
    dump(out/'normalization.json',{'mean':mean.tolist(),'sd':sd.tolist(),'empirical':empirical.tolist(),'training_counts':counts,'mask_sha256':hashlib.sha256(maskfile.read_bytes()).hexdigest(),'raw_mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()})
    print('PREPARATION COMPLETE',counts,flush=True)
def evaluation_features(scenario):
    d=anchor('test');mask=load(V/'diagnostics'/f'mask_{scenario}.npz')['mask'];X=np.empty((len(d['y']),169),np.float32)
    for si,ix,x in histories(d):X[ix]=window_features(x,mask[ix],si,d['X'][ix])
    return d,X,mask
def save(path,p,d):
    p=np.maximum(0,np.sort(p,axis=-1)).astype('float32');assert np.isfinite(p).all()
    np.savez_compressed(path,prediction=p,target=d['y'],site=d['site'],origin=d['origin'])
if __name__=='__main__':prepare()
