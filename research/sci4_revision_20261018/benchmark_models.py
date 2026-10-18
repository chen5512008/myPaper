"""Common CPU model-only latency; feature extraction and foundation inference excluded."""
import importlib.util,platform,time,torch
import pandas as pd
import lightgbm as lgb
from fair_common import *
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings
torch.set_num_threads(2);out=R/'resources';out.mkdir(exist_ok=True)
d=anchor('test');ids=np.random.default_rng(20261018).choice(len(d['y']),512,replace=False);small={k:v[ids] for k,v in d.items()}
hist=np.empty((512,168,8),np.float32)
for _,ix,x in histories(small):hist[ix]=x
mask=np.ones((512,168,4),bool);X=np.empty((512,169),np.float32)
t0=time.perf_counter()
for si in range(6):
    ix=np.flatnonzero(small['site']==SITES[si]);X[ix]=window_features(hist[ix],mask[ix],si,small['X'][ix])
feature_seconds=time.perf_counter()-t0
rows=[];predictions={}
spec=importlib.util.spec_from_file_location('review_adapter',V/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
for label in ['lightgbm','tabm','tabm_flags','adapter']:
    folder=R/'runs'/f'{label}_17'
    if label=='lightgbm':
        files=list(folder.glob('h*_q*.txt'));models=[[lgb.Booster(model_file=str(folder/f'h{h}_q{q}.txt')) for q in range(1,10)] for h in H];nparams=None
        def infer(n):return np.maximum(0,np.sort(np.stack([np.stack([b.predict(X[:n],num_threads=2) for b in ms],-1) for ms in models],1),axis=-1))
    elif label.startswith('tabm'):
        files=[folder/'model.pt'];ck=torch.load(files[0],map_location='cpu',weights_only=False);model=TabM.make(n_num_features=ck.get('n_num_features',168),cat_cardinalities=[6],d_out=36,num_embeddings=PiecewiseLinearEmbeddings(ck['bins'],d_embedding=8,activation=False,version='B'),n_blocks=2,d_block=256,dropout=.1,k=16);model.load_state_dict(ck['state']);model.eval();nparams=sum(p.numel() for p in model.parameters())
        @torch.no_grad()
        def infer(n):
            xx=np.delete(X[:n],136,axis=1);missing_cols=ck.get('missing_cols',[])
            if len(missing_cols):xx=np.concatenate([xx,np.isnan(xx[:,missing_cols]).astype('float32')],1)
            xx=(np.where(np.isfinite(xx),xx,ck['mean'])-ck['mean'])/ck['sd'];v=model(torch.tensor(xx,dtype=torch.float32),torch.tensor(X[:n,136:137].astype('int64')))
            return np.maximum(0,np.sort(v.mean(1).reshape(-1,4,9).numpy()*ck['scale']+small['last'][:n,None,None],axis=-1))
    else:
        files=[folder/'model.pt'];ck=torch.load(files[0],map_location='cpu',weights_only=False);model=mod.DistributionQueryAdapter(ck['sd'][0],ck['mean'][0],ck['variant'],empirical_reference=ck['empirical_reference']);model.load_state_dict(ck['state_dict']);model.eval();nparams=sum(p.numel() for p in model.parameters())
        @torch.no_grad()
        def infer(n):
            x=((hist[:n]-np.array(ck['mean']))/np.array(ck['sd'])).astype('float32')
            return model(torch.tensor(x),torch.zeros((n,4,9)),torch.tensor(mask[:n]))[0].numpy()
    for n in [1,512]:
        for _ in range(2):infer(n)
        times=[]
        for _ in range(9):
            t0=time.perf_counter();p=infer(n);times.append(time.perf_counter()-t0)
        assert np.isfinite(p).all()
        rows.append({'model':label,'batch':n,'median_ms':1000*np.median(times),'p25_ms':1000*np.quantile(times,.25),'p75_ms':1000*np.quantile(times,.75),'origins_per_second':n/np.median(times),'parameters':nparams,'checkpoint_bytes':sum(f.stat().st_size for f in files),'checkpoint_files':len(files),'threads':2,'repeats':9,'device':'CPU'})
    del infer
    if label!='lightgbm':del model,ck
    else:del models
pd.DataFrame(rows).to_csv(out/'model_only_latency.csv',index=False)
dump(out/'protocol.json',{'created_utc':now(),'platform':platform.platform(),'processor':platform.processor(),'torch':torch.__version__,'threads':2,'warmups':2,'repetitions':9,'synchronization':'CPU synchronous calls','input':'Same 512 origin sample, available historical observations; model-specific representation','measurement':'Model normalization/imputation and prediction/postprocessing. Excludes disk IO, history gathering, causal feature construction, foundation model inference and full seven-expert latency. No deployment speed claim for the seven-expert system.','feature_169_construction_batch512_seconds_single_measurement':feature_seconds,'host_caveat':'Shared desktop, timings are indicative and may vary with load.'})
print(pd.DataFrame(rows).to_string(index=False),flush=True)
