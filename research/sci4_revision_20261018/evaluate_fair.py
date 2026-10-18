"""Score the prespecified same-mask retraining controls, including unfavourable results."""
import copy,gc,importlib.util,time,torch
import pandas as pd
import lightgbm as lgb
from fair_common import *
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings
torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
out=R/'evaluation';out.mkdir(exist_ok=True)
configs=[('lightgbm',17)]+[(name,seed) for name in ['tabm','tabm_flags','adapter'] for seed in [17,29,43]]
assert all((R/'runs'/f'{name}_{seed}'/'completed.json').exists() for name,seed in configs),'Finish every prespecified training run before evaluation.'
checkpoints={}
for name,seed in configs:
    folder=R/'runs'/f'{name}_{seed}'
    for p in folder.iterdir():
        if p.suffix in ['.txt','.pt']:checkpoints[p.relative_to(R).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
dump(out/'frozen_checkpoints.json',{'created_utc':now(),'checkpoints':checkpoints,'protocol_sha256':hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()})
spec=importlib.util.spec_from_file_location('review_adapter',V/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
for name,seed in configs:
    folder=R/'runs'/f'{name}_{seed}'
    if name=='lightgbm':models=[[lgb.Booster(model_file=str(folder/f'h{h}_q{q}.txt')) for q in range(1,10)] for h in H]
    elif name.startswith('tabm'):
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        model=TabM.make(n_num_features=ck.get('n_num_features',168),cat_cardinalities=[6],d_out=36,num_embeddings=PiecewiseLinearEmbeddings(ck['bins'],d_embedding=8,activation=False,version='B'),n_blocks=2,d_block=256,dropout=.1,k=16).to(device)
        model.load_state_dict(ck['state']);model.eval()
    else:
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        model=mod.DistributionQueryAdapter(ck['sd'][0],ck['mean'][0],ck['variant'],empirical_reference=ck['empirical_reference']).to(device);model.load_state_dict(ck['state_dict']);model.eval()
    for scenario in ['complete','random30','last24','all_missing']:
        dest=out/f'{name}_{seed}__{scenario}.npz'
        if dest.exists():continue
        d,X,masks=evaluation_features(scenario);p=np.empty((len(d['y']),4,9),np.float32)
        if name=='lightgbm':p=np.stack([np.stack([b.predict(X,num_threads=2) for b in ms],-1) for ms in models],1)
        elif name.startswith('tabm'):
            xx=np.delete(X,136,axis=1);missing_cols=ck.get('missing_cols',[])
            if len(missing_cols):xx=np.concatenate([xx,np.isnan(xx[:,missing_cols]).astype('float32')],1)
            xx=((np.where(np.isfinite(xx),xx,ck['mean'])-ck['mean'])/ck['sd']).astype('float32')
            with torch.no_grad():
                for a in range(0,len(xx),256):
                    ids=slice(a,a+256);v=model(torch.tensor(xx[ids],device=device),torch.tensor(X[ids,136:137].astype('int64'),device=device))
                    p[ids]=v.mean(1).reshape(-1,4,9).cpu().numpy()*ck['scale']+d['last'][ids,None,None]
            del xx
        else:
            for si,ix,history in histories(d,batch=256):
                xx=((history-np.array(ck['mean']))/np.array(ck['sd'])).astype('float32')
                with torch.no_grad():p[ix]=model(torch.tensor(xx,device=device),torch.zeros((len(ix),4,9),device=device),torch.tensor(masks[ix],device=device))[0].cpu().numpy()
        save(dest,p,d);print('FAIR SCORED',name,seed,scenario,float(pin(np.maximum(0,np.sort(p,-1)),d['y']).mean()),flush=True)
        del X,p,masks;gc.collect()
    if name!='lightgbm':del model,ck;torch.cuda.empty_cache()
    else:del models
d=anchor('test');threshold={s:json.loads((D/'prepared'/f'{s}.json').read_text())['training_high_wave_threshold_m'] for s in SITES};high=d['y']>=np.array([threshold[str(s)] for s in d['site']])[:,None]
rows=[];cis=[];bycell=[]
for scenario in ['complete','random30','last24','all_missing']:
    forecasts={}
    for name,seed in configs:
        z=load(out/f'{name}_{seed}__{scenario}.npz');p=align(z,d)['prediction'];forecasts[(name,seed)]=p
        rows.append({'model':name,'seed':seed,'scenario':scenario,**metrics(p,d['y'],high)})
        for s in SITES:
            ix=d['site']==s
            for j,h in enumerate(H):bycell.append({'model':name,'seed':seed,'scenario':scenario,'site':s,'horizon':h,**metrics(p[ix,j:j+1],d['y'][ix,j:j+1],high[ix,j:j+1])})
    for seed in [17,29,43]:
        for ref in ['lightgbm','tabm','tabm_flags']:
            reference=forecasts[ref,17 if ref=='lightgbm' else seed];diff=(pin(forecasts['adapter',seed],d['y'])-pin(reference,d['y'])).mean((1,2))
            for days in [7,28]:cis.append({'scenario':scenario,'seed':seed,'model':'adapter','reference':ref,**block_ci(diff,d['origin'],days)})
    for ref in ['lightgbm','tabm','tabm_flags']:
        differences=[]
        for seed in [17,29,43]:
            reference=forecasts[ref,17 if ref=='lightgbm' else seed]
            differences.append((pin(forecasts['adapter',seed],d['y'])-pin(reference,d['y'])).mean((1,2)))
        diff=np.mean(differences,axis=0)
        for days in [7,28]:cis.append({'scenario':scenario,'seed':'mean3','model':'adapter','reference':ref,**block_ci(diff,d['origin'],days)})
pd.DataFrame(rows).to_csv(out/'scores_by_seed.csv',index=False)
pd.DataFrame(bycell).to_csv(out/'scores_by_site_lead.csv',index=False)
pd.DataFrame(cis).to_csv(out/'paired_intervals.csv',index=False)
df=pd.DataFrame(rows);df.groupby(['model','scenario']).agg(pinball_mean=('pinball','mean'),pinball_sd=('pinball','std'),median_MAE=('median_MAE','mean'),coverage80=('coverage80','mean'),width80=('width80','mean'),interval_score80=('interval_score80','mean'),highwave_MAE=('highwave_MAE','mean')).reset_index().to_csv(out/'summary.csv',index=False)
print(df[['model','seed','scenario','pinball']].to_string(index=False),flush=True)
