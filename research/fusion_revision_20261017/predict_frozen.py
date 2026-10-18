"""Inference only after the fusion selection has been frozen."""
import sys,os,gc,argparse
import torch
from common import *

def raw_data(role):
    if role=='test':
        result={}
        for s in SITES:
            z=load(D/'prepared'/f'{s}.npz');z['indices']=load(D/'forecasts'/f'{s}_test_base.npz')['indices'];result[s]=z
        return result
    folder=A/'confirmation_2026' if role=='h1' else R/'reserved_q3'
    return {s:load(folder/f'{s}.npz') for s in SITES}

def chronos_new(data,out):
    os.environ.setdefault('HF_HOME',str(D.parent/'experiments_20261016/hf_cache'));os.environ['HF_HUB_DISABLE_XET']='1'
    if not all((out/f'{s}_{mode}.npz').exists() for s in SITES for mode in ['base','native']):
        from chronos import Chronos2Pipeline
        pipe=Chronos2Pipeline.from_pretrained('amazon/chronos-2',revision='29ec3766d36d6f73f0696f85560a422f50e8498c',local_files_only=True,device_map='cuda',torch_dtype=torch.float32)
        for s,z in data.items():
            raw=z['raw'];ids=z['indices']
            for mode in ['base','native']:
                dest=out/f'{s}_{mode}.npz'
                if dest.exists():continue
                p=[]
                for a in range(0,len(ids),128):
                    tasks=[]
                    for idx in ids[a:a+128]:
                        hist=raw[idx-167:idx+1];task={'target':hist[:,0].copy()}
                        if mode=='native':task['past_covariates']={f'x{j}':hist[:,j].copy() for j in range(1,8)}
                        tasks.append(task)
                    qs,_=pipe.predict_quantiles(tasks,prediction_length=48,quantile_levels=Q.tolist(),batch_size=128,context_length=168,cross_learning=False)
                    p.extend(q[0,H-1,:].numpy() for q in qs)
                np.savez_compressed(dest,prediction=np.maximum(0,np.sort(np.stack(p),-1)),target=raw[ids[:,None]+H,0],site=np.repeat(s,len(ids)),origin=z['time'][ids])
                print('Q3_REFERENCE',s,mode,flush=True)
        del pipe;gc.collect();torch.cuda.empty_cache()
    return {mode:{k:np.concatenate([load(out/f'{s}_{mode}.npz')[k] for s in SITES]) for k in ['prediction','target','site','origin']} for mode in ['base','native']}

def neural_prediction(path,data,base,d):
    sys.path.insert(0,str(A));from model_factory import build_model
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model(ck['sd'][0],ck['mean'][0],ck['variant'],ck['empirical_reference']).to('cuda');model.load_state_dict(ck['state_dict']);model.eval()
    xx=[]
    for s in SITES:
        z=data[s];x=((z['raw']-np.array(ck['mean']))/np.array(ck['sd'])).astype('float32');xx.append(x[z['indices'][:,None]-np.arange(167,-1,-1)])
    xx=np.concatenate(xx);sid=np.array([SITES.index(str(s)) for s in d['site']]);result=[]
    with torch.no_grad():
        for lo in range(0,len(xx),512):
            kw={'site_ids':torch.tensor(sid[lo:lo+512],device='cuda',dtype=torch.long)} if getattr(model,'needs_site',False) else {}
            result.append(model(torch.tensor(xx[lo:lo+512],device='cuda'),torch.tensor(base[lo:lo+512],device='cuda'),**kw)[0].cpu().numpy())
    del model;gc.collect();torch.cuda.empty_cache();return np.concatenate(result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['test','h1','q3'],required=True);ap.add_argument('--seed',type=int,default=17);args=ap.parse_args();role=args.role;seed=args.seed
    assert (R/'selection.json').exists(),'Freeze fusion selection before test inference.'
    torch.set_num_threads(2);out=R/'predictions'/role if seed==17 else R/'predictions'/f'seed{seed}'/role;out.mkdir(parents=True,exist_ok=True)
    d=load(R/'prepared'/f'{role}.npz');anchor={'target':d['y'],'site':d['site'],'origin':d['origin']}
    def save(name,p):save_predictions(out/f'{name}.npz',p,d);print('PREDICTED',role,name,flush=True)
    for name,folder in [('XGBoost',f'xgb_{seed}'),('CatBoost_RMSEcal',f'catpoint_{seed}')]:
        if (out/f'{name}.npz').exists():continue
        pred=np.zeros((len(d['y']),4,9),dtype='float32')
        if name=='XGBoost':
            import xgboost as xgb
            for j,h in enumerate(H):
                model=xgb.Booster();model.load_model(R/'experts'/folder/f'h{h}.ubj');model.set_param({'device':'cpu','nthread':2})
                pred[:,j]=model.inplace_predict(d['X'],iteration_range=(0,int(model.best_iteration)+1))+d['last'][:,None]
        else:
            import pandas as pd
            from catboost import CatBoostRegressor,Pool
            pp=d['X'];offsets=np.load(R/'experts'/folder/'offsets.npy')
            for j,h in enumerate(H):
                model=CatBoostRegressor();model.load_model(str(R/'experts'/folder/f'h{h}.cbm'));pred[:,j]=model.predict(pp,thread_count=2)[:,None]+d['last'][:,None]+offsets[j]
        save(name,pred)
    if not (out/'TabM.npz').exists():
        from train_tabm import make_model,predict
        folder=R/'experts'/f'tabm_{seed}';run=json.loads((folder/'run.json').read_text());rule=run['checkpoint_rule'];p=None;torch.backends.cuda.matmul.allow_tf32=True
        for key in (['global'] if rule=='global' else [f'h{h}' for h in H]):
            ck=torch.load(folder/f'{key}.pt',map_location='cpu',weights_only=False);model=make_model(ck['config'],ck['bins']).to('cuda');model.load_state_dict(ck['state']);cur=predict(model,d,ck['mean'],ck['sd'],ck['scale'],'cuda')
            if rule=='global':p=cur
            else:
                if p is None:p=np.zeros_like(cur)
                j=list(H).index(int(key[1:]));p[:,j]=cur[:,j]
            del model;gc.collect();torch.cuda.empty_cache()
        save('TabM',p);torch.backends.cuda.matmul.allow_tf32=False
    if seed!=17:
        for name in ['LightGBM','Neural3','Chronos_adapter','Chronos_native']:
            save(name,align(load(R/'predictions'/role/f'{name}.npz'),anchor)['prediction'])
        dump(out/'completed.json',{'timestamp_utc':now(),'role':role,'seed':seed,'n':len(d['y']),'selection_sha256':hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()});return
    if role in ['test','h1']:
        paths={'LightGBM':A/'strong_lightgbm/test.npz','Chronos_adapter':A/'runs/chronos_summary_flex_gated_stable_17/complete.npz'} if role=='test' else {'LightGBM':A/'confirmation_2026/predictions/LightGBM_extended.npz'}
        for name,path in paths.items():
            if not (out/f'{name}.npz').exists():save(name,align(load(path),anchor)['prediction'])
        if not (out/'Neural3.npz').exists():
            paths=[A/f'runs/empirical_summary_flex_only_region_stable_{s}/complete.npz' for s in [17,29,43]] if role=='test' else [A/f'confirmation_2026/predictions/selected_{s}.npz' for s in [17,29,43]]
            save('Neural3',np.mean([align(load(path),anchor)['prediction'] for path in paths],axis=0))
        if not (out/'Chronos_native.npz').exists():
            if role=='h1':z=load(A/'confirmation_2026/predictions/Chronos_native.npz')
            else:z={k:np.concatenate([load(D/'forecasts'/f'{s}_test_native.npz')[k] for s in SITES]) for k in ['prediction','target','site','origin']}
            save('Chronos_native',align(z,anchor)['prediction'])
    data=None;base=None
    if role=='q3':
        data=raw_data(role);refs=chronos_new(data,out)
        base=align(refs['base'],anchor)['prediction']
        if not (out/'Chronos_native.npz').exists():save('Chronos_native',align(refs['native'],anchor)['prediction'])
        if not (out/'LightGBM.npz').exists():
            import lightgbm as lgb
            p=np.zeros((len(d['y']),4,9),dtype='float32')
            for j,h in enumerate(H):
                for qi in range(9):p[:,j,qi]=lgb.Booster(model_file=str(A/'strong_lightgbm'/f'h{h}_q{qi+1}.txt')).predict(d['X'],num_threads=2)
            save('LightGBM',p)
        if not (out/'Neural3.npz').exists():
            ps=[neural_prediction(A/f'runs/empirical_summary_flex_only_region_stable_{s}/model.pt',data,base,d) for s in [17,29,43]];save('Neural3',np.mean(ps,axis=0))
    if not (out/'Chronos_adapter.npz').exists():
        if data is None:data=raw_data(role)
        if base is None:
            folder=A/'confirmation_2026/predictions';parts=[load(folder/f'{s}_base.npz') for s in SITES];z={k:np.concatenate([p[k] for p in parts]) for k in ['prediction','target','site','origin']};base=align(z,anchor)['prediction']
        save('Chronos_adapter',neural_prediction(A/'runs/chronos_summary_flex_gated_stable_17/model.pt',data,base,d))
    dump(out/'completed.json',{'timestamp_utc':now(),'role':role,'n':len(d['y']),'selection_sha256':hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()})

if __name__=='__main__':main()
