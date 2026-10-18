"""One-pass fixed-checkpoint evaluation on reserved 2026 H1 observations."""
from pathlib import Path
import os,json,time,hashlib
import numpy as np
import torch,lightgbm as lgb
from model_factory import build_model
R=Path(__file__).parent;D=R.parent;O=R/'confirmation_2026';P=O/'predictions';P.mkdir(exist_ok=True)
selection=json.loads((R/'selection.json').read_text());baseline_selection=json.loads((R/'strong_lightgbm/selection.json').read_text())
for s in [17,29,43]:assert (R/'runs'/f"{selection['variant']}_{s}/model.pt").exists()
os.environ.setdefault('HF_HOME',str(D.parent/'experiments_20261016/hf_cache'));os.environ['HF_HUB_DISABLE_XET']='1'
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
SITES=json.loads((D/'protocol.json').read_text())['sites'];H=np.array([6,12,24,48]);Q=np.arange(1,10)/10
data={s:dict(np.load(O/f'{s}.npz')) for s in SITES}
frozen={'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'selection':selection,'baseline_selection':baseline_selection,'checkpoints':{f.name:hashlib.sha256((f/'model.pt').read_bytes()).hexdigest() for f in (R/'runs').glob('*') if (f/'model.pt').exists()},'instruction':'No retraining in response to this evaluation; report all seeds and negative results.'}
if not (O/'frozen_models.json').exists():(O/'frozen_models.json').write_text(json.dumps(frozen,indent=2))
if not all((P/f'{s}_{mode}.npz').exists() for s in SITES for mode in ['base','native']):
    from chronos import Chronos2Pipeline
    pipe=Chronos2Pipeline.from_pretrained('amazon/chronos-2',revision='29ec3766d36d6f73f0696f85560a422f50e8498c',local_files_only=True,device_map=DEVICE,torch_dtype=torch.float32)
    for site,z in data.items():
        raw=z['raw'];ids=z['indices']
        for mode in ['base','native']:
            path=P/f'{site}_{mode}.npz'
            if path.exists():continue
            pred=[];start=time.perf_counter()
            for a in range(0,len(ids),128):
                tasks=[]
                for idx in ids[a:a+128]:
                    hist=raw[idx-167:idx+1];task={'target':hist[:,0].copy()}
                    if mode=='native':task['past_covariates']={f'x{j}':hist[:,j].copy() for j in range(1,8)}
                    tasks.append(task)
                qs,_=pipe.predict_quantiles(tasks,prediction_length=48,quantile_levels=Q.tolist(),batch_size=128,context_length=168,cross_learning=False)
                pred.extend(q[0,H-1,:].numpy() for q in qs)
            np.savez_compressed(path,prediction=np.maximum(0,np.sort(np.stack(pred),axis=-1)),target=raw[ids[:,None]+H,0],site=np.repeat(site,len(ids)),origin=z['time'][ids])
            print('CONFIRM_REFERENCE',site,mode,round(time.perf_counter()-start,1),flush=True)
    del pipe;__import__('gc').collect();torch.cuda.empty_cache()
def combined(mode):
    parts=[dict(np.load(P/f'{s}_{mode}.npz')) for s in SITES]
    return {k:np.concatenate([a[k] for a in parts]) for k in ['prediction','target','site','origin']}
q0=combined('base');native=combined('native');np.savez_compressed(P/'Chronos_native.npz',**native)
checkpoints=[(f"selected_{s}",R/'runs'/f"{selection['variant']}_{s}/model.pt",None) for s in [17,29,43]]
checkpoints += [('original_empirical',D/'transformer_revision/runs/pooled_empirical_reference_17/model.pt','empirical_control'),('original_chronos',D/'transformer_revision/runs/pooled_full_17/model.pt','chronos_control')]
checkpoints += [(f'original_empirical_{s}',D/f'transformer_revision/runs/pooled_empirical_reference_{s}/model.pt','empirical_control') for s in [29,43]]
for label,path,override in checkpoints:
    out=P/f'{label}.npz'
    if out.exists():continue
    ck=torch.load(path,map_location=DEVICE,weights_only=False);mode=override or ck['variant']
    model=build_model(ck['sd'][0],ck['mean'][0],mode,ck['empirical_reference']).to(DEVICE);model.load_state_dict(ck['state_dict']);model.eval()
    xx=[]
    for s in SITES:
        z=data[s];x=((z['raw']-np.array(ck['mean']))/np.array(ck['sd'])).astype('float32');xx.append(x[z['indices'][:,None]-np.arange(167,-1,-1)])
    x=torch.as_tensor(np.concatenate(xx),device=DEVICE);q=torch.as_tensor(q0['prediction'],device=DEVICE);pred=[]
    with torch.no_grad():
        for a in range(0,len(x),512):
            kw={'site_ids':torch.as_tensor([SITES.index(str(s)) for s in q0['site'][a:a+512]],device=DEVICE,dtype=torch.long)} if getattr(model,'needs_site',False) else {}
            pred.append(model(x[a:a+512],q[a:a+512],**kw)[0].cpu().numpy())
    np.savez_compressed(out,prediction=np.concatenate(pred),target=q0['target'],site=q0['site'],origin=q0['origin']);print('CONFIRM_ADAPTER',label,flush=True)
pieces=[]
for si,s in enumerate(SITES):
    z=data[s];raw=z['raw'];ids=z['indices'];feat=[raw[ids-l] for l in [0,1,3,6,12,24,48,72,167]]
    for w in [6,24,72,168]:
        hist=np.stack([raw[ids-l] for l in range(w)],1);feat.extend([hist.mean(1),hist.std(1)])
    pieces.append(np.concatenate(feat+[np.full((len(ids),1),si)]+[raw[ids]-raw[ids-l] for l in [6,24,72]],1))
x=np.concatenate(pieces)
for label,folder,features in [('LightGBM_original',D/'tabular',137),('LightGBM_extended',R/'strong_lightgbm',161)]:
    out=P/f'{label}.npz'
    if out.exists():continue
    pred=np.zeros((len(x),4,9),dtype='float32')
    for hi,h in enumerate(H):
        for qi in range(9):pred[:,hi,qi]=lgb.Booster(model_file=str(folder/f'h{h}_q{qi+1}.txt')).predict(x[:,:features])
    np.savez_compressed(out,prediction=np.maximum(0,np.sort(pred,axis=-1)),target=q0['target'],site=q0['site'],origin=q0['origin']);print('CONFIRM_BASELINE',label,flush=True)
(O/'completed.json').write_text(json.dumps({'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'samples':len(q0['target']),'no_tuning':True},indent=2))
