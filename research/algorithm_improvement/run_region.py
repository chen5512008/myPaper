"""Second validation-only stage; original protocol and selection preserved."""
from pathlib import Path
import json,time,copy,hashlib,datetime,argparse
import numpy as np
import torch
from model_region import RegionAdapter as RevisedAdapter
from model import pinball
R=Path(__file__).parent;D=R.parent
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);DEVICE='cuda' if torch.cuda.is_available() else 'cpu';BATCH=1024
SITES=json.loads((D/'protocol.json').read_text())['sites']
MODES=['empirical_control_region_stable','empirical_summary_flex_only_region_stable','empirical_summary_flex_gated_region_stable']
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['select','repeat'],required=True);args=ap.parse_args()
protocol={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'trigger':'LightGBM has site identity but existing neural candidates do not; add explicit site conditioning to equalize this static information. Reserved 2026 not used for selection.','amendment':'Three additional validation configurations to isolate static-site information, including a no-attention control. Prior stage protocols and selections retained; no new neural test or 2026 scoring before final selection.','candidates':MODES,'epochs_max':40,'patience':8,'batch':BATCH,'optimizer':'AdamW','learning_rate':.0006,'weight_decay':.01,'schedule':'cosine, minimum 0.00006','ema_decay_per_update':.995,'selection':'Minimum 2020-2021 validation pinball across raw and EMA epoch checkpoints; select among both stages before test evaluation.','training_mask':'Unchanged:50% complete,25% random30,25% terminal3/6/12/24/168','model_v1_sha256':hashlib.sha256((R/'model.py').read_bytes()).hexdigest(),'model_v2_sha256':hashlib.sha256((R/'model_v2.py').read_bytes()).hexdigest(),'model_region_sha256':hashlib.sha256((R/'model_region.py').read_bytes()).hexdigest(),'seed_selection':17,'confirmation_policy':'Evaluate reserved 2026 only if selected validation loss beats the strongest validation-selected LightGBM. Otherwise preserve the reserved period and report validation failure.'}
if not (R/'region_protocol.json').exists():
    (R/'region_protocol.json').write_text(json.dumps(protocol,indent=2))
    (R/'selection_stage2.json').write_bytes((R/'selection.json').read_bytes())
raws={s:dict(np.load(D/'prepared'/f'{s}.npz')) for s in SITES}
tr=np.concatenate([raws[s]['raw'][raws[s]['time']<'2020'] for s in SITES]).astype('float64');mean=tr.mean(0);sd=np.maximum(tr.std(0),1e-8);del tr
def dataset(role):
    parts=[]
    for s in SITES:
        z=raws[s];c=dict(np.load(D/'forecasts'/f'{s}_{role}_base.npz'));ids=c['indices'];x=((z['raw']-mean)/sd).astype('float32')
        parts.append({'x':x[ids[:,None]-np.arange(167,-1,-1)],'q':c['prediction'],'y':c['target'],'site':c['site'],'origin':c['origin']})
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for k in ['x','q','y']:d[k]=torch.as_tensor(d[k],device=DEVICE,dtype=torch.float32)
    d['sid']=torch.as_tensor([SITES.index(str(s)) for s in d['site']],device=DEVICE,dtype=torch.long)
    return d
def mask_for(x,scenario):
    m=torch.ones_like(x[...,4:])
    if scenario=='random30':m=(torch.rand_like(m)>.3).float()
    elif scenario.startswith('last'):m[:,-int(scenario[4:]):]=0
    return m
@torch.no_grad()
def infer(model,d,collect=False):
    model.eval();total=0.;n=0;pred=[]
    for a in range(0,len(d['y']),BATCH):
        x=d['x'][a:a+BATCH];p=model(x,d['q'][a:a+BATCH],site_ids=d['sid'][a:a+BATCH])[0];assert torch.isfinite(p).all()
        total+=float(pinball(p,d['y'][a:a+BATCH]))*len(x);n+=len(x)
        if collect:pred.append(p.cpu().numpy())
    return total/n,np.concatenate(pred) if collect else None
data={role:dataset(role) for role in ['train','validation']}
residual=(data['train']['y']-(data['train']['x'][:,-1,0]*sd[0]+mean[0])[:,None]).cpu().numpy();empirical=np.quantile(residual,np.arange(1,10)/10,axis=0).T.astype('float32')
choices=[(m,17) for m in MODES] if args.stage=='select' else [(json.loads((R/'selection.json').read_text())['variant'],s) for s in [29,43]]
for mode,seed in choices:
    out=R/'runs'/f'{mode}_{seed}';out.mkdir(exist_ok=True)
    if (out/'run.json').exists():continue
    torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    model=RevisedAdapter(sd[0],mean[0],mode,empirical).to(DEVICE);ema=copy.deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=.00006)
    best,_=infer(model,data['validation']);state=copy.deepcopy(model.state_dict());trace=[];stale=0;best_kind='initial';start=time.perf_counter()
    print('START_REGION',mode,seed,flush=True)
    for epoch in range(1,41):
        model.train();d=data['train'];total=0.;n=0
        for ids in torch.randperm(len(d['y']),device=DEVICE).split(BATCH):
            x=d['x'][ids];kind=int(rng.integers(0,4));scenario='complete' if kind<2 else ('random30' if kind==2 else 'last'+str(int(rng.choice([3,6,12,24,168]))))
            opt.zero_grad(set_to_none=True);loss=pinball(model(x,d['q'][ids],mask_for(x,scenario),d['sid'][ids])[0],d['y'][ids]);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():
                for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,.005)
            total+=float(loss.detach())*len(ids);n+=len(ids)
        val,_=infer(model,data['validation']);smooth,_=infer(ema,data['validation']);value=min(val,smooth);kind='ema' if smooth<val else 'raw'
        trace.append({'epoch':epoch,'train_pinball':total/n,'raw_validation_pinball':val,'ema_validation_pinball':smooth,'validation_pinball':value,'learning_rate':opt.param_groups[0]['lr']});scheduler.step()
        print(mode,seed,epoch,'raw',round(val,6),'ema',round(smooth,6),'seconds',round(time.perf_counter()-start,1),flush=True)
        if value<best:best=value;state=copy.deepcopy((ema if kind=='ema' else model).state_dict());best_kind=kind;stale=0
        else:stale+=1
        if stale>=8:break
    model.load_state_dict(state);_,p=infer(model,data['validation'],True);d=data['validation']
    torch.save({'state_dict':state,'mean':mean.tolist(),'sd':sd.tolist(),'seed':seed,'variant':mode,'empirical_reference':empirical.tolist(),'selected_weights':best_kind},out/'model.pt')
    np.savez_compressed(out/'validation.npz',prediction=p,target=d['y'].cpu().numpy(),site=d['site'],origin=d['origin'])
    (out/'run.json').write_text(json.dumps({**protocol,'variant':mode,'seed':seed,'trace':trace,'best_validation':best,'selected_weights':best_kind,'seconds':time.perf_counter()-start,'parameters':sum(a.numel() for a in model.parameters()),'checkpoint_sha256':hashlib.sha256((out/'model.pt').read_bytes()).hexdigest()},indent=2));print('SAVED_REGION',mode,seed,best,flush=True)
if args.stage=='select':
    runs=[json.loads(f.read_text()) for f in (R/'runs').glob('*_17/run.json')];runs.sort(key=lambda a:a['best_validation'])
    baseline=json.loads((R/'strong_lightgbm/selection.json').read_text());best_baseline=min(baseline['original_validation'],baseline['extended_validation'])
    selection={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':runs[0]['variant'],'ranking':[{'variant':a['variant'],'validation_pinball':a['best_validation']} for a in runs],'rule':'Best validation among all disclosed stages including static-site controls; no test access for selection','test_evaluated':False,'beats_strong_baseline_validation':runs[0]['best_validation']<best_baseline,'baseline_validation':best_baseline}
    (R/'selection.json').write_text(json.dumps(selection,indent=2));print('FINAL_VALIDATION_SELECTION',json.dumps(selection),flush=True)
