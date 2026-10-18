from pathlib import Path
import json,time,copy,gc,argparse,hashlib,datetime
import numpy as np
import torch
from model import ImprovedAdapter,pinball
from model_factory import build_model
ROOT=Path(__file__).resolve().parent;DATA=ROOT.parent
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu';BATCH=512
SITES=json.loads((DATA/'protocol.json').read_text())['sites']
CANDIDATES=['chronos_control','empirical_control','chronos_flex','chronos_summary','chronos_summary_flex','empirical_summary_flex','empirical_summary_flex_only']
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['select','repeat','evaluate'],required=True);args=ap.parse_args()
protocol={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'Exploratory on previously inspected 2022-2025 benchmark. Test never used by this script for model selection.','candidates':CANDIDATES,'selection':'Smallest complete 2020-2021 validation pinball among seed17 candidate runs; freeze choice before any new test inference. Repeat selected model at seeds29,43; report all seeds.','train':'2000-2019','validation':'2020-2021','test':'2022-2025','epochs_max':40,'patience':8,'learning_rate':.001,'batch':512,'mask_training':'50% complete,25% random30,25% terminal3/6/12/24/168','gap_floor_standardized':.02,'budget':'7 configurations seed17 + selected two repetitions, no test-driven retuning','comparison':'Keep existing LightGBM, native, LCA, and empirical controls. No deletion of unfavorable runs. Primary pinball; high-wave MAE, coverage, width, and interval score secondary.','model_sha256':hashlib.sha256((ROOT/'model.py').read_bytes()).hexdigest()}
if not (ROOT/'protocol.json').exists():(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
else:
    prior=json.loads((ROOT/'protocol.json').read_text());assert prior['model_sha256']==protocol['model_sha256']
raws={s:dict(np.load(DATA/'prepared'/f'{s}.npz')) for s in SITES}
tr=np.concatenate([raws[s]['raw'][raws[s]['time']<'2020'] for s in SITES]).astype('float64')
mean=tr.mean(0);sd=np.maximum(tr.std(0),1e-8);del tr
def dataset(role):
    pieces=[]
    for s in SITES:
        z=raws[s];c=dict(np.load(DATA/'forecasts'/f'{s}_{role}_base.npz'));ids=c['indices'];x=((z['raw']-mean)/sd).astype('float32')
        pieces.append({'x':x[ids[:,None]-np.arange(167,-1,-1)],'q':c['prediction'],'y':c['target'],'site':c['site'],'origin':c['origin']})
    d={k:np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
    for k in ['x','q','y']:d[k]=torch.from_numpy(d[k]).float().to(DEVICE)
    d['sid']=torch.as_tensor([SITES.index(str(s)) for s in d['site']],device=DEVICE,dtype=torch.long)
    return d
def mask_for(x,scenario,generator=None):
    m=torch.ones_like(x[...,4:])
    if scenario=='random30':m=(torch.rand(m.shape,device=DEVICE,generator=generator)>.3).float()
    elif scenario.startswith('last'):m[:,-int(scenario[4:]):]=0
    elif scenario=='all_missing':m.zero_()
    return m
@torch.no_grad()
def infer(model,d,scenario='complete',collect=False):
    model.eval();ps=[];total=0.;n=0;gen=torch.Generator(device=DEVICE).manual_seed(777)
    for a in range(0,len(d['y']),BATCH):
        x=d['x'][a:a+BATCH];kw={'site_ids':d['sid'][a:a+BATCH]} if getattr(model,'needs_site',False) else {}
        p=model(x,d['q'][a:a+BATCH],mask_for(x,scenario,gen),**kw)[0]
        assert torch.isfinite(p).all()
        total+=float(pinball(p,d['y'][a:a+BATCH]))*len(x);n+=len(x)
        if collect:ps.append(p.cpu().numpy())
    return total/n,np.concatenate(ps) if collect else None
if args.stage in ['select','repeat']:
    data={role:dataset(role) for role in ['train','validation']}
    residual=(data['train']['y']-(data['train']['x'][:,-1,0]*sd[0]+mean[0])[:,None]).cpu().numpy()
    empirical=np.quantile(residual,np.arange(1,10)/10,axis=0).T.astype('float32')
    choices=[(m,17) for m in CANDIDATES] if args.stage=='select' else [(json.loads((ROOT/'selection.json').read_text())['variant'],s) for s in [29,43]]
    for mode,seed in choices:
        out=ROOT/'runs'/f'{mode}_{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'run.json').exists():continue
        torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
        model=ImprovedAdapter(sd[0],mean[0],mode,empirical).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        best,_=infer(model,data['validation']);state=copy.deepcopy(model.state_dict());trace=[];stale=0;start=time.perf_counter()
        print('START',mode,seed,'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(1,41):
            model.train();d=data['train'];loss_sum=0.;n=0
            for ids in torch.randperm(len(d['y']),device=DEVICE).split(BATCH):
                x=d['x'][ids];kind=int(rng.integers(0,4));scenario='complete' if kind<2 else ('random30' if kind==2 else 'last'+str(int(rng.choice([3,6,12,24,168]))))
                opt.zero_grad(set_to_none=True);loss=pinball(model(x,d['q'][ids],mask_for(x,scenario))[0],d['y'][ids]);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_sum+=float(loss.detach())*len(ids);n+=len(ids)
            value,_=infer(model,data['validation']);trace.append({'epoch':epoch,'train_pinball':loss_sum/n,'validation_pinball':value})
            print(mode,seed,epoch,round(value,6),'seconds',round(time.perf_counter()-start,1),flush=True)
            if value<best:best=value;state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
            if stale>=8:break
        model.load_state_dict(state)
        torch.save({'state_dict':state,'mean':mean.tolist(),'sd':sd.tolist(),'seed':seed,'variant':mode,'empirical_reference':empirical.tolist()},out/'model.pt')
        _,p=infer(model,data['validation'],collect=True);d=data['validation']
        np.savez_compressed(out/'validation.npz',prediction=p,target=d['y'].cpu().numpy(),site=d['site'],origin=d['origin'])
        report={**protocol,'seed':seed,'variant':mode,'trace':trace,'best_validation':best,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start,'checkpoint_sha256':hashlib.sha256((out/'model.pt').read_bytes()).hexdigest()}
        (out/'run.json').write_text(json.dumps(report,indent=2));print('SAVED',mode,seed,'best',best,flush=True)
    if args.stage=='select':
        ranked=sorted([json.loads((ROOT/'runs'/f'{m}_17/run.json').read_text()) for m in CANDIDATES],key=lambda a:a['best_validation'])
        selection={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':ranked[0]['variant'],'ranking':[{'variant':a['variant'],'validation_pinball':a['best_validation']} for a in ranked],'rule':protocol['selection'],'test_evaluated':False}
        (ROOT/'selection.json').write_text(json.dumps(selection,indent=2));print('SELECTION',json.dumps(selection),flush=True)
else:
    selection=json.loads((ROOT/'selection.json').read_text())
    for s in [17,29,43]:assert (ROOT/'runs'/f"{selection['variant']}_{s}/run.json").exists()
    data=dataset('test')
    for out in sorted((ROOT/'runs').glob('*')):
        if not (out/'run.json').exists():continue
        if (out/'test_scores.json').exists():continue
        ck=torch.load(out/'model.pt',map_location=DEVICE,weights_only=False)
        model=build_model(sd[0],mean[0],ck['variant'],ck['empirical_reference']).to(DEVICE);model.load_state_dict(ck['state_dict'])
        scores={}
        for scenario in ['complete','last24','random30','all_missing']:
            score,p=infer(model,data,scenario,True);scores[scenario]=score
            np.savez_compressed(out/f'{scenario}.npz',prediction=p,target=data['y'].cpu().numpy(),origin=data['origin'],site=data['site'])
        (out/'test_scores.json').write_text(json.dumps(scores,indent=2));print(out.name,scores,flush=True)
