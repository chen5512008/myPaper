from pathlib import Path
import sys,json,time,copy,gc,argparse,hashlib
import numpy as np
import torch
from model import DistributionQueryAdapter,pinball

ROOT=Path(__file__).resolve().parent;DATA=ROOT.parent
sys.path.insert(0,str(DATA/'models'))
from adapter import WeatherAdapter
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu';BATCH=512
SITES=json.loads((DATA/'protocol.json').read_text())['sites']
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pooled','holdout'],default='pooled');args=parser.parse_args()
raws={s:dict(np.load(DATA/'prepared'/f'{s}.npz')) for s in SITES}

class Legacy(torch.nn.Module):
    def __init__(self,scale):super().__init__();self.net=WeatherAdapter(target_scale=scale)
    def forward(self,x,q,mask):return self.net(torch.cat([x[...,:4],x[...,4:]*mask],-1),q)

def dataset(sites,role,mean,sd):
    pieces=[]
    for s in sites:
        z=raws[s];c=dict(np.load(DATA/'forecasts'/f'{s}_{role}_base.npz'));ids=c['indices']
        x=((z['raw']-mean)/sd).astype('float32')
        pieces.append({'x':x[ids[:,None]-np.arange(167,-1,-1)],'q':c['prediction'],'y':c['target'],'site':c['site'],'origin':c['origin']})
    d={k:np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
    for k in ['x','q','y']:d[k]=torch.from_numpy(d[k]).float().to(DEVICE)
    return d

def make_mask(x,scenario,generator=None):
    m=torch.ones_like(x[...,4:])
    if scenario=='random30':m=(torch.rand(m.shape,device=DEVICE,generator=generator)>.3).float()
    elif scenario.startswith('last'):m[:,-int(scenario[4:]):]=0
    elif scenario=='all_missing':m.zero_()
    return m

@torch.no_grad()
def infer(model,d,scenario='complete',collect=True):
    model.eval();ps=[];total=0.;n=0;gen=torch.Generator(device=DEVICE).manual_seed(777)
    for a in range(0,len(d['y']),BATCH):
        x=d['x'][a:a+BATCH];m=make_mask(x,scenario,gen)
        p=model(x,d['q'][a:a+BATCH],m)[0]
        if not torch.isfinite(p).all():raise RuntimeError('Nonfinite prediction')
        total+=float(pinball(p,d['y'][a:a+BATCH]))*len(x);n+=len(x)
        if collect:ps.append(p.cpu().numpy())
    return total/n,np.concatenate(ps) if collect else None

groups=[('pooled',SITES,SITES)] if args.stage=='pooled' else [(f'holdout_{s}',[a for a in SITES if a!=s],[s]) for s in SITES]
for group,sources,targets in groups:
    choices=([('full',17),('affine',17),('no_availability',17),('no_distribution_query',17),('empirical_reference',17),('legacy_matched',17),('full',29),('full',43)] if group=='pooled' else [('full',17)])
    if all((ROOT/'runs'/f'{group}_{mode}_{seed}/run.json').exists() for mode,seed in choices):continue
    tr=np.concatenate([raws[s]['raw'][raws[s]['time']<'2020'] for s in sources]).astype('float64')
    mean=tr.mean(0);sd=np.maximum(tr.std(0),1e-8);del tr
    data={role:dataset(targets if role=='test' else sources,role,mean,sd) for role in ['train','validation','test']}
    residual=(data['train']['y']-(data['train']['x'][:,-1,0]*sd[0]+mean[0])[:,None]).cpu().numpy()
    empirical=np.quantile(residual,np.arange(1,10)/10,axis=0).T.astype('float32')
    for mode,seed in choices:
        out=ROOT/'runs'/f'{group}_{mode}_{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'run.json').exists():continue
        torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
        model=(Legacy(float(sd[0])) if mode=='legacy_matched' else DistributionQueryAdapter(sd[0],mean[0],mode,empirical_reference=empirical)).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        best,_=infer(model,data['validation'],collect=False);state=copy.deepcopy(model.state_dict());trace=[];stale=0;start=time.perf_counter()
        print('START',group,mode,seed,'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(1,31):
            model.train();d=data['train'];loss_sum=0.;n=0
            for ids in torch.randperm(len(d['y']),device=DEVICE).split(BATCH):
                x=d['x'][ids];kind=int(rng.integers(0,4))
                scenario='complete' if kind<2 else ('random30' if kind==2 else 'last'+str(int(rng.choice([3,6,12,24,168]))))
                m=make_mask(x,scenario)
                opt.zero_grad(set_to_none=True);loss=pinball(model(x,d['q'][ids],m)[0],d['y'][ids])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                loss_sum+=float(loss.detach())*len(ids);n+=len(ids)
            value,_=infer(model,data['validation'],collect=False)
            trace.append({'epoch':epoch,'train_pinball':loss_sum/n,'validation_pinball':value})
            print(group,mode,seed,epoch,round(value,6),'seconds',round(time.perf_counter()-start,1),flush=True)
            if value<best:best=value;state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
            if stale>=5:break
        model.load_state_dict(state)
        torch.save({'state_dict':state,'mean':mean.tolist(),'sd':sd.tolist(),'sources':sources,'seed':seed,'variant':mode,'empirical_reference':empirical.tolist()},out/'model.pt')
        scores={}
        for scenario in ['complete','last24','random30','all_missing']:
            score,p=infer(model,data['test'],scenario);scores[scenario]=score
            d=data['test'];np.savez_compressed(out/f'{scenario}.npz',prediction=p,target=d['y'].cpu().numpy(),origin=d['origin'],site=d['site'])
        report={'sources':sources,'targets':targets,'seed':seed,'variant':mode,'mean':mean.tolist(),'sd':sd.tolist(),'trace':trace,'best_validation':best,'test_scores':scores,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start,'selection':'complete source validation only','epochs_max':30,'patience':5,'learning_rate':.001,'batch':BATCH,'gradient_clip':1.,'model_sha256':hashlib.sha256((ROOT/'model.py').read_bytes()).hexdigest()}
        (out/'run.json').write_text(json.dumps(report,indent=2))
        print('SAVED',group,mode,seed,round(time.perf_counter()-start,1),flush=True)
    del data,model,opt,state;gc.collect();torch.cuda.empty_cache()
