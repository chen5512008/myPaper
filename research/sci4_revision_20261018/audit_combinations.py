"""Equal-recipe state gates refitted exclusively on 2020/2021 validation data."""
import copy,time,torch
import pandas as pd
from fair_common import *
from bank import validation_bank
from fusion import fit_weights,pool
from gate import Gate,gate_features,apply_gate

torch.set_num_threads(2)
out=R/'combinations';out.mkdir(exist_ok=True)
d,bank=validation_bank();names=list(bank);fit,sel=year_masks(d)
configs=[('TabM_LightGBM',['TabM','LightGBM']),('TabM_LightGBM_ChronosAdapter',['TabM','LightGBM','Chronos_adapter']),('TabM_LightGBM_Neural3',['TabM','LightGBM','Neural3']),('TabM_LightGBM_BothAdapters',['TabM','LightGBM','Neural3','Chronos_adapter']),('All',names),('All_without_adapters',[n for n in names if n not in ['Neural3','Chronos_adapter']])]+[('All_without_'+n,[k for k in names if k!=n]) for n in names]
protocol={'created_utc':now(),'status':'Exploratory follow-up after all test periods were seen; no new confirmatory claim','fit':'2020 only','early_stopping':'2021 only','penalty':.001,'prior':'horizon convex weights fitted on 2020, squared uniform penalty .001','seed':17,'gate':'Same Gate architecture, features, learning rate, batch, patience and logit bound as original selected gate','subsets':configs,'test_use':'All prespecified subsets reported. No test-based subset selection. All old frozen horizon-pool deletion diagnostics retained separately.'}
if not (out/'protocol.json').exists():dump(out/'protocol.json',protocol)
objects=[]
for label,members in configs:
    dest=out/label;dest.mkdir(exist_ok=True);path=dest/'model.pt'
    p=np.stack([bank[n] for n in members],-1)
    if not path.exists():
        prior=np.stack([fit_weights(p[fit,h],d['y'][fit,h],penalty=.001)[0] for h in range(4)])
        x=gate_features(p,d);mean=x[fit].mean((0,1));sd=np.maximum(x[fit].std((0,1)),1e-5);x=(x-mean)/sd
        torch.manual_seed(17);m=Gate(x.shape[-1],p.shape[-1],prior);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
        data={role:tuple(torch.tensor(a[ix],dtype=torch.float32) for a in [x,p,d['y']]) for role,ix in [('fit',fit),('sel',sel)]};tau=torch.tensor(Q,dtype=torch.float32);best=float('inf');stale=0;trace=[]
        for epoch in range(1,101):
            m.train();xx,pp,yy=data['fit']
            for ids in torch.randperm(len(xx)).split(512):
                opt.zero_grad();w=m(xx[ids]);pred=(pp[ids]*w[:,:,None]).sum(-1);e=yy[ids,:,None]-pred
                loss=torch.maximum(tau*e,(tau-1)*e).mean()+.001*(w*(w.clamp_min(1e-8).log()-m.prior.log()[None])).sum(-1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
            m.eval()
            with torch.no_grad():
                xx,pp,yy=data['sel'];e=yy[:,:,None]-(pp*m(xx)[:,:,None]).sum(-1);value=float(torch.maximum(tau*e,(tau-1)*e).mean())
            trace.append({'epoch':epoch,'validation2021':value})
            if value<best-1e-7:best=value;stale=0;state=copy.deepcopy(m.state_dict());best_epoch=epoch
            else:stale+=1
            if stale>=12:break
        torch.save({'state':state,'mean':mean,'sd':sd,'n_features':x.shape[-1],'n_experts':p.shape[-1],'prior':prior,'penalty':.001,'seed':17},path)
        dump(dest/'fit.json',{'members':members,'best_validation2021':best,'best_epoch':best_epoch,'trace':trace})
        print('COMBINATION FIT',label,best,best_epoch,flush=True)
    objects.append((label,members,path))
# Freeze all checkpoints before any test-bank access in this script.
dump(out/'fit_completed.json',{'timestamp_utc':now(),'checkpoints':{label:hashlib.sha256(path.read_bytes()).hexdigest() for label,_,path in objects}})
rows=[];cis=[]
for role in ['test','h1','q3']:
    z=anchor(role);ppbank={n:align(load(F/'predictions'/role/f'{n}.npz'),z)['prediction'] for n in names}
    thresholds={s:json.loads((D/'prepared'/f'{s}.json').read_text())['training_high_wave_threshold_m'] for s in SITES}
    high=z['y']>=np.array([thresholds[str(s)] for s in z['site']])[:,None]
    preds={'TabM':ppbank['TabM']}
    for label,members,path in objects:
        p=np.stack([ppbank[n] for n in members],-1);chunks=[]
        for a in range(0,len(z['y']),1024):
            dz={key:value[a:a+1024] for key,value in z.items()};forecast,_=apply_gate(p[a:a+1024],dz,path);chunks.append(forecast)
        forecast=np.concatenate(chunks);preds[label]=forecast
        save(out/f'{role}__{label}.npz',forecast,z)
    pairs=[(n,'TabM') for n in preds if n!='TabM']+[(n,'TabM_LightGBM') for n in ['TabM_LightGBM_ChronosAdapter','TabM_LightGBM_Neural3','TabM_LightGBM_BothAdapters']]+[('All','All_without_adapters')]+[('All','All_without_'+n) for n in names]
    for label,p in preds.items():rows.append({'period':role,'model':label,**metrics(p,z['y'],high)})
    for a,b in pairs:
        diff=(pin(preds[a],z['y'])-pin(preds[b],z['y'])).mean((1,2))
        for days in [7,28]:cis.append({'period':role,'model':a,'reference':b,**block_ci(diff,z['origin'],days)})
    print('COMBINATION SCORES',role,{a:round(float(pin(p,z['y']).mean()),8) for a,p in preds.items()},flush=True)
pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
pd.DataFrame(cis).to_csv(out/'paired_intervals.csv',index=False)
