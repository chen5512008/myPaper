"""TabM with memberwise quantile loss; stopping uses 2020 targets only."""
import argparse,copy,time,random
import torch
from tabm import TabM
from rtdl_num_embeddings import compute_bins,PiecewiseLinearEmbeddings
from common import *
from fusion import year_masks

def make_model(config,bins):
    return TabM.make(n_num_features=160,cat_cardinalities=[6],d_out=36,
        num_embeddings=PiecewiseLinearEmbeddings(bins,d_embedding=8,activation=False,version='B'),
        n_blocks=2,d_block=config['d_block'],dropout=.1,k=config['k'])

def predict(model,d,mean,sd,scale,device):
    x=(np.delete(d['X'],136,axis=1)-mean)/sd
    cat=d['X'][:,136:137].astype('int64');result=[];model.eval()
    with torch.no_grad():
        for lo in range(0,len(x),512):
            v=model(torch.tensor(x[lo:lo+512],device=device),torch.tensor(cat[lo:lo+512],device=device))
            result.append(v.mean(1).reshape(-1,4,9).cpu().numpy()*scale+d['last'][lo:lo+512,None,None])
    return np.maximum(0,np.sort(np.concatenate(result),axis=-1)).astype('float32')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=17);ap.add_argument('--checkpoint-rule',choices=['global','horizon']);args=ap.parse_args()
    out=R/'experts'/f'tabm_{args.seed}';out.mkdir(parents=True,exist_ok=True)
    cfg={'seed':args.seed,'d_block':256,'k':16,'epochs':60,'batch':512,'lr':.001,'weight_decay':.0003,'patience':8,'bins':16,'embedding':8,'stopping':'2020 only; global and per-horizon checkpoints','training_loss':'mean of individual member quantile losses; prediction averages member outputs'}
    dump(out/'config.json',{'timestamp_utc':now(),**cfg})
    torch.set_num_threads(2);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    tr=load(R/'prepared/train.npz');va=load(R/'prepared/validation.npz');fit,sel=year_masks(va)
    xn=np.delete(tr['X'],136,axis=1);mean=xn.mean(0);sd=np.maximum(xn.std(0),1e-5)
    xn=(xn-mean)/sd;bins=compute_bins(torch.from_numpy(xn),n_bins=16)
    scale=float(np.std(tr['y']-tr['last'][:,None]));device='cuda'
    model=make_model(cfg,bins).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    X=torch.tensor(xn,device=device);C=torch.tensor(tr['X'][:,136:137].astype('int64'),device=device)
    Y=torch.tensor((tr['y']-tr['last'][:,None])/scale,device=device);quant=torch.tensor(Q,dtype=torch.float32,device=device)
    vf={k:v[fit] for k,v in va.items()};best=float('inf');best_h=np.full(4,np.inf);epoch_h=np.zeros(4,int);stale=0;history=[];t0=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        model.train();perm=torch.randperm(len(X),device=device);running=0.
        for ix in perm.split(cfg['batch']):
            opt.zero_grad(set_to_none=True);p=model(X[ix],C[ix]).reshape(-1,cfg['k'],4,9)
            e=Y[ix,None,:,None]-p;loss=torch.maximum(quant*e,(quant-1)*e).mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();running+=float(loss.detach())*len(ix)
        pv=predict(model,vf,mean,sd,scale,device);hv=pin(pv,vf['y']).mean((0,2));v=float(hv.mean())
        payload={'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'config':cfg,'bins':bins,'mean':mean,'sd':sd,'scale':scale,'epoch':epoch}
        if v<best-1e-7:
            best=v;stale=0;torch.save(payload,out/'global.pt')
        else:stale+=1
        for j in range(4):
            if hv[j]<best_h[j]-1e-7:
                best_h[j]=hv[j];epoch_h[j]=epoch;torch.save(payload,out/f'h{H[j]}.pt')
        history.append({'epoch':epoch,'train_loss_scaled':running/len(X),'validation2020':v,'by_horizon':hv.tolist()})
        dump(out/'history.json',history)
        print('TABM',args.seed,epoch,'val2020',round(v,7),'best',round(best,7),'seconds',round(time.perf_counter()-t0,1),flush=True)
        if stale>=cfg['patience']:break
    pglobal=None;ph=np.zeros((len(va['y']),4,9),dtype='float32')
    for key in ['global']+[f'h{h}' for h in H]:
        ck=torch.load(out/f'{key}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
        p=predict(model,va,mean,sd,scale,device)
        if key=='global':pglobal=p
        else:j=list(H).index(int(key[1:]));ph[:,j]=p[:,j]
    save_predictions(out/'validation_global.npz',pglobal,va);save_predictions(out/'validation_horizon.npz',ph,va)
    scores={'global':score(pglobal[sel],va['y'][sel]),'horizon':score(ph[sel],va['y'][sel])};choice=args.checkpoint_rule or min(scores,key=scores.get)
    save_predictions(out/'validation.npz',{'global':pglobal,'horizon':ph}[choice],va)
    dump(out/'run.json',{'timestamp_utc':now(),'model':'TabM','config':cfg,'version':'0.0.3','parameters':sum(p.numel() for p in model.parameters()),'n_train':len(X),'n_early_stopping':int(fit.sum()),'2021_scores':scores,'checkpoint_rule':choice,'best2020_per_horizon_epoch':epoch_h.tolist(),'seconds':time.perf_counter()-t0,'history':history})
    print('DONE TABM',scores,choice,flush=True)

if __name__=='__main__':main()
