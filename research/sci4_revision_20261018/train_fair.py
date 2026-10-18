"""Fixed same-mask controls; no test access before each checkpoint is saved."""
import argparse,time,copy,random,gc,importlib.util
import torch
from fair_common import *
ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['lightgbm','tabm','adapter'],required=True);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--missing-indicators',action='store_true');args=ap.parse_args()
torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed);device='cuda' if torch.cuda.is_available() else 'cpu'
tr=load(R/'prepared/train.npz');va=load(R/'prepared/validation.npz');fit,select=year_masks(va)
tag='tabm_flags' if args.missing_indicators else args.model
out=R/'runs'/f'{tag}_{args.seed}';out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();logs=[]
if (out/'completed.json').exists():print('ALREADY COMPLETE',out,flush=True);raise SystemExit
if args.model=='lightgbm':
    import lightgbm as lgb
    pred=np.empty((len(va['y']),4,9),np.float32)
    for hi,h in enumerate(H):
        for qi,tau in enumerate(Q):
            dest=out/f'h{h}_q{qi+1}.txt'
            if dest.exists():b=lgb.Booster(model_file=str(dest));iteration=b.current_iteration()
            else:
                m=lgb.LGBMRegressor(objective='quantile',alpha=float(tau),n_estimators=600,num_leaves=31,learning_rate=.05,min_child_samples=50,colsample_bytree=1.,random_state=args.seed,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
                m.fit(tr['X'],tr['y'][:,hi],eval_set=[(va['X'][fit],va['y'][fit,hi])],callbacks=[lgb.early_stopping(40,verbose=False)])
                b=m.booster_;iteration=m.best_iteration_;b.save_model(str(dest))
            pred[:,hi,qi]=b.predict(va['X'],num_threads=2);logs.append({'horizon':h,'quantile':float(tau),'iteration':iteration})
            print('FAIR_LGBM',h,qi+1,iteration,'seconds',round(time.perf_counter()-start,1),flush=True)
    save(out/'validation.npz',pred,va)
elif args.model=='tabm':
    from tabm import TabM
    from rtdl_num_embeddings import compute_bins,PiecewiseLinearEmbeddings
    xn=np.delete(tr['X'],136,axis=1)
    missing_cols=np.flatnonzero(np.isnan(xn).any(0)) if args.missing_indicators else np.array([],dtype=int)
    if len(missing_cols):xn=np.concatenate([xn,np.isnan(xn[:,missing_cols]).astype('float32')],1)
    n_num=xn.shape[1];mean=np.nanmean(xn,axis=0);sd=np.maximum(np.nanstd(xn,axis=0),1e-5)
    xn=np.where(np.isfinite(xn),xn,mean);xn=((xn-mean)/sd).astype('float32')
    bins=compute_bins(torch.from_numpy(xn),n_bins=16);scale=float(np.std(tr['y']-tr['last'][:,None]));k=16
    model=TabM.make(n_num_features=n_num,cat_cardinalities=[6],d_out=36,num_embeddings=PiecewiseLinearEmbeddings(bins,d_embedding=8,activation=False,version='B'),n_blocks=2,d_block=256,dropout=.1,k=k).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
    X=torch.tensor(xn,device=device);C=torch.tensor(tr['X'][:,136:137].astype('int64'),device=device);Y=torch.tensor((tr['y']-tr['last'][:,None])/scale,device=device);tau=torch.tensor(Q,dtype=torch.float32,device=device)
    @torch.no_grad()
    def infer(d):
        num=np.delete(d['X'],136,axis=1)
        if len(missing_cols):num=np.concatenate([num,np.isnan(num[:,missing_cols]).astype('float32')],1)
        num=(np.where(np.isfinite(num),num,mean)-mean)/sd;p=[];model.eval()
        for a in range(0,len(num),512):
            ids=slice(a,a+512);v=model(torch.tensor(num[ids],dtype=torch.float32,device=device),torch.tensor(d['X'][ids,136:137].astype('int64'),device=device))
            p.append(v.mean(1).reshape(-1,4,9).cpu().numpy()*scale+d['last'][ids,None,None])
        return np.maximum(0,np.sort(np.concatenate(p),axis=-1))
    best=float('inf');stale=0;vf={key:value[fit] for key,value in va.items()}
    for epoch in range(1,61):
        model.train()
        for ids in torch.randperm(len(X),device=device).split(512):
            opt.zero_grad(set_to_none=True);p=model(X[ids],C[ids]).reshape(-1,k,4,9);e=Y[ids,None,:,None]-p;loss=torch.maximum(tau*e,(tau-1)*e).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        pv=infer(vf);value=float(pin(pv,vf['y']).mean());logs.append({'epoch':epoch,'validation2020':value})
        if value<best-1e-7:best=value;stale=0;state={key:v.detach().cpu().clone() for key,v in model.state_dict().items()};bestepoch=epoch
        else:stale+=1
        print('FAIR_TABM',args.seed,epoch,value,'seconds',round(time.perf_counter()-start,1),flush=True)
        if stale>=8:break
    torch.save({'state':state,'mean':mean,'sd':sd,'scale':scale,'bins':bins,'n_num_features':n_num,'missing_cols':missing_cols,'seed':args.seed,'epoch':bestepoch},out/'model.pt')
    model.load_state_dict(state);save(out/'validation.npz',infer(va),va)
else:
    spec=importlib.util.spec_from_file_location('review_adapter',V/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    norm=json.loads((R/'prepared/normalization.json').read_text());mean=np.array(norm['mean']);sd=np.array(norm['sd']);emp=np.array(norm['empirical'],np.float32)
    data={};del tr['X'];del va['X']
    for role,d in [('train',anchor('train')),('validation',anchor('validation'))]:
        history_path=R/'prepared'/f'{role}_history.f32'
        if not history_path.exists():
            x=np.memmap(history_path,mode='w+',shape=(len(d['y']),168,8),dtype=np.float32)
            for _,ix,history in histories(d):x[ix]=((history-mean)/sd).astype('float32')
            x.flush();del x
        data[role]={'x':np.memmap(history_path,mode='r',shape=(len(d['y']),168,8),dtype=np.float32),'y':torch.from_numpy(d['y']).to(device)}
    del d
    masks=load(R/'prepared/shared_training_mask.npz')['mask']
    model=mod.DistributionQueryAdapter(sd[0],mean[0],'empirical_reference',empirical_reference=emp).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    dummy=torch.zeros((512,4,9),device=device)
    @torch.no_grad()
    def infer():
        model.eval();p=[];d=data['validation']
        for a in range(0,len(d['y']),512):
            x=torch.tensor(np.asarray(d['x'][a:a+512]),device=device);p.append(model(x,dummy[:len(x)],torch.ones_like(x[...,4:],dtype=torch.bool))[0].cpu().numpy())
        return np.concatenate(p)
    p=infer();best=float(pin(p[fit],va['y'][fit]).mean());state=copy.deepcopy(model.state_dict());bestepoch=0;stale=0
    for epoch in range(1,31):
        model.train();d=data['train']
        for ids in torch.randperm(len(d['y'])).split(512):
            ix=ids.numpy();x=torch.tensor(d['x'][ix],device=device);mm=torch.tensor(masks[ix],device=device)
            opt.zero_grad(set_to_none=True);p=model(x,dummy[:len(ids)],mm)[0];loss=mod.pinball(p,d['y'][ids.to(device)]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        p=infer();value=float(pin(p[fit],va['y'][fit]).mean());logs.append({'epoch':epoch,'validation2020':value})
        if value<best:best=value;state={key:v.detach().clone() for key,v in model.state_dict().items()};bestepoch=epoch;stale=0
        else:stale+=1
        print('FAIR_ADAPTER',args.seed,epoch,value,'seconds',round(time.perf_counter()-start,1),flush=True)
        if stale>=5:break
    model.load_state_dict(state);torch.save({'state_dict':state,'mean':mean.tolist(),'sd':sd.tolist(),'variant':'empirical_reference','empirical_reference':emp.tolist(),'seed':args.seed,'epoch':bestepoch},out/'model.pt');save(out/'validation.npz',infer(),va)
p=load(out/'validation.npz')['prediction'];dump(out/'completed.json',{'completed_utc':now(),'model':tag,'seed':args.seed,'seconds':time.perf_counter()-start,'validation2020':float(pin(p[fit],va['y'][fit]).mean()),'validation2021_report_only':float(pin(p[select],va['y'][select]).mean()),'trace':logs,'test_evaluated':False})
print('TRAINING COMPLETE',args.model,args.seed,flush=True)
