"""Small causal state gate, regularized toward a convex horizon pool."""
import torch
from common import *
from fusion import year_masks

def gate_features(p,d):
    # Every feature is available at the forecast origin. No future weather.
    n,h,q,m=p.shape
    dynamic=np.concatenate([p[:,:,4],p[:,:,-1]-p[:,:,0],p[:,:,4].std(-1,keepdims=True)],-1)
    history=np.repeat(d['X'][:,None,:],4,axis=1)
    site=np.eye(6,dtype='float32')[[SITES.index(str(s)) for s in d['site']]]
    return np.concatenate([history,dynamic,np.repeat(site[:,None],4,axis=1),np.repeat(np.eye(4,dtype='float32')[None],n,axis=0)],-1).astype('float32')

class Gate(torch.nn.Module):
    def __init__(self,n,m,prior):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(n,32),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(32,m))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('prior',torch.tensor(np.maximum(prior,.005)/np.maximum(prior,.005).sum(-1,keepdims=True),dtype=torch.float32))
    def forward(self,x):return torch.softmax(self.prior.log()[None]+2*torch.tanh(self.net(x)),-1)

def fit_gates(p,d,prior,out):
    torch.set_num_threads(2);fit,select=year_masks(d);x=gate_features(p,d)
    mean=x[fit].mean((0,1));sd=np.maximum(x[fit].std((0,1)),1e-5);x=(x-mean)/sd
    data={role:tuple(torch.from_numpy(a[ix].astype('float32')) for a in [x,p,d['y']]) for role,ix in [('fit',fit),('select',select)]}
    tau=torch.tensor(Q,dtype=torch.float32);rows=[];objects=[]
    config={'timestamp_utc':now(),'penalties':[.001,.01],'epochs':100,'patience':12,'hidden':32,'seed':17,'lr':.001,'batch':512,'regularization':'KL to horizon convex prior','gate_logit_bound':2,'training':'2020','stopping':'2021','input':'causal161 features, expert median/width/disagreement, site and horizon indicators'}
    dump(out/'gate_protocol.json',config)
    for lam in config['penalties']:
        torch.manual_seed(17);model=Gate(x.shape[-1],p.shape[-1],prior);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        best=float('inf');stale=0;trace=[]
        for epoch in range(1,101):
            model.train();xx,pp,yy=data['fit']
            for ids in torch.randperm(len(xx)).split(512):
                opt.zero_grad();w=model(xx[ids]);pred=(pp[ids]*w[:,:,None]).sum(-1);e=yy[ids,:,None]-pred
                loss=torch.maximum(tau*e,(tau-1)*e).mean()+lam*(w*(w.clamp_min(1e-8).log()-model.prior.log()[None])).sum(-1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            model.eval()
            with torch.no_grad():
                xx,pp,yy=data['select'];pred=(pp*model(xx)[:,:,None]).sum(-1);e=yy[:,:,None]-pred;v=float(torch.maximum(tau*e,(tau-1)*e).mean())
            trace.append({'epoch':epoch,'selection_loss':v})
            if v<best-1e-7:
                best=v;stale=0;state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch
            else:stale+=1
            if stale>=12:break
        path=out/f'gate_{lam}.pt';torch.save({'state':state,'mean':mean,'sd':sd,'n_features':x.shape[-1],'n_experts':p.shape[-1],'prior':prior,'penalty':lam,'seed':17},path)
        rows.append({'method':f'gate_{lam}','selection_loss':best,'best_epoch':best_epoch,'epochs':epoch,'trace':trace})
        objects.append({'method':f'gate_{lam}','path':str(path)})
        print('GATE',lam,'best',best,'epoch',best_epoch,flush=True)
    return rows,objects

def apply_gate(p,d,path):
    local=R/'selection'/Path(path).name
    if local.exists():path=local
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=DistributionGate(ck['n_features'],ck['n_experts'],ck['prior'],ck['target_scale']) if ck.get('ordered_correction') else Gate(ck['n_features'],ck['n_experts'],ck['prior'])
    model.load_state_dict(ck['state']);model.eval()
    x=(gate_features(p,d)-ck['mean'])/ck['sd']
    with torch.no_grad():
        if ck.get('ordered_correction'):
            pred,w,_=model.distribution(torch.tensor(x),torch.tensor(p));return pred.numpy(),w.numpy()
        w=model(torch.tensor(x)).numpy()
    return np.einsum('nhqm,nhm->nhq',p,w),w

class DistributionGate(Gate):
    def __init__(self,n,m,prior,scale):
        super().__init__(n,m,prior);self.correction=torch.nn.Sequential(torch.nn.Linear(n,32),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(32,9))
        torch.nn.init.zeros_(self.correction[-1].weight);torch.nn.init.zeros_(self.correction[-1].bias);self.scale=float(scale)
    def distribution(self,x,p):
        w=self(x);q0=(p*w[:,:,None,:]).sum(-1);theta=self.correction(x)
        median=q0[...,4:5]+self.scale*.2*torch.tanh(theta[...,:1])
        gap=(q0[...,1:]-q0[...,:-1])*torch.exp(.5*torch.tanh(theta[...,1:]))
        lower=median-torch.flip(torch.cumsum(torch.flip(gap[...,:4],[-1]),-1),[-1]);upper=median+torch.cumsum(gap[...,4:],-1)
        return torch.cat([lower,median,upper],-1).clamp_min(0),w,theta

def fit_ordered_gates(p,d,prior,out):
    torch.set_num_threads(2);fit,sel=year_masks(d);x=gate_features(p,d);mean=x[fit].mean((0,1));sd=np.maximum(x[fit].std((0,1)),1e-5);x=(x-mean)/sd
    tr=load(R/'prepared/train.npz');scale=float(tr['last'].std());data={r:tuple(torch.tensor(a[ix],dtype=torch.float32) for a in [x,p,d['y']]) for r,ix in [('fit',fit),('sel',sel)]};tau=torch.tensor(Q,dtype=torch.float32)
    config={'timestamp_utc':now(),'purpose':'Integrate ordered median/gap correction from the original adapter with state-dependent expert aggregation','penalties':[.001,.01],'theta_penalty':.01,'epochs':100,'patience':12,'median_bound_training_sd':.2,'log_gap_bound':.5,'fit_targets':'2020','selection_targets':'2021','test_read':False}
    dump(out/'ordered_gate_protocol.json',config);rows=[];objects=[]
    for lam in config['penalties']:
        torch.manual_seed(17);model=DistributionGate(x.shape[-1],p.shape[-1],prior,scale);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);best=float('inf');stale=0;trace=[]
        for epoch in range(1,101):
            model.train();xx,pp,yy=data['fit']
            for ids in torch.randperm(len(xx)).split(512):
                opt.zero_grad();pred,w,theta=model.distribution(xx[ids],pp[ids]);e=yy[ids,:,None]-pred
                loss=torch.maximum(tau*e,(tau-1)*e).mean()+lam*(w*(w.clamp_min(1e-8).log()-model.prior.log()[None])).sum(-1).mean()+.01*theta.square().mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            model.eval()
            with torch.no_grad():
                xx,pp,yy=data['sel'];pred,_,_=model.distribution(xx,pp);e=yy[:,:,None]-pred;v=float(torch.maximum(tau*e,(tau-1)*e).mean())
            trace.append({'epoch':epoch,'selection_loss':v})
            if v<best-1e-7:best=v;stale=0;state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch
            else:stale+=1
            if stale>=12:break
        path=out/f'ordered_gate_{lam}.pt';torch.save({'state':state,'mean':mean,'sd':sd,'n_features':x.shape[-1],'n_experts':p.shape[-1],'prior':prior,'penalty':lam,'seed':17,'ordered_correction':True,'target_scale':scale},path)
        rows.append({'method':f'ordered_gate_{lam}','selection_loss':best,'best_epoch':best_epoch,'epochs':epoch,'trace':trace});objects.append({'method':f'ordered_gate_{lam}','path':str(path)})
        print('ORDERED_GATE',lam,best,best_epoch,flush=True)
    return rows,objects
