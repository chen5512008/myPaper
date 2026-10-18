"""Convex quantile aggregation; identical weights for all quantile levels."""
import numpy as np
from scipy.optimize import minimize
from common import Q,pin
def year_masks(d):
    t=np.asarray(d['origin']).astype('datetime64[h]');a=t+np.timedelta64(6,'h');b=t+np.timedelta64(48,'h')
    return (a>=np.datetime64('2020-01-01'))&(b<np.datetime64('2021-01-01')),(a>=np.datetime64('2021-01-01'))&(b<np.datetime64('2022-01-01'))
def fit_weights(pred,y,prior=None,penalty=0.):
    # pred: N,H,Q,M or N,Q,M; y: N,H or N.
    n=pred.shape[-1];prior=np.ones(n)/n if prior is None else np.asarray(prior)
    p=pred.reshape(-1,9,n).astype('float64');target=y.reshape(-1,1)
    def f(w):
        z=np.einsum('nqm,m->nq',p,w);e=target-z
        value=np.maximum(Q*e,(Q-1)*e).mean()+penalty*np.square(w-prior).sum()
        g=np.einsum('nq,nqm->m',(z>target)-Q,p)/(p.shape[0]*9)+2*penalty*(w-prior)
        return value,g
    result=minimize(f,prior,jac=True,method='SLSQP',bounds=[(0,1)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)}],options={'ftol':1e-11,'maxiter':250})
    w=np.maximum(0,result.x);w/=w.sum();assert np.isfinite(w).all()
    return w,{'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit)}
def pool(p,w,site_ids=None):
    if w.ndim==1:return np.einsum('nhqm,m->nhq',p,w)
    if w.ndim==2:return np.einsum('nhqm,hm->nhq',p,w)
    return np.einsum('nhqm,nhm->nhq',p,w[site_ids])
def linear_candidates(p,d,sites):
    fit,select=year_masks(d);y=d['y'];m=p.shape[-1];rows=[];objects=[]
    rows.append({'method':'uniform','selection_loss':float(pin(pool(p,np.ones(m)/m)[select],y[select]).mean())});objects.append({'method':'uniform','weights':np.ones(m)/m})
    for penalty in [0.,.0001,.001]:
        w,info=fit_weights(p[fit],y[fit],penalty=penalty)
        rows.append({'method':f'global_{penalty}','selection_loss':float(pin(pool(p,w)[select],y[select]).mean()),'fit_info':info});objects.append({'method':f'global_{penalty}','weights':w})
        wh=[];infos=[]
        for h in range(4):
            a,b=fit_weights(p[fit,h],y[fit,h],penalty=penalty);wh.append(a);infos.append(b)
        wh=np.stack(wh)
        rows.append({'method':f'horizon_{penalty}','selection_loss':float(pin(pool(p,wh)[select],y[select]).mean()),'fit_info':infos});objects.append({'method':f'horizon_{penalty}','weights':wh})
    wh=objects[2]['weights']
    for penalty in [.001,.01]:
        ws=[];infos=[]
        for s in sites:
            ix=fit&(d['site']==s);ss=[]
            for h in range(4):
                a,b=fit_weights(p[ix,h],y[ix,h],prior=wh[h],penalty=penalty);ss.append(a);infos.append(b)
            ws.append(ss)
        ws=np.array(ws);sid=np.array([sites.index(str(s)) for s in d['site']])
        rows.append({'method':f'site_horizon_{penalty}','selection_loss':float(pin(pool(p,ws,sid)[select],y[select]).mean()),'fit_info':infos});objects.append({'method':f'site_horizon_{penalty}','weights':ws})
    best=np.argmin([a['selection_loss'] for a in rows])
    return rows,objects,int(best)
