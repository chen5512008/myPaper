"""Give the adapter the static site identity already used by LightGBM."""
import torch
from torch import nn
from torch.nn import functional as F
from model_v2 import RevisedAdapter

class RegionAdapter(RevisedAdapter):
    needs_site=True
    def __init__(self,scale,mean,variant,empirical):
        super().__init__(scale,mean,variant,empirical)
        self.site_embedding=nn.Embedding.from_pretrained(torch.zeros(6,self.d),freeze=False)
    def forward(self,x,base,observed=None,site_ids=None):
        if site_ids is None:raise ValueError('Explicit site IDs required; no implicit geographic inference.')
        b=x.shape[0];mask=torch.ones_like(x[...,4:]) if observed is None else observed.to(x.dtype)
        weather=x[...,4:]*mask;reliability=mask.reshape(b,28,24).mean(-1)
        q0=base.sort(-1).values
        if self.variant.startswith('empirical'):
            last=x[:,-1,0]*self.target_scale+self.target_mean;q0=(last[:,None,None]+self.empirical[None]).clamp_min(0)
        state=x[:,-1,:4,None].transpose(1,2).expand(-1,4,-1)
        qfeatures=torch.cat([q0/self.target_scale,state,self.leads[None,:,None].expand(b,-1,-1)],-1)
        query=self.query(qfeatures)+self.site_embedding(site_ids.long())[:,None,:]
        theta=torch.zeros((b,4,9),device=x.device,dtype=x.dtype)
        if not self.summary_only:
            wave=self.wave_block(self.wave_embed(x[...,:4].reshape(b,28,24))+self.position)
            met=self.weather_block(self.weather_embed(torch.cat([weather.reshape(b,28,24),mask.reshape(b,28,24)],-1))+self.position,reliability)
            wc,_=self.wave_cross(query,wave);mc,_=self.weather_cross(query+wc,met,reliability)
            theta=self.wave_head(torch.cat([query,wc],-1))+reliability.mean(-1)[:,None,None]*self.weather_head(torch.cat([query,wc,mc],-1))
        if self.shortcut:
            xx=torch.cat([x[...,:4],weather],-1);lags=[0,1,3,6,12,24,48,72,167];feats=[xx[:,-1-l] for l in lags]
            for w in [6,24,72,168]:feats.extend([xx[:,-w:].mean(1),xx[:,-w:].std(1,unbiased=False)])
            feats.extend([mask[:,-1-l] for l in lags]);feats.extend([mask[:,-w:].mean(1) for w in [6,24,72,168]])
            feats.extend([xx[:,-1]-xx[:,-1-l] for l in [6,24,72]])
            history=torch.cat(feats,-1)[:,None].expand(-1,4,-1);z=torch.cat([history,qfeatures],-1)
            branch=self.short.branch if hasattr(self.short,'branch') else self.short
            # Share the same 32-dimensional site embedding across attention and four hidden groups.
            bias=self.site_embedding(site_ids.long())[:,None,:].repeat(1,4,4)
            delta=branch[2:](F.gelu(branch[0](z)+bias))
            if hasattr(self.short,'gate'):delta=delta*torch.sigmoid(self.short.gate(z[...,-38:]))
            theta=theta+delta
        median=q0[...,4:5]+self.target_scale*theta[...,:1];basegap=(q0[...,1:]-q0[...,:-1]).clamp_min(0)
        if self.flexible:
            g=(basegap/self.target_scale).clamp_min(.02);gaps=self.target_scale*F.softplus(g+torch.log(-torch.expm1(-g))+theta[...,1:])
        else:gaps=basegap*torch.exp(2*torch.tanh(theta[...,1:]))
        lower=median-torch.flip(torch.cumsum(torch.flip(gaps[...,:4],[-1]),-1),[-1]);upper=median+torch.cumsum(gaps[...,4:],-1)
        return torch.cat([lower,median,upper],-1).clamp_min(0),{}
