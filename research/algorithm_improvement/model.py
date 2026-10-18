"""Incremental historical distribution adaptation; no future covariates."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
_spec=importlib.util.spec_from_file_location('original_distribution_adapter',Path(__file__).resolve().parent.parent/'transformer_revision/model.py')
_original=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_original)
OriginalAdapter=_original.DistributionQueryAdapter
pinball=_original.pinball

class ImprovedAdapter(OriginalAdapter):
    def __init__(self, scale, mean, variant, empirical):
        reference='empirical_reference' if variant.startswith('empirical') else 'full'
        super().__init__(scale,mean,reference,empirical_reference=empirical)
        self.mode=variant
        self.flexible='flex' in variant
        self.shortcut='summary' in variant
        self.summary_only='only' in variant
        if self.shortcut:
            # 136 historical values/statistics, 52 observed-mask summaries, 24 changes, 14 query entries.
            self.short=nn.Sequential(nn.Linear(226,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,9))
            nn.init.zeros_(self.short[-1].weight);nn.init.zeros_(self.short[-1].bias)

    def forward(self,x,base,observed=None):
        b=x.shape[0]
        mask=torch.ones_like(x[...,4:]) if observed is None else observed.to(x.dtype)
        weather=x[...,4:]*mask
        reliability=mask.reshape(b,28,24).mean(-1)
        q0=base.sort(-1).values
        if self.variant.startswith('empirical'):
            last=x[:,-1,0]*self.target_scale+self.target_mean
            q0=(last[:,None,None]+self.empirical[None]).clamp_min(0)
        state=x[:,-1,:4,None].transpose(1,2).expand(-1,4,-1)
        qfeatures=torch.cat([q0/self.target_scale,state,self.leads[None,:,None].expand(b,-1,-1)],-1)
        query=self.query(qfeatures)
        theta=torch.zeros((b,4,9),device=x.device,dtype=x.dtype)
        if not self.summary_only:
            wave=self.wave_block(self.wave_embed(x[...,:4].reshape(b,28,24))+self.position)
            met=self.weather_block(self.weather_embed(torch.cat([weather.reshape(b,28,24),mask.reshape(b,28,24)],-1))+self.position,reliability)
            wc,_=self.wave_cross(query,wave)
            mc,_=self.weather_cross(query+wc,met,reliability)
            theta=self.wave_head(torch.cat([query,wc],-1))+reliability.mean(-1)[:,None,None]*self.weather_head(torch.cat([query,wc,mc],-1))
        if self.shortcut:
            xx=torch.cat([x[...,:4],weather],-1)
            lags=[0,1,3,6,12,24,48,72,167]
            feats=[xx[:,-1-l] for l in lags]
            for w in [6,24,72,168]:feats.extend([xx[:,-w:].mean(1),xx[:,-w:].std(1,unbiased=False)])
            feats.extend([mask[:,-1-l] for l in lags])
            feats.extend([mask[:,-w:].mean(1) for w in [6,24,72,168]])
            feats.extend([xx[:,-1]-xx[:,-1-l] for l in [6,24,72]])
            history=torch.cat(feats,-1)[:,None].expand(-1,4,-1)
            theta=theta+self.short(torch.cat([history,qfeatures],-1))
        median=q0[...,4:5]+self.target_scale*theta[...,:1]
        basegap=(q0[...,1:]-q0[...,:-1]).clamp_min(0)
        if self.flexible:
            g=(basegap/self.target_scale).clamp_min(.02)
            inverse=g+torch.log(-torch.expm1(-g))
            gaps=self.target_scale*F.softplus(inverse+theta[...,1:])
        else:gaps=basegap*torch.exp(2*torch.tanh(theta[...,1:]))
        lower=median-torch.flip(torch.cumsum(torch.flip(gaps[...,:4],[-1]),-1),[-1])
        upper=median+torch.cumsum(gaps[...,4:],-1)
        return torch.cat([lower,median,upper],-1).clamp_min(0),{}
