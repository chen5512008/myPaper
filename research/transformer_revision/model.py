"""Historical-only distribution-query Transformer adapter. Research prototype."""
import math
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self,d=32,heads=4):
        super().__init__();self.heads=heads;self.d=d
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
    def forward(self,query,memory,reliability=None):
        b,n,_=query.shape;m=memory.shape[1];h=self.heads
        q=self.q(query).reshape(b,n,h,-1).transpose(1,2)
        k=self.k(memory).reshape(b,m,h,-1).transpose(1,2)
        v=self.v(memory).reshape(b,m,h,-1).transpose(1,2)
        score=(q@k.transpose(-1,-2))/math.sqrt(self.d/h)
        if reliability is not None:score=score+reliability.clamp_min(1e-8).log()[:,None,None,:]
        w=score.softmax(-1)
        z=(w@v).transpose(1,2).reshape(b,n,self.d)
        return self.out(z),w.mean(1)

class Block(nn.Module):
    def __init__(self,d=32):
        super().__init__();self.attn=Attention(d);self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(self,x,reliability=None):
        x=x+self.attn(self.n1(x),self.n1(x),reliability)[0]
        return x+self.ff(self.n2(x))

class DistributionQueryAdapter(nn.Module):
    def __init__(self,target_scale,target_mean=0.,variant='full',hidden=32,empirical_reference=None):
        super().__init__();self.variant=variant;self.patch=6;self.d=hidden
        self.register_buffer('target_scale',torch.tensor(float(target_scale)))
        self.register_buffer('target_mean',torch.tensor(float(target_mean)))
        self.register_buffer('leads',torch.tensor([6,12,24,48],dtype=torch.float32)/48)
        self.register_buffer('empirical',torch.zeros(4,9) if empirical_reference is None else torch.as_tensor(empirical_reference,dtype=torch.float32))
        self.wave_embed=nn.Linear(24,hidden);self.weather_embed=nn.Linear(48,hidden)
        self.position=nn.Parameter(torch.randn(1,28,hidden)*.02)
        self.wave_block=Block(hidden);self.weather_block=Block(hidden)
        self.query=nn.Sequential(nn.Linear(14,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.wave_cross=Attention(hidden);self.weather_cross=Attention(hidden)
        self.wave_head=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,9))
        self.weather_head=nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Linear(hidden,9))
        for head in [self.wave_head,self.weather_head]:
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    def forward(self,x,base,observed=None):
        b=x.shape[0]
        mask=torch.ones_like(x[...,4:]) if observed is None else observed.to(x.dtype)
        weather=x[...,4:]*mask
        # Explicit missingness is an input; standardized zero may also be a valid observation.
        if self.variant=='no_availability':mask=torch.ones_like(mask)
        reliability=mask.reshape(b,28,24).mean(-1)
        wave=self.wave_embed(x[...,:4].reshape(b,28,24))+self.position
        met=self.weather_embed(torch.cat([weather.reshape(b,28,24),mask.reshape(b,28,24)],-1))+self.position
        wave=self.wave_block(wave)
        met=self.weather_block(met,reliability)
        q0=base.sort(-1).values
        if self.variant=='empirical_reference':
            last=x[:,-1,0]*self.target_scale+self.target_mean
            q0=(last[:,None,None]+self.empirical[None]).clamp_min(0)
        qfeatures=q0/self.target_scale
        if self.variant=='no_distribution_query':qfeatures=torch.zeros_like(qfeatures)
        state=x[:,-1,:4,None].transpose(1,2).expand(-1,4,-1)
        query=self.query(torch.cat([qfeatures,state,self.leads[None,:,None].expand(b,-1,-1)],-1))
        wave_context,wa=self.wave_cross(query,wave)
        met_context,ma=self.weather_cross(query+wave_context,met,reliability)
        wave_theta=self.wave_head(torch.cat([query,wave_context],-1))
        weather_theta=self.weather_head(torch.cat([query,wave_context,met_context],-1))
        available=reliability.mean(-1)[:,None,None]
        theta=wave_theta+available*weather_theta
        if self.variant=='affine':theta=torch.cat([theta[...,:1],theta[...,1:].mean(-1,keepdim=True).expand(-1,-1,8)],-1)
        median=q0[...,4:5]+self.target_scale*theta[...,:1]
        gaps=(q0[...,1:]-q0[...,:-1]).clamp_min(0)*torch.exp(2*torch.tanh(theta[...,1:]))
        lower=median-torch.flip(torch.cumsum(torch.flip(gaps[...,:4],[-1]),-1),[-1])
        upper=median+torch.cumsum(gaps[...,4:],-1)
        prediction=torch.cat([lower,median,upper],-1).clamp_min(0)
        return prediction,{'weather_weights':ma,'wave_weights':wa,'availability':available,'gap_multiplier':torch.exp(2*torch.tanh(theta[...,1:]))}

def pinball(p,y):
    levels=torch.arange(1,10,device=p.device,dtype=p.dtype)/10
    residual=y[...,None]-p
    return torch.maximum(levels*residual,(levels-1)*residual).mean()
