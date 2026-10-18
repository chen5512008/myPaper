"""Query/trend gating for the explicit historical summary branch."""
import torch
from torch import nn
from model import ImprovedAdapter

class GatedSummary(nn.Module):
    def __init__(self,branch):
        super().__init__();self.branch=branch
        self.gate=nn.Sequential(nn.Linear(38,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2.)
    def forward(self,z):
        # Last 24 historical entries are 6/24/72-hour changes; final 14 are reference/state/lead.
        return self.branch(z)*torch.sigmoid(self.gate(z[...,-38:]))

class RevisedAdapter(ImprovedAdapter):
    def __init__(self,scale,mean,variant,empirical):
        super().__init__(scale,mean,variant,empirical)
        if '_gated_' in variant:self.short=GatedSummary(self.short)

def build_model(scale,mean,variant,empirical):
    return (RevisedAdapter if variant.endswith('_stable') else ImprovedAdapter)(scale,mean,variant,empirical)
