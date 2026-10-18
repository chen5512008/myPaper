"""Numerical diagnostics on saved forecasts and structural missingness invariance."""
from pathlib import Path
import json
import numpy as np
import torch
from model import DistributionQueryAdapter
torch.set_num_threads(2)
R=Path(__file__).parent;D=R.parent
checkpoint=torch.load(R/'runs/pooled_full_17/model.pt',map_location='cpu',weights_only=False)
model=DistributionQueryAdapter(checkpoint['sd'][0],checkpoint['mean'][0],empirical_reference=checkpoint['empirical_reference'])
model.load_state_dict(checkpoint['state_dict']);model.eval()
z=np.load(D/'prepared/BH01.npz');c=np.load(D/'forecasts/BH01_validation_base.npz')
ids=c['indices'][:64];a=(z['raw']-checkpoint['mean'])/checkpoint['sd']
x=torch.as_tensor(a[ids[:,None]-np.arange(167,-1,-1)],dtype=torch.float32)
q=torch.as_tensor(c['prediction'][:64],dtype=torch.float32);mask=torch.zeros_like(x[...,4:])
with torch.no_grad():
    p1=model(x,q,mask)[0];xx=x.clone();xx[...,4:]=123.
    p2=model(xx,q,mask)[0]
    complete=model(x,q,torch.ones_like(mask))[0]
report={'validation_windows':64,'site':'BH01','maximum_change_after_replacing_unobserved_weather':float((p1-p2).abs().max()),'minimum_adjacent_gap_complete':float(complete.diff(dim=-1).min()),'minimum_adjacent_gap_no_weather':float(p1.diff(dim=-1).min()),'minimum_prediction':float(torch.minimum(complete.min(),p1.min())),'note':'Finite input structural check, not an error or calibration guarantee.'}
(R/'results/structural_diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
