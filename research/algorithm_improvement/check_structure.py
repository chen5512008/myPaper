"""Numerical checks using source validation histories, not test outcomes."""
from pathlib import Path
import json,time
import numpy as np,torch
from model_factory import build_model
R=Path(__file__).parent;D=R.parent;O=R/'results';O.mkdir(exist_ok=True)
torch.set_num_threads(2)
mode=json.loads((R/'selection.json').read_text())['variant']
ck=torch.load(R/f'runs/{mode}_17/model.pt',map_location='cpu',weights_only=False)
model=build_model(ck['sd'][0],ck['mean'][0],mode,ck['empirical_reference']);model.load_state_dict(ck['state_dict']);model.eval()
z=np.load(D/'prepared/BH01.npz');c=np.load(D/'forecasts/BH01_validation_base.npz');ids=c['indices'][:64]
xx=((z['raw']-ck['mean'])/ck['sd']).astype('float32');x=torch.as_tensor(xx[ids[:,None]-np.arange(167,-1,-1)]);q=torch.as_tensor(c['prediction'][:64])
mask=torch.zeros_like(x[...,4:])
kw={'site_ids':torch.zeros(len(x),dtype=torch.long)} if getattr(model,'needs_site',False) else {}
with torch.no_grad():
    p=model(x,q,mask,**kw)[0];perturbed=x.clone();perturbed[...,4:]=123.;pp=model(perturbed,q,mask,**kw)[0];complete=model(x,q,**kw)[0]
assert torch.isfinite(complete).all() and torch.isfinite(p).all()
assert (p-pp).abs().max()==0
assert min(float(p.min()),float(complete.min()))>=0
assert min(float(p.diff(dim=-1).min()),float(complete.diff(dim=-1).min()))>=-1e-6
model.zero_grad();model(x,q,**kw)[0].sum().backward()
active=sum(a.numel() for a in model.parameters() if a.grad is not None)
report={'variant':mode,'validation_histories':64,'maximum_change_after_replacing_fully_masked_weather':float((p-pp).abs().max()),'minimum_complete_gap':float(complete.diff(dim=-1).min()),'minimum_missing_gap':float(p.diff(dim=-1).min()),'stored_parameters':sum(a.numel() for a in model.parameters()),'parameters_with_gradient_path':active,'scope':'Finite-value structural invariance and quantile-order checks, not forecast accuracy evidence.'}
(O/'structural_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
