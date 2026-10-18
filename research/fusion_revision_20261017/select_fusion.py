"""Freeze a validation-selected fusion before test inference."""
import torch
from common import *
from bank import validation_bank,controls,apply_controls
from fusion import *
from gate import fit_gates,fit_ordered_gates

out=R/'selection';out.mkdir(exist_ok=True)
assert not (R/'selection.json').exists(),'Selection already frozen; do not overwrite after test exposure.'
d,members=validation_bank();names=list(members);p=np.stack(list(members.values()),-1);fit,sel=year_masks(d)
rows,objects,best=linear_candidates(p,d,SITES)
prior=next(o['weights'] for o in objects if o['method']=='horizon_0.001')
gr,go=fit_gates(p,d,prior,out);rows.extend(gr);objects.extend(go)
gr,go=fit_ordered_gates(p,d,prior,out);rows.extend(gr);objects.extend(go)
best=int(np.argmin([r['selection_loss'] for r in rows]));choice=objects[best]
c=controls(d,members['LightGBM'],fit);np.savez_compressed(out/'controls.npz',**c)
if 'weights' in choice:np.save(out/'weights.npy',choice['weights'])
single={n:score(v[sel],d['y'][sel]) for n,v in members.items()}
single.update({n:score(v[sel],d['y'][sel]) for n,v in apply_controls(d,members['LightGBM'],c).items()})
manifest={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (R/'experts').rglob('*') if f.is_file() and f.suffix in ['.pt','.ubj','.cbm','.json']}
result={'timestamp_utc':now(),'members':names,'single_2021':single,'candidates':rows,'selected':rows[best],'object':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in choice.items()},'fit_targets':'2020','selection_targets':'2021','new_Q3_evaluated':False,'new_models_seed':17,'existing_Neural3':'Prediction average of independent seeds17,29,43; explicitly an ensemble','new_models_manifest':manifest,'protocol':json.loads((R/'protocol.json').read_text())}
dump(R/'selection.json',result)
print('FROZEN',rows[best]['method'],rows[best]['selection_loss'],'SINGLES',single,flush=True)
