"""Replay saved checkpoints and verify recorded scores without changing selection."""
import torch,pandas as pd
from common import *
from train_tabm import make_model,predict
from gate import apply_gate
frozen=json.loads((R/'frozen_before_evaluation.json').read_text())
assert hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()==frozen['selection_sha256']
for name,sha in frozen['checkpoints'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==sha,name
for name,sha in frozen['prepared_inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,name
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
d=load(R/'prepared/q3.npz');folder=R/'experts/tabm_17';ck=torch.load(folder/'global.pt',map_location='cpu',weights_only=False)
model=make_model(ck['config'],ck['bins']).to('cuda');model.load_state_dict(ck['state']);p=predict(model,d,ck['mean'],ck['sd'],ck['scale'],'cuda')
saved=load(R/'predictions/q3/TabM.npz')['prediction'];error=float(np.abs(p-saved).max());assert error<1e-6
del model;torch.cuda.empty_cache()
choice=json.loads((R/'selection.json').read_text());pp=np.stack([load(R/'predictions/q3'/f'{n}.npz')['prediction'] for n in choice['members']],-1)
pred,w=apply_gate(pp,d,choice['object']['path']);saved=load(R/'results/q3_seed17_Selected_fusion.npz')['prediction'];gate_error=float(np.abs(pred-saved).max());assert gate_error<1e-6
frame=pd.read_csv(R/'results/scores_by_seed.csv');scores=[]
for _,row in frame[frame.model=='Selected_fusion'].iterrows():
    dd=load(R/'results'/f'{row.period}_seed{row.seed}_Selected_fusion.npz');v=score(dd['prediction'],dd['target']);assert abs(v-row.pinball)<1e-7;scores.append({'period':row.period,'seed':int(row.seed),'loss':v})
dump(R/'results/checkpoint_reproduction_check.json',{'timestamp_utc':now(),'frozen_model_hashes_unchanged':True,'prepared_input_hashes_unchanged':True,'fusion_selection_hash_unchanged':True,'TabM_Q3_replay_max_abs_error':error,'gate_Q3_replay_max_abs_error':gate_error,'all_nine_recorded_fusion_scores_recomputed':scores,'tolerance_metres':1e-6,'not_a_retraining_claim':'Checkpoint replay and cached figures are deterministic; CatBoost GPU retraining can differ in floating point reductions.'})
print('REPRODUCED CHECKPOINTS AND NINE FUSION SCORES',error,gate_error,flush=True)
