import sys,subprocess
from common import *
assert (R/'selection.json').exists()
files=[f for f in (R/'experts').rglob('*') if f.suffix in ['.pt','.cbm','.ubj']]+list((R/'selection').glob('*.pt'))+list((A/'strong_lightgbm').glob('h*_q*.txt'))
files+=[A/f'runs/empirical_summary_flex_only_region_stable_{s}/model.pt' for s in [17,29,43]]+[A/'runs/chronos_summary_flex_gated_stable_17/model.pt']
checkpoints={str(f.relative_to(D)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
inputs={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['prepared','reserved_q3'] for f in (R/folder).glob('*.npz')}
freeze={'timestamp_utc':now(),'selection_sha256':hashlib.sha256((R/'selection.json').read_bytes()).hexdigest(),'checkpoints':checkpoints,'prepared_inputs':inputs,'seeds':[17,29,43],'no_test_driven_changes':True}
dest=R/'frozen_before_evaluation.json'
if dest.exists():
    old=json.loads(dest.read_text());assert old['selection_sha256']==freeze['selection_sha256'] and old['checkpoints']==checkpoints and old['prepared_inputs']==inputs,'Frozen files changed; do not silently overwrite the evaluation provenance.'
else:dump(dest,freeze)
for seed in [17,29,43]:
    for role in ['test','h1','q3']:
        print('PREDICTION_STAGE',seed,role,flush=True)
        r=subprocess.run([sys.executable,'-u',str(R/'predict_frozen.py'),'--role',role,'--seed',str(seed)],cwd=R)
        if r.returncode:raise SystemExit(r.returncode)
for name in ['evaluate.py','evaluate_repeats.py','figures.py','report.py']:
    if not (R/name).exists():continue
    r=subprocess.run([sys.executable,'-u',str(R/name)],cwd=R)
    if r.returncode:raise SystemExit(r.returncode)
dump(R/'completed.json',{'timestamp_utc':now(),'training_repetitions':3,'evaluation_complete':True,'selection_sha256':hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()})
