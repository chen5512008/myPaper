from pathlib import Path
import argparse,os,subprocess,sys
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['figures','review-training','review-trees','review-adapters','review-metrics','fusion-metrics','sci-prepare','sci-training','sci-evaluate','sci-components','sci-resources'],default='figures');a=p.parse_args()
env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
v=root/'research/review_followup_20261018';f=root/'research/fusion_revision_20261017'
jobs={'figures':[(v/'method_figure.py',[]),(v/'figures.py',[]),(f/'figures.py',[])],'review-training':[(v/'run_repeats.py',[])],'review-trees':[(v/'diagnostics.py',['--phase','trees'])],'review-adapters':[(v/'diagnostics.py',['--phase','adapters'])],'review-metrics':[(v/'summarize_repeats.py',[]),(v/'diagnostics.py',['--phase','analyze'])],'fusion-metrics':[(f/'evaluate_repeats.py',[])]}
s=root/'research/sci4_revision_20261018'
jobs['figures'].append((s/'figures.py',[]))
jobs.update({'sci-prepare':[(s/'fair_common.py',[])],'sci-training':[(s/'train_fair_lightgbm.py',[])]+[(s/'train_fair.py',['--model',name,'--seed',str(seed),*(['--missing-indicators'] if flags else [])]) for name,flags in [('tabm',False),('tabm',True),('adapter',False)] for seed in [17,29,43]],'sci-evaluate':[(s/'evaluate_fair.py',[])],'sci-components':[(s/'audit_combinations.py',[])],'sci-resources':[(s/'benchmark_models.py',[])]})
if a.stage!='figures' and not (root/'research/prepared/BH01.npz').exists():raise SystemExit('Extract reproduction-assets.zip into this directory first. See README.md.')
for script,args in jobs[a.stage]:subprocess.run([sys.executable,'-u',str(script),*args],env=env,check=True)
