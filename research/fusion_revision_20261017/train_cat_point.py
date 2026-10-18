"""Memory-bounded CatBoost residual mean with fixed 2020 residual quantiles."""
import argparse,time
from catboost import CatBoostRegressor
import catboost
from common import *
from fusion import year_masks
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=17);args=ap.parse_args()
out=R/'experts'/f'catpoint_{args.seed}';out.mkdir(parents=True,exist_ok=True)
tr=load(R/'prepared/train.npz');va=load(R/'prepared/validation.npz');fit,sel=year_masks(va)
cfg={'iterations':1200,'depth':6,'learning_rate':.05,'l2_leaf_reg':5,'loss_function':'RMSE','task_type':'GPU','devices':'0','gpu_ram_part':.5,'random_seed':args.seed,'allow_writing_files':False,'od_type':'Iter','od_wait':80,'verbose':200,'border_count':64}
dump(out/'config.json',{'timestamp_utc':now(),'model':'CatBoost residual mean plus 2020 quantile residual calibration','params':cfg,'fallback':'CPU native MultiQuantile failed with bad allocation; two parallel horizon workers stopped to free RAM; all failure records retained','calibration':'Per-horizon additive 2020 residual quantiles; 2021 untouched by fitting'})
p=np.zeros((len(va['y']),4,9),dtype='float32');offsets=[];logs=[];start=time.perf_counter()
for j,h in enumerate(H):
    model=CatBoostRegressor(**cfg);dest=out/f'h{h}.cbm'
    if dest.exists():model.load_model(str(dest))
    else:
        model.fit(tr['X'],tr['y'][:,j]-tr['last'],eval_set=(va['X'][fit],va['y'][fit,j]-va['last'][fit]),use_best_model=True)
        model.save_model(str(dest))
    point=model.predict(va['X'])+va['last'];off=np.quantile(va['y'][fit,j]-point[fit],Q);offsets.append(off)
    p[:,j]=point[:,None]+off;logs.append({'horizon':int(h),'best_iteration':int(model.tree_count_)})
    print('CATPOINT_SAVED',h,model.tree_count_,round(time.perf_counter()-start,1),flush=True)
np.save(out/'offsets.npy',np.array(offsets));save_predictions(out/'validation.npz',p,va)
dump(out/'run.json',{'timestamp_utc':now(),'version':catboost.__version__,'seed':args.seed,'params':cfg,'logs':logs,'offsets':np.array(offsets).tolist(),'validation2021':score(np.maximum(0,np.sort(p,-1))[sel],va['y'][sel]),'seconds':time.perf_counter()-start})
print('DONE CATPOINT',flush=True)
