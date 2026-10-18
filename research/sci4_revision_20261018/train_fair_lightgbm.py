"""CPU-only runner for the same frozen LightGBM recipe, avoiding a CUDA runtime."""
import time,gc
from concurrent.futures import ThreadPoolExecutor,as_completed
import lightgbm as lgb
from fair_common import *
out=R/'runs/lightgbm_17';out.mkdir(exist_ok=True)
tr=load(R/'prepared/train.npz');va=load(R/'prepared/validation.npz');fit,select=year_masks(va);eval_x=va['X'][fit]
if (out/'completed.json').exists():print('ALREADY COMPLETE');raise SystemExit
start=time.perf_counter();pred=np.empty((len(va['y']),4,9),np.float32)
def job(hi,qi):
    h=H[hi];tau=Q[qi];dest=out/f'h{h}_q{qi+1}.txt'
    if dest.exists():b=lgb.Booster(model_file=str(dest));iteration=b.current_iteration()
    else:
        m=lgb.LGBMRegressor(objective='quantile',alpha=float(tau),n_estimators=600,num_leaves=31,learning_rate=.05,min_child_samples=50,colsample_bytree=1.,random_state=17,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
        m.fit(tr['X'],tr['y'][:,hi],eval_X=eval_x,eval_y=va['y'][fit,hi],callbacks=[lgb.early_stopping(40,verbose=False)])
        b=m.booster_;iteration=m.best_iteration_;b.save_model(str(dest))
    pred[:,hi,qi]=b.predict(va['X'],num_threads=2)
    log={'horizon':h,'quantile':float(tau),'iteration':iteration};print('FAIR_LGBM',h,qi+1,iteration,'seconds',round(time.perf_counter()-start,1),flush=True)
    return log
logs=[]
with ThreadPoolExecutor(max_workers=1) as pool:
    for future in as_completed([pool.submit(job,hi,qi) for hi in range(4) for qi in range(9)]):logs.append(future.result())
save(out/'validation.npz',pred,va)
pred=load(out/'validation.npz')['prediction'];dump(out/'completed.json',{'completed_utc':now(),'model':'lightgbm','seed':17,'seconds':time.perf_counter()-start,'validation2020':float(pin(pred[fit],va['y'][fit]).mean()),'validation2021_report_only':float(pin(pred[select],va['y'][select]).mean()),'trace':sorted(logs,key=lambda x:(x['horizon'],x['quantile'])),'test_evaluated':False,'resource_runner':'CPU-only sequential quantile jobs with two threads; healthy completed models reused after documented interrupted resource attempts; statistical recipe unchanged'})
print('TRAINING COMPLETE lightgbm 17',flush=True)
