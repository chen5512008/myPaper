"""New residual multi-quantile experts; 2020 stopping, no test scoring."""
import argparse,time
import pandas as pd
from common import *
from fusion import year_masks
ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['xgb','catboost'],required=True);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--only-horizon',type=int,choices=H.tolist());args=ap.parse_args()
out=R/'experts'/f'{args.model}_{args.seed}';out.mkdir(parents=True,exist_ok=True)
train=load(R/'prepared/train.npz');valid=load(R/'prepared/validation.npz');fit,sel=year_masks(valid)
residual=train['y']-train['last'][:,None];val_residual=valid['y']-valid['last'][:,None]
pred=np.zeros((len(valid['y']),4,9),dtype='float32');logs=[];start=time.perf_counter()
if args.model=='xgb':
    import xgboost as xgb
    params={'objective':'reg:quantileerror','quantile_alpha':Q.tolist(),'tree_method':'hist','device':'cuda','max_depth':5,'eta':.04,'subsample':.9,'colsample_bytree':.9,'min_child_weight':20,'reg_lambda':10,'seed':args.seed,'nthread':2,'max_bin':256}
    for j,h in enumerate(H):
        if args.only_horizon is not None and h!=args.only_horizon:continue
        dest=out/f'h{h}.ubj'
        if dest.exists():model=xgb.Booster();model.load_model(dest)
        else:
            tr=xgb.QuantileDMatrix(train['X'],label=residual[:,j]);va=xgb.QuantileDMatrix(valid['X'][fit],label=val_residual[fit,j],ref=tr)
            model=xgb.train(params,tr,num_boost_round=1200,evals=[(va,'validation2020')],early_stopping_rounds=80,verbose_eval=200);model.save_model(dest)
        end=int(model.best_iteration)+1
        pred[:,j]=model.inplace_predict(valid['X'],iteration_range=(0,end))+valid['last'][:,None]
        logs.append({'horizon':int(h),'best_iteration':end});print('XGB_SAVED',h,end,'seconds',round(time.perf_counter()-start,1),flush=True)
    version=xgb.__version__
else:
    from catboost import CatBoostRegressor,Pool
    import catboost
    params={'iterations':1200,'depth':6,'learning_rate':.05,'l2_leaf_reg':5,'loss_function':'MultiQuantile:alpha='+','.join(f'{q:.1f}' for q in Q),'task_type':'CPU','thread_count':4,'random_seed':args.seed,'allow_writing_files':False,'od_type':'Iter','od_wait':80,'verbose':200}
    def frame(a):
        f=pd.DataFrame(a,columns=[f'f{k}' for k in range(a.shape[1])]);f['f136']=f['f136'].astype('int32');return f
    tf=frame(train['X']);vf=frame(valid['X']);vff=vf.loc[fit]
    for j,h in enumerate(H):
        if args.only_horizon is not None and h!=args.only_horizon:continue
        dest=out/f'h{h}.cbm';model=CatBoostRegressor(**params)
        if dest.exists():model.load_model(dest)
        else:model.fit(Pool(tf,label=residual[:,j],cat_features=['f136']),eval_set=Pool(vff,label=val_residual[fit,j],cat_features=['f136']),use_best_model=True);model.save_model(dest)
        pred[:,j]=model.predict(Pool(vf,cat_features=['f136']))+valid['last'][:,None]
        logs.append({'horizon':int(h),'best_iteration':int(model.tree_count_)});print('CATBOOST_SAVED',h,model.tree_count_,'seconds',round(time.perf_counter()-start,1),flush=True)
    version=catboost.__version__
suffix='' if args.only_horizon is None else f'_h{args.only_horizon}'
save_predictions(out/f'validation{suffix}.npz',pred,valid)
dump(out/f'run{suffix}.json',{'timestamp_utc':now(),'model':args.model,'version':version,'seed':args.seed,'params':params,'logs':logs,'features':161,'target':'future wave minus current wave; current wave added back before scoring','n_train':len(train['y']),'n_early_stopping':int(fit.sum()),'stopping_targets':'2020','fusion_selection_targets':'2021','partial_horizon':args.only_horizon,'validation2021_score':score(np.maximum(0,np.sort(pred,axis=-1))[sel],valid['y'][sel]) if args.only_horizon is None else None,'seconds':time.perf_counter()-start})
print('DONE',args.model,args.seed,flush=True)
