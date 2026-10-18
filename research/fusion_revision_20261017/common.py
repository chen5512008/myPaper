from pathlib import Path
import json,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent;D=R.parent;A=D/'algorithm_improvement'
SITES=json.loads((D/'protocol.json').read_text())['sites'];H=np.array([6,12,24,48]);Q=np.arange(1,10)/10
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def pin(p,y):
    e=y[...,None]-p;return np.maximum(Q*e,(Q-1)*e)
def score(p,y):return float(pin(p,y).mean())
def load(path):return dict(np.load(path))
def features(raw,ids,si):
    pieces=[raw[ids-l] for l in [0,1,3,6,12,24,48,72,167]]
    for w in [6,24,72,168]:
        hist=np.stack([raw[ids-l] for l in range(w)],axis=1);pieces.extend([hist.mean(1),hist.std(1)])
    return np.concatenate(pieces+[np.full((len(ids),1),si)]+[raw[ids]-raw[ids-l] for l in [6,24,72]],axis=1).astype('float32')
def save_predictions(path,p,d):
    p=np.maximum(0,np.sort(p,axis=-1)).astype('float32');assert np.isfinite(p).all()
    np.savez_compressed(path,prediction=p,target=d['y'],site=d['site'],origin=d['origin'])
def align(z,anchor):
    index={(str(s),str(t)):i for i,(s,t) in enumerate(zip(z['site'],z['origin']))};assert len(index)==len(z['site'])
    ids=[index[(str(s),str(t))] for s,t in zip(anchor['site'],anchor['origin'])]
    out={k:np.asarray(z[k])[ids] for k in ['prediction','target','site','origin']}
    assert np.allclose(out['target'],anchor.get('target',anchor.get('y')))
    return out
