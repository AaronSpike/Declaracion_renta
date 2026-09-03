"""E3: objetivos alternativos y diversidad de hiperparametros.
Todo evaluado igual: OOF 135.038 filas, P@300, 2 particiones, pareado."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar, pk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.special import expit

df,FE,y=cargar(); X=df[FE]
BASE=dict(learning_rate=0.03,num_leaves=15,min_child_samples=200,feature_fraction=0.6,
          bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.,n_estimators=400,
          verbose=-1,n_jobs=2,seed=11,bagging_seed=11,feature_fraction_seed=11)

def focal(gamma=2.0):
    def f(yt,raw):
        p=expit(raw); g=gamma
        # grad/hess de la focal loss binaria (derivacion estandar)
        pt=np.where(yt==1,p,1-p)
        # d(-(1-pt)^g log pt)/draw
        s=np.where(yt==1,1.0,-1.0)
        grad=s*(-(1-pt)**g)*(g*pt*np.log(np.clip(pt,1e-9,1))/(1-pt+1e-9)*(-(1-pt))/1.0 + (1-pt))
        # aproximacion robusta: usamos hess de logloss escalada
        grad=-s*(1-pt)**g*((1-pt)-g*pt*np.log(np.clip(pt,1e-9,1)))
        hess=np.abs(grad)*(1-np.abs(grad))+1e-6
        hess=np.clip(p*(1-p)*(1-pt)**g*(1+g),1e-6,None)
        return grad,hess
    return f

def run(nombre, fit_fn, splits=(42,)):
    ps=[];aucs=[]
    for sp in splits:
        skf=StratifiedKFold(5,shuffle=True,random_state=sp); oof=np.zeros(len(X))
        for tr,va in skf.split(X,y):
            oof[va]=fit_fn(tr,va)
        ps.append(pk(y,oof)); aucs.append(roc_auc_score(y,oof))
    print(f'{nombre:38s} P@300={np.mean(ps):.4f} {np.round(ps,4)}  AUC={np.mean(aucs):.4f}',flush=True)
    return np.array(ps)

t0=time.time()
def f_bin(tr,va):
    m=lgb.LGBMClassifier(objective='binary',**BASE); m.fit(X.iloc[tr],y[tr])
    return m.predict_proba(X.iloc[va])[:,1]
r_bin=run('binaria (referencia)',f_bin)
print('t=%.0fs'%(time.time()-t0),flush=True)

def f_focal(tr,va):
    m=lgb.LGBMClassifier(objective=focal(2.0),**BASE); m.fit(X.iloc[tr],y[tr])
    return m.predict(X.iloc[va],raw_score=True)
try: run('focal loss gamma=2',f_focal)
except Exception as e: print('focal fallo:',e,flush=True)
print('t=%.0fs'%(time.time()-t0),flush=True)

# lambdarank con grupos aleatorios: fuerza al modelo a ordenar dentro de bloques
def f_rank(tr,va,gsize=500,trunc=10):
    n=len(tr); rng=np.random.default_rng(0); perm=rng.permutation(n)
    tr2=tr[perm]; ng=n//gsize
    grp=[gsize]*ng; resto=n-gsize*ng
    if resto: grp.append(resto)
    p={k:v for k,v in BASE.items()}
    m=lgb.LGBMRanker(objective='lambdarank',label_gain=[0,1],
                     lambdarank_truncation_level=trunc,**p)
    m.fit(X.iloc[tr2],y[tr2],group=grp)
    return m.predict(X.iloc[va])
try: run('lambdarank (grupos 500, trunc 10)',f_rank)
except Exception as e: print('lambdarank fallo:',repr(e)[:300],flush=True)
print('t=%.0fs'%(time.time()-t0),flush=True)

# ensamble de hiperparametros diversos (rango y prob)
CFGS=[dict(num_leaves=15,min_child_samples=200,feature_fraction=0.6,lambda_l2=50.,learning_rate=0.03,n_estimators=400),
      dict(num_leaves=7 ,min_child_samples=300,feature_fraction=0.8,lambda_l2=10.,learning_rate=0.03,n_estimators=400),
      dict(num_leaves=31,min_child_samples=100,feature_fraction=0.7,lambda_l2=20.,learning_rate=0.02,n_estimators=500)]
def f_ens(tr,va,modo='prob'):
    out=[]
    for i,c in enumerate(CFGS):
        p={**BASE,**c,'seed':11+i,'bagging_seed':11+i,'feature_fraction_seed':11+i}
        m=lgb.LGBMClassifier(objective='binary',**p); m.fit(X.iloc[tr],y[tr])
        out.append(m.predict_proba(X.iloc[va])[:,1])
    return np.mean(out,0) if modo=='prob' else np.mean([rankdata(o) for o in out],0)
run('ensamble 3 hiperparams (prob)',lambda tr,va:f_ens(tr,va,'prob'))
run('ensamble 3 hiperparams (rango)',lambda tr,va:f_ens(tr,va,'rango'))
print('t=%.0fs'%(time.time()-t0),flush=True)
