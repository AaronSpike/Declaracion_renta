"""E4: la regularizacion como palanca de CABEZA.
Hipotesis: lo que hizo subir P@100 de 15,1% a 21,3% no fue 'menos AUC' sino
impedir que hojas con pocos casos manden ruido al top. Si es asi, empujar mas
en esa direccion (min_child_samples, lambda_l2, min_gain_to_split) deberia
seguir moviendo la cabeza aunque el AUC no se mueva."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar, pk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
df,FE,y=cargar(); X=df[FE]
BASE=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=200,
          feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.,
          n_estimators=400,verbose=-1,n_jobs=1)
VAR=[('mcs=200 (produccion)',{}),
     ('mcs=500',{'min_child_samples':500}),
     ('mcs=1000',{'min_child_samples':1000}),
     ('mcs=2000',{'min_child_samples':2000}),
     ('mcs=500 + l2=200',{'min_child_samples':500,'lambda_l2':200.}),
     ('mcs=1000 + hojas 31',{'min_child_samples':1000,'num_leaves':31})]
t0=time.time()
for nom,ov in VAR:
    ps=[];aucs=[]
    for sp,sd in [(42,1),(7,2)]:
        skf=StratifiedKFold(5,shuffle=True,random_state=sp); oof=np.zeros(len(X))
        for tr,va in skf.split(X,y):
            m=lgb.LGBMClassifier(**{**BASE,**ov,'seed':sd,'bagging_seed':sd,'feature_fraction_seed':sd})
            m.fit(X.iloc[tr],y[tr]); oof[va]=m.predict_proba(X.iloc[va])[:,1]
        ps.append(pk(y,oof)); aucs.append(roc_auc_score(y,oof))
    print(f'{nom:24s} P@300={np.mean(ps):.4f} {np.round(ps,4)}  AUC={np.mean(aucs):.4f}  ({time.time()-t0:.0f}s)',flush=True)
