"""E8: tercera particion independiente para las dos variantes de regularizacion
que subieron en E4. Sirve para ver si el signo se repite 3 de 3."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar, pk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
df,FE,y=cargar(); X=df[FE]
B=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=200,
       feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.,
       n_estimators=400,verbose=-1,n_jobs=2)
V=[('mcs=200 l2=50 (produccion)',{}),('mcs=500 l2=200',{'min_child_samples':500,'lambda_l2':200.}),
   ('mcs=1000 l2=50',{'min_child_samples':1000})]
t0=time.time()
for nom,ov in V:
    ps=[]
    for sp,sd in [(2024,3),(11,4)]:
        oof=np.zeros(len(X))
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=sp).split(X,y):
            m=lgb.LGBMClassifier(**{**B,**ov,'seed':sd,'bagging_seed':sd,'feature_fraction_seed':sd})
            m.fit(X.iloc[tr],y[tr]); oof[va]=m.predict_proba(X.iloc[va])[:,1]
        ps.append(pk(y,oof))
    print(f'{nom:28s} P@300={np.mean(ps):.4f} {np.round(ps,4)}  ({time.time()-t0:.0f}s)',flush=True)
