"""E5: reordenamiento en dos etapas. Etapa 1 sobre todo, etapa 2 solo sobre el
top 20% de candidatos. Protocolo honesto: los candidatos del bloque de
entrenamiento se definen con OOF interno, nunca con el modelo que los vio."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar, pk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
df,FE,y=cargar(); X=df[FE]
P=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=200,
       feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.,
       n_estimators=400,verbose=-1,n_jobs=1,seed=11,bagging_seed=11,feature_fraction_seed=11)
P2={**P,'num_leaves':7,'min_child_samples':50,'n_estimators':300}
FRAC=0.20
t0=time.time()
for sp in [42,7]:
    skf=StratifiedKFold(5,shuffle=True,random_state=sp)
    oof1=np.zeros(len(X)); oof2=np.zeros(len(X)); oofb=np.zeros(len(X))
    for tr,va in skf.split(X,y):
        # OOF interno para elegir candidatos dentro del bloque de entrenamiento
        inner=np.zeros(len(tr))
        for itr,iva in StratifiedKFold(3,shuffle=True,random_state=0).split(X.iloc[tr],y[tr]):
            mi=lgb.LGBMClassifier(**P); mi.fit(X.iloc[tr[itr]],y[tr[itr]])
            inner[iva]=mi.predict_proba(X.iloc[tr[iva]])[:,1]
        cand=tr[np.argsort(-inner)[:int(FRAC*len(tr))]]
        m1=lgb.LGBMClassifier(**P); m1.fit(X.iloc[tr],y[tr])
        s1=m1.predict_proba(X.iloc[va])[:,1]; oof1[va]=s1
        m2=lgb.LGBMClassifier(**P2); m2.fit(X.iloc[cand],y[cand])
        s2=m2.predict_proba(X.iloc[va])[:,1]
        # solo reordena dentro del top FRAC de la etapa 1; el resto conserva su orden
        umbral=np.quantile(s1,1-FRAC)
        nuevo=s1.copy(); msk=s1>=umbral
        r=(s2[msk].argsort().argsort()+1)/msk.sum()
        nuevo[msk]=umbral+r*(s1.max()-umbral+1e-9)
        oof2[va]=nuevo
        oofb[va]=np.where(msk,(s1-s1.min())/(s1.max()-s1.min())*0.5+ (s2-s2.min())/(s2.max()-s2.min()+1e-9)*0.5, -1)
    print(f'split{sp}  etapa1 {pk(y,oof1):.4f}  dos-etapas(reordena top20%) {pk(y,oof2):.4f}  '
          f'AUC1 {roc_auc_score(y,oof1):.4f}  ({time.time()-t0:.0f}s)',flush=True)
