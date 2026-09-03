"""E2: curva de aprendizaje en la CABEZA. Cuanto gana P@300 al entrenar con mas datos?
Se compara K-fold con K=3,5,10,20 (entrena con 67/80/90/95% de las etiquetas).
El conjunto de evaluacion es SIEMPRE las 135.038 filas OOF -> comparacion pareada.
Pregunta practica: el pipeline en produccion entrena cada modelo con el 64% de las
etiquetas (5 folds sobre el 80% de desarrollo). Reajustar con el 100% ayuda?"""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar, pk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

df,FE,y=cargar(); X=df[FE]
P=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=200,
       feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.,
       n_estimators=400,verbose=-1,n_jobs=1)
t0=time.time()
for K in [3,5,10,20]:
    ps=[];aucs=[]
    for sd in [42,7]:
        skf=StratifiedKFold(K,shuffle=True,random_state=sd)
        oof=np.zeros(len(X))
        for tr,va in skf.split(X,y):
            m=lgb.LGBMClassifier(**{**P,'seed':11,'bagging_seed':11,'feature_fraction_seed':11})
            m.fit(X.iloc[tr],y[tr]); oof[va]=m.predict_proba(X.iloc[va])[:,1]
        ps.append(pk(y,oof)); aucs.append(roc_auc_score(y,oof))
    print(f'K={K:2d}  entrena con {100*(K-1)/K:4.1f}% de las etiquetas  '
          f'P@300={np.mean(ps):.4f} {np.round(ps,4)}  AUC={np.mean(aucs):.4f}  ({time.time()-t0:.0f}s)',flush=True)
