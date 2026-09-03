"""E7: la comparacion que decide la receta de ENTREGA.

baseline_final.py entrega el promedio de 5 modelos, cada uno entrenado sobre un
fold del 80% de desarrollo -> cada miembro ve el 64% de las etiquetas.
Alternativa: promediar 5 modelos entrenados cada uno sobre el 100% de las
etiquetas disponibles, diversificados solo por la semilla de bagging.

Ambas recetas se evaluan con el MISMO OOF de 135.038 filas -> diferencia pareada.
"""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar, pk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
df,FE,y=cargar(); X=df[FE]
P=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=200,
       feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.,
       n_estimators=400,verbose=-1,n_jobs=1)
M=5
t0=time.time()
for sp in [42]:
    oofP=np.zeros(len(X)); oofN=np.zeros(len(X)); oof1=np.zeros(len(X))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=sp).split(X,y):
        # receta produccion: 5 submodelos sobre folds internos (80% de tr = 64% del total)
        acc=np.zeros(len(va))
        for i,(itr,_) in enumerate(StratifiedKFold(M,shuffle=True,random_state=42).split(X.iloc[tr],y[tr])):
            m=lgb.LGBMClassifier(**{**P,'seed':42,'bagging_seed':42,'feature_fraction_seed':42})
            m.fit(X.iloc[tr[itr]],y[tr[itr]]); acc+=m.predict_proba(X.iloc[va])[:,1]
        oofP[va]=acc/M
        # receta nueva: 5 modelos sobre TODO tr, diversificados por semilla
        acc=np.zeros(len(va))
        for sd in range(1,M+1):
            m=lgb.LGBMClassifier(**{**P,'seed':sd,'bagging_seed':sd,'feature_fraction_seed':sd})
            m.fit(X.iloc[tr],y[tr]); acc+=m.predict_proba(X.iloc[va])[:,1]
            if sd==1: oof1[va]=m.predict_proba(X.iloc[va])[:,1]
        oofN[va]=acc/M
    print(f'split{sp}  1 modelo sobre 80%: {pk(y,oof1):.4f} | '
          f'PRODUCCION 5x64%: {pk(y,oofP):.4f} | NUEVA 5x80%: {pk(y,oofN):.4f}  '
          f'| AUC {roc_auc_score(y,oofP):.4f} vs {roc_auc_score(y,oofN):.4f}  ({time.time()-t0:.0f}s)',flush=True)
