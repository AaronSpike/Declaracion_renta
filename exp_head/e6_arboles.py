"""E6: numero de arboles como palanca de CABEZA, a la tasa de aprendizaje de
produccion (lr=0,01). El AUC sigue subiendo casi hasta el final; la pregunta es
si P@300 (=P@100 sobre 44.962) tiene su maximo antes de los 1200 arboles."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar, pk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
df,FE,y=cargar(); X=df[FE]
P=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=200,
       feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.,
       n_estimators=1600,verbose=-1,n_jobs=2)
ITERS=[300,500,700,900,1100,1300,1600]
t0=time.time()
acum={t:[] for t in ITERS}; aucs={t:[] for t in ITERS}
for sp,sd in [(42,1),(42,2),(7,3)]:
    skf=StratifiedKFold(5,shuffle=True,random_state=sp)
    oof={t:np.zeros(len(X)) for t in ITERS}
    for tr,va in skf.split(X,y):
        m=lgb.LGBMClassifier(**{**P,'seed':sd,'bagging_seed':sd,'feature_fraction_seed':sd})
        m.fit(X.iloc[tr],y[tr])
        for t in ITERS: oof[t][va]=m.predict_proba(X.iloc[va],num_iteration=t)[:,1]
    for t in ITERS:
        acum[t].append(pk(y,oof[t])); aucs[t].append(roc_auc_score(y,oof[t]))
    print(f'  corrida sp{sp} sd{sd} lista ({time.time()-t0:.0f}s)  '+
          ' '.join(f'{t}:{acum[t][-1]:.4f}' for t in ITERS),flush=True)
print('\narboles   P@300 (media 3 corridas)   AUC')
for t in ITERS:
    print(f'{t:5d}     {np.mean(acum[t]):.4f}  {np.round(acum[t],4)}   {np.mean(aucs[t]):.4f}')
