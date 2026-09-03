"""B. Sesgo de seleccion: elegir por P@100 en el mismo set donde se valida.
Entrena 12 configuraciones una sola vez, luego simula 400 veces
'selecciono en A / reporto en B' con A y B independientes."""
import numpy as np, pandas as pd, lightgbm as lgb, time, warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

PCT = 100/44962           # percentil de la metrica de premiacion
df, FE, y = cargar()
X = df[FE]
Xtr, Xpo, ytr, ypo = train_test_split(X, y, test_size=0.5, stratify=y, random_state=11)
print(f'train {len(ytr)}  pool {len(ypo)}  base {ypo.mean():.4f}', flush=True)

W = dict(learning_rate=0.01, num_leaves=15, min_child_samples=200, feature_fraction=0.6,
         bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, n_estimators=1200)
CFG = {}
for s in range(1,7):
    CFG[f'ganadora_s{s}'] = dict(W, seed=s, bagging_seed=s, feature_fraction_seed=s)
CFG['someros']    = dict(learning_rate=0.03,num_leaves=7,min_child_samples=300,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,n_estimators=800,seed=1)
CFG['lento_reg']  = dict(learning_rate=0.02,num_leaves=31,min_child_samples=100,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20.0,n_estimators=600,seed=1)
CFG['pos_w5']     = dict(learning_rate=0.02,num_leaves=31,min_child_samples=100,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20.0,n_estimators=600,scale_pos_weight=5.0,seed=1)
CFG['mcs1000']    = dict(W, min_child_samples=1000, seed=1)
CFG['sobreajuste']= dict(learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,n_estimators=2000,seed=1)
CFG['corta_400']  = dict(W, n_estimators=400, seed=1)

P = {}
for nm,pa in CFG.items():
    t=time.time()
    m=lgb.LGBMClassifier(objective='binary',verbose=-1,n_jobs=3,**pa)
    m.fit(Xtr,ytr)
    P[nm]=m.predict_proba(Xpo)[:,1]
    print(f'  {nm:16s} AUC_pool={roc_auc_score(ypo,P[nm]):.4f} '
          f'AP={average_precision_score(ypo,P[nm]):.4f} ({time.time()-t:.0f}s)', flush=True)
names=list(CFG); M=len(names)
np.save('/home/user/Declaracion_renta/val/preds_pool.npy', np.array([P[n] for n in names]))
np.save('/home/user/Declaracion_renta/val/y_pool.npy', ypo)
with open('/home/user/Declaracion_renta/val/names.txt','w') as f: f.write('\n'.join(names))
