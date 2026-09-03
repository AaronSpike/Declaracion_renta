"""Celdas de banderas evaluadas OOF sobre las 135.038, P@300 (mismo percentil), 5 semillas.
Comparable directamente con el LightGBM medido igual."""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('exp/base_train.parquet'); fl=pd.read_parquet('exp/flags.parquet')
df=df.merge(fl,on='id_cuenta'); FLAGS=[c for c in fl.columns if c.startswith('f_')]
y=df.is_fraud.values; K=300
cell=df[FLAGS].astype(str).agg(''.join,axis=1).values
nb=df[FLAGS].sum(axis=1).values
print('celdas distintas:',len(np.unique(cell)))
def oof_te(keys,m,seed):
    oof=np.zeros(len(y)); skf=StratifiedKFold(5,shuffle=True,random_state=seed)
    for tr,va in skf.split(keys,y):
        p=y[tr].mean(); s=pd.Series(y[tr]).groupby(keys[tr]).agg(['size','mean'])
        sm=(s['mean']*s['size']+p*m)/(s['size']+m)
        oof[va]=pd.Series(keys[va]).map(sm).fillna(p).values
    return oof
rng=np.random.default_rng(0)
for m in (0,5,10,20,50):
    ps=[];aus=[]
    for sd in (42,7,2024,11,99):
        o=oof_te(cell,m,sd)+rng.normal(0,1e-9,len(y))
        ps.append(y[np.argsort(-o)[:K]].mean()); aus.append(roc_auc_score(y,o))
    ps=np.array(ps)
    print(f'celdas m={m:3d}  AUC {np.mean(aus):.4f}  P@300 {ps.mean():.4f} +/-{ps.std(ddof=1)/np.sqrt(5):.4f}  {np.round(ps,3)}')
# conteo simple como referencia
o=nb.astype(float)+rng.normal(0,1e-9,len(y))
print(f'\nconteo n_banderas       AUC {roc_auc_score(y,o):.4f}  P@300 {y[np.argsort(-o)[:K]].mean():.4f}')
