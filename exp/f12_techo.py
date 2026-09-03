"""Techo de las fuentes estaticas. Estima P(fraude|celda) en DEV y rankea HOLDOUT.
k=60 sobre 27.008 = mismo percentil (0,222%) que P@100 sobre 44.962 -> comparable con el 23,33% reportado."""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('exp/base_train.parquet'); fl=pd.read_parquet('exp/flags.parquet')
df=df.merge(fl,on='id_cuenta'); FLAGS=[c for c in fl.columns if c.startswith('f_')]
y=df.is_fraud.values
itr,ih=train_test_split(np.arange(len(df)),test_size=0.20,stratify=y,random_state=42)
yd,yh=y[itr],y[ih]; K=60
cell=df[FLAGS].astype(str).agg(''.join,axis=1)
dv=pd.DataFrame({'c':cell.values[itr],'y':yd}); hl=pd.DataFrame({'c':cell.values[ih],'y':yh})
# suavizado bayesiano hacia la tasa base
prior=yd.mean()
for m in (0,5,20,50):
    g=dv.groupby('c').y.agg(['size','mean'])
    g['sm']=(g['mean']*g['size']+prior*m)/(g['size']+m)
    sc=hl.c.map(g.sm).fillna(prior).values
    rng=np.random.default_rng(0); sc=sc+rng.normal(0,1e-9,len(sc))
    p=yh[np.argsort(-sc)[:K]].mean()
    print(f'celdas de banderas, suavizado m={m:3d}:  P@60 holdout = {p:.4f}   AUC {roc_auc_score(yh,sc):.4f}')
print(f'\nbaseline actual (reportado):     P@60 holdout = 0.2333   AUC 0.6195')
print(f'tasa base: {yh.mean():.4f}')
# techo teorico si conocieramos la celda perfecta (usando holdout, TRAMPA, solo como cota superior)
g2=hl.groupby('c').y.agg(['size','mean']); g2=g2[g2['size']>=5].sort_values('mean',ascending=False)
acum=0; n=0
for _,r in g2.iterrows():
    take=min(K-n,r['size']); acum+=take*r['mean']; n+=take
    if n>=K: break
print(f'\ncota superior con celdas ajustadas EN el holdout (trampa, no alcanzable): P@60 = {acum/K:.4f}')
