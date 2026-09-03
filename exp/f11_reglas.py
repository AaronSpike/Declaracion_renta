"""Busca reglas de 2-3 banderas con alta precision. Descubre en DEV, valida en HOLDOUT.
Sin validacion fuera de muestra el maximo sobre ~220 reglas esta sesgado ~+3 sigma."""
import pandas as pd, numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
df=pd.read_parquet('exp/base_train.parquet'); fl=pd.read_parquet('exp/flags.parquet')
df=df.merge(fl,on='id_cuenta'); FLAGS=[c for c in fl.columns if c.startswith('f_')]
y=df.is_fraud.values
itr,ih=train_test_split(np.arange(len(df)),test_size=0.20,stratify=y,random_state=42)
dev=df.iloc[itr]; hol=df.iloc[ih]; yd=y[itr]; yh=y[ih]
print(f'dev {len(dev)} (base {yd.mean():.4f})   holdout {len(hol)} (base {yh.mean():.4f})\n')
res=[]
for r in (2,3):
    for combo in combinations(FLAGS,r):
        md=np.ones(len(dev),bool); mh=np.ones(len(hol),bool)
        for c in combo: md&=dev[c].values.astype(bool); mh&=hol[c].values.astype(bool)
        nd=md.sum()
        if nd<200: continue
        res.append((combo,nd,yd[md].mean(),mh.sum(),yh[mh].mean() if mh.sum()>0 else np.nan))
res.sort(key=lambda x:-x[2])
print(f'{"regla":48s} {"n_dev":>6s} {"tasa_dev":>9s} {"n_hol":>6s} {"tasa_hol":>9s}')
print('-'*86)
for combo,nd,td,nh,th in res[:18]:
    print(f'{"+".join(c[2:] for c in combo):48s} {nd:6d} {td:9.4f} {nh:6d} {th:9.4f}')
print(f'\ntasa base: dev {yd.mean():.4f}  holdout {yh.mean():.4f}')
# correlacion descubrimiento vs validacion: si es baja, las reglas son ruido
top=[r for r in res[:20]]
import scipy.stats as st
a=[r[2] for r in top]; b=[r[4] for r in top]
print(f'\ncorrelacion tasa_dev vs tasa_holdout en el top-20: {np.corrcoef(a,b)[0,1]:.3f}')
print(f'media tasa_dev top-20 {np.mean(a):.4f}  ->  media tasa_holdout {np.mean(b):.4f}  (encogimiento {np.mean(a)-np.mean(b):+.4f})')
