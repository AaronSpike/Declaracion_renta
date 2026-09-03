import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('exp/base_train.parquet'); fl=pd.read_parquet('exp/flags.parquet')
df=df.merge(fl,on='id_cuenta'); FLAGS=[c for c in fl.columns if c.startswith('f_')]
df['nb']=df[FLAGS].sum(axis=1); y=df.is_fraud.values
itr,ih=train_test_split(np.arange(len(df)),test_size=0.20,stratify=y,random_state=42)
ph=np.load('data/pred_hold.npy'); yh=np.load('data/y_hold.npy')
assert (yh==y[ih]).all(), 'indices no coinciden'
print('AUC baseline holdout:',round(roc_auc_score(yh,ph),4))
h=df.iloc[ih].copy(); h['score']=ph
top=h.nlargest(100,'score')
print('\n=== composicion del TOP-100 del modelo actual (holdout) ===')
print('aciertos:',int(top.is_fraud.sum()),'/100')
print('n_banderas en el top-100:'); print(top.nb.value_counts().sort_index().to_string())
print('\n=== que hay realmente arriba por n_banderas (holdout completo) ===')
print(h.groupby('nb').is_fraud.agg(['size','mean']).to_string())
print('\n=== ranking por n_banderas puro (holdout): top-100 ===')
t2=h.sample(frac=1,random_state=0).nlargest(100,'nb')
print('aciertos por conteo de banderas:',int(t2.is_fraud.sum()),'/100')
print('\n=== score del modelo por nivel de n_banderas (percentil medio) ===')
h['pct_score']=h.score.rank(pct=True)
print(h.groupby('nb').pct_score.mean().round(3).to_string())
print('\ncuentas con nb>=5 en holdout:',int((h.nb>=5).sum()),' fraude:',round(h[h.nb>=5].is_fraud.mean(),4))
print('de esas, cuantas entran al top-100 del modelo:',int(top[top.nb>=5].shape[0]))
