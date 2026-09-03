import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
D='/home/user/Declaracion_renta/data'
c=pd.read_parquet(f'{D}/customers.parquet')
d=pd.read_parquet(f'{D}/device_mobile_activity.parquet')
l=pd.read_parquet(f'{D}/fraud_labels_train.parquet')[['id_cuenta','is_fraud']]
u=pd.read_parquet(f'{D}/data_update_history.parquet')
m=pd.read_parquet(f'{D}/merchants.parquet')
print(m.head()); print(m.dtypes); print('merchant_city n:',m.merchant_city.nunique())
print(m.merchant_country.value_counts().head())

df=l.merge(c,on='id_cuenta').merge(u,on='id_cuenta')
y=df.is_fraud.values
print('\n--- cardinalidad y tasa por categoria ---')
for col in ['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal']:
    g=df.groupby(col).is_fraud.agg(['size','mean'])
    print(f'\n{col}: {df[col].nunique()} niveles')
    print(g.sort_values('mean',ascending=False).to_string())
