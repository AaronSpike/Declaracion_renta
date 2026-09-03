import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D='/home/user/Declaracion_renta/data'
c=pd.read_parquet(f'{D}/customers.parquet')
d=pd.read_parquet(f'{D}/device_mobile_activity.parquet')
l=pd.read_parquet(f'{D}/fraud_labels_train.parquet')[['id_cuenta','is_fraud']]
s=pd.read_parquet(f'{D}/accounts_to_score.parquet')
u=pd.read_parquet(f'{D}/data_update_history.parquet')

# --- grafo de dispositivos sobre las 180k cuentas (train+test), sin usar etiquetas
dg = d[['id_cuenta','device_id']].drop_duplicates()
cuentas_por_dev = dg.groupby('device_id')['id_cuenta'].nunique().rename('n_ctas_dev')
dg = dg.merge(cuentas_por_dev, on='device_id')

net = dg.groupby('id_cuenta').agg(
    max_ctas_por_dispositivo=('n_ctas_dev','max'),
    sum_ctas_por_dispositivo=('n_ctas_dev','sum'),
    mean_ctas_por_dispositivo=('n_ctas_dev','mean'),
).reset_index()
net['comparte_dispositivo']=(net['max_ctas_por_dispositivo']>1).astype(int)
# vecinos distintos = cuentas alcanzables a 1 salto
viz = dg.merge(dg, on='device_id')
viz = viz[viz.id_cuenta_x!=viz.id_cuenta_y]
nv = viz.groupby('id_cuenta_x')['id_cuenta_y'].nunique().rename('n_vecinos_dev')
net = net.merge(nv, left_on='id_cuenta', right_index=True, how='left')
net['n_vecinos_dev']=net['n_vecinos_dev'].fillna(0)

base = l.merge(net, on='id_cuenta', how='left')
y=base.is_fraud.values
print('tasa base', y.mean(), 'n', len(y))
print('\n--- AUC univariado features de red de dispositivos ---')
for col in ['max_ctas_por_dispositivo','sum_ctas_por_dispositivo','mean_ctas_por_dispositivo','comparte_dispositivo','n_vecinos_dev']:
    v=base[col].fillna(base[col].median()).values
    a=roc_auc_score(y,v)
    print(f'{col:34s} AUC {a:.4f}  (inv {1-a:.4f})')

# tasa de fraude por nivel de compartición
print('\n--- tasa de fraude por max_ctas_por_dispositivo ---')
print(base.groupby('max_ctas_por_dispositivo').agg(n=('is_fraud','size'), fraude=('is_fraud','mean')))
net.to_parquet('/home/user/Declaracion_renta/exp/net_dev.parquet')
