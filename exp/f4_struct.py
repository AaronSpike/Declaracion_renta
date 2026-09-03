import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('/home/user/Declaracion_renta/exp/base_train.parquet')
y=df.is_fraud.values
u=['num_actualizaciones_12m','num_cambios_telefono','num_cambios_email','num_cambios_direccion']
print(df[u].describe().to_string())
print('\nsuma tel+email+dir vs num_actualizaciones_12m:')
s=df[['num_cambios_telefono','num_cambios_email','num_cambios_direccion']].sum(1)
print('  iguales:', (s==df.num_actualizaciones_12m).mean(), ' <= :', (s<=df.num_actualizaciones_12m).mean())
print('  corr:', np.corrcoef(s,df.num_actualizaciones_12m)[0,1].round(4))
print('\ncorrelacion entre las 4:'); print(df[u].corr().round(3).to_string())

print('\n=== tasa de fraude por num_actualizaciones_12m ===')
print(df.groupby('num_actualizaciones_12m').is_fraud.agg(['size','mean']).to_string())
print('\n=== tasa por num_cambios_telefono ===')
print(df.groupby('num_cambios_telefono').is_fraud.agg(['size','mean']).to_string())
print('\n=== tasa por pct_accesos_fuera_ciudad (deciles) ===')
q=pd.qcut(df.pct_accesos_fuera_ciudad,10,duplicates='drop')
print(df.groupby(q,observed=True).is_fraud.agg(['size','mean']).to_string())
print('\n=== tasa por num_ciudades_acceso_30d ===')
print(df.groupby('num_ciudades_acceso_30d').is_fraud.agg(['size','mean']).to_string())
