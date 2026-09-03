import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('exp/base_train.parquet'); y=df.is_fraud.values
print('id_cuenta ejemplos:',df.id_cuenta.head(3).tolist(),'| id_cliente:',df.id_cliente.head(3).tolist())
# orden de generacion
num=df.id_cuenta.str.extract(r'(\d+)')[0].astype(float)
print('AUC indice id_cuenta:',round(roc_auc_score(y,num),4))
d=pd.read_parquet('data/device_mobile_activity.parquet')
dn=d.groupby('id_cuenta').device_id.first().str.extract(r'(\d+)')[0].astype(float)
t=df[['id_cuenta']].merge(dn.rename('dev_idx'),left_on='id_cuenta',right_index=True,how='left')
print('AUC indice device_id:',round(roc_auc_score(y,t.dev_idx.fillna(t.dev_idx.median())),4))

print('\n=== INTERACCIONES Y RAZONES (AUC univariado) ===')
C={}
C['tel_x_ciudades']  = df.num_cambios_telefono*df.num_ciudades_acceso_30d
C['tel_x_fuera']     = df.num_cambios_telefono*df.pct_accesos_fuera_ciudad
C['act_x_disp']      = df.num_actualizaciones_12m*df.num_dispositivos
C['act_x_fuera']     = df.num_actualizaciones_12m*df.pct_accesos_fuera_ciudad
C['contacto_x_disp'] = (df.num_cambios_telefono+df.num_cambios_email+df.num_cambios_direccion)*df.num_dispositivos
C['tel_sobre_act']   = df.num_cambios_telefono/df.num_actualizaciones_12m.replace(0,np.nan)
C['sesiones_x_fuera']= df.num_sesiones_30d*df.pct_accesos_fuera_ciudad
C['fuera_x_ciudades']= df.pct_accesos_fuera_ciudad*df.num_ciudades_acceso_30d
C['max_contacto']    = df[['num_cambios_telefono','num_cambios_email','num_cambios_direccion']].max(axis=1)
C['n_tipos_contacto']= (df[['num_cambios_telefono','num_cambios_email','num_cambios_direccion']]>0).sum(axis=1)
C['act_menos_cont']  = df.num_actualizaciones_12m-(df.num_cambios_telefono+df.num_cambios_email+df.num_cambios_direccion)
C['contacto_total']  = df.num_cambios_telefono+df.num_cambios_email+df.num_cambios_direccion
for k,v in C.items():
    v=pd.Series(v).astype(float); v=v.fillna(v.median())
    print(f'{k:20s} AUC {roc_auc_score(y,v):.4f}')

print('\n=== DESVIACION RESPECTO AL GRUPO (segmento / ciudad) ===')
for g in ['segmento','ciudad']:
    for col in ['num_actualizaciones_12m','num_sesiones_30d','pct_accesos_fuera_ciudad']:
        z=(df[col]-df.groupby(g)[col].transform('mean'))/df.groupby(g)[col].transform('std')
        print(f'z({col}) por {g:9s} AUC {roc_auc_score(y,z.fillna(0)):.4f}   (crudo {roc_auc_score(y,df[col].fillna(df[col].median())):.4f})')
