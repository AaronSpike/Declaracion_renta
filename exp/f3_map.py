import pandas as pd, numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score
D='/home/user/Declaracion_renta/data'
c=pd.read_parquet(f'{D}/customers.parquet'); d=pd.read_parquet(f'{D}/device_mobile_activity.parquet')
l=pd.read_parquet(f'{D}/fraud_labels_train.parquet')[['id_cuenta','is_fraud']]
u=pd.read_parquet(f'{D}/data_update_history.parquet')
CUT=pd.Timestamp(pd.read_parquet(f'{D}/accounts_to_score.parquet')['fecha_corte'].max())

d=d.copy(); d['ultimo_login']=pd.to_datetime(d['ultimo_login'],errors='coerce')
d['sin_cambio_dispositivo']=(d['dias_desde_ultimo_cambio']==999).astype(int)
d.loc[d['dias_desde_ultimo_cambio']==999,'dias_desde_ultimo_cambio']=np.nan
agg=d.groupby('id_cuenta').agg(num_dispositivos=('device_id','size'),num_tipos_dispositivo=('device_type','nunique'),
 is_rooted_or_jailbreak=('is_rooted_or_jailbreak','max'),is_emulator=('is_emulator','max'),
 num_cambios_dispositivo_12m=('num_cambios_dispositivo_12m','max'),dias_desde_ultimo_cambio=('dias_desde_ultimo_cambio','min'),
 sin_cambio_dispositivo=('sin_cambio_dispositivo','min'),num_sesiones_30d=('num_sesiones_30d','sum'),
 num_ciudades_acceso_30d=('num_ciudades_acceso_30d','max'),pct_accesos_fuera_ciudad=('pct_accesos_fuera_ciudad','max'),
 ultimo_login=('ultimo_login','max'),device_type=('device_type','first'),ip_country_ultimo=('ip_country_ultimo','first')).reset_index()
agg['ip_ultima_fuera_co']=(agg['ip_country_ultimo']!='CO').astype(int)
dias=(CUT-agg['ultimo_login']).dt.days
agg['dias_desde_ultimo_login']=dias.where(dias>=0,np.nan)
agg=agg.drop(columns=['ultimo_login'])

df=l.merge(c,on='id_cuenta').merge(u,on='id_cuenta').merge(agg,on='id_cuenta',how='left')
y=df.is_fraud.values

print('=== CHI2 de independencia por categorica (H0: sin relacion con fraude) ===')
for col in ['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal','device_type','ip_country_ultimo']:
    t=pd.crosstab(df[col],df.is_fraud)
    chi2,p,dof,_=chi2_contingency(t)
    print(f'{col:22s} niveles={df[col].nunique():3d}  chi2={chi2:7.2f} dof={dof:3d}  p={p:.4f}  {"SEÑAL" if p<0.05 else "ruido"}')

print('\n=== AUC univariado, TODAS las numericas actuales ===')
num=[x for x in df.columns if df[x].dtype.kind in 'ifb' and x not in ('is_fraud',)]
res=[]
for col in num:
    v=df[col].astype(float)
    if v.notna().sum()<1000 or v.nunique()<2: continue
    v=v.fillna(v.median()).values
    a=roc_auc_score(y,v); res.append((col,a,max(a,1-a)))
for col,a,ab in sorted(res,key=lambda r:-r[2]):
    print(f'{col:32s} AUC {a:.4f}   |dev| {abs(a-0.5):.4f}')
df.to_parquet('/home/user/Declaracion_renta/exp/base_train.parquet')
