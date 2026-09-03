import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D='data'
d=pd.read_parquet(f'{D}/device_mobile_activity.parquet')
l=pd.read_parquet(f'{D}/fraud_labels_train.parquet')[['id_cuenta','is_fraud']]
d['ultimo_login']=pd.to_datetime(d['ultimo_login'],errors='coerce')
n=d.groupby('id_cuenta').size()
multi=set(n[n>1].index)
print('cuentas con >1 dispositivo:',len(multi))
g=d.groupby('id_cuenta')
A=pd.DataFrame({
 'n_ip_distintas': g.ip_country_ultimo.nunique(),
 'n_tipos': g.device_type.nunique(),
 'rango_sesiones': g.num_sesiones_30d.max()-g.num_sesiones_30d.min(),
 'algun_root': g.is_rooted_or_jailbreak.max(),
 'todos_root': g.is_rooted_or_jailbreak.min(),
 'algun_emu': g.is_emulator.max(),
 'span_login_dias': (g.ultimo_login.max()-g.ultimo_login.min()).dt.days,
 'max_ciudades': g.num_ciudades_acceso_30d.max(),
 'min_ciudades': g.num_ciudades_acceso_30d.min(),
 'max_fuera': g.pct_accesos_fuera_ciudad.max(),
 'min_fuera': g.pct_accesos_fuera_ciudad.min(),
 'sum_sesiones': g.num_sesiones_30d.sum(),
 'max_sesiones': g.num_sesiones_30d.max(),
})
A['ip_heterogenea']=(A.n_ip_distintas>1).astype(int)
# el device_type / ip del login MAS RECIENTE (el pipeline actual usa 'first', arbitrario)
rec=d.sort_values('ultimo_login').groupby('id_cuenta').tail(1).set_index('id_cuenta')
A['ip_reciente_fuera_co']=(rec.ip_country_ultimo!='CO').astype(int)
first=d.groupby('id_cuenta').first()
A['ip_first_fuera_co']=(first.ip_country_ultimo!='CO').astype(int)
print('  coinciden first vs reciente (ip fuera CO):',(A.ip_reciente_fuera_co==A.ip_first_fuera_co).mean().round(4))
t=l.merge(A,left_on='id_cuenta',right_index=True,how='left'); y=t.is_fraud.values
print('\n=== AUC univariado (todas las cuentas) ===')
for c in A.columns:
    v=t[c].astype(float); v=v.fillna(v.median())
    if v.nunique()<2: continue
    print(f'{c:24s} AUC {roc_auc_score(y,v):.4f}')
sub=t[t.id_cuenta.isin(multi)]; ys=sub.is_fraud.values
print(f'\n=== SOLO las {len(sub)} cuentas multi-dispositivo (tasa {ys.mean():.4f}) ===')
for c in ['n_ip_distintas','n_tipos','rango_sesiones','span_login_dias','ip_heterogenea','todos_root','max_fuera','min_fuera']:
    v=sub[c].astype(float); v=v.fillna(v.median())
    if v.nunique()<2: continue
    print(f'{c:24s} AUC {roc_auc_score(ys,v):.4f}   media_fraude {v[ys==1].mean():.3f} vs {v[ys==0].mean():.3f}')
