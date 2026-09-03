import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('/home/user/Declaracion_renta/exp/base_train.parquet'); y=df.is_fraud.values
for col in ['num_cambios_email','num_cambios_direccion','num_cambios_dispositivo_12m','num_dispositivos','num_tipos_dispositivo','is_emulator','is_rooted_or_jailbreak','sin_cambio_dispositivo']:
    print(f'--- {col}'); print(df.groupby(col).is_fraud.agg(['size','mean']).to_string())

# banderas con cortes ESTRUCTURALES (>=2 en conteos) o percentil de la distribucion, no ajustados a la etiqueta
F={}
F['f_tel2']   = (df.num_cambios_telefono>=2).astype(int)
F['f_mail2']  = (df.num_cambios_email>=2).astype(int)
F['f_dir2']   = (df.num_cambios_direccion>=2).astype(int)
F['f_act3']   = (df.num_actualizaciones_12m>=3).astype(int)
F['f_ciud2']  = (df.num_ciudades_acceso_30d>=2).astype(int)
F['f_fuera']  = (df.pct_accesos_fuera_ciudad>df.pct_accesos_fuera_ciudad.quantile(0.90)).astype(int)
F['f_disp2']  = (df.num_dispositivos>=2).astype(int)
F['f_cdisp2'] = (df.num_cambios_dispositivo_12m>=2).astype(int)
F['f_emu']    = df.is_emulator.astype(int)
F['f_root']   = df.is_rooted_or_jailbreak.astype(int)
F['f_camdev'] = (1-df.sin_cambio_dispositivo).astype(int)
Fd=pd.DataFrame(F)
print('\n=== AUC de cada bandera y prevalencia ===')
for k in Fd: print(f'{k:10s} prev {Fd[k].mean():.3f}  AUC {roc_auc_score(y,Fd[k]):.4f}  tasa_1 {y[Fd[k]==1].mean():.4f}')

n=Fd.sum(1)
print(f'\n### CONTEO DE BANDERAS: AUC univariado = {roc_auc_score(y,n):.4f}')
print(df.assign(nb=n).groupby('nb').is_fraud.agg(['size','mean']).to_string())
# referencia: mejor variable individual
print(f'\nreferencia num_actualizaciones_12m AUC = {roc_auc_score(y,df.num_actualizaciones_12m):.4f}')
# P@100 univariado del conteo (con desempate aleatorio)
rng=np.random.default_rng(0)
def pk(sc,y,k=100,n_sim=300):
    o=[]
    for _ in range(n_sim):
        s=sc+rng.normal(0,1e-9,len(sc))
        o.append(y[np.argsort(-s)[:k]].mean())
    return np.mean(o)
print(f'P@100 (todo train, k=100) conteo banderas: {pk(n.values.astype(float),y):.4f}')
print(f'P@100 num_actualizaciones_12m: {pk(df.num_actualizaciones_12m.values.astype(float),y):.4f}')
Fd.assign(id_cuenta=df.id_cuenta.values).to_parquet('/home/user/Declaracion_renta/exp/flags.parquet')
