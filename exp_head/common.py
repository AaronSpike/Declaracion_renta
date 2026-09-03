import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA='/home/user/Declaracion_renta/data'
K_EVAL=300           # mismo percentil que P@100 sobre 44.962 (100/44962*135038=300,3)
N_SCORE=44962

def cargar():
    df=pd.read_parquet('/home/user/Declaracion_renta/exp/base_train.parquet')
    df['cambios_contacto_total']=df.num_cambios_telefono+df.num_cambios_email+df.num_cambios_direccion
    df['ratio_cambios_contacto']=df.cambios_contacto_total/df.num_actualizaciones_12m.replace(0,np.nan)
    df['kyc_vencido_1y']=(df.dias_desde_ultimo_kyc>365).astype(int)
    df['sesiones_por_ciudad']=df.num_sesiones_30d/df.num_ciudades_acceso_30d.replace(0,np.nan)
    df['cuenta_nueva_90d']=(df.antiguedad_dias<=90).astype(int)
    df['sin_2fa']=1-df.tiene_2fa
    df['intensidad_cambios']=df.cambios_contacto_total/(df.antiguedad_dias+1)
    CATS=['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal','device_type','ip_country_ultimo']
    for c in CATS: df[c]=df[c].astype('category')
    FE=[c for c in df.columns if c not in ('id_cuenta','id_cliente','is_fraud')]
    return df, FE, df.is_fraud.values.astype(int)

def pk(y,s,k=K_EVAL):
    return y[np.argsort(-s,kind='stable')[:k]].mean()

def hits(y,s,k=K_EVAL):
    return int(y[np.argsort(-s,kind='stable')[:k]].sum())

def rankavg(mats):
    from scipy.stats import rankdata
    return np.mean([rankdata(m) for m in mats],axis=0)
