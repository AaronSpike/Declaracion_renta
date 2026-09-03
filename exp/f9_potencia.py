"""Evaluacion de alta potencia: OOF sobre las 135.038 cuentas, P@300.
P@300 sobre 135.038 = mismo percentil del ranking (0,222%) que P@100 sobre 44.962,
pero con 300 casos en vez de 60 -> ~2,2x menos ruido. Promediado sobre 3 semillas."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
SEED=42; K300=300
df=pd.read_parquet('exp/base_train.parquet'); fl=pd.read_parquet('exp/flags.parquet')
df=df.merge(fl,on='id_cuenta'); FLAGS=[c for c in fl.columns if c.startswith('f_')]
df['n_banderas']=df[FLAGS].sum(axis=1)
df['cambios_contacto_total']=df.num_cambios_telefono+df.num_cambios_email+df.num_cambios_direccion
df['ratio_cambios_contacto']=df.cambios_contacto_total/df.num_actualizaciones_12m.replace(0,np.nan)
df['kyc_vencido_1y']=(df.dias_desde_ultimo_kyc>365).astype(int)
df['sesiones_por_ciudad']=df.num_sesiones_30d/df.num_ciudades_acceso_30d.replace(0,np.nan)
df['cuenta_nueva_90d']=(df.antiguedad_dias<=90).astype(int)
df['sin_2fa']=1-df.tiene_2fa
df['intensidad_cambios']=df.cambios_contacto_total/(df.antiguedad_dias+1)
df['act_x_disp']=df.num_actualizaciones_12m*df.num_dispositivos
CATS=['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal','device_type','ip_country_ultimo']
for c in CATS: df[c]=df[c].astype('category')
BASE=[c for c in df.columns if c not in ['id_cuenta','id_cliente','is_fraud','n_banderas','act_x_disp']+FLAGS]
RUIDO=['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal','device_type',
 'ip_country_ultimo','edad','antiguedad_dias','dias_desde_ultimo_kyc','tiene_2fa','sin_2fa',
 'kyc_vencido_1y','cuenta_nueva_90d','ip_ultima_fuera_co','intensidad_cambios']
PODADO=[c for c in BASE if c not in RUIDO]
y=df.is_fraud.values
P=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=200,feature_fraction=0.6,
       bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.0,n_estimators=1200,verbose=-1,n_jobs=2)
def evalua(feats,name,params=None,seeds=(42,7,2024)):
    pr=params or P; aucs=[]; ps=[]
    for sd in seeds:
        oof=np.zeros(len(df)); skf=StratifiedKFold(5,shuffle=True,random_state=sd)
        for tr,va in skf.split(df,y):
            m=lgb.LGBMClassifier(**{**pr,'seed':sd}); m.fit(df.iloc[tr][feats],y[tr])
            oof[va]=m.predict_proba(df.iloc[va][feats])[:,1]
        aucs.append(roc_auc_score(y,oof)); ps.append(y[np.argsort(-oof)[:K300]].mean())
    ps=np.array(ps)
    print(f'{name:36s} AUC {np.mean(aucs):.4f}  P@300 {ps.mean():.4f} +/-{ps.std(ddof=1)/np.sqrt(len(ps)):.4f}  {np.round(ps,4)}',flush=True)
    return ps.mean()
print('n=',len(df),'  P@300 == mismo percentil que P@100 sobre 44.962\n')
evalua(BASE,'A base (actual en produccion)')
evalua(PODADO,'C podado (sin ruido medido)')
evalua(PODADO+['n_banderas'],'D podado + n_banderas')
evalua(PODADO+FLAGS+['n_banderas'],'E podado + banderas + conteo')
evalua(BASE+FLAGS+['n_banderas'],'F base + banderas + conteo')
