import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
D='/home/user/Declaracion_renta/data'; SEED=42; K=100; N_SCORE=44962
df=pd.read_parquet('/home/user/Declaracion_renta/exp/base_train.parquet')
fl=pd.read_parquet('/home/user/Declaracion_renta/exp/flags.parquet')
df=df.merge(fl,on='id_cuenta')
FLAGCOLS=[c for c in fl.columns if c.startswith('f_')]
df['n_banderas']=df[FLAGCOLS].sum(axis=1)
# derivadas del baseline actual
df['cambios_contacto_total']=df.num_cambios_telefono+df.num_cambios_email+df.num_cambios_direccion
df['ratio_cambios_contacto']=df.cambios_contacto_total/df.num_actualizaciones_12m.replace(0,np.nan)
df['kyc_vencido_1y']=(df.dias_desde_ultimo_kyc>365).astype(int)
df['sesiones_por_ciudad']=df.num_sesiones_30d/df.num_ciudades_acceso_30d.replace(0,np.nan)
df['cuenta_nueva_90d']=(df.antiguedad_dias<=90).astype(int)
df['sin_2fa']=1-df.tiene_2fa
df['intensidad_cambios']=df.cambios_contacto_total/(df.antiguedad_dias+1)
CATS=['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal','device_type','ip_country_ultimo']
for c in CATS: df[c]=df[c].astype('category')

BASE=[c for c in df.columns if c not in ['id_cuenta','id_cliente','is_fraud','n_banderas']+FLAGCOLS]
RUIDO=['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal','device_type',
       'ip_country_ultimo','edad','antiguedad_dias','dias_desde_ultimo_kyc','tiene_2fa','sin_2fa',
       'kyc_vencido_1y','cuenta_nueva_90d','ip_ultima_fuera_co','intensidad_cambios']
PODADO=[c for c in BASE if c not in RUIDO]
print('BASE',len(BASE),'| PODADO',len(PODADO),'->',PODADO)

y=df.is_fraud.values
idx=np.arange(len(df))
itr,ihold=train_test_split(idx,test_size=0.20,stratify=y,random_state=SEED)
P=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=200,
       feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.0,
       n_estimators=1200,verbose=-1,n_jobs=2)
rng=np.random.default_rng(SEED)
def pk(yt,yp,n_sim=400):
    n=len(yt); ns=min(N_SCORE,n); k=max(1,int(round(K*ns/N_SCORE))); o=np.empty(n_sim)
    for i in range(n_sim):
        s=rng.choice(n,size=ns,replace=False); o[i]=yt[s[np.argsort(-yp[s])[:k]]].mean()
    return o.mean(), o.std()/np.sqrt(n_sim)

def run(feats,name,seeds=(42,7,2024)):
    Xd=df.iloc[itr][feats]; yd=y[itr]; Xh=df.iloc[ihold][feats]; yh=y[ihold]
    ph=np.zeros(len(Xh))
    for sd in seeds:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd)
        for tr,va in skf.split(Xd,yd):
            m=lgb.LGBMClassifier(**P,seed=sd); m.fit(Xd.iloc[tr],yd[tr])
            ph+=m.predict_proba(Xh)[:,1]/(5*len(seeds))
    a=roc_auc_score(yh,ph); p,se=pk(yh,ph)
    print(f'{name:34s} AUC {a:.4f}   P@100 {p:.4f} +/-{se:.4f}')
    return ph

r={}
r['A base (reproduccion)']      = run(BASE,'A base (reproduccion)')
r['B base + n_banderas']        = run(BASE+['n_banderas'],'B base + n_banderas')
r['C podado']                   = run(PODADO,'C podado')
r['D podado + n_banderas']      = run(PODADO+['n_banderas'],'D podado + n_banderas')
r['E podado + banderas + cont'] = run(PODADO+FLAGCOLS+['n_banderas'],'E podado + banderas + conteo')
np.save('/home/user/Declaracion_renta/exp/preds.npy',np.array([r[k] for k in r]))
np.save('/home/user/Declaracion_renta/exp/yhold.npy',y[ihold])
