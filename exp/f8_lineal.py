import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D='/home/user/Declaracion_renta/data'; SEED=42; K=100; N_SCORE=44962
df=pd.read_parquet('exp/base_train.parquet'); fl=pd.read_parquet('exp/flags.parquet')
df=df.merge(fl,on='id_cuenta'); FLAGS=[c for c in fl.columns if c.startswith('f_')]
df['n_banderas']=df[FLAGS].sum(axis=1)
y=df.is_fraud.values; idx=np.arange(len(df))
itr,ih=train_test_split(idx,test_size=0.20,stratify=y,random_state=SEED)
rng=np.random.default_rng(SEED)
def pk(yt,yp,n_sim=400):
    n=len(yt); ns=min(N_SCORE,n); k=max(1,int(round(K*ns/N_SCORE))); o=np.empty(n_sim)
    for i in range(n_sim):
        s=rng.choice(n,size=ns,replace=False); o[i]=yt[s[np.argsort(-yp[s])[:k]]].mean()
    return o.mean(),o.std()/np.sqrt(n_sim)
yh=y[ih]
def rep(name,ph):
    a=roc_auc_score(yh,ph); p,se=pk(yh,ph); print(f'{name:42s} AUC {a:.4f}  P@100 {p:.4f} +/-{se:.4f}'); return ph

# 1) conteo crudo de banderas (sin ajustar nada)
rep('conteo banderas (sin entrenar)', df.n_banderas.values[ih].astype(float)+rng.normal(0,1e-6,len(ih)))
# 2) logistica sobre las banderas, pesos estimados SOLO en dev
lr=LogisticRegression(C=1.0,max_iter=2000).fit(df.iloc[itr][FLAGS],y[itr])
rep('logistica sobre banderas', lr.predict_proba(df.iloc[ih][FLAGS])[:,1])
print('  pesos:',dict(zip(FLAGS,lr.coef_[0].round(3))))
# 3) logistica sobre banderas + numericas con senal
NUM=['num_actualizaciones_12m','num_cambios_telefono','num_cambios_email','num_cambios_direccion',
     'num_dispositivos','num_sesiones_30d','num_ciudades_acceso_30d','pct_accesos_fuera_ciudad']
Xn=df[NUM].fillna(0); Xn=(Xn-Xn.iloc[itr].mean())/Xn.iloc[itr].std()
XF=pd.concat([df[FLAGS],Xn],axis=1)
lr2=LogisticRegression(C=1.0,max_iter=3000).fit(XF.iloc[itr],y[itr])
rep('logistica banderas+numericas', lr2.predict_proba(XF.iloc[ih])[:,1])
# 4) banderas + todas sus interacciones de 2 vias
from itertools import combinations
XI=df[FLAGS].copy()
for a,b in combinations(FLAGS,2): XI[f'{a}*{b}']=df[a]*df[b]
lr3=LogisticRegression(C=0.5,max_iter=4000).fit(XI.iloc[itr],y[itr])
rep('logistica banderas + interacc 2 vias', lr3.predict_proba(XI.iloc[ih])[:,1])
np.save('exp/idx_hold.npy',ih); np.save('exp/idx_tr.npy',itr)
