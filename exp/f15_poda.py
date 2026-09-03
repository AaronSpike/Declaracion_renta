"""Pregunta decisiva: podar las 17 features sin senal medida, mejora P@100?
Presupuesto reducido (lr 0,03 x 400 = mismo lr*n que 0,01 x 1200) para comparacion RELATIVA.
P@300 sobre OOF de 135.038 = mismo percentil del ranking que P@100 sobre 44.962."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

df = pd.read_parquet('/home/user/Declaracion_renta/exp/base_train.parquet')
fl = pd.read_parquet('/home/user/Declaracion_renta/exp/flags.parquet')
df = df.merge(fl, on='id_cuenta')
FLAGS = [c for c in fl.columns if c.startswith('f_')]
df['n_banderas'] = df[FLAGS].sum(axis=1)
df['cambios_contacto_total'] = df.num_cambios_telefono + df.num_cambios_email + df.num_cambios_direccion
df['ratio_cambios_contacto'] = df.cambios_contacto_total / df.num_actualizaciones_12m.replace(0, np.nan)
df['kyc_vencido_1y'] = (df.dias_desde_ultimo_kyc > 365).astype(int)
df['sesiones_por_ciudad'] = df.num_sesiones_30d / df.num_ciudades_acceso_30d.replace(0, np.nan)
df['cuenta_nueva_90d'] = (df.antiguedad_dias <= 90).astype(int)
df['sin_2fa'] = 1 - df.tiene_2fa
df['intensidad_cambios'] = df.cambios_contacto_total / (df.antiguedad_dias + 1)
CATS = ['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal','device_type','ip_country_ultimo']
for c in CATS:
    df[c] = df[c].astype('category')

BASE = [c for c in df.columns if c not in ['id_cuenta','id_cliente','is_fraud','n_banderas'] + FLAGS]
RUIDO = ['genero','ciudad','segmento','nivel_educativo','ocupacion','producto_principal','device_type',
         'ip_country_ultimo','edad','antiguedad_dias','dias_desde_ultimo_kyc','tiene_2fa','sin_2fa',
         'kyc_vencido_1y','cuenta_nueva_90d','ip_ultima_fuera_co','intensidad_cambios']
PODADO = [c for c in BASE if c not in RUIDO]
y = df.is_fraud.values
K = 300
P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_child_samples=200,
         feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0,
         n_estimators=400, verbose=-1, n_jobs=4)

def evalua(feats, name, seeds=(42, 7, 2024)):
    aucs, ps = [], []
    for sd in seeds:
        oof = np.zeros(len(df))
        skf = StratifiedKFold(5, shuffle=True, random_state=sd)
        for tr, va in skf.split(df, y):
            m = lgb.LGBMClassifier(**{**P, 'seed': sd})
            m.fit(df.iloc[tr][feats], y[tr])
            oof[va] = m.predict_proba(df.iloc[va][feats])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        ps.append(y[np.argsort(-oof)[:K]].mean())
    ps = np.array(ps)
    print(f'{name:32s} n_feat {len(feats):2d}  AUC {np.mean(aucs):.4f}  '
          f'P@300 {ps.mean():.4f} +/-{ps.std(ddof=1)/np.sqrt(len(ps)):.4f}  {np.round(ps,3)}', flush=True)
    return ps

evalua(BASE, 'A base (35 feats, produccion)')
evalua(PODADO, 'C podado (18 feats)')
evalua(PODADO + ['n_banderas'], 'D podado + n_banderas')
evalua(PODADO + FLAGS + ['n_banderas'], 'E podado + banderas')
