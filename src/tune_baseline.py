"""
Afinamiento del baseline optimizando Precision@100, no ROC-AUC.

Motivacion: la metrica de premiacion solo mira las 100 cuentas de la cabeza del
ranking. Detener el entrenamiento por AUC optimiza el orden global, que no es lo
mismo. Aqui se compara un conjunto de configuraciones midiendo directamente la
metrica de negocio, estimada por remuestreo para no leer ruido como senal.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "/home/user/Declaracion_renta/data"
SEED = 42
K = 100
N_SCORE = 44962

# ---------------------------------------------------------------------------
# Datos (identico al baseline)
# ---------------------------------------------------------------------------
labels = pd.read_parquet(f"{DATA}/fraud_labels_train.parquet")
score = pd.read_parquet(f"{DATA}/accounts_to_score.parquet")
customers = pd.read_parquet(f"{DATA}/customers.parquet")
updates = pd.read_parquet(f"{DATA}/data_update_history.parquet")
device = pd.read_parquet(f"{DATA}/device_mobile_activity.parquet")
CUTOFF = pd.Timestamp(score["fecha_corte"].max())

device["ultimo_login"] = pd.to_datetime(device["ultimo_login"], errors="coerce")
dev = device.groupby("id_cuenta").agg(
    num_dispositivos=("device_id", "size"),
    num_tipos_dispositivo=("device_type", "nunique"),
    is_rooted_or_jailbreak=("is_rooted_or_jailbreak", "max"),
    is_emulator=("is_emulator", "max"),
    num_cambios_dispositivo_12m=("num_cambios_dispositivo_12m", "max"),
    dias_desde_ultimo_cambio=("dias_desde_ultimo_cambio", "min"),
    num_sesiones_30d=("num_sesiones_30d", "sum"),
    num_ciudades_acceso_30d=("num_ciudades_acceso_30d", "max"),
    pct_accesos_fuera_ciudad=("pct_accesos_fuera_ciudad", "max"),
    ultimo_login=("ultimo_login", "max"),
    device_type=("device_type", "first"),
    ip_country_ultimo=("ip_country_ultimo", "first"),
).reset_index()
dev["ip_ultima_fuera_co"] = (dev["ip_country_ultimo"] != "CO").astype(int)
dev["dias_desde_ultimo_login"] = (CUTOFF - dev["ultimo_login"]).dt.days
dev["flag_login_post_corte"] = (dev["ultimo_login"] > CUTOFF).astype(int)
dev = dev.drop(columns=["ultimo_login"])

df = (labels[["id_cuenta", "is_fraud"]]
      .merge(customers, on="id_cuenta", how="left")
      .merge(updates, on="id_cuenta", how="left")
      .merge(dev, on="id_cuenta", how="left"))

df["cambios_contacto_total"] = (df["num_cambios_telefono"] + df["num_cambios_email"]
                                + df["num_cambios_direccion"])
df["ratio_cambios_contacto"] = df["cambios_contacto_total"] / df["num_actualizaciones_12m"].replace(0, np.nan)
df["kyc_vencido_1y"] = (df["dias_desde_ultimo_kyc"] > 365).astype(int)
df["sesiones_por_ciudad"] = df["num_sesiones_30d"] / df["num_ciudades_acceso_30d"].replace(0, np.nan)
df["cuenta_nueva_90d"] = (df["antiguedad_dias"] <= 90).astype(int)
df["sin_2fa"] = 1 - df["tiene_2fa"]
df["intensidad_cambios"] = df["cambios_contacto_total"] / (df["antiguedad_dias"] + 1)

for c in ["genero", "ciudad", "segmento", "nivel_educativo", "ocupacion",
          "producto_principal", "device_type", "ip_country_ultimo"]:
    df[c] = df[c].astype("category")

FEATURES = [c for c in df.columns if c not in ("id_cuenta", "id_cliente", "is_fraud")]
X, y = df[FEATURES], df["is_fraud"].astype(int).values
print(f"{len(FEATURES)} features, {len(X)} filas, tasa base {y.mean():.4f}")

rng = np.random.default_rng(SEED)


def p_at_k(y_true, y_pred, n_sim=500):
    """Precision@100 estimada por remuestreo al tamano real del set a calificar."""
    idx = np.arange(len(y_true))
    out = np.empty(n_sim)
    for i in range(n_sim):
        s = rng.choice(idx, size=N_SCORE, replace=False)
        out[i] = y_true[s[np.argsort(-y_pred[s])[:K]]].mean()
    return out.mean(), out.std() / np.sqrt(n_sim)   # media y error estandar


CONFIGS = {
    "actual (early stop AUC)": dict(
        learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, n_estimators=2000),
    "lento + regularizado": dict(
        learning_rate=0.02, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0, n_estimators=600),
    "muy lento + hojas pocas": dict(
        learning_rate=0.01, num_leaves=15, min_child_samples=200, feature_fraction=0.6,
        bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, n_estimators=1200),
    "arboles someros": dict(
        learning_rate=0.03, num_leaves=7, min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, n_estimators=800),
    "peso positivo x5": dict(
        learning_rate=0.02, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0, n_estimators=600,
        scale_pos_weight=5.0),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
resultados = []

for nombre, params in CONFIGS.items():
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        # sin early stopping: el numero de arboles es parte de la configuracion,
        # asi se compara la metrica de negocio y no el criterio de parada
        m = lgb.LGBMClassifier(objective="binary", verbose=-1, seed=SEED, n_jobs=-1, **params)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    p, se = p_at_k(y, oof)
    resultados.append({"config": nombre, "auc": auc, "ap": ap, "p100": p, "se": se,
                       "top100_crudo": y[np.argsort(-oof)[:K]].mean()})
    print(f"{nombre:28s} AUC={auc:.4f}  AP={ap:.4f}  P@100={p:.4f}+/-{se:.4f}  "
          f"(top100 crudo {resultados[-1]['top100_crudo']:.3f})")

r = pd.DataFrame(resultados).sort_values("p100", ascending=False)
print("\n" + "=" * 78)
print("ORDENADO POR LA METRICA DE PREMIACION (Precision@100)")
print("=" * 78)
print(r.to_string(index=False))

mejor = r.iloc[0]
base = r[r.config == "actual (early stop AUC)"].iloc[0]
dif = mejor["p100"] - base["p100"]
sig = abs(dif) > 2 * np.sqrt(mejor["se"] ** 2 + base["se"] ** 2)
print(f"\nMejor: '{mejor['config']}'  P@100={mejor['p100']:.4f}")
print(f"Diferencia frente a la configuracion actual: {dif:+.4f} "
      f"({'significativa' if sig else 'DENTRO DEL RUIDO: no se puede afirmar mejora'})")
r.to_csv(f"{DATA}/tuning_baseline.csv", index=False)
