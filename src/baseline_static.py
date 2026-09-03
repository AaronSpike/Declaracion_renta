"""
Baseline SIN features transaccionales - Hackathon Asobancaria 2026.

Objetivo: (a) dejar un submission valido de respaldo desde temprano y
(b) medir cuanta senal aportan las fuentes NO transaccionales por si solas,
para poder atribuir despues la ganancia real de las transacciones.

Metrica oficial de negocio: Precision@100 sobre 44.962 cuentas.
Se simula en validacion cruzada muestreando repetidamente subconjuntos
out-of-fold del mismo tamano que el set de evaluacion.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "/home/user/Declaracion_renta/data"
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Carga y construccion de la tabla de features (una fila por cuenta)
# ---------------------------------------------------------------------------

labels = pd.read_parquet(f"{DATA}/fraud_labels_train.parquet")
score = pd.read_parquet(f"{DATA}/accounts_to_score.parquet")
customers = pd.read_parquet(f"{DATA}/customers.parquet")
updates = pd.read_parquet(f"{DATA}/data_update_history.parquet")
device = pd.read_parquet(f"{DATA}/device_mobile_activity.parquet")

CUTOFF = pd.Timestamp(score["fecha_corte"].max())
print(f"Fecha de corte comun: {CUTOFF.date()}")

# device tiene 2-3 filas para 4.019 cuentas -> agregar antes de unir
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
# El snapshot de dispositivos puede traer logins posteriores al corte:
# se marca explicitamente como hallazgo de gobernanza de datos.
dev["flag_login_post_corte"] = (dev["ultimo_login"] > CUTOFF).astype(int)
dev = dev.drop(columns=["ultimo_login"])
assert dev["id_cuenta"].is_unique

master = pd.concat([
    labels[["id_cuenta", "is_fraud"]].assign(split="train"),
    score[["id_cuenta"]].assign(is_fraud=np.nan, split="score"),
], ignore_index=True)

df = (master
      .merge(customers, on="id_cuenta", how="left")
      .merge(updates, on="id_cuenta", how="left")
      .merge(dev, on="id_cuenta", how="left"))
assert len(df) == 180000, f"joins duplicaron filas: {len(df)}"

# Features derivadas de los snapshots
df["ratio_cambios_contacto"] = (
    df["num_cambios_telefono"] + df["num_cambios_email"] + df["num_cambios_direccion"]
) / df["num_actualizaciones_12m"].replace(0, np.nan)
df["cambios_contacto_total"] = (
    df["num_cambios_telefono"] + df["num_cambios_email"] + df["num_cambios_direccion"]
)
df["kyc_vencido_1y"] = (df["dias_desde_ultimo_kyc"] > 365).astype(int)
df["sesiones_por_ciudad"] = df["num_sesiones_30d"] / df["num_ciudades_acceso_30d"].replace(0, np.nan)
df["antiguedad_anios"] = df["antiguedad_dias"] / 365.25
df["cuenta_nueva_90d"] = (df["antiguedad_dias"] <= 90).astype(int)
df["sin_2fa"] = 1 - df["tiene_2fa"]

CATS = ["genero", "ciudad", "segmento", "nivel_educativo", "ocupacion",
        "producto_principal", "device_type", "ip_country_ultimo"]
for c in CATS:
    df[c] = df[c].astype("category")

DROP = ["id_cuenta", "id_cliente", "is_fraud", "split"]
FEATURES = [c for c in df.columns if c not in DROP]
print(f"{len(FEATURES)} features (solo fuentes NO transaccionales)")

train = df[df.split == "train"].reset_index(drop=True)
test = df[df.split == "score"].reset_index(drop=True)
y = train["is_fraud"].astype(int).values
X = train[FEATURES]
X_test = test[FEATURES]

# ---------------------------------------------------------------------------
# 2. Validacion cruzada + simulacion de Precision@100
# ---------------------------------------------------------------------------

N_SCORE = len(test)          # 44.962 cuentas reales a calificar
K = 100                      # capacidad operativa: 100 alertas/dia

def precision_at_k_simulado(y_true, y_pred, n_subset, k=K, n_sim=300, rng=rng):
    """Simula la metrica oficial: se muestrean subconjuntos del tamano real del
    set de evaluacion y se mide la precision en el top-k de cada uno."""
    idx = np.arange(len(y_true))
    out = []
    for _ in range(n_sim):
        s = rng.choice(idx, size=min(n_subset, len(idx)), replace=False)
        top = s[np.argsort(-y_pred[s])[:k]]
        out.append(y_true[top].mean())
    return float(np.mean(out)), float(np.std(out))

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, n_estimators=2000, verbose=-1, seed=SEED, n_jobs=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
best_iters = []

for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X.iloc[tr], y[tr],
              eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
    best_iters.append(model.best_iteration_)
    print(f"  fold {fold}: AUC={roc_auc_score(y[va], oof[va]):.4f} iters={model.best_iteration_}")

auc = roc_auc_score(y, oof)
ap = average_precision_score(y, oof)
p100, p100_sd = precision_at_k_simulado(y, oof, N_SCORE)

print("\n" + "=" * 60)
print("BASELINE (sin transacciones)")
print("=" * 60)
print(f"ROC-AUC (OOF)            : {auc:.4f}")
print(f"Average Precision (OOF)  : {ap:.4f}")
print(f"Precision@100 simulada   : {p100:.4f} (+/- {p100_sd:.4f})")
print(f"  -> aciertos esperados en 100 alertas: {p100*100:.1f}")
print(f"Tasa base de fraude      : {y.mean():.4f}  (lift: {p100/y.mean():.1f}x)")

imp = pd.DataFrame({"feature": FEATURES,
                    "gain": model.booster_.feature_importance("gain")}
                   ).sort_values("gain", ascending=False)
print("\nTop 15 features por ganancia:")
print(imp.head(15).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Submission de respaldo
# ---------------------------------------------------------------------------

sub = pd.DataFrame({"id_cuenta": test["id_cuenta"], "score_probabilidad_fraude": test_pred})
sample = pd.read_csv(f"{DATA}/sample_submission.csv")
sub = sample[["id_cuenta"]].merge(sub, on="id_cuenta", how="left")
assert len(sub) == len(sample) and sub["score_probabilidad_fraude"].notna().all()
sub.to_csv("/home/user/Declaracion_renta/submissions/submission_baseline_static.csv", index=False)
print(f"\nSubmission guardado: submissions/submission_baseline_static.csv ({len(sub)} filas)")

np.save("/home/user/Declaracion_renta/data/oof_baseline.npy", oof)
imp.to_csv("/home/user/Declaracion_renta/data/importancia_baseline.csv", index=False)
