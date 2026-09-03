"""
Pipeline principal - Hackathon Asobancaria 2026 (Banco AuditPlus).

Consume account_features.parquet (generado en Kaggle por
kaggle_notebook/build_account_features.py) y produce el submission final.

Estrategia frente a la metrica oficial (Precision@100):
- Lo unico que importa es el ORDEN en la cabeza del ranking, no la calibracion.
- Se combinan tres vistas del problema por promedio de rangos:
    A. LightGBM binario (is_fraud).
    B. LightGBM multiclase sobre tipo_fraude -> score = 1 - P(none).
       Cada tipologia (ATO, apertura fraudulenta, mula, suplantacion) tiene
       firma distinta; modelar las cabezas por separado ordena mejor el tope.
    C. LightGBM binario con submuestreo del negativo (enfasis en la frontera).
- Multi-semilla para reducir la varianza del top-100, que con solo 100 casos
  es alta por construccion.

Uso:  python3 src/train_full.py <ruta_account_features.parquet>
"""

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "/home/user/Declaracion_renta/data"
OUT = "/home/user/Declaracion_renta/submissions"
SEED = 42
N_SEEDS = 3
K = 100

FEATURES_PATH = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/account_features.parquet"

# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------

df = pd.read_parquet(FEATURES_PATH)
print(f"account_features: {df.shape}")
assert len(df) == 180000, f"Se esperaban 180000 filas, hay {len(df)}"

# ---------------------------------------------------------------------------
# 2. Features derivadas adicionales (baratas, sobre la tabla ya agregada)
# ---------------------------------------------------------------------------

def add_derived(d):
    d = d.copy()
    eps = 1e-9
    if {"monto_total", "num_tx_total"} <= set(d.columns):
        d["ticket_promedio"] = d["monto_total"] / (d["num_tx_total"] + eps)
    if {"num_tx_total", "dias_antiguedad_transaccional"} <= set(d.columns):
        d["tx_por_dia"] = d["num_tx_total"] / (d["dias_antiguedad_transaccional"] + 1)
    if {"num_tx_30d", "num_tx_total", "dias_antiguedad_transaccional"} <= set(d.columns):
        # aceleracion: actividad reciente vs. ritmo historico de la cuenta
        ritmo_hist = d["num_tx_total"] / (d["dias_antiguedad_transaccional"] + 1)
        d["aceleracion_30d"] = (d["num_tx_30d"] / 30.0) / (ritmo_hist + eps)
    if {"monto_30d", "monto_total"} <= set(d.columns):
        d["concentracion_monto_reciente"] = d["monto_30d"] / (d["monto_total"] + eps)
    if {"num_sesiones_30d", "num_tx_30d"} <= set(d.columns):
        # muchas sesiones sin transaccionar: patron de reconocimiento/ATO
        d["sesiones_por_tx_30d"] = d["num_sesiones_30d"] / (d["num_tx_30d"] + 1)
    if {"cambios_contacto_total", "antiguedad_dias"} <= set(d.columns):
        d["intensidad_cambios_contacto"] = d["cambios_contacto_total"] / (d["antiguedad_dias"] + 1)
    if {"num_cambios_telefono", "num_cambios_email", "num_cambios_direccion"} <= set(d.columns):
        d["cambios_contacto_total"] = (d["num_cambios_telefono"].fillna(0)
                                       + d["num_cambios_email"].fillna(0)
                                       + d["num_cambios_direccion"].fillna(0))
    if {"dias_desde_ultimo_kyc"} <= set(d.columns):
        d["kyc_vencido_1y"] = (d["dias_desde_ultimo_kyc"] > 365).astype(int)
    if {"antiguedad_dias"} <= set(d.columns):
        d["cuenta_nueva_90d"] = (d["antiguedad_dias"] <= 90).astype(int)
    if {"tiene_2fa"} <= set(d.columns):
        d["sin_2fa"] = 1 - d["tiene_2fa"]
    if {"num_sesiones_30d", "num_ciudades_acceso_30d"} <= set(d.columns):
        d["sesiones_por_ciudad"] = d["num_sesiones_30d"] / (d["num_ciudades_acceso_30d"] + eps)
    # cuentas sin actividad transaccional en la ventana: marca explicita
    if "num_tx_total" in d.columns:
        d["sin_transacciones"] = d["num_tx_total"].isna().astype(int)
    return d

df = add_derived(df)

DROP = ["id_cuenta", "id_cliente", "is_fraud", "split", "tipo_fraude",
        "fecha_confirmacion_fraude", "fecha_corte", "primera_tx", "ultima_tx"]
CATS = [c for c in ["genero", "ciudad", "segmento", "nivel_educativo", "ocupacion",
                    "producto_principal", "device_type", "ip_country_ultimo"] if c in df.columns]
for c in CATS:
    df[c] = df[c].astype("category")

FEATURES = [c for c in df.columns if c not in DROP and df[c].dtype.name != "datetime64[ns]"]
print(f"{len(FEATURES)} features")

train = df[df.split == "train"].reset_index(drop=True)
test = df[df.split == "score"].reset_index(drop=True)
y = train["is_fraud"].astype(int).values
X, X_test = train[FEATURES], test[FEATURES]

# etiqueta multiclase por tipologia
tipo = train["tipo_fraude"].fillna("none").astype(str)
TIPOS = sorted(tipo.unique())
tipo_idx = {t: i for i, t in enumerate(TIPOS)}
y_multi = tipo.map(tipo_idx).values
IDX_NONE = tipo_idx.get("none", 0)
print(f"tipologias: {TIPOS}")

N_SCORE = len(test)
rng = np.random.default_rng(SEED)


def precision_at_k_simulado(y_true, y_pred, n_subset=N_SCORE, k=K, n_sim=400):
    idx = np.arange(len(y_true))
    out = []
    for _ in range(n_sim):
        s = rng.choice(idx, size=min(n_subset, len(idx)), replace=False)
        out.append(y_true[s[np.argsort(-y_pred[s])[:k]]].mean())
    return float(np.mean(out)), float(np.std(out))


BASE = dict(learning_rate=0.03, num_leaves=63, min_child_samples=40,
            feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=5.0, n_estimators=3000, verbose=-1, n_jobs=-1)


def run_cv(kind, seed):
    """Devuelve (oof, test_pred) para una de las tres vistas del problema."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    tp = np.zeros(len(test))
    strat = y_multi if kind == "multi" else y
    for tr, va in skf.split(X, strat):
        if kind == "multi":
            m = lgb.LGBMClassifier(objective="multiclass", num_class=len(TIPOS),
                                   seed=seed, **BASE)
            m.fit(X.iloc[tr], y_multi[tr], eval_X=X.iloc[va], eval_y=y_multi[va],
                  eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = 1.0 - m.predict_proba(X.iloc[va])[:, IDX_NONE]
            tp += (1.0 - m.predict_proba(X_test)[:, IDX_NONE]) / skf.n_splits
        else:
            params = dict(BASE, objective="binary", seed=seed)
            if kind == "subsample":
                params["scale_pos_weight"] = 3.0
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr], y[tr], eval_X=X.iloc[va], eval_y=y[va], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            tp += m.predict_proba(X_test)[:, 1] / skf.n_splits
    return oof, tp, m


results = {}
last_model = None
for kind in ["binary", "multi", "subsample"]:
    oofs, tps = [], []
    for s in range(N_SEEDS):
        o, t, m = run_cv(kind, SEED + s)
        oofs.append(rankdata(o) / len(o))
        tps.append(rankdata(t) / len(t))
        last_model = m
    oof = np.mean(oofs, axis=0)
    tp = np.mean(tps, axis=0)
    auc = roc_auc_score(y, oof)
    p100, sd = precision_at_k_simulado(y, oof)
    results[kind] = (oof, tp)
    print(f"{kind:10s} AUC={auc:.4f}  AP={average_precision_score(y, oof):.4f}  "
          f"P@100={p100:.4f} (+/-{sd:.4f})")

# ---------------------------------------------------------------------------
# 3. Ensamble por promedio de rangos
# ---------------------------------------------------------------------------

oof_ens = np.mean([rankdata(results[k][0]) / len(train) for k in results], axis=0)
test_ens = np.mean([rankdata(results[k][1]) / len(test) for k in results], axis=0)

auc = roc_auc_score(y, oof_ens)
ap = average_precision_score(y, oof_ens)
p100, sd = precision_at_k_simulado(y, oof_ens)

print("\n" + "=" * 60)
print("ENSAMBLE FINAL")
print("=" * 60)
print(f"ROC-AUC (OOF)          : {auc:.4f}")
print(f"Average Precision (OOF): {ap:.4f}")
print(f"Precision@100 simulada : {p100:.4f} (+/- {sd:.4f})  -> {p100*100:.1f} aciertos por 100 alertas")
print(f"Tasa base              : {y.mean():.4f}  (lift {p100/y.mean():.1f}x)")

imp = pd.DataFrame({"feature": FEATURES, "gain": last_model.booster_.feature_importance("gain")}
                   ).sort_values("gain", ascending=False)
print("\nTop 20 features:")
print(imp.head(20).to_string(index=False))
imp.to_csv(f"{DATA}/importancia_full.csv", index=False)

sample = pd.read_csv(f"{DATA}/sample_submission.csv")
sub = sample[["id_cuenta"]].merge(
    pd.DataFrame({"id_cuenta": test["id_cuenta"], "score_probabilidad_fraude": test_ens}),
    on="id_cuenta", how="left")
assert len(sub) == len(sample) and sub["score_probabilidad_fraude"].notna().all()
sub.to_csv(f"{OUT}/submission_final.csv", index=False)
np.save(f"{DATA}/oof_full.npy", oof_ens)
print(f"\nSubmission guardado: {OUT}/submission_final.csv")
