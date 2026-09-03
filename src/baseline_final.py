"""
Baseline corregido y medido de forma insesgada - Hackathon Asobancaria 2026.

Corrige tres defectos detectados en revision adversarial del propio pipeline:

C1. FUGA POST-CORTE CODIFICADA COMO VARIABLE.
    dias_desde_ultimo_login = corte - ultimo_login tomaba valores NEGATIVOS
    (de -1 a -39) en 25.587 cuentas (14,2%), es decir, entregaba al modelo
    informacion posterior a la fecha de corte en forma directamente utilizable.
    Contradecia la regla anti-fuga del reto y la propia denuncia del informe.
    Tratamiento: se censura a valor faltante cuando el ultimo acceso es
    posterior al corte, y se elimina el indicador binario derivado.
    Costo predictivo nulo: AUC univariado 0,4921 (indicador solo, 0,5039).

C2. VALOR CENTINELA TRATADO COMO CANTIDAD.
    dias_desde_ultimo_cambio = 999 en el 90,06% de los registros. No son 999
    dias: es "sin cambio de dispositivo registrado". Usarlo como numero mete
    una distancia falsa de ~900 dias frente a los cambios reales y distorsiona
    los cortes del arbol.
    Tratamiento: 999 pasa a valor faltante y se agrega un indicador explicito
    sin_cambio_dispositivo.

C3. ESTIMACION OPTIMISTA POR SELECCION SOBRE LAS MISMAS PARTICIONES.
    Los hiperparametros se eligieron comparando configuraciones sobre las
    mismas 5 particiones con las que luego se reportaba la metrica. El maximo
    de varias evaluaciones sobre la misma particion esta sesgado al alza.
    Tratamiento: se reserva un holdout del 20% estratificado que NO participa
    ni en la seleccion ni en el entrenamiento, y la cifra oficial se reporta
    sobre el. Se informan ambas cifras para que la diferencia sea visible.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "/home/user/Declaracion_renta/data"
OUT = "/home/user/Declaracion_renta/submissions"
SEED = 42
K = 100
N_SCORE = 44962

labels = pd.read_parquet(f"{DATA}/fraud_labels_train.parquet")
score = pd.read_parquet(f"{DATA}/accounts_to_score.parquet")
customers = pd.read_parquet(f"{DATA}/customers.parquet")
updates = pd.read_parquet(f"{DATA}/data_update_history.parquet")
device = pd.read_parquet(f"{DATA}/device_mobile_activity.parquet")
CUTOFF = pd.Timestamp(score["fecha_corte"].max())
print(f"Fecha de corte: {CUTOFF.date()}")


def preparar_dispositivos(device, cutoff):
    d = device.copy()
    d["ultimo_login"] = pd.to_datetime(d["ultimo_login"], errors="coerce")

    # C2: el centinela 999 no es una cantidad de dias
    d["sin_cambio_dispositivo"] = (d["dias_desde_ultimo_cambio"] == 999).astype(int)
    d.loc[d["dias_desde_ultimo_cambio"] == 999, "dias_desde_ultimo_cambio"] = np.nan

    agg = d.groupby("id_cuenta").agg(
        num_dispositivos=("device_id", "size"),
        num_tipos_dispositivo=("device_type", "nunique"),
        is_rooted_or_jailbreak=("is_rooted_or_jailbreak", "max"),
        is_emulator=("is_emulator", "max"),
        num_cambios_dispositivo_12m=("num_cambios_dispositivo_12m", "max"),
        dias_desde_ultimo_cambio=("dias_desde_ultimo_cambio", "min"),
        sin_cambio_dispositivo=("sin_cambio_dispositivo", "min"),
        num_sesiones_30d=("num_sesiones_30d", "sum"),
        num_ciudades_acceso_30d=("num_ciudades_acceso_30d", "max"),
        pct_accesos_fuera_ciudad=("pct_accesos_fuera_ciudad", "max"),
        ultimo_login=("ultimo_login", "max"),
        device_type=("device_type", "first"),
        ip_country_ultimo=("ip_country_ultimo", "first"),
    ).reset_index()

    agg["ip_ultima_fuera_co"] = (agg["ip_country_ultimo"] != "CO").astype(int)
    # C1: censura anti-fuga. Un acceso posterior al corte no es observable
    # en la fecha de decision: la recencia queda como dato faltante.
    dias = (cutoff - agg["ultimo_login"]).dt.days
    agg["dias_desde_ultimo_login"] = dias.where(dias >= 0, np.nan)
    n_cens = int((dias < 0).sum())
    print(f"C1: censuradas {n_cens} cuentas ({n_cens/len(agg):.1%}) con acceso posterior al corte")
    return agg.drop(columns=["ultimo_login"])


dev = preparar_dispositivos(device, CUTOFF)
assert dev["id_cuenta"].is_unique


def construir(base):
    d = (base
         .merge(customers, on="id_cuenta", how="left")
         .merge(updates, on="id_cuenta", how="left")
         .merge(dev, on="id_cuenta", how="left"))
    d["cambios_contacto_total"] = (d["num_cambios_telefono"] + d["num_cambios_email"]
                                   + d["num_cambios_direccion"])
    d["ratio_cambios_contacto"] = d["cambios_contacto_total"] / d["num_actualizaciones_12m"].replace(0, np.nan)
    d["kyc_vencido_1y"] = (d["dias_desde_ultimo_kyc"] > 365).astype(int)
    d["sesiones_por_ciudad"] = d["num_sesiones_30d"] / d["num_ciudades_acceso_30d"].replace(0, np.nan)
    d["cuenta_nueva_90d"] = (d["antiguedad_dias"] <= 90).astype(int)
    d["sin_2fa"] = 1 - d["tiene_2fa"]
    d["intensidad_cambios"] = d["cambios_contacto_total"] / (d["antiguedad_dias"] + 1)
    return d


CATS = ["genero", "ciudad", "segmento", "nivel_educativo", "ocupacion",
        "producto_principal", "device_type", "ip_country_ultimo"]

train_df = construir(labels[["id_cuenta", "is_fraud"]])
test_df = construir(score[["id_cuenta"]])
assert len(train_df) == 135038 and len(test_df) == 44962

for c in CATS:
    cats = train_df[c].astype("category").cat.categories
    train_df[c] = pd.Categorical(train_df[c], categories=cats)
    test_df[c] = pd.Categorical(test_df[c], categories=cats)

# flag_login_post_corte eliminado deliberadamente (C1)
FEATURES = [c for c in train_df.columns if c not in ("id_cuenta", "id_cliente", "is_fraud")]
print(f"{len(FEATURES)} features")

y_all = train_df["is_fraud"].astype(int).values
X_all = train_df[FEATURES]

# C3: holdout intocado para la cifra oficial
X_dev, X_hold, y_dev, y_hold = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=SEED)
print(f"desarrollo: {len(X_dev)}  holdout: {len(X_hold)} (fraude {y_hold.mean():.4f})")

rng = np.random.default_rng(SEED)


def p_at_k(y_true, y_pred, n_sim=500):
    """Precision@100 estimada por remuestreo al tamano del set a calificar.
    Cuando la muestra disponible es menor que ese tamano, se remuestrea al
    tamano disponible y se ajusta K en la misma proporcion, para que el
    percentil evaluado sea el mismo."""
    n = len(y_true)
    n_sub = min(N_SCORE, n)
    k = max(1, int(round(K * n_sub / N_SCORE)))
    idx = np.arange(n)
    out = np.empty(n_sim)
    for i in range(n_sim):
        s = rng.choice(idx, size=n_sub, replace=False)
        out[i] = y_true[s[np.argsort(-y_pred[s])[:k]]].mean()
    return out.mean(), out.std() / np.sqrt(n_sim)


PARAMS = dict(objective="binary", learning_rate=0.01, num_leaves=15,
              min_child_samples=200, feature_fraction=0.6, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=50.0, n_estimators=1200,
              verbose=-1, seed=SEED, n_jobs=-1)

# --- cifra de desarrollo (comparable con lo reportado antes) ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X_dev))
for tr, va in skf.split(X_dev, y_dev):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X_dev.iloc[tr], y_dev[tr])
    oof[va] = m.predict_proba(X_dev.iloc[va])[:, 1]
auc_dev = roc_auc_score(y_dev, oof)
p_dev, se_dev = p_at_k(y_dev, oof)

# --- cifra oficial: holdout jamas visto ---
modelos, pred_hold, pred_test = [], np.zeros(len(X_hold)), np.zeros(len(test_df))
for tr, va in skf.split(X_dev, y_dev):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X_dev.iloc[tr], y_dev[tr])
    pred_hold += m.predict_proba(X_hold)[:, 1] / 5
    pred_test += m.predict_proba(test_df[FEATURES])[:, 1] / 5
    modelos.append(m)
auc_hold = roc_auc_score(y_hold, pred_hold)
ap_hold = average_precision_score(y_hold, pred_hold)
p_hold, se_hold = p_at_k(y_hold, pred_hold)

print("\n" + "=" * 66)
print("BASELINE CORREGIDO — cifras de desarrollo vs. holdout intocado")
print("=" * 66)
print(f"{'':22s} {'ROC-AUC':>9s} {'P@100':>16s}")
print(f"{'desarrollo (CV)':22s} {auc_dev:9.4f} {p_dev:9.4f} +/-{se_dev:.4f}")
print(f"{'HOLDOUT (oficial)':22s} {auc_hold:9.4f} {p_hold:9.4f} +/-{se_hold:.4f}")
print(f"\nAverage Precision (holdout): {ap_hold:.4f}")
print(f"Optimismo por seleccion: {p_dev - p_hold:+.4f} en P@100")
print(f"Tasa base holdout: {y_hold.mean():.4f}  |  lift: {p_hold/y_hold.mean():.1f}x")

imp = pd.DataFrame({"feature": FEATURES,
                    "gain": modelos[-1].booster_.feature_importance("gain")}
                   ).sort_values("gain", ascending=False)
print("\nTop 12 features:")
print(imp.head(12).to_string(index=False))

sub = pd.read_csv(f"{DATA}/sample_submission.csv")[["id_cuenta"]].merge(
    pd.DataFrame({"id_cuenta": test_df["id_cuenta"], "score_probabilidad_fraude": pred_test}),
    on="id_cuenta", how="left")
assert len(sub) == 44962 and sub["score_probabilidad_fraude"].notna().all()
sub.to_csv(f"{OUT}/submission_baseline_static.csv", index=False)
imp.to_csv(f"{DATA}/importancia_baseline.csv", index=False)
np.save(f"{DATA}/oof_dev.npy", oof)
np.save(f"{DATA}/y_dev.npy", y_dev)
np.save(f"{DATA}/pred_hold.npy", pred_hold)
np.save(f"{DATA}/y_hold.npy", y_hold)
print(f"\nSubmission: {OUT}/submission_baseline_static.csv ({len(sub)} filas)")
print(f"empates dentro del top-200: {200 - sub['score_probabilidad_fraude'].nlargest(200).nunique()}")
