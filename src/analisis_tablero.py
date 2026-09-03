"""
Genera los datos reales que alimentan el tablero de auditoria.
Salida: data/tablero.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

DATA = "/home/user/Declaracion_renta/data"

labels = pd.read_parquet(f"{DATA}/fraud_labels_train.parquet")
score = pd.read_parquet(f"{DATA}/accounts_to_score.parquet")
customers = pd.read_parquet(f"{DATA}/customers.parquet")
device = pd.read_parquet(f"{DATA}/device_mobile_activity.parquet")
updates = pd.read_parquet(f"{DATA}/data_update_history.parquet")

oof = np.load(f"{DATA}/oof_baseline.npy")
y = labels["is_fraud"].astype(int).values
N_SCORE = len(score)
BASE = y.mean()

out = {}

# ---------------------------------------------------------------------------
# 1. Composicion del universo y tipologias
# ---------------------------------------------------------------------------
tip = labels["tipo_fraude"].value_counts()
out["universo"] = {
    "total_cuentas": 180000,
    "train": int(len(labels)),
    "score": int(N_SCORE),
    "fraudes": int(y.sum()),
    "tasa_base": float(BASE),
    "capacidad_diaria": 100,
}
out["tipologias"] = [
    {"tipo": t, "n": int(n), "pct_de_fraudes": float(n / y.sum())}
    for t, n in tip.items() if t != "none"
]

# ---------------------------------------------------------------------------
# 2. Curva Precision@K (la metrica que importa) y ganancias acumuladas
# ---------------------------------------------------------------------------
orden = np.argsort(-oof)
y_ord = y[orden]
acum = np.cumsum(y_ord)
ks = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
# escalar K al tamano del set de evaluacion: el top-K de 44.962 cuentas
# equivale al top-K*(135038/44962) sobre las 135.038 de entrenamiento
factor = len(y) / N_SCORE
out["precision_at_k"] = [
    {
        "k": k,
        "k_equivalente_train": int(round(k * factor)),
        "precision": float(acum[int(round(k * factor)) - 1] / int(round(k * factor))),
        "precision_aleatoria": float(BASE),
        "lift": float((acum[int(round(k * factor)) - 1] / int(round(k * factor))) / BASE),
    }
    for k in ks if int(round(k * factor)) <= len(y)
]

# Curva de ganancias acumuladas (decil a decil)
deciles = []
n = len(y)
for d in range(1, 11):
    corte = int(n * d / 10)
    deciles.append({
        "decil": d,
        "pct_poblacion": d * 10,
        "pct_fraudes_capturados": float(acum[corte - 1] / y.sum() * 100),
        "captura_aleatoria": d * 10,
    })
out["ganancias_deciles"] = deciles

# Curva ROC (submuestreada para el grafico)
fpr, tpr, _ = roc_curve(y, oof)
idx = np.linspace(0, len(fpr) - 1, 100).astype(int)
out["curva_roc"] = {
    "auc": float(roc_auc_score(y, oof)),
    "puntos": [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx],
}

# ---------------------------------------------------------------------------
# 3. Senal univariada: que variables discriminan por si solas
# ---------------------------------------------------------------------------
dev = device.groupby("id_cuenta").agg(
    num_dispositivos=("device_id", "size"),
    is_emulator=("is_emulator", "max"),
    is_rooted_or_jailbreak=("is_rooted_or_jailbreak", "max"),
    num_cambios_dispositivo_12m=("num_cambios_dispositivo_12m", "max"),
    num_ciudades_acceso_30d=("num_ciudades_acceso_30d", "max"),
    pct_accesos_fuera_ciudad=("pct_accesos_fuera_ciudad", "max"),
    num_sesiones_30d=("num_sesiones_30d", "sum"),
).reset_index()

base = (labels[["id_cuenta", "is_fraud"]]
        .merge(customers, on="id_cuenta", how="left")
        .merge(updates, on="id_cuenta", how="left")
        .merge(dev, on="id_cuenta", how="left"))

NUMS = ["edad", "antiguedad_dias", "tiene_2fa", "num_actualizaciones_12m",
        "num_cambios_telefono", "num_cambios_email", "num_cambios_direccion",
        "dias_desde_ultimo_kyc", "num_dispositivos", "is_emulator",
        "is_rooted_or_jailbreak", "num_cambios_dispositivo_12m",
        "num_ciudades_acceso_30d", "pct_accesos_fuera_ciudad", "num_sesiones_30d"]

univar = []
yb = base["is_fraud"].values
for c in NUMS:
    v = base[c].astype(float)
    if v.notna().sum() < 1000 or v.nunique() < 2:
        continue
    m = v.notna()
    auc = roc_auc_score(yb[m], v[m])
    univar.append({
        "variable": c,
        "auc": float(max(auc, 1 - auc)),
        "direccion": "mayor = mas riesgo" if auc > 0.5 else "menor = mas riesgo",
        "media_fraude": float(v[yb == 1].mean()),
        "media_legitima": float(v[yb == 0].mean()),
    })
out["senal_univariada"] = sorted(univar, key=lambda r: -r["auc"])

# ---------------------------------------------------------------------------
# 4. Contrastes por tipologia: cada fraude tiene firma distinta
# ---------------------------------------------------------------------------
perfil = []
for t in ["account_takeover", "apertura_fraudulenta", "cuenta_mula", "suplantacion"]:
    sub = base[labels["tipo_fraude"].values == t]
    leg = base[labels["tipo_fraude"].values == "none"]
    perfil.append({
        "tipologia": t,
        "n": int(len(sub)),
        "antiguedad_dias": {"tipologia": float(sub["antiguedad_dias"].mean()),
                            "legitimas": float(leg["antiguedad_dias"].mean())},
        "num_cambios_telefono": {"tipologia": float(sub["num_cambios_telefono"].mean()),
                                 "legitimas": float(leg["num_cambios_telefono"].mean())},
        "num_dispositivos": {"tipologia": float(sub["num_dispositivos"].mean()),
                             "legitimas": float(leg["num_dispositivos"].mean())},
        "pct_accesos_fuera_ciudad": {"tipologia": float(sub["pct_accesos_fuera_ciudad"].mean()),
                                     "legitimas": float(leg["pct_accesos_fuera_ciudad"].mean())},
        "tasa_emulador": {"tipologia": float(sub["is_emulator"].mean()),
                          "legitimas": float(leg["is_emulator"].mean())},
        "dias_desde_ultimo_kyc": {"tipologia": float(sub["dias_desde_ultimo_kyc"].mean()),
                                  "legitimas": float(leg["dias_desde_ultimo_kyc"].mean())},
    })
out["perfil_tipologias"] = perfil

# ---------------------------------------------------------------------------
# 5. Calidad de datos (evidencia de los hallazgos)
# ---------------------------------------------------------------------------
device["ultimo_login"] = pd.to_datetime(device["ultimo_login"], errors="coerce")
CUTOFF = pd.Timestamp(score["fecha_corte"].max())
conf = pd.to_datetime(labels["fecha_confirmacion_fraude"], errors="coerce")
out["calidad_datos"] = {
    "fecha_corte": str(CUTOFF.date()),
    "device_filas": int(len(device)),
    "device_cuentas": int(device["id_cuenta"].nunique()),
    "cuentas_multi_dispositivo": int((device["id_cuenta"].value_counts() > 1).sum()),
    "logins_posteriores_al_corte": int((device["ultimo_login"] > CUTOFF).sum()),
    "pct_logins_post_corte": float((device["ultimo_login"] > CUTOFF).mean()),
    "ultimo_login_maximo": str(device["ultimo_login"].max().date()),
    "fraudes_confirmados_post_corte": int((conf > CUTOFF).sum()),
    "fraudes_confirmados_pre_corte": int((conf <= CUTOFF).sum()),
    "rango_confirmacion": [str(conf.min().date()), str(conf.max().date())],
}

# ---------------------------------------------------------------------------
# 6. Desempeno del baseline
# ---------------------------------------------------------------------------
imp = pd.read_csv(f"{DATA}/importancia_baseline.csv")
out["modelo_baseline"] = {
    "auc": float(roc_auc_score(y, oof)),
    "precision_at_100": float(acum[int(round(100 * factor)) - 1] / int(round(100 * factor))),
    "top_features": imp.head(12).to_dict("records"),
}

with open(f"{DATA}/tablero.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(json.dumps({k: (v if not isinstance(v, list) else f"[{len(v)} items]")
                  for k, v in out.items()}, indent=2, ensure_ascii=False)[:2500])
