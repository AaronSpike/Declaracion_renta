"""
Modelo final - Hackathon Asobancaria 2026 (Banco AuditPlus)

Reemplaza a baseline_final.py corrigiendo cinco defectos que encontro la
auditoria adversarial del propio pipeline. Cada correccion esta respaldada por
una medicion, no por criterio.

D1  ESTIMADOR DEGENERADO. p_at_k usaba n_sub = min(N_SCORE, n) y muestreaba sin
    reemplazo: cuando n < N_SCORE eso es una permutacion del conjunto entero,
    las 500 simulaciones daban el MISMO valor y el error estandar impreso era
    exactamente 0. Se reemplaza por bootstrap con reemplazo, que si mide la
    incertidumbre real. Verificado: 200 simulaciones -> 1 valor distinto, sd 0,0.

D2  METRICA MEDIDA DONDE SE ELIGIO. El 21-23% reportado salia del set de
    desarrollo, el mismo donde se seleccionaron los hiperparametros. Ahora la
    cifra reportable se calcula fuera de muestra por validacion cruzada sobre
    las 135.038 filas, con K reescalado al mismo percentil (300 sobre 135.038
    equivale a 100 sobre 44.962): mismo percentil, 3x mas eventos, menos ruido.

D3  FUGA POST-CORTE RENOMBRADA, NO ELIMINADA. Censurar la recencia de acceso a
    nulo hizo que {es nulo} coincidiera EXACTAMENTE con {login posterior al
    corte} en las 180.000 cuentas. LightGBM enruta los nulos por rama propia, de
    modo que la senal seguia disponible: el top-100 entregado tenia 31,0% de
    cuentas con login post-corte contra 14,5% de la base (2,14x). Ahora se
    recorta a 0: la cuenta se trata como si su ultimo acceso fuera la fecha de
    corte, que es el estado mas reciente observable en el momento de la decision.

D4  VARIABLES SIN RELACION CON EL OBJETIVO. 17 variables no muestran relacion
    con is_fraud (chi2 e AUC univariado) y sin embargo tres de ellas se llevaban
    el 37% del gain del modelo. Se podan. Efecto medido: P@300 sube de 0,2156 a
    0,2322 y su dispersion se reduce a la mitad (+/-0,0118 -> +/-0,0048), con el
    AUC intacto (0,6223 -> 0,6225).
    Nota de auditoria: las variables podadas son justamente las sensibles
    (genero, ciudad, nivel educativo, ocupacion, edad). La correccion de
    desempeno y la de riesgo de trato discriminatorio resultaron ser la misma.

D5  DUPLICADO EXACTO. num_dispositivos == num_cambios_dispositivo_12m + 1 en el
    100% de las filas. Eran la misma variable contada dos veces; con
    feature_fraction < 1 se sorteaban como si fueran independientes. Se deja una.

D6  ENTREGA CON DATOS INCOMPLETOS. El modelo enviado se entrenaba sobre el 64%
    de las etiquetas (80% de desarrollo x 80% de cada pliegue). Una vez
    congelada la configuracion, el modelo de entrega usa el 100%.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "/home/user/Declaracion_renta/data"
SEED = 42
N_SCORE = 44962          # cuentas a calificar
K_NEGOCIO = 100          # capacidad operativa diaria
CUTOFF = pd.Timestamp("2026-04-29")
rng = np.random.default_rng(SEED)

# Configuracion congelada. Elegida con P@300 fuera de muestra; las diferencias
# entre las configuraciones cercanas NO son significativas (z < 1), asi que se
# prefiere la mas regularizada por estabilidad, no por su punto estimado.
PARAMS = dict(
    objective="binary", learning_rate=0.01, num_leaves=15,
    min_child_samples=500, feature_fraction=0.6, bagging_fraction=0.7,
    bagging_freq=1, lambda_l2=200.0, n_estimators=1200,
    verbose=-1, n_jobs=-1,
)

# D4/D5: solo las variables con relacion medida con el objetivo.
FEATURES = [
    "num_actualizaciones_12m", "num_cambios_telefono", "num_cambios_email",
    "num_cambios_direccion", "cambios_contacto_total", "ratio_cambios_contacto",
    "num_dispositivos", "num_tipos_dispositivo", "is_rooted_or_jailbreak",
    "is_emulator", "dias_desde_ultimo_cambio", "sin_cambio_dispositivo",
    "num_sesiones_30d", "num_ciudades_acceso_30d", "pct_accesos_fuera_ciudad",
    "sesiones_por_ciudad", "dias_desde_ultimo_login",
]


# ---------------------------------------------------------------------------
# Metrica de negocio, medida honestamente
# ---------------------------------------------------------------------------

def precision_at_capacidad(y, pred, n_boot=2000):
    """Precision en la cabeza del ranking, al percentil que corresponde a
    revisar 100 de 44.962 cuentas, con intervalo de confianza por bootstrap.

    Devuelve (punto, ic_bajo, ic_alto, aciertos, k). Se reporta tambien el
    conteo crudo de aciertos: '65/300' hace visible de inmediato sobre cuantos
    eventos descansa la cifra, cosa que un porcentaje esconde.
    """
    n = len(y)
    k = max(1, int(round(K_NEGOCIO * n / N_SCORE)))
    aciertos = int(y[np.argsort(-pred)[:k]].sum())
    boot = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)                      # con reemplazo: si mide
        boot[i] = y[s][np.argsort(-pred[s])[:k]].mean()
    return aciertos / k, np.percentile(boot, 2.5), np.percentile(boot, 97.5), aciertos, k


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def preparar(ids):
    """Construye la matriz de features para un conjunto de id_cuenta."""
    cus = pd.read_parquet(f"{DATA}/customers.parquet")
    upd = pd.read_parquet(f"{DATA}/data_update_history.parquet")
    dev = pd.read_parquet(f"{DATA}/device_mobile_activity.parquet")
    dev["ultimo_login"] = pd.to_datetime(dev["ultimo_login"], errors="coerce")

    # D3: recorte a 0 en vez de censura a nulo. Un acceso posterior al corte se
    # trata como acceso en la fecha de corte; asi la variable no codifica
    # "ocurrio despues" de forma recuperable por el patron de faltantes.
    dev["dias_login"] = (CUTOFF - dev["ultimo_login"]).dt.days.clip(lower=0)

    # dias_desde_ultimo_cambio usa 999 como centinela ("sin cambio registrado")
    # en el 90,06% de los registros. Se separa el centinela de la cantidad.
    dev["cambio_real"] = dev["dias_desde_ultimo_cambio"].where(
        dev["dias_desde_ultimo_cambio"] != 999)

    # device tiene 185.004 filas para 180.000 cuentas: agregar antes de unir.
    agg = dev.groupby("id_cuenta").agg(
        num_dispositivos=("device_id", "size"),
        num_tipos_dispositivo=("device_type", "nunique"),
        is_rooted_or_jailbreak=("is_rooted_or_jailbreak", "max"),
        is_emulator=("is_emulator", "max"),
        dias_desde_ultimo_cambio=("cambio_real", "min"),
        num_sesiones_30d=("num_sesiones_30d", "sum"),
        num_ciudades_acceso_30d=("num_ciudades_acceso_30d", "max"),
        pct_accesos_fuera_ciudad=("pct_accesos_fuera_ciudad", "max"),
        dias_desde_ultimo_login=("dias_login", "min"),
    ).reset_index()
    agg["sin_cambio_dispositivo"] = agg["dias_desde_ultimo_cambio"].isna().astype(int)

    df = (ids.merge(cus[["id_cuenta"]], on="id_cuenta", how="left")
             .merge(upd, on="id_cuenta", how="left")
             .merge(agg, on="id_cuenta", how="left"))

    df["cambios_contacto_total"] = (df["num_cambios_telefono"]
                                    + df["num_cambios_email"]
                                    + df["num_cambios_direccion"])
    df["ratio_cambios_contacto"] = (df["cambios_contacto_total"]
                                    / df["num_actualizaciones_12m"].replace(0, np.nan))
    df["sesiones_por_ciudad"] = (df["num_sesiones_30d"]
                                 / df["num_ciudades_acceso_30d"].replace(0, np.nan))

    assert len(df) == len(ids), f"el cruce cambio el numero de filas: {len(df)} vs {len(ids)}"
    return df


labels = pd.read_parquet(f"{DATA}/fraud_labels_train.parquet")
score = pd.read_parquet(f"{DATA}/accounts_to_score.parquet")

train = preparar(labels[["id_cuenta"]])
test = preparar(score[["id_cuenta"]])
y = labels["is_fraud"].astype(int).values

X, X_test = train[FEATURES], test[FEATURES]
print(f"{len(FEATURES)} features | train {len(X)} | score {len(X_test)} | tasa base {y.mean():.4f}")

# ---------------------------------------------------------------------------
# D2: medicion fuera de muestra sobre TODAS las etiquetas
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(seed=SEED, **PARAMS)
    m.fit(X.iloc[tr], y[tr])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]

auc = roc_auc_score(y, oof)
ap = average_precision_score(y, oof)
p, lo, hi, ac, k = precision_at_capacidad(y, oof)

print("\n" + "=" * 66)
print("DESEMPENO FUERA DE MUESTRA (135.038 cuentas, validacion cruzada)")
print("=" * 66)
print(f"  ROC-AUC              {auc:.4f}")
print(f"  Average Precision    {ap:.4f}")
print(f"  Precision@capacidad  {p:.4f}   ({ac}/{k} aciertos)")
print(f"    IC 95% bootstrap   [{lo:.4f}, {hi:.4f}]")
print(f"  Tasa base            {y.mean():.4f}")
print(f"  Lift                 {p / y.mean():.2f}x")

# ---------------------------------------------------------------------------
# D6: el modelo de entrega usa el 100% de las etiquetas
# ---------------------------------------------------------------------------
pred = np.zeros(len(X_test))
SEMILLAS = [42, 7, 123, 2024, 999]      # promediar semillas reduce la varianza
for s in SEMILLAS:
    m = lgb.LGBMClassifier(seed=s, **{**PARAMS, "bagging_seed": s, "feature_fraction_seed": s})
    m.fit(X, y)
    pred += m.predict_proba(X_test)[:, 1] / len(SEMILLAS)

sub = pd.DataFrame({"id_cuenta": test["id_cuenta"], "score_probabilidad_fraude": pred})
plantilla = pd.read_csv(f"{DATA}/sample_submission.csv")[["id_cuenta"]]
sub = plantilla.merge(sub, on="id_cuenta", how="left")
assert len(sub) == N_SCORE, f"filas: {len(sub)}"
assert sub["score_probabilidad_fraude"].notna().all(), "hay predicciones faltantes"
assert sub["score_probabilidad_fraude"].between(0, 1).all(), "scores fuera de [0,1]"
sub.to_csv("/home/user/Declaracion_renta/submissions/submission_final.csv", index=False)

np.save(f"{DATA}/oof_final.npy", oof)
imp = pd.DataFrame({"feature": FEATURES,
                    "gain": m.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
imp.to_csv(f"{DATA}/importancia_final.csv", index=False)

print(f"\nsubmission_final.csv escrito ({len(sub)} filas)")
print("\nvariables por aporte:")
print(imp.to_string(index=False))
