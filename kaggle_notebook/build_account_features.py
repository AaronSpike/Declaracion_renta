"""
Feature engineering a nivel de cuenta - Hackathon Asobancaria 2026 (Banco AuditPlus)

COMO USAR:
1. Crea/abre un Kaggle Notebook adjuntado a la competencia
   "hackaton-asobancaria-auditores-2026" (Add Data -> la competencia).
2. Pega TODO este archivo en una celda y ejecutalo (Run All).
   No necesitas subir ni descargar transactions.parquet: ya esta montado
   en /kaggle/input dentro del propio notebook.
3. Al terminar, queda un archivo /kaggle/working/account_features.parquet
   (deberia pesar pocos MB, NO cientos de MB). Descargalo desde el panel
   "Output" del notebook y subelo de vuelta en el chat.

DECISIONES DE DISENO (anti-leakage), documentadas para el jurado:
- Cada cuenta tiene una "fecha de corte" (cutoff): solo se usan transacciones
  con timestamp <= cutoff.
  * Cuentas de evaluacion (accounts_to_score): cutoff = fecha_corte (dada).
  * Cuentas fraudulentas de train (is_fraud=1): cutoff = fecha_confirmacion_fraude.
    Se usa la fecha de CONFIRMACION del caso (no despues) para no filtrar el
    comportamiento posterior a la contencion/bloqueo del banco.
  * Cuentas legitimas de train (is_fraud=0, sin fecha de confirmacion):
    no tienen un evento natural de corte. Se les asigna un cutoff global
    GLOBAL_SNAPSHOT_DATE = MAX(timestamp) observado en transactions.parquet,
    simulando "hoy" como fecha de observacion. Esta es una decision de
    negocio explicita: se documenta en el pitch como supuesto auditado.
- Las tablas customers, credit_indebtedness, device_mobile_activity y
  data_update_history son snapshots sin fecha por evento (no son logs
  transaccionales), por lo que NO se pueden filtrar por cutoff con precision.
  Se asume que reflejan el estado mas reciente disponible en la extraccion.
  RIESGO DE AUDITORIA: para cuentas de fraude confirmado esto puede introducir
  fuga leve post-evento (ej. cambio de dispositivo tras el bloqueo). Se deja
  registrado como hallazgo de gobernanza de datos, no se intenta corregir
  con supuestos adicionales no soportados por los datos.
"""

import glob
import os
import subprocess
import sys

try:
    import duckdb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckdb"])
    import duckdb

# ---------------------------------------------------------------------------
# 1. Localizar archivos de entrada (autodeteccion por nombre bajo /kaggle/input)
# ---------------------------------------------------------------------------

def find_file(name):
    matches = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"No se encontro {name} bajo /kaggle/input. Verifica el Add Data.")
    return matches[0]

FILES = {
    "transactions": find_file("transactions.parquet"),
    "customers": find_file("customers.parquet"),
    "credit_indebtedness": find_file("credit_indebtedness.parquet"),
    "device_mobile_activity": find_file("device_mobile_activity.parquet"),
    "data_update_history": find_file("data_update_history.parquet"),
    "merchants": find_file("merchants.parquet"),
    "fraud_labels_train": find_file("fraud_labels_train.parquet"),
    "accounts_to_score": find_file("accounts_to_score.parquet"),
}
for k, v in FILES.items():
    print(f"{k}: {v}")

OUT_PATH = "/kaggle/working/account_features.parquet"

con = duckdb.connect()
con.execute("PRAGMA threads=4")

# ---------------------------------------------------------------------------
# 2. Tabla maestra de cuentas + cutoff (anti-leakage)
# ---------------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TABLE accounts_master AS
WITH global_snapshot AS (
    SELECT MAX(TRY_CAST(timestamp AS TIMESTAMP)) AS snapshot_date
    FROM read_parquet('{FILES["transactions"]}')
),
train AS (
    SELECT
        id_cuenta,
        is_fraud,
        tipo_fraude,
        TRY_CAST(fecha_confirmacion_fraude AS TIMESTAMP) AS fecha_confirmacion_fraude,
        'train' AS split
    FROM read_parquet('{FILES["fraud_labels_train"]}')
),
score AS (
    SELECT
        id_cuenta,
        CAST(NULL AS INTEGER) AS is_fraud,
        CAST(NULL AS VARCHAR) AS tipo_fraude,
        TRY_CAST(fecha_corte AS TIMESTAMP) AS fecha_corte,
        'score' AS split
    FROM read_parquet('{FILES["accounts_to_score"]}')
)
SELECT
    t.id_cuenta,
    t.split,
    t.is_fraud,
    t.tipo_fraude,
    CASE
        WHEN t.is_fraud = 1 AND t.fecha_confirmacion_fraude IS NOT NULL
            THEN t.fecha_confirmacion_fraude
        ELSE (SELECT snapshot_date FROM global_snapshot)
    END AS cutoff_date
FROM train t
UNION ALL
SELECT
    s.id_cuenta,
    s.split,
    s.is_fraud,
    s.tipo_fraude,
    s.fecha_corte AS cutoff_date
FROM score s
""")

print("accounts_master:", con.execute("SELECT split, COUNT(*), MIN(cutoff_date), MAX(cutoff_date) FROM accounts_master GROUP BY split").fetchall())

# ---------------------------------------------------------------------------
# 3. Transacciones filtradas por cutoff (anti-leakage) + merchants
# ---------------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TABLE tx_filtered AS
SELECT
    m.id_cuenta,
    m.cutoff_date,
    TRY_CAST(t.timestamp AS TIMESTAMP) AS ts,
    t.amount,
    t.account_type,
    t.canal,
    t.is_international,
    t.is_new_ip,
    t.ip_country,
    t.id_comercio,
    mc.merchant_category,
    mc.merchant_country AS merchant_country,
    mc.merchant_risk_score
FROM read_parquet('{FILES["transactions"]}') t
JOIN accounts_master m ON m.id_cuenta = t.id_cuenta
LEFT JOIN read_parquet('{FILES["merchants"]}') mc ON mc.id_comercio = t.id_comercio
WHERE TRY_CAST(t.timestamp AS TIMESTAMP) <= m.cutoff_date
""")

print("tx_filtered rows:", con.execute("SELECT COUNT(*) FROM tx_filtered").fetchone())

# ---------------------------------------------------------------------------
# 4. Agregados transaccionales por cuenta
# ---------------------------------------------------------------------------

con.execute("""
CREATE OR REPLACE TABLE agg_base AS
SELECT
    id_cuenta,
    ANY_VALUE(cutoff_date) AS cutoff_date,
    COUNT(*) AS num_tx_total,
    SUM(amount) AS monto_total,
    AVG(amount) AS monto_avg,
    STDDEV_POP(amount) AS monto_std,
    MAX(amount) AS monto_max,
    MIN(amount) AS monto_min,
    MEDIAN(amount) AS monto_mediana,
    MIN(ts) AS primera_tx,
    MAX(ts) AS ultima_tx,
    AVG(is_international) AS pct_internacional,
    AVG(is_new_ip) AS pct_ip_nueva,
    COUNT(DISTINCT ip_country) AS num_paises_ip_distintos,
    COUNT(DISTINCT canal) AS num_canales_distintos,
    COUNT(DISTINCT id_comercio) AS num_comercios_distintos,
    COUNT(DISTINCT merchant_category) AS num_categorias_comercio_distintos,
    COUNT(DISTINCT account_type) AS num_tipos_cuenta_distintos,
    AVG(merchant_risk_score) AS merchant_risk_avg,
    MAX(merchant_risk_score) AS merchant_risk_max,
    AVG(CASE WHEN EXTRACT(hour FROM ts) BETWEEN 0 AND 5 THEN 1.0 ELSE 0.0 END) AS pct_horario_nocturno,
    AVG(CASE WHEN dayofweek(ts) IN (0, 6) THEN 1.0 ELSE 0.0 END) AS pct_fin_semana,
    SUM(CASE WHEN ts >= cutoff_date - INTERVAL 7 DAY THEN 1 ELSE 0 END) AS num_tx_7d,
    SUM(CASE WHEN ts >= cutoff_date - INTERVAL 30 DAY THEN 1 ELSE 0 END) AS num_tx_30d,
    SUM(CASE WHEN ts >= cutoff_date - INTERVAL 90 DAY THEN 1 ELSE 0 END) AS num_tx_90d,
    SUM(CASE WHEN ts >= cutoff_date - INTERVAL 7 DAY THEN amount ELSE 0 END) AS monto_7d,
    SUM(CASE WHEN ts >= cutoff_date - INTERVAL 30 DAY THEN amount ELSE 0 END) AS monto_30d,
    SUM(CASE WHEN ts >= cutoff_date - INTERVAL 90 DAY THEN amount ELSE 0 END) AS monto_90d,
    DATE_DIFF('day', MIN(ts), ANY_VALUE(cutoff_date)) AS dias_antiguedad_transaccional,
    DATE_DIFF('day', MAX(ts), ANY_VALUE(cutoff_date)) AS dias_desde_ultima_tx
FROM tx_filtered
GROUP BY id_cuenta
""")

# Indices de concentracion (HHI) por canal / comercio / tipo_cuenta:
# HHI cercano a 1 = toda la actividad concentrada en una sola categoria
# (util para detectar canalizacion tipica de cuentas mula).
for col, name in [("canal", "hhi_canal"), ("id_comercio", "hhi_comercio"), ("account_type", "hhi_tipo_cuenta")]:
    con.execute(f"""
    CREATE OR REPLACE TABLE hhi_{name} AS
    WITH counts AS (
        SELECT id_cuenta, {col} AS cat, COUNT(*) AS n
        FROM tx_filtered
        GROUP BY id_cuenta, {col}
    ),
    totals AS (
        SELECT id_cuenta, SUM(n) AS total FROM counts GROUP BY id_cuenta
    )
    SELECT c.id_cuenta, SUM(POWER(c.n * 1.0 / t.total, 2)) AS {name}
    FROM counts c JOIN totals t ON c.id_cuenta = t.id_cuenta
    GROUP BY c.id_cuenta
    """)

con.execute("""
CREATE OR REPLACE TABLE agg_tx AS
SELECT
    a.*,
    hc.hhi_canal,
    hm.hhi_comercio,
    ht.hhi_tipo_cuenta,
    a.num_tx_7d * 1.0 / NULLIF(a.num_tx_90d, 0) AS ratio_tx_7d_90d,
    a.monto_max / NULLIF(a.monto_avg, 0) AS ratio_pico_promedio
FROM agg_base a
LEFT JOIN hhi_hhi_canal hc ON a.id_cuenta = hc.id_cuenta
LEFT JOIN hhi_hhi_comercio hm ON a.id_cuenta = hm.id_cuenta
LEFT JOIN hhi_hhi_tipo_cuenta ht ON a.id_cuenta = ht.id_cuenta
""")

# ---------------------------------------------------------------------------
# 5. Union con cuentas SIN transacciones en la ventana (deben quedar en la
#    tabla final con NULLs/ceros, no desaparecer)
# ---------------------------------------------------------------------------

con.execute("""
CREATE OR REPLACE TABLE agg_tx_full AS
SELECT m.id_cuenta, m.split, m.is_fraud, m.tipo_fraude, m.cutoff_date, a.* EXCLUDE (id_cuenta, cutoff_date)
FROM accounts_master m
LEFT JOIN agg_tx a ON a.id_cuenta = m.id_cuenta
""")

# ---------------------------------------------------------------------------
# 6. Join con tablas estaticas (customers, credit_indebtedness,
#    device_mobile_activity, data_update_history)
# ---------------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TABLE final_features AS
SELECT
    f.*,
    c.id_cliente, c.edad, c.genero, c.ciudad, c.antiguedad_dias, c.segmento,
    c.nivel_educativo, c.ocupacion, c.tiene_2fa, c.producto_principal,
    ci.score_crediticio_externo, ci.dias_mora_max, ci.num_moras_12m, ci.capacidad_pago_pct,
    d.device_type, d.is_rooted_or_jailbreak, d.is_emulator,
    d.num_cambios_dispositivo_12m, d.dias_desde_ultimo_cambio,
    d.num_sesiones_30d, d.num_ciudades_acceso_30d, d.pct_accesos_fuera_ciudad,
    d.ip_country_ultimo,
    h.num_actualizaciones_12m, h.num_cambios_telefono, h.num_cambios_email,
    h.num_cambios_direccion, h.dias_desde_ultimo_kyc
FROM agg_tx_full f
LEFT JOIN read_parquet('{FILES["customers"]}') c ON c.id_cuenta = f.id_cuenta
LEFT JOIN read_parquet('{FILES["credit_indebtedness"]}') ci ON ci.id_cuenta = f.id_cuenta
LEFT JOIN read_parquet('{FILES["device_mobile_activity"]}') d ON d.id_cuenta = f.id_cuenta
LEFT JOIN read_parquet('{FILES["data_update_history"]}') h ON h.id_cuenta = f.id_cuenta
""")

n_rows, n_cols = con.execute("SELECT COUNT(*), (SELECT COUNT(*) FROM pragma_table_info('final_features')) FROM final_features").fetchone()
print(f"final_features: {n_rows} filas x {n_cols} columnas")

con.execute(f"COPY final_features TO '{OUT_PATH}' (FORMAT PARQUET)")
size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"Guardado: {OUT_PATH} ({size_mb:.1f} MB)")
print("Descarga este archivo desde el panel Output del notebook y subelo al chat.")
