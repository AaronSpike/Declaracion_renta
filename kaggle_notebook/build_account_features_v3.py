"""
Feature engineering a nivel de cuenta - Hackathon Asobancaria 2026 (Banco AuditPlus)
VERSION 2 (corregida tras EDA)

COMO USAR:
1. Crea/abre un Kaggle Notebook adjuntado a la competencia
   "hackaton-asobancaria-auditores-2026" (Add Data -> la competencia).
2. Pega TODO este archivo en una celda y ejecutalo.
   No necesitas subir ni descargar transactions.parquet: ya esta montado
   en /kaggle/input dentro del propio notebook.
3. Al terminar queda /kaggle/working/account_features.parquet (pocos MB).
   Descargalo del panel "Output" del notebook y subelo de vuelta al chat.

-----------------------------------------------------------------------------
DECISIONES DE DISENO (anti-leakage) - documentadas para el jurado
-----------------------------------------------------------------------------
1) FECHA DE CORTE UNICA Y COMUN = MAX(accounts_to_score.fecha_corte) = 2026-04-29.
   Se aplica la MISMA fecha de corte a las cuentas de entrenamiento y a las de
   evaluacion. Solo se usan transacciones con timestamp <= corte.

   Por que NO se usa fecha_confirmacion_fraude como corte por cuenta:
   el rango de confirmaciones va de 2025-10-11 a 2026-06-07, mientras que
   TODAS las cuentas a calificar tienen corte 2026-04-29. Si a cada cuenta
   fraudulenta se le recortara la ventana en su fecha de confirmacion y a las
   legitimas no, el LARGO DE LA VENTANA DE OBSERVACION quedaria correlacionado
   casi perfectamente con la etiqueta. El modelo aprenderia "ventana corta =
   fraude", que es un artefacto de construccion del dataset y no una senal de
   riesgo: colapsaria al aplicarse sobre el set de evaluacion, donde todas las
   ventanas miden lo mismo. El corte comun elimina ese artefacto y hace que las
   features de train y de scoring sean directamente comparables.

2) LIMITACION CONOCIDA (hallazgo de auditoria, se reporta, no se "parcha"):
   para las cuentas cuyo fraude fue confirmado ANTES del 2026-04-29, la ventana
   de observacion incluye comportamiento posterior al evento (rastro del fraude
   ya materializado y de la contencion del banco). Esto cumple la regla del reto
   (informacion <= fecha_corte) pero infla el desempeno frente a un uso de
   deteccion temprana real. Recomendacion para produccion: reentrenar con
   ventana de observacion y ventana de desempeno separadas.

3) LIMITACION CONOCIDA (hallazgo de auditoria):
   customers, credit_indebtedness, device_mobile_activity y data_update_history
   son SNAPSHOTS sin fecha por evento (no son logs). Sus campos de ventana movil
   (num_sesiones_30d, num_cambios_dispositivo_12m, ultimo_login, etc.) reflejan
   el estado al momento de la extraccion, que puede ser posterior al corte:
   25.587 cuentas (14,2%) tienen ultimo_login posterior al 2026-04-29, con
   maximo en 2026-06-07, es decir 39 dias despues. No es reconstruible el
   estado exacto al 2026-04-29 con los datos entregados. Se usan igual porque
   el reto las entrega como fuentes validas, pero se deja constancia del riesgo
   de fuga post-corte en estas fuentes.

5) CENSURA ANTI-FUGA DE LA RECENCIA DE ACCESO:
   calcular dias_desde_ultimo_login = corte - ultimo_login produce valores
   NEGATIVOS (de -1 a -39) en esas 25.587 cuentas, entregando al modelo
   informacion posterior al corte en forma directamente utilizable. Aqui se
   censura a NULO cuando el acceso es posterior al corte. Costo predictivo
   nulo (AUC univariado de la variable: 0,4921).

6) VALOR CENTINELA: dias_desde_ultimo_cambio vale 999 en el 90,06% de los
   registros. No son 999 dias: significa "sin cambio de dispositivo
   registrado". Se convierte a NULO y se agrega un indicador explicito, para
   no introducir una distancia falsa de ~900 dias frente a los cambios reales.

4) device_mobile_activity tiene MAS filas que cuentas (185.004 vs 180.000):
   4.019 cuentas tienen 2-3 dispositivos. Se AGREGA por cuenta antes de unir
   (un LEFT JOIN directo duplicaria cuentas y corromperia el dataset).
   El numero de dispositivos se conserva como feature (num_dispositivos).
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
# 1. Localizar archivos de entrada
# ---------------------------------------------------------------------------

def find_file(name):
    matches = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"No se encontro {name} bajo /kaggle/input. Verifica el Add Data.")
    return matches[0]

FILES = {k: find_file(f"{k}.parquet") for k in [
    "transactions", "customers", "credit_indebtedness", "device_mobile_activity",
    "data_update_history", "merchants", "fraud_labels_train", "accounts_to_score",
]}
for k, v in FILES.items():
    print(f"{k}: {v}")

OUT_PATH = "/kaggle/working/account_features.parquet"

con = duckdb.connect()
con.execute("PRAGMA threads=4")
# P0-a: sin temp_directory, DuckDB en memoria NO puede volcar a disco y muere por OOM
# al ordenar 40M de filas. Con esto degrada a disco en vez de abortar la corrida unica.
os.makedirs("/kaggle/working/ddbtmp", exist_ok=True)
con.execute("SET temp_directory='/kaggle/working/ddbtmp'")
con.execute("SET preserve_insertion_order=false")

# ---------------------------------------------------------------------------
# 2. Fecha de corte comun
# ---------------------------------------------------------------------------

CUTOFF = con.execute(f"""
    SELECT MAX(TRY_CAST(fecha_corte AS DATE)) FROM read_parquet('{FILES["accounts_to_score"]}')
""").fetchone()[0]
print(f"\nFECHA DE CORTE COMUN: {CUTOFF}")

# ---------------------------------------------------------------------------
# 3. Tabla maestra de cuentas (train + score) con la misma fecha de corte
# ---------------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TABLE accounts_master AS
SELECT id_cuenta, 'train' AS split, is_fraud, tipo_fraude,
       TRY_CAST(fecha_confirmacion_fraude AS DATE) AS fecha_confirmacion_fraude
FROM read_parquet('{FILES["fraud_labels_train"]}')
UNION ALL
SELECT id_cuenta, 'score' AS split, CAST(NULL AS INTEGER) AS is_fraud,
       CAST(NULL AS VARCHAR) AS tipo_fraude, CAST(NULL AS DATE) AS fecha_confirmacion_fraude
FROM read_parquet('{FILES["accounts_to_score"]}')
""")
print("accounts_master:", con.execute(
    "SELECT split, COUNT(*), AVG(is_fraud) FROM accounts_master GROUP BY split").fetchall())

# ---------------------------------------------------------------------------
# 4. Transacciones filtradas por el corte comun + datos de comercio
#    (filtro por constante: una sola pasada, sin join previo)
# ---------------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TABLE tx AS
SELECT
    t.id_cuenta,
    TRY_CAST(t.timestamp AS TIMESTAMP) AS ts,
    t.amount, t.account_type, t.canal,
    t.is_international, t.is_new_ip, t.ip_country, t.id_comercio,
    mc.merchant_category, mc.merchant_country, mc.merchant_risk_score
FROM read_parquet('{FILES["transactions"]}') t
LEFT JOIN read_parquet('{FILES["merchants"]}') mc ON mc.id_comercio = t.id_comercio
WHERE TRY_CAST(t.timestamp AS TIMESTAMP) <= TIMESTAMP '{CUTOFF} 23:59:59'
""")
print("transacciones dentro del corte:", con.execute("SELECT COUNT(*) FROM tx").fetchone()[0])
print("rango de fechas:", con.execute("SELECT MIN(ts), MAX(ts) FROM tx").fetchone())

# ---------------------------------------------------------------------------
# 5. Agregados transaccionales por cuenta
# ---------------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TABLE agg_base AS
SELECT
    id_cuenta,
    COUNT(*) AS num_tx_total,
    SUM(amount) AS monto_total,
    AVG(amount) AS monto_avg,
    STDDEV_POP(amount) AS monto_std,
    MAX(amount) AS monto_max,
    MIN(amount) AS monto_min,
    MEDIAN(amount) AS monto_mediana,
    QUANTILE_CONT(amount, 0.95) AS monto_p95,
    AVG(is_international) AS pct_internacional,
    AVG(is_new_ip) AS pct_ip_nueva,
    SUM(is_new_ip) AS num_ip_nuevas,
    COUNT(DISTINCT ip_country) AS num_paises_ip_distintos,
    COUNT(DISTINCT canal) AS num_canales_distintos,
    COUNT(DISTINCT id_comercio) AS num_comercios_distintos,
    COUNT(DISTINCT merchant_category) AS num_categorias_comercio,
    COUNT(DISTINCT merchant_country) AS num_paises_comercio,
    COUNT(DISTINCT account_type) AS num_tipos_cuenta_distintos,
    AVG(merchant_risk_score) AS merchant_risk_avg,
    MAX(merchant_risk_score) AS merchant_risk_max,
    STDDEV_POP(merchant_risk_score) AS merchant_risk_std,
    QUANTILE_CONT(merchant_risk_score, 0.9) AS merchant_risk_p90,
    AVG(CASE WHEN EXTRACT(hour FROM ts) BETWEEN 0 AND 5 THEN 1.0 ELSE 0.0 END) AS pct_horario_nocturno,
    AVG(CASE WHEN dayofweek(ts) IN (0, 6) THEN 1.0 ELSE 0.0 END) AS pct_fin_semana,
    -- ventanas recientes respecto al corte comun
    SUM(CASE WHEN ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 7 DAY  THEN 1 ELSE 0 END) AS num_tx_7d,
    SUM(CASE WHEN ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY THEN 1 ELSE 0 END) AS num_tx_30d,
    SUM(CASE WHEN ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 90 DAY THEN 1 ELSE 0 END) AS num_tx_90d,
    SUM(CASE WHEN ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 7 DAY  THEN amount ELSE 0 END) AS monto_7d,
    SUM(CASE WHEN ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY THEN amount ELSE 0 END) AS monto_30d,
    SUM(CASE WHEN ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 90 DAY THEN amount ELSE 0 END) AS monto_90d,
    SUM(CASE WHEN ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY THEN is_international ELSE 0 END) AS num_intl_30d,
    SUM(CASE WHEN ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY THEN is_new_ip ELSE 0 END) AS num_ip_nueva_30d,
    -- F4a ESTRUCTURACION: montos redondos. Comparacion por distancia, no por MOD
    -- sobre un CAST truncado (que daria falsos positivos con decimales).
    AVG(CASE WHEN ABS(amount - ROUND(amount/1000.0)*1000.0)     < 0.01 THEN 1.0 ELSE 0.0 END) AS pct_monto_redondo_1k,
    AVG(CASE WHEN ABS(amount - ROUND(amount/10000.0)*10000.0)   < 0.01 THEN 1.0 ELSE 0.0 END) AS pct_monto_redondo_10k,
    AVG(CASE WHEN ABS(amount - ROUND(amount/100000.0)*100000.0) < 0.01 THEN 1.0 ELSE 0.0 END) AS pct_monto_redondo_100k,
    -- F3b MULA: transaccion sin comercio = no es compra (transferencia / retiro).
    -- Ademas desambigua el HHI: hoy una cuenta 100% transferencias y una cuenta
    -- concentrada en un solo comercio dan las dos hhi_comercio = 1.
    AVG(CASE WHEN id_comercio IS NULL THEN 1.0 ELSE 0.0 END) AS pct_tx_sin_comercio,
    SUM(CASE WHEN id_comercio IS NULL THEN amount ELSE 0 END) / NULLIF(SUM(amount), 0) AS pct_monto_sin_comercio,
    -- F5 DIRECCIONALIDAD: no hay columna de sentido ni de contraparte en el esquema.
    -- Si amount trae signo, esto reconstruye entrada/salida (fan-in/fan-out de mula).
    -- Si amount es siempre positivo, quedan constantes y LightGBM las ignora: coste cero.
    AVG(CASE WHEN amount < 0 THEN 1.0 ELSE 0.0 END) AS pct_tx_monto_negativo,
    SUM(amount) / NULLIF(SUM(ABS(amount)), 0) AS flujo_neto_norm,
    SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS monto_entradas,
    SUM(CASE WHEN amount < 0 THEN -amount ELSE 0 END) AS monto_salidas,
    COUNT(*) FILTER (WHERE amount > 0) AS n_entradas,
    COUNT(*) FILTER (WHERE amount < 0) AS n_salidas,
    MIN(ts) AS primera_tx,
    MAX(ts) AS ultima_tx,
    DATE_DIFF('day', MIN(ts), TIMESTAMP '{CUTOFF}') AS dias_antiguedad_transaccional,
    DATE_DIFF('day', MAX(ts), TIMESTAMP '{CUTOFF}') AS dias_desde_ultima_tx,
    COUNT(DISTINCT DATE_TRUNC('day', ts)) AS num_dias_activos
FROM tx
GROUP BY id_cuenta
""")

# Indices de concentracion Herfindahl (HHI): 1 = toda la actividad en una sola
# categoria. Senal tipica de cuenta mula (embudo hacia un solo destino/canal).
for col, name in [("canal", "hhi_canal"), ("id_comercio", "hhi_comercio"),
                  ("account_type", "hhi_tipo_cuenta"), ("merchant_category", "hhi_categoria")]:
    con.execute(f"""
    CREATE OR REPLACE TABLE t_{name} AS
    WITH counts AS (
        SELECT id_cuenta, {col} AS cat, COUNT(*) AS n FROM tx
        WHERE {col} IS NOT NULL GROUP BY 1, 2
    ), totals AS (
        SELECT id_cuenta, SUM(n) AS total FROM counts GROUP BY 1
    )
    SELECT c.id_cuenta, SUM(POWER(c.n * 1.0 / t.total, 2)) AS {name}
    FROM counts c JOIN totals t USING (id_cuenta)
    GROUP BY c.id_cuenta
    """)


# ---------------------------------------------------------------------------
# 5-bis. FEATURES NUEVAS (revision del frente transaccional)
# ---------------------------------------------------------------------------

# F1 RAFAGA / VELOCIDAD INTRADIA / DORMANCIA.
# La granularidad mas fina del script era 7 dias; el cash-out de una toma de
# control ocurre en minutos. Ventana deslizante real (RANGE + INTERVAL), no
# baldes fijos, para que una rafaga a caballo entre dos horas no se parta.
con.execute(f"""
CREATE OR REPLACE TABLE agg_burst AS
WITH s AS (
    SELECT id_cuenta, ts, amount,
           COUNT(*)    OVER w1  AS c1h,
           SUM(amount) OVER w1  AS m1h,
           COUNT(*)    OVER w24 AS c24h,
           DATE_DIFF('second', LAG(ts) OVER (PARTITION BY id_cuenta ORDER BY ts), ts) AS gap_seg
    FROM tx
    WINDOW w1  AS (PARTITION BY id_cuenta ORDER BY ts RANGE BETWEEN INTERVAL 1  HOUR PRECEDING AND CURRENT ROW),
           w24 AS (PARTITION BY id_cuenta ORDER BY ts RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW)
)
SELECT id_cuenta,
    MAX(c1h)  AS max_tx_1h,
    MAX(m1h)  AS max_monto_1h,
    MAX(c24h) AS max_tx_24h,
    MAX(c1h)  FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS max_tx_1h_30d,
    MAX(m1h)  FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS max_monto_1h_30d,
    MIN(gap_seg) AS gap_min_seg,
    QUANTILE_CONT(gap_seg, 0.10) AS gap_p10_seg,
    MAX(gap_seg) / 86400.0 AS gap_max_dias,
    AVG(CASE WHEN gap_seg <=  300 THEN 1.0 ELSE 0.0 END) AS pct_tx_a_menos_5min,
    AVG(CASE WHEN gap_seg <= 3600 THEN 1.0 ELSE 0.0 END) AS pct_tx_a_menos_1h
FROM s GROUP BY id_cuenta
""")

# F2 DESVIACION CONTRA LA PROPIA LINEA BASE.
# Todo lo que habia era NIVEL. La cabeza de un ranking de fraude no la ocupan
# las cuentas grandes sino las que CAMBIARON. Compara ultimos 30d contra la
# historia previa de la MISMA cuenta, que es su propio grupo de control.
con.execute(f"""
CREATE OR REPLACE TABLE agg_delta AS
WITH b AS (
  SELECT id_cuenta,
    AVG(amount)        FILTER (WHERE ts <  TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS m_avg_h,
    STDDEV_POP(amount) FILTER (WHERE ts <  TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS m_std_h,
    AVG(amount)        FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS m_avg_r,
    MAX(amount)        FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS m_max_r,
    AVG(CASE WHEN EXTRACT(hour FROM ts) BETWEEN 0 AND 5 THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE ts <  TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS noc_h,
    AVG(CASE WHEN EXTRACT(hour FROM ts) BETWEEN 0 AND 5 THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS noc_r,
    AVG(is_new_ip * 1.0)        FILTER (WHERE ts <  TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS ipn_h,
    AVG(is_new_ip * 1.0)        FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS ipn_r,
    AVG(is_international * 1.0) FILTER (WHERE ts <  TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS intl_h,
    AVG(is_international * 1.0) FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS intl_r,
    AVG(merchant_risk_score)    FILTER (WHERE ts <  TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS mr_h,
    AVG(merchant_risk_score)    FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS mr_r,
    COUNT(*) FILTER (WHERE ts >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS n_r,
    COUNT(*) FILTER (WHERE ts <  TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS n_h,
    DATE_DIFF('day', MIN(ts), TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY)   AS dias_h
  FROM tx GROUP BY id_cuenta
)
SELECT id_cuenta,
    (m_max_r - m_avg_h) / NULLIF(m_std_h, 0)        AS z_monto_max_30d,
    m_avg_r / NULLIF(m_avg_h, 0)                    AS ratio_ticket_30d_hist,
    noc_r  - noc_h                                  AS delta_nocturno_30d,
    ipn_r  - ipn_h                                  AS delta_ip_nueva_30d,
    intl_r - intl_h                                 AS delta_intl_30d,
    mr_r   - mr_h                                   AS delta_merchant_risk_30d,
    (n_r / 30.0) / NULLIF(n_h * 1.0 / NULLIF(dias_h, 0), 0) AS aceleracion_30d_vs_hist,
    CASE WHEN n_h = 0 THEN 1 ELSE 0 END             AS sin_historia_previa_30d
FROM b
""")

# F3 CONTRAPARTES NUEVAS.
# El unico proxy de contraparte en el esquema es id_comercio. Comercio que
# aparece por primera vez en la ventana reciente = destino nuevo (toma de
# control) o punto de drenaje recien montado (mula).
con.execute(f"""
CREATE OR REPLACE TABLE agg_nuevos AS
WITH prim AS (
    SELECT id_cuenta, id_comercio, MIN(ts) AS primera, COUNT(*) AS n, SUM(amount) AS monto
    FROM tx WHERE id_comercio IS NOT NULL GROUP BY 1, 2
)
SELECT id_cuenta,
    COUNT(*) FILTER (WHERE primera >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS num_comercios_nuevos_30d,
    COUNT(*) FILTER (WHERE primera >= TIMESTAMP '{CUTOFF}' - INTERVAL  7 DAY) AS num_comercios_nuevos_7d,
    COUNT(*) FILTER (WHERE primera >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) * 1.0
        / NULLIF(COUNT(*), 0)                                        AS pct_comercios_nuevos_30d,
    COALESCE(SUM(monto) FILTER (WHERE primera >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY), 0)
        / NULLIF(SUM(monto), 0)                                      AS pct_monto_comercios_nuevos_30d,
    MAX(monto) / NULLIF(SUM(monto), 0)                               AS share_monto_top_comercio,
    MAX(n)     / NULLIF(SUM(n), 0.0)                                 AS share_tx_top_comercio
FROM prim GROUP BY id_cuenta
""")

# F4b MONTO REPETIDO EXACTO: pass-through automatizado (misma cifra N veces).
con.execute("""
CREATE OR REPLACE TABLE agg_repet AS
WITH r AS (SELECT id_cuenta, amount, COUNT(*) AS n FROM tx GROUP BY 1, 2)
SELECT id_cuenta,
    MAX(n) AS max_repeticiones_mismo_monto,
    SUM(CASE WHEN n >= 3 THEN n ELSE 0 END) * 1.0 / NULLIF(SUM(n), 0) AS pct_tx_monto_repetido,
    COUNT(*) * 1.0 / NULLIF(SUM(n), 0) AS ratio_montos_distintos_tx
FROM r GROUP BY id_cuenta
""")

con.execute("""
CREATE OR REPLACE TABLE agg_tx AS
SELECT a.*,
       c.hhi_canal, m.hhi_comercio, tc.hhi_tipo_cuenta, cat.hhi_categoria,
       b.* EXCLUDE (id_cuenta),
       dl.* EXCLUDE (id_cuenta),
       nv.* EXCLUDE (id_cuenta),
       rp.* EXCLUDE (id_cuenta),
       a.num_tx_7d  * 1.0 / NULLIF(a.num_tx_90d, 0) AS ratio_tx_7d_90d,
       a.num_tx_30d * 1.0 / NULLIF(a.num_tx_total, 0) AS ratio_tx_30d_total,
       a.monto_30d / NULLIF(a.monto_total, 0) AS ratio_monto_30d_total,
       a.monto_max / NULLIF(a.monto_avg, 0) AS ratio_pico_promedio,
       a.monto_std / NULLIF(a.monto_avg, 0) AS coef_variacion_monto,
       a.num_tx_total * 1.0 / NULLIF(a.num_dias_activos, 0) AS tx_por_dia_activo,
       a.num_comercios_distintos * 1.0 / NULLIF(a.num_tx_total, 0) AS ratio_comercios_tx
FROM agg_base a
LEFT JOIN t_hhi_canal c USING (id_cuenta)
LEFT JOIN t_hhi_comercio m USING (id_cuenta)
LEFT JOIN t_hhi_tipo_cuenta tc USING (id_cuenta)
LEFT JOIN t_hhi_categoria cat USING (id_cuenta)
LEFT JOIN agg_burst  b  USING (id_cuenta)
LEFT JOIN agg_delta  dl USING (id_cuenta)
LEFT JOIN agg_nuevos nv USING (id_cuenta)
LEFT JOIN agg_repet  rp USING (id_cuenta)
""")

# ---------------------------------------------------------------------------
# 6. device_mobile_activity AGREGADO por cuenta (evita duplicar filas)
# ---------------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TABLE dev_agg AS
SELECT
    id_cuenta,
    COUNT(*) AS num_dispositivos,
    COUNT(DISTINCT device_type) AS num_tipos_dispositivo,
    MAX(is_rooted_or_jailbreak) AS is_rooted_or_jailbreak,
    MAX(is_emulator) AS is_emulator,
    MAX(num_cambios_dispositivo_12m) AS num_cambios_dispositivo_12m,
    -- 999 es centinela de "sin cambio registrado", no una cantidad de dias
    MIN(NULLIF(dias_desde_ultimo_cambio, 999)) AS dias_desde_ultimo_cambio,
    MIN(CASE WHEN dias_desde_ultimo_cambio = 999 THEN 1 ELSE 0 END) AS sin_cambio_dispositivo,
    SUM(num_sesiones_30d) AS num_sesiones_30d,
    MAX(num_ciudades_acceso_30d) AS num_ciudades_acceso_30d,
    MAX(pct_accesos_fuera_ciudad) AS pct_accesos_fuera_ciudad,
    MAX(TRY_CAST(ultimo_login AS TIMESTAMP)) AS ultimo_login,
    ANY_VALUE(device_type) AS device_type,
    ANY_VALUE(ip_country_ultimo) AS ip_country_ultimo,
    MAX(CASE WHEN ip_country_ultimo <> 'CO' THEN 1 ELSE 0 END) AS ip_ultima_fuera_co
FROM read_parquet('{FILES["device_mobile_activity"]}')
GROUP BY id_cuenta
""")
print("dev_agg filas:", con.execute("SELECT COUNT(*) FROM dev_agg").fetchone()[0], "(debe ser 180000)")

# ---------------------------------------------------------------------------
# 7. Tabla final: cuentas x (features transaccionales + snapshots)
#    LEFT JOIN desde accounts_master para no perder cuentas sin transacciones
# ---------------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TABLE final_features AS
SELECT
    am.id_cuenta, am.split, am.is_fraud, am.tipo_fraude, am.fecha_confirmacion_fraude,
    DATE '{CUTOFF}' AS fecha_corte,
    a.* EXCLUDE (id_cuenta),
    c.* EXCLUDE (id_cuenta),
    ci.* EXCLUDE (id_cuenta),
    d.* EXCLUDE (id_cuenta, ultimo_login),
    h.* EXCLUDE (id_cuenta),
    -- Censura anti-fuga: un acceso posterior al corte no es observable en la
    -- fecha de decision, asi que la recencia queda como NULO en vez de negativa.
    CASE WHEN d.ultimo_login <= TIMESTAMP '{CUTOFF} 23:59:59'
         THEN DATE_DIFF('day', d.ultimo_login, TIMESTAMP '{CUTOFF}')
         ELSE NULL END AS dias_desde_ultimo_login
FROM accounts_master am
LEFT JOIN agg_tx a USING (id_cuenta)
LEFT JOIN read_parquet('{FILES["customers"]}') c USING (id_cuenta)
LEFT JOIN read_parquet('{FILES["credit_indebtedness"]}') ci USING (id_cuenta)
LEFT JOIN dev_agg d USING (id_cuenta)
LEFT JOIN read_parquet('{FILES["data_update_history"]}') h USING (id_cuenta)
""")

n_rows = con.execute("SELECT COUNT(*) FROM final_features").fetchone()[0]
n_cols = len(con.execute("DESCRIBE final_features").fetchall())
print(f"\nfinal_features: {n_rows} filas x {n_cols} columnas  (filas esperadas: 180000)")

print("\nsin transacciones en la ventana:",
      con.execute("SELECT COUNT(*) FROM final_features WHERE num_tx_total IS NULL").fetchone()[0])
print("tasa de fraude por split:",
      con.execute("SELECT split, COUNT(*), AVG(is_fraud) FROM final_features GROUP BY split").fetchall())

# ---------------------------------------------------------------------------
# 8. Export
#    - DOUBLE -> FLOAT: mitad de bytes, cero perdida util (LightGBM binea a 255).
#    - Se excluyen los TIMESTAMP crudos: primera_tx / ultima_tx no aportan nada
#      que dias_antiguedad_transaccional y dias_desde_ultima_tx no digan ya, y
#      ultimo_login es informacion POSTERIOR al corte (hasta +39 dias) que
#      reintroduciria por la puerta de atras la fuga que se censuro a proposito.
#    - fecha_confirmacion_fraude se excluye: solo existe para los positivos,
#      es una etiqueta disfrazada de fecha.
#    - Se ESCRIBE ANTES de validar: si una asercion falla, la corrida unica de
#      40M de filas no se pierde.
# ---------------------------------------------------------------------------
cols = con.execute("DESCRIBE final_features").fetchall()
EXCLUIR = {"primera_tx", "ultima_tx", "ultimo_login", "fecha_confirmacion_fraude", "id_cliente"}
sel = []
for name, dtype, *_ in cols:
    if name in EXCLUIR:
        continue
    sel.append(f'CAST("{name}" AS FLOAT) AS "{name}"' if dtype == "DOUBLE" else f'"{name}"')
con.execute(f"COPY (SELECT {', '.join(sel)} FROM final_features) TO '{OUT_PATH}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD)")
print(f"\nGuardado: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB, {len(sel)} columnas)")
print("Descarga este archivo desde el panel Output del notebook y subelo al chat.")

assert n_rows == 180000, f"ERROR: se esperaban 180000 filas, hay {n_rows} (revisa duplicados en los joins)"
