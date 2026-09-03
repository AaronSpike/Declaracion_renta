# Parche de poda para src/baseline_final.py

Evidencia: chi2 de independencia sobre 135.038 filas y AUC univariado.
Ninguna de estas 17 variables muestra relacion con is_fraud, y entre las tres
juntas se llevan el 37,0% del gain del modelo en produccion.

## Cambio 1 — eliminar las categoricas (todas independientes del fraude)

    genero              chi2 p=0,6621
    ciudad              chi2 p=0,6620   <- 2a feature por gain del modelo actual
    segmento            chi2 p=0,9197
    nivel_educativo     chi2 p=0,8539
    ocupacion           chi2 p=0,3844
    producto_principal  chi2 p=0,6687
    device_type         chi2 p=0,2869
    ip_country_ultimo   chi2 p=0,0499   <- no sobrevive Bonferroni (0,0499x8=0,40)

En baseline_final.py:

    CATS = []   # antes: 8 categoricas

y el bucle de casteo a Categorical queda sin efecto.

## Cambio 2 — eliminar las numericas planas y sus derivadas

    edad                    AUC 0,5002
    antiguedad_dias         AUC 0,5033
    dias_desde_ultimo_kyc   AUC 0,5012
    tiene_2fa               AUC 0,5041
    ip_ultima_fuera_co      AUC 0,4928

derivadas de estas, tambien fuera:

    sin_2fa, kyc_vencido_1y, cuenta_nueva_90d (gain 0,0), intensidad_cambios

Reemplazar la lista FEATURES por una lista explicita:

    FEATURES = [
        "num_actualizaciones_12m", "num_cambios_telefono", "num_cambios_email",
        "num_cambios_direccion", "num_dispositivos", "num_tipos_dispositivo",
        "is_rooted_or_jailbreak", "is_emulator", "dias_desde_ultimo_cambio",
        "sin_cambio_dispositivo", "num_sesiones_30d", "num_ciudades_acceso_30d",
        "pct_accesos_fuera_ciudad", "dias_desde_ultimo_login",
        "cambios_contacto_total", "ratio_cambios_contacto", "sesiones_por_ciudad",
    ]

## Cambio 3 — quitar el duplicado exacto

    num_dispositivos == num_cambios_dispositivo_12m + 1  en el 100,0% de las filas
    (correlacion 1,0000)

Son la misma variable. Con num_tipos_dispositivo (corr 0,74) el modelo tiene
tres copias de la misma senal, que con feature_fraction=0,6 se sortean como si
fueran independientes. Dejar solo num_dispositivos.

## Cambio 4 — revisar feature_fraction

feature_fraction=0,6 se sintonizo con 35 features (21 por arbol). Con 17
features pasa a 10 por arbol. Conviene reevaluar en {0,6 0,8 1,0}.
