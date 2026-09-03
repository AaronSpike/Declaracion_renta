# PARCHE PARA src/baseline_final.py  (reemplaza el bloque "cifra oficial")
# ---------------------------------------------------------------------------
# Antes: la prediccion de entrega salia de 5 modelos entrenados sobre folds
# del 80% de desarrollo -> cada modelo ve el 64% de las etiquetas, y una sola
# semilla decide que cuentas quedan en la cabeza.
#
# Despues: el holdout se sigue usando para MEDIR (cifra honesta), pero la
# entrega se reajusta con el 100% de las etiquetas y se promedian S semillas.
# El coste es lineal en S y no hay riesgo de fuga: no se usa ninguna etiqueta
# que no estuviera ya disponible.

SEEDS_ENTREGA = [1, 2, 3, 4, 5, 6, 7, 8]   # subir mientras alcance el tiempo

# 1) medicion honesta (igual que antes, pero promediando semillas tambien aqui
#    para que la cifra reportada corresponda al modelo que de verdad se entrega)
pred_hold = np.zeros(len(X_hold))
for sd in SEEDS_ENTREGA:
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=sd).split(X_dev, y_dev):
        m = lgb.LGBMClassifier(**{**PARAMS, "seed": sd, "bagging_seed": sd,
                                  "feature_fraction_seed": sd})
        m.fit(X_dev.iloc[tr], y_dev[tr])
        pred_hold += m.predict_proba(X_hold)[:, 1]
pred_hold /= (5 * len(SEEDS_ENTREGA))

# 2) entrega: 100% de las etiquetas
pred_test = np.zeros(len(test_df))
n_mod = 0
for sd in SEEDS_ENTREGA:
    for tr, _ in StratifiedKFold(5, shuffle=True, random_state=sd).split(X_all, y_all):
        m = lgb.LGBMClassifier(**{**PARAMS, "seed": sd, "bagging_seed": sd,
                                  "feature_fraction_seed": sd})
        m.fit(X_all.iloc[tr], y_all[tr])
        pred_test += m.predict_proba(test_df[FEATURES])[:, 1]; n_mod += 1
pred_test /= n_mod
print(f"entrega: {n_mod} modelos, 100% de las etiquetas")

# 3) intervalo honesto para la cifra reportada (sustituye a p_at_k)
#    P@k tiene desviacion binomial sqrt(p(1-p)/k). Con k=60 sobre el holdout
#    eso son ~5,4 pp: reportarlo con +/-0,0015 es precision inventada.
def p_at_k_honesto(y_true, y_pred, k):
    p = y_true[np.argsort(-y_pred)[:k]].mean()
    return p, np.sqrt(p * (1 - p) / k)
