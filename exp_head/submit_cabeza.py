"""Pipeline de ENTREGA optimizado para la cabeza del ranking.

Dos cambios frente a src/baseline_final.py, ambos medidos:

1. REAJUSTE CON EL 100% DE LAS ETIQUETAS.
   baseline_final.py entrena los modelos de entrega sobre folds del 80% de
   desarrollo: cada modelo ve el 64% de las etiquetas. El holdout sirve para
   MEDIR, no hay razon para que ademas recorte los datos de la entrega.
   Curva medida (P@300 OOF, mismo percentil que P@100 sobre 44.962):
       67% de etiquetas -> 0,2000
       80% de etiquetas -> 0,2183
   (ver exp_head/e2.log para el resto de la curva)

2. PROMEDIADO DE SEMILLAS.
   El top-100 de una sola semilla es en buena parte un sorteo. Promediar S
   semillas divide entre S la varianza del ruido de score y con ella el
   riesgo de que la entrega caiga en el lado malo del sorteo (ver exp_head/e1.log).

El holdout de C3 se mantiene, pero solo para reportar la cifra honesta.
"""
import numpy as np, pandas as pd, lightgbm as lgb, sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/home/user/Declaracion_renta/src')

DATA="/home/user/Declaracion_renta/data"; OUT="/home/user/Declaracion_renta/submissions"
SEEDS=[1,2,3,4,5,6,7,8]     # subir mientras haya tiempo; el coste es lineal
NFOLDS=5
PARAMS=dict(objective="binary",learning_rate=0.01,num_leaves=15,min_child_samples=200,
            feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.0,
            n_estimators=1200,verbose=-1,n_jobs=2)

# --- construir X_all, y_all, test_df exactamente como baseline_final.py ---
# (importar las funciones de alli o copiar el bloque construir()/preparar_dispositivos())
# Aqui se asume que X_all (135.038 filas), y_all y test_df ya existen.

def entregar(X_all, y_all, test_df, FEATURES):
    pred=np.zeros(len(test_df)); n=0
    for sd in SEEDS:
        skf=StratifiedKFold(NFOLDS,shuffle=True,random_state=sd)
        for tr,_ in skf.split(X_all,y_all):          # <- 100% de las etiquetas disponibles
            m=lgb.LGBMClassifier(**{**PARAMS,'seed':sd,'bagging_seed':sd,
                                    'feature_fraction_seed':sd})
            m.fit(X_all.iloc[tr],y_all[tr])
            pred+=m.predict_proba(test_df[FEATURES])[:,1]; n+=1
    pred/=n
    print(f'{n} modelos promediados; empates en el top-200: '
          f'{200-len(np.unique(np.sort(pred)[-200:]))}')
    return pred
