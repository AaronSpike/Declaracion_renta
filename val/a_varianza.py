"""A. Cuanta varianza REAL tiene Precision@100. Remuestreo sobre predicciones ya guardadas."""
import numpy as np
D='/home/user/Declaracion_renta/data'
rng=np.random.default_rng(0)
oof=np.load(f'{D}/oof_dev.npy'); ydev=np.load(f'{D}/y_dev.npy')
ph =np.load(f'{D}/pred_hold.npy'); yh=np.load(f'{D}/y_hold.npy')
N_SCORE=44962; K=100
print(f'dev  n={len(ydev)} base={ydev.mean():.4f}')
print(f'hold n={len(yh)} base={yh.mean():.4f}')

def top_prec(y,s,k):
    return y[np.argsort(-s,kind='stable')[:k]].mean()

# --- A1: la SD de UNA realizacion vs el SE de la media de 500 sims
def sim(y,s,n_sub,k,n_sim=4000):
    idx=np.arange(len(y)); out=np.empty(n_sim)
    for i in range(n_sim):
        sel=rng.choice(idx,size=n_sub,replace=False)
        out[i]=y[sel[np.argsort(-s[sel],kind='stable')[:k]]].mean()
    return out

o=sim(ydev,oof,N_SCORE,K)
print('\n--- A1. Distribucion de P@100 sobre un set de calificacion de 44.962 (dev OOF) ---')
print(f'media {o.mean():.4f}  SD de UNA realizacion {o.std():.4f}  '
      f'SE de la media de 500 sims {o.std()/np.sqrt(500):.4f}')
print(f'p05-p95 [{np.percentile(o,5):.3f}, {np.percentile(o,95):.3f}]  '
      f'min {o.min():.2f} max {o.max():.2f}')
print(f'aciertos: media {o.mean()*100:.1f} de 100, rango {o.min()*100:.0f}-{o.max()*100:.0f}')
print(f'RATIO SD_realizacion / SE_reportado = {np.sqrt(500):.1f}x')

# --- A2: cota binomial pura (ruido irreducible de mirar solo 100 casos)
for p in (0.15,0.20,0.2135,0.25):
    print(f'  binomial n=100 p={p:.3f}: SD={np.sqrt(p*(1-p)/100):.4f} '
          f'IC95 aprox [{p-1.96*np.sqrt(p*(1-p)/100):.3f},{p+1.96*np.sqrt(p*(1-p)/100):.3f}]')

# --- A3: el holdout reportado (k=60 por el reescalado del codigo)
k_h=max(1,int(round(K*len(yh)/N_SCORE)))
ph_p=top_prec(yh,ph,k_h)
print(f'\n--- A3. Holdout: el codigo evalua k={k_h}, no 100 ---')
print(f'P@{k_h} holdout = {ph_p:.4f}  = {int(ph_p*k_h)}/{k_h} aciertos')
print(f'SE binomial de ESA cifra = {np.sqrt(ph_p*(1-ph_p)/k_h):.4f}  '
      f'-> IC95 [{ph_p-1.96*np.sqrt(ph_p*(1-ph_p)/k_h):.3f},{ph_p+1.96*np.sqrt(ph_p*(1-ph_p)/k_h):.3f}]')
# bootstrap de cuentas en el holdout
bs=np.empty(4000)
n=len(yh); idx=np.arange(n)
for i in range(4000):
    s=rng.choice(idx,size=n,replace=True)
    bs[i]=top_prec(yh[s],ph[s],k_h)
print(f'bootstrap (remuestreo de cuentas) SD={bs.std():.4f} IC95 [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]')

# --- A4: estabilidad de AP y AUC en el mismo holdout, para comparar
from sklearn.metrics import average_precision_score, roc_auc_score
apb=np.empty(1000); aucb=np.empty(1000)
for i in range(1000):
    s=rng.choice(idx,size=n,replace=True)
    apb[i]=average_precision_score(yh[s],ph[s]); aucb[i]=roc_auc_score(yh[s],ph[s])
ap0=average_precision_score(yh,ph); auc0=roc_auc_score(yh,ph)
print(f'\n--- A4. Estabilidad relativa de las metricas en el mismo holdout ---')
print(f'P@{k_h}: valor {ph_p:.4f}  SD {bs.std():.4f}  CV {bs.std()/ph_p:.3f}')
print(f'AP    : valor {ap0:.4f}  SD {apb.std():.4f}  CV {apb.std()/ap0:.3f}')
print(f'AUC   : valor {auc0:.4f}  SD {aucb.std():.4f}  CV {aucb.std()/auc0:.3f}')
print(f'-> AP es {bs.std()/ph_p/(apb.std()/ap0):.1f}x mas estable que P@k en terminos relativos')
