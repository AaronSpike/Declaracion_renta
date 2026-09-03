"""D. Loteria del holdout + verificacion del estimador de error usado en el codigo."""
import numpy as np
V='/home/user/Declaracion_renta/val'; D='/home/user/Declaracion_renta/data'
Pm=np.load(f'{V}/preds_pool.npy'); y=np.load(f'{V}/y_pool.npy'); names=open(f'{V}/names.txt').read().split('\n')
s=Pm[0]; rng=np.random.default_rng(5); PCT=100/44962
pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
def pk(yy,ss,k): return yy[np.argsort(-ss,kind='stable')[:k]].mean()

# D1: si el holdout hubiera salido con otra semilla
n_h=27008; k_h=60
out=[]
for i in range(2000):
    npos=int(round(n_h*y.mean()))
    sel=np.concatenate([rng.choice(pos,npos,replace=False),rng.choice(neg,n_h-npos,replace=False)])
    out.append(pk(y[sel],s[sel],k_h))
out=np.array(out)
print('--- D1. La misma configuracion, holdouts de 27.008 cuentas distintos (k=60) ---')
print(f'P@60 media {out.mean():.4f}  SD {out.std():.4f}  '
      f'p05-p95 [{np.percentile(out,5):.3f},{np.percentile(out,95):.3f}]  min {out.min():.3f} max {out.max():.3f}')
print(f'-> cambiar random_state del holdout mueve la cifra reportada en un rango de '
      f'{(np.percentile(out,95)-np.percentile(out,5))*100:.1f} pp sin cambiar NADA del modelo')

# D2: el estimador p_at_k del codigo en produccion
N_SCORE=44962; K=100
rng2=np.random.default_rng(42)
def p_at_k_codigo(y_true,y_pred,n_sim=500):
    n=len(y_true); n_sub=min(N_SCORE,n); k=max(1,int(round(K*n_sub/N_SCORE)))
    idx=np.arange(n); o=np.empty(n_sim)
    for i in range(n_sim):
        sel=rng2.choice(idx,size=n_sub,replace=False)
        o[i]=y_true[sel[np.argsort(-y_pred[sel])[:k]]].mean()
    return o.mean(), o.std()/np.sqrt(n_sim), o.std()
yh=np.load(f'{D}/y_hold.npy'); ph=np.load(f'{D}/pred_hold.npy')
m,se,sd=p_at_k_codigo(yh,ph)
print('\n--- D2. Que devuelve exactamente p_at_k() sobre el holdout de 27.008 ---')
print(f'media {m:.4f}   se reportado {se:.6f}   SD entre simulaciones {sd:.6f}')
print('   n_sub = min(44962, 27008) = 27008 = n  ->  rng.choice(sin reemplazo, tamano n) es una '
      'PERMUTACION:\n   las 500 "simulaciones" evaluan el mismo top-60. El +/- reportado es cero por construccion.')
yd=np.load(f'{D}/y_dev.npy'); od=np.load(f'{D}/oof_dev.npy')
m2,se2,sd2=p_at_k_codigo(yd,od)
print(f'\n   sobre desarrollo (n=108.030): media {m2:.4f}  se reportado {se2:.4f}  SD real de una '
      f'realizacion {sd2:.4f}  ({sd2/se2:.0f}x mayor)')
