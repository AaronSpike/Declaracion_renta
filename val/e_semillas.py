"""E. Promediar semillas: la unica intervencion con diseno pareado. Verificacion independiente."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
V='/home/user/Declaracion_renta/val'
Pm=np.load(f'{V}/preds_pool.npy'); y=np.load(f'{V}/y_pool.npy'); names=open(f'{V}/names.txt').read().split('\n')
S=[names.index(f'ganadora_s{s}') for s in range(1,7)]
PCT=100/44962; rng=np.random.default_rng(9); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
def pk(yy,ss,k): return yy[np.argsort(-ss,kind='stable')[:k]].mean()
prom={n:Pm[S[:n]].mean(0) for n in (1,2,3,5,6)}
prom['rank6']=np.mean([rankdata(Pm[i]) for i in S],axis=0)
R=500; res={k:np.zeros(R) for k in list(prom)+['ind_media']}
for r in range(R):
    ip=rng.permutation(pos); ineg=rng.permutation(neg)
    B=np.concatenate([ip[:len(pos)//2], ineg[:len(neg)//2]]); k=max(1,int(round(PCT*len(B))))
    for key,s in prom.items(): res[key][r]=pk(y[B],s[B],k)
    res['ind_media'][r]=np.mean([pk(y[B],Pm[i,B],k) for i in S])
print(f'500 mitades de 33.760 cuentas, k={k}. Base {y.mean():.4f}')
b=res['ind_media']
for key in [1,2,3,5,6,'rank6']:
    d=res[key]-b
    lab=f'promedio de {key} semillas' if key!='rank6' else 'promedio de rangos (6)'
    print(f'{lab:26s} P@k {res[key].mean():.4f}  vs semilla individual {b.mean():.4f}  '
          f'dif {d.mean()*100:+5.2f} pp  SD pareada {d.std()*100:4.2f}  t={d.mean()/(d.std()/np.sqrt(R)):5.1f}')
print(f'\nAUC(pool): 1 semilla {roc_auc_score(y,prom[1]):.4f}  6 semillas {roc_auc_score(y,prom[6]):.4f}  '
      f'| AP: {average_precision_score(y,prom[1]):.4f} -> {average_precision_score(y,prom[6]):.4f}')
