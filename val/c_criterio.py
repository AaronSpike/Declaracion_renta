"""C. Que criterio medido en A predice mejor la P@100 en B."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
V='/home/user/Declaracion_renta/val'
Pm=np.load(f'{V}/preds_pool.npy'); y=np.load(f'{V}/y_pool.npy')
names=open(f'{V}/names.txt').read().split('\n'); M=Pm.shape[0]; PCT=100/44962
rng=np.random.default_rng(21); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
def pk(yy,ss,k): return yy[np.argsort(-ss,kind='stable')[:k]].mean()
def pauc(yy,ss,frac=0.05):  # AUC parcial en la cabeza: recall entre los peor rankeados
    n=len(yy); k=int(n*frac); o=np.argsort(-ss,kind='stable')[:k]
    return yy[o].sum()/yy.sum()

CRIT=['P@k100','P@k300','P@k750','P@k1500','P@2pct','AP','AUC','recall@5pct']
R=300
selB={c:[] for c in CRIT}; rho={c:[] for c in CRIT}; fijo=[]
for r in range(R):
    ip=rng.permutation(pos); ineg=rng.permutation(neg)
    A=np.concatenate([ip[:len(pos)//2], ineg[:len(neg)//2]]); B=np.concatenate([ip[len(pos)//2:], ineg[len(neg)//2:]])
    yA,yB=y[A],y[B]; nA=len(A); kB=max(1,int(round(PCT*len(B))))
    pkB=np.array([pk(yB,Pm[m,B],kB) for m in range(M)])
    val={c:np.zeros(M) for c in CRIT}
    for m in range(M):
        sA=Pm[m,A]
        val['P@k100'][m]=pk(yA,sA,int(round(PCT*nA)))
        val['P@k300'][m]=pk(yA,sA,int(round(3*PCT*nA)))
        val['P@k750'][m]=pk(yA,sA,int(round(7.5*PCT*nA)))
        val['P@k1500'][m]=pk(yA,sA,int(round(15*PCT*nA)))
        val['P@2pct'][m]=pk(yA,sA,int(nA*0.02))
        val['AP'][m]=average_precision_score(yA,sA)
        val['AUC'][m]=roc_auc_score(yA,sA)
        val['recall@5pct'][m]=pauc(yA,sA)
    for c in CRIT:
        selB[c].append(pkB[val[c].argmax()]); rho[c].append(spearmanr(val[c],pkB).statistic)
    fijo.append(pkB[0])
print(f'{R} particiones A/B de 33.760 cuentas. Se selecciona 1 de {M} configs en A, se mide P@100 en B.\n')
print(f'{"criterio en A":14s} {"P@100 en B":>11s} {"rho con verdad":>15s}')
for c in CRIT:
    print(f'{c:14s} {np.mean(selB[c]):11.4f} {np.mean(rho[c]):15.3f}')
print(f'{"(no elegir)":14s} {np.mean(fijo):11.4f}')
