"""B2. 'Selecciono en A, reporto en B' con A y B independientes. 600 repeticiones."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
V='/home/user/Declaracion_renta/val'
Pm=np.load(f'{V}/preds_pool.npy'); y=np.load(f'{V}/y_pool.npy')
names=open(f'{V}/names.txt').read().split('\n')
M,N=Pm.shape; PCT=100/44962
rng=np.random.default_rng(7)
pos=np.where(y==1)[0]; neg=np.where(y==0)[0]

def pk(yy,ss,k): return yy[np.argsort(-ss,kind='stable')[:k]].mean()

R=300
res={key:np.zeros(R) for key in ['maxA_pk','B_de_ganadorA_pk','B_de_ganadorA_ap','B_de_ganadorA_auc',
                                 'B_medio','B_max','B_ganadora_s1','rho_pkA_pkB','rho_apA_pkB','rho_aucA_pkB']}
acierto={'pk':0,'ap':0,'auc':0}
gan_por_pk=np.zeros(M); gan_por_ap=np.zeros(M)
for r in range(R):
    ip=rng.permutation(pos); ineg=rng.permutation(neg)
    A=np.concatenate([ip[:len(pos)//2], ineg[:len(neg)//2]])
    B=np.concatenate([ip[len(pos)//2:], ineg[len(neg)//2:]])
    kA=max(1,int(round(PCT*len(A)))); kB=max(1,int(round(PCT*len(B))))
    yA,yB=y[A],y[B]
    pkA=np.array([pk(yA,Pm[m,A],kA) for m in range(M)])
    pkB=np.array([pk(yB,Pm[m,B],kB) for m in range(M)])
    apA=np.array([average_precision_score(yA,Pm[m,A]) for m in range(M)])
    aucA=np.array([roc_auc_score(yA,Pm[m,A]) for m in range(M)])
    wpk,wap,wauc=pkA.argmax(),apA.argmax(),aucA.argmax()
    res['maxA_pk'][r]=pkA.max()
    res['B_de_ganadorA_pk'][r]=pkB[wpk]; res['B_de_ganadorA_ap'][r]=pkB[wap]; res['B_de_ganadorA_auc'][r]=pkB[wauc]
    res['B_medio'][r]=pkB.mean(); res['B_max'][r]=pkB.max(); res['B_ganadora_s1'][r]=pkB[0]
    res['rho_pkA_pkB'][r]=spearmanr(pkA,pkB).statistic
    res['rho_apA_pkB'][r]=spearmanr(apA,pkB).statistic
    res['rho_aucA_pkB'][r]=spearmanr(aucA,pkB).statistic
    gan_por_pk[wpk]+=1; gan_por_ap[wap]+=1
    best=pkB.argmax()
    acierto['pk']+= (wpk==best); acierto['ap']+=(wap==best); acierto['auc']+=(wauc==best)

print(f'M={M} configs, {R} repeticiones, kA=kB~{kA} (mismo percentil que top-100 de 44.962)')
print(f'\n--- Sesgo del ganador (winner s curse) ---')
print(f'max P@k en A (lo que se reportaria)      {res["maxA_pk"].mean():.4f}')
print(f'ese mismo modelo medido en B (honesto)   {res["B_de_ganadorA_pk"].mean():.4f}')
print(f'OPTIMISMO por seleccionar y reportar en el mismo set: '
      f'{(res["maxA_pk"].mean()-res["B_de_ganadorA_pk"].mean())*100:+.2f} pp')
print(f'P@k medio de las 12 configs en B         {res["B_medio"].mean():.4f}')
print(f'mejor posible en B (oraculo)             {res["B_max"].mean():.4f}')
print(f'ganadora_s1 (config de produccion) en B  {res["B_ganadora_s1"].mean():.4f}')
print(f'\n--- Que criterio de seleccion en A produce mejor P@k en B ---')
for c,lab in [('pk','P@k en A'),('ap','AP en A'),('auc','AUC en A')]:
    v=res[f'B_de_ganadorA_{c}']
    print(f'  seleccionar por {lab:9s} -> P@k en B = {v.mean():.4f} '
          f'(SD {v.std():.4f})  acierta el mejor real {acierto[c]/R:.1%} de las veces')
print(f'  no seleccionar (config fija)        -> P@k en B = {res["B_ganadora_s1"].mean():.4f}')
print(f'\n--- Correlacion de rangos entre lo medido en A y la verdad en B (12 configs) ---')
for c,lab in [('pk','P@k'),('ap','AP'),('auc','AUC')]:
    v=res[f'rho_{c}A_pkB' if c!='pk' else 'rho_pkA_pkB']
    print(f'  rho( {lab:4s} en A , P@k en B ) = {v.mean():+.3f}  (SD {v.std():.3f})')
print(f'\n--- Con que frecuencia gana cada config al seleccionar por P@k en A ---')
for i,n in enumerate(names):
    print(f'  {n:14s} por P@k {gan_por_pk[i]/R:6.1%}   por AP {gan_por_ap[i]/R:6.1%}')
np.save(f'{V}/res_sim.npy',np.array([res[k] for k in res]))
