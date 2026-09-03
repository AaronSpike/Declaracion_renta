"""B3. Detalle: verdad por config, ruido pareado, curva de sesgo vs numero de candidatos."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
V='/home/user/Declaracion_renta/val'
Pm=np.load(f'{V}/preds_pool.npy'); y=np.load(f'{V}/y_pool.npy')
names=open(f'{V}/names.txt').read().split('\n')
M,N=Pm.shape; PCT=100/44962
rng=np.random.default_rng(3)
pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
def pk(yy,ss,k): return yy[np.argsort(-ss,kind='stable')[:k]].mean()

R=400
PKB=np.zeros((R,M)); PKA=np.zeros((R,M))
for r in range(R):
    ip=rng.permutation(pos); ineg=rng.permutation(neg)
    A=np.concatenate([ip[:len(pos)//2], ineg[:len(neg)//2]])
    B=np.concatenate([ip[len(pos)//2:], ineg[len(neg)//2:]])
    kA=max(1,int(round(PCT*len(A)))); kB=max(1,int(round(PCT*len(B))))
    for m in range(M):
        PKA[r,m]=pk(y[A],Pm[m,A],kA); PKB[r,m]=pk(y[B],Pm[m,B],kB)

print(f'k={kA} por mitad ({len(A)} cuentas). {R} particiones A/B.')
print('\n--- Calidad "verdadera" de cada config (media sobre 400 mitades) vs metricas globales ---')
print(f'{"config":15s} {"P@k medio":>10s} {"SD/rep":>8s} {"AP(pool)":>9s} {"AUC(pool)":>10s}')
orden=np.argsort(-PKB.mean(0))
for m in orden:
    print(f'{names[m]:15s} {PKB[:,m].mean():10.4f} {PKB[:,m].std():8.4f} '
          f'{average_precision_score(y,Pm[m]):9.4f} {roc_auc_score(y,Pm[m]):10.4f}')

print('\n--- Ruido PAREADO: diferencia de P@k entre dos configs medida en la misma mitad ---')
pares=[('ganadora_s1','someros'),('ganadora_s1','sobreajuste'),('ganadora_s1','lento_reg'),
       ('ganadora_s1','ganadora_s2'),('ganadora_s1','corta_400'),('ganadora_s1','pos_w5')]
for a,b in pares:
    i,j=names.index(a),names.index(b); d=PKA[:,i]-PKA[:,j]
    print(f'  {a} - {b:12s}: dif media {d.mean()*100:+6.2f} pp  SD del par {d.std()*100:5.2f} pp  '
          f'|  se ve al reves en {(np.sign(d)!=np.sign(d.mean())).mean():5.1%} de las mitades')

print('\n--- Cuanto sesgo agrega probar M candidatos (todos evaluados y reportado el mejor) ---')
seeds=[names.index(f'ganadora_s{s}') for s in range(1,7)]
for Mtry in [2,3,5,8,12]:
    opt=[]
    for r in range(R):
        c=rng.choice(M,size=Mtry,replace=False)
        w=PKA[r,c].argmax(); opt.append(PKA[r,c][w]-PKB[r,c[w]])
    print(f'  M={Mtry:2d} candidatos cualesquiera : optimismo medio {np.mean(opt)*100:+5.2f} pp')
opt=[]
for r in range(R):
    w=PKA[r,seeds].argmax(); opt.append(PKA[r,seeds][w]-PKB[r,seeds[w]])
print(f'  M= 6 SOLO SEMILLAS (misma calidad real): optimismo medio {np.mean(opt)*100:+5.2f} pp '
      f'<- puro ruido, no hay nada que elegir')

print('\n--- Tamano de evaluacion necesario para detectar una diferencia real ---')
i,j=names.index('ganadora_s1'),names.index('someros')
sd_par=(PKA[:,i]-PKA[:,j]).std()
print(f'  con k={kA} (mitad de 33.760 cuentas) la SD de la diferencia pareada es {sd_par*100:.2f} pp')
print(f'  minima diferencia detectable al 95% (2 SD): {2*sd_par*100:.2f} pp')
for dif in [0.02,0.03,0.05]:
    kn=kA*(2*sd_par/dif)**2
    print(f'  para detectar {dif*100:.0f} pp harian falta k~{kn:.0f} '
          f'-> set de evaluacion de ~{kn/PCT/1000:.0f}k cuentas (hay 135k etiquetadas)')
