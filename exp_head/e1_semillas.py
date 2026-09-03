"""E1/E2: cuanto vale promediar semillas, y rango vs probabilidad.
Estimador: OOF sobre las 135.038 filas, P@300 = mismo percentil que P@100/44.962.
Comparaciones PAREADAS (mismas particiones, mismos datos) para bajar varianza."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/home/user/Declaracion_renta/exp_head')
from common import cargar, pk, hits, K_EVAL
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

df,FE,y=cargar(); X=df[FE]
P=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=200,
       feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.,
       n_estimators=400,verbose=-1,n_jobs=2)
SPLITS=[42,7,2024]; SEEDS=[1,2,3,4,5]
ITERS=[100,200,300,400]
res={}
t0=time.time()
for sp in SPLITS:
    skf=StratifiedKFold(5,shuffle=True,random_state=sp)
    folds=list(skf.split(X,y))
    oofs=[]; oof_it={t:[] for t in ITERS}
    for sd in SEEDS:
        oof=np.zeros(len(X)); ooft={t:np.zeros(len(X)) for t in ITERS}
        for tr,va in folds:
            m=lgb.LGBMClassifier(**{**P,'seed':sd,'bagging_seed':sd,'feature_fraction_seed':sd})
            m.fit(X.iloc[tr],y[tr])
            for t in ITERS:
                ooft[t][va]=m.predict_proba(X.iloc[va],num_iteration=t)[:,1]
            oof[va]=ooft[400][va]
        oofs.append(oof)
        for t in ITERS: oof_it[t].append(ooft[t])
        print(f'  split{sp} seed{sd}  P@300={pk(y,oof):.4f} AUC={roc_auc_score(y,oof):.4f}  ({time.time()-t0:.0f}s)',flush=True)
    oofs=np.array(oofs)
    ind=[pk(y,o) for o in oofs]
    prob=pk(y,oofs.mean(0))
    rank=pk(y,np.mean([rankdata(o) for o in oofs],axis=0))
    # promedio de 2 y de 3 semillas para ver la curva
    p2=np.mean([pk(y,oofs[[i,j]].mean(0)) for i in range(5) for j in range(i+1,5)])
    p3=np.mean([pk(y,oofs[[0,1,2]].mean(0)),pk(y,oofs[[1,2,3]].mean(0)),pk(y,oofs[[2,3,4]].mean(0))])
    # solapamiento de top-300 entre semillas
    tops=[set(np.argsort(-o)[:K_EVAL]) for o in oofs]
    ov=np.mean([len(tops[i]&tops[j])/K_EVAL for i in range(5) for j in range(i+1,5)])
    curva={t:pk(y,np.array(oof_it[t]).mean(0)) for t in ITERS}
    res[sp]=dict(ind=ind,prob=prob,rank=rank,p2=p2,p3=p3,ov=ov,curva=curva,
                 auc_ind=np.mean([roc_auc_score(y,o) for o in oofs]),auc_prob=roc_auc_score(y,oofs.mean(0)))
    print(f'SPLIT {sp}: individual {np.mean(ind):.4f} (sd {np.std(ind,ddof=1):.4f} rango {min(ind):.3f}-{max(ind):.3f}) '
          f'| 2sem {p2:.4f} | 3sem {p3:.4f} | 5sem-prob {prob:.4f} | 5sem-rango {rank:.4f} | solape top300 {ov:.3f}',flush=True)
    print(f'   curva arboles: '+'  '.join(f'{t}:{v:.4f}' for t,v in curva.items()),flush=True)

print('\n=== RESUMEN (3 particiones) ===')
ind=np.array([np.mean(res[s]['ind']) for s in SPLITS])
sdw=np.array([np.std(res[s]['ind'],ddof=1) for s in SPLITS])
prob=np.array([res[s]['prob'] for s in SPLITS]); rank=np.array([res[s]['rank'] for s in SPLITS])
p2=np.array([res[s]['p2'] for s in SPLITS]); p3=np.array([res[s]['p3'] for s in SPLITS])
print(f'P@300 semilla unica (media)      {ind.mean():.4f}')
print(f'  desv. entre semillas (misma part.) {sdw.mean():.4f}  <- ruido autoinfligido')
print(f'P@300 promedio 2 semillas        {p2.mean():.4f}  ({(p2-ind).mean()*100:+.2f} pp)')
print(f'P@300 promedio 3 semillas        {p3.mean():.4f}  ({(p3-ind).mean()*100:+.2f} pp)')
print(f'P@300 promedio 5 semillas (prob) {prob.mean():.4f}  ({(prob-ind).mean()*100:+.2f} pp) pareado sd {np.std(prob-ind,ddof=1):.4f}')
print(f'P@300 promedio 5 semillas (rango){rank.mean():.4f}  ({(rank-prob).mean()*100:+.2f} pp vs prob)')
print(f'solape medio top-300 entre semillas {np.mean([res[s]["ov"] for s in SPLITS]):.3f}')
print(f'AUC ind {np.mean([res[s]["auc_ind"] for s in SPLITS]):.4f} -> AUC 5sem {np.mean([res[s]["auc_prob"] for s in SPLITS]):.4f}')
np.save('/home/user/Declaracion_renta/exp_head/e1_res.npy',res,allow_pickle=True)
