import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, f1_score
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
b=pd.read_csv(OUT+"biotx100_scored.csv")
y=b.triples_ok_full.to_numpy()
print("n=",len(b),"pos=",y.sum(),"prior=",y.mean())
def logit(p): return np.log(p/(1-p))
def thr(tp,te): return 1/(1+np.exp(logit(te)-logit(tp)))
tv3=thr(0.3428772394389507, y.mean()); tv2=thr(0.28895, y.mean())
print("thr V3 %.4f  thr V2 %.4f"%(tv3,tv2))
cols=[c for c in b.columns if c.startswith("p_")]
for c in cols:
    p=b[c].to_numpy(); t=tv3 if "V3" in c else tv2
    pred=(p>=t).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
    fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f1=2*pr*rc/max(pr+rc,1e-9)
    print(f"{c:10} thr={t:.3f} TP{tp:3d} FP{fp:3d} FN{fn:3d} TN{tn:3d}  P={pr:.4f} R={rc:.4f} F1={f1:.4f} AUPRC={average_precision_score(y,p):.4f}")
# mean-of-seeds ensemble
for tag,t in (("V3",tv3),("V2",tv2)):
    p=b[[c for c in cols if tag in c]].mean(axis=1).to_numpy()
    pred=(p>=t).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
    fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
    print(f"{tag}_mean3  thr={t:.3f} TP{tp:3d} FP{fp:3d} FN{fn:3d} TN{tn:3d}  P={pr:.4f} R={rc:.4f} F1={2*pr*rc/(pr+rc):.4f} AUPRC={average_precision_score(y,p):.4f}")
# V1
v1=b.classifier.to_numpy()
tp=int(((v1==1)&(y==1)).sum()); fp=int(((v1==1)&(y==0)).sum())
fn=int(((v1==0)&(y==1)).sum()); tn=int(((v1==0)&(y==0)).sum())
print(f"V1 recorded TP{tp} FP{fp} FN{fn} TN{tn} P={tp/(tp+fp):.4f} R={tp/(tp+fn):.4f} F1={2*tp/(2*tp+fp+fn):.4f}")
