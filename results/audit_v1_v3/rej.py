import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
r=pd.read_csv(OUT+"reject50_scored.csv")
y=r.gold_label.to_numpy()
print("n",len(r),"pos",y.sum())
def logit(p): return np.log(p/(1-p))
def thr(tp,te): return 1/(1+np.exp(logit(te)-logit(tp)))
tv3=thr(0.3428772394389507,y.mean()); tv2=thr(0.28895,y.mean())
print("prior-shift thr on reject50: V3 %.4f V2 %.4f"%(tv3,tv2))
for pref in ("p_","pform_"):
    for tag,t in (("V3",tv3),("V2",tv2)):
        cols=[c for c in r.columns if c.startswith(pref) and tag in c]
        p=r[cols].mean(axis=1).to_numpy()
        for thrname,tt in (("prior",t),("biotx0.219",0.2193 if tag=="V3" else 0.1795)):
            pred=(p>=tt).astype(int)
            tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
            fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
            print(f"{pref}{tag} thr={thrname}({tt:.3f}) TP{tp:3d} FP{fp:3d} FN{fn:3d} TN{tn:3d} recall={tp/max(tp+fn,1):.3f} AUPRC={average_precision_score(y,p):.4f}")
