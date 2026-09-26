import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
b=pd.read_csv(OUT+"biotx100_audit.csv"); y=b.triples_ok_full.to_numpy()
v1=b.classifier.to_numpy()
for name,col,t in (("V3 canon",'p3',0.21931),("V3 form",'pf1',0.21931),("V3 canon@0.003",'p3',0.003),("V3 canon@0.01",'p3',0.01)):
    v3=(b[col]>=t).astype(int).to_numpy()
    c1=int(((v1==y)&(v3!=y)).sum()); c2=int(((v1!=y)&(v3==y)).sum())
    p=binomtest(min(c1,c2),c1+c2,0.5).pvalue if c1+c2>0 else 1.0
    print(f"McNemar V1 vs {name}: V1-only-correct={c1} {name}-only-correct={c2} exact p={p:.2e}")
# bootstrap CIs
rng=np.random.default_rng(20260922)
def boot(pred,y,n=4000):
    f=[];pr=[];rc=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if y[i].sum()==0: continue
        f.append(f1_score(y[i],pred[i],zero_division=0)); pr.append(precision_score(y[i],pred[i],zero_division=0)); rc.append(recall_score(y[i],pred[i],zero_division=0))
    q=lambda a:(np.percentile(a,2.5),np.percentile(a,97.5))
    return q(f),q(pr),q(rc)
for name,pred in (("V1",v1),("V3@0.219",(b.p3>=0.21931).astype(int).to_numpy()),("V3@0.003",(b.p3>=0.003).astype(int).to_numpy()),("V3form@0.219",(b.pf1>=0.21931).astype(int).to_numpy())):
    f,pr,rc=boot(pred,y)
    print(f"{name:13} F1 {f1_score(y,pred):.4f} [{f[0]:.3f},{f[1]:.3f}]  P {precision_score(y,pred):.4f} [{pr[0]:.3f},{pr[1]:.3f}]  R {recall_score(y,pred):.4f} [{rc[0]:.3f},{rc[1]:.3f}]")
# V2 vs V3 head to head
b["p2"]=b[[c for c in b.columns if c.startswith("p_V2")]].mean(axis=1)
v2=(b.p2>=0.17953).astype(int).to_numpy(); v3=(b.p3>=0.21931).astype(int).to_numpy()
c1=int(((v2==y)&(v3!=y)).sum()); c2=int(((v2!=y)&(v3==y)).sum())
print(f"\nMcNemar V2 vs V3 (both at own prior-shift thr): V2-only={c1} V3-only={c2} disagreements={c1+c2}")
