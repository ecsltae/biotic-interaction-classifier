import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import binomtest, fisher_exact
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
b=pd.read_csv(OUT+"biotx100_audit.csv"); y=b.triples_ok_full.to_numpy()
b["pmax"]=b[["pf1","prev"]].max(axis=1)
TH=0.21931
print("=== direction-agnostic scoring: max(forward, reversed) form query ===")
for name,col in (("form forward","pf1"),("max(fwd,rev)","pmax"),("canonical","p3")):
    p=b[col].to_numpy(); pred=(p>=TH).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
    print(f"  {name:14} TP{tp:3d} FP{fp:3d} FN{fn:3d} TN{tn:3d} P={pr:.4f} R={rc:.4f} F1={2*pr*rc/(pr+rc):.4f} AUPRC={average_precision_score(y,p):.4f}")
pos=b[y==1]
flip=pos[(pos.pf1<TH)&(pos.prev>=TH)]
print(f"\ngold positives V3 rejects forward but accepts REVERSED: {len(flip)} -> #{list(flip['count'])}")

print("\n=== FULL PR TABLE, V3 canonical query, thresholds spanning the actual score range ===")
ths=[0.0003,0.001,0.003,0.01,0.03,0.05,0.1,0.2193,0.5,0.75,0.9,0.99]
rows=[]
for t in ths:
    pred=(b.p3>=t).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
    pv=binomtest(tp,tp+fp,0.65,alternative='greater').pvalue if tp+fp>0 else 1.0
    rows.append(dict(thr=t,kept=tp+fp,pct_kept=f"{(tp+fp)}%",TP=tp,FP=fp,FN=fn,TN=tn,
        precision=round(pr,4),recall=round(rc,4),F1=round(2*pr*rc/max(pr+rc,1e-9),4),p=f"{pv:.1e}"))
print(pd.DataFrame(rows).to_string(index=False))
print("\nscore mass: fraction of all 100 rows with p3 in [0.05,0.60] =",
      round(float(((b.p3>=0.05)&(b.p3<=0.60)).mean()),3),
      "| p3<0.05:",int((b.p3<0.05).sum()),"| p3>0.6:",int((b.p3>0.6).sum()))
print("positives with p3<0.05:",int(((y==1)&(b.p3<0.05)).sum()),"negatives with p3>0.6:",int(((y==0)&(b.p3>0.6)).sum()))

# ---- reject50 with first-element form ----
r=pd.read_csv(OUT+"reject50_audit.csv"); yr=r.gold_label.to_numpy()
r["pmax"]=r[["pf1","prev"]].max(axis=1)
print("\n=== reject50 (11 pos / 39 neg): V3 by query representation, thr 0.2193 ===")
for name,col in (("canonical  ","p3"),("form (1st) ","pf1"),("max(fwd,rev)","pmax"),("stringified list (as in eval_v3.py)","p3f")):
    p=r[col].to_numpy(); pred=(p>=TH).astype(int)
    tp=int(((pred==1)&(yr==1)).sum()); fp=int(((pred==1)&(yr==0)).sum()); fn=int(((pred==0)&(yr==1)).sum()); tn=int(((pred==0)&(yr==0)).sum())
    print(f"  {name:36} TP{tp:3d} FP{fp:3d} FN{fn:3d} TN{tn:3d} recall={tp/max(tp+fn,1):.3f} spec={tn/max(tn+fp,1):.3f} AUPRC={average_precision_score(yr,p):.4f}")
print("\n=== combined high-rank test (Biotx100 positives + reject50 positives), form query ===")
A=pd.DataFrame(dict(nh=b[y==1].n_high, hit=(b[y==1].pf1>=TH).astype(int)))
B=pd.DataFrame(dict(nh=r[yr==1].n_high, hit=(r[yr==1].pf1>=TH).astype(int)))
C=pd.concat([A,B])
print(C.groupby(C.nh>0).hit.agg(["size","sum","mean"]))
t=pd.crosstab(C.nh>0,C.hit); print(t); print("fisher p =",fisher_exact(t.values)[1])
r.to_csv(OUT+"reject50_audit.csv",index=False); b.to_csv(OUT+"biotx100_audit.csv",index=False)
