import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import binomtest, fisher_exact
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
b=pd.read_csv(OUT+"biotx100_audit.csv"); y=b.triples_ok_full.to_numpy()
TH=0.21931
print("=== Biotx100: three query representations, V3 (mean of 3 seeds), thr 0.2193 ===")
for name,col in (("canonical term  ","p3"),("surface form(1st)","pf1"),("form REVERSED   ","prev"),("canon REVERSED  ","pcrev")):
    p=b[col].to_numpy(); pred=(p>=TH).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
    fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
    pv=binomtest(tp,tp+fp,0.65,alternative='greater').pvalue
    print(f"  {name} TP{tp:3d} FP{fp:3d} FN{fn:3d} TN{tn:3d}  P={pr:.4f} R={rc:.4f} F1={2*pr*rc/(pr+rc):.4f} AUPRC={average_precision_score(y,p):.4f} p={pv:.2e}")
print("\n=== argument-order sensitivity: forward vs reversed form query ===")
b["dir_gap"]=b.pf1-b.prev
print("positives mean gap %.3f | negatives mean gap %.3f"%(b[y==1].dir_gap.mean(),b[y==0].dir_gap.mean()))
print("rows where REVERSED scores higher than forward by >0.3:",int((b.dir_gap<-0.3).sum()))
sub=b[(y==1)&(b.pf1<TH)]
print("\n=== the %d positives V3 still misses with the FORM query ===" % len(sub))
for _,x in sub.sort_values("pf1").iterrows():
    print(f"  #{x['count']} pf1={x.pf1:.3f} reversed={x.prev:.3f} canon={x.p3:.3f} nhigh={x.n_high} [{x.species1_form} | {x.interaction_form} | {x.species2_form}]")
    print("     ",str(x.sentence)[:250])
print("\n=== of the 11 canonical-query FNs, how many are recovered by the form query? ===")
fn=b[(y==1)&(b.p3<TH)]
print("recovered:",int((fn.pf1>=TH).sum()),"/",len(fn))
print("of those, reversed-order triples (reversed query scores higher):",int((fn.dir_gap<0).sum()))

# ---------- PR table ----------
print("\n=== V3 precision-recall table on Biotx100 (canonical-term query, as deployed today) ===")
rows=[]
for t in [0.05,0.10,0.15,0.2193,0.30,0.40,0.50,0.60,0.75,0.90]:
    pred=(b.p3>=t).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
    pv=binomtest(tp,tp+fp,0.65,alternative='greater').pvalue if tp+fp>0 else 1
    rows.append(dict(thr=t,kept=tp+fp,TP=tp,FP=fp,FN=fn,TN=tn,precision=round(pr,4),recall=round(rc,4),
                     F1=round(2*pr*rc/max(pr+rc,1e-9),4),p_vs_accept_all=f"{pv:.1e}"))
print(pd.DataFrame(rows).to_string(index=False))
print("\n=== same, FORM query ===")
rows=[]
for t in [0.05,0.10,0.15,0.2193,0.30,0.40,0.50,0.60,0.75,0.90]:
    pred=(b.pf1>=t).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
    pv=binomtest(tp,tp+fp,0.65,alternative='greater').pvalue if tp+fp>0 else 1
    rows.append(dict(thr=t,kept=tp+fp,TP=tp,FP=fp,FN=fn,TN=tn,precision=round(pr,4),recall=round(rc,4),
                     F1=round(2*pr*rc/max(pr+rc,1e-9),4),p_vs_accept_all=f"{pv:.1e}"))
print(pd.DataFrame(rows).to_string(index=False))
