import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import binomtest
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
b=pd.read_csv(OUT+"biotx100_scored.csv"); y=b.triples_ok_full.to_numpy()
def logit(p): return np.log(p/(1-p))
def thr(tp,te): return 1/(1+np.exp(logit(te)-logit(tp)))
T={"V3":thr(0.3428772394389507,0.65),"V2":thr(0.28895,0.65)}
rows=[]
for pref,lbl in (("p_","canonical term"),("pform_","surface form")):
    for tag in ("V3","V2"):
        p=b[[c for c in b.columns if c.startswith(pref) and tag in c]].mean(axis=1).to_numpy()
        t=T[tag]; pred=(p>=t).astype(int)
        tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
        fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
        pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
        bt=binomtest(tp,tp+fp,0.65,alternative='greater').pvalue
        rows.append(dict(query=lbl,model=tag,thr=round(t,4),TP=tp,FP=fp,FN=fn,TN=tn,
                         precision=round(pr,4),recall=round(rc,4),F1=round(2*pr*rc/(pr+rc),4),
                         AUPRC=round(average_precision_score(y,p),4),p_binom=round(bt,5)))
print(pd.DataFrame(rows).to_string(index=False))
# agreement between representations
for tag in ("V3","V2"):
    pa=b[[c for c in b.columns if c.startswith("p_") and tag in c]].mean(axis=1).to_numpy()
    pb=b[[c for c in b.columns if c.startswith("pform_") and tag in c]].mean(axis=1).to_numpy()
    print(tag,"mean |delta prob| term->form:",round(np.abs(pa-pb).mean(),4),
          " rows flipping decision at thr:",int(((pa>=T[tag])!=(pb>=T[tag])).sum()),
          " spearman:",round(pd.Series(pa).corr(pd.Series(pb),method='spearman'),4))
