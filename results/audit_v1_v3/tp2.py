import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
b=pd.read_csv(OUT+"biotx100_audit.csv"); y=b.triples_ok_full.to_numpy()
pos=b[y==1].copy()
print("=== V3 recall (canonical-term query) by #canonical terms literally ABSENT from the sentence ===")
g=pos.groupby("nterm_missing").agg(n=("p3","size"), acc=("cellV3",lambda s:(s=="TP").sum()), meanp=("p3","mean"))
g["recall"]=g.acc/g.n; print(g)
t=pd.crosstab(pos.nterm_missing>=2, pos.cellV3)
print(t); print("fisher p =", fisher_exact(t.values)[1])
print("\n=== V3 on the 35 negatives: p3 by mechanism ===")
neg=b[y==0]
print(neg.groupby("mech").p3.agg(["size","mean","median","max"]))
print("\n=== the 23 EXTRA true negatives V3 makes that V1 missed ===")
extra=b[(b.cellV1=="FP")&(b.cellV3=="TN")]
print("n =",len(extra))
print(extra.mech.value_counts().to_dict())
print("field:",extra.field.value_counts().to_dict(),"type:",extra.type.value_counts().to_dict())
print("mean p3 = %.4f  max p3 = %.4f"%(extra.p3.mean(),extra.p3.max()))
print("lexicon fires on:",int(extra.lex_hit.sum()),"/",len(extra))
print("same clause:",int(extra.same_clause.sum()),"/",len(extra))
for _,x in extra.sort_values("p3",ascending=False).iterrows():
    print(f"  #{x['count']:>3} p3={x.p3:.3f} {x.mech:<12} [{x.species1_term} | {x.interaction_term} | {x.species2_term}] ({x.field})")
    print("      ", str(x.sentence)[:230])
