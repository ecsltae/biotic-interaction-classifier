import sys, re, numpy as np, pandas as pd
sys.path.insert(0,"/home/egaillac/MetaP/classifier/src")
from data.interaction_lexicon import score_sentence, count_species_mentions
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
b=pd.read_csv(OUT+"biotx100_scored.csv")
y=b.triples_ok_full.to_numpy()
b["p3"]=b[[c for c in b.columns if c.startswith("p_V3")]].mean(axis=1)
b["p3f"]=b[[c for c in b.columns if c.startswith("pform_V3")]].mean(axis=1)
b["p2"]=b[[c for c in b.columns if c.startswith("p_V2")]].mean(axis=1)
TH3=0.21931
b["v3"]=(b.p3>=TH3).astype(int)
b["v1"]=b.classifier.astype(int)
def mech(r):
    if r.triples_ok_full==1: return "positive"
    if r.species1_eval==0 or r.species2_eval==0: return "entity"
    if r.interaction_eval==0: return "predicate"
    return "pair-binding"
b["mech"]=b.apply(mech,axis=1)
b["cellV1"]=np.where((b.v1==1)&(y==1),"TP",np.where((b.v1==1)&(y==0),"FP",np.where((b.v1==0)&(y==1),"FN","TN")))
b["cellV3"]=np.where((b.v3==1)&(y==1),"TP",np.where((b.v3==1)&(y==0),"FP",np.where((b.v3==0)&(y==1),"FN","TN")))
print("=== negatives: mechanism x V1 cell ===")
print(pd.crosstab(b[y==0].mech,b[y==0].cellV1))
print("\n=== negatives: mechanism x V3 cell ===")
print(pd.crosstab(b[y==0].mech,b[y==0].cellV3))
print("\n=== V1 FPs (26) fixed by V3, by mechanism ===")
fp1=b[(b.cellV1=="FP")]
print(pd.crosstab(fp1.mech,fp1.cellV3))
print("\n=== the 3 V3 FPs ===")
for _,r in b[b.cellV3=="FP"].iterrows():
    print(f"  #{r['count']} p3={r.p3:.3f} mech={r.mech} [{r.species1_term} | {r.interaction_term} | {r.species2_term}] ({r.field})")
    print("    ",r.sentence[:300])
print("\n=== V3 FN on Biotx100 (11) ===")
for _,r in b[b.cellV3=="FN"].iterrows():
    print(f"  #{r['count']} p3={r.p3:.3f} p3form={r.p3f:.3f} [{r.species1_term} | {r.interaction_term} | {r.species2_term}] ({r.field}, {r.type})")
    print("    ",r.sentence[:330])
b.to_csv(OUT+"biotx100_audit.csv",index=False)
