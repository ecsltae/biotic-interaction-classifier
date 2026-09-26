import sqlite3, numpy as np, pandas as pd
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
con=sqlite3.connect("/home/egaillac/MetaP/classifier/data/processed/ott_index.sqlite")
def rank(oid):
    if not isinstance(oid,str): return None
    o=oid.replace("ott:","").strip()
    r=con.execute("select rank,pref from concepts where ott_id=?",(o,)).fetchone()
    return r
b=pd.read_csv(OUT+"biotx100_audit.csv")
miss=0
for c in ("species1","species2"):
    rr=[rank(x) for x in b[c+"_id"]]
    b[c+"_rank"]=[x[0] if x else "UNRESOLVED" for x in rr]
    b[c+"_pref"]=[x[1] if x else "" for x in rr]
print(b.species1_rank.value_counts().to_dict())
print(b.species2_rank.value_counts().to_dict())
ABOVE={"genus","family","order","class","phylum","kingdom","domain","subfamily","tribe","superfamily","infraorder","suborder","subclass","subphylum","no rank","subgenus","infraclass","superorder","cohort","section","series","division","superphylum","UNRESOLVED"}
b["s1_high"]=~b.species1_rank.isin(["species","subspecies","varietas","forma","species group","species subgroup","no rank - terminal"])
b["s2_high"]=~b.species2_rank.isin(["species","subspecies","varietas","forma","species group","species subgroup","no rank - terminal"])
b["n_high"]=b.s1_high.astype(int)+b.s2_high.astype(int)
b["pair_high"]=b.n_high>0
b.to_csv(OUT+"biotx100_audit.csv",index=False)
y=b.triples_ok_full.to_numpy()
print("\n=== V3 recall on positives by taxonomic rank of the pair ===")
pos=b[y==1]
for k,g in pos.groupby("n_high"):
    print(f"  n_high_rank_taxa={k}: n={len(g)}  V3 accepts {int((g.cellV3=='TP').sum())}  recall={(g.cellV3=='TP').mean():.3f}  mean p3={g.p3.mean():.3f}")
print("\n=== V3 precision on negatives by rank ===")
neg=b[y==0]
for k,g in neg.groupby("n_high"):
    print(f"  n_high_rank_taxa={k}: n={len(g)}  V3 rejects {int((g.cellV3=='TN').sum())}  mean p3={g.p3.mean():.3f}")
from scipy.stats import fisher_exact
t=pd.crosstab(pos.pair_high,pos.cellV3)
print("\npositives: pair_high x V3 cell\n",t)
print("fisher:",fisher_exact(t.values))
