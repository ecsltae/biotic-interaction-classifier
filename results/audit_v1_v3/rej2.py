import ast, sqlite3, numpy as np, pandas as pd
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
con=sqlite3.connect("/home/egaillac/MetaP/classifier/data/processed/ott_index.sqlite")
def rk(o):
    r=con.execute("select rank,pref from concepts where ott_id=?",(o.replace("ott:","").strip(),)).fetchone()
    return r or ("UNRESOLVED","")
r=pd.read_csv(OUT+"reject50_scored.csv")
r["p3"]=r[[c for c in r.columns if c.startswith("p_V3")]].mean(axis=1)
r["p3f"]=r[[c for c in r.columns if c.startswith("pform_V3")]].mean(axis=1)
r["p2f"]=r[[c for c in r.columns if c.startswith("pform_V2")]].mean(axis=1)
ids=r.triplet_key.str.split(";")
r["s1rank"]=[rk(x[0])[0] for x in ids]; r["s2rank"]=[rk(x[1])[0] for x in ids]
SP={"species","subspecies","varietas","forma","species group","no rank - terminal"}
r["n_high"]=(~r.s1rank.isin(SP)).astype(int)+(~r.s2rank.isin(SP)).astype(int)
y=r.gold_label.to_numpy()
print("rank counts s1",r.s1rank.value_counts().to_dict())
print("rank counts s2",r.s2rank.value_counts().to_dict())
print("\n=== reject50 positives (11): V3 score with form-query, thr 0.219 ===")
for _,x in r[y==1].sort_values("p3f").iterrows():
    print(f"  {x.id} p3_canon={x.p3:.3f} p3_form={x.p3f:.3f} nhigh={x.n_high} [{x.species1} | {x.interaction} | {x.species2}]")
    print(f"     forms {x.species1_form} | {x.interaction_form} | {x.species2_form}")
    print("     ",str(x.sentence)[:300])
print("\n=== rank vs recall on the 11 positives (form query, thr .219) ===")
pos=r[y==1].copy(); pos["hit"]=(pos.p3f>=0.2193).astype(int)
print(pos.groupby("n_high").hit.agg(["size","sum","mean"]))
print("\n=== negatives: V3 form-query FPs at thr .219 ===")
for _,x in r[(y==0)&(r.p3f>=0.2193)].iterrows():
    print(f"  {x.id} p3f={x.p3f:.3f} s1e={x.species1_eval} s2e={x.species2_eval} ie={x.interaction_eval} [{x.species1} | {x.interaction} | {x.species2}]")
    print("     ",str(x.sentence)[:260])
print("\n=== negatives by failure layer (from component evals) ===")
neg=r[y==0].copy()
def lay(x):
    if x.species1_eval==0 or x.species2_eval==0: return "entity"
    if x.interaction_eval==0: return "predicate"
    return "pair-binding"
neg["layer"]=neg.apply(lay,axis=1)
print(neg.layer.value_counts())
neg["rej"]=(neg.p3f<0.2193).astype(int)
print(neg.groupby("layer").rej.agg(["size","sum","mean"]))
r.to_csv(OUT+"reject50_audit.csv",index=False)
