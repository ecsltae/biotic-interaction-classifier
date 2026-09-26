#!/usr/bin/env python3
"""Build the V3 training set: add the supervision the V2 set provably lacked.

Measured gaps in the V2 (20,000-row) set:
  * ZERO passages carry more than one labelled pair, so nothing teaches
    "right sentence, wrong pair" -- the failure mode that is 16 of 35
    deployment errors. This was caused by sampling one row per taxon pair.
  * No reversed triples, so direction is learned only incidentally.
  * 155,588 pool passages were never labelled; the student's uncertainty band
    [0.15,0.85] holds 8,219 triplets the teacher has never seen.

Three additions, all excluding benchmark passages and benchmark taxon pairs:
  A. PAIR-CONTRAST   same passage, different candidate pair (teacher-labelled)
  B. REVERSAL        gold-order triples with the arguments swapped; for a
                     directed relation the reverse is almost always false, so
                     these are free hard negatives AND direction supervision
  C. ACTIVE          uncertainty-band triplets, highest information per call
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEED = 20260901
rng = np.random.RandomState(SEED)

SYMMETRIC = {"symbiosis","symbiotically interacts with","interacts with",
             "associated with","commensalist of","co-occurs with","mutualism"}

def norm(s): return " ".join(str(s).split()).strip()
def first(x):
    if isinstance(x,(list,np.ndarray)): return str(x[0]).strip() if len(x) else ""
    return str(x).strip() if pd.notna(x) else ""

pool = pd.read_parquet(REPO/"data/training/distill/med25_pool.parquet")
for c in ("species1_form","species2_form","interaction_form"): pool[c]=pool[c].map(first)
pool["passage"]=pool.passage.map(norm)
pool = pool[(pool.species1_form!="")&(pool.species2_form!="")&(pool.interaction_form!="")]

# ---- benchmark guards (text AND taxon pair) ----
btxt, bpair = set(), set()
for f,tc,kw,pc in [("data/evaluation/biotic_interaction_test_set.csv","sentence",{},None),
                   ("data/evaluation/test500_paired.csv","sentence",{},("species1","species2")),
                   ("globi-relax_passages-triplets_2024-02-28_curation_EP.tsv","sentence",dict(sep="\t",encoding="latin-1"),("species1_term","species2_term")),
                   ("data/evaluation/biotx_retrieval_eval_100.csv","sentence",dict(sep=";",encoding="utf-8-sig"),("species1_term","species2_term"))]:
    p=REPO/f
    if not p.exists(): continue
    x=pd.read_csv(p,**kw)
    if tc in x.columns: btxt |= set(x[tc].astype(str).map(norm))
    if pc and pc[0] in x.columns:
        bpair |= {tuple(sorted([norm(a).lower(),norm(b).lower()]))
                  for a,b in zip(x[pc[0]],x[pc[1]]) if pd.notna(a) and pd.notna(b)}
for f in (REPO/"data/evaluation").glob("*curation_EP*.tsv"):
    try:
        x=pd.read_csv(f,sep="\t",encoding="latin-1"); btxt |= set(x["sentence"].astype(str).map(norm))
        if "species1_term" in x.columns:
            bpair |= {tuple(sorted([norm(a).lower(),norm(b).lower()])) for a,b in zip(x.species1_term,x.species2_term) if pd.notna(a) and pd.notna(b)}
    except Exception: pass

pool["pk"]=[tuple(sorted([a.lower(),b.lower()])) for a,b in zip(pool.species1_form,pool.species2_form)]
pool = pool[~pool.passage.isin(btxt) & ~pool.pk.isin(bpair)]
print(f"eligible pool after benchmark guards: {len(pool):,} passages")

old = pd.read_csv(REPO/"data/training/distill/student_train.csv")
old_keys = set(zip(old.text.map(norm), old.source_species.astype(str), old.target_species.astype(str)))

def rows(df, kind):
    return pd.DataFrame({"text":df.passage, "source_species":df.species1_form,
                         "target_species":df.species2_form,
                         "interaction_type":df.interaction_form,
                         "triplet_key":df.triplet_key, "doc_id":df.doc_id, "kind":kind})

# ---- A. pair-contrast: every row of every passage that carries >1 pair ----
n_per = pool.groupby("passage").triplet_key.transform("nunique")
A = rows(pool[n_per>1], "pair_contrast")
A = A[~pd.Series([k in old_keys for k in zip(A.text, A.source_species.astype(str), A.target_species.astype(str))], index=A.index)]
print(f"A. pair-contrast rows      : {len(A):,} over {A.text.nunique():,} passages")

# ---- B. reversal negatives from the EXISTING labelled positives ----
lab = pd.read_csv(REPO/"data/training/distill/teacher_labels.csv")
pos = old.reset_index().rename(columns={"index":"row"}).merge(lab, on="row")
pos = pos[(pos.label_y==1) & (~pos.interaction_type.str.lower().isin(SYMMETRIC))]
B = pd.DataFrame({"text":pos.text, "source_species":pos.target_species,
                  "target_species":pos.source_species, "interaction_type":pos.interaction_type,
                  "triplet_key":pos.get("triplet_key",""), "doc_id":pos.get("doc_id",""),
                  "kind":"reversal"})
print(f"B. reversal candidates     : {len(B):,} (from directed positives; symmetric relations excluded)")

# ---- C. active learning: uncertainty band, never labelled ----
sc = pd.read_csv(REPO/"experiments/knowledge_graph/results/verified_kg/all_candidates_scored.csv")
unc = set(sc[(sc.max_conf>=0.15)&(sc.max_conf<=0.85)].triplet_key)
C = rows(pool[pool.triplet_key.isin(unc)], "active")
C = C[~pd.Series([k in old_keys for k in zip(C.text, C.source_species.astype(str), C.target_species.astype(str))], index=C.index)]
C = C.drop_duplicates("triplet_key")
print(f"C. uncertainty-band actives: {len(C):,}")

new = pd.concat([A,B,C], ignore_index=True).drop_duplicates(
        subset=["text","source_species","target_species","interaction_type"])
out = REPO/"data/training/distill/v3_to_label.csv"
new.to_csv(out, index=False)
print(f"\nwrote {out}: {len(new):,} rows to label")
print(new.kind.value_counts().to_string())
print(f"\nat 2.44 it/s -> {len(new)/2.44/3600:.1f} GPU-hours")
