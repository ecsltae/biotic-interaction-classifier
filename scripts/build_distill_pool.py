#!/usr/bin/env python3
"""Sample a PAIR-DIVERSE slice of the SIBiLS retrieval pool for teacher labelling.

Why pair-diverse: the paired training corpora carry 606 (D3) and 11,602 (D1)
distinct taxon pairs against 79,460 in the retrieval pool. Supervised pair
conditioning failed on D3 because the pair slot was ~10 tokens of out-of-
distribution noise. Sampling by PAIR rather than by row is the fix.

Exclusions, both mandatory:
  - benchmark passage text (normalised exact match)
  - benchmark TAXON PAIRS — 22.7% of test299 pairs occur in this pool, and
    since the benchmark asks whether a specific pair interacts, training on a
    shared pair lets the model memorise pair identity instead of reading.
"""
import re, sys, hashlib, json
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED = 20260826
N_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
MAX_PER_PAIR = 3          # cap so no pair dominates

def norm(s): return " ".join(str(s).split()).strip()
def first(x):
    if isinstance(x, (list, np.ndarray)): return norm(x[0]) if len(x) else ""
    return norm(x)

d = pd.read_parquet(REPO/"data/training/distill/med25_pool.parquet")
print(f"pool: {len(d):,}")
for c in ("species1_form","species2_form","interaction_form"):
    d[c] = d[c].map(first)
d["passage"] = d["passage"].map(norm)
d = d[(d.species1_form!="")&(d.species2_form!="")&(d.interaction_form!="")]

# ---- benchmark exclusions ----
btxt, bpair = set(), set()
for f,tc,kw,pc in [("data/evaluation/biotic_interaction_test_set.csv","sentence",{},None),
                   ("data/evaluation/test500_paired.csv","sentence",{},("species1","species2")),
                   ("globi-relax_passages-triplets_2024-02-28_curation_EP.tsv","sentence",dict(sep="\t",encoding="latin-1"),("species1_term","species2_term")),
                   ("data/evaluation/biotx_retrieval_eval_100.csv","sentence",dict(sep=";",encoding="utf-8-sig"),("species1_term","species2_term"))]:
    p = REPO/f
    if not p.exists(): continue
    x = pd.read_csv(p, **kw)
    if tc in x.columns: btxt |= set(x[tc].astype(str).map(norm))
    if pc and pc[0] in x.columns:
        bpair |= {tuple(sorted([norm(a).lower(), norm(b).lower()]))
                  for a,b in zip(x[pc[0]], x[pc[1]]) if pd.notna(a) and pd.notna(b)}
for f in (REPO/"data/evaluation").glob("*curation_EP*.tsv"):
    try:
        x = pd.read_csv(f, sep="\t", encoding="latin-1")
        btxt |= set(x["sentence"].astype(str).map(norm))
        if "species1_term" in x.columns:
            bpair |= {tuple(sorted([norm(a).lower(), norm(b).lower()]))
                      for a,b in zip(x.species1_term, x.species2_term) if pd.notna(a) and pd.notna(b)}
    except Exception: pass
print(f"exclusions: {len(btxt)} passages, {len(bpair)} taxon pairs")

d["pk"] = [tuple(sorted([a.lower(), b.lower()])) for a,b in zip(d.species1_form, d.species2_form)]
before = len(d)
d = d[~d.passage.isin(btxt)]
n_txt = before - len(d)
before = len(d)
d = d[~d.pk.isin(bpair)]
print(f"dropped {n_txt} on text, {before-len(d)} on taxon pair -> {len(d):,} eligible")

# ---- pair-diverse sampling: round-robin over pairs, capped ----
rng = np.random.RandomState(SEED)
d = d.sample(frac=1, random_state=SEED)
d["rank_in_pair"] = d.groupby("pk").cumcount()
d = d[d.rank_in_pair < MAX_PER_PAIR]
pairs = d.pk.unique(); rng.shuffle(pairs)
order = {p:i for i,p in enumerate(pairs)}
d["po"] = d.pk.map(order)
d = d.sort_values(["rank_in_pair","po"]).head(N_TARGET)

out = REPO/"data/training/distill/distill_pool.csv"
d[["passage","species1_form","species2_form","interaction_form","triplet_key","doc_id","field"]].to_csv(out, index=False)
sha = hashlib.sha256(open(out,'rb').read()).hexdigest()
man = {"output": str(out.relative_to(REPO)), "sha256": sha, "seed": SEED,
       "rows": int(len(d)), "distinct_pairs": int(d.pk.nunique()),
       "max_rows_per_pair": MAX_PER_PAIR,
       "excluded_benchmark_passages": int(n_txt), "excluded_benchmark_pairs": len(bpair),
       "source": "sibils_eval.med25_r1_v_ep_passages (cached pull, 175,588 rows)"}
out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=2))
print(f"\nwrote {out}")
print(f"  {len(d):,} rows over {d.pk.nunique():,} distinct taxon pairs")
print(f"  (D3 had 606, D1 had 11,602)")
