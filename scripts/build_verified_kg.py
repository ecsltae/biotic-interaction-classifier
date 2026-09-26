#!/usr/bin/env python3
"""Build a verified biotic-interaction knowledge graph.

Every edge carries: the taxon pair, the ROBI-style predicate, the supporting
passage, its source document, and the verifier's calibrated confidence. This is
the artefact the thesis has claimed and never produced -- Tier-2 emitted zero
triples because its NER head had no HOST/PATHOGEN supervision.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
from eval.core import threshold_from_prior  # noqa

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/student/xenc_s1"
OUT = REPO/"experiments/knowledge_graph/results/verified_kg"
OUT.mkdir(parents=True, exist_ok=True)

def first(x):
    if isinstance(x, (list, np.ndarray)): return str(x[0]).strip() if len(x) else ""
    return str(x).strip() if pd.notna(x) else ""

d = pd.read_parquet(REPO/"data/training/distill/med25_pool.parquet")
for c in ("species1_form","species2_form","interaction_form"):
    d[c] = d[c].map(first)
d = d[(d.species1_form!="")&(d.species2_form!="")&(d.interaction_form!="")].reset_index(drop=True)
print(f"candidates: {len(d):,} passages, {d.triplet_key.nunique():,} triplets, {d.doc_id.nunique():,} docs", flush=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
tok = AutoTokenizer.from_pretrained(REPO/MODEL, local_files_only=True)
m = AutoModelForSequenceClassification.from_pretrained(REPO/MODEL, local_files_only=True).to(dev).eval()
cfg = json.loads((REPO/MODEL/"student_config.json").read_text())

q = [f"{a} [SEP] {r} [SEP] {b}" for a,r,b in zip(d.species1_form, d.interaction_form, d.species2_form)]
p = d.passage.astype(str).tolist()
probs, t0, BS = [], time.time(), 256
with torch.no_grad():
    for i in range(0, len(q), BS):
        e = tok(q[i:i+BS], p[i:i+BS], truncation="only_second", max_length=256,
                padding=True, return_tensors="pt").to(dev)
        probs.extend(torch.softmax(m(**e).logits.float(), -1)[:,1].cpu().numpy())
        if (i//BS) % 100 == 0 and i:
            el = time.time()-t0
            print(f"  {i:,}/{len(q):,}  {i/el:.0f} pass/s  eta {(len(q)-i)/(i/el)/60:.1f}m", flush=True)
d["confidence"] = np.array(probs)
print(f"scored {len(d):,} in {time.time()-t0:.0f}s ({len(d)/(time.time()-t0):.0f} pass/s)", flush=True)

thr = float(cfg["threshold_dev"])
d["verified"] = d.confidence >= thr
print(f"\nthreshold {thr:.3f} (dev-derived): {int(d.verified.sum()):,} passages verified ({d.verified.mean():.1%})")

# ---- edges: aggregate passages per triplet ----
d["pair"] = [tuple(sorted([a.lower(), b.lower()])) for a,b in zip(d.species1_form, d.species2_form)]
g = d.groupby("triplet_key")
edges = g.agg(species1=("species1_form","first"), species2=("species2_form","first"),
              predicate=("interaction_form","first"), n_passages=("passage","size"),
              max_conf=("confidence","max"), mean_conf=("confidence","mean"),
              n_verified=("verified","sum"), docs=("doc_id", lambda x: sorted(set(x))[:20])).reset_index()
edges["n_docs"] = edges.docs.map(len)
edges["accepted"] = edges.n_verified > 0
best = d.sort_values("confidence", ascending=False).groupby("triplet_key").first()
edges["best_passage"] = edges.triplet_key.map(best.passage)
edges["best_doc"] = edges.triplet_key.map(best.doc_id)

kg = edges[edges.accepted].copy()
print(f"\n=== VERIFIED KG ===")
print(f"  edges: {len(kg):,} of {len(edges):,} candidate triplets ({len(kg)/len(edges):.1%})")
nodes = set(kg.species1.str.lower()) | set(kg.species2.str.lower())
print(f"  nodes (distinct taxon surface forms): {len(nodes):,}")
print(f"  distinct predicates: {kg.predicate.str.lower().nunique():,}")
print(f"  source documents: {len(set().union(*kg.docs)):,}")
print(f"  edges with >1 supporting passage: {int((kg.n_passages>1).sum()):,} ({(kg.n_passages>1).mean():.1%})")
print(f"  edges with >1 supporting document: {int((kg.n_docs>1).sum()):,} ({(kg.n_docs>1).mean():.1%})")
print(f"\n  top predicates:")
for pr,c in kg.predicate.str.lower().value_counts().head(10).items():
    print(f"    {c:6}  {pr}")

kg_out = kg.drop(columns=["docs"]).assign(docs=kg.docs.map(lambda x: "|".join(map(str,x))))
kg_out.to_csv(OUT/"verified_kg_edges.csv", index=False)
edges.drop(columns=["docs"]).assign(docs=edges.docs.map(lambda x:"|".join(map(str,x)))).to_csv(OUT/"all_candidates_scored.csv", index=False)
(OUT/"build_manifest.json").write_text(json.dumps({
    "model": MODEL, "threshold": thr, "threshold_source": "student dev split",
    "train_pos_rate": cfg.get("train_pos_rate"),
    "candidates_passages": int(len(d)), "candidate_triplets": int(len(edges)),
    "verified_edges": int(len(kg)), "nodes": int(len(nodes)),
    "predicates": int(kg.predicate.str.lower().nunique()),
    "source_documents": int(len(set().union(*kg.docs))),
    "multi_passage_edges": int((kg.n_passages>1).sum()),
    "multi_doc_edges": int((kg.n_docs>1).sum()),
}, indent=2))
print(f"\nwrote {OUT}/verified_kg_edges.csv")
