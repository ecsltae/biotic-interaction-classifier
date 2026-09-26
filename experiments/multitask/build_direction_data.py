"""Build direction supervision for the whole med25 pool from a dependency parse.

For every candidate row we ask the parser which taxon fills the SUBJECT-side
argument slot of the relation phrase. That is the annotation target itself
("which organism is the SUBJECT of the relation"), so the label needs no
composition step and no taxon-role ontology.

Output columns
  triplet_key doc_id passage s1 s2 rel ro pol syn_idx ev quality
    syn_idx  0 -> s1 is the subject, 1 -> s2 is the subject, -1 -> parser abstained
    quality  BOTH (both arguments resolved, high precision) | ONE | NONE
    pol      relation polarity from direction_lexicon (0 = symmetric -> not_applicable)
"""
import sys, os, argparse, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import direction_syntax as S, direction_lexicon as L

POOL = "/home/egaillac/MetaP/classifier/data/training/distill/med25_pool.parquet"

def first(a):
    try:
        return str(a[0])
    except Exception:
        return str(a)

def run(shard, nshard, out):
    p = pd.read_parquet(POOL)
    p = p.iloc[shard::nshard].reset_index(drop=True)
    s1 = p.species1_form.map(first); s2 = p.species2_form.map(first)
    rel = p.interaction_form.map(first)
    ro = p.triplet_key.str.rsplit(";", n=1).str[-1]
    nlp = S.nlp()
    texts = p.passage.astype(str).tolist()
    idxs, evs = [], []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        try:
            idx, ev = S.syn_subject_side(doc.text, s1[i], s2[i], rel[i], doc=doc)
        except Exception:
            idx, ev = None, "err"
        idxs.append(-1 if idx is None else idx); evs.append(ev)
        if i and i % 5000 == 0:
            print(f"shard {shard}: {i}/{len(p)}", flush=True)
    out_df = pd.DataFrame(dict(
        triplet_key=p.triplet_key, doc_id=p.doc_id, passage=p.passage,
        s1=s1, s2=s2, rel=rel, ro=ro, pol=ro.map(lambda r: L.polarity(r)),
        syn_idx=idxs, ev=evs))
    out_df["quality"] = np.where(out_df.ev.str.startswith("BOTH"), "BOTH",
                         np.where(out_df.syn_idx >= 0, "ONE", "NONE"))
    out_df.to_parquet(out, index=False)
    print("wrote", out, len(out_df), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.shard, a.nshard, a.out)
