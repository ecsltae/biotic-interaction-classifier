"""Load the human direction gold and compute baselines.

Gold schema (unified): SUBJECT_IS in {1, 2, None}
  1 = species1 is the subject of the relation, 2 = species2, None = not decidable.
direction_curation_220.xlsx uses DIRECTION F/R/N/U; v2 and SHORT use SUBJECT_IS 1/2/?.
"""
import pandas as pd, numpy as np, math, sys, os
EVAL = "/home/egaillac/MetaP/classifier/data/evaluation"

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def load_gold():
    rows = []
    d = pd.read_excel(f"{EVAL}/direction_curation_220.xlsx", sheet_name="annotate")
    for _, r in d[d.DIRECTION.notna()].iterrows():
        v = str(r.DIRECTION).strip().upper()
        s = 1 if v.startswith("F") else 2 if v.startswith("R") else None
        rows.append(dict(block="220", row_id=r.row_id, sentence=r.sentence,
                         species1=r.species1, species2=r.species2, relation=r.relation,
                         relation_canonical=r.relation_canonical, subject_is=s, raw=v))
    for fn, blk in (("direction_curation_v2.xlsx", "v2"), ("direction_curation_SHORT.xlsx", "short")):
        p = f"{EVAL}/{fn}"
        if not os.path.exists(p): continue
        d = pd.read_excel(p, sheet_name="annotate")
        if "SUBJECT_IS" not in d.columns: continue
        for _, r in d[d.SUBJECT_IS.notna()].iterrows():
            v = str(r.SUBJECT_IS).strip()
            s = 1 if v.startswith("1") else 2 if v.startswith("2") else None
            rows.append(dict(block=blk, row_id=r.row_id, sentence=r.sentence,
                             species1=r.species1, species2=r.species2, relation=r.relation,
                             relation_canonical=None, subject_is=s, raw=v))
    g = pd.DataFrame(rows)
    # the three workbooks are nested (SHORT and v2 are subsets of the 220 row_ids)
    g = g.drop_duplicates(subset=["row_id"], keep="first").reset_index(drop=True)
    return g

if __name__ == "__main__":
    g = load_gold()
    dec = g[g.subject_is.notna()]
    print(f"gold items {len(g)}  decidable {len(dec)}  undecidable {len(g)-len(dec)} "
          f"({(len(g)-len(dec))/len(g):.1%})")
    print("subject_is:", dec.subject_is.value_counts().to_dict())
    k = int((dec.subject_is == 1).sum()); n = len(dec)
    lo, hi = wilson(k, n)
    print(f"stored-order baseline (always species1): {k}/{n} = {k/n:.3f}  95% Wilson [{lo:.3f}, {hi:.3f}]")
