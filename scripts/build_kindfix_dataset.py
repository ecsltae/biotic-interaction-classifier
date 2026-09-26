#!/usr/bin/env python3
"""Use the measured per-kind flip rates to relabel rather than discard.

For a full-triple-NO row of kind k, the teacher probe gives p_k = P(species-level YES).
Keeping the row as NEGATIVE is right with probability 1-p_k; relabelling it POSITIVE is
right with probability p_k. So the best use of each kind is decided by its measured rate,
not by a blanket policy:

    reversal      p=0.835  -> relabel positive   (83.5% correct, vs 16.5% as-is)
    active        p=0.635  -> relabel positive   (63.5% correct, vs 36.5% as-is)
    pair_contrast p=0.485  -> drop               (a coin toss either way: no information)
    v2_base       p=0.320  -> keep negative      (68.0% correct)

Rows that carry an actual species-level judgement are used directly and never guessed.
"""
import argparse
import pandas as pd
from pathlib import Path

REPO = Path("/home/egaillac/MetaP/classifier")
SC = Path("/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad")
COLS = ["text", "source_species", "target_species", "interaction_type", "label"]

# measured, n=200 per kind (v2_base n=2000)
FLIP = {"reversal": 0.835, "active": 0.635, "pair_contrast": 0.485, "v2_base": 0.320}
DROP_BAND = (0.40, 0.60)          # too close to a coin toss to carry information

ap = argparse.ArgumentParser()
ap.add_argument("--add-slice", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

df = pd.read_csv(REPO/"data/training/distill/v3_combined_train.csv").reset_index().rename(columns={"index": "row"})
df["known"] = False

known = []
rl = REPO/"data/training/distill/species_relabel.csv"
if rl.exists():
    known.append(pd.read_csv(rl).drop_duplicates("row")[["row", "label_species"]])
kp = SC/"kindprobe_labels.csv"
if kp.exists():
    k = pd.read_csv(kp).drop_duplicates("uid")
    k["row"] = k.uid.str.lstrip("k").astype(int)
    known.append(k[["row", "label_species"]])
if known:
    kn = pd.concat(known).drop_duplicates("row")
    df = df.merge(kn, on="row", how="left")
    hit = df.label_species.notna()
    df.loc[hit, "label"] = df.loc[hit, "label_species"].astype(int)
    df.loc[hit, "known"] = True
    print(f"applied {int(hit.sum())} measured species-level judgements")
    df = df.drop(columns=["label_species"])

keep = []
for kind, p in FLIP.items():
    neg = df[(df.kind == kind) & (df.label == 0) & (~df.known)]
    if DROP_BAND[0] <= p <= DROP_BAND[1]:
        print(f"  {kind:14s} p={p:.3f}  DROP {len(neg)} rows (no information either way)")
        continue
    if p > DROP_BAND[1]:
        n = neg.copy(); n["label"] = 1
        print(f"  {kind:14s} p={p:.3f}  RELABEL POSITIVE {len(n)} rows ({p:.1%} correct, was {1-p:.1%})")
        keep.append(n)
    else:
        print(f"  {kind:14s} p={p:.3f}  KEEP NEGATIVE {len(neg)} rows ({1-p:.1%} correct)")
        keep.append(neg)

rest = df[(df.label == 1) | df.known]
out = pd.concat([rest] + keep, ignore_index=True)[COLS]

if a.add_slice:
    lab = pd.read_csv(SC/"work_labels.csv").drop_duplicates("uid")
    pool = pd.read_parquet(SC/"pool_scored.parquet").reset_index(drop=True)
    pool["uid"] = "p" + pool.index.astype(str)
    s = lab[lab.uid.str.startswith("p")].merge(pool[["uid", "pas", "sp1", "sp2", "rel"]], on="uid")
    if len(s):
        sl = s.rename(columns={"pas": "text", "sp1": "source_species", "sp2": "target_species",
                               "rel": "interaction_type", "label_species": "label"})[COLS]
        print(f"adding {len(sl)} teacher-labelled pool rows, pos {sl.label.mean():.3f}")
        out = pd.concat([out, sl], ignore_index=True)

out = out.dropna(subset=["text", "source_species", "target_species"])
out["interaction_type"] = out.interaction_type.fillna("interacts with")
out.to_csv(REPO/a.out, index=False)
print(f"wrote {a.out}  n={len(out)}  pos={out.label.mean():.4f}")
