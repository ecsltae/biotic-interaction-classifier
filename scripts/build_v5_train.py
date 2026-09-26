#!/usr/bin/env python3
"""Assemble a training set from the v3 base plus the two interventions.

  --relabel   apply species-level labels to the base rows the re-labelling has
              reached. The base labels answer a FULL-TRIPLE question ("is the
              stated relation the right type, in the right direction"); the test
              set asks a species-level one ("do these two taxa interact"). About a
              third of the base NO rows carry the opposite label under the question
              we actually evaluate, and those rows are the entire class of positives
              V3 scores ~0.00 on.
  --targeted  add the V1-error-targeted families from generate_targeted_v5.py.
              co_participant rows are used only where the teacher has adjudicated
              them; synonym/ancestor/authority are negative by construction.

Everything is capped and seeded so volume can be ablated independently of content.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--relabel", default=None, help="species_relabel.csv")
    ap.add_argument("--targeted", default=None, help="v5_targeted_candidates.csv")
    ap.add_argument("--targeted-labels", default=None, help="teacher labels for co_participant")
    ap.add_argument("--cap-co", type=int, default=4000)
    ap.add_argument("--cap-syn", type=int, default=2500)
    ap.add_argument("--cap-anc", type=int, default=2500)
    ap.add_argument("--cap-auth", type=int, default=1000)
    ap.add_argument("--cap-coord", type=int, default=0)
    ap.add_argument("--rel-invariance", type=int, default=0,
                    help="augment N label=1 rows with a RESAMPLED relation term, label kept")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)

    base = pd.read_csv(REPO/a.base)
    base = base.reset_index().rename(columns={"index": "row"})
    print(f"base {len(base):,}  pos {base.label.mean():.4f}")

    if a.relabel:
        rl = pd.read_csv(REPO/a.relabel).drop_duplicates(subset=["row"], keep="last")
        rl = rl[rl.raw.astype(str).str.upper().str.startswith(("YES", "NO"))]
        m = base.row.isin(set(rl.row))
        base = base.merge(rl[["row", "label_species"]], on="row", how="left")
        flipped = ((base.label == 0) & (base.label_species == 1)).sum()
        base["label"] = np.where(base.label_species.notna(),
                                 base.label_species.fillna(base.label), base.label).astype(int)
        base = base.drop(columns=["label_species"])
        print(f"relabel: {m.sum():,} base rows re-judged, {flipped:,} flipped NO->YES "
              f"({flipped/max(m.sum(),1):.1%}); pos now {base.label.mean():.4f}")

    parts = [base[["text", "source_species", "target_species", "interaction_type", "label", "kind"]]]

    if a.targeted:
        T = pd.read_csv(REPO/a.targeted)
        if a.targeted_labels and Path(REPO/a.targeted_labels).exists():
            tl = pd.read_csv(REPO/a.targeted_labels).drop_duplicates(subset=["row"], keep="last")
            tl = tl[tl.raw.astype(str).str.upper().str.startswith(("YES", "NO"))]
            T = T.reset_index().rename(columns={"index": "row"}).merge(
                tl[["row", "label_species"]], on="row", how="left")
            got = T.label_species.notna()
            T.loc[got, "label"] = T.loc[got, "label_species"].astype(int)
            T = T.drop(columns=["label_species", "row"])
        T = T[T.label >= 0]                       # drop un-adjudicated co_participant rows
        caps = {"co_participant": a.cap_co, "synonym": a.cap_syn,
                "ancestor": a.cap_anc, "authority": a.cap_auth,
                "coordination": a.cap_coord}
        keep = []
        for k, c in caps.items():
            g = T[T.kind == k]
            if len(g) > c: g = g.sample(c, random_state=a.seed)
            if len(g): keep.append(g)
            print(f"  targeted {k:14s} {len(g):,}  pos {g.label.mean():.3f}" if len(g)
                  else f"  targeted {k:14s} 0")
        if keep:
            T = pd.concat(keep, ignore_index=True)
            parts.append(T[["text", "source_species", "target_species", "interaction_type", "label", "kind"]])

    if a.rel_invariance:
        # Species-level truth does not depend on which relation word was retrieved.
        # A full-triple YES implies a species-level YES (species-level is the weaker
        # condition), so for label=1 rows the relation string can be resampled freely
        # and the label still holds. This is the same monotonicity the re-labelling
        # relies on, used in the opposite direction -- and it needs no teacher.
        # It targets the dominant new-FN class: pairs that do interact but were
        # retrieved with the wrong relation term or direction, which V3 scores ~0.00.
        # never augment a passage that appears in the evaluation set: the base already
        # carries 15 such passages (the known leak, flagged in_train on the test side)
        # and resampling them would multiply that leak rather than leave it alone.
        def _n(x): return " ".join(str(x).split()).strip()
        tset = set(pd.read_csv(REPO/"data/evaluation/unified_test_set.csv").sentence.map(_n))
        pos = base[(base.label == 1) & (~base.text.map(_n).isin(tset))]
        rels = base.interaction_type.dropna().astype(str)
        rels = rels[rels.str.len() > 1].value_counts()
        rvocab = rels.index.to_numpy(); rprob = (rels.to_numpy()/rels.sum())
        n = min(a.rel_invariance, len(pos))
        g = pos.sample(n, random_state=a.seed).copy()
        draw = rng.choice(rvocab, size=len(g), p=rprob)
        same = draw == g.interaction_type.astype(str).to_numpy()
        while same.any():
            draw[same] = rng.choice(rvocab, size=int(same.sum()), p=rprob)
            same = draw == g.interaction_type.astype(str).to_numpy()
        g["interaction_type"] = draw
        g["kind"] = "rel_invariance"
        parts.append(g[["text", "source_species", "target_species", "interaction_type", "label", "kind"]])
        print(f"  rel_invariance  {len(g):,} rows (label=1, relation resampled)")

    D = pd.concat(parts, ignore_index=True)
    D = D.drop_duplicates(subset=["text", "source_species", "target_species", "interaction_type"])
    D = D.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    out = REPO/a.out
    D.to_csv(out, index=False)
    print(f"\nwrote {out}  {len(D):,} rows  pos {D.label.mean():.4f}")
    print(D.groupby("kind").agg(n=("label", "size"), pos=("label", "mean")).round(3).to_string())

if __name__ == "__main__": main()
