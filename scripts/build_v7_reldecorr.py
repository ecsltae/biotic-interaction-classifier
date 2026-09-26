#!/usr/bin/env python3
"""Decorrelate the relation term from the label, in place, on the species-labelled set.

WHY. The species-level label is a function of (passage, taxon1, taxon2) ONLY: the
teacher prompt in scripts/relabel_species_level.py never shows the retrieved relation
term. Yet in data/training/distill/v4_species_train.csv the relation term explains
eta^2 = 0.106 of the label variance -- the per-term positive rate runs from 0.293
("positively regulates") to 0.981 ("ectoparasite of"), sd 0.159 over the 145 terms
that carry 98.4% of the rows. That prior is a pure artefact of which triples GloBI
retrieval happened to emit, and the benchmark does not have it: within a source block
the term barely separates gold positives from gold negatives (reject50 0.600 vs 0.587,
biotx100 0.779 vs 0.732). The model learns it anyway -- among gold-POSITIVE benchmark
rows, corr(training term prior, V3 score) = +0.616, and V3 recall on gold positives
runs 0.700 / 0.933 / 0.978 as the term prior rises through <0.60 / 0.60-0.75 / >=0.75.

WHAT IT DOES. For a fraction q of rows, replace the relation string with one drawn
from the corpus-wide relation marginal, independent of the label. Rows are edited IN
PLACE -- no duplication -- so the row count, the passage set, the pair-grouped dev
split and the in_train leak status are all identical to the input file, and any arm
trained on the output is directly paired with the v4 arm.

WHY IT IS LABEL-SAFE, unlike build_v5_train.py --rel-invariance. That flag augments
label==1 rows only (monotonicity: a full-triple YES implies a species-level YES), which
cannot flatten the marginal -- it only pushes every term's rate up, unevenly. It was
measured null at 5.6% dilution (results/scale_lever/measurements.json). Both classes
can be resampled now: data/training/distill/species_relabel.csv covers 31,764 of
31,764 NO rows (100%), so every label in v4_species_train is species-certain, which
removes the blocker that note records ("pair_contrast negatives turned out not to be
species-safe").
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]


def eta2(rel: pd.Series, y: pd.Series) -> float:
    """Fraction of label variance explained by the relation term."""
    gm = y.mean()
    g = pd.DataFrame({"r": rel, "y": y}).groupby("r").y.agg(["size", "mean"])
    return float((g["size"] * (g["mean"] - gm) ** 2).sum() / ((y - gm) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/distill/v4_species_train.csv")
    ap.add_argument("--q", type=float, default=0.75,
                    help="fraction of rows whose relation term is resampled label-independently")
    ap.add_argument("--min-n", type=int, default=20,
                    help="terms rarer than this are not used as resampling targets")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)

    d = pd.read_csv(REPO / a.data)
    rel0 = d.interaction_type.astype(str).str.lower().str.strip()
    print(f"{len(d):,} rows  pos {d.label.mean():.4f}  distinct relation terms {rel0.nunique():,}")
    print(f"BEFORE  eta^2(relation -> label) = {eta2(rel0, d.label):.4f}")
    g0 = d.assign(r=rel0).groupby("r").label.agg(["size", "mean"])
    g0 = g0[g0["size"] >= a.min_n]
    print(f"        per-term positive rate over {len(g0)} terms (n>={a.min_n}): "
          f"min {g0['mean'].min():.3f}  median {g0['mean'].median():.3f}  "
          f"max {g0['mean'].max():.3f}  sd {g0['mean'].std():.3f}")

    # resampling distribution: the corpus relation marginal, restricted to terms the
    # model has actually seen enough of. Drawing from the marginal (not uniform) keeps
    # the surface distribution of segment A unchanged; only its association with the
    # label is destroyed.
    vocab = g0.index.to_numpy()
    prob = (g0["size"].to_numpy() / g0["size"].sum())

    take = rng.rand(len(d)) < a.q
    draw = rng.choice(vocab, size=int(take.sum()), p=prob)
    out = d.copy()
    orig = out.loc[take, "interaction_type"].astype(str).to_numpy()
    out.loc[take, "interaction_type"] = draw
    out["kind"] = np.where(take, out["kind"].astype(str) + "+reldecorr", out["kind"].astype(str))

    rel1 = out.interaction_type.astype(str).str.lower().str.strip()
    print(f"\nresampled {int(take.sum()):,} of {len(d):,} rows ({take.mean():.1%}); "
          f"{int((draw == np.char.lower(orig.astype(str))).sum()):,} redraws happened to match the original")
    print(f"AFTER   eta^2(relation -> label) = {eta2(rel1, out.label):.4f}")
    g1 = out.assign(r=rel1).groupby("r").label.agg(["size", "mean"])
    g1 = g1[g1["size"] >= a.min_n]
    print(f"        per-term positive rate over {len(g1)} terms: "
          f"min {g1['mean'].min():.3f}  median {g1['mean'].median():.3f}  "
          f"max {g1['mean'].max():.3f}  sd {g1['mean'].std():.3f}")

    assert len(out) == len(d), "row count must not change"
    assert out.label.equals(d.label), "labels must not change"
    assert out.text.equals(d.text), "passages must not change -- leak status is inherited"
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(f"\nwrote {p}  {len(out):,} rows  pos {out.label.mean():.4f}")


if __name__ == "__main__":
    main()
