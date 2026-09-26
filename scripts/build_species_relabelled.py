#!/usr/bin/env python3
"""Build the species-level corrected training file.

Takes v3_combined_train.csv and flips label 0 -> 1 for every row whose index appears
in species_relabel.csv with label_species=1. Monotonicity: full-triple YES implies
species-level YES, so only NO rows can move.

If the re-label is still partial, rows not yet judged keep their original label and
the script reports the coverage so the result can be read honestly.
"""
import argparse
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--relabel", default="data/training/distill/species_relabel.csv")
    ap.add_argument("--out", default="data/training/distill/v3_species_train.csv")
    ap.add_argument("--only-judged", action="store_true",
                    help="drop NO rows that have not been re-judged instead of keeping them as NO")
    a = ap.parse_args()

    df = pd.read_csv(REPO/a.data)
    rl = pd.read_csv(REPO/a.relabel).drop_duplicates("row", keep="last")
    rl = rl[rl.raw.astype(str).str.upper().isin(["YES", "NO"])]        # drop ERR rows
    flip = set(rl.loc[rl.label_species == 1, "row"].astype(int))
    judged = set(rl.row.astype(int))

    no_idx = set(df.index[df.label == 0])
    cov = len(judged & no_idx)/len(no_idx)
    print(f"{len(df):,} rows; {len(no_idx):,} NO; {len(judged & no_idx):,} re-judged "
          f"({cov:.1%} coverage); {len(flip & no_idx):,} flip to YES "
          f"({len(flip & no_idx)/max(len(judged & no_idx),1):.1%} of judged)")

    out = df.copy()
    out.loc[out.index.isin(flip), "label"] = 1
    out["relabelled"] = out.index.isin(judged & no_idx)
    if a.only_judged:
        keep = (out.label == 1) | out.relabelled
        print(f"--only-judged: dropping {(~keep).sum():,} un-judged NO rows")
        out = out[keep]

    print(f"OUT {len(out):,} rows, positive rate {out.label.mean():.4f} "
          f"(was {df.label.mean():.4f})")
    print(out.groupby('kind').label.agg(['size', 'mean']).round(4))
    out.drop(columns=["relabelled"]).to_csv(REPO/a.out, index=False)
    print("wrote", REPO/a.out)

if __name__ == "__main__":
    main()
