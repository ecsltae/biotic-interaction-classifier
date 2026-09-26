#!/usr/bin/env python3
"""Fix the label contradictions that species-level semantics makes provable.

The evaluated label is "do these two taxa interact?" -- symmetric in the pair and
independent of the retrieved relation term. Under that semantics, two rows with the
SAME passage and the SAME unordered taxon pair must carry the SAME label. The v3
training file violates this for 3,891 rows: every `reversal` negative (direction
flipped) and 2,130 `pair_contrast` negatives (relation term swapped) have an
identical-text, identical-pair twin labelled 1.

Those negatives are therefore mislabelled for the task we evaluate, and no LLM call
is needed to know it. This builds the corrected file.
"""
import argparse
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--out", default="data/training/distill/v3_contradiction_fix.csv")
    ap.add_argument("--mode", choices=["flip", "drop"], default="flip",
                    help="flip the contradictory negatives to 1, or drop them entirely")
    a = ap.parse_args()

    df = pd.read_csv(REPO/a.data)
    pk = [tuple(sorted([str(x).lower(), str(y).lower()]))
          for x, y in zip(df.source_species, df.target_species)]
    tk = df.text.astype(str).str.strip().str[:200]
    key = list(zip(tk, pk))
    df["_key"] = key
    poskeys = set(df.loc[df.label == 1, "_key"])
    bad = (df.label == 0) & df._key.isin(poskeys)
    print(f"{len(df):,} rows; {int(bad.sum()):,} contradictory negatives "
          f"({bad.sum()/max((df.label==0).sum(),1):.1%} of all negatives)")
    print(df[bad].kind.value_counts().to_string())

    if a.mode == "flip":
        df.loc[bad, "label"] = 1
    else:
        df = df[~bad]
    df = df.drop(columns=["_key"])
    print(f"\nOUT {len(df):,} rows, positive rate {df.label.mean():.4f}")
    print(df.groupby("kind").label.agg(["size", "mean"]).round(4).to_string())
    df.to_csv(REPO/a.out, index=False)
    print("wrote", REPO/a.out)

if __name__ == "__main__":
    main()
