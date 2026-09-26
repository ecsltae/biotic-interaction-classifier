#!/usr/bin/env python3
"""Species-level augmentation that cannot introduce label noise.

Monotonicity: a full-triple YES is necessarily a species-level YES (species-level is
the weaker condition). And the evaluated label is symmetric in the pair. Therefore,
for every positive row, the entity-swapped copy is also a species-level positive --
known with certainty, no teacher call required.

This teaches the pair-symmetry invariance the evaluation demands, using only rows
whose species-level label is provable. Negatives are left untouched, so unlike the
contradiction fix this cannot inject inconsistent supervision.
"""
import argparse
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--out", default="data/training/distill/v3_symaug.csv")
    a = ap.parse_args()

    df = pd.read_csv(REPO/a.data)
    pos = df[df.label == 1].copy()
    sw = pos.copy()
    sw["source_species"], sw["target_species"] = pos.target_species.values, pos.source_species.values
    sw["kind"] = sw.kind.astype(str) + "_swap"

    # drop swaps that already exist verbatim (reversal rows already supply some)
    key = lambda x: list(zip(x.text.astype(str).str.strip().str[:200],
                             x.source_species.astype(str).str.lower(),
                             x.target_species.astype(str).str.lower()))
    have = set(key(df))
    sw = sw[[k not in have for k in key(sw)]]

    out = pd.concat([df, sw], ignore_index=True)
    print(f"base {len(df):,} rows (pos rate {df.label.mean():.4f})")
    print(f"added {len(sw):,} swapped positives")
    print(f"OUT {len(out):,} rows, pos rate {out.label.mean():.4f}")
    out.to_csv(REPO/a.out, index=False)
    print("wrote", REPO/a.out)

if __name__ == "__main__":
    main()
