#!/usr/bin/env python3
"""Resolve label contradictions that only exist under the `pair` input format.

Under the verified-winning `pair` query ("s1 [SEP] s2" + passage) the ROBI relation
term is no longer an input, so every row that differed ONLY by its relation term
collapses onto an identical model input. In v3_combined_train.csv that makes 8,218
rows (17.0%) members of 2,040 groups whose inputs are byte-identical but whose
labels disagree -- the model is asked to emit 0 and 1 for the same tensor.

That is not a labelling opinion, it is an inconsistency created by the format, and
it does not exist under `triple` (939 rows, 1.9%). Three resolutions, plus the
prior-matched control needed to tell a label fix from a prior shift:

  cmax      contradictory groups harmonised to label 1 (species-level reading),
            all rows kept -> prior rises, size unchanged
  cdrop     contradictory groups removed entirely -> prior falls, size shrinks
  ccollapse each unique pair-format input becomes ONE row carrying the max label;
            this is the species-level dataset in pair space, with no duplicate
            upweighting. The principled version.

Usage: python3 scripts/build_pairkey_datasets.py [--with-targeted]
"""
import argparse, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts"))
import xenc_format as xf


def pair_key(df):
    A, B = xf.build_many("pair", df.source_species, df.interaction_type,
                         df.target_species, df.text.astype(str))
    return pd.Series([a + "\x00" + b for a, b in zip(A, B)], index=df.index)


def report(tag, df):
    print(f"  {tag:<28} rows {len(df):>6}  prior {df.label.mean():.4f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--targeted", default=None,
                    help="optional dataset whose EXTRA rows (by kind) are appended, e.g. v5_rules_train.csv")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    d = pd.read_csv(REPO/a.base)
    if a.targeted:
        t = pd.read_csv(REPO/a.targeted)
        extra = t[~t.kind.isin(set(d.kind))]
        print(f"appending {len(extra)} targeted rows, kinds {sorted(set(extra.kind))}, "
              f"labels {extra.label.value_counts().to_dict()}")
        d = pd.concat([d, extra], ignore_index=True)
    report("input", d)

    k = pair_key(d)
    nun = d.groupby(k).label.transform("nunique")
    contra = nun > 1
    print(f"  contradictory pair-format inputs: {int(k[contra].nunique())} groups, "
          f"{int(contra.sum())} rows ({contra.mean():.1%})")

    out = REPO/"data/training/distill"
    sfx = a.suffix

    cmax = d.copy()
    cmax.loc[contra, "label"] = 1
    report("cmax", cmax).to_csv(out/f"v6_pair_cmax{sfx}.csv", index=False)

    cdrop = d[~contra].reset_index(drop=True)
    report("cdrop", cdrop).to_csv(out/f"v6_pair_cdrop{sfx}.csv", index=False)

    # one row per unique pair-format input, carrying the max label; keep the first
    # row's metadata so the trainer's pair-grouped split still works
    g = d.assign(_k=k).sort_values("label", ascending=False).groupby("_k", as_index=False).first()
    coll = g.drop(columns=["_k"]).reset_index(drop=True)
    report("ccollapse", coll).to_csv(out/f"v6_pair_ccollapse{sfx}.csv", index=False)

    # pos_weight that reproduces cmax's positive rate from the ORIGINAL labels:
    # eff = w*P / (w*P + N)  ->  w = eff/(1-eff) * N/P
    P, N = d.label.sum(), (1 - d.label).sum()
    for tag, ref in (("cmax", cmax), ("ccollapse", coll)):
        eff = ref.label.mean()
        w = eff / (1 - eff) * N / P
        print(f"  control pos_weight matching {tag} prior {eff:.4f}: {w:.4f}")


if __name__ == "__main__":
    main()
