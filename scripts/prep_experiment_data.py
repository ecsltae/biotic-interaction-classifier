#!/usr/bin/env python3
"""
Prepare datasets for multi-task training experiments (Phase 0).

Steps:
  0a. distillation_44k.csv  — first 44,178 rows of distillation_soft_labels.csv (pre-harvest-augmentation)
  0b. v7_data.csv           — v7_llm_cleaned rows from v14 training data
  0c. (soft labels for v7 produced separately by score_all.py)
  0d. v7_distill_mix.csv    — v7_softlabels.csv + distillation_44k.csv (created after 0c)
  0e. nonv7_v14.csv         — non-v7 v14 rows (for Option D)

Usage:
    python classifier/scripts/prep_experiment_data.py [--step {all,0a,0b,0d,0e}]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "classifier/data/training"

DISTILLATION_CSV = DATA / "distillation_soft_labels.csv"
V14_CSV          = DATA / "training_data_v14.csv"
OUT_44K          = DATA / "distillation_44k.csv"
OUT_V7           = DATA / "v7_data.csv"
OUT_V7_SL        = DATA / "v7_softlabels.csv"  # produced by score_all.py
OUT_MIX          = DATA / "v7_distill_mix.csv"
OUT_NONV7        = DATA / "nonv7_v14.csv"

N_PRE_AUGMENT = 44178  # rows before harvest augmentation was added


def step_0a():
    print("Step 0a: Creating distillation_44k.csv (first 44,178 rows) ...")
    df = pd.read_csv(DISTILLATION_CSV)
    print(f"  Full distillation CSV: {len(df):,} rows, {(df.hard_label == 1).sum()} positive")
    out = df.head(N_PRE_AUGMENT)
    out.to_csv(OUT_44K, index=False)
    print(f"  Saved {len(out):,} rows → {OUT_44K.name}")
    print(f"  Positive count: {(out.hard_label == 1).sum()}")


def step_0b():
    print("Step 0b: Extracting v7_llm_cleaned rows from v14 ...")
    v14 = pd.read_csv(V14_CSV)
    v7 = v14[v14.source == "v7_llm_cleaned"].copy()
    v7.to_csv(OUT_V7, index=False)
    pos = (v7.label == 1).sum()
    neg = (v7.label == 0).sum()
    print(f"  Saved {len(v7):,} rows → {OUT_V7.name}  ({pos} pos / {neg} neg)")


def step_0d():
    if not OUT_V7_SL.exists():
        print(f"ERROR: {OUT_V7_SL} not found — run score_all.py first (step 0c)")
        sys.exit(1)
    print("Step 0d: Creating v7_distill_mix.csv ...")

    v7_sl = pd.read_csv(OUT_V7_SL)
    # Normalize label column name
    if "label" in v7_sl.columns and "hard_label" not in v7_sl.columns:
        v7_sl = v7_sl.rename(columns={"label": "hard_label"})
    v7_part = v7_sl[["text", "hard_label", "p_ensemble"]].copy()

    if not OUT_44K.exists():
        print(f"  distillation_44k.csv not found — running step 0a first ...")
        step_0a()
    distill = pd.read_csv(OUT_44K)[["text", "hard_label", "p_ensemble"]]

    mixed = pd.concat([v7_part, distill], ignore_index=True)
    mixed.to_csv(OUT_MIX, index=False)
    pos = (mixed.hard_label == 1).sum()
    neg = (mixed.hard_label == 0).sum()
    print(f"  Saved {len(mixed):,} rows → {OUT_MIX.name}  ({pos} pos / {neg} neg)")
    print(f"  v7 part: {len(v7_part):,} rows   distillation part: {len(distill):,} rows")


def step_0e():
    print("Step 0e: Extracting non-v7 rows from v14 (Option D) ...")
    v14 = pd.read_csv(V14_CSV)
    nonv7 = v14[v14.source != "v7_llm_cleaned"].copy()
    nonv7.to_csv(OUT_NONV7, index=False)
    pos = (nonv7.label == 1).sum()
    print(f"  Saved {len(nonv7):,} rows → {OUT_NONV7.name}  ({pos} pos)")


def main():
    parser = argparse.ArgumentParser(description="Prepare experiment datasets")
    parser.add_argument("--step", default="all",
                        choices=["all", "0a", "0b", "0d", "0e"],
                        help="Which step(s) to run (default: all except 0d which needs score_all.py first)")
    args = parser.parse_args()

    if args.step in ("all", "0a"):
        step_0a()
    if args.step in ("all", "0b"):
        step_0b()
    if args.step == "0d":
        step_0d()
    if args.step in ("all", "0e"):
        step_0e()

    if args.step == "all":
        print("\nSteps 0a, 0b, 0e done.")
        print("Next: run score_all.py on v7_data.csv to produce v7_softlabels.csv,")
        print("      then run this script with --step 0d to create v7_distill_mix.csv.")


if __name__ == "__main__":
    main()
