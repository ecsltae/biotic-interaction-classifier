#!/usr/bin/env python3
"""
Build v16 training dataset.

Changes from v14:
1. Append commensal/symbiosis harvest (commensal_harvest_20260703.csv)
2. Entity-prepended variants for positives with known species pairs
   Format: "Entity1: {sp1}. Entity2: {sp2}. Sentence: {text}"
   (kept alongside originals so model handles both prompt styles)

Input:
    classifier/data/training/training_data_v14.csv          (~34K rows)
    classifier/data/training/commensal_harvest_20260703.csv (~486 rows)

Output:
    classifier/data/training/training_data_v16.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "classifier"))

V14 = ROOT / "classifier/data/training/training_data_v14.csv"
HARVEST = ROOT / "classifier/data/training/commensal_harvest_20260703.csv"
OUT = ROOT / "classifier/data/training/training_data_v16.csv"


def entity_prepend(row: pd.Series) -> str:
    sp1 = str(row.get("source_species", "") or "").strip()
    sp2 = str(row.get("target_species", "") or "").strip()
    text = str(row.get("text", "")).strip()
    if sp1 and sp2 and sp1 != "nan" and sp2 != "nan":
        return f"Entity1: {sp1}. Entity2: {sp2}. Sentence: {text}"
    return text


def build(add_entity_prepend: bool = True, dry_run: bool = False) -> pd.DataFrame:
    print("Loading v14 ...")
    v14 = pd.read_csv(V14)
    print(f"  v14: {len(v14)} rows  (pos={v14['label'].sum()}, neg={(v14['label']==0).sum()})")

    print("Loading commensal harvest ...")
    harvest = pd.read_csv(HARVEST)
    # Align columns to v14 schema
    for col in v14.columns:
        if col not in harvest.columns:
            harvest[col] = ""
    harvest = harvest[v14.columns]
    pos_harvest = harvest[harvest["label"] == 1]
    neg_harvest = harvest[harvest["label"] == 0]
    print(f"  harvest: {len(harvest)} rows (pos={len(pos_harvest)}, neg={len(neg_harvest)})")

    combined = pd.concat([v14, harvest], ignore_index=True)
    print(f"  after merge: {len(combined)} rows")

    # Entity-prepended variants (positives with both species known)
    if add_entity_prepend:
        positives = combined[combined["label"] == 1].copy()
        has_both = (
            positives["source_species"].notna() & (positives["source_species"] != "") &
            positives["target_species"].notna() & (positives["target_species"] != "")
        )
        prepend_candidates = positives[has_both].copy()
        prepend_candidates["text"] = prepend_candidates.apply(entity_prepend, axis=1)
        prepend_candidates["source"] = prepend_candidates["source"].astype(str) + "_entity_prepend"
        print(f"  entity-prepend variants: {len(prepend_candidates)} new rows")
        combined = pd.concat([combined, prepend_candidates], ignore_index=True)

    # Deduplicate on text (keep first occurrence)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text"], keep="first")
    print(f"  after dedup: {len(combined)} rows (removed {before_dedup - len(combined)} duplicates)")

    pos = combined["label"].sum()
    neg = (combined["label"] == 0).sum()
    print(f"\nv16 summary:")
    print(f"  Total:     {len(combined):,}")
    print(f"  Positive:  {int(pos):,}")
    print(f"  Negative:  {int(neg):,}")
    print(f"  Balance:   {pos/len(combined)*100:.1f}% positive")

    if "interaction_type" in combined.columns:
        print(f"\nInteraction type distribution (positives):")
        itype = combined[combined["label"]==1]["interaction_type"].value_counts()
        comm = itype.get("commensalistOf", 0)
        symb = itype.get("symbioticWith", 0)
        print(f"  commensalistOf: {comm}")
        print(f"  symbioticWith:  {symb}")
        print(f"  (top 5): {itype.head(5).to_dict()}")

    return combined


def main() -> None:
    p = argparse.ArgumentParser(description="Build v16 training dataset")
    p.add_argument("--no-entity-prepend", action="store_true",
                   help="Skip entity-prepended variants")
    p.add_argument("--dry-run", action="store_true",
                   help="Build but don't write output")
    p.add_argument("--output", default=str(OUT),
                   help=f"Output path (default: {OUT})")
    args = p.parse_args()

    df = build(add_entity_prepend=not args.no_entity_prepend, dry_run=args.dry_run)

    if not args.dry_run:
        out = Path(args.output)
        df.to_csv(out, index=False)
        print(f"\n✓ Saved {len(df):,} rows → {out.relative_to(ROOT)}")
    else:
        print("\n[DRY RUN] Not writing output.")


if __name__ == "__main__":
    main()
