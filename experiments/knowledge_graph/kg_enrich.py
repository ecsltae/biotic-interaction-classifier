#!/usr/bin/env python3
"""
KG enrichment: prepend GloBI context to sentences where a known species pair is detected.

For each sentence:
  1. Detect species names via Aho-Corasick (reuse automaton from data.py)
  2. For each detected pair, check bidirectional GloBI lookup
  3. If found: prepend "[KG: sp1 CATEGORY sp2]" to the text

Produces:
  - results/kg_enrich/v14_enriched.csv   (drop-in replacement for training)
  - results/kg_enrich/ep_enriched.tsv    (drop-in for evaluation)
  - results/kg_enrich/coverage_stats.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
import time

csv.field_size_limit(sys.maxsize)
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "classifier"))
sys.path.insert(0, str(ROOT / "classifier" / "experiments" / "knowledge_graph"))
sys.path.insert(0, str(ROOT / "classifier" / "experiments" / "multitask"))

from kg_utils import map_interaction_type_to_category

RELEVANT_CATEGORIES = {
    "INFECTION", "PARASITISM", "ENDOPARASITISM", "PARASITOIDISM",
    "VECTOR", "PREDATION", "POLLINATION", "HERBIVORY", "SYMBIOSIS",
}

# ── Build GloBI lookup ─────────────────────────────────────────────────────────

def build_globi_lookup(globi_path: Path) -> Dict[Tuple[str, str], str]:
    """Parse GloBI TSV → {(norm_sp1, norm_sp2): category} bidirectional. Caches to disk."""
    cache_path = globi_path.parent / "globi_lookup_cache.pkl"
    if cache_path.exists():
        print(f"Loading cached GloBI lookup from {cache_path} ...", flush=True)
        with open(cache_path, "rb") as f:
            lookup = pickle.load(f)
        print(f"  Loaded {len(lookup):,} pairs from cache", flush=True)
        return lookup

    print(f"Building GloBI lookup from {globi_path} ...", flush=True)
    t0 = time.time()
    lookup: Dict[Tuple[str, str], str] = {}
    skipped = 0
    n = 0

    with gzip.open(str(globi_path), "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = (row.get("sourceTaxonSpeciesName") or row.get("sourceTaxonName") or "").strip()
            tgt = (row.get("targetTaxonSpeciesName") or row.get("targetTaxonName") or "").strip()
            itype = row.get("interactionTypeName", "").strip()
            if not src or not tgt or src == tgt:
                skipped += 1
                continue
            category = map_interaction_type_to_category(itype)
            if category not in RELEVANT_CATEGORIES:
                skipped += 1
                continue
            src_n = src.lower()
            tgt_n = tgt.lower()
            # Store: prefer more specific category if pair seen multiple times
            for k in ((src_n, tgt_n), (tgt_n, src_n)):
                if k not in lookup:
                    lookup[k] = category
            n += 1
            if n % 500_000 == 0:
                print(f"  {n:,} rows processed, {len(lookup):,} pairs so far ...", flush=True)

    print(f"  GloBI lookup: {len(lookup):,} pairs from {n:,} interactions "
          f"({skipped:,} skipped) in {time.time()-t0:.1f}s", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached to {cache_path}", flush=True)
    return lookup


# ── Species detection ──────────────────────────────────────────────────────────

def build_automaton():
    """Reuse automaton from multitask/data.py."""
    from data import get_automata
    species_aut, _ = get_automata()
    return species_aut


def detect_species(text: str, automaton) -> list[str]:
    """Return list of normalised species names found in text."""
    found = set()
    text_lower = text.lower()
    for end_idx, orig_name in automaton.iter(text_lower):
        found.add(orig_name.lower())
    return list(found)


# ── Context injection ──────────────────────────────────────────────────────────

def enrich_text(text: str, automaton, lookup: Dict[Tuple[str, str], str]) -> Tuple[str, bool]:
    """
    Returns (enriched_text, was_enriched).
    Prepends '[KG: sp1 CATEGORY sp2]' for first matching pair found.
    """
    species = detect_species(text, automaton)
    if len(species) < 2:
        return text, False

    # Check all pairs (order-independent via bidirectional lookup)
    for i, sp1 in enumerate(species):
        for sp2 in species[i + 1:]:
            cat = lookup.get((sp1, sp2)) or lookup.get((sp2, sp1))
            if cat:
                prefix = f"[KG: {sp1} {cat} {sp2}] "
                return prefix + text, True

    return text, False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrich sentences with GloBI KG context")
    parser.add_argument("--globi-tsv",  required=True)
    parser.add_argument("--v14-path",   required=True)
    parser.add_argument("--ep-relax",   required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Build GloBI lookup
    lookup = build_globi_lookup(Path(args.globi_tsv))

    # 2. Build species automaton
    print("Building species automaton ...", flush=True)
    automaton = build_automaton()

    stats: dict = {
        "globi_pairs": len(lookup),
        "v14": {},
        "ep_relax": {},
    }

    # 3. Enrich v14 training data
    print(f"\nEnriching v14 training data: {args.v14_path}", flush=True)
    df = pd.read_csv(args.v14_path)
    enriched_texts = []
    enriched_flags = []
    for text in df["text"].astype(str):
        et, flag = enrich_text(text, automaton, lookup)
        enriched_texts.append(et)
        enriched_flags.append(flag)

    df["text"] = enriched_texts
    v14_out = out / "v14_enriched.csv"
    df.to_csv(v14_out, index=False)

    n_enriched = sum(enriched_flags)
    pos_enriched = sum(f and l == 1 for f, l in zip(enriched_flags, df["label"].tolist()))
    stats["v14"] = {
        "total": len(df),
        "enriched": n_enriched,
        "enriched_pct": round(n_enriched / len(df) * 100, 1),
        "pos_enriched": pos_enriched,
    }
    print(f"  v14: {n_enriched}/{len(df)} sentences enriched "
          f"({n_enriched/len(df)*100:.1f}%), {pos_enriched} positives", flush=True)

    # 4. Enrich EP-relax
    print(f"\nEnriching EP-relax: {args.ep_relax}", flush=True)
    ep_rows = []
    ep_enriched = 0
    ep_pos_enriched = 0
    with open(args.ep_relax, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        for row in reader:
            text = row.get("passage") or row.get("text") or ""
            label = int(row.get("label", 0))
            et, flag = enrich_text(text, automaton, lookup)
            row_copy = dict(row)
            if "passage" in row_copy:
                row_copy["passage"] = et
            elif "text" in row_copy:
                row_copy["text"] = et
            ep_rows.append(row_copy)
            if flag:
                ep_enriched += 1
                if label == 1:
                    ep_pos_enriched += 1

    ep_out = out / "ep_enriched.tsv"
    with open(ep_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(ep_rows)

    stats["ep_relax"] = {
        "total": len(ep_rows),
        "enriched": ep_enriched,
        "enriched_pct": round(ep_enriched / len(ep_rows) * 100, 1) if ep_rows else 0,
        "pos_enriched": ep_pos_enriched,
    }
    print(f"  EP-relax: {ep_enriched}/{len(ep_rows)} sentences enriched "
          f"({stats['ep_relax']['enriched_pct']}%), {ep_pos_enriched} positives", flush=True)

    # 5. Save stats
    with open(out / "coverage_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone. Enriched files written to {out}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
