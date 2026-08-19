#!/usr/bin/env python3
"""
Test-set contamination check: exact-match, near-duplicate, and taxon-pair
leakage between the 500-sentence test set and all training corpora actually
used in the pipeline.

Usage:
    python classifier/scripts/check_test_contamination.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "classifier/data/evaluation"
TRAIN_DIR = ROOT / "classifier/data/training"
OUT = ROOT / "classifier/results/contamination_check.json"

NEAR_DUP_THRESHOLD = 0.85


def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def normalize_pair(a, b) -> tuple:
    a, b = str(a).strip().lower(), str(b).strip().lower()
    return tuple(sorted([a, b]))


def main():
    results = {}

    # ── Load test set (final assembled + 5 raw sources) ────────────────────
    test_final = pd.read_csv(EVAL_DIR / "biotic_interaction_test_set.csv")
    print(f"Final assembled test set: {len(test_final)} rows")

    sources = {
        "EP-A":         EVAL_DIR / "globi-relax_passages-triplets_2024-02-28_curation_EP.tsv",
        "EP-passage":   EVAL_DIR / "globi-passage_passages-triplets_2024-02-28_curation_EP.tsv",
        "BioTx-random": EVAL_DIR / "biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv",
        "eval-100":     EVAL_DIR / "eval_100.tsv",
        "gen-set-100":  EVAL_DIR / "gen_set_100.csv",
    }
    raw = {}
    for name, path in sources.items():
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        raw[name] = df
        print(f"  {name}: {len(df)} rows, cols={list(df.columns)[:6]}...")

    # Check the 499 vs 500 discrepancy
    all_raw_sentences = pd.concat([df["sentence"].astype(str) for df in raw.values()], ignore_index=True)
    results["row_count_check"] = {
        "sum_of_5_sources": len(all_raw_sentences),
        "final_assembled_test_set": len(test_final),
        "difference": len(all_raw_sentences) - len(test_final),
    }
    # Find which raw sentence(s) are missing from the final assembled set
    final_norm = set(test_final["sentence"].astype(str).map(normalize))
    missing = [s for s in all_raw_sentences if normalize(s) not in final_norm]
    results["row_count_check"]["missing_sentences_sample"] = missing[:5]
    print(f"\nRow count: 5 sources sum to {len(all_raw_sentences)}, "
          f"final test set has {len(test_final)} ({len(missing)} missing/dropped)")

    # ── Load training corpora actually used in the pipeline ────────────────
    train_sets = {
        "distillation_44k (champion training data)":
            pd.read_csv(TRAIN_DIR / "distillation_soft_labels.csv").head(44178)[["text"]],
        "v7_template_corpus (single-task baseline training data)":
            pd.read_csv(TRAIN_DIR / "training_data_globi_v7_llm_cleaned.csv")[["text"]],
        "v14_broader_corpus":
            pd.read_csv(TRAIN_DIR / "training_data_v14.csv")[["text"]],
    }
    for name, df in train_sets.items():
        print(f"  {name}: {len(df)} rows")

    test_sentences = test_final["sentence"].astype(str).tolist()
    test_norm = [normalize(s) for s in test_sentences]

    # ── 1. Exact-match check ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. EXACT-MATCH CHECK (normalized string equality)")
    print("=" * 60)
    exact_results = {}
    for name, df in train_sets.items():
        train_norm = set(df["text"].astype(str).map(normalize))
        hits = [s for s, n in zip(test_sentences, test_norm) if n in train_norm]
        exact_results[name] = {"n_exact_matches": len(hits), "examples": hits[:5]}
        print(f"  {name}: {len(hits)} exact matches out of {len(test_sentences)} test sentences")
    results["exact_match"] = exact_results

    # ── 2. Near-duplicate check (TF-IDF char n-gram cosine similarity) ─────
    print("\n" + "=" * 60)
    print(f"2. NEAR-DUPLICATE CHECK (TF-IDF char 3-5gram cosine > {NEAR_DUP_THRESHOLD})")
    print("=" * 60)
    near_dup_results = {}
    for name, df in train_sets.items():
        train_texts = df["text"].astype(str).tolist()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        all_texts = test_sentences + train_texts
        tfidf = vec.fit_transform(all_texts)
        test_tfidf = tfidf[:len(test_sentences)]
        train_tfidf = tfidf[len(test_sentences):]

        near_dups = []
        batch = 50
        for i in range(0, len(test_sentences), batch):
            sims = cosine_similarity(test_tfidf[i:i + batch], train_tfidf)
            for j, row in enumerate(sims):
                max_sim = row.max()
                if max_sim > NEAR_DUP_THRESHOLD:
                    best_train_idx = row.argmax()
                    near_dups.append({
                        "test_sentence": test_sentences[i + j],
                        "train_match": train_texts[best_train_idx],
                        "similarity": round(float(max_sim), 4),
                    })
        near_dup_results[name] = {"n_near_duplicates": len(near_dups), "examples": near_dups[:5]}
        print(f"  {name}: {len(near_dups)} near-duplicates (cosine > {NEAR_DUP_THRESHOLD})")
        for d in near_dups[:3]:
            print(f"    sim={d['similarity']:.3f}")
            print(f"      test:  {d['test_sentence'][:90]}")
            print(f"      train: {d['train_match'][:90]}")
    results["near_duplicate"] = near_dup_results

    # ── 3. Taxon-pair leakage check (GloBI-seeded sources only) ────────────
    print("\n" + "=" * 60)
    print("3. TAXON-PAIR LEAKAGE CHECK (GloBI-seeded test sources vs training pairs)")
    print("=" * 60)
    v7_df = pd.read_csv(TRAIN_DIR / "training_data_globi_v7_llm_cleaned.csv")
    v7_pairs = set()
    for _, row in v7_df.dropna(subset=["source_species", "target_species"]).iterrows():
        v7_pairs.add(normalize_pair(row["source_species"], row["target_species"]))
    print(f"  v7 template corpus: {len(v7_pairs)} unique taxon pairs")

    v14_df = pd.read_csv(TRAIN_DIR / "training_data_v14.csv")
    v14_pairs = set()
    for _, row in v14_df.dropna(subset=["source_species", "target_species"]).iterrows():
        v14_pairs.add(normalize_pair(row["source_species"], row["target_species"]))
    print(f"  v14 broader corpus: {len(v14_pairs)} unique taxon pairs")

    pair_leakage = {}
    for name in ["EP-A", "EP-passage", "BioTx-random"]:
        df = raw[name]
        if "species1_term" not in df.columns or "species2_term" not in df.columns:
            continue
        test_pairs = [normalize_pair(r["species1_term"], r["species2_term"])
                      for _, r in df.dropna(subset=["species1_term", "species2_term"]).iterrows()]
        n_total = len(test_pairs)
        in_v7 = sum(1 for p in test_pairs if p in v7_pairs)
        in_v14 = sum(1 for p in test_pairs if p in v14_pairs)
        pair_leakage[name] = {
            "n_test_pairs": n_total,
            "n_overlap_with_v7_template_pairs": in_v7,
            "n_overlap_with_v14_pairs": in_v14,
            "pct_overlap_v7": round(100 * in_v7 / n_total, 1) if n_total else None,
        }
        print(f"  {name}: {n_total} taxon pairs, {in_v7} ({100*in_v7/n_total:.1f}%) "
              f"also seed a v7 template sentence; {in_v14} also appear in v14")
    results["taxon_pair_leakage"] = pair_leakage

    # ── Save ─────────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {OUT}")


if __name__ == "__main__":
    main()
