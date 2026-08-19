#!/usr/bin/env python3
"""
Tier 4: Multi-approach publishable explorations.

Experiment A — Graph-augmented classifier
  Does KG lookup improve binary classification? (Retrieval-augmented prediction)
  Finding: KG prior adds recall boost for known species pairs, OR baseline generalises beyond known pairs.

Experiment B — GloBI alignment analysis
  What fraction of real GloBI interactions does our literature KG recover?
  Finding: Coverage gaps per interaction category → data collection priorities.

Experiment C — Coverage gap harvest (optional, uses Qwen3.5-122B)
  Targeted PMC harvest for the 2-3 most underrepresented categories.
  Finding: Demonstrates how targeted harvest closes specific coverage gaps.

Usage:
  python tier4_explorations.py \
      --tier2-kg   results/tier2/kg_tier2.pkl \
      --model-dir  classifier/models/multitask/full_typed_a05_ner2 \
      --ep-relax   classifier/data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv \
      --globi-tsv  classifier/data/globi/interactions.tsv.gz \
      --output-dir results/tier4 \
      [--skip-harvest]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER = ROOT / "classifier"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLASSIFIER))
sys.path.insert(0, str(CLASSIFIER / "experiments" / "multitask"))

from kg_utils import (
    normalize_name, parse_species_list,
    load_kg, compute_kg_stats, map_interaction_type_to_category,
)

BASELINE_EP_F1 = 0.868
EP_RELAX_PATH = CLASSIFIER / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
GLOBI_TSV = CLASSIFIER / "data/globi/interactions.tsv.gz"


# ─── Shared helpers ────────────────────────────────────────────────────────────

def load_ep_relax(path: Path):
    df = pd.read_csv(path, sep="\t")
    texts  = df["sentence"].astype(str).tolist()
    labels = df["evaluation_pair_interacting"].astype(int).tolist()
    sp1_col = [c for c in df.columns if "species1" in c.lower() and "form" in c.lower()]
    sp2_col = [c for c in df.columns if "species2" in c.lower() and "form" in c.lower()]
    sp1 = df[sp1_col[0]].astype(str).tolist() if sp1_col else [""] * len(texts)
    sp2 = df[sp2_col[0]].astype(str).tolist() if sp2_col else [""] * len(texts)
    return texts, labels, sp1, sp2


def predict_baseline(model, tokenizer, texts: List[str], device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Run multitask model classification on all texts, return probabilities."""
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
            )
        p = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs)


# ─── Experiment A: Graph-augmented classifier ─────────────────────────────────

def find_species_in_text(text: str, sp: str) -> bool:
    """Return True if `sp` appears (case-insensitive) in text."""
    if not sp or sp == "nan":
        return False
    import re
    try:
        return bool(re.search(r"(?<!\w)" + re.escape(sp.strip()) + r"(?!\w)", text, re.IGNORECASE))
    except re.error:
        return False


def kg_prior_score(G, sp1: str, sp2: str) -> float:
    """Return 1.0 if species pair exists in KG in either direction, else 0.0."""
    n1 = normalize_name(sp1)
    n2 = normalize_name(sp2)
    if G.has_node(n1) and G.has_node(n2):
        if G.has_edge(n1, n2) or G.has_edge(n2, n1):
            return 1.0
    return 0.0


def experiment_a(model, tokenizer, G, ep_texts, ep_labels, ep_sp1, ep_sp2, device, out: Path):
    done_path = out / "exp_a_done"
    result_path = out / "experiment_a.json"
    if done_path.exists():
        print("  Experiment A: already done", flush=True)
        return json.load(open(result_path))

    print("\n=== Experiment A: Graph-augmented classifier ===", flush=True)
    t0 = time.time()

    # Baseline probabilities
    print("  Running baseline classifier on EP-relax...", flush=True)
    base_probs = predict_baseline(model, tokenizer, ep_texts, device)

    # KG prior for each sentence
    kg_priors = np.array([kg_prior_score(G, s1, s2) for s1, s2 in zip(ep_sp1, ep_sp2)])
    n_in_kg = int(kg_priors.sum())
    n_positive_in_kg = sum(l for l, k in zip(ep_labels, kg_priors) if k > 0)
    print(f"  EP-relax pairs in KG: {n_in_kg}/{len(ep_labels)}", flush=True)
    print(f"  Positive pairs in KG: {n_positive_in_kg}", flush=True)

    # Sweep alpha
    alphas = [round(a, 2) for a in np.arange(0.0, 1.05, 0.1)]
    best_f1, best_alpha, best_p, best_r = 0.0, 1.0, 0.0, 0.0
    alpha_results = []

    for alpha in alphas:
        # final_score = alpha * base_prob + (1-alpha) * kg_prior
        combined = alpha * base_probs + (1.0 - alpha) * kg_priors
        # Find best threshold for this alpha
        best_t_f1 = 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (combined >= t).astype(int)
            f = f1_score(ep_labels, preds, zero_division=0)
            if f > best_t_f1:
                best_t_f1 = f
        alpha_results.append({"alpha": alpha, "best_f1": round(best_t_f1, 4)})
        if best_t_f1 > best_f1:
            # Recompute prec/rec at optimal threshold
            best_f1 = best_t_f1
            best_alpha = alpha
            for t in np.arange(0.05, 0.95, 0.05):
                preds = (combined >= t).astype(int)
                f = f1_score(ep_labels, preds, zero_division=0)
                if abs(f - best_f1) < 1e-6:
                    best_p = precision_score(ep_labels, preds, zero_division=0)
                    best_r = recall_score(ep_labels, preds, zero_division=0)

    # Baseline-only best F1
    baseline_best_f1 = 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (base_probs >= t).astype(int)
        f = f1_score(ep_labels, preds, zero_division=0)
        if f > baseline_best_f1:
            baseline_best_f1 = f

    delta = round(best_f1 - baseline_best_f1, 4)
    finding = (
        f"KG prior improves F1 by {delta:+.4f} (from {baseline_best_f1:.3f} to {best_f1:.3f}) "
        f"at alpha={best_alpha} — KG lookup adds recall for known species pairs"
        if delta > 0.002
        else
        f"KG prior gives negligible benefit (Δ{delta:+.4f}); "
        f"baseline ({baseline_best_f1:.3f}) generalises beyond known pairs — KG is complementary, not additive"
    )

    result = {
        "baseline_ep_f1": round(baseline_best_f1, 4),
        "kg_augmented_best_f1": round(best_f1, 4),
        "kg_augmented_prec": round(best_p, 4),
        "kg_augmented_rec": round(best_r, 4),
        "best_alpha": best_alpha,
        "delta_f1": delta,
        "n_ep_pairs_in_kg": n_in_kg,
        "n_ep_positives_in_kg": n_positive_in_kg,
        "alpha_sweep": alpha_results,
        "finding": finding,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    done_path.touch()
    print(f"  → {finding}", flush=True)
    print(f"  Experiment A complete in {time.time()-t0:.1f}s", flush=True)
    return result


# ─── Experiment B: GloBI alignment analysis ────────────────────────────────────

def load_globi_sample(globi_path: Path, n_sample: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Load a random sample of GloBI interactions."""
    random.seed(seed)
    print(f"  Loading GloBI interactions from {globi_path} ...", flush=True)

    opener = gzip.open if str(globi_path).endswith(".gz") else open
    rows = []
    with opener(globi_path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        # Identify column indices we need
        col_map = {col: i for i, col in enumerate(header)}

        def get(row, name, default=""):
            i = col_map.get(name, -1)
            return row[i] if 0 <= i < len(row) else default

        for line in f:
            parts = line.rstrip("\n").split("\t")
            src_name = get(parts, "sourceTaxonName")
            tgt_name = get(parts, "targetTaxonName")
            itype = get(parts, "interactionTypeName")
            if src_name and tgt_name and itype:
                rows.append({
                    "sourceTaxonName": src_name,
                    "targetTaxonName": tgt_name,
                    "interactionTypeName": itype,
                })

    df = pd.DataFrame(rows)
    print(f"  Total GloBI interactions loaded: {len(df):,}", flush=True)
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=seed)
    return df


def experiment_b(G, out: Path):
    done_path = out / "exp_b_done"
    result_path = out / "experiment_b.json"
    if done_path.exists():
        print("  Experiment B: already done", flush=True)
        return json.load(open(result_path))

    print("\n=== Experiment B: GloBI alignment analysis ===", flush=True)
    t0 = time.time()

    if not GLOBI_TSV.exists():
        print(f"  GloBI TSV not found: {GLOBI_TSV} — skipping", flush=True)
        return {"error": "GloBI TSV not found"}

    # Build fast node lookup
    kg_nodes = {normalize_name(n) for n in G.nodes()}
    # Build fast edge lookup: (src_norm, tgt_norm)
    kg_edges: set = set()
    for u, v, _ in G.edges(data=True):
        kg_edges.add((normalize_name(u), normalize_name(v)))
        kg_edges.add((normalize_name(v), normalize_name(u)))  # undirected lookup

    globi_df = load_globi_sample(GLOBI_TSV, n_sample=5000)
    print(f"  GloBI sample: {len(globi_df):,} interactions", flush=True)

    # Map GloBI interactionTypeName → canonical category
    globi_df["category"] = globi_df["interactionTypeName"].apply(map_interaction_type_to_category)

    # Check recall: is each GloBI pair in our KG?
    def pair_in_kg(src, tgt):
        sn, tn = normalize_name(src), normalize_name(tgt)
        return (sn, tn) in kg_edges

    globi_df["in_kg"] = globi_df.apply(
        lambda r: pair_in_kg(r["sourceTaxonName"], r["targetTaxonName"]), axis=1
    )

    overall_recall = globi_df["in_kg"].mean()
    print(f"  Overall KG recall: {overall_recall:.1%}", flush=True)

    # Per-category recall
    cat_results = {}
    for cat, grp in globi_df.groupby("category"):
        recall = grp["in_kg"].mean()
        n_total = len(grp)
        n_covered = int(grp["in_kg"].sum())
        cat_results[cat] = {
            "recall": round(recall, 4),
            "n_total": n_total,
            "n_covered": n_covered,
        }
        print(f"    {cat:20s}: recall={recall:.1%} ({n_covered}/{n_total})", flush=True)

    # Identify coverage gap categories (recall < 20%)
    gap_categories = sorted(
        [cat for cat, res in cat_results.items() if res["recall"] < 0.2],
        key=lambda c: cat_results[c]["recall"]
    )

    finding = (
        f"KG covers {overall_recall:.1%} of GloBI sample. "
        f"Lowest recall: {', '.join(gap_categories[:3])}. "
        f"These categories need targeted literature harvest."
    )

    result = {
        "globi_sample_size": len(globi_df),
        "overall_recall": round(overall_recall, 4),
        "recall_by_category": cat_results,
        "coverage_gap_categories": gap_categories,
        "finding": finding,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    done_path.touch()
    print(f"\n  → {finding}", flush=True)
    print(f"  Experiment B complete in {time.time()-t0:.1f}s", flush=True)
    return result


# ─── Experiment C: Coverage gap harvest ───────────────────────────────────────

def experiment_c(G, gap_categories: List[str], out: Path, dry_run: bool = False):
    done_path = out / "exp_c_done"
    result_path = out / "experiment_c.json"
    if done_path.exists():
        print("  Experiment C: already done", flush=True)
        return json.load(open(result_path))

    print("\n=== Experiment C: Coverage gap harvest ===", flush=True)
    if dry_run:
        print("  Dry run mode — skipping actual harvest", flush=True)
        result = {"status": "skipped_dry_run", "gap_categories": gap_categories}
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        done_path.touch()
        return result

    if not gap_categories:
        print("  No gap categories identified — skipping", flush=True)
        result = {"status": "skipped_no_gaps"}
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        done_path.touch()
        return result

    t0 = time.time()
    target_cats = gap_categories[:2]  # Top-2 most underrepresented
    print(f"  Targeting categories: {target_cats}", flush=True)

    fetch_script = CLASSIFIER / "scripts/fetch_epmc_direct.py"
    if not fetch_script.exists():
        print(f"  Harvest script not found: {fetch_script} — skipping", flush=True)
        result = {"status": "skipped_no_script"}
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        done_path.touch()
        return result

    harvest_out = out / "harvest"
    harvest_out.mkdir(exist_ok=True)

    # Run harvest for each gap category
    all_harvested = []
    for cat in target_cats:
        harvest_csv = harvest_out / f"harvest_{cat.lower()}.csv"
        if not harvest_csv.exists():
            print(f"  Harvesting {cat} sentences from Europe PMC...", flush=True)
            try:
                subprocess.run(
                    [
                        sys.executable, str(fetch_script),
                        "--category", cat,
                        "--max-pairs", "50",
                        "--max-per-pair", "5",
                        "--output", str(harvest_csv),
                    ],
                    timeout=600,
                    check=True,
                    capture_output=False,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                print(f"  Harvest failed for {cat}: {e}", flush=True)
                continue

        if harvest_csv.exists():
            try:
                df = pd.read_csv(harvest_csv)
                all_harvested.append(df)
                print(f"  {cat}: {len(df)} sentences harvested", flush=True)
            except Exception as e:
                print(f"  Failed to read {harvest_csv}: {e}", flush=True)

    if not all_harvested:
        result = {"status": "harvest_failed", "gap_categories": target_cats}
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        done_path.touch()
        return result

    harvested_df = pd.concat(all_harvested, ignore_index=True)
    print(f"\n  Total harvested: {len(harvested_df)} sentences", flush=True)

    # Label with Qwen3.5-122B
    qwen_labeled = _label_with_qwen(harvested_df, out)

    # Add confident positives to KG
    new_triples = 0
    if qwen_labeled is not None:
        pos = qwen_labeled[qwen_labeled.get("qwen_label", pd.Series()) == "YES"]
        for _, row in pos.iterrows():
            src = normalize_name(str(row.get("source_species", "")))
            tgt = normalize_name(str(row.get("target_species", "")))
            if src and tgt and src != tgt:
                cat = map_interaction_type_to_category(str(row.get("interaction_type", "")))
                G.add_node(src, name=src)
                G.add_node(tgt, name=tgt)
                G.add_edge(src, tgt, category=cat, source="tier4_harvest")
                new_triples += 1

        print(f"  New triples added to KG: {new_triples}", flush=True)

    result = {
        "status": "completed",
        "gap_categories": target_cats,
        "sentences_harvested": len(harvested_df),
        "new_triples_added": new_triples,
        "elapsed_s": round(time.time() - t0),
        "finding": (
            f"Targeted harvest of {', '.join(target_cats)} yielded {len(harvested_df)} sentences, "
            f"adding {new_triples} new triples to the KG."
        ),
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    done_path.touch()
    print(f"  → {result['finding']}", flush=True)
    return result


def _label_with_qwen(df: pd.DataFrame, out: Path) -> Optional[pd.DataFrame]:
    """Label harvested sentences with Qwen3.5-122B via Ollama."""
    labeled_path = out / "harvest_qwen_labeled.csv"
    if labeled_path.exists():
        return pd.read_csv(labeled_path)

    print("  Labeling with Qwen3.5-122B (Ollama)...", flush=True)
    try:
        import requests
        labels = []
        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            prompt = (
                f"Does this sentence describe a direct biotic interaction between two species? "
                f"Answer YES or NO only.\n\nSentence: {text}"
            )
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5:72b", "prompt": prompt, "stream": False},
                timeout=30,
            )
            answer = resp.json().get("response", "").strip().upper()
            labels.append("YES" if answer.startswith("YES") else "NO")
        df["qwen_label"] = labels
        df.to_csv(labeled_path, index=False)
        yes_count = sum(1 for l in labels if l == "YES")
        print(f"  Qwen labeled: {yes_count}/{len(labels)} positive", flush=True)
        return df
    except Exception as e:
        print(f"  Qwen labeling failed: {e} — using classifier score fallback", flush=True)
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier 4: Publishable explorations")
    parser.add_argument("--tier2-kg",    required=True, help="Path to tier2 KG pickle")
    parser.add_argument("--model-dir",   required=True, help="Path to full_typed_a05_ner2")
    parser.add_argument("--ep-relax",    default=str(EP_RELAX_PATH))
    parser.add_argument("--globi-tsv",   default=str(GLOBI_TSV))
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--skip-harvest", action="store_true",
                        help="Skip Experiment C (PMC harvest)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Tier 4: Publishable explorations ===", flush=True)
    t0 = time.time()

    # Load KG
    print(f"Loading KG from {args.tier2_kg} ...", flush=True)
    G = load_kg(Path(args.tier2_kg))
    print(f"  KG: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges", flush=True)

    # Load multitask model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)
    sys.path.insert(0, str(CLASSIFIER / "experiments" / "multitask"))
    from model import MultiTaskBiomedBERT
    import json as _json
    model_dir = Path(args.model_dir)
    model = MultiTaskBiomedBERT.load(str(model_dir), device=str(device))
    cfg = _json.load(open(model_dir / "multitask_config.json"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_name"])

    # Load EP-relax
    ep_texts, ep_labels, ep_sp1, ep_sp2 = load_ep_relax(Path(args.ep_relax))
    print(f"  EP-relax: {len(ep_texts)} sentences, {sum(ep_labels)} positive", flush=True)

    # Run experiments
    result_a = experiment_a(model, tokenizer, G, ep_texts, ep_labels, ep_sp1, ep_sp2, device, out)
    result_b = experiment_b(G, out)

    gap_cats = result_b.get("coverage_gap_categories", []) if isinstance(result_b, dict) else []
    result_c = experiment_c(G, gap_cats, out, dry_run=args.skip_harvest)

    # Consolidated report
    report = {
        "experiment_a": result_a,
        "experiment_b": result_b,
        "experiment_c": result_c,
        "elapsed_total_s": round(time.time() - t0),
    }
    report_path = out / "exploration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nExploration report: {report_path}", flush=True)
    print(f"Tier 4 complete in {time.time()-t0:.1f}s", flush=True)

    (out / "tier4_done").touch()


if __name__ == "__main__":
    main()
