#!/usr/bin/env python3
"""
Tier 2: Extract triples from all v14 sentences using the multi-task NER model.

Uses full_typed_a05_ner2 to:
  1. Classify sentences (binary: interaction yes/no, threshold 0.13)
  2. Decode BIO NER tags → HOST / PATHOGEN / SPECIES character spans
  3. Scan GloBI terms for interaction category
  4. Emit (HOST, category, PATHOGEN) triples with provenance
  5. Augment the Tier 1 KG with discovered triples

Input:  results/tier1/kg_v14.pkl
        classifier/data/training/training_data_v14.csv
        classifier/models/multitask/full_typed_a05_ner2/
Output: results/tier2/triples.jsonl
        results/tier2/kg_tier2.pkl
        results/tier2/kg_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import networkx as nx
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_utils import (
    normalize_name, parse_species_list,
    save_kg, load_kg, compute_kg_stats, map_interaction_type_to_category,
)

# Bring in classifier modules
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "classifier"))
sys.path.insert(0, str(ROOT / "classifier" / "experiments" / "multitask"))

# NER label set (must match full_typed scheme)
NER_LABELS = [
    "O",
    "B-HOST", "I-HOST",
    "B-PATHOGEN", "I-PATHOGEN",
    "B-SPECIES", "I-SPECIES",
    "B-INT", "I-INT",
]
NER_LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}

# A5: this was a hardcoded 0.13, which is exactly the F1-argmax on the EP-relax
# test set — the test set was deciding which KG edges got admitted. The threshold
# now comes from the checkpoint's recorded dev value (written by train_v2.py).
CLS_THRESHOLD = None  # resolved from the checkpoint by _threshold_from_checkpoint()


def _threshold_from_checkpoint(model_dir, override=None):
    import json as _json
    from pathlib import Path as _P
    if override is not None:
        return float(override)
    cfg = _P(model_dir) / "multitask_config.json"
    if cfg.exists():
        c = _json.loads(cfg.read_text())
        for k in ("threshold_dev", "best_threshold"):
            if c.get(k) is not None:
                return float(c[k])
    raise ValueError(
        f"No threshold recorded in {cfg}. Refusing to fall back to a literal: "
        f"V1's literal was the EP-relax argmax. Retrain with train_v2.py, or pass "
        f"an explicit threshold.")


def load_multitask_model(model_dir: Path, device: torch.device):
    from model import MultiTaskBiomedBERT
    print(f"  Loading MultiTaskBiomedBERT from {model_dir} ...", flush=True)
    model = MultiTaskBiomedBERT.load(str(model_dir), device=str(device))
    model.eval()

    cfg = json.load(open(model_dir / "multitask_config.json"))
    # Try local tokenizer first (saved by model.save()), fall back to encoder name
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_name"])
    return model, tokenizer


def decode_bio_spans(
    text: str,
    pred_label_ids: List[int],
    offset_mapping: List[Tuple[int, int]],
) -> Dict[str, List[dict]]:
    """Decode BIO tag sequence → entity spans grouped by type."""
    result: Dict[str, List[dict]] = {"HOST": [], "PATHOGEN": [], "SPECIES": [], "INT": []}
    current = None

    for label_id, (char_s, char_e) in zip(pred_label_ids, offset_mapping):
        if char_s == 0 and char_e == 0:  # special token (CLS/SEP/PAD)
            if current:
                result[current["type"]].append(current)
                current = None
            continue

        label = NER_LABELS[label_id]

        if label.startswith("B-"):
            if current:
                result[current["type"]].append(current)
            etype = label[2:]
            current = {"type": etype, "char_start": char_s, "char_end": char_e}

        elif label.startswith("I-"):
            etype = label[2:]
            if current and current["type"] == etype:
                current["char_end"] = char_e
            else:
                # Broken I-tag — treat as B-
                if current:
                    result[current["type"]].append(current)
                current = {"type": etype, "char_start": char_s, "char_end": char_e}

        else:  # O or INT
            if current:
                result[current["type"]].append(current)
                current = None

    if current:
        result[current["type"]].append(current)

    # Add span text
    for etype, spans in result.items():
        for span in spans:
            span["text"] = text[span["char_start"]:span["char_end"]].strip()

    return result


def scan_globi_terms(text: str) -> str:
    """Get interaction category from GloBI term scan."""
    try:
        from src.data.interaction_taxonomy import classify_interaction_type, _ensure_loaded
        from src.data.interaction_taxonomy import scan_globi_terms as _scan
        _ensure_loaded()
        terms = _scan(text)
        return classify_interaction_type(terms) or "GENERIC"
    except Exception:
        return "GENERIC"


def extract_triples(
    texts: List[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> List[dict]:
    """Run batch inference and extract triples from positive sentences."""
    all_triples = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_maps = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
            )

        cls_probs = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().tolist()
        ner_preds = out["ner_logits"].argmax(dim=-1).cpu().tolist()

        for j, (text, cls_prob) in enumerate(zip(batch, cls_probs)):
            if cls_prob < CLS_THRESHOLD:
                continue

            spans = decode_bio_spans(text, ner_preds[j], offset_maps[j])
            hosts = spans["HOST"]
            pathogens = spans["PATHOGEN"]
            species = spans["SPECIES"]

            # Get interaction category from GloBI scanner
            category = scan_globi_terms(text)

            # Primary: typed HOST + PATHOGEN spans
            for host in hosts:
                for pathogen in pathogens:
                    all_triples.append({
                        "source": pathogen["text"],
                        "target": host["text"],
                        "category": category,
                        "cls_prob": round(cls_prob, 4),
                        "sentence": text,
                        "evidence_idx": i + j,
                        "confidence": "high",
                        "entity_detection": "typed",
                    })

            # Fallback: HOST found but no PATHOGEN — use SPECIES as targets
            if hosts and not pathogens:
                for host in hosts:
                    for sp in species:
                        all_triples.append({
                            "source": sp["text"],
                            "target": host["text"],
                            "category": category,
                            "cls_prob": round(cls_prob, 4),
                            "sentence": text,
                            "evidence_idx": i + j,
                            "confidence": "low",
                            "entity_detection": "species_fallback",
                        })

        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(texts))
            print(f"  [{done}/{len(texts)}] triples so far: {len(all_triples)}", flush=True)

    return all_triples


def augment_kg(G: nx.MultiDiGraph, triples: List[dict]) -> nx.MultiDiGraph:
    """Add extracted triples to existing KG."""
    added = 0
    for t in triples:
        src_n = normalize_name(t["source"])
        tgt_n = normalize_name(t["target"])
        if not src_n or not tgt_n or src_n == tgt_n:
            continue
        G.add_node(src_n, name=src_n)
        G.add_node(tgt_n, name=tgt_n)
        G.add_edge(
            src_n, tgt_n,
            category=t["category"],
            source="tier2_ner",
            cls_prob=t["cls_prob"],
            confidence=t["confidence"],
            text=t["sentence"][:300],
        )
        added += 1
    print(f"  Added {added} new edges from NER extraction", flush=True)
    return G


def main():
    parser = argparse.ArgumentParser(description="Tier 2: NER-based triple extraction")
    parser.add_argument("--v14-path",   required=True)
    parser.add_argument("--model-dir",  required=True)
    parser.add_argument("--tier1-kg",   required=True, help="Path to tier1 KG pickle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("=== Tier 2: NER-based triple extraction ===", flush=True)
    t0 = time.time()

    # Load model
    model, tokenizer = load_multitask_model(Path(args.model_dir), device)

    global CLS_THRESHOLD
    CLS_THRESHOLD = _threshold_from_checkpoint(Path(args.model_dir))
    print(f'  cls threshold from checkpoint: {CLS_THRESHOLD}', flush=True)
    # Load sentences
    print(f"Loading {args.v14_path} ...", flush=True)
    df = pd.read_csv(args.v14_path)
    texts = df["text"].astype(str).tolist()
    print(f"  {len(texts):,} sentences to process", flush=True)

    # Extract triples
    triples_path = out / "triples.jsonl"
    # Resume: check if triples already written
    if triples_path.exists():
        print(f"  Resuming from existing {triples_path}", flush=True)
        triples = [json.loads(l) for l in open(triples_path)]
        print(f"  Loaded {len(triples)} existing triples", flush=True)
    else:
        print("Extracting triples...", flush=True)
        triples = extract_triples(texts, model, tokenizer, device, args.batch_size)
        with open(triples_path, "w") as f:
            for t in triples:
                f.write(json.dumps(t) + "\n")

    print(f"  Total triples extracted: {len(triples)}", flush=True)

    # Breakdown
    by_conf = {}
    for t in triples:
        c = t["confidence"]
        by_conf[c] = by_conf.get(c, 0) + 1
    print(f"  By confidence: {by_conf}", flush=True)

    # Augment KG
    print(f"Loading tier1 KG from {args.tier1_kg} ...", flush=True)
    G = load_kg(Path(args.tier1_kg))
    G = augment_kg(G, [t for t in triples if t["confidence"] == "high"])

    kg_path = out / "kg_tier2.pkl"
    save_kg(G, kg_path)
    print(f"  Augmented KG saved: {kg_path}", flush=True)

    # Stats
    stats = compute_kg_stats(G)
    stats["n_high_confidence_triples"] = by_conf.get("high", 0)
    stats["n_low_confidence_triples"] = by_conf.get("low", 0)
    stats_path = out / "kg_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nGraph: {stats['n_nodes']:,} nodes, {stats['n_edges']:,} edges", flush=True)
    print(f"Tier 2 complete in {time.time()-t0:.1f}s", flush=True)

    (out / "tier2_done").touch()


if __name__ == "__main__":
    main()
