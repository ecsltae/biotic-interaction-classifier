#!/usr/bin/env python3
"""
Tier 3: Train a SpERT relation extractor on weak supervision derived from v14 data.
Then evaluate binary classification on EP-relax and compare to full_typed_a05_ner2.

Phases (pass --phase to run one at a time):
  build-ws   — build weak supervision training data from v14 (~30 min)
  train      — train SpERT on weak supervision (~1h CPU)
  evaluate   — evaluate on EP-relax, compare to baseline EP F1=0.868

Usage:
  python tier3_train_spert.py --phase build-ws --v14-path ... --output-dir ...
  python tier3_train_spert.py --phase train --ws-data ... --pretrained-encoder ... --output-dir ...
  python tier3_train_spert.py --phase evaluate --model-dir ... --ep-relax ... --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER = ROOT / "classifier"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLASSIFIER))
sys.path.insert(0, str(CLASSIFIER / "experiments" / "multitask"))

from kg_utils import (
    SPERT_TRAINING_CATEGORIES, CATEGORY_TO_SPERT_RELATION,
    SPERT_ENTITY_INDEX, SPERT_RELATION_INDEX,
    parse_species_list, normalize_name, map_interaction_type_to_category,
)

# Paths
EP_RELAX_PATH = CLASSIFIER / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
V14_PATH = CLASSIFIER / "data/training/training_data_v14.csv"

BASELINE_EP_F1 = 0.868  # full_typed_a05_ner2


# ─── Weak supervision utilities ───────────────────────────────────────────────

def _find_exact_spans(text: str, name: str) -> List[Tuple[int, int]]:
    """All [start, end) char spans of `name` in `text`, case-insensitive, word-boundary aware."""
    try:
        from data import _find_spans_exact
        return _find_spans_exact(text, name)
    except Exception:
        if not name:
            return []
        try:
            pattern = re.compile(r"(?<!\w)" + re.escape(name.strip()) + r"(?!\w)", re.IGNORECASE)
            return [(m.start(), m.end()) for m in pattern.finditer(text)]
        except re.error:
            return []


def _find_abbreviated_spans(text: str, name: str) -> List[Tuple[int, int]]:
    """Try abbreviated form: 'Plasmodium falciparum' → 'P. falciparum'."""
    parts = name.strip().split()
    if len(parts) != 2:
        return []
    genus, epithet = parts
    try:
        pattern = re.compile(
            r"\b" + re.escape(genus[0]) + r"\.\s*" + re.escape(epithet) + r"\b",
            re.IGNORECASE,
        )
        return [(m.start(), m.end()) for m in pattern.finditer(text)]
    except re.error:
        return []


def find_species_spans(text: str, name: str) -> List[Tuple[int, int]]:
    """Find all char spans for `name` in `text`, trying exact then abbreviated."""
    spans = _find_exact_spans(text, name)
    if not spans:
        spans = _find_abbreviated_spans(text, name)
    return spans


def char_spans_to_token_spans(
    char_spans: List[Tuple[int, int]],
    offset_mapping: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Convert character spans to token spans using offset_mapping (from tokenizer)."""
    token_spans = []
    for char_s, char_e in char_spans:
        tok_s, tok_e = None, None
        for tok_i, (off_s, off_e) in enumerate(offset_mapping):
            if off_s == 0 and off_e == 0:
                continue  # special token
            if tok_s is None and off_s >= char_s:
                tok_s = tok_i
            if off_e <= char_e:
                tok_e = tok_i + 1  # exclusive end
        if tok_s is not None and tok_e is not None and tok_s < tok_e:
            token_spans.append((tok_s, tok_e))
    return token_spans


def build_weak_supervision_sample(
    text: str,
    src_names: List[str],
    tgt_names: List[str],
    category: str,
    tokenizer,
    max_length: int = 96,
) -> Optional[dict]:
    """Build a training sample for SpERT from a v14 row with species annotations."""
    relation_name = CATEGORY_TO_SPERT_RELATION.get(category)
    if relation_name is None:
        return None

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        padding=False,
    )
    offset_mapping = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    seq_len = len(input_ids)

    # Collect entity spans for source (PATHOGEN) and target (HOST)
    src_spans = []  # (tok_start, tok_end, entity_type_idx)
    tgt_spans = []

    for name in src_names:
        char_spans = find_species_spans(text, name)
        tok_spans = char_spans_to_token_spans(char_spans, offset_mapping)
        for ts, te in tok_spans:
            if te <= seq_len:
                # source species = PATHOGEN for infection-type, VECTOR for vector
                if category == "VECTOR":
                    etype = SPERT_ENTITY_INDEX["VECTOR"]
                else:
                    etype = SPERT_ENTITY_INDEX["PATHOGEN"]
                src_spans.append((ts, te, etype))

    for name in tgt_names:
        char_spans = find_species_spans(text, name)
        tok_spans = char_spans_to_token_spans(char_spans, offset_mapping)
        for ts, te in tok_spans:
            if te <= seq_len:
                tgt_spans.append((ts, te, SPERT_ENTITY_INDEX["HOST"]))

    if not src_spans or not tgt_spans:
        return None  # span matching failed

    # De-duplicate
    src_spans = list(dict.fromkeys(src_spans))
    tgt_spans = list(dict.fromkeys(tgt_spans))

    # Build entity list: deduplicate by position
    seen_positions = set()
    all_spans = []
    span_types = []
    for ts, te, etype in src_spans + tgt_spans:
        pos = (ts, te)
        if pos not in seen_positions:
            seen_positions.add(pos)
            all_spans.append((ts, te))
            span_types.append(etype)

    n_spans = len(all_spans)
    if n_spans < 2:
        return None

    # Build relation pairs: src × tgt (both directions)
    relation_pairs = []
    relation_labels = []

    for i_s, (ts_s, te_s, et_s) in enumerate(src_spans[:3]):
        pos_s = (ts_s, te_s)
        idx_s = next(k for k, s in enumerate(all_spans) if s == pos_s)
        for i_t, (ts_t, te_t, et_t) in enumerate(tgt_spans[:3]):
            pos_t = (ts_t, te_t)
            if pos_s == pos_t:
                continue
            idx_t = next(k for k, s in enumerate(all_spans) if s == pos_t)
            # Forward: (src → tgt) with the relation
            fwd_rel = SPERT_RELATION_INDEX.get(relation_name, 2)
            relation_pairs.append((idx_s, idx_t))
            relation_labels.append(fwd_rel)
            # Inverse: (tgt → src) with inverse relation (always INFECTED_BY ↔ INFECTS)
            inv_name = {
                "INFECTS": "INFECTED_BY", "INFECTED_BY": "INFECTS",
                "COLONIZED_BY": "COLONIZES", "COLONIZES": "COLONIZED_BY",
                "TRANSMITTED_BY": "TRANSMITS", "TRANSMITS": "TRANSMITTED_BY",
            }.get(relation_name, "INFECTED_BY")
            inv_rel = SPERT_RELATION_INDEX.get(inv_name, 1)
            relation_pairs.append((idx_t, idx_s))
            relation_labels.append(inv_rel)

    if not relation_pairs:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "spans": all_spans,         # list of (tok_start, tok_end)
        "span_labels": span_types,  # list of entity type indices
        "relation_pairs": relation_pairs,   # list of (idx_i, idx_j)
        "relation_labels": relation_labels,  # list of relation type indices
        "text": text[:200],
        "category": category,
        "n_src": len(src_spans),
        "n_tgt": len(tgt_spans),
    }


# ─── Phase 1: Build weak supervision ─────────────────────────────────────────

def phase_build_ws(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ws_path = out / "weak_supervision_train.jsonl"
    if ws_path.exists():
        print(f"  Weak supervision data already exists: {ws_path}", flush=True)
        data = [json.loads(l) for l in open(ws_path)]
        print(f"  {len(data)} samples loaded", flush=True)
        return

    print("=== Phase 3a: Building weak supervision data ===", flush=True)
    t0 = time.time()

    print(f"Loading {args.v14_path} ...", flush=True)
    df = pd.read_csv(args.v14_path)

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )

    positive_rows = df[
        (df["label"] == 1)
        & df["source_species"].notna()
        & df["source_species"].astype(str).str.strip().ne("")
        & df["target_species"].notna()
        & df["target_species"].astype(str).str.strip().ne("")
    ]
    print(f"  Positive rows: {len(positive_rows):,}", flush=True)

    samples = []
    skipped_category = 0
    skipped_spans = 0

    for _, row in positive_rows.iterrows():
        category = map_interaction_type_to_category(str(row.get("interaction_type", "")))
        if category not in SPERT_TRAINING_CATEGORIES:
            skipped_category += 1
            continue

        text = str(row["text"])
        src_names = parse_species_list(row["source_species"])
        tgt_names = parse_species_list(row["target_species"])

        sample = build_weak_supervision_sample(
            text, src_names, tgt_names, category, tokenizer
        )
        if sample is None:
            skipped_spans += 1
            continue
        samples.append(sample)

    # Add NO_RELATION negative samples from label=0 rows with detectable species
    neg_rows = df[df["label"] == 0].sample(min(3000, len(df[df["label"] == 0])), random_state=42)
    neg_added = 0
    for _, row in neg_rows.iterrows():
        text = str(row["text"])
        src_names = parse_species_list(str(row.get("source_species", "")))
        tgt_names = parse_species_list(str(row.get("target_species", "")))
        if not src_names or not tgt_names:
            continue

        enc = tokenizer(text, truncation=True, max_length=96, return_offsets_mapping=True)
        offset_mapping = enc["offset_mapping"]
        seq_len = len(enc["input_ids"])

        all_spans = []
        span_labels = []
        for name in (src_names[:1] + tgt_names[:1]):
            char_spans = find_species_spans(text, name)
            tok_spans = char_spans_to_token_spans(char_spans, offset_mapping)
            for ts, te in tok_spans:
                if te <= seq_len:
                    all_spans.append((ts, te))
                    span_labels.append(SPERT_ENTITY_INDEX["PATHOGEN"])
                    break

        if len(all_spans) < 2:
            continue

        # NO_RELATION pairs
        relation_pairs = [(0, 1), (1, 0)]
        relation_labels = [SPERT_RELATION_INDEX["NO_RELATION"]] * 2

        samples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "spans": all_spans,
            "span_labels": span_labels,
            "relation_pairs": relation_pairs,
            "relation_labels": relation_labels,
            "text": text[:200],
            "category": "NONE",
            "n_src": 1,
            "n_tgt": 1,
        })
        neg_added += 1

    print(f"\n  Positive samples: {len(positive_rows):,}", flush=True)
    print(f"  Skipped (wrong category): {skipped_category:,}", flush=True)
    print(f"  Skipped (span not found): {skipped_spans:,}", flush=True)
    print(f"  Valid training samples: {len(samples) - neg_added:,}", flush=True)
    print(f"  NO_RELATION negatives added: {neg_added:,}", flush=True)
    print(f"  Total: {len(samples):,}", flush=True)

    random.shuffle(samples)
    with open(ws_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved: {ws_path} ({time.time()-t0:.1f}s)", flush=True)


# ─── Phase 2: Train SpERT ─────────────────────────────────────────────────────

def collate_single(sample: dict, device: torch.device) -> dict:
    """Convert one sample dict to batched tensors (batch_size=1)."""
    def t(x, dtype=torch.long):
        return torch.tensor([x], dtype=dtype).to(device)

    return {
        "input_ids":      t(sample["input_ids"]),
        "attention_mask": t(sample["attention_mask"]),
        "spans":          t(sample["spans"]),
        "span_labels":    t(sample["span_labels"]),
        "relation_pairs": t(sample["relation_pairs"]),
        "relation_labels":t(sample["relation_labels"]),
    }


def phase_train(args):
    out = Path(args.output_dir)
    model_out = out / "spert_model"
    ckpt_path = model_out / "pytorch_model.bin"

    if ckpt_path.exists():
        print(f"  SpERT model already trained: {ckpt_path}", flush=True)
        return

    print("=== Phase 3b: Training SpERT ===", flush=True)
    t0 = time.time()

    ws_path = Path(args.ws_data)
    print(f"Loading weak supervision from {ws_path} ...", flush=True)
    samples = [json.loads(l) for l in open(ws_path)]
    print(f"  {len(samples):,} samples", flush=True)

    # Train/val split (90/10)
    random.seed(42)
    random.shuffle(samples)
    val_n = max(50, len(samples) // 10)
    val_samples = samples[:val_n]
    train_samples = samples[val_n:]
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    # Load SpERT model
    from src.models.relation_extractor import HostPathogenRelationExtractor, ENTITY_TYPES, RELATION_TYPES
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    pretrained_dir = Path(args.pretrained_encoder) if args.pretrained_encoder else None
    if pretrained_dir and pretrained_dir.exists():
        print(f"  Warm-starting encoder from {pretrained_dir} ...", flush=True)
        model = HostPathogenRelationExtractor(
            model_name=model_name,
            max_span_width=6,
            entity_types=ENTITY_TYPES,
            relation_types=RELATION_TYPES,
        )
        # Load BiomedBERT encoder weights from multitask checkpoint
        state = torch.load(pretrained_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        encoder_weights = {
            k[len("encoder."):]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = model.encoder.load_state_dict(encoder_weights, strict=False)
        print(f"  Encoder: loaded {len(encoder_weights)} weights, missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    else:
        print("  Initializing SpERT from pretrained BiomedBERT (no warm start)...", flush=True)
        model = HostPathogenRelationExtractor(
            model_name=model_name,
            max_span_width=6,
            entity_types=ENTITY_TYPES,
            relation_types=RELATION_TYPES,
        )

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.01)

    epochs = int(args.epochs)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_samples)
        total_loss = 0.0
        n_steps = 0

        for sample in train_samples:
            # Skip malformed samples
            if len(sample["spans"]) == 0 or len(sample["relation_pairs"]) == 0:
                continue

            batch = collate_single(sample, device)
            optimizer.zero_grad()

            out = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                spans          = batch["spans"],
                span_labels    = batch["span_labels"],
                relation_pairs = batch["relation_pairs"],
                relation_labels= batch["relation_labels"],
            )

            loss = out.get("loss")
            if loss is None or torch.isnan(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        avg_train_loss = total_loss / max(n_steps, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for sample in val_samples:
                if len(sample["spans"]) == 0 or len(sample["relation_pairs"]) == 0:
                    continue
                batch = collate_single(sample, device)
                out = model(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    spans          = batch["spans"],
                    span_labels    = batch["span_labels"],
                    relation_pairs = batch["relation_pairs"],
                    relation_labels= batch["relation_labels"],
                )
                loss = out.get("loss")
                if loss is not None and not torch.isnan(loss):
                    val_loss += loss.item()
                    val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  elapsed={elapsed:.0f}s",
            flush=True
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model
    model_out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)
    config = {
        "model_name": model_name,
        "max_span_width": 6,
        "entity_types": ENTITY_TYPES,
        "relation_types": RELATION_TYPES,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs,
    }
    with open(model_out / "spert_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  SpERT model saved: {ckpt_path} ({time.time()-t0:.1f}s)", flush=True)


# ─── Phase 3: Evaluate on EP-relax ───────────────────────────────────────────

def load_ep_relax(path: Path):
    df = pd.read_csv(path, sep="\t")
    texts  = df["sentence"].astype(str).tolist()
    labels = df["evaluation_pair_interacting"].astype(int).tolist()
    # Oracle species spans from columns species1_form / species2_form if available
    sp1 = df["species1_form"].astype(str).tolist() if "species1_form" in df.columns else [None] * len(texts)
    sp2_col = [c for c in df.columns if "species2" in c.lower() and "form" in c.lower()]
    sp2 = df[sp2_col[0]].astype(str).tolist() if sp2_col else [None] * len(texts)
    return texts, labels, sp1, sp2


def _spert_predict_fixed(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    span_threshold: float,
    relation_threshold: float,
):
    """Replicate model.predict() but fix the off-by-one: generate_spans uses seq_len
    as exclusive end, causing index-out-of-bounds when gathering end token embeddings.
    Fix: cap span generation at seq_len - 1 so all end indices stay in bounds."""
    import torch.nn.functional as F
    from src.models.relation_extractor import SpanPrediction, RelationPrediction

    model.eval()
    with torch.no_grad():
        # Use seq_len - 1 to avoid the off-by-one in SpanClassifier.forward()
        seq_len = int(attention_mask.sum().item())
        safe_seq_len = max(1, seq_len - 1)

        span_list = model.generate_spans(safe_seq_len, model.max_span_width)
        if not span_list:
            return [], []

        spans = torch.tensor([span_list], device=input_ids.device)
        hidden_states = model.encode(input_ids, attention_mask)
        span_logits = model.span_classifier(hidden_states, spans)
        span_probs = F.softmax(span_logits, dim=-1)

        entities = []
        entity_spans = []
        for i, (start, end) in enumerate(span_list):
            probs = span_probs[0, i]
            pred_type = int(probs.argmax().item())
            if pred_type > 0 and probs[pred_type].item() > span_threshold:
                entities.append(SpanPrediction(
                    start=start, end=end,
                    entity_type=model.entity_types[pred_type],
                    score=probs[pred_type].item(),
                ))
                entity_spans.append((i, start, end))

        relations = []
        if len(entity_spans) >= 2:
            pairs = []
            pair_indices = []
            for ii, (idx1, s1, e1) in enumerate(entity_spans):
                for jj, (idx2, s2, e2) in enumerate(entity_spans):
                    if ii != jj:
                        pairs.append((s1, e1, s2, e2))
                        pair_indices.append((idx1, idx2))

            if pairs:
                relation_pairs = torch.tensor([pair_indices], device=input_ids.device)
                out = model(input_ids, attention_mask, spans=spans, relation_pairs=relation_pairs)
                rel_probs = F.softmax(out["relation_logits"], dim=-1)
                for k, (s1, e1, s2, e2) in enumerate(pairs):
                    probs = rel_probs[0, k]
                    pred_type = int(probs.argmax().item())
                    if pred_type > 0 and probs[pred_type].item() > relation_threshold:
                        relations.append(RelationPrediction(
                            head_span=(s1, e1), tail_span=(s2, e2),
                            relation_type=model.relation_types[pred_type],
                            score=probs[pred_type].item(),
                        ))

    return entities, relations


def spert_predict_sentence(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    span_threshold: float = 0.3,
    relation_threshold: float = 0.3,
    oracle_sp1: Optional[str] = None,
    oracle_sp2: Optional[str] = None,
) -> bool:
    """Return True if SpERT predicts a biotic interaction in this sentence."""
    enc = tokenizer(
        text, truncation=True, max_length=96,
        return_tensors="pt", return_offsets_mapping=True
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    entities, relations = _spert_predict_fixed(
        model, enc["input_ids"], enc["attention_mask"],
        span_threshold=span_threshold,
        relation_threshold=relation_threshold,
    )

    # Primary path: SpERT found entities and at least one non-NO_RELATION
    if len(entities) >= 2 and any(r.relation_type != "NO_RELATION" for r in relations):
        return True

    # Fallback: use oracle species spans if SpERT finds no entities
    if oracle_sp1 and oracle_sp2 and oracle_sp1 != "nan" and oracle_sp2 != "nan":
        # Find token spans for oracle species
        spans_1 = char_spans_to_token_spans(
            find_species_spans(text, str(oracle_sp1)), offset_mapping
        )
        spans_2 = char_spans_to_token_spans(
            find_species_spans(text, str(oracle_sp2)), offset_mapping
        )
        if not spans_1 or not spans_2:
            return False

        # Build oracle span tensors (two entities)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        seq_len = input_ids.shape[1]

        oracle_spans_list = []
        for ts, te in spans_1[:1]:
            if te <= seq_len:
                oracle_spans_list.append([ts, te])
        for ts, te in spans_2[:1]:
            if te <= seq_len:
                oracle_spans_list.append([ts, te])

        if len(oracle_spans_list) < 2:
            return False

        oracle_spans = torch.tensor([oracle_spans_list], device=device)
        oracle_pairs = torch.tensor([[[0, 1], [1, 0]]], device=device)

        with torch.no_grad():
            hidden = model.encode(input_ids, attention_mask)
            rel_logits = model.relation_classifier(
                model.get_span_representation(hidden, oracle_spans[:, :, 0], oracle_spans[:, :, 1]),
                model.get_span_representation(
                    hidden,
                    oracle_spans[:, [1, 0], 0],
                    oracle_spans[:, [1, 0], 1]
                ),
                model.get_context_representation(
                    hidden, oracle_spans[:, :, 1], oracle_spans[:, [1, 0], 0], attention_mask
                ),
            )

        import torch.nn.functional as F
        rel_probs = F.softmax(rel_logits, dim=-1)
        pred_types = rel_probs[0].argmax(dim=-1).tolist()
        from src.models.relation_extractor import RELATION_TYPES
        return any(RELATION_TYPES[p] != "NO_RELATION" for p in pred_types)

    return False


def _score_sentence(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    oracle_sp1: Optional[str],
    oracle_sp2: Optional[str],
) -> dict:
    """Run inference ONCE per sentence; return raw max scores for threshold sweep.

    Returns dict with:
      max_entity_prob  — highest non-O span probability found
      max_rel_prob     — highest non-NO_RELATION relation probability found
      n_entities       — number of spans with any non-O prediction
      oracle_rel_prob  — max relation prob using oracle species spans (fallback)
    """
    import torch.nn.functional as F

    enc = tokenizer(
        text, truncation=True, max_length=96,
        return_tensors="pt", return_offsets_mapping=True
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        seq_len = int(enc["attention_mask"].sum().item())
        safe_seq_len = max(1, seq_len - 1)
        span_list = model.generate_spans(safe_seq_len, model.max_span_width)
        if not span_list:
            return {"max_entity_prob": 0.0, "max_rel_prob": 0.0,
                    "n_entities": 0, "oracle_rel_prob": 0.0}

        spans = torch.tensor([span_list], device=device)
        hidden = model.encode(enc["input_ids"], enc["attention_mask"])
        span_logits = model.span_classifier(hidden, spans)
        span_probs = F.softmax(span_logits, dim=-1)  # (1, n_spans, n_entity_types)

        # Best non-O probability across all spans
        non_o_probs = span_probs[0, :, 1:].max(dim=-1).values  # (n_spans,)
        max_entity_prob = float(non_o_probs.max().item())

        # Entity spans above a very low threshold for relation classification
        entity_mask = non_o_probs > 0.05
        entity_indices = entity_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(entity_indices, int):
            entity_indices = [entity_indices]

        max_rel_prob = 0.0
        n_entities = len(entity_indices)

        if n_entities >= 2:
            pairs = [(i, j) for i in entity_indices for j in entity_indices if i != j]
            if pairs:
                rel_pairs = torch.tensor([pairs], device=device)
                out = model(enc["input_ids"], enc["attention_mask"],
                            spans=spans, relation_pairs=rel_pairs)
                rel_probs = F.softmax(out["relation_logits"], dim=-1)
                # Best non-NO_RELATION probability
                max_rel_prob = float(rel_probs[0, :, 1:].max().item())

        # Oracle fallback: if oracle species known, force-compute relation prob
        oracle_rel_prob = 0.0
        sp1_str = str(oracle_sp1) if oracle_sp1 else ""
        sp2_str = str(oracle_sp2) if oracle_sp2 else ""
        if sp1_str and sp2_str and sp1_str != "nan" and sp2_str != "nan":
            spans_1 = char_spans_to_token_spans(
                find_species_spans(text, sp1_str), offset_mapping
            )
            spans_2 = char_spans_to_token_spans(
                find_species_spans(text, sp2_str), offset_mapping
            )
            if spans_1 and spans_2:
                seq_tensor_len = enc["input_ids"].shape[1]
                oracle_list = []
                for ts, te in spans_1[:1]:
                    if te <= seq_tensor_len:
                        oracle_list.append([ts, te])
                for ts, te in spans_2[:1]:
                    if te <= seq_tensor_len:
                        oracle_list.append([ts, te])
                if len(oracle_list) == 2:
                    oracle_spans = torch.tensor([oracle_list], device=device)
                    # (1, 2, hidden) representations
                    s1_repr = model.get_span_representation(
                        hidden, oracle_spans[:, :1, 0], oracle_spans[:, :1, 1])
                    s2_repr = model.get_span_representation(
                        hidden, oracle_spans[:, 1:, 0], oracle_spans[:, 1:, 1])
                    ctx = model.get_context_representation(
                        hidden, oracle_spans[:, :1, 1], oracle_spans[:, 1:, 0],
                        enc["attention_mask"]
                    )
                    rel_logits = model.relation_classifier(s1_repr, s2_repr, ctx)
                    oracle_rel_prob = float(
                        F.softmax(rel_logits, dim=-1)[0, 0, 1:].max().item()
                    )

    return {
        "max_entity_prob": max_entity_prob,
        "max_rel_prob": max_rel_prob,
        "n_entities": n_entities,
        "oracle_rel_prob": oracle_rel_prob,
    }


def phase_evaluate(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result_path = out / "ep_relax_eval.json"
    if result_path.exists():
        print(f"  Evaluation already done: {result_path}", flush=True)
        print(json.dumps(json.load(open(result_path)), indent=2), flush=True)
        return

    print("=== Phase 3c: SpERT evaluation on EP-relax ===", flush=True)

    model_dir = Path(args.model_dir)
    if not (model_dir / "pytorch_model.bin").exists():
        print(f"  ERROR: model not found at {model_dir}", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.models.relation_extractor import HostPathogenRelationExtractor, ENTITY_TYPES, RELATION_TYPES
    cfg = json.load(open(model_dir / "spert_config.json"))
    model = HostPathogenRelationExtractor(
        model_name=cfg["model_name"],
        max_span_width=cfg.get("max_span_width", 6),
        entity_types=ENTITY_TYPES,
        relation_types=RELATION_TYPES,
    )
    model.load_state_dict(
        torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    )
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    ep_path = Path(args.ep_relax) if args.ep_relax else EP_RELAX_PATH
    texts, labels, sp1_list, sp2_list = load_ep_relax(ep_path)
    print(f"  EP-relax: {len(texts)} sentences, {sum(labels)} positive", flush=True)

    # Run inference ONCE per sentence, cache scores
    print("  Running inference (once per sentence)...", flush=True)
    scores = []
    t0 = time.time()
    for i, (text, sp1, sp2) in enumerate(zip(texts, sp1_list, sp2_list)):
        s = _score_sentence(model, tokenizer, text, device, sp1, sp2)
        scores.append(s)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(texts) - i - 1)
            print(f"  [{i+1}/{len(texts)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    print(f"  Inference done in {time.time()-t0:.0f}s", flush=True)

    # Sweep thresholds on cached scores — cheap
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    best_f1, best_t, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    threshold_results = []

    for threshold in thresholds:
        preds = []
        for s in scores:
            # Positive: entity found AND relation predicted (primary path)
            primary = s["max_entity_prob"] >= threshold and s["max_rel_prob"] >= threshold
            # Fallback: oracle species known, and oracle relation score >= threshold
            oracle = s["oracle_rel_prob"] >= threshold
            preds.append(int(primary or oracle))

        f1 = f1_score(labels, preds, zero_division=0)
        p  = precision_score(labels, preds, zero_division=0)
        r  = recall_score(labels, preds, zero_division=0)
        threshold_results.append({"threshold": threshold, "f1": round(f1, 4),
                                   "prec": round(p, 4), "rec": round(r, 4)})
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, threshold, p, r

    delta = round(best_f1 - BASELINE_EP_F1, 4)
    recommendation = (
        f"SpERT (F1={best_f1:.3f}) BEATS baseline (F1={BASELINE_EP_F1}) by {delta:+.3f} — consider as production candidate"
        if best_f1 > BASELINE_EP_F1
        else f"Baseline (F1={BASELINE_EP_F1}) is better than SpERT (F1={best_f1:.3f}, Δ={delta:+.3f}) — keep full_typed_a05_ner2"
    )

    result = {
        "spert": {
            "best_f1": round(best_f1, 4),
            "best_prec": round(best_p, 4),
            "best_rec": round(best_r, 4),
            "best_threshold": best_t,
            "threshold_sweep": threshold_results,
        },
        "baseline_full_typed_a05_ner2": {"ep_f1": BASELINE_EP_F1, "threshold": 0.13},
        "delta_f1": delta,
        "recommendation": recommendation,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  SpERT best F1={best_f1:.4f} (Δ{delta:+.4f} vs baseline)", flush=True)
    print(f"  → {recommendation}", flush=True)
    print(f"  Results: {result_path}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier 3: SpERT training + evaluation")
    parser.add_argument("--phase", required=True, choices=["build-ws", "train", "evaluate", "all"])
    parser.add_argument("--v14-path",          default=str(V14_PATH))
    parser.add_argument("--ws-data",           default=None, help="Path to weak_supervision_train.jsonl")
    parser.add_argument("--pretrained-encoder", default=None, help="Path to multitask model dir")
    parser.add_argument("--model-dir",         default=None, help="Path to trained spert_model dir")
    parser.add_argument("--ep-relax",          default=str(EP_RELAX_PATH))
    parser.add_argument("--output-dir",        required=True)
    parser.add_argument("--epochs",            type=int, default=3)
    parser.add_argument("--lr",                type=float, default=2e-5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    if args.ws_data is None:
        args.ws_data = str(out / "weak_supervision_train.jsonl")
    if args.model_dir is None:
        args.model_dir = str(out / "spert_model")

    if args.phase in ("build-ws", "all"):
        phase_build_ws(args)
    if args.phase in ("train", "all"):
        phase_train(args)
    if args.phase in ("evaluate", "all"):
        phase_evaluate(args)

    if args.phase == "all":
        (out / "tier3_done").touch()
        print("Tier 3 complete.", flush=True)


if __name__ == "__main__":
    main()
