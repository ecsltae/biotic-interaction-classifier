#!/usr/bin/env python3
"""
Run champion model (full_typed_a05_ner2) on pending curation queue items
to add reasoning hints and auto-approve high-confidence predictions.

Rules:
  - prob > 0.90  → auto-approve as POSITIVE  (confidence=0.85)
  - prob < 0.05  → auto-approve as NEGATIVE  (confidence=0.85)
  - otherwise    → UPDATE reasoning hint only, keep status='pending'

Targets: sibils_diverse, globi_sibils (NOT v15_test_batch*)
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "classifier" if (ROOT / "classifier").exists() else ROOT.parent / "classifier"
sys.path.insert(0, str(CLASSIFIER / "experiments" / "multitask"))

from model import MultiTaskBiomedBERT  # noqa: E402

DB_PATH = CLASSIFIER / "data" / "training" / "curation.db"
CHAMPION = CLASSIFIER / "models" / "multitask" / "full_typed_a05_ner2"
BATCH_SIZE = 64
AUTO_POS_THRESH = 0.90
AUTO_NEG_THRESH = 0.05
AUTO_CONFIDENCE = 0.85


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_model(device):
    cfg = json.load(open(CHAMPION / "multitask_config.json"))
    model = MultiTaskBiomedBERT.load(str(CHAMPION), device=str(device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["encoder_name"])
    return model, tok


def predict_batch(model, tok, texts: list, device) -> np.ndarray:
    enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
    return torch.softmax(out["cls_logits"], -1)[:, 1].cpu().numpy()


def fetch_pending(source: str, db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, text FROM curation_queue WHERE status='pending' AND source=?",
            (source,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_hint(item_id: int, hint: str, db_path: Path):
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE curation_queue SET reasoning=?, updated_at=? WHERE id=?",
            (hint, _now(), item_id),
        )


def approve(item_id: int, label: int, confidence: float, reasoning: str, db_path: Path):
    status = "approved" if confidence >= 0.7 else "uncertain"
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE curation_queue SET label=?, confidence=?, reasoning=?, author=?, status=?, updated_at=? WHERE id=?",
            (label, confidence, reasoning, "champion_model", status, _now(), item_id),
        )


def run_source(source: str, model, tok, device, db_path: Path, dry_run: bool):
    items = fetch_pending(source, db_path)
    if not items:
        print(f"  No pending items for source '{source}'")
        return

    print(f"\n  Source '{source}': {len(items)} pending items")
    texts = [it["text"] for it in items]
    ids = [it["id"] for it in items]

    all_probs: list[float] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        probs = predict_batch(model, tok, batch, device)
        all_probs.extend(probs.tolist())
        print(f"    Batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1} done", flush=True)

    auto_pos = auto_neg = hints = 0
    for item_id, prob in zip(ids, all_probs):
        prob_str = f"{prob:.3f}"
        if prob > AUTO_POS_THRESH:
            reason = f"Champion model: POSITIVE (prob={prob_str})"
            if not dry_run:
                approve(item_id, 1, AUTO_CONFIDENCE, reason, db_path)
            auto_pos += 1
        elif prob < AUTO_NEG_THRESH:
            reason = f"Champion model: NEGATIVE (prob={prob_str})"
            if not dry_run:
                approve(item_id, 0, AUTO_CONFIDENCE, reason, db_path)
            auto_neg += 1
        else:
            hint = f"Champion model hint: {'POSITIVE' if prob >= 0.13 else 'NEGATIVE'} (prob={prob_str})"
            if not dry_run:
                update_hint(item_id, hint, db_path)
            hints += 1

    tag = " [DRY RUN]" if dry_run else ""
    print(f"    {tag} Auto-approved POSITIVE: {auto_pos}")
    print(f"    {tag} Auto-approved NEGATIVE: {auto_neg}")
    print(f"    {tag} Hints added (still pending): {hints}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+",
        default=["sibils_diverse", "globi_sibils"],
        help="Sources to prelabel (never touch v15_test_batch*)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    args = parser.parse_args()

    # Safety: never prelabel test batches
    safe_sources = [s for s in args.sources if not s.startswith("v15_test_batch")]
    if len(safe_sources) < len(args.sources):
        blocked = set(args.sources) - set(safe_sources)
        print(f"WARNING: Skipping protected sources: {blocked}")
    if not safe_sources:
        print("No sources to process. Exiting.")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading champion model from {CHAMPION} ...")
    model, tok = load_model(device)
    print("Model loaded.")

    for source in safe_sources:
        run_source(source, model, tok, device, DB_PATH, dry_run=args.dry_run)

    del model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
