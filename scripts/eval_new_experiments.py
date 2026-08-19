#!/usr/bin/env python3
"""
Evaluate new multi-task training experiments on the 500-sentence test set.

Reads threshold from each model's train_summary.json (best_threshold field).
Outputs results to classifier/results/new_testset/new_experiments.json and
prints a sorted comparison table.

Usage:
    python classifier/scripts/eval_new_experiments.py --gpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "classifier/experiments/multitask"))
from model import MultiTaskBiomedBERT  # noqa: E402
from transformers import AutoTokenizer

TEST_SET = ROOT / "classifier/data/evaluation/biotic_interaction_test_set.csv"
OUT_DIR  = ROOT / "classifier/results/new_testset"

NEW_EXPERIMENTS = [
    "mt_v7_softlabels",
    "mt_v7_softlabels_warm",
    "mt_distill_warm_ner1",
    "mt_distill_warm_ner0",
    "mt_v7_distill_mix",
    "mt_v7_distill_mix_warm",
    "mt_v14teacher",       # Option D (may not exist yet)
]

BASELINES = {
    "full_typed_a05_ner2":         {"threshold": 0.090, "f1": 0.807, "p": 0.741, "r": 0.886},
    "full_typed_a05_ner2_warmstart": {"threshold": 0.52, "f1": 0.835, "p": None,  "r": None},
    "BiomedBERT_v7_singletask":    {"threshold": 0.50,  "f1": 0.865, "p": None,  "r": None},
}


def predict_multitask(model_path: Path, texts: list, device) -> np.ndarray:
    cfg = json.load(open(model_path / "multitask_config.json"))
    model = MultiTaskBiomedBERT.load(str(model_path), device=str(device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["encoder_name"])
    probs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        enc = tok(batch, truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"],
                        enc.get("token_type_ids"))
        probs.extend(torch.softmax(out["cls_logits"], -1)[:, 1].cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def compute_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    labels_arr = np.array(labels)
    f1 = f1_score(labels_arr, preds, zero_division=0)
    p  = precision_score(labels_arr, preds, zero_division=0)
    r  = recall_score(labels_arr, preds, zero_division=0)
    return {"f1": round(float(f1), 4), "precision": round(float(p), 4),
            "recall": round(float(r), 4), "threshold": threshold,
            "n_pred_pos": int(preds.sum())}


def get_threshold(model_path: Path) -> float:
    # train_summary.json lives in results dir, mirroring the model dir name
    results_dir = ROOT / "classifier/results/multitask" / model_path.name
    for candidate in [model_path / "train_summary.json",
                      results_dir / "train_summary.json"]:
        if candidate.exists():
            return json.load(open(candidate)).get("best_threshold", 0.5)
    return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = pd.read_csv(TEST_SET)
    texts  = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    print(f"Test set: {len(texts)} sentences, {sum(labels)} positives\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name in NEW_EXPERIMENTS:
        model_path = ROOT / "classifier/models/multitask" / name
        if not model_path.exists():
            print(f"  SKIP {name} — not found")
            continue
        threshold = get_threshold(model_path)
        print(f"  Loading {name} (τ={threshold:.3f}) ...", flush=True)
        try:
            probs = predict_multitask(model_path, texts, device)
            results[name] = compute_metrics(labels, probs, threshold)
            m = results[name]
            print(f"  → F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}"
                  f"  pos={m['n_pred_pos']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    out_file = OUT_DIR / "new_experiments.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_file}")

    # Print sorted comparison table
    print("\n" + "=" * 72)
    print(f"{'Model':<38} {'Data':>6} {'Warm':>4} {'NER':>3} {'F1':>6} {'P':>6} {'R':>6}")
    print("-" * 72)

    metadata = {
        "mt_v7_softlabels":          ("v7",  "cold", 2),
        "mt_v7_softlabels_warm":     ("v7",  "warm", 2),
        "mt_distill_warm_ner1":      ("44k", "warm", 1),
        "mt_distill_warm_ner0":      ("44k", "warm", 0),
        "mt_v7_distill_mix":         ("mix", "cold", 2),
        "mt_v7_distill_mix_warm":    ("mix", "warm", 1),
        "mt_v14teacher":             ("44k", "cold", 2),
    }

    all_rows = []
    for name, m in results.items():
        if "error" in m:
            continue
        meta = metadata.get(name, ("?", "?", "?"))
        all_rows.append((m["f1"], name, meta, m))

    # Sort by F1 descending
    for f1, name, meta, m in sorted(all_rows, reverse=True):
        data, warm, ner = meta
        print(f"  {name:<36} {data:>6} {warm:>4} {str(ner):>3} {m['f1']:>6.4f} "
              f"{m['precision']:>6.4f} {m['recall']:>6.4f}  ← NEW")

    print("-" * 72)
    print(f"  {'champion (full_typed_a05_ner2)':<36} {'44k':>6} {'cold':>4} {'2':>3} "
          f"{0.807:>6.4f} {0.741:>6.4f} {0.886:>6.4f}  (baseline)")
    print(f"  {'warmstart (full_typed_a05_ner2_warm)':<36} {'44k':>6} {'warm':>4} {'2':>3} "
          f"{0.835:>6.4f} {'?':>6} {'?':>6}  (baseline)")
    print(f"  {'BiomedBERT_v7_singletask':<36} {'v7':>6} {'cold':>4} {'-':>3} "
          f"{0.865:>6.4f} {'?':>6} {'?':>6}  (target)")


if __name__ == "__main__":
    main()
