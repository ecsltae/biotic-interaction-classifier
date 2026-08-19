#!/usr/bin/env python3
"""
EP-relax error analysis for champion model (full_typed_a05_ner2).
Identifies FP/FN sentences and groups them by interaction type and error pattern.
Results written to classifier/results/error_analysis/ep_relax/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "classifier" if (ROOT / "classifier").exists() else ROOT.parent / "classifier"
sys.path.insert(0, str(CLASSIFIER / "experiments" / "multitask"))

from model import MultiTaskBiomedBERT

CHAMPION = CLASSIFIER / "models" / "multitask" / "full_typed_a05_ner2"
EP_RELAX = CLASSIFIER / "data" / "evaluation" / "globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
THRESHOLD = 0.13  # optimal from eval JSON
OUT = CLASSIFIER / "results" / "error_analysis" / "ep_relax"
OUT.mkdir(parents=True, exist_ok=True)


def predict_multitask(model_path: Path, texts: list, device) -> np.ndarray:
    cfg = json.load(open(model_path / "multitask_config.json"))
    model = MultiTaskBiomedBERT.load(str(model_path), device=str(device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["encoder_name"])
    probs = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        enc = tok(
            batch,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
        probs.extend(torch.softmax(out["cls_logits"], -1)[:, 1].cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = pd.read_csv(EP_RELAX, sep="\t")
    labels = df["evaluation_pair_interacting"].values
    sentences = df["sentence"].tolist()
    print(f"EP-relax: {len(df)} sentences, {labels.sum()} positive", flush=True)

    print("Running champion model ...", flush=True)
    probs = predict_multitask(CHAMPION, sentences, device)

    preds = (probs >= THRESHOLD).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    print(f"  Threshold={THRESHOLD}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn} TN={tn})")

    df["prob"] = probs
    df["pred"] = preds
    df["label"] = labels
    df["error_type"] = "OK"
    df.loc[(df["pred"] == 1) & (df["label"] == 0), "error_type"] = "FP"
    df.loc[(df["pred"] == 0) & (df["label"] == 1), "error_type"] = "FN"

    # Save all predictions
    df[["sentence", "interaction_term", "species1_term", "species2_term",
        "label", "pred", "prob", "error_type", "comment"]].to_csv(
        OUT / "all_predictions.csv", index=False
    )

    fps = df[df["error_type"] == "FP"].sort_values("prob", ascending=False)
    fns = df[df["error_type"] == "FN"].sort_values("prob")

    print(f"\n=== FALSE POSITIVES ({len(fps)}) — negative sentences predicted positive ===")
    for _, row in fps.iterrows():
        print(f"  [{row['prob']:.3f}] {row['species1_term']} × {row['species2_term']} "
              f"({row['interaction_term']})")
        print(f"    SENTENCE: {row['sentence'][:150]}")
        if str(row.get("comment", "")).strip() not in ("", "nan"):
            print(f"    NOTE: {row['comment']}")
        print()

    print(f"=== FALSE NEGATIVES ({len(fns)}) — positive sentences predicted negative ===")
    for _, row in fns.iterrows():
        print(f"  [{row['prob']:.3f}] {row['species1_term']} × {row['species2_term']} "
              f"({row['interaction_term']})")
        print(f"    SENTENCE: {row['sentence'][:150]}")
        if str(row.get("comment", "")).strip() not in ("", "nan"):
            print(f"    NOTE: {row['comment']}")
        print()

    # Interaction type breakdown
    print("\n=== ERRORS BY INTERACTION TYPE ===")
    err_df = df[df["error_type"].isin(["FP", "FN"])]
    breakdown = (
        err_df.groupby(["interaction_term", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    print(breakdown.to_string(index=False))

    # Save error analysis
    results = {
        "threshold": THRESHOLD,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_positives": fps[["sentence", "species1_term", "species2_term",
                                "interaction_term", "prob", "comment"]].to_dict("records"),
        "false_negatives": fns[["sentence", "species1_term", "species2_term",
                                "interaction_term", "prob", "comment"]].to_dict("records"),
        "interaction_breakdown": breakdown.to_dict("records"),
    }
    with open(OUT / "error_analysis.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
