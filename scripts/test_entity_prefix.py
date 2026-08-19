#!/usr/bin/env python3
"""
Test entity-prepended inference on EP-relax.
Formats input as: "[SP1: {sp1}] [SP2: {sp2}] {sentence}" and re-evaluates.
If this improves FP rate without retraining, it motivates a v16 training run.
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

from model import MultiTaskBiomedBERT  # noqa: E402

CHAMPION = CLASSIFIER / "models" / "multitask" / "full_typed_a05_ner2"
EP_RELAX = CLASSIFIER / "data" / "evaluation" / "globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
THRESHOLD = 0.13
OUT = CLASSIFIER / "results" / "error_analysis" / "ep_relax"


def format_entity_prefix(sp1: str, sp2: str, text: str) -> str:
    return f"[SP1: {sp1}] [SP2: {sp2}] {text}"


def format_qa(sp1: str, sp2: str, text: str) -> str:
    return f"Do {sp1} and {sp2} interact? {text}"


def format_natural(sp1: str, sp2: str, text: str) -> str:
    return f"{sp1} interacts with {sp2}. {text}"


def predict(model, tok, texts: list, device) -> np.ndarray:
    probs = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        enc = tok(batch, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
        probs.extend(torch.softmax(out["cls_logits"], -1)[:, 1].cpu().tolist())
    return np.array(probs)


def score(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return {"P": round(float(prec), 4), "R": round(float(rec), 4),
            "F1": round(float(f1), 4), "TP": int(tp), "FP": int(fp), "FN": int(fn)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(EP_RELAX, sep="\t")
    labels = df["evaluation_pair_interacting"].values
    sentences = df["sentence"].tolist()
    sp1_list = df["species1_term"].tolist()
    sp2_list = df["species2_term"].tolist()

    cfg = json.load(open(CHAMPION / "multitask_config.json"))
    model = MultiTaskBiomedBERT.load(str(CHAMPION), device=str(device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["encoder_name"])

    # Baseline: plain text
    print("\n=== BASELINE (plain text) ===")
    probs_base = predict(model, tok, sentences, device)
    res_base = score(probs_base, labels, THRESHOLD)
    print(f"  {res_base}")

    def eval_format(name: str, fmt_texts: list, base_preds: np.ndarray):
        probs = predict(model, tok, fmt_texts, device)
        res = score(probs, labels, THRESHOLD)
        print(f"\n=== {name} ===")
        print(f"  @threshold={THRESHOLD}: {res}")
        # Best threshold sweep
        best_f1, best_thr = 0, THRESHOLD
        for thr in np.arange(0.05, 0.95, 0.01):
            r = score(probs, labels, thr)
            if r["F1"] > best_f1:
                best_f1, best_thr = r["F1"], thr
        res_best = score(probs, labels, best_thr)
        print(f"  Best thr={best_thr:.2f}: {res_best}")
        preds = (probs >= THRESHOLD).astype(int)
        fixed_fp = ((base_preds == 1) & (preds == 0) & (labels == 0)).sum()
        new_fp   = ((base_preds == 0) & (preds == 1) & (labels == 0)).sum()
        fixed_fn = ((base_preds == 0) & (preds == 1) & (labels == 1)).sum()
        new_fn   = ((base_preds == 1) & (preds == 0) & (labels == 1)).sum()
        print(f"  FPs fixed={fixed_fp} new={new_fp}  FNs fixed={fixed_fn} new={new_fn}")
        return probs, res, res_best

    preds_base = (probs_base >= THRESHOLD).astype(int)

    # Format 1: bracket prefix
    ep_texts = [format_entity_prefix(s1, s2, t) for s1, s2, t in zip(sp1_list, sp2_list, sentences)]
    probs_ep, res_ep, res_ep_best = eval_format("BRACKET PREFIX ([SP1: X] [SP2: Y] text)", ep_texts, preds_base)

    # Format 2: QA question
    qa_texts = [format_qa(s1, s2, t) for s1, s2, t in zip(sp1_list, sp2_list, sentences)]
    probs_qa, res_qa, res_qa_best = eval_format("QA FORMAT (Do X and Y interact? text)", qa_texts, preds_base)

    # Format 3: natural sentence prepend
    nat_texts = [format_natural(s1, s2, t) for s1, s2, t in zip(sp1_list, sp2_list, sentences)]
    probs_nat, res_nat, res_nat_best = eval_format("NATURAL PREFIX (X interacts with Y. text)", nat_texts, preds_base)

    results = {
        "baseline": res_base,
        "bracket_prefix": {"at_013": res_ep, "best": {"threshold": 0.0, **res_ep_best}},
        "qa_format": {"at_013": res_qa, "best": {"threshold": 0.0, **res_qa_best}},
        "natural_prefix": {"at_013": res_nat, "best": {"threshold": 0.0, **res_nat_best}},
    }
    out_path = OUT / "entity_prefix_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
