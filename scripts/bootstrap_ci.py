#!/usr/bin/env python3
"""
Bootstrap confidence intervals for classifier F1/P/R on EP-relax and eval_sets.
Also runs McNemar's test between every pair of models.

Usage:
    python classifier/scripts/bootstrap_ci.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "classifier/experiments/multitask"))
from model import MultiTaskBiomedBERT  # noqa: E402

DEVICE = torch.device("cpu")  # run on CPU — GPU reserved for v15 training
N_BOOT = 10_000
ALPHA  = 0.05
SEED   = 42

EP_RELAX  = ROOT / "classifier/data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
EVAL_SETS = ROOT / "classifier/data/evaluation/eval_sets_qwen_validated.csv"

MODELS = {
    "multitask_best": {
        "type": "multitask",
        "path": ROOT / "classifier/models/multitask/full_typed_a05_ner2",
        "threshold": 0.13,
    },
    "distilled_v2": {
        "type": "distilled",
        "path": ROOT / "classifier/models/distilled_BiomedBERT_v2",
        "threshold": 0.25,
    },
    "BiomedBERT_v7": {
        "type": "distilled",
        "path": ROOT / "classifier/models/transformer_BiomedBERT_cv_regularized",
        "threshold": 0.50,
    },
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_ep_relax():
    df = pd.read_csv(EP_RELAX, sep="\t")
    texts  = df["sentence"].astype(str).tolist()
    labels = df["evaluation_pair_interacting"].astype(int).tolist()
    return texts, labels


def load_eval_sets():
    df = pd.read_csv(EVAL_SETS)
    # gold_label is authoritative
    texts  = df["text"].astype(str).tolist()
    labels = df["gold_label"].astype(int).tolist()
    return texts, labels


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_multitask(model_path, texts):
    cfg = json.load(open(model_path / "multitask_config.json"))
    model = MultiTaskBiomedBERT.load(str(model_path), device=str(DEVICE))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["encoder_name"])
    probs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        enc = tok(batch, truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"],
                        enc.get("token_type_ids"))
        probs.extend(torch.softmax(out["cls_logits"], -1)[:, 1].cpu().tolist())
    return np.array(probs)


def predict_seq_cls(model_path, texts):
    tok   = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True).to(DEVICE).eval()
    probs = []
    for i in range(0, len(texts), 32):
        batch = [t.lower() for t in texts[i:i+32]]
        enc = tok(batch, truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, -1)[:, 1].cpu().tolist()
        probs.extend(p)
    return np.array(probs)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred_bin, n=N_BOOT, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    y_true = np.array(y_true)
    y_pred_bin = np.array(y_pred_bin)
    idx = np.arange(len(y_true))
    f1s, ps, rs = [], [], []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        # skip degenerate resamples with no positives or all positives
        if y_true[s].sum() == 0 or y_true[s].sum() == len(s):
            continue
        f1s.append(f1_score(y_true[s], y_pred_bin[s], zero_division=0))
        ps.append(precision_score(y_true[s], y_pred_bin[s], zero_division=0))
        rs.append(recall_score(y_true[s], y_pred_bin[s], zero_division=0))
    lo, hi = ALPHA / 2 * 100, (1 - ALPHA / 2) * 100
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_ci":   [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
        "p_mean":  float(np.mean(ps)),
        "p_ci":    [float(np.percentile(ps, lo)),  float(np.percentile(ps, hi))],
        "r_mean":  float(np.mean(rs)),
        "r_ci":    [float(np.percentile(rs, lo)),  float(np.percentile(rs, hi))],
        "n_boot":  len(f1s),
    }


# ── McNemar ───────────────────────────────────────────────────────────────────

def mcnemar(y_true, pred_a, pred_b):
    """Two-tailed McNemar's test with continuity correction."""
    y_true = np.array(y_true)
    a = np.array(pred_a)
    b = np.array(pred_b)
    n01 = int(((a == y_true) & (b != y_true)).sum())  # A right, B wrong
    n10 = int(((a != y_true) & (b == y_true)).sum())  # A wrong, B right
    n_disagree = n01 + n10
    if n_disagree == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p_value": 1.0, "note": "identical predictions"}
    stat = (abs(n01 - n10) - 1) ** 2 / n_disagree
    p    = 1 - chi2.cdf(stat, df=1)
    return {
        "n01": n01, "n10": n10,
        "chi2": float(stat), "p_value": float(p),
        "note": f"A correct & B wrong={n01}; B correct & A wrong={n10}; total disagree={n_disagree}",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_on_dataset(name, texts, labels):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  ({len(texts)} sentences, {sum(labels)} pos)")
    print(f"{'='*60}")

    all_probs = {}
    all_preds = {}
    point_metrics = {}

    for mname, cfg in MODELS.items():
        print(f"\n  Loading {mname} ...", flush=True)
        if cfg["type"] == "multitask":
            probs = predict_multitask(cfg["path"], texts)
        else:
            probs = predict_seq_cls(cfg["path"], texts)

        t = cfg["threshold"]
        preds = (probs >= t).astype(int)
        all_probs[mname] = probs
        all_preds[mname] = preds

        f1 = f1_score(labels, preds, zero_division=0)
        p  = precision_score(labels, preds, zero_division=0)
        r  = recall_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, probs)
        point_metrics[mname] = {"f1": f1, "precision": p, "recall": r, "auc": auc, "threshold": t}
        print(f"    t={t:.2f}  F1={f1:.4f}  P={p:.4f}  R={r:.4f}  AUC={auc:.4f}")

    # Bootstrap CIs
    print("\n  Bootstrap CIs (n=10,000):", flush=True)
    ci_results = {}
    rng = np.random.default_rng(SEED)
    for mname, preds in all_preds.items():
        ci = bootstrap_ci(labels, preds, rng=rng)
        ci_results[mname] = ci
        f1_lo, f1_hi = ci["f1_ci"]
        print(f"    {mname:25s}  F1={ci['f1_mean']:.3f}  95% CI [{f1_lo:.3f}, {f1_hi:.3f}]"
              f"  P=[{ci['p_ci'][0]:.3f},{ci['p_ci'][1]:.3f}]"
              f"  R=[{ci['r_ci'][0]:.3f},{ci['r_ci'][1]:.3f}]")

    # McNemar pairwise
    print("\n  McNemar pairwise tests:", flush=True)
    names = list(all_preds.keys())
    mcnemar_results = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            res = mcnemar(labels, all_preds[a], all_preds[b])
            key = f"{a}_vs_{b}"
            mcnemar_results[key] = res
            sig = "**SIGNIFICANT**" if res["p_value"] < 0.05 else "not significant"
            print(f"    {a} vs {b}: χ²={res['chi2']:.3f}  p={res['p_value']:.4f}  {sig}")
            print(f"      ({res['note']})")

    return {
        "point_metrics": point_metrics,
        "bootstrap_ci":  ci_results,
        "mcnemar":       mcnemar_results,
    }


def main():
    print(f"Device: {DEVICE}", flush=True)

    ep_texts, ep_labels   = load_ep_relax()
    ev_texts, ev_labels   = load_eval_sets()

    results = {}
    results["ep_relax"]   = run_on_dataset("EP-relax (99 sentences)", ep_texts, ep_labels)
    results["eval_sets"]  = run_on_dataset("eval_sets_qwen (599 sentences)", ev_texts, ev_labels)

    out = ROOT / "classifier/results/bootstrap_ci_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY — F1 with 95% bootstrap CI")
    print("="*70)
    for dname, dres in results.items():
        print(f"\n{dname}:")
        for mname in MODELS:
            pm = dres["point_metrics"][mname]
            ci = dres["bootstrap_ci"][mname]
            print(f"  {mname:25s}  F1={pm['f1']:.3f}  [{ci['f1_ci'][0]:.3f}, {ci['f1_ci'][1]:.3f}]")


if __name__ == "__main__":
    main()
