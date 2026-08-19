#!/usr/bin/env python3
"""
Statistical significance tests for new multi-task experiments vs key baselines.

Bootstrap 95% CI (n=10,000) + McNemar's test for:
  - mt_distill_warm_ner0  (new best)
  - full_typed_a05_ner2   (champion, from probs_main.npz)
  - full_typed_a05_ner2_warmstart  (existing warmstart)
  - BiomedBERT_v7_singletask  (target, from probs_main.npz)
  - multitask_hardce  (hard-CE ablation, from probs_main.npz)

Usage:
    python classifier/scripts/stat_test_new_models.py --gpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "classifier/experiments/multitask"))
from model import MultiTaskBiomedBERT  # noqa: E402

TEST_SET  = ROOT / "classifier/data/evaluation/biotic_interaction_test_set.csv"
PROBS_NPZ = ROOT / "classifier/results/new_testset/probs_main.npz"
OUT_DIR   = ROOT / "classifier/results/new_testset"

N_BOOT = 10_000
SEED   = 42
ALPHA  = 0.05


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


def get_threshold(model_path: Path) -> float:
    results_dir = ROOT / "classifier/results/multitask" / model_path.name
    for candidate in [model_path / "train_summary.json",
                      results_dir / "train_summary.json"]:
        if candidate.exists():
            return json.load(open(candidate)).get("best_threshold", 0.5)
    return 0.5


def bootstrap_ci(labels, preds, n=N_BOOT, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    labels = np.array(labels)
    preds  = np.array(preds)
    idx = np.arange(len(labels))
    f1s, ps, rs = [], [], []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if labels[s].sum() == 0 or labels[s].sum() == len(s):
            continue
        f1s.append(f1_score(labels[s], preds[s], zero_division=0))
        ps.append(precision_score(labels[s], preds[s], zero_division=0))
        rs.append(recall_score(labels[s], preds[s], zero_division=0))
    lo, hi = ALPHA / 2 * 100, (1 - ALPHA / 2) * 100
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_ci":   [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
        "p_ci":    [float(np.percentile(ps,  lo)), float(np.percentile(ps,  hi))],
        "r_ci":    [float(np.percentile(rs,  lo)), float(np.percentile(rs,  hi))],
        "n_boot":  len(f1s),
    }


def mcnemar(labels, preds_a, preds_b):
    """Two-tailed McNemar with continuity correction."""
    y = np.array(labels)
    a, b = np.array(preds_a), np.array(preds_b)
    n01 = int(((a == y) & (b != y)).sum())
    n10 = int(((a != y) & (b == y)).sum())
    n_d = n01 + n10
    if n_d == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / n_d
    p = 1 - chi2.cdf(stat, df=1)
    return {"n01": n01, "n10": n10, "n_disagree": n_d,
            "chi2": round(float(stat), 3), "p_value": round(float(p), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = pd.read_csv(TEST_SET)
    texts  = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    print(f"Test set: {len(texts)} sentences, {sum(labels)} positives\n")

    npz = np.load(PROBS_NPZ)
    all_probs = {}
    all_thresholds = {}

    # Load baselines from probs_main.npz
    npz_models = {
        "champion":       ("multitask_champion",      0.090),
        "hardce":         ("multitask_hardce",         0.510),
        "BiomedBERT_v7":  ("BiomedBERT_v7_singletask", 0.500),
    }
    for name, (key, t) in npz_models.items():
        if key in npz:
            all_probs[name] = npz[key]
            all_thresholds[name] = t
            print(f"  Loaded {name} from npz (τ={t:.3f})")
        else:
            print(f"  SKIP {name} — key '{key}' not in npz")

    # Infer new multitask models
    mt_models = {
        "mt_distill_warm_ner0":         ROOT / "classifier/models/multitask/mt_distill_warm_ner0",
        "mt_distill_warm_ner2":         ROOT / "classifier/models/multitask/mt_distill_warm_ner2",
        "warmstart_existing":           ROOT / "classifier/models/multitask/full_typed_a05_ner2_warmstart",
    }
    for name, path in mt_models.items():
        if not path.exists():
            print(f"  SKIP {name} — model not found")
            continue
        t = get_threshold(path)
        print(f"  Inferring {name} (τ={t:.3f}) ...", flush=True)
        all_probs[name] = predict_multitask(path, texts, device)
        all_thresholds[name] = t

    # Compute binary predictions + point metrics
    all_preds = {}
    point = {}
    print("\nPoint metrics:")
    for name, probs in all_probs.items():
        t = all_thresholds[name]
        preds = (probs >= t).astype(int)
        all_preds[name] = preds
        f1 = f1_score(labels, preds, zero_division=0)
        p  = precision_score(labels, preds, zero_division=0)
        r  = recall_score(labels, preds, zero_division=0)
        point[name] = {"f1": round(float(f1), 4), "precision": round(float(p), 4),
                       "recall": round(float(r), 4), "threshold": t}
        print(f"  {name:<30}  τ={t:.3f}  F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

    # Bootstrap CIs
    print(f"\nBootstrap 95% CI (n={N_BOOT:,}) ...")
    rng = np.random.default_rng(SEED)
    ci_results = {}
    for name, preds in all_preds.items():
        ci = bootstrap_ci(labels, preds, rng=rng)
        ci_results[name] = ci
        lo, hi = ci["f1_ci"]
        print(f"  {name:<30}  F1={ci['f1_mean']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # McNemar pairwise — focus on meaningful comparisons
    print("\nMcNemar pairwise (new best vs each baseline):")
    mcnemar_results = {}
    comparisons = [
        # (model_a, model_b, description)
        ("mt_distill_warm_ner0", "champion",          "new_best vs champion"),
        ("mt_distill_warm_ner0", "warmstart_existing","new_best vs warmstart"),
        ("mt_distill_warm_ner0", "BiomedBERT_v7",     "new_best vs BiomedBERT_v7"),
        ("mt_distill_warm_ner0", "hardce",             "new_best vs hardce"),
        ("mt_distill_warm_ner2", "champion",           "ner2_warm vs champion"),
        ("mt_distill_warm_ner2", "mt_distill_warm_ner0", "ner2_warm vs ner0_warm"),
    ]
    for a, b, desc in comparisons:
        if a not in all_preds or b not in all_preds:
            print(f"  SKIP {desc}")
            continue
        res = mcnemar(labels, all_preds[a], all_preds[b])
        mcnemar_results[f"{a}_vs_{b}"] = res
        sig = "p<0.05 SIGNIFICANT" if res["p_value"] < 0.05 else "not significant"
        f1_a = point[a]["f1"]
        f1_b = point[b]["f1"] if b in point else "?"
        print(f"  {desc:<40}  χ²={res['chi2']:.3f}  p={res['p_value']:.4f}  {sig}")
        print(f"    ({a} F1={f1_a:.4f} vs {b} F1={f1_b:.4f};"
              f" A-right-B-wrong={res['n01']}, B-right-A-wrong={res['n10']},"
              f" disagree={res['n_disagree']})")

    out = {
        "dataset": f"biotic_interaction_test_set ({len(texts)} sentences, {sum(labels)} positives)",
        "point_metrics":  point,
        "bootstrap_ci":   ci_results,
        "mcnemar":        mcnemar_results,
    }
    out_file = OUT_DIR / "stat_test_new_models.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — F1 with 95% bootstrap CI (500-sentence test set)")
    print("=" * 70)
    order = ["mt_distill_warm_ner0", "mt_distill_warm_ner2", "warmstart_existing",
             "champion", "BiomedBERT_v7", "hardce"]
    for name in order:
        if name not in point:
            continue
        pm = point[name]
        ci = ci_results[name]
        print(f"  {name:<30}  F1={pm['f1']:.4f}  [{ci['f1_ci'][0]:.4f}, {ci['f1_ci'][1]:.4f}]"
              f"  P={pm['precision']:.4f}  R={pm['recall']:.4f}")


if __name__ == "__main__":
    main()
