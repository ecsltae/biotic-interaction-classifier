#!/usr/bin/env python3
"""Re-run paperA's query ablation on the full clean benchmark, pair-level labels.

The claim under test is "the query, not the encoder": if the verifier's advantage came from a
stronger encoder rather than from being told which triple to check, degrading the query at
inference would not hurt it much. Weights and threshold are unchanged in every row of the table;
only the text of segment A moves.

Arms
  retrieved   the triple the pipeline actually issued            (deployed behaviour)
  generic     relation replaced by "interacts with", pair kept   (is the relation carrying it?)
  shuffled    relation permuted across items, pair kept          (is it the *right* relation?)
  rel_only    relation kept, both taxa blanked                   (is the pair carrying it?)
  empty       segment A empty, passage only                      (sentence-level baseline)

The empty arm is the deployed filter's formulation: passage in, verdict out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xenc_format  # noqa: E402
from eval_unified import model_format  # noqa: E402

OUT = REPO / "results/paperA_rebuild_2026-09-28"
RNG_SEED = 0          # fixed: the shuffle must be reproducible, and Date/random are avoidable


def clean_benchmark() -> pd.DataFrame:
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    scan = pd.read_csv(REPO / "results/test_contamination_scan.csv")
    keep = ~(d.in_train.to_numpy() | (scan.maxj.to_numpy() > 0.5))
    return d[keep].reset_index(drop=True)


def arms(d: pd.DataFrame) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(len(d))
    blank = pd.Series([""] * len(d))
    return {
        "retrieved": (d.species1, d.relation, d.species2),
        "generic":   (d.species1, pd.Series(["interacts with"] * len(d)), d.species2),
        "shuffled":  (d.species1, d.relation.iloc[perm].reset_index(drop=True), d.species2),
        "rel_only":  (blank, d.relation, blank),
        "empty":     (blank, blank, blank),
    }


def score_arm(md: Path, d: pd.DataFrame, s1, rel, s2, dev, bs=64) -> np.ndarray:
    fmt = model_format(md)
    q, p = xenc_format.build_many(fmt, s1, rel, s2, d.sentence.astype(str))
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(md, local_files_only=True).to(dev).eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(q), bs):
            if p is None:
                e = tok(q[i:i + bs], truncation=True, max_length=256,
                        padding=True, return_tensors="pt").to(dev)
            else:
                e = tok(q[i:i + bs], p[i:i + bs], truncation="only_second", max_length=256,
                        padding=True, return_tensors="pt").to(dev)
            P.extend(torch.softmax(m(**e).logits.float(), -1)[:, 1].cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.array(P)


def main() -> None:
    mds = [REPO / f"models/student_v3/xenc_s{i}" for i in (1, 2, 3)]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    d = clean_benchmark()
    y = d.label.to_numpy()
    THR = 0.04   # the block-held-out operating point; identical across every arm

    rows = []
    for name, (s1, rel, s2) in arms(d).items():
        per_seed = [score_arm(md, d, s1, rel, s2, dev) for md in mds]
        S = np.mean(per_seed, axis=0)
        pred = (S >= THR).astype(int)
        rows.append({
            "arm": name,
            "auprc": float(average_precision_score(y, S)),
            "auprc_per_seed_sd": float(np.std([average_precision_score(y, s) for s in per_seed], ddof=1)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "n_accept": int(pred.sum()),
        })
        r = rows[-1]
        print(f"{name:10} AUPRC {r['auprc']:.4f} (sd {r['auprc_per_seed_sd']:.4f})  "
              f"P {r['precision']:.4f}  R {r['recall']:.4f}  F1 {r['f1']:.4f}")

    base = float(y.mean())
    print(f"\nbase rate (accept everything): AUPRC floor {base:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation.json").write_text(json.dumps(
        {"n": int(len(d)), "threshold": THR, "base_rate": base, "arms": rows}, indent=2))
    print(f"wrote {OUT/'ablation.json'}")


if __name__ == "__main__":
    main()
