#!/usr/bin/env python3
"""Sandbox: the SENTENCE-level question on the 437-row biodiversity benchmark.

Gold, until the expert's sentence labels arrive, is PROVISIONAL ("silver"): a candidate that is
pair-positive makes its sentence positive; for the others, the majority answer of three local LLMs
(Qwen3-32B, Qwen3.5-122B, Qwen3.8-27B) to the sentence question. `--human` reads the expert's
SENTENCE column from the two blind sheets instead, where filled.

Sentence scores:
  sentence  the paper's sentence-input arm (one score per passage)
  pair-max  the paper's pair-conditioned arm, max over every pair of taxon mentions TaxoNERD finds
            in the passage (plus the benchmark's own pair): sentence knowledge from pair verification
  pair-own  the pair arm on the benchmark's own candidate only (reference)
Operating points: block-held-out thresholds, as in the paper.

Usage
  python3 sandbox/sentence_level/biodiv_sentence.py [--human]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score as ap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from eval.core import clean_benchmark  # noqa: E402
from paperA_tables import block_held_out, order_free, prf  # noqa: E402

OUT = REPO / "sandbox/sentence_level/results"
LLMS = ("qwen3-32b", "qwen3.5-122b", "qwen3.8-27b-v035")


def silver(d: pd.DataFrame) -> np.ndarray:
    votes = np.mean([pd.read_csv(REPO / f"results/paperA_v2/llm/biodiv_{m}_sentence.csv").verdict.to_numpy()
                     for m in LLMS], axis=0)
    return np.where(d.label.to_numpy() == 1, 1, (votes >= 0.5).astype(int))


def mentions(texts) -> list[list[str]]:
    from taxonerd import TaxoNERD
    t = TaxoNERD(prefer_gpu=False); t.load("en_ner_eco_biobert")
    out = []
    for s in texts:
        df = t.find_in_text(str(s))
        seen, ms = set(), []
        for m in (df["text"].tolist() if len(df) else []):
            k = m.strip().lower()
            if k and k not in seen:
                seen.add(k); ms.append(m.strip())
        out.append(ms)
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--human", action="store_true", help="use the expert's sentence labels where filled")
    a = ap_.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = clean_benchmark(); y_pair = d.label.to_numpy(); blocks = d.source.to_numpy()
    y = silver(d)
    cache = OUT / "taxonerd_mentions.json"
    if cache.exists():
        ments = json.loads(cache.read_text())
    else:
        ments = mentions(d.sentence); cache.write_text(json.dumps(ments))
    rows = []
    for i, (s, a1, a2, ms) in enumerate(zip(d.sentence, d.species1, d.species2, ments)):
        pairs = {(a1, a2)} | {tuple(sorted(p)) for p in itertools.combinations(ms, 2)}
        rows += [{"i": i, "sentence": s, "species1": p[0], "species2": p[1], "relation": ""} for p in pairs]
    P = pd.DataFrame(rows)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed = []
    for k in (1, 2, 3):
        sc = order_free(REPO / f"models/pair_baseline/xenc_s{k}", P, dev)
        per_seed.append(P.assign(s=sc).groupby("i").s.max().reindex(range(len(d))).to_numpy())
    S = {"sentence": np.load(REPO / "results/paperA_v2/S_biodiv_sentence.npy"),
         "pair-max": np.mean(per_seed, axis=0),
         "pair-own": np.load(REPO / "results/paperA_v2/S_biodiv_pair.npy")}
    res = {"labels": "silver (pair-positive, else majority of three LLMs' sentence answers)",
           "n": int(len(y)), "sentence_positive_rate": float(y.mean()), "pair_positive_rate": float(y_pair.mean()),
           "precision_ceiling_of_a_perfect_sentence_filter_on_pair_gold": float(y_pair.sum() / y.sum()),
           "pairs_scored": int(len(P)), "mean_pairs_per_sentence": float(len(P) / len(d)),
           "pair_max_auprc_per_seed": [float(ap(y, s)) for s in per_seed]}
    for name, s in S.items():
        pred, thr = block_held_out(y, s, blocks)
        res[name] = {"auprc": float(ap(y, s)), **prf(y, pred), "thresholds": thr}
        print(f"{name:9} sentence-level AUPRC {res[name]['auprc']:.3f} | block-held-out P {res[name]['P']:.3f} "
              f"R {res[name]['R']:.3f} F1 {res[name]['F1']:.3f}", flush=True)
    print(f"sentence-positive {y.mean():.3f} vs pair-positive {y_pair.mean():.3f}: a perfect sentence filter "
          f"accepting every sentence-positive candidate has pair-level precision {y_pair.sum() / y.sum():.3f}")
    (OUT / "biodiv_sentence_silver.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
