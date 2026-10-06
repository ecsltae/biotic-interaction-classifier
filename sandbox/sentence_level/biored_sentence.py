#!/usr/bin/env python3
"""Sandbox: the SENTENCE-level question on BioRED, where its gold is exact.

A sentence is positive when it co-mentions at least one pair of concepts that BioRED relates
(any candidate of the sentence-level benchmark is positive). Three ways to answer it, scored on the
BC8 test sentences:
  sentlab   a sentence-input model trained on sentence-level labels (one row per sentence)
  sentence  the paper's sentence-input arm (trained on candidate rows): its score is the same for
            every candidate of a sentence
  pair-max  the paper's pair-conditioned arm, max over the sentence's candidate pairs: sentence-level
            knowledge read off pair-level verification
Thresholds come from the development split (BioRED's test split, BC8 protocol).

Usage
  python3 sandbox/sentence_level/biored_sentence.py --build      # write the sentence tables
  python3 sandbox/sentence_level/biored_sentence.py --score      # after training sentlab
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
SRC = REPO / "data/benchmarks/biored_bc8"
DST = REPO / "data/benchmarks/biored_bc8_sentence"
OUT = REPO / "sandbox/sentence_level/results"
GRID = np.arange(0.01, 1.0, 0.01)


def sentence_table(split: str) -> pd.DataFrame:
    d = pd.read_csv(SRC / f"{split}.csv", dtype={"pmid": str})
    g = d.groupby("sent_id").agg(text=("text", "first"), label=("label", "max"), n_candidates=("label", "size"))
    return g.reset_index().assign(source_species="", target_species="", interaction_type="")


def best_thr(y, s):
    return float(max(GRID, key=lambda t: f1_score(y, (s >= t).astype(int), zero_division=0)))


def prf(y, p):
    return {"P": float(precision_score(y, p, zero_division=0)), "R": float(recall_score(y, p, zero_division=0)),
            "F1": float(f1_score(y, p, zero_division=0))}


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--build", action="store_true"); ap_.add_argument("--score", action="store_true")
    a = ap_.parse_args()
    if a.build:
        DST.mkdir(parents=True, exist_ok=True)
        for s in ("train", "dev", "test"):
            t = sentence_table(s); t.to_csv(DST / f"{s}.csv", index=False)
            print(f"{s}: {len(t)} sentences with >= 1 candidate pair, {t.label.mean():.3f} positive")
    if a.score:
        from paperA_tables import order_free
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        OUT.mkdir(parents=True, exist_ok=True)
        res = {}
        cand = {s: pd.read_csv(SRC / f"{s}.csv", dtype={"pmid": str}).rename(columns={
            "source_species": "species1", "target_species": "species2", "interaction_type": "relation",
            "text": "sentence"}).assign(relation="") for s in ("dev", "test")}
        sent = {s: sentence_table(s) for s in ("dev", "test")}
        y = sent["test"].label.to_numpy(); ydev = sent["dev"].label.to_numpy()

        def per_sentence(split, scores):            # max over a sentence's candidates
            return (cand[split].assign(s=scores).groupby("sent_id").s.max()
                    .reindex(sent[split].sent_id).to_numpy())
        arms = {"sentlab": [REPO / f"models/sandbox_sentence/biored_sentlab_s{k}" for k in (1, 2, 3)],
                "sentence": [REPO / f"models/biored_bc8/sentence_s{k}" for k in (1, 2, 3)],
                "pair-max": [REPO / f"models/biored_bc8/pair_s{k}" for k in (1, 2, 3)]}
        for arm, mds in arms.items():
            per_seed, sd, st = [], [], []
            for md in mds:
                if arm == "sentlab":                   # scores sentences directly
                    q = {s: sent[s].rename(columns={"text": "sentence", "source_species": "species1",
                                                     "target_species": "species2", "interaction_type": "relation"})
                         for s in ("dev", "test")}
                    sdv, sts = order_free(md, q["dev"], dev), order_free(md, q["test"], dev)
                else:
                    sdv, sts = per_sentence("dev", order_free(md, cand["dev"], dev)), per_sentence("test", order_free(md, cand["test"], dev))
                sd.append(sdv); st.append(sts); per_seed.append(float(ap(y, sts)))
            Sd, St = np.mean(sd, axis=0), np.mean(st, axis=0)
            thr = best_thr(ydev, Sd)
            res[arm] = {"auprc_per_seed": per_seed, "auprc_mean": float(np.mean(per_seed)),
                        "auprc_sd": float(np.std(per_seed, ddof=1)), "auprc_ensemble": float(ap(y, St)),
                        "threshold_dev": thr, **prf(y, (St >= thr).astype(int))}
            print(f"{arm:9} AUPRC {res[arm]['auprc_mean']:.3f} ± {res[arm]['auprc_sd']:.3f} | at dev threshold "
                  f"{thr:.2f}: P {res[arm]['P']:.3f} R {res[arm]['R']:.3f} F1 {res[arm]['F1']:.3f}", flush=True)
        res["n_sentences"] = int(len(y)); res["positive_rate"] = float(y.mean())
        (OUT / "biored_sentence.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
