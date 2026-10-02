#!/usr/bin/env python3
"""Compare any arm against V1 on the WHOLE clean benchmark, not just the 150 recorded rows.

Why this script exists
  V1's recorded decisions cover 150 of 449 benchmark rows, and those 150 are exactly
  biotx100 + reject50 -- the two blocks V1 was evaluated on. Every McNemar against V1
  so far ran on that 141-row clean overlap, where reject50 (a deliberately adversarial
  recall set on which BOTH systems are near chance) is 29% of the rows against 9% of
  the full benchmark. The test was also underpowered: V1 makes only 24 errors there, so
  any arm that breaks 11+ of V1's correct answers cannot reach p<0.05 at all.

  V1 is a checkpoint, not just a column of recorded answers, so it can simply be run on
  all 440 rows. That is what this does.

Two V1s, and they are not the same thing
  recorded -- the decisions the deployed pipeline produced (rules + NER + classifier).
              Authoritative, but only on 150 rows.
  stored   -- V1's own probabilities on the 299 test299 rows, saved when V1 was scored
              there (results/v2/base299_V1_champion.json), at the threshold recorded with
              them. Also authoritative. Together these cover every clean row, so no
              re-run reconstruction is needed and none is used.

Thresholds
  challenger -- 0.5, pre-specified, never fitted on this set.
  V1         -- not chosen here at all. Every V1 decision is one V1 actually made:
                the recorded pipeline output, or its stored test299 probabilities at the
                threshold 0.28 recorded alongside them.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_unified import mcnemar                                      # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.core import clean_mask, to_clean                            # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score,       # noqa: E402
                             precision_score, recall_score)

REPO = Path(__file__).resolve().parents[1]
# Authoritative V1 decisions on every clean row, built by scripts/build_v1_decisions.py:
#   150 rows -- the decisions the deployed pipeline recorded (the `v1` column)
#   299 rows -- V1's own stored probabilities for test299 from
#               results/v2/base299_V1_champion.json, thresholded at its recorded 0.28.
#               These reproduce that file's confusion (TP119/FP19/FN44/TN117) against its
#               own copy of the labels. The two benchmark files differ only at row 400
#               (gold revised 1 -> 0 on 2026-09-25); on current gold V1's test299 confusion is
#               TP118/FP20/FN44/TN117.
# Nothing here is a re-run reconstruction.
V1_DECISIONS = REPO / "results/v1_decisions_449.npy"   # full length, -1 = no decision
CHALLENGER_THR = 0.5


def _norm(s):
    return re.sub(r"\W+", " ", str(s)).lower().strip()


def clean_benchmark():
    """Benchmark minus every row that is a near-duplicate of a training passage.

    The shipped `in_train` column flags 9 rows by exact match. A 5-gram Jaccard scan
    against all 48,338 training passages (results/test_contamination_scan.csv) finds 3
    more, all in biotx100 and all at Jaccard 1.000 -- exact duplicates that differ only
    in mojibake ("na\u221a\u00d8ve" vs "naive"), which unicode-aware \W normalisation
    preserves and ASCII-stripping collapses.

    test299 is clean by this measure: 0 rows above 0.5, max 0.176, mean 0.008. Since the
    whole V1 result lives in test299, that matters more than the total count.
    """
    keep = clean_mask()               # the shared loader of src/eval/core.py
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    return d[keep].reset_index(drop=True), keep, int((~keep).sum())


def compare(scores, name, d, sel):
    y = d.label.to_numpy()
    S = to_clean(scores)              # 440-long arrays (older runs) or 437-long
    assert len(S) == len(d), f"{name}: {len(S)} scores for {len(d)} rows"
    V = np.load(V1_DECISIONS)[sel]
    assert (V >= 0).all(), "a kept row has no V1 decision"
    pred = (S >= CHALLENGER_THR).astype(int)
    out = {"arm": name, "n": int(len(y)), "auprc": float(average_precision_score(y, S)),
           "f1": float(f1_score(y, pred)), "precision": float(precision_score(y, pred)),
           "recall": float(recall_score(y, pred)), "blocks": {}}
    for src in list(d.source.unique()) + ["ALL"]:
        m = (d.source == src).to_numpy() if src != "ALL" else np.ones(len(y), bool)
        p, k, m_ = mcnemar(V[m], pred[m], y[m])
        out["blocks"][src] = {
            "n": int(m.sum()), "pos": int(y[m].sum()),
            "v1_f1": float(f1_score(y[m], V[m], zero_division=0)),
            "arm_f1": float(f1_score(y[m], pred[m], zero_division=0)),
            "k_arm_fixes": k, "m_arm_breaks": m_, "mcnemar_p": p}
    # the authoritative comparison, on the rows where V1's real decisions are recorded
    hv = d.v1.notna().to_numpy()
    if hv.sum():
        rec = d.v1.fillna(0).to_numpy().astype(int)[hv]
        p, k, m_ = mcnemar(rec, pred[hv], y[hv])
        out["vs_recorded_v1"] = {"n": int(hv.sum()), "k": k, "m": m_, "mcnemar_p": p,
                                 "v1_f1": float(f1_score(y[hv], rec))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="name=path/to/scores.npy")
    ap.add_argument("--out", default="results/vs_v1_full.json")
    a = ap.parse_args()
    d, sel, nleak = clean_benchmark()
    print(f"clean benchmark: {len(d)} rows ({nleak} leaked rows removed)\n")
    res = []
    for spec in a.scores:
        name, path = spec.split("=", 1)
        res.append(compare(np.load(REPO / path), name, d, sel))
    hdr = f'{"arm":18} {"AUPRC":>7} {"F1":>6} {"P":>6} {"R":>6} {"k":>4} {"m":>4} {"p(437)":>10} {"p(rec141)":>10}'
    print(hdr); print("-" * len(hdr))
    for r in res:
        b = r["blocks"]["ALL"]; v = r.get("vs_recorded_v1", {})
        print(f'{r["arm"]:18} {r["auprc"]:7.4f} {r["f1"]:6.3f} {r["precision"]:6.3f} '
              f'{r["recall"]:6.3f} {b["k_arm_fixes"]:4d} {b["m_arm_breaks"]:4d} '
              f'{b["mcnemar_p"]:10.2e} {v.get("mcnemar_p", float("nan")):10.4f}')
    (REPO / a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
