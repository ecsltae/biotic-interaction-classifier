#!/usr/bin/env python3
"""Rebuild every empirical number in paperA from the non-superseded harness.

What this fixes relative to the numbers currently in paperA.tex (a 22 Sep build that
predates docs/EVAL_AUDIT_2026-09-25.md):

  1. Label column. Scores against the benchmark's own species-level labels, which is
     what BENCHMARKS.json pins and what the deployed filter is actually for. The old
     numbers used `triples_ok_full` on biotx100 only.
  2. Full benchmark. All 437 clean rows, not the 150-row biotx100+reject50 slice.
     The clean set is 449 minus 9 `in_train` minus 3 exact duplicates found by the
     5-gram Jaccard scan -- all three of those are in biotx100, which is why the
     paper's "biotx100 shares no passage with training data (0/100)" is wrong.
  3. Whole passage. The full `sentence` column is the B segment for every row; the
     query is the triple. Rows whose passage does not survive 256 wordpieces are
     counted and reported rather than silently truncated.
  4. Seeds, honestly. Per-seed metrics AND the probability-ensemble are reported
     separately. Averaging three seeds' probabilities and reporting the result is an
     ensemble, not "means over three seeds", and the two differ.
  5. Threshold, without the leak. Three policies are reported side by side:
       pre-specified   tau = 0.5, fixed in advance, never fitted
       block-held-out  tau chosen on two source blocks, applied to the third
       oracle          tau maximising F1 on the reporting set -- an UPPER BOUND, not
                       a result, and labelled as such wherever it appears
     The superseded scripts/eval_v3.py took tau from the reporting set's own gold
     prior, which is why its numbers cannot be published.

Usage
  python3 scripts/rebuild_paperA_numbers.py --models models/student_v3/xenc_s1 \
      models/student_v3/xenc_s2 models/student_v3/xenc_s3 --name Verifier-PC
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score)
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_unified import mcnemar, model_format, score  # noqa: E402

V1_DECISIONS = REPO / "results/v1_decisions_449.npy"
CONTAM = REPO / "results/test_contamination_scan.csv"
JACCARD_MAX = 0.5
PRESPECIFIED_THR = 0.5


def clean_benchmark() -> pd.DataFrame:
    """The 437 rows that are neither flagged in_train nor near-duplicates of training text."""
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    d["v1_decision"] = np.load(V1_DECISIONS)
    scan = pd.read_csv(CONTAM)
    if len(scan) != len(d):
        raise SystemExit(f"contamination scan covers {len(scan)} rows, benchmark has {len(d)}")
    leaked = d.in_train.to_numpy() | (scan.maxj.to_numpy() > JACCARD_MAX)
    kept = d[~leaked].reset_index(drop=True)
    kept.attrs["n_dropped"] = int(leaked.sum())
    kept.attrs["n_dropped_by_jaccard_only"] = int(((scan.maxj.to_numpy() > JACCARD_MAX)
                                                   & ~d.in_train.to_numpy()).sum())
    return kept


def truncation_report(d: pd.DataFrame, md: Path) -> dict:
    """How many rows lose passage text at 256 wordpieces, and where they sit."""
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    lens = [len(tok(str(s), add_special_tokens=False)["input_ids"]) for s in d.sentence]
    lens = np.array(lens)
    over = lens > 240  # 256 minus room for the query segment and specials
    return {"max_passage_wordpieces": int(lens.max()),
            "median": int(np.median(lens)),
            "n_over_240": int(over.sum()),
            "pos_rate_over": float(d.label.to_numpy()[over].mean()) if over.any() else None}


def metrics(y, s, thr) -> dict:
    p = (s >= thr).astype(int)
    return {"thr": float(thr),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)),
            "n_accept": int(p.sum())}


def best_thr(y, s, grid) -> float:
    return float(max(grid, key=lambda t: f1_score(y, (s >= t).astype(int), zero_division=0)))


def block_held_out(d, y, S, grid) -> dict:
    """tau chosen on the other two source blocks, applied to the held-out one.

    Uses every row exactly once as test, and never fits tau on the rows it scores.
    """
    rows, preds = [], np.zeros(len(y), int)
    for blk in sorted(d.source.unique()):
        te = (d.source == blk).to_numpy()
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        t = best_thr(y[tr], S[tr], grid)
        preds[te] = (S[te] >= t).astype(int)
        rows.append({"block": blk, "n": int(te.sum()), "thr_from_other_blocks": t,
                     **{k: v for k, v in metrics(y[te], S[te], t).items() if k != "thr"}})
    pooled = {"precision": float(precision_score(y, preds, zero_division=0)),
              "recall": float(recall_score(y, preds, zero_division=0)),
              "f1": float(f1_score(y, preds, zero_division=0))}
    return {"per_block": rows, "pooled": pooled, "_preds": preds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default="results/paperA_rebuild_2026-09-28")
    a = ap.parse_args()

    outdir = REPO / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    d = clean_benchmark()
    y = d.label.to_numpy()
    mds = [REPO / m for m in a.models]
    grid = np.arange(0.01, 1.00, 0.01)

    per_seed_scores = [score(md, d, dev) for md in mds]
    S_ens = np.mean(per_seed_scores, axis=0)

    out = {
        "name": a.name,
        "checkpoints": [str(m.relative_to(REPO)) for m in mds],
        "input_format": sorted({model_format(m) for m in mds}),
        "benchmark": {
            "n": int(len(d)), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "n_dropped_as_contaminated": d.attrs["n_dropped"],
            "dropped_by_jaccard_beyond_in_train": d.attrs["n_dropped_by_jaccard_only"],
            "per_source": {s: {"n": int((d.source == s).sum()),
                               "pos": int(y[(d.source == s).to_numpy()].sum())}
                           for s in sorted(d.source.unique())},
            "label_semantics": "species-level: do these two taxa interact in this passage",
        },
        "truncation": truncation_report(d, mds[0]),
    }

    # ---- per seed, then the ensemble, never conflated ----
    seed_rows = []
    for md, s in zip(mds, per_seed_scores):
        seed_rows.append({
            "checkpoint": str(md.relative_to(REPO)),
            "auprc": float(average_precision_score(y, s)),
            "prespecified": metrics(y, s, PRESPECIFIED_THR),
            "oracle_UPPER_BOUND": metrics(y, s, best_thr(y, s, grid)),
        })
    f1s = [r["prespecified"]["f1"] for r in seed_rows]
    aps = [r["auprc"] for r in seed_rows]
    out["per_seed"] = {
        "seeds": seed_rows,
        "mean_auprc": float(np.mean(aps)),
        "sd_auprc": float(np.std(aps, ddof=1)) if len(aps) > 1 else None,
        "mean_f1_at_prespecified": float(np.mean(f1s)),
        "sd_f1_at_prespecified": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
    }
    out["ensemble"] = {
        "note": "probabilities averaged across seeds; this is an ensemble, not a per-seed mean",
        "auprc": float(average_precision_score(y, S_ens)),
        "prespecified": metrics(y, S_ens, PRESPECIFIED_THR),
        "block_held_out": block_held_out(d, y, S_ens, grid),
        "oracle_UPPER_BOUND": metrics(y, S_ens, best_thr(y, S_ens, grid)),
    }
    bho_preds = out["ensemble"]["block_held_out"].pop("_preds")

    # ---- V1 on every clean row, at the decisions V1 actually made ----
    v1 = d.v1_decision.to_numpy().astype(int)
    have = v1 >= 0
    if have.all():
        v1b = v1.astype(int)
        cmp = {}
        for policy, pred in (("prespecified", (S_ens >= PRESPECIFIED_THR).astype(int)),
                             ("block_held_out", bho_preds)):
            p, n01, n10 = mcnemar(v1b, pred, y)
            cmp[policy] = {"mcnemar_p": p, "arm_fixes": n01, "arm_breaks": n10,
                           "arm": metrics(y, pred.astype(float), 0.5),
                           "v1_f1": float(f1_score(y, v1b, zero_division=0)),
                           "v1_precision": float(precision_score(y, v1b, zero_division=0)),
                           "v1_recall": float(recall_score(y, v1b, zero_division=0))}
        # per block, so the recall-side set is not hidden inside the pool
        cmp["per_block"] = {}
        pred = bho_preds
        for blk in sorted(d.source.unique()):
            m = (d.source == blk).to_numpy()
            p, n01, n10 = mcnemar(v1b[m], pred[m], y[m])
            cmp["per_block"][blk] = {
                "n": int(m.sum()), "mcnemar_p": p, "arm_fixes": n01, "arm_breaks": n10,
                "v1_f1": float(f1_score(y[m], v1b[m], zero_division=0)),
                "arm_f1": float(f1_score(y[m], pred[m], zero_division=0))}
        out["vs_v1"] = cmp
    else:
        out["vs_v1"] = {"error": f"V1 decision missing on {int((~have).sum())} rows"}

    np.save(outdir / f"S_{a.name}_ensemble.npy", S_ens)
    np.save(outdir / f"S_{a.name}_per_seed.npy", np.array(per_seed_scores))
    (outdir / f"rebuild_{a.name}.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
