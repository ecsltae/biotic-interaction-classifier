#!/usr/bin/env python3
"""Relation-marginalised (TTA) evaluation of cross-encoders on the unified test set.

Companion to scripts/eval_unified.py. Same label semantics (SPECIES-LEVEL), same
leaked-row exclusion, same McNemar-vs-V1 protocol; it reproduces eval_unified.py
exactly when run with --probes 0 --pool none --agg prob.

WHY THIS EXISTS
---------------
Every training label answers the FULL-TRIPLE question ("is this the right relation,
in this direction?"). The unified test set asks the SPECIES-LEVEL question ("do these
two taxa interact?"). At inference we can convert one into the other without any new
training data: re-query the same frozen cross-encoder with a bank of candidate
relations and pool. max-pooling over the bank asks "is there ANY relation under which
this passage links these two taxa?", which is the species-level question.

Thresholds are never chosen on the evaluation rows: --crossfit picks the threshold on
4/5 of the rows and scores the held-out 5th, rotating, repeated over 10 seeds.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import chi2, rankdata
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import xenc_format as _X
from eval_unified import model_format as _model_format
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = Path(__file__).resolve().parents[1]

# Ten most frequent interaction types among POSITIVE rows of
# data/training/distill/v3_combined_train.csv. Order matters: --probes N takes a prefix.
PROBE_BANK = ["pathogens", "prey", "exposed to", "infestation", "parasite of",
              "predator", "pathogen of", "affecting", "infested", "transmitted by"]


def score_queries(model, tok, queries: list[str], passages: list[str],
                  device: torch.device, batch_size: int = 128) -> np.ndarray:
    """Return the class-1-minus-class-0 margin for each (query, passage) pair."""
    out = []
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            enc = tok(queries[i:i + batch_size], passages[i:i + batch_size],
                      truncation="only_second", max_length=256, padding=True,
                      return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    logits = np.concatenate(out, 0)
    return logits[:, 1] - logits[:, 0]


def pool_probes(margins: np.ndarray, rule: str) -> np.ndarray:
    """Pool a (n_probes, n_rows) margin matrix down to (n_rows,)."""
    if rule == "max":
        return margins.max(0)
    if rule == "top3":
        return np.sort(margins, 0)[-3:].mean(0)
    if rule == "mean":
        return margins.mean(0)
    if rule == "frac":                      # fraction of probes firing, rescaled
        return (margins > 0).mean(0) * 10 - 5
    raise ValueError(f"unknown pool rule {rule!r}")


def mcnemar(a_pred: np.ndarray, b_pred: np.ndarray, y: np.ndarray):
    """Returns (p, n_b_fixes, n_b_breaks) for b relative to a."""
    a, b = (a_pred == y), (b_pred == y)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(chi2.sf((abs(n01 - n10) - 1) ** 2 / (n01 + n10), 1)), n01, n10


def _f1_curve(scores, y, idx, thresholds):
    s, yl = scores[idx], y[idx]
    order = np.argsort(-s)
    sy, ss = yl[order], s[order]
    tp, fp, n_pos = np.cumsum(sy), np.cumsum(1 - sy), yl.sum()
    k = np.searchsorted(-ss, -np.asarray(thresholds), side="right")
    tpk, fpk = np.concatenate([[0], tp])[k], np.concatenate([[0], fp])[k]
    denom = 2 * tpk + fpk + (n_pos - tpk)
    return np.where(denom > 0, 2 * tpk / np.maximum(denom, 1e-9), 0.0)


def _thresholds(scores):
    u = np.unique(scores)
    if len(u) < 2:
        return u
    return np.concatenate([[u.min() - 1e-9], (u[:-1] + u[1:]) / 2, [u.max() + 1e-9]])


def crossfit_predictions(scores, y, strata, n_splits=5, seeds=range(10)):
    """Out-of-fold decisions with the threshold fitted only on the other folds."""
    ths = _thresholds(scores)
    preds = []
    for seed in seeds:
        oof = np.zeros(len(y), dtype=int)
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for tr, te in splitter.split(scores.reshape(-1, 1), strata):
            curve = _f1_curve(scores, y, tr, ths)
            oof[te] = (scores[te] >= ths[int(np.argmax(curve))]).astype(int)
        preds.append(oof)
    return np.stack(preds)


def build_scores(model_dirs, data, probes, pool, agg, device, batch_size=128):
    """Score every row under the original query, or under `probes` relation probes."""
    passages = data.sentence.astype(str).tolist()
    s1 = data.species1.astype(str).tolist()
    s2 = data.species2.astype(str).tolist()
    rel = data.relation.astype(str).tolist()
    bank = PROBE_BANK[:probes]
    # Formats whose segment A does not contain the relation string. Marginalising over a
    # relation bank is a mathematical no-op for these -- every probe builds a byte-identical
    # query -- so a "TTA" run would silently be its own control. Refuse instead.
    RELATION_FREE = {"pair", "pair_canon", "mark_canon", "mark_only"}

    per_model = []
    for md in model_dirs:
        fmt = _model_format(md)                     # honour what the checkpoint was trained with
        tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            md, local_files_only=True).to(device).eval()
        if bank and fmt in RELATION_FREE:
            raise ValueError(
                f"{md}: input_format {fmt!r} does not put the relation in the query, so "
                f"{len(bank)} relation probes would produce identical inputs. "
                f"Relation-probe TTA is undefined for this checkpoint; run with --probes 0.")
        if bank:
            mat = []
            for probe in bank:
                q, pas = _X.build_many(fmt, s1, [probe] * len(s1), s2, passages)
                mat.append(score_queries(model, tok, q, pas, device, batch_size))
            per_model.append(pool_probes(np.stack(mat), pool))
        else:
            q, pas = _X.build_many(fmt, s1, rel, s2, passages)
            per_model.append(score_queries(model, tok, q, pas, device, batch_size))
        del model
        torch.cuda.empty_cache()
    per_model = np.stack(per_model)
    if agg == "logit":
        return per_model.mean(0)
    if agg == "prob":
        return (1 / (1 + np.exp(-per_model))).mean(0)
    if agg == "rank":
        return np.mean([rankdata(m) for m in per_model], 0) / per_model.shape[1]
    raise ValueError(f"unknown agg {agg!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model dirs, relative to the classifier/ repo root")
    ap.add_argument("--name", required=True)
    ap.add_argument("--probes", type=int, default=10,
                    help="how many relation probes to marginalise over (0 = original query)")
    ap.add_argument("--pool", default="max", choices=["max", "top3", "mean", "frac"])
    ap.add_argument("--agg", default="logit", choices=["logit", "prob", "rank"],
                    help="how to combine models (logit = margin averaging, the best measured)")
    ap.add_argument("--baseline-scores", default=None,
                    help=".npy of a comparison system's scores on the same rows (e.g. V2)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=None, help="write the results JSON here")
    ap.add_argument("--save-scores", default=None, help="write the raw scores here as .npy")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    data = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    data = data[~data.in_train].reset_index(drop=True)     # 440 clean rows
    y = data.label.to_numpy()
    strata = np.array([f"{s}_{l}" for s, l in zip(data.source, y)])

    scores = build_scores([REPO / m for m in args.models], data,
                          args.probes, args.pool, args.agg, device, args.batch_size)
    if args.save_scores:
        np.save(REPO / args.save_scores, scores)

    ths = _thresholds(scores)
    curve = _f1_curve(scores, y, np.arange(len(y)), ths)
    oof = crossfit_predictions(scores, y, strata)
    cf_f1 = [f1_score(y, o, zero_division=0) for o in oof]

    res = {
        "model": args.name,
        "config": {"models": args.models, "probes": args.probes,
                   "pool": args.pool, "agg": args.agg},
        "n": int(len(data)), "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, scores)),
        "best_f1_oracle_threshold": float(curve.max()),
        "best_thr": float(ths[int(np.argmax(curve))]),
        "crossfit_f1_mean": float(np.mean(cf_f1)),
        "crossfit_f1_sd": float(np.std(cf_f1)),
        "per_source": {}, "vs_v1": {}, "vs_baseline": {},
    }
    for s, _ in data.groupby("source"):
        m = (data.source == s).to_numpy()
        res["per_source"][s] = {
            "n": int(m.sum()), "pos": int(y[m].sum()),
            "auprc": float(average_precision_score(y[m], scores[m])) if len(set(y[m])) > 1 else None,
            "crossfit_f1": float(np.median([f1_score(y[m], o[m], zero_division=0) for o in oof])),
        }

    # V1 is only recorded on biotx100 + the clean part of reject50 (141 of the 440 rows).
    hv = data.v1.notna().to_numpy()
    if hv.sum():
        v1 = data.v1.fillna(0).to_numpy().astype(int)
        stats = [mcnemar(v1[hv], o[hv], y[hv]) for o in oof]
        res["vs_v1"] = {
            "n": int(hv.sum()),
            "v1_precision": float(precision_score(y[hv], v1[hv], zero_division=0)),
            "v1_recall": float(recall_score(y[hv], v1[hv], zero_division=0)),
            "v1_f1": float(f1_score(y[hv], v1[hv], zero_division=0)),
            "model_precision": float(np.median([precision_score(y[hv], o[hv], zero_division=0) for o in oof])),
            "model_recall": float(np.median([recall_score(y[hv], o[hv], zero_division=0) for o in oof])),
            "model_f1": float(np.median([f1_score(y[hv], o[hv], zero_division=0) for o in oof])),
            "mcnemar_p_median": float(np.median([s[0] for s in stats])),
            "k_v1_errors_fixed": float(np.median([s[1] for s in stats])),
            "m_new_errors": float(np.median([s[2] for s in stats])),
        }

    if args.baseline_scores:
        base = np.load(REPO / args.baseline_scores)
        boof = crossfit_predictions(base, y, strata)
        stats = [mcnemar(boof[i], oof[i], y) for i in range(len(oof))]
        res["vs_baseline"] = {
            "file": args.baseline_scores, "n": int(len(y)),
            "baseline_auprc": float(average_precision_score(y, base)),
            "baseline_crossfit_f1": float(np.median([f1_score(y, o, zero_division=0) for o in boof])),
            "mcnemar_p_median": float(np.median([s[0] for s in stats])),
            "k_baseline_errors_fixed": float(np.median([s[1] for s in stats])),
            "m_new_errors": float(np.median([s[2] for s in stats])),
        }

    print(json.dumps(res, indent=2, default=float))
    if args.out:
        p = REPO / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
