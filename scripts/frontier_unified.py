#!/usr/bin/env python3
"""Precision/recall frontier wrapper around scripts/eval_unified.py.

Does NOT re-implement evaluation: it imports eval_unified.evaluate(), so scoring,
leak exclusion and label semantics are identical and the numbers are comparable.
What it adds is the full threshold frontier (not just best F1) and, at every
threshold, the McNemar test against V1 on the 141 clean rows where V1's decision
is recorded -- including the high-recall region (recall >= V1's 0.8941), which is
where a win against V1 has to come from.

  python scripts/frontier_unified.py --models <dir> [<dir> ...] --name <tag> \
      [--out results/loss_shaping/<tag>.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from eval_unified import evaluate, mcnemar, wilson  # noqa: E402

V1_RECALL = 0.8941176470588236  # measured on the 141 clean V1 rows


def frontier(model_dirs, name):
    o, S, d = evaluate([REPO / m if not str(m).startswith("/") else Path(m) for m in model_dirs], name)
    y = d.label.to_numpy()
    hv = d.v1.notna().to_numpy()
    v1v = d.v1.fillna(0).to_numpy().astype(int)[hv]
    yv = y[hv]

    rows = []
    # grid reaches far below 0.01: several models cannot reach V1's recall even at
    # threshold 0.01, and truncating the grid there would hide that.
    grid = np.concatenate([np.array([1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3]),
                           np.arange(0.01, 1.0, 0.005)])
    for t in grid:
        pred = (S >= t).astype(int)
        predv = (S[hv] >= t).astype(int)
        p_, n01, n10 = mcnemar(v1v, predv, yv)
        tp = int(((predv == 1) & (yv == 1)).sum()); fp = int(((predv == 1) & (yv == 0)).sum())
        fn = int(((predv == 0) & (yv == 1)).sum()); tn = int(((predv == 0) & (yv == 0)).sum())
        rows.append(dict(
            thr=round(float(t), 4),
            # full 440-row clean set
            precision=float(precision_score(y, pred, zero_division=0)),
            recall=float(recall_score(y, pred, zero_division=0)),
            f1=float(f1_score(y, pred, zero_division=0)),
            # 141-row V1 subset
            v1sub_precision=float(precision_score(yv, predv, zero_division=0)),
            v1sub_recall=float(recall_score(yv, predv, zero_division=0)),
            v1sub_f1=float(f1_score(yv, predv, zero_division=0)),
            tp=tp, fp=fp, fn=fn, tn=tn,
            mcnemar_p=float(p_), model_fixes_v1=int(n01), v1_beats_model=int(n10)))
    F = pd.DataFrame(rows)

    res = {k: v for k, v in o.items() if k != "by_threshold"}
    res["n_v1_rows"] = int(hv.sum())
    # eval_unified's own coarse grid (0.01..0.99 step 0.01) -- the comparable headline
    res["best_f1_evalunified"] = float(o["best_f1"])
    res["best_thr_evalunified"] = float(o["best_thr"])
    # fraction of true positives the model scores near zero: the failure mode that makes
    # every high-precision threshold destroy as many of V1's correct calls as it fixes
    res["dead_positive_rate"] = {f"lt_{t}": float((S[y == 1] < t).mean())
                                 for t in (0.005, 0.01, 0.05, 0.1)}
    # best F1 on the full clean set over the wide grid
    res["best_f1"] = float(F.f1.max())
    res["best_thr"] = float(F.loc[F.f1.idxmax(), "thr"])
    # high-recall region on the FULL clean set
    hi = F[F.recall >= V1_RECALL]
    res["high_recall_region"] = None if hi.empty else dict(
        n_thresholds=int(len(hi)),
        max_precision_at_recall_ge_v1=float(hi.precision.max()),
        best_f1_in_region=float(hi.f1.max()),
        thr_of_best=float(hi.loc[hi.f1.idxmax(), "thr"]))
    # best achievable McNemar against V1, and the best one in the high-recall region
    # signed: only thresholds where the model fixes MORE of V1's errors than it breaks
    W = F[F.model_fixes_v1 > F.v1_beats_model]
    best_m = F.loc[F.mcnemar_p.idxmin()] if W.empty else W.loc[W.mcnemar_p.idxmin()]
    res["best_mcnemar"] = dict(thr=float(best_m.thr), p=float(best_m.mcnemar_p),
                               model_wins=bool(best_m.model_fixes_v1 > best_m.v1_beats_model),
                               fixes=int(best_m.model_fixes_v1), breaks=int(best_m.v1_beats_model),
                               v1sub_f1=float(best_m.v1sub_f1),
                               v1sub_precision=float(best_m.v1sub_precision),
                               v1sub_recall=float(best_m.v1sub_recall))
    hiv = F[(F.v1sub_recall >= V1_RECALL) & (F.model_fixes_v1 > F.v1_beats_model)]
    res["best_mcnemar_at_v1_recall"] = None if hiv.empty else dict(
        thr=float(hiv.loc[hiv.mcnemar_p.idxmin(), "thr"]),
        p=float(hiv.mcnemar_p.min()),
        fixes=int(hiv.loc[hiv.mcnemar_p.idxmin(), "model_fixes_v1"]),
        breaks=int(hiv.loc[hiv.mcnemar_p.idxmin(), "v1_beats_model"]),
        v1sub_f1=float(hiv.loc[hiv.mcnemar_p.idxmin(), "v1sub_f1"]),
        max_v1sub_precision=float(hiv.v1sub_precision.max()))
    res["max_recall_full"] = float(F.recall.max())
    res["max_recall_v1sub"] = float(F.v1sub_recall.max())
    # F1 at the threshold each model picked on its own dev split (no test-set tuning)
    devts = []
    for m in model_dirs:
        cfg = (REPO / m if not str(m).startswith("/") else Path(m)) / "student_config.json"
        if cfg.exists():
            devts.append(json.loads(cfg.read_text())["threshold_dev"])
    if devts:
        t = float(np.mean(devts))
        pred = (S >= t).astype(int)
        predv = (S[hv] >= t).astype(int)
        p_, n01, n10 = mcnemar(v1v, predv, yv)
        res["at_dev_threshold"] = dict(
            thr=t, precision=float(precision_score(y, pred, zero_division=0)),
            recall=float(recall_score(y, pred, zero_division=0)),
            f1=float(f1_score(y, pred, zero_division=0)),
            mcnemar_p=float(p_), fixes=int(n01), breaks=int(n10))
    return res, F, S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res, F, S = frontier(a.models, a.name)
    print(json.dumps(res, indent=2, default=float))
    if a.out:
        p = REPO / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2, default=float))
        F.to_csv(str(p).replace(".json", "_frontier.csv"), index=False)
        np.save(str(p).replace(".json", "_scores.npy"), S)
