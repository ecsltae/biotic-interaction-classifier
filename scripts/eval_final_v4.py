#!/usr/bin/env python3
"""Honest evaluation for the final combined model.

Wraps scripts/eval_unified.py -- it imports `score` and reuses the same label
semantics, the same leaked-row exclusion and the same McNemar definition, so every
number stays comparable with the recorded V1/V2/V3 baselines.

What it adds, all of it about NOT choosing the operating point on the rows we report:

  cross-fitted threshold   5-fold stratified on source x label; the threshold is
                           maximised on 4/5 and applied to the held-out 5th, so a
                           row's decision never saw its own label. Repeated over
                           R fold seeds; we report the median and the full range.
  dev threshold            the checkpoint's own `threshold_dev`, averaged over seeds.
                           Fully test-blind, reported alongside.
  paired tests             McNemar on out-of-fold decisions vs V1 (its 141 rows),
                           and vs V2 / V3 (all 440), plus a paired bootstrap on AUPRC.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2
import eval_unified as EU

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.core import clean_benchmark, to_clean  # noqa: E402  (the one 437-row loader)
CACHE = REPO/"results/final_v4/scores"


def load_clean():
    return clean_benchmark()          # 437 rows: in_train and near-duplicates dropped


def scores_for(model_dirs, d, tag):
    """Mean probability over checkpoints, cached per checkpoint."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    S = []
    for md in model_dirs:
        # The cache key must bind the benchmark too. Keyed on the model dir alone, a
        # regenerated or corrected test set silently reuses stale score vectors -- and the
        # benchmark HAS been corrected (row 400, 2026-09-25).
        import hashlib as _h
        _bench = _h.sha256(
            pd.util.hash_pandas_object(d[["sentence", "species1", "species2"]], index=False)
            .values.tobytes()).hexdigest()[:12]
        key = str(md).replace("/", "__").strip("_") + f"__{_bench}__n{len(d)}"
        f = CACHE/f"{key}.npy"
        if f.exists():
            S.append(np.load(f))
        else:
            s = EU.score(REPO/md, d, dev)
            np.save(f, s); S.append(s)
    return np.mean(S, axis=0)


def crossfit(S, y, strat, R=20, seed0=0):
    """Out-of-fold decisions from a threshold fitted only on the other folds."""
    grid = np.arange(0.005, 1.0, 0.005)
    preds = []
    for r in range(R):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed0+r)
        p = np.zeros(len(y), int)
        for tr, te in skf.split(S.reshape(-1, 1), strat):
            f1s = [f1_score(y[tr], (S[tr] >= g).astype(int), zero_division=0) for g in grid]
            p[te] = (S[te] >= grid[int(np.argmax(f1s))]).astype(int)
        preds.append(p)
    return np.array(preds)


def mcnemar(a_pred, b_pred, y):
    """(p, n_b_only_right, n_a_only_right) -- b is the challenger."""
    a, b = (a_pred == y), (b_pred == y)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10, 0.0
    c2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(chi2.sf(c2, 1)), n01, n10, float(c2)


def boot_auprc(y, Sa, Sb, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(average_precision_score(y[i], Sb[i]) - average_precision_score(y[i], Sa[i]))
    d = np.array(d)
    return dict(delta=float(average_precision_score(y, Sb) - average_precision_score(y, Sa)),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p_le0=float((d <= 0).mean()))


def centred_scores(model_dirs, d, dev):
    """Ensemble score centred on each member's own dev threshold. Decide at 0.0.

    dev_threshold()/scores_for() below average thresholds and average probabilities
    separately, and those two aggregations do not compose: cutting the mean probability at
    the mean threshold is not the decision any member would make. Member thresholds here
    span 0.29-0.70, and the damage shows as the ~0.13 dev-to-oracle F1 gap on every
    ensemble row of results/final_v4/arms.csv.

    Centring in logit space fixes it: z_i = logit(p_i) - logit(t_i), average the z_i,
    decide at 0. Every member then contributes at its own calibrated boundary, the
    ensemble's operating point is 0.0 by construction and fully test-blind, and for a
    single checkpoint this reduces exactly to that checkpoint's recorded threshold_dev.

    A precision floor is expressed as a single global offset b on z, fitted on dev --
    never on the reporting rows.
    """
    def _logit(x):
        x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    Z = []
    for md in model_dirs:
        c = REPO/md/"student_config.json"
        t = json.loads(c.read_text()).get("threshold_dev") if c.exists() else None
        if t is None:
            raise ValueError(f"{md}: no threshold_dev; cannot centre it")
        Z.append(_logit(scores_for([md], d, dev)) - _logit(t))
    return np.mean(Z, axis=0)


def dev_threshold(model_dirs):
    ts = []
    for md in model_dirs:
        c = REPO/md/"student_config.json"
        if c.exists():
            t = json.loads(c.read_text()).get("threshold_dev")
            if t is not None:
                ts.append(t)
    return float(np.mean(ts)) if ts else None


def summarise(tag, model_dirs, d, R=20):
    y = d.label.to_numpy()
    strat = (d.source.astype(str) + "_" + d.label.astype(str)).to_numpy()
    S = scores_for(model_dirs, d, tag)
    P = crossfit(S, y, strat, R=R)
    cf_f1 = np.array([f1_score(y, p, zero_division=0) for p in P])
    grid = np.arange(0.005, 1.0, 0.005)
    orc = max(f1_score(y, (S >= g).astype(int), zero_division=0) for g in grid)
    tdev = dev_threshold(model_dirs)
    o = dict(tag=tag, n=len(d), n_models=len(model_dirs),
             auprc=float(average_precision_score(y, S)),
             cf_f1_med=float(np.median(cf_f1)), cf_f1_sd=float(cf_f1.std()),
             cf_f1_min=float(cf_f1.min()), cf_f1_max=float(cf_f1.max()),
             oracle_f1=float(orc), dev_thr=tdev)
    if tdev is not None:
        o["dev_f1_440"] = float(f1_score(y, (S >= tdev).astype(int), zero_division=0))
    o["per_source_auprc"] = {s: float(average_precision_score(y[(d.source == s).to_numpy()],
                                                             S[(d.source == s).to_numpy()]))
                             for s in sorted(d.source.unique())}
    return o, S, P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    d = load_clean()
    o, S, P = summarise(a.name, a.models, d, R=a.repeats)
    print(json.dumps(o, indent=2))
