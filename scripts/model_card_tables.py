#!/usr/bin/env python3
"""Every table of docs/MODEL_CARD_joint_v2.md that is computed on the 437-row benchmark.

Internal: the card compares the shipped checkpoint with the deployed filter (V1), which no
paper uses as a baseline. Scores are the stored per-row arrays of each arm; V1's decisions are
results/v1_decisions_449.npy (scripts/build_v1_decisions.py). Thresholds are pre-specified
(0.5) unless a column says "oracle", which sweeps the threshold on this set and is an upper
bound, not an operating point.

`--labels pre_goldfix` scores against the labels before the 2026-09-25 revision of row 400, so
the identity of every stored array can be checked against the numbers the card used to print.

Usage
  python3 scripts/model_card_tables.py
  python3 scripts/model_card_tables.py --labels pre_goldfix
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from eval.core import clean_mask, to_clean  # noqa: E402

GRID = np.arange(0.01, 1.0, 0.01)          # generations table
GRID_V1 = np.arange(0.005, 1.0, 0.005)     # V1-oracle figures, as first computed
ARMS = {  # card row -> stored per-row scores
    "V2 cross-encoder, triple": ("npz", "V2_(triple)"),
    "V3 cross-encoder, triple": ("npz", "V3_(triple)"),
    "V4 12-checkpoint ensemble": ("npz", "ENS4FMT_(V4)"),
    "+ species-level relabel": ("npy", "models/encoder_sweep/CONTROL_triple_s1.scores.npy"),
    "+ joint mark_canon (shipped)": ("npy", "results/dirhead/joint_a05_s1_binary_scores.npy"),
    "+ joint, frozen trunk": ("npy", "results/dirhead/detach_s1_binary_scores.npy"),
}
SPREAD = {"joint_a05_s1": "+ joint mark_canon (shipped)", "detach_s1": "+ joint, frozen trunk",
          "species-relabel, 1 seed": "+ species-level relabel"}


def mcnemar(a_pred, b_pred, y):
    ca, cb = a_pred == y, b_pred == y
    k, m = int((~ca & cb).sum()), int((ca & ~cb).sum())
    return (1.0 if k + m == 0 else float(chi2.sf((abs(k - m) - 1) ** 2 / (k + m), 1))), k, m


def prf(y, p):
    return (float(f1_score(y, p, zero_division=0)), float(precision_score(y, p, zero_division=0)),
            float(recall_score(y, p, zero_division=0)))


def oracle(y, s, mask=None, grid=GRID):
    m = np.ones(len(y), bool) if mask is None else mask
    return float(grid[int(np.argmax([f1_score(y[m], (s[m] >= g).astype(int), zero_division=0) for g in grid]))])


def key(sentence, s1, s2):
    n = lambda s: re.sub(r"\W+", " ", str(s)).lower().strip()  # noqa: E731
    return f"{n(sentence)}||{n(s1)}||{n(s2)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=("current", "pre_goldfix"), default="current")
    a = ap.parse_args()
    keep = clean_mask()
    bench = REPO / ("data/evaluation/unified_test_set.csv" if a.labels == "current"
                    else "data/evaluation/unified_test_set.pre_goldfix.csv")
    d = pd.read_csv(bench)[keep].reset_index(drop=True)
    y = d.label.to_numpy()
    V = np.load(REPO / "results/v1_decisions_449.npy")[keep]
    z = np.load(REPO / "results/final_v4/scores.npz")
    S = {name: to_clean(z[k] if kind == "npz" else np.load(REPO / k)) for name, (kind, k) in ARMS.items()}
    shipped = S["+ joint mark_canon (shipped)"]
    pred = (shipped >= 0.5).astype(int)
    out = {"labels": a.labels, "n": int(len(y)), "positives": int(y.sum())}

    f, p, r = prf(y, pred)
    vf, vp, vr = prf(y, V)
    out["interaction"] = {"shipped@0.5": {"auprc": float(average_precision_score(y, shipped)), "F1": f, "P": p, "R": r},
                          "V1": {"F1": vf, "P": vp, "R": vr}}

    blocks = {}
    for b in ("biotx100", "reject50", "test299", "ALL"):
        m = (d.source == b).to_numpy() if b != "ALL" else np.ones(len(y), bool)
        pv, k, mm = mcnemar(V[m], pred[m], y[m])
        blocks[b] = {"n": int(m.sum()), "prevalence": float(y[m].mean()), "V1_F1": prf(y[m], V[m])[0],
                     "model_F1": prf(y[m], pred[m])[0], "k": k, "m": mm, "p": pv}
    out["blocks"] = blocks

    # test299's three source sets, from the paired file it was assembled from
    t5 = pd.read_csv(REPO / "data/evaluation/test500_paired.csv")
    sub_of = dict(zip((key(*r) for r in zip(t5.sentence, t5.species1, t5.species2)), t5.source))
    sub = np.array([sub_of.get(key(*r), "") if s == "test299" else s
                    for s, *r in zip(d.source, d.sentence, d.species1, d.species2)])
    assert (sub[(d.source == "test299").to_numpy()] != "").all(), "a test299 row has no source set"
    out["sub_blocks"] = {}
    for b in ("biotx100", "reject50", "EP-A", "EP-passage", "eval-100/BioTx-random"):
        m = sub == b
        pv, k, mm = mcnemar(V[m], pred[m], y[m])
        out["sub_blocks"][b] = {"n": int(m.sum()), "prevalence": float(y[m].mean()), "V1_F1": prf(y[m], V[m])[0],
                                "model_F1": prf(y[m], pred[m])[0], "k": k, "m": mm, "p": pv}

    # V1 handed its gold-fitted oracle threshold on test299 (its stored probabilities)
    j = json.loads((REPO / "results/v2/base299_V1_champion.json").read_text())["test299"]
    src = pd.read_csv(j["benchmark_path"])
    prob_of = dict(zip((key(*r) for r in zip(src.sentence, src.species1, src.species2)), j["_probs"]))
    m299 = (d.source == "test299").to_numpy()
    pv1 = np.array([prob_of[key(*r)] for r in zip(d.sentence[m299], d.species1[m299], d.species2[m299])])
    t = oracle(y[m299], pv1, grid=GRID_V1)
    v1o = (pv1 >= t).astype(int)
    pv, k, mm = mcnemar(v1o, pred[m299], y[m299])
    out["v1_oracle_test299"] = {"threshold": t, "V1_F1": prf(y[m299], v1o)[0], "k": k, "m": mm, "p": pv}
    # ... and a separate oracle threshold for each of test299's source sets
    out["v1_per_block_oracle"] = {}
    for b in ("EP-A", "EP-passage", "eval-100/BioTx-random"):
        mb = sub[m299] == b
        tb = oracle(y[m299][mb], pv1[mb], grid=GRID_V1)
        vb = (pv1[mb] >= tb).astype(int)
        pv, k, mm = mcnemar(vb, pred[m299][mb], y[m299][mb])
        out["v1_per_block_oracle"][b] = {"threshold": tb, "V1_F1": prf(y[m299][mb], vb)[0], "k": k, "m": mm, "p": pv}

    # one fixed threshold across the three test299 source sets
    out["prevalence_spread"] = {}
    for name, arm in [("V1", None)] + list(SPREAD.items()):
        pr = V if arm is None else (S[arm] >= 0.5).astype(int)
        f1s = {b: prf(y[sub == b], pr[sub == b])[0] for b in ("eval-100/BioTx-random", "EP-A", "EP-passage")}
        out["prevalence_spread"][name] = {**f1s, "spread": max(f1s.values()) - min(f1s.values())}

    # generations
    out["generations"] = {}
    for name, s in S.items():
        pr = (s >= 0.5).astype(int)
        t = oracle(y, s)
        po = (s >= t).astype(int)
        out["generations"][name] = {"auprc": float(average_precision_score(y, s)),
                                    **dict(zip(("F1", "P", "R"), prf(y, pr))), "p@0.5": mcnemar(V, pr, y)[0],
                                    "oracle_thr": t, "oracle_F1": prf(y, po)[0], "p@oracle": mcnemar(V, po, y)[0]}

    # operating points of the shipped model
    out["operating_points"] = {}
    for t in (0.5, 0.9, 0.95):
        pr = (shipped >= t).astype(int)
        f, p, r = prf(y, pr)
        out["operating_points"][str(t)] = {"P": p, "R": r, "F1": f, "p_vs_V1": mcnemar(V, pr, y)[0]}

    # encoder choice: every stored sweep array, AUPRC on these rows and F1 at the checkpoint's
    # own development threshold (sweep_results.json, else the checkpoint's student_config.json)
    from scipy.stats import mannwhitneyu, ttest_ind
    sweep = REPO / "models/encoder_sweep"
    thr = {x["ckpt"]: x["dev_thr"] for x in json.loads((sweep / "sweep_results.json").read_text())}
    thr["CONTROL_biomedbert"] = thr.pop("biomedbert(CONTROL)", None)
    enc = {}
    for f in sorted(sweep.glob("*.scores.npy")):
        name = f.name.replace(".scores.npy", "")
        sc = to_clean(np.load(f))
        t = thr.get(name)
        cfg = sweep / name / "student_config.json"
        if t is None and cfg.exists():
            t = json.loads(cfg.read_text()).get("threshold_dev")
        enc[name] = {"auprc": float(average_precision_score(y, sc)), "dev_thr": t,
                     "F1@dev": prf(y, (sc >= t).astype(int))[0] if t is not None else None,
                     "p@0.5_vs_V1": mcnemar(V, (sc >= 0.5).astype(int), y)[0]}
    fam = {"BiomedBERT": [f"CONTROL_triple_s{i}" for i in range(1, 6)],
           "BioLinkBERT": [f"biolinkbert_s{i}" for i in range(1, 6)],
           "BiodivBERT": [f"biodivbert_s{i}" for i in range(1, 6)],
           "DeBERTa-v3": [f"debertav3_s{i}" for i in range(1, 4)]}
    fams = {k: [enc[c]["auprc"] for c in v if c in enc] for k, v in fam.items()}
    base = fams["BiomedBERT"]
    out["encoders"] = {"per_checkpoint": enc, "families": {
        k: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "auprc": v,
            **({} if k == "BiomedBERT" else {"t_p": float(ttest_ind(v, base).pvalue),
                                             "mannwhitney_p": float(mannwhitneyu(v, base).pvalue),
                                             "delta": float(np.mean(v) - np.mean(base))})}
        for k, v in fams.items()}}

    (REPO / f"results/model_card_tables_{a.labels}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
