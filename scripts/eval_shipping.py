#!/usr/bin/env python3
"""Score a shippable checkpoint exactly as the handoff package scores it.

Runs handoff/biotic_verifier/predict.py's own predict() -- same input builder, same polarity
mapping, same candidate rules, same four-way direction decoding -- on

  1. the 437 clean rows of the benchmark (interaction), at the checkpoint's threshold;
  2. the expert direction gold (direction_curation_v2.xlsx at the classifier root, plus the
     labelled rows of data/evaluation/direction_curation_220.xlsx), in four categories.

Gold codes: 1/F -> FORWARD, 2/R -> REVERSE, ? -> UNCERTAIN, bidirectionnal -> BIDIRECTIONAL;
N (the pair does not interact) and ! (flagged) are reported separately, never scored as a
direction.

Usage
  python3 scripts/eval_shipping.py --model models/dirhead/joint_a05_s1 --name a05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[1]
HANDOFF = REPO / "handoff/biotic_verifier"
sys.path.insert(0, str(HANDOFF))
import predict as PR  # noqa: E402

OUT = REPO / "results/shipping_2026-10-02"
CODE = {"1": "FORWARD", "F": "FORWARD", "2": "REVERSE", "R": "REVERSE", "?": "UNCERTAIN",
        "BIDIRECTIONNAL": "BIDIRECTIONAL", "BIDIRECTIONAL": "BIDIRECTIONAL"}


def benchmark():
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    scan = pd.read_csv(REPO / "results/test_contamination_scan.csv")
    keep = ~(d.in_train.to_numpy() | (scan.maxj.to_numpy() > 0.5))
    return d[keep].reset_index(drop=True)


def direction_gold():
    rows = []
    v2 = pd.read_excel(REPO / "direction_curation_v2.xlsx", sheet_name="annotate")
    for _, r in v2[v2.SUBJECT_IS.notna()].iterrows():
        rows.append(dict(row_id=r.row_id, sentence=r.sentence, species1=r.species1,
                         relation=r.relation, species2=r.species2,
                         raw=str(r.SUBJECT_IS).strip(), note=r.get("NOTE"), block="v2"))
    w = pd.read_excel(REPO / "data/evaluation/direction_curation_220.xlsx", sheet_name="annotate")
    for _, r in w[w.DIRECTION.notna()].iterrows():
        rows.append(dict(row_id=r.row_id, sentence=r.sentence, species1=r.species1,
                         relation=r.relation, species2=r.species2,
                         raw=str(r.DIRECTION).strip(), note=r.get("NOTE"), block="220"))
    g = pd.DataFrame(rows).drop_duplicates("row_id", keep="first").reset_index(drop=True)
    g["gold"] = g.raw.str.upper().map(CODE)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="interaction threshold; default the checkpoint's recorded threshold_dev")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, tok, cfg = PR.load(REPO / a.model, device=dev)
    thr = a.threshold if a.threshold is not None else float(cfg.get("threshold_dev", 0.5))
    PR.INTERACT_THR = thr
    max_len = int(cfg.get("max_len", 256))
    out = {"model": a.model, "name": a.name, "threshold": thr, "n_pol": m.dir.n_pol,
           "has_undirected_branch": m.dir.h is not None}

    # ---- interaction, 437 rows -------------------------------------------------------------
    d = benchmark(); y = d.label.to_numpy()
    for rules in (False, True):
        r = PR.predict(m, tok, d.species1, d.species2, d.relation, d.sentence, device=dev,
                       bs=32, max_len=max_len, rules=rules)
        p = r.interacts.to_numpy()
        key = "with_rules" if rules else "no_rules"
        out[key] = {"P": precision_score(y, p), "R": recall_score(y, p), "F1": f1_score(y, p),
                    "accepts": int(p.sum()), "fp": int(((p == 1) & (y == 0)).sum())}
        if not rules:
            out["auprc"] = float(average_precision_score(y, r.p_interact))
            out["truncated_rows"] = int(r.truncated.sum())
            out["taxon_not_located_rows"] = int((r.both_taxa_located == 0).sum())
            r.assign(label=y, source=d.source).to_csv(OUT / f"bench_{a.name}.csv", index=False)

    # ---- direction, expert gold, four categories --------------------------------------------
    g = direction_gold()
    r = PR.predict(m, tok, g.species1, g.species2, g.relation, g.sentence, device=dev,
                   bs=32, max_len=max_len, rules=True)
    g["pred"] = r.direction.values
    g["interacts"] = r.interacts.values
    # the direction head's own answer, before the interaction gate
    head = np.where(r.symmetric_relation == 1, "BIDIRECTIONAL",
                    np.where(r.direction_confidence.fillna(0) < PR.DIR_ABSTAIN, "UNCERTAIN",
                             np.where(r.p_species1_is_subject.fillna(0.5) >= 0.5, "FORWARD", "REVERSE")))
    head = np.where(r.both_taxa_located == 1, head, "UNCERTAIN")
    g["pred_head"] = head
    g.to_csv(OUT / f"direction_{a.name}.csv", index=False)
    scored = g[g.gold.notna()]
    dr = scored[scored.gold.isin(["FORWARD", "REVERSE"])]
    res = {}
    for col in ("pred", "pred_head"):
        answered = dr[col].isin(["FORWARD", "REVERSE"])
        correct = (dr[col] == dr.gold)
        res[col] = {
            "directional_items": int(len(dr)),
            "answered": int(answered.sum()),
            "coverage": float(answered.mean()),
            "accuracy_on_answered": float(correct[answered].mean()) if answered.any() else None,
            "accuracy_counting_abstention_as_wrong": float(correct.mean()),
            "uncertain_gold_items": int((scored.gold == "UNCERTAIN").sum()),
            "uncertain_gold_answered_uncertain": int(((scored.gold == "UNCERTAIN") & (scored[col] == "UNCERTAIN")).sum()),
            "bidirectional_gold_items": int((scored.gold == "BIDIRECTIONAL").sum()),
            "bidirectional_gold_answered_bidirectional": int(((scored.gold == "BIDIRECTIONAL") & (scored[col] == "BIDIRECTIONAL")).sum()),
            "confusion": pd.crosstab(scored.gold, scored[col]).to_dict(),
        }
    out["direction"] = res
    out["direction_no_interaction_gold"] = {
        "items": int((g.raw.str.upper() == "N").sum()),
        "model_says_no_interaction": int(((g.raw.str.upper() == "N") & (g.interacts == 0)).sum())}
    (OUT / f"eval_{a.name}.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "direction"}, indent=2, default=float))
    for col, v in res.items():
        print(f"\n[{col}] directional {v['directional_items']}: coverage {v['coverage']:.3f}, "
              f"accuracy on answered {v['accuracy_on_answered']:.3f}, strict {v['accuracy_counting_abstention_as_wrong']:.3f}; "
              f"'?' gold -> UNCERTAIN {v['uncertain_gold_answered_uncertain']}/{v['uncertain_gold_items']}; "
              f"bidirectional gold -> BIDIRECTIONAL {v['bidirectional_gold_answered_bidirectional']}/{v['bidirectional_gold_items']}")


if __name__ == "__main__":
    main()
