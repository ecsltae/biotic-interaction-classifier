#!/usr/bin/env python3
"""Score the dev-holdout joint checkpoints at thresholds chosen WITHOUT the reporting set.

The shipped joint_a05_s1 was trained on 100% of the binary data with `threshold_dev`
hardcoded to 0.5, so its operating point was pre-specified but never *chosen* -- which is why
the precision-first policy could not be expressed honestly. train_direction.py now holds out a
pair-grouped 10% dev split and records two thresholds from it:

    threshold_dev              maximises F1 on dev
    threshold_precision_floor  the lowest threshold whose dev precision >= --precision-floor

Both are fitted on dev only. This script reports the benchmark numbers at each, plus 0.5 for
continuity, and seed-aggregates so the operating point is not read off one run.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"experiments"/"direction"))
sys.path.insert(0, str(REPO/"experiments"/"multitask"))
from eval_direction import load, run                                    # noqa: E402
from eval_unified import mcnemar                                        # noqa: E402
from sklearn.metrics import (precision_score, recall_score, f1_score,   # noqa: E402
                             average_precision_score)


def clean():
    d = pd.read_csv(REPO/"data/evaluation/unified_test_set.csv")
    d["maxj"] = pd.read_csv(REPO/"results/test_contamination_scan.csv").maxj.values
    leak = (d.maxj >= 0.5) | d.in_train
    return d[~leak].reset_index(drop=True), (~leak).to_numpy()


def main():
    d, sel = clean()
    y = d.label.to_numpy()
    V = np.load(REPO/"results/v1_decisions_449.npy")[sel]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(p for p in (REPO/"models/dirhead").glob("joint_dev_s*")
                   if (p/"student_config.json").exists())
    if not ckpts:
        print("no dev-holdout checkpoints yet"); return
    rows, store = [], {}
    for md in ckpts:
        cfg = json.loads((md/"student_config.json").read_text())
        m, tok, _ = load(md, dev)
        S, _, _ = run(m, tok, d.species1, d.species2, d.relation, d.sentence, pol=None, dev=dev)
        store[md.name] = S
        np.save(REPO/f"results/dirhead/{md.name}_binary_scores.npy", S)
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        for nm, t in (("0.5 (old, hardcoded)", 0.5),
                      ("dev max-F1", cfg.get("threshold_dev")),
                      (f"dev P>={cfg.get('precision_floor')}", cfg.get("threshold_precision_floor"))):
            if t is None:
                continue
            pr = (S >= t).astype(int); p, k, mm = mcnemar(V, pr, y)
            rows.append(dict(ckpt=md.name, rule=nm, thr=round(float(t), 3),
                             auprc=round(float(average_precision_score(y, S)), 4),
                             P=round(float(precision_score(y, pr, zero_division=0)), 4),
                             R=round(float(recall_score(y, pr, zero_division=0)), 4),
                             F1=round(float(f1_score(y, pr, zero_division=0)), 4),
                             k=k, m=mm, mcnemar_p=float(p),
                             dev_auprc=cfg.get("dev_auprc")))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    if len(store) > 1:
        E = np.mean(list(store.values()), axis=0)
        print(f"\n{len(store)}-seed mean-probability ensemble:")
        for th in (0.5,):
            pr = (E >= th).astype(int); p, k, mm = mcnemar(V, pr, y)
            print(f"  thr {th}: AUPRC {average_precision_score(y,E):.4f} "
                  f"P {precision_score(y,pr):.4f} R {recall_score(y,pr):.4f} "
                  f"F1 {f1_score(y,pr):.4f}  k={k} m={mm} p={p:.2e}")
    t.to_csv(REPO/"results/dirhead/joint_dev_summary.csv", index=False)
    print("\nwrote results/dirhead/joint_dev_summary.csv")


if __name__ == "__main__":
    main()
