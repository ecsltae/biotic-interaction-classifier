#!/usr/bin/env python3
"""The SemEval 2x2 design ablation, three seeds per cell."""
import numpy as np
from eval_direction_general import evaluate

CELLS = [("canonical", "symmetric"), ("canonical", "uncon"),
         ("text", "symmetric"), ("text", "uncon")]
print(f"{'input':10} {'head':10} {'dir acc':>16} {'swap':>16} {'und AUC':>16} {'max dev':>10}")
print("-" * 84)
rows = []
for inp, hd in CELLS:
    A, S, U, M = [], [], [], []
    for s in (1, 2, 3):
        r = evaluate(f"semeval_{inp}_{hd}_s{s}", "semeval_test_ours.csv")
        A.append(r["acc"]); S.append(r["swap"]); U.append(r["auc"]); M.append(r["maxdev"])
    f = lambda v: f"{np.mean(v):.4f} ±{np.std(v, ddof=1):.4f}"
    print(f"{inp:10} {hd:10} {f(A):>16} {f(S):>16} {f(U):>16} {max(M):10.2e}")
    rows.append(dict(input=inp, head=hd, acc=float(np.mean(A)), acc_sd=float(np.std(A, ddof=1)),
                     swap=float(np.mean(S)), swap_sd=float(np.std(S, ddof=1)),
                     auc=float(np.mean(U)), auc_sd=float(np.std(U, ddof=1)), maxdev=max(M)))
import json; json.dump(rows, open("SEMEVAL_2X2.json", "w"), indent=2)
print("\nwrote SEMEVAL_2X2.json")
