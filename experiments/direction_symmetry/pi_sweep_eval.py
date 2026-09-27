#!/usr/bin/env python3
"""Evaluate the controlled pi sweep and emit the dose-response table.

Every arm is trained on the same SemEval passages, pairs and relations; only the order in
which the two arguments are presented differs, and the test set is identical throughout.
"""
import json
import numpy as np
from eval_direction_general import evaluate

PIS = [("050", 0.50), ("060", 0.60), ("070", 0.70), ("080", 0.80), ("090", 0.90), ("100", 1.00)]
TEST = "semeval_test_ours.csv"

rows = []
print(f"{'pi':>5}  {'arm':22} {'dir acc':>16} {'swap consistency':>18}")
print("-" * 66)
for tag, pi in PIS:
    A, S = [], []
    for s in (1, 2):
        try:
            r = evaluate(f"pi_text_uncon_{tag}_s{s}", TEST)
        except Exception:
            continue
        A.append(r["acc"]); S.append(r["swap"])
    if not A:
        print(f"{pi:5.2f}  (not finished)"); continue
    f = lambda v: (f"{np.mean(v):.4f} ±{np.std(v, ddof=1):.4f}" if len(v) > 1
                   else f"{np.mean(v):.4f}        ")
    print(f"{pi:5.2f}  {'text + uncon.':22} {f(A):>16} {f(S):>18}")
    rows.append(dict(pi=pi, arm="text_uncon", k=len(A),
                     acc=float(np.mean(A)), acc_sd=float(np.std(A, ddof=1)) if len(A) > 1 else None,
                     swap=float(np.mean(S)), swap_sd=float(np.std(S, ddof=1)) if len(S) > 1 else None))

print()
for tag, pi in (("050", 0.50), ("100", 1.00)):
    try:
        r = evaluate(f"pi_canon_sym_{tag}_s1", TEST)
    except Exception:
        print(f"{pi:5.2f}  canonical control (not finished)"); continue
    print(f"{pi:5.2f}  {'canonical + sym.':22} {r['acc']:16.4f} {r['swap']:18.4f}")
    rows.append(dict(pi=pi, arm="canonical_sym", k=1, acc=r["acc"], acc_sd=None,
                     swap=r["swap"], swap_sd=None))

json.dump(rows, open("PI_SWEEP.json", "w"), indent=2)
print("\nwrote PI_SWEEP.json")
