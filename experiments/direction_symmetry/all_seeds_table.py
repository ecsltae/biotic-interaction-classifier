#!/usr/bin/env python3
"""Every per-seed number quoted in the paper, emitted as LaTeX appendix rows."""
import numpy as np, pandas as pd
from eval_direction_general import evaluate

ARMS = [("SemEval", "canon.", "sym.",   "semeval_canonical_symmetric_s"),
        ("SemEval", "canon.", "uncon.", "semeval_canonical_uncon_s"),
        ("SemEval", "text",   "sym.",   "semeval_text_symmetric_s"),
        ("SemEval", "text",   "uncon.", "semeval_text_uncon_s")]
rows = []
for corpus, inp, hd, pre in ARMS:
    for s in (1, 2, 3):
        r = evaluate(f"{pre}{s}", "semeval_test_ours.csv")
        rows.append(dict(corpus=corpus, input=inp, head=hd, seed=s,
                         acc=r["acc"], swap=r["swap"], auc=r["auc"]))
df = pd.DataFrame(rows)
df.to_csv("per_seed_semeval.csv", index=False)
for (c, i, h), g in df.groupby(["corpus", "input", "head"], sort=False):
    a = " & ".join(f"{v:.4f}" for v in g.acc)
    w = " & ".join(f"{v:.4f}" for v in g.swap)
    u = " & ".join(f"{v:.4f}" for v in g.auc)
    print(f"{i} & {h} & acc & {a} & {g.acc.mean():.4f} & {g.acc.std(ddof=1):.4f} \\\\")
    print(f"       &        & swap & {w} & {g.swap.mean():.4f} & {g.swap.std(ddof=1):.4f} \\\\")
    print(f"       &        & AUC & {u} & {g.auc.mean():.4f} & {g.auc.std(ddof=1):.4f} \\\\")
    print("\\midrule")
