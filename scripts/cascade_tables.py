#!/usr/bin/env python3
"""Escalation: the deployed verifier decides, and only its least confident candidates go to an LLM.

The deployed configuration (handoff `predict()` scores of joint_a05_s1 at its fixed threshold 0.5,
with the candidate rules applied first; results/shipping_2026-10-02/bench_a05.csv) keeps its own
decision except on the band of candidates whose score lies nearest 0.5; those take the zero-shot
LLM's greedy answer to the pair question (results/paperA_v2/llm/biodiv_<model>_pair.csv). A rule
rejection is final. The band is reported as a dial -- several fractions, none chosen on these rows.

Usage
  python3 scripts/cascade_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "handoff/biotic_verifier"))
from eval.core import clean_benchmark  # noqa: E402
import candidate_rules as CR  # noqa: E402

BANDS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
LLMS = ("qwen3-32b", "qwen3.5-122b")


def prf(y, p):
    tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn)
    return {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0, "FP": fp, "FN": fn}


def mcnemar(a, b, y):
    ca, cb = a == y, b == y
    k, m = int((~ca & cb).sum()), int((ca & ~cb).sum())
    return (1.0 if k + m == 0 else float(chi2.sf((abs(k - m) - 1) ** 2 / (k + m), 1))), k, m


def main() -> None:
    d = clean_benchmark()
    y = d.label.to_numpy()
    st = pd.read_csv(REPO / "results/shipping_2026-10-02/bench_a05.csv")
    assert (st.species1.values == d.species1.values).all(), "deployed scores misaligned"
    s = st.p_interact.to_numpy()
    rej = np.array([CR.reject_reason(str(p), a, r, b) is not None
                    for p, a, r, b in zip(d.sentence, d.species1, d.relation, d.species2)])
    base = (s >= 0.5).astype(int) * ~rej
    order = np.argsort(np.abs(s - 0.5), kind="stable")          # least confident first
    out = {"n": int(len(y)), "positives": int(y.sum()), "student": prf(y, base), "llm": {}}
    for tag in LLMS:
        L = pd.read_csv(REPO / f"results/paperA_v2/llm/biodiv_{tag}_pair.csv")
        assert (L.species1.values == d.species1.values).all(), f"{tag} answers misaligned"
        v = L.verdict.to_numpy()
        rows = []
        for f in BANDS:
            k = int(round(f * len(y)))
            band = np.zeros(len(y), bool); band[order[:k]] = True
            pred = np.where(band, v, (s >= 0.5).astype(int)) * ~rej
            p, fx, br = mcnemar(base, pred, y)
            rows.append({"band": f, "llm_calls": k, **prf(y, pred), "mcnemar_vs_student": p, "fixed": fx, "broken": br})
        out["llm"][tag] = rows
    (REPO / "results/paperA_v2/cascade.json").write_text(json.dumps(out, indent=2))
    print(f"student alone: {out['student']}")
    for tag, rows in out["llm"].items():
        print(f"\n{tag}:  band  calls     P      R     F1   FP   p(vs student)")
        for r in rows:
            print(f"       {r['band']:4.0%} {r['llm_calls']:5d}  {r['P']:.3f}  {r['R']:.3f}  {r['F1']:.3f}  {r['FP']:3d}  "
                  f"{r['mcnemar_vs_student']:.3g} ({r['fixed']}/{r['broken']})")


if __name__ == "__main__":
    main()
