#!/usr/bin/env python3
"""Item-by-item confusion delta against V1 on the rows where V1's decision is recorded.

Aggregate F1 hides whether a model fixed V1's errors or merely traded them. This
prints, for every one of V1's errors, whether the model fixed it, and for every row
V1 got right, whether the model broke it -- which is exactly the k and m that decide
the McNemar test.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import chi2
REPO = Path(__file__).resolve().parents[1]

def mcnemar(a_ok, b_ok):
    n01 = int((~a_ok & b_ok).sum()); n10 = int((a_ok & ~b_ok).sum())
    if n01 + n10 == 0: return 1.0, n01, n10
    return float(chi2.sf((abs(n01-n10)-1)**2/(n01+n10), 1)), n01, n10

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help=".npy of scores on the 440 clean rows")
    ap.add_argument("--thr", type=float, default=None, help="default: threshold maximising F1 on the 141 V1 rows")
    ap.add_argument("--name", default="model")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--baseline", nargs="*", default=[],
                    help="name=scores.npy of another model, for a paired McNemar on all 440")
    a = ap.parse_args()

    d = pd.read_csv(REPO/"data/evaluation/unified_test_set.csv")
    d = d[~d.in_train].reset_index(drop=True)
    S = np.load(REPO/a.scores if not Path(a.scores).is_absolute() else a.scores)
    assert len(S) == len(d), f"{len(S)} scores vs {len(d)} rows"
    hv = d.v1.notna().to_numpy()
    y = d.label.to_numpy(); v1 = d.v1.fillna(0).to_numpy().astype(int)

    if a.thr is None:
        grid = np.arange(0.005, 1.0, 0.005)
        a.thr = float(grid[int(np.argmax([f1_score(y[hv], (S[hv] >= g).astype(int), zero_division=0) for g in grid]))])
    pred = (S >= a.thr).astype(int)

    v1_ok = (v1[hv] == y[hv]); m_ok = (pred[hv] == y[hv])
    p, n01, n10 = mcnemar(v1_ok, m_ok)
    sub = d[hv].reset_index(drop=True)
    sub["v1_pred"] = v1[hv]; sub["m_score"] = S[hv]; sub["m_pred"] = pred[hv]
    sub["v1_ok"] = v1_ok; sub["m_ok"] = m_ok

    def cls(r):
        if r.v1_ok and r.m_ok: return "both_right"
        if not r.v1_ok and r.m_ok: return "FIXED"
        if r.v1_ok and not r.m_ok: return "BROKE"
        return "both_wrong"
    sub["delta"] = sub.apply(cls, axis=1)

    print(f"=== {a.name}  thr={a.thr:.3f}  (V1 rows n={hv.sum()}) ===")
    print(f"V1     P={precision_score(y[hv],v1[hv],zero_division=0):.4f} R={recall_score(y[hv],v1[hv],zero_division=0):.4f} F1={f1_score(y[hv],v1[hv],zero_division=0):.4f}")
    print(f"{a.name:6s} P={precision_score(y[hv],pred[hv],zero_division=0):.4f} R={recall_score(y[hv],pred[hv],zero_division=0):.4f} F1={f1_score(y[hv],pred[hv],zero_division=0):.4f}")
    print(f"\nk (V1 errors FIXED)      = {n01}")
    print(f"m (V1 correct BROKEN)    = {n10}")
    stat = (abs(n01-n10)-1)**2/(n01+n10) if n01+n10 else 0
    print(f"McNemar chi2={stat:.3f}  p={p:.4f}   {'PASSES' if p<0.05 and n01>n10 else 'does not pass'} the 3.84 bar")
    print(f"\nall 440 clean rows: best-F1 on full set = "
          f"{max(f1_score(y,(S>=g).astype(int),zero_division=0) for g in np.arange(0.005,1,0.005)):.4f}")
    print(sub.delta.value_counts().to_string())

    for tag in ("FIXED", "BROKE"):
        g = sub[sub.delta == tag]
        print(f"\n--- {tag} ({len(g)}) ---")
        for _, r in g.iterrows():
            kind = "FP" if r.v1_pred == 1 and r.label == 0 else ("FN" if r.v1_pred == 0 and r.label == 1 else "ok")
            print(f"  [{r.source:9s}] gold={r.label} v1={r.v1_pred}({kind}) m={r.m_pred} s={r.m_score:.3f} | "
                  f"{str(r.species1)[:28]} --{str(r.relation)[:16]}--> {str(r.species2)[:28]}")
            if a.verbose: print(f"      {str(r.sentence)[:190]}")
    # paired comparison against other MODELS on the full clean set. V1 has no
    # decision on test299, so that comparison stays on its 141; a model-vs-model
    # comparison can use all 440.
    for spec in a.baseline:
        bname, bpath = spec.split("=", 1)
        B = np.load(REPO/bpath if not Path(bpath).is_absolute() else bpath)
        grid = np.arange(0.005, 1.0, 0.005)
        bt = float(grid[int(np.argmax([f1_score(y, (B >= g).astype(int), zero_division=0) for g in grid]))])
        bp = (B >= bt).astype(int)
        pv, n01, n10 = mcnemar(bp == y, pred == y)
        print(f"\nvs {bname} on all {len(y)} clean rows (each at its own best-F1 threshold, "
              f"{bname}={bt:.3f} {a.name}={a.thr:.3f}):")
        print(f"   {bname} F1={f1_score(y,bp,zero_division=0):.4f}   {a.name} F1={f1_score(y,pred,zero_division=0):.4f}")
        print(f"   {a.name} fixes {n01}, breaks {n10}, McNemar p={pv:.4f}"
              f"   {'PASSES' if pv<0.05 and n01>n10 else 'does not pass'}")

    out = REPO/f"results/v4_targeted/delta_{a.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}")

if __name__ == "__main__": main()
