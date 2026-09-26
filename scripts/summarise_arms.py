#!/usr/bin/env python3
"""One table over every evaluated arm: AUPRC and best F1 on the 440 clean rows,
and the paired McNemar against V1 on the 141 rows where V1's decision is recorded."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import chi2
REPO = Path(__file__).resolve().parents[1]
R = REPO/"results/v4_targeted"

d = pd.read_csv(REPO/"data/evaluation/unified_test_set.csv")
d = d[~d.in_train].reset_index(drop=True)
y = d.label.to_numpy(); hv = d.v1.notna().to_numpy(); v1 = d.v1.fillna(0).to_numpy().astype(int)

def mcn(a_ok, b_ok):
    n01 = int((~a_ok & b_ok).sum()); n10 = int((a_ok & ~b_ok).sum())
    if n01+n10 == 0: return 1.0, n01, n10
    return float(chi2.sf((abs(n01-n10)-1)**2/(n01+n10), 1)), n01, n10

rows = []
for f in sorted(glob.glob(str(R/"scores_*.npy"))):
    name = Path(f).stem.replace("scores_", "")
    S = np.load(f)
    if len(S) != len(d): continue
    grid = np.arange(0.005, 1.0, 0.005)
    f1s = [f1_score(y, (S >= g).astype(int), zero_division=0) for g in grid]
    bt = float(grid[int(np.argmax(f1s))]); bf = float(max(f1s))
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y, S)
    # threshold chosen on the V1 subset, which is where the paired test lives
    fv = [f1_score(y[hv], (S[hv] >= g).astype(int), zero_division=0) for g in grid]
    tv = float(grid[int(np.argmax(fv))])
    pred = (S[hv] >= tv).astype(int)
    p, k, m = mcn(v1[hv] == y[hv], pred == y[hv])
    rows.append(dict(arm=name, auprc=round(ap, 4), best_f1_440=round(bf, 4), thr=bt,
                     f1_on_141=round(max(fv), 4), k_fixed=k, m_broken=m,
                     chi2=round((abs(k-m)-1)**2/(k+m), 2) if k+m else 0.0,
                     mcnemar_p=round(p, 4), beats_v1=("YES" if p < 0.05 and k > m else "no")))
t = pd.DataFrame(rows).sort_values("best_f1_440", ascending=False)
pd.set_option("display.width", 200)
print(f"V1 on its 141 rows: P=0.8352 R=0.8941 F1=0.8636   (bar: chi2>3.84)")
print(t.to_string(index=False))
t.to_csv(R/"summary_arms.csv", index=False)
print(f"\nwrote {R/'summary_arms.csv'}")
