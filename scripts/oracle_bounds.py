#!/usr/bin/env python3
"""Upper bounds for the 'target V1's error classes' lever, computed without training.

Each targeted family is applied to V3's scores as a PERFECT detector (an oracle that
never misfires). If the oracle cannot beat V1 significantly, no training data built
for that family can either, because training data can only approximate the oracle.

This is what makes the lever's verdict a bound rather than one noisy run.
"""
import re, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from scipy.stats import chi2
REPO = Path(__file__).resolve().parents[1]
R = REPO/"results/v4_targeted"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.core import clean_benchmark, to_clean  # noqa: E402  (the one 437-row loader)
d = clean_benchmark()
y = d.label.to_numpy(); hv = d.v1.notna().to_numpy(); v1 = d.v1.fillna(0).to_numpy().astype(int)
V3 = to_clean(np.load(R/"scores_V3.npy"))

def sp(t, n):
    n = str(n).split("|")[0].strip()
    return [m.span() for m in re.finditer(re.escape(n), str(t), re.I)] if n else []
CLADE = re.compile(r"(aceae|ales|idae|inae|oidea|ida|iformes|mycota|mycetes|virales|viridae|phyta|opsida|acea)$", re.I)
BREAK = set("""of in on by with from to as that which for is are was were be been being has have had
infect infects infected infecting parasit parasite preys prey predator transmit transmits host hosts
causes cause caused associated feeding feeds pollinat symbio vector isolated found during after before
between within than such including like against via through""".split())

def f_syn(r):
    return any(0 <= (q2-e1 if q2 >= e1 else s1-e2) <= 3
               for s1, e1 in sp(r.sentence, r.species1) for q2, e2 in sp(r.sentence, r.species2))
def f_anc(r):
    t = str(r.sentence)
    for x, z in ((str(r.species1), str(r.species2)), (str(r.species2), str(r.species1))):
        if CLADE.search(x.strip()):
            for s, e in sp(t, z):
                if re.match(r"\s*\(", t[e:e+3]) and re.search(re.escape(x.split("|")[0].strip()), t[e:e+90], re.I):
                    return True
    return False
def f_auth(r):
    t = str(r.sentence)
    return any(len(x.split()) == 1 and x[:1].isupper()
               and re.search(re.escape(x)+r"\s*,?\s*(1[6-9]\d\d|20[0-2]\d)", t)
               for x in (str(r.species1), str(r.species2)))
def f_colist(r):
    t = str(r.sentence)
    for s1, e1 in sp(t, r.species1):
        for s2, e2 in sp(t, r.species2):
            lo, hi = (e1, s2) if s2 >= e1 else (e2, s1)
            if hi <= lo or hi-lo > 80: continue
            mid = t[lo:hi]
            if not re.fullmatch(r"[\sA-Za-z0-9.,;&'()\-‐-―]*", mid): continue
            toks = [w.strip(".,;()&'-").lower() for w in mid.split()]
            if any(any(w.startswith(b) for b in BREAK) for w in toks if w): continue
            if not re.search(r"[,;]|\band\b|\bor\b", mid, re.I): continue
            return True
    return False

FAM = {"synonym": f_syn, "ancestor": f_anc, "authority": f_auth, "co_participant(co-listed)": f_colist}
for k, f in FAM.items(): d[k] = d.apply(f, axis=1)

def mc(a_ok, b_ok):
    n01 = int((~a_ok & b_ok).sum()); n10 = int((a_ok & ~b_ok).sum())
    st = (abs(n01-n10)-1)**2/(n01+n10) if n01+n10 else 0.0
    return (1.0 if n01+n10 == 0 else float(chi2.sf(st, 1))), n01, n10, st

grid = np.arange(0.005, 1.0, 0.005)
def row(name, mask):
    f1s = [f1_score(y, ((V3 >= g) & ~mask).astype(int), zero_division=0) for g in grid]
    tv = [f1_score(y[hv], ((V3[hv] >= g) & ~mask[hv]).astype(int), zero_division=0) for g in grid]
    t2 = float(grid[int(np.argmax(tv))]); p2 = ((V3[hv] >= t2) & ~mask[hv]).astype(int)
    p, k, m, st = mc(v1[hv] == y[hv], p2 == y[hv])
    return dict(oracle=name, n_flagged=int(mask.sum()),
                flagged_gold_neg=int((y[mask] == 0).sum()) if mask.sum() else 0,
                f1_440=round(max(f1s), 4), f1_141=round(max(tv), 4),
                k_fixed=k, m_broken=m, chi2=round(st, 2), mcnemar_p=round(p, 4),
                beats_v1="YES" if p < 0.05 and k > m else "no")
out = [row("(none) V3 as-is", np.zeros(len(d), bool))]
allm = np.zeros(len(d), bool)
for k in FAM:
    mk = d[k].to_numpy(); allm |= mk
    out.append(row(f"perfect {k}", mk))
out.append(row("all four families combined", allm))
t = pd.DataFrame(out)
pd.set_option("display.width", 220)
print("ORACLE UPPER BOUNDS -- perfect detection of each targeted family, applied to V3")
print(f"V1 on its 141 rows: F1=0.8636.  Significance bar: chi2 > 3.84\n")
print(t.to_string(index=False))
t.to_csv(R/"oracle_bounds.csv", index=False)
print(f"\nwrote {R/'oracle_bounds.csv'}")
