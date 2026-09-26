#!/usr/bin/env python3
"""Pre-registered falsifier for the pair-key contradiction fix.

The species-relabel lever's post-mortem showed its contradiction fix (CFIX) lifted
target positives and gold negatives by the SAME absolute amount -- it had taught
global permissiveness, not pair-specific symmetry. This applies that same test here.

TARGET  : the 5 biotx100 positives the pair ensemble scores near zero (relation
          mismatch -- the class the fix is supposed to rescue).
CONTROL : the 4 reject50 negatives it already scores high (must NOT rise), and all
          gold negatives on the 440 (must not rise as a block).

A fix that is real moves TARGET much more than CONTROL. A prior shift moves both.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import eval_final_v4 as E

TARGET = [("Mycoplasma", "gut metagenome"), ("human", "Entamoeba histolytica"),
          ("Alternaria alternata", "Pear"), ("human", "Mycobacterium leprae"),
          ("Blastocystis", "human")]
CONTROL = [("wasps", "spiders"), ("nervous necrosis virus", "European sea bass"),
           ("Sitobion miscanthi", "aphid"), ("head lice", "anise")]


def mask_for(d, pairs):
    key = list(zip(d.species1.astype(str), d.species2.astype(str)))
    return np.array([k in pairs for k in key])


def compare(d, S_base, S_new, label=""):
    y = d.label.to_numpy()
    mt, mc = mask_for(d, set(TARGET)), mask_for(d, set(CONTROL))
    neg = (y == 0)
    rows = [("TARGET  5 biotx pos (want UP)", mt),
            ("CONTROL 4 reject neg (want FLAT)", mc),
            ("all gold negatives (want FLAT)", neg),
            ("all gold positives", (y == 1))]
    print(f"\n{label}  mean score shift")
    out = {}
    for nm, m in rows:
        a, b = S_base[m].mean(), S_new[m].mean()
        print(f"  {nm:<34} {a:.4f} -> {b:.4f}   delta {b-a:+.4f}  (n={int(m.sum())})")
        out[nm] = (float(a), float(b), float(b-a))
    dt = out["TARGET  5 biotx pos (want UP)"][2]
    dn = out["all gold negatives (want FLAT)"][2]
    ratio = dt/dn if abs(dn) > 1e-9 else float("inf")
    print(f"  --> target/negative shift ratio = {ratio:.2f}   "
          f"({'pair-specific' if ratio > 2 else 'GLOBAL PERMISSIVENESS -- falsified'})")
    return out, ratio
