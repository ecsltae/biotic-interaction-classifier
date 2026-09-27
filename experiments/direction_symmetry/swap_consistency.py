#!/usr/bin/env python3
"""Swap consistency of the released BioREDirect model on its own benchmark.

The test: relabel Src<->Tgt on every entity marker, changing exactly which entity is
DESIGNATED the source, and nothing else. A model that understands direction rather than
memorising surface position must then flip Left_to_Right <-> Right_to_Left, and must leave
No_Direct and None alone.

This is not a re-implementation. It is the published artifact, on the published test set.
"""
import csv, sys
import numpy as np, pandas as pd

DIR = ["None", "Left_to_Right", "Right_to_Left", "No_Direct"]
MIRROR = {"Left_to_Right": "Right_to_Left", "Right_to_Left": "Left_to_Right",
          "No_Direct": "No_Direct", "None": "None"}


def preds(path):
    d = pd.read_csv(path, sep="\t")
    return d[DIR].to_numpy(), d


def main():
    import sys as _s
    swapped = _s.argv[1] if len(_s.argv) > 1 else "/tmp/biored_swapped.pred.tsv"
    a, da = preds("/tmp/biored_test.pred.tsv")
    b, db = preds(swapped)
    print(f"comparing against {swapped}\n")
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    gold = []
    for r in csv.reader(open("bioredirect/processed/test.tsv"), delimiter="\t",
                        quoting=csv.QUOTE_NONE):
        if len(r) < 9:
            gold.append(None); continue
        e1, e2, rel, subj = r[3], r[4], r[6], r[8]
        gold.append(None if rel in ("None", "") else
                    ("Left_to_Right" if subj == e1 else
                     "Right_to_Left" if subj == e2 else "No_Direct"))
    gold = np.array(gold[:n], dtype=object)

    pa = np.array([DIR[i] for i in a.argmax(1)])
    pb = np.array([DIR[i] for i in b.argmax(1)])
    want = np.array([MIRROR[x] for x in pa])

    has_rel = np.array([g is not None for g in gold])
    directed = np.array([g in ("Left_to_Right", "Right_to_Left") for g in gold])

    def report(mask, name):
        if mask.sum() == 0:
            return
        cons = (pb[mask] == want[mask]).mean()
        print(f"  {name:44} n={mask.sum():5d}   swap consistency {cons:.3f}")

    print("SWAP CONSISTENCY of the released BioREDirect model, BioRED test\n")
    report(np.ones(n, bool), "all candidate pairs")
    report(has_rel, "pairs with a gold relation")
    report(directed, "pairs with a gold DIRECTION (L2R or R2L)")

    print("\n  what the model does to a directed prediction when the roles are swapped:")
    sub = directed & np.isin(pa, ["Left_to_Right", "Right_to_Left"])
    if sub.sum():
        flipped = (pb[sub] == want[sub]).sum()
        same = (pb[sub] == pa[sub]).sum()
        other = sub.sum() - flipped - same
        print(f"    correctly mirrored : {flipped:5d} / {sub.sum()}  ({flipped/sub.sum():.1%})")
        print(f"    unchanged (contradicts itself) : {same:5d}  ({same/sub.sum():.1%})")
        print(f"    moved to None/No_Direct        : {other:5d}  ({other/sub.sum():.1%})")

    acc_a = (pa[directed] == gold[directed]).mean()
    print(f"\n  direction accuracy on the {directed.sum()} gold-directed pairs: {acc_a:.3f}")


if __name__ == "__main__":
    main()
