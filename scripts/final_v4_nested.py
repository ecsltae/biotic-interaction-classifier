#!/usr/bin/env python3
"""Nested cross-fit: the ensemble config AND its threshold are both chosen inside
the training fold, so the reported decisions survive config-selection leakage.

Outer 5-fold stratified on source x label. Inside each training fold we pick the
(config, threshold) pair maximising F1 on that fold only, then apply it to the
held-out fold. Repeated over R fold seeds. The resulting out-of-fold decision
vector is what the McNemar tests use.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import eval_final_v4 as E

GRID = np.arange(0.005, 1.0, 0.005)


def nested(configs, y, strat, R=20, seed0=0):
    """configs: {name: score_vector}. Returns (preds RxN, chosen config names per fold)."""
    names = list(configs)
    M = np.stack([configs[n] for n in names])       # C x N
    preds, picks = [], []
    for r in range(R):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed0 + r)
        p = np.zeros(len(y), int)
        for tr, te in skf.split(y.reshape(-1, 1), strat):
            best = (-1, 0, 0.5)
            for ci in range(len(names)):
                s = M[ci]
                for g in GRID:
                    f = f1_score(y[tr], (s[tr] >= g).astype(int), zero_division=0)
                    if f > best[0]:
                        best = (f, ci, g)
            _, ci, g = best
            p[te] = (M[ci][te] >= g).astype(int)
            picks.append(names[ci])
        preds.append(p)
    return np.array(preds), picks
