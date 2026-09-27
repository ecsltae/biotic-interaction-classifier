#!/usr/bin/env python3
"""McNemar between our canonical-symmetric arm and the order-sensitive arm on BioRED.

Each arm is summarised by the majority vote of its three seeds before the test.
"""
import numpy as np, pandas as pd, torch
from scipy.stats import chi2
from transformers import AutoTokenizer

import train_ours_on_biored as OURS
import train_biored_ordersens as ORDS
import eval_ours_on_biored as EO
import eval_ordersens as EORD


def arm_preds(mod, ev, md, te, canonical):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(mod.ENC)
    m = mod.Model().to(dev)
    m.load_state_dict(torch.load(f"{md}/model.pt", map_location=dev)); m.eval()
    sd, _, _, _ = ev.scores(m, tok, te, dev)
    if canonical:
        at = np.array([str(a).lower() <= str(b).lower() for a, b in zip(te.s1, te.s2)])
    else:
        at = np.ones(len(te), bool)
    p_s1 = np.where(at, 1/(1+np.exp(-sd)), 1-1/(1+np.exp(-sd)))
    return np.where(p_s1 >= 0.5, 1, 2)


te = pd.read_csv("biored_test_ours.csv")
A = np.array([arm_preds(OURS, EO,   f"ours_on_biored_s{s}",   te, True)  for s in (1, 2, 3)])
B = np.array([arm_preds(ORDS, EORD, f"biored_ordersens_s{s}", te, False) for s in (1, 2, 3)])
maj = lambda v: np.where(v.mean(0) <= 1.5, 1, 2)
directed = te.directed.to_numpy() == 1
gold = te.subject_is.to_numpy()
ca = (maj(A) == gold)[directed]; cb = (maj(B) == gold)[directed]
b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
p = 1.0 if b + c == 0 else float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
print(f"\n  n directed                  : {directed.sum():,}")
print(f"  ours (canonical+sym) acc    : {ca.mean():.4f}")
print(f"  order-sensitive uncon. acc  : {cb.mean():.4f}")
print(f"  discordant: ours-only={b}, uncon-only={c}")
print(f"  McNemar (continuity-corrected) p = {p:.4f}")
