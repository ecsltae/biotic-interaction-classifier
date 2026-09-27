#!/usr/bin/env python3
"""McNemar (continuity-corrected) between the canonical-symmetric and unconstrained arms.

Per-item correctness is taken from the majority vote over the three seeds of each arm, so the
test reflects the arm rather than one lucky initialisation.
"""
import sys, json
import numpy as np, pandas as pd, torch
from scipy.stats import chi2
from transformers import AutoTokenizer


def preds(md, test_csv, mod):
    cfg = json.load(open(f"{md}/config.json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(mod.ENC)
    rel2i = {r: i for i, r in enumerate(cfg["rels"])}
    m = mod.Model(len(cfg["rels"]), head=cfg["head"]).to(dev)
    m.load_state_dict(torch.load(f"{md}/model.pt", map_location=dev)); m.eval()
    te = pd.read_csv(test_csv)
    import eval_direction_general as E
    sd, _ = E.scores(m, tok, te, rel2i, cfg["input"], dev)
    if cfg["input"] == "canonical":
        at = np.array([str(a).lower() <= str(b).lower() for a, b in zip(te.s1, te.s2)])
    else:
        at = np.ones(len(te), bool)
    p_s1 = np.where(at, 1/(1+np.exp(-sd)), 1-1/(1+np.exp(-sd)))
    return np.where(p_s1 >= 0.5, 1, 2), te


def mcnemar(c_a, c_b):
    b = int((c_a & ~c_b).sum()); c = int((~c_a & c_b).sum())
    if b + c == 0:
        return b, c, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(chi2.sf(stat, 1))


if __name__ == "__main__":
    test, pa, pb = sys.argv[1], sys.argv[2], sys.argv[3]   # prefixes, seeds 1..3 appended
    import train_direction_general as mod
    out = {}
    for tag, pre in (("A", pa), ("B", pb)):
        P = []
        for s in (1, 2, 3):
            pr, te = preds(f"{pre}{s}", test, mod)
            P.append(pr)
        out[tag] = np.array(P)
    directed = te.directed.to_numpy() == 1
    gold = te.subject_is.to_numpy()
    # majority vote over seeds (labels are 1/2, so the mean side decides)
    maj = {t: np.where(v.mean(0) <= 1.5, 1, 2) for t, v in out.items()}
    ca = (maj["A"] == gold)[directed]; cb = (maj["B"] == gold)[directed]
    b, c, p = mcnemar(ca, cb)
    print(f"\n  n directed              : {directed.sum():,}")
    print(f"  A = {pa}* acc : {ca.mean():.4f}")
    print(f"  B = {pb}* acc : {cb.mean():.4f}")
    print(f"  discordant: A-only-right={b}, B-only-right={c}")
    print(f"  McNemar (continuity-corrected) p = {p:.4f}")
