#!/usr/bin/env python3
"""Our architecture on BioRED test: direction accuracy and swap consistency.

Head to head against the released BioREDirect model on the same 1,163 rows:
  their direction accuracy on gold-directed pairs : 0.766
  their swap consistency on those pairs           : 0.700
"""
import json, sys
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer
from train_ours_on_biored import Model, DS, ENC, R2I, RELS
from sklearn.metrics import roc_auc_score

def scores(m, tok, df, dev, bs=32):
    ds = DS(df, tok)
    SD, SU, OK = [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            b = ds.collate([ds[j] for j in range(i, min(i+bs, len(ds)))])
            b = {k: v.to(dev) for k, v in b.items()}
            y, r, d = b.pop("y"), b.pop("r"), b.pop("d")
            sd, su, ok = m(b["input_ids"], b["attention_mask"], b.get("token_type_ids"), r=r)
            SD.extend(sd.float().cpu().numpy()); SU.extend(su.float().cpu().numpy())
            OK.extend(ok.float().cpu().numpy())
    return np.array(SD), np.array(SU), np.array(OK), ds

def main(md="ours_on_biored_s1"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENC)
    m = Model().to(dev); m.load_state_dict(torch.load(f"{md}/model.pt", map_location=dev)); m.eval()
    te = pd.read_csv("biored_test_ours.csv")

    sd, su, ok, ds = scores(m, tok, te, dev)
    at = np.array([str(a).lower() <= str(b).lower() for a, b in zip(te.s1, te.s2)])
    p_at = 1/(1+np.exp(-sd)); p_s1 = np.where(at, p_at, 1-p_at)
    pred = np.where(p_s1 >= 0.5, 1, 2)
    directed = (te.directed.to_numpy() == 1)
    truth = te.subject_is.to_numpy()

    acc = (pred[directed] == truth[directed]).mean()
    print(f"OUR architecture on BioRED test ({len(te):,} rows)\n")
    print(f"  direction accuracy on {int(directed.sum()):,} gold-directed pairs : {acc:.4f}")
    print(f"    (released BioREDirect on the same pairs                : 0.7660)")

    # swap: exchange the two entities. Canonical sorting should make this a no-op.
    sw = te.copy()
    sw["s1"], sw["s2"] = te.s2.values, te.s1.values
    sw["subject_is"] = np.where(te.subject_is == 1, 2, np.where(te.subject_is == 2, 1, -1))
    sd2, su2, ok2, _ = scores(m, tok, sw, dev)
    at2 = np.array([str(a).lower() <= str(b).lower() for a, b in zip(sw.s1, sw.s2)])
    p_at2 = 1/(1+np.exp(-sd2)); p_s1_sw = 1 - np.where(at2, p_at2, 1-p_at2)
    same = ((p_s1 >= 0.5) == (p_s1_sw >= 0.5))
    print(f"\n  swap consistency, all pairs            : {same.mean():.4f}")
    print(f"  swap consistency, gold-directed pairs  : {same[directed].mean():.4f}")
    print(f"    (released BioREDirect                              : 0.7000)")
    print(f"  max |p(s1 subject) - mirrored|         : {np.abs(p_s1 - p_s1_sw).max():.3e}")

    und = ~directed
    if und.sum():
        p_dir = 1/(1+np.exp(-su))
        print(f"\n  directedness head: AUC separating {int(directed.sum()):,} directed from "
              f"{int(und.sum()):,} undirected : {roc_auc_score(directed.astype(int), p_dir):.4f}")

if __name__ == "__main__":
    main(*sys.argv[1:])
