#!/usr/bin/env python3
"""Score every direction arm on the human gold: direction accuracy + undirectedness + swap consistency.

Arms differ on two axes, one per BioREDirect-style design choice:
  input order   canonical (ours: alphabetical sort, byte-identical under swap)
                text      (theirs: stored order with role markers, so the input carries direction)
  head style    symmetric     (ours: antisymmetric direction + symmetric directedness)
                unconstrained (theirs: one 3-way softmax, no symmetry guarantee)

Swap consistency is the point of the whole design: feed the same pair the other way round and
the answer must mirror. Ours is 1.0 by algebra; an unconstrained head has to learn it, and the
measurement says how well it does.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"experiments"/"direction"))
sys.path.insert(0, str(REPO/"experiments"/"multitask"))
import xenc_format as X, polarity                                  # noqa: E402
from transformers import AutoTokenizer                             # noqa: E402
from scipy.stats import binomtest                                  # noqa: E402
from sklearn.metrics import roc_auc_score                          # noqa: E402


def pol3(r):
    p, _ = polarity.polarity(str(r))
    return 2 if (p is None or p == 0) else (1 if p > 0 else 0)


def load(md):
    cfg = json.loads((Path(md)/"student_config.json").read_text())
    style = cfg.get("head_style", "symmetric")
    if style == "unconstrained":
        from train_direction_ablate import Student
        m = Student(enc=str(md), detach_dir=cfg.get("detach_dir", False), head_style=style)
    else:
        from train_direction_v3 import Student
        m = Student(enc=str(md), detach_dir=cfg.get("detach_dir", False))
    m.dir.load_state_dict(torch.load(Path(md)/"direction_head.pt", map_location="cpu"))
    return m.eval(), AutoTokenizer.from_pretrained(md, local_files_only=True), cfg


@torch.no_grad()
def run(m, tok, s1, s2, rel, txt, fmt, dev, bs=16):
    a, b = X.build_many(fmt, list(s1), list(rel), list(s2), list(txt))
    pol = np.array([pol3(r) for r in rel], dtype="int64")
    SD, SU, OK = [], [], []
    for i in range(0, len(a), bs):
        e = tok(a[i:i+bs], b[i:i+bs], truncation="only_second", max_length=256,
                padding=True, return_tensors="pt").to(dev)
        p = torch.tensor(pol[i:i+bs], dtype=torch.long, device=dev)
        out = m(e["input_ids"], e["attention_mask"], e.get("token_type_ids"), pol=p)
        _, sd, su, ok = out
        SD.extend(sd.float().cpu().numpy()); SU.extend(su.float().cpu().numpy())
        OK.extend(ok.float().cpu().numpy())
    return np.array(SD), np.array(SU), np.array(OK)


def main():
    G = pd.read_csv(REPO/"data/evaluation/direction_gold_v2.csv")
    G = G[G.sentence.notna()].reset_index(drop=True)
    dec = G.decidable.astype(bool).to_numpy()
    und = (G.raw_code.astype(str) == "?").to_numpy()
    gold = G.gold.astype(str).to_numpy()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arms = [("joint_v3_s1", "canonical", "symmetric")]
    for tag, order, style in (("abl_text_uncon", "text", "unconstrained"),
                              ("abl_canon_uncon", "canonical", "unconstrained"),
                              ("abl_text_sym", "text", "symmetric")):
        if (REPO/f"models/dirhead/{tag}/student_config.json").exists():
            arms.append((tag, order, style))

    print(f"{'arm':18} {'input':10} {'head':14} {'dir acc':>9} {'p':>9} "
          f"{'undir AUC':>10} {'swap cons.':>11}")
    print("-" * 88)
    for tag, order, style in arms:
        md = REPO/f"models/dirhead/{tag}"
        m, tok, _ = load(md); m = m.to(dev)
        fmt = "mark_canon" if order == "canonical" else "triple_mark"

        sd, su, _ = run(m, tok, G.species1, G.species2, G.relation, G.sentence, fmt, dev)
        if order == "canonical":
            at = np.array([str(a).lower() <= str(b).lower() for a, b in zip(G.species1, G.species2)])
        else:
            at = np.ones(len(G), bool)
        p_s1 = np.where(at, 1/(1+np.exp(-sd)), 1-1/(1+np.exp(-sd)))
        pred = np.where(p_s1 >= 0.5, "FORWARD", "REVERSE")
        corr = (pred[dec] == gold[dec])
        k, n = int(corr.sum()), int(dec.sum())

        # swap: same pair, arguments exchanged. A consistent model must mirror its answer.
        sd2, su2, _ = run(m, tok, G.species2, G.species1, G.relation, G.sentence, fmt, dev)
        if order == "canonical":
            at2 = np.array([str(b).lower() <= str(a).lower() for a, b in zip(G.species1, G.species2)])
        else:
            at2 = np.ones(len(G), bool)
        p_s1_sw = 1 - np.where(at2, 1/(1+np.exp(-sd2)), 1-1/(1+np.exp(-sd2)))
        swap = float(((p_s1 >= 0.5) == (p_s1_sw >= 0.5)).mean())

        mask = dec | und
        auc = roc_auc_score(dec[mask].astype(int), (1/(1+np.exp(-su)))[mask]) if und.sum() else float("nan")
        print(f"{tag:18} {order:10} {style:14} {k:4d}/{n:<4d} {k/n:.3f} "
              f"{binomtest(k,n,0.5,alternative='greater').pvalue:9.1e} {auc:10.3f} {swap:11.3f}")
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print("\nswap consistency = fraction of pairs whose direction answer is unchanged when the two")
    print("arguments are exchanged. 1.000 by construction for the symmetric head; learned otherwise.")


if __name__ == "__main__":
    main()
