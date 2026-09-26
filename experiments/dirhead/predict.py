#!/usr/bin/env python3
"""Deployment entry point: one pass, binary verdict + direction, CPU-friendly.

    python experiments/dirhead/predict.py --model models/dirhead/joint_cons_s1 \
        --in candidates.csv --out scored.csv          # needs s1,s2,rel,text

Output contract, one row per candidate, every field always present:
    interacts, interacts_score          the existing species-level verdict (unchanged semantics)
    direction   a->b | b->a | no_direction | abstain
    direction_score                     P(the emitted arrow) or null
    subject, object                     the taxa, or null when no arrow is emitted
The two taxa are written in canonical (alphabetical) order, because the stored order in
the upstream pipeline is a text-position artefact and carries no meaning.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_dirhead as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir-threshold", type=float, default=0.90,
                    help="abstain below this |P(fwd)-P(rev)| margin")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = pd.read_csv(a.inp)
    m, tok, cfg = E.load(a.model, device="cpu")
    PB, PD, OK = E.score(m, tok, d, device="cpu")
    margin = np.abs(PD[:, 0] - PD[:, 1]) / np.clip(PD[:, 0] + PD[:, 1], 1e-9, None)
    nodir = PD.argmax(1) == 2
    fwd = PD[:, 0] >= PD[:, 1]
    lab = np.where(~OK, "abstain",
          np.where(nodir, "no_direction",
          np.where(margin < a.dir_threshold, "abstain",
          np.where(fwd, "s1->s2", "s2->s1"))))
    out = d.copy()
    out["interacts_score"] = PB
    out["interacts"] = (PB >= cfg.get("threshold_dev", 0.5)).astype(int)
    out["direction"] = lab
    out["direction_score"] = np.where(np.isin(lab, ["s1->s2", "s2->s1"]),
                                      np.maximum(PD[:, 0], PD[:, 1]), np.nan)
    out["subject"] = np.where(lab == "s1->s2", d.s1, np.where(lab == "s2->s1", d.s2, None))
    out["object"] = np.where(lab == "s1->s2", d.s2, np.where(lab == "s2->s1", d.s1, None))
    out.to_csv(a.out, index=False)
    print(out.direction.value_counts().to_dict())


if __name__ == "__main__":
    main()
