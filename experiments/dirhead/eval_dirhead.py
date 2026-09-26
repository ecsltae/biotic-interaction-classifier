#!/usr/bin/env python3
"""Score and evaluate a joint binary+direction checkpoint."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "scripts"))
import fmt, data as D, dirlib
from model import DirModel


def load(md, device="cuda"):
    md = Path(md)
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    cfg = json.loads((md / "student_config.json").read_text())
    m = DirModel(cfg["encoder"])
    m.load_state_dict(torch.load(md / "pytorch_model.bin", map_location="cpu"))
    return m.to(device).eval(), tok, cfg


def score(m, tok, df, device="cuda", bs=64, names_only=False, mask_relation=False,
          anon=False, max_len=256):
    """df needs s1, s2, rel, text.  Returns (p_bin, P_dir[n,3], subj_pred over STORED order)."""
    d = df.copy()
    d["y_bin"], d["y_dir"] = -100, -100
    ds = D.Joint(d, tok, max_len, augment=False, names_only=names_only,
                 mask_relation=mask_relation, anon=anon)
    cf = D.collate(tok)
    PB, PD, OK = [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            b = cf([ds[j] for j in range(i, min(i + bs, len(ds)))])
            ok = b.pop("ok"); b.pop("y_bin"); b.pop("y_dir")
            b = {k: v.to(device) for k, v in b.items()}
            bl, dl = m(**b)
            PB.append(torch.softmax(bl.float(), -1)[:, 1].cpu().numpy())
            PD.append(torch.softmax(dl.float(), -1).cpu().numpy())
            OK.append(ok.numpy())
    PB, PD, OK = np.concatenate(PB), np.concatenate(PD), np.concatenate(OK)
    a_is_s1 = np.array([str(x).lower() <= str(y).lower() for x, y in zip(d.s1, d.s2)])
    # class 0 = alphabetically-first span is the subject
    p1 = np.where(a_is_s1, PD[:, 0], PD[:, 1])
    p2 = np.where(a_is_s1, PD[:, 1], PD[:, 0])
    return PB, np.stack([p1, p2, PD[:, 2]], 1), OK


def gold_frame():
    g = dirlib.load_gold()
    g = g.rename(columns={"sentence": "text", "species1": "s1", "species2": "s2",
                          "relation": "rel"})
    return g


def risk_coverage(conf, correct, grid=None):
    order = np.argsort(-conf)
    c = np.array(correct)[order]
    out = []
    for k in (grid or range(1, len(c) + 1)):
        out.append(dict(n=k, coverage=k / len(c), acc=float(c[:k].mean()),
                        thr=float(conf[order][k - 1])))
    return out
