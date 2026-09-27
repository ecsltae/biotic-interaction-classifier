#!/usr/bin/env python3
"""Direction accuracy, swap consistency and undirectedness AUC for a trained arm."""
import json, sys
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer
from train_direction_general import Model, DS, ENC
from sklearn.metrics import roc_auc_score


def scores(m, tok, df, rel2i, order, dev, bs=48):
    ds = DS(df, tok, rel2i, order=order)
    SD, SU = [], []
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            b = ds.collate([ds[j] for j in range(i, min(i+bs, len(ds)))])
            b = {k: v.to(dev) for k, v in b.items()}
            b.pop("y"); r = b.pop("r"); b.pop("d")
            sd, su, ok = m(b["input_ids"], b["attention_mask"], b.get("token_type_ids"), r=r)
            SD.extend(sd.float().cpu().numpy()); SU.extend(su.float().cpu().numpy())
    return np.array(SD), np.array(SU)


def evaluate(md, test_csv):
    cfg = json.load(open(f"{md}/config.json"))
    rel2i = {r: i for i, r in enumerate(cfg["rels"])}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENC)
    m = Model(len(cfg["rels"]), head=cfg["head"]).to(dev)
    m.load_state_dict(torch.load(f"{md}/model.pt", map_location=dev)); m.eval()
    te = pd.read_csv(test_csv)
    order = cfg["input"]

    sd, su = scores(m, tok, te, rel2i, order, dev)
    if order == "canonical":
        at = np.array([str(a).lower() <= str(b).lower() for a, b in zip(te.s1, te.s2)])
    else:
        at = np.ones(len(te), bool)
    p_s1 = np.where(at, 1/(1+np.exp(-sd)), 1-1/(1+np.exp(-sd)))
    pred = np.where(p_s1 >= 0.5, 1, 2)
    directed = te.directed.to_numpy() == 1
    acc = (pred[directed] == te.subject_is.to_numpy()[directed]).mean()

    sw = te.copy(); sw["s1"], sw["s2"] = te.s2.values, te.s1.values
    sw["subject_is"] = np.where(te.subject_is == 1, 2, np.where(te.subject_is == 2, 1, -1))
    sd2, _ = scores(m, tok, sw, rel2i, order, dev)
    if order == "canonical":
        at2 = np.array([str(a).lower() <= str(b).lower() for a, b in zip(sw.s1, sw.s2)])
    else:
        at2 = np.ones(len(sw), bool)
    p_sw = 1 - np.where(at2, 1/(1+np.exp(-sd2)), 1-1/(1+np.exp(-sd2)))
    cons = ((p_s1 >= 0.5) == (p_sw >= 0.5))

    auc = (roc_auc_score(directed.astype(int), 1/(1+np.exp(-su)))
           if (~directed).sum() and directed.sum() else float("nan"))
    return dict(arm=md, input=cfg["input"], head=cfg["head"], seed=cfg["seed"],
                n=int(directed.sum()), acc=float(acc),
                swap=float(cons[directed].mean()),
                maxdev=float(np.abs(p_s1 - p_sw).max()), auc=float(auc))


if __name__ == "__main__":
    test = sys.argv[1]; arms = sys.argv[2:]
    rows = [evaluate(a, test) for a in arms]
    print(f"\n{'arm':26} {'input':10} {'head':10} {'n':>6} {'dir acc':>8} "
          f"{'swap':>7} {'max dev':>10} {'und AUC':>8}")
    print("-"*92)
    for r in rows:
        print(f"{r['arm']:26} {r['input']:10} {r['head']:10} {r['n']:6d} {r['acc']:8.4f} "
              f"{r['swap']:7.4f} {r['maxdev']:10.2e} {r['auc']:8.4f}")
    pd.DataFrame(rows).to_csv("direction_arms_results.csv", index=False)
