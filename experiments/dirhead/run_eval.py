#!/usr/bin/env python3
"""Full evaluation of a joint binary+direction checkpoint. Writes results/dirhead/<tag>.json."""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "scripts"))
import dirlib, eval_dirhead as E
import eval_unified


def dir_metrics(P, truth, name, ok=None):
    """P[n,3] over STORED order (p_subj1, p_subj2, p_nodir); truth in {1,2,nan}."""
    m = np.array([t in (1, 2) for t in truth])
    P, truth = P[m], np.array(truth)[m].astype(int)
    if ok is not None:
        ok = np.array(ok)[m]
    pred = np.where(P[:, 0] >= P[:, 1], 1, 2)
    corr = (pred == truth)
    conf = np.abs(P[:, 0] - P[:, 1]) / np.clip(P[:, 0] + P[:, 1], 1e-9, None)
    lo, hi = dirlib.wilson(int(corr.sum()), len(corr))
    out = dict(name=name, n=int(len(corr)), correct=int(corr.sum()),
               acc=float(corr.mean()), lo=float(lo), hi=float(hi),
               p_vs_chance=float(binomtest(int(corr.sum()), len(corr), 0.5,
                                           alternative="greater").pvalue),
               mean_nodir=float(P[:, 2].mean()),
               frac_nodir_argmax=float((P.argmax(1) == 2).mean()))
    if len(corr) >= 8:
        cov = [0.25, 0.5, 0.6, 0.75, 0.9, 1.0]
        rc = []
        order = np.argsort(-conf)
        c = corr[order]
        for f in cov:
            k = max(1, int(round(f * len(c))))
            l, h = dirlib.wilson(int(c[:k].sum()), k)
            rc.append(dict(coverage=round(k / len(c), 3), n=k, acc=float(c[:k].mean()),
                           lo=float(l), hi=float(h), thr=float(conf[order][k - 1])))
        out["risk_coverage"] = rc
    return out, pred, corr


def mcnemar_exact(a_ok, b_ok):
    a_ok, b_ok = np.asarray(a_ok), np.asarray(b_ok)
    n01 = int((~a_ok & b_ok).sum()); n10 = int((a_ok & ~b_ok).sum())
    if n01 + n10 == 0:
        return 1.0, n10, n01
    p = binomtest(n10, n01 + n10, 0.5).pvalue
    return float(p), n10, n01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skip-binary", action="store_true")
    a = ap.parse_args()
    (REPO / "results/dirhead").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, tok, cfg = E.load(REPO / a.model, dev)
    R = dict(model=a.model, tag=a.tag, config={k: cfg[k] for k in
             ("dir_source", "split", "epochs", "alpha", "names_only", "no_binary",
              "no_direction", "seed", "max_fr", "max_sym") if k in cfg})

    # ---------------- human gold
    g = E.gold_frame()
    _, PD, OK = E.score(m, tok, g, dev)
    R["gold"] = {}
    o, pred, corr = dir_metrics(PD, g.subj, "direction head (full passage)")
    R["gold"]["head"] = o
    for lab, kw in (("names_only", dict(names_only=True)),
                    ("relation_masked", dict(mask_relation=True))):
        _, P2, _ = E.score(m, tok, g, dev, **kw)
        R["gold"][lab] = dir_metrics(P2, g.subj, f"head, {lab}")[0]
    # baselines on the same items
    base = {}
    bl_ok = {}
    for nm, fn in (("stored_order", lambda r: 1),
                   ("text_order", dirlib.base_textorder),
                   ("syntactic_rule", dirlib.base_syntactic)):
        pr = [fn(r) for _, r in g.rename(columns={"text": "sentence", "s1": "species1",
                                                  "s2": "species2", "rel": "relation"}).iterrows()]
        base[nm] = dirlib.report(nm, pr, g.subj)
        dm = [t in (1, 2) for t in g.subj]
        bl_ok[nm] = np.array([p == t for p, t in zip(np.array(pr, dtype=object)[dm],
                                                     np.array(g.subj)[dm])])
    R["gold"]["baselines"] = base
    R["gold"]["mcnemar_vs"] = {}
    for nm, ok in bl_ok.items():
        p, w, l = mcnemar_exact(ok, corr)
        R["gold"]["mcnemar_vs"][nm] = dict(p=p, baseline_only_right=w, head_only_right=l)

    # ---------------- silver, taxon-disjoint and seen-taxa
    md = REPO / a.model
    for slice_name, f in (("taxon_disjoint_test", "dir_test.parquet"),
                          ("seen_taxa_test", "dir_test_seen.parquet")):
        p = md / f
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if len(d) > 4000:
            d = d.sample(4000, random_state=0).reset_index(drop=True)
        d = d.rename(columns={"text": "text"})
        truth = np.where(d.y_dir == 0, 1, np.where(d.y_dir == 1, 2, np.nan))
        # y_dir is over canonical order; convert to stored order
        a_is_s1 = np.array([str(x).lower() <= str(y).lower() for x, y in zip(d.s1, d.s2)])
        truth = np.where(np.isnan(truth), np.nan, np.where(a_is_s1, truth, 3 - truth))
        _, P, _ = E.score(m, tok, d, dev)
        R[slice_name] = dict(head=dir_metrics(P, truth, slice_name)[0])
        for lab, kw in (("names_only", dict(names_only=True)),
                        ("relation_masked", dict(mask_relation=True))):
            _, P2, _ = E.score(m, tok, d, dev, **kw)
            R[slice_name][lab] = dir_metrics(P2, truth, f"{slice_name} {lab}")[0]
        # 3-way including the no-direction class
        y3 = d.y_dir.to_numpy()
        _, P3, _ = E.score(m, tok, d, dev)
        # rebuild canonical-order posterior for the 3-way call
        pc = np.stack([np.where(a_is_s1, P3[:, 0], P3[:, 1]),
                       np.where(a_is_s1, P3[:, 1], P3[:, 0]), P3[:, 2]], 1)
        R[slice_name]["three_way_acc"] = float((pc.argmax(1) == y3).mean())
        R[slice_name]["three_way_n"] = int(len(y3))
        R[slice_name]["nodir_recall"] = float((pc[y3 == 2].argmax(1) == 2).mean()) if (y3 == 2).any() else None

    # ---------------- binary task
    if not a.skip_binary:
        d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
        d = d[~d.in_train].reset_index(drop=True)
        dd = d.rename(columns={"sentence": "text", "species1": "s1", "species2": "s2",
                               "relation": "rel"})
        PB, _, _ = E.score(m, tok, dd, dev)
        o, S, _ = eval_unified.evaluate([], a.tag, scores=PB)
        R["binary_unified"] = {k: v for k, v in o.items() if k != "by_threshold"}
        np.save(REPO / f"results/dirhead/{a.tag}_binary_scores.npy", PB)

    (REPO / "results/dirhead").mkdir(parents=True, exist_ok=True)
    (REPO / f"results/dirhead/{a.tag}.json").write_text(json.dumps(R, indent=2, default=float))
    print(json.dumps(R, indent=2, default=float))


if __name__ == "__main__":
    main()
