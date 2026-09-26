#!/usr/bin/env python3
"""Evaluate an antisymmetric joint binary+direction checkpoint from train_direction.py.

train_direction.py saves the encoder with save_pretrained (model.safetensors) and the
direction head separately (direction_head.pt). experiments/dirhead/eval_dirhead.py cannot
load that layout -- it hardcodes pytorch_model.bin and builds the older 3-way DirModel --
which is why detach_s1 and joint_a05_s1 were trained but never scored.

Reports both tasks:
  binary    -- unified test set, mark_canon input, AUPRC + oracle-best F1 + McNemar vs V1
  direction -- the 20-row human gold (17 decidable), accuracy with Wilson CI and an exact
               binomial test against chance

The binary threshold here is NOT dev-derived: train_direction.py trains on 100% of
v4_species_train and hardcodes threshold_dev=0.5. The prior-shift threshold is reported
as the principled operating point; oracle F1 is a labelled upper bound, not achievable.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

REPO = Path("/home/egaillac/MetaP/classifier")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "dirhead"))
sys.path.insert(0, str(REPO / "experiments" / "direction"))
sys.path.insert(0, str(REPO / "experiments" / "multitask"))
import xenc_format as X                                     # noqa: E402
from train_direction import Student, AT_ID, HASH_ID, span_pool  # noqa: E402
from transformers import AutoTokenizer                      # noqa: E402
import eval_unified as EU                                   # noqa: E402


def load(md, dev):
    md = Path(md)
    cfg = json.loads((md / "student_config.json").read_text())
    m = Student(enc=str(md), detach_dir=cfg.get("detach_dir", False))
    m.dir.load_state_dict(torch.load(md / "direction_head.pt", map_location="cpu"))
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    return m.to(dev).eval(), tok, cfg


@torch.no_grad()
def run(m, tok, s1, s2, rel, text, pol=None, dev="cuda", bs=32, max_len=256):
    """Returns (p_interact, s_dir, markers_ok). s_dir is the logit that the @ taxon is subject."""
    a, b = X.build_many("mark_canon", list(s1), list(rel), list(s2), [str(t) for t in text])
    PB, SD, OK = [], [], []
    for i in range(0, len(a), bs):
        e = tok(a[i:i+bs], b[i:i+bs], truncation="only_second", max_length=max_len,
                padding=True, return_tensors="pt").to(dev)
        p = None
        if pol is not None:
            p = torch.tensor(np.asarray(pol[i:i+bs]), dtype=torch.long, device=dev)
        logits, s, ok = m(e["input_ids"], e["attention_mask"], e.get("token_type_ids"), pol=p)
        PB.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        SD.extend((s.float().cpu().numpy() if s is not None else [np.nan] * len(logits)))
        OK.extend(ok.float().cpu().numpy())
    return np.array(PB), np.array(SD), np.array(OK)


def wilson(k, n):
    from scipy.stats import norm
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(0.975); p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def direction_eval(m, tok, dev):
    import polarity
    g = pd.read_csv(REPO / "data/evaluation/direction_gold_combined.csv")
    g = g[g.decidable.astype(bool) & g.gold.isin(["FORWARD", "REVERSE"])].reset_index(drop=True)

    pol_raw, pol_src = [], []
    for r, ro in zip(g.relation, g.get("robi_id", pd.Series([None] * len(g)))):
        try:
            a, b = polarity.polarity(str(r), ro if isinstance(ro, str) else None)
        except Exception:
            a, b = None, "error"
        pol_raw.append(np.nan if a is None else float(a)); pol_src.append(b)
    pol_raw = np.array(pol_raw, dtype=float)
    covered = np.isfinite(pol_raw) & (pol_raw != 0)
    pol = np.where(pol_raw > 0, 1, 0).astype("int64")   # undefined -> 0; flagged via `covered`

    _, s, ok = run(m, tok, g.species1, g.species2, g.relation, g.sentence, pol=pol, dev=dev)

    at_is_s1 = np.array([str(x).lower() <= str(y).lower() for x, y in zip(g.species1, g.species2)])
    p_at_subj = 1 / (1 + np.exp(-s))
    p_s1_subj = np.where(at_is_s1, p_at_subj, 1 - p_at_subj)
    pred = np.where(p_s1_subj >= 0.5, "FORWARD", "REVERSE")
    corr = (pred == g.gold.to_numpy())

    def blk(mask, name):
        n = int(mask.sum()); k = int(corr[mask].sum())
        lo, hi = wilson(k, n)
        p = float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else 1.0
        return dict(name=name, n=n, correct=k, acc=(k / n if n else None),
                    ci95=[lo, hi], p_vs_chance=p)

    return {
        "all_decidable": blk(np.ones(len(g), bool), "all decidable"),
        "polarity_covered": blk(covered, "relation polarity in lexicon"),
        "polarity_missing": blk(~covered, "relation polarity NOT in lexicon"),
        "n_polarity_covered": int(covered.sum()),
        "markers_found_frac": float((ok > 0).mean()),
        "items": [dict(item=int(i), s1=str(a), rel=str(r), s2=str(b), gold=str(gd),
                       pred=str(pd_), correct=bool(c), p_s1_subject=float(ps),
                       polarity=(None if not np.isfinite(pr) else float(pr)), polarity_src=str(src),
                       markers_ok=bool(o > 0))
                  for i, a, r, b, gd, pd_, c, ps, pr, src, o in zip(
                      g.item_id, g.species1, g.relation, g.species2, g.gold, pred, corr,
                      p_s1_subj, pol_raw, pol_src, ok)],
    }


def binary_eval(m, tok, dev, cfg):
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    d = d[~d.in_train].reset_index(drop=True)
    y = d.label.to_numpy()
    S, _, _ = run(m, tok, d.species1, d.species2, d.relation, d.sentence, pol=None, dev=dev)

    train_pos = pd.read_csv(cfg["bin_data"], usecols=["label"]).label.mean()

    # 0.5 is the ONLY pre-specified operating point here. A prior-shift threshold would
    # need the DEPLOYMENT prior; deriving it from y.mean() would read the reporting set's
    # own gold, which is the leak this harness exists to prevent. Not computed.
    out = {"n": int(len(d)), "prevalence": float(y.mean()),
           "auprc": float(average_precision_score(y, S)),
           "train_pos_rate": float(train_pos),
           "threshold_policy": "0.5 pre-specified; no threshold is fitted on this set"}
    best = max(((f1_score(y, (S >= t).astype(int), zero_division=0), float(t))
                for t in np.arange(0.01, 1.0, 0.01)))
    out["oracle_best_f1"], out["oracle_thr"] = float(best[0]), best[1]
    out["_oracle_note"] = "labelled upper bound, threshold swept on the reporting set; NOT achievable"
    for nm, t in (("at_0.5", 0.5),):
        pr = (S >= t).astype(int)
        out[nm] = dict(thr=float(t), P=float(precision_score(y, pr, zero_division=0)),
                       R=float(recall_score(y, pr, zero_division=0)),
                       F1=float(f1_score(y, pr, zero_division=0)))
    hv = d.v1.notna().to_numpy()
    if hv.sum():
        v1 = d.v1.fillna(0).to_numpy().astype(int)[hv]; yv = y[hv]
        cmp = []
        for t in np.arange(0.01, 1.0, 0.01):
            pr = (S[hv] >= t).astype(int)
            p, n01, n10 = EU.mcnemar(v1, pr, yv)
            cmp.append(dict(thr=float(t), f1=float(f1_score(yv, pr, zero_division=0)),
                            precision=float(precision_score(yv, pr, zero_division=0)),
                            recall=float(recall_score(yv, pr, zero_division=0)),
                            mcnemar_p=p, v1_only_right=n10, model_only_right=n01))
        out["v1_subset_n"] = int(hv.sum())
        out["v1_f1"] = float(f1_score(yv, v1, zero_division=0))
        out["vs_v1_top5_by_f1"] = sorted(cmp, key=lambda c: -c["f1"])[:5]
        at05 = [c for c in cmp if abs(c["thr"] - 0.5) < 1e-9][0]
        out["vs_v1_prespecified"] = at05          # the only p that may be quoted as a p-value
        # min over the grid is threshold-shopping across 99 correlated tests; kept for
        # diagnosis only, under a name that cannot be mistaken for a p-value.
        out["_vs_v1_min_p_oracle_selected"] = min(c["mcnemar_p"] for c in cmp)
        out["_vs_v1_n_thresholds_p_lt_05"] = sum(1 for c in cmp if c["mcnemar_p"] < 0.05)
    return out, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skip-binary", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, tok, cfg = load(a.model, dev)
    R = {"model": a.model, "tag": a.tag, "config": cfg}
    R["direction"] = direction_eval(m, tok, dev)
    if not a.skip_binary:
        R["binary"], S = binary_eval(m, tok, dev, cfg)
        np.save(REPO / f"results/dirhead/{a.tag}_binary_scores.npy", S)
    outp = REPO / f"results/dirhead/{a.tag}.json"
    outp.write_text(json.dumps(R, indent=2, default=float))
    dd = R["direction"]["all_decidable"]
    print(f'[{a.tag}] direction {dd["correct"]}/{dd["n"]} = {dd["acc"]:.3f} '
          f'(p={dd["p_vs_chance"]:.4f})')
    if not a.skip_binary:
        b = R["binary"]
        print(f'[{a.tag}] binary AUPRC {b["auprc"]:.4f}  oracleF1 {b["oracle_best_f1"]:.4f}  '
              f'F1@0.5 {b["at_0.5"]["F1"]:.4f}  McNemar@0.5 p={b["vs_v1_prespecified"]["mcnemar_p"]:.4f}')
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
