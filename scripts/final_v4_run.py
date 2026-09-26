#!/usr/bin/env python3
"""Produce every number in the final report, in one pass.

Threshold policy, stated once and applied everywhere: cross-fitted. A 5-fold
stratified split on source x label picks the threshold on 4/5 of the rows and
applies it to the held-out 5th, repeated over 20 fold seeds, so no reported
decision was made by a threshold that saw its own row's label. The oracle
(test-maximised) F1 is printed alongside ONLY so the gap is visible; it is never
the basis of a claim. The checkpoint's own dev-derived threshold is also printed,
as the fully test-blind alternative.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import eval_final_v4 as E, final_v4_nested as N, final_v4_mechanism as M

REPO = Path(__file__).resolve().parents[1]
R = 20
S3 = (1, 2, 3)

SINGLE = {
    "V2 (triple)":            [f"models/student/xenc_s{i}" for i in S3],
    "V3 (triple)":            [f"models/student_v3/xenc_s{i}" for i in S3],
    "PAIR":                   [f"models/lever_arch/fmt_pair_s{i}" for i in S3],
    "PAIRMARK":               [f"models/lever_arch/fmt_pair_mark_s{i}" for i in S3],
    "CTLW  prior control":    [f"models/final_v4/ctlw_s{i}" for i in S3],
    "CMAX  contra->1":        [f"models/final_v4/cmax_s{i}" for i in S3],
    "CDROP contra removed":   [f"models/final_v4/cdrop_s{i}" for i in S3],
    "CCOLL collapsed":        [f"models/final_v4/ccoll_s{i}" for i in S3],
    "CCOLLTGT coll+targeted": [f"models/final_v4/ccolltgt_s{i}" for i in S3],
    "CANONCOLL":              [f"models/final_v4/canoncoll_s{i}" for i in S3],
    "CANONCOLLTGT":           [f"models/final_v4/canoncolltgt_s{i}" for i in S3],
    "CANONRAW  canon, no fix": [f"models/final_v4/canonraw_s{i}" for i in S3],
    "MARKCANON canon+markers": [f"models/final_v4/markcanon_s{i}" for i in S3],
}
ENSEMBLES = {
    "ENS6 PAIR+PAIRMARK":  SINGLE["PAIR"] + SINGLE["PAIRMARK"],
    "ENS PAIR+CDROP":      SINGLE["PAIR"] + SINGLE["CDROP contra removed"],
    "ENS PAIR+CCOLL":      SINGLE["PAIR"] + SINGLE["CCOLL collapsed"],
    "ENS6+CDROP":          SINGLE["PAIR"] + SINGLE["PAIRMARK"] + SINGLE["CDROP contra removed"],
    "ENS6+CANONCOLL":      SINGLE["PAIR"] + SINGLE["PAIRMARK"] + SINGLE["CANONCOLL"],
    "ENS ALLPAIR":         SINGLE["PAIR"] + SINGLE["PAIRMARK"] + SINGLE["CDROP contra removed"]
                           + SINGLE["CCOLL collapsed"] + SINGLE["CANONCOLL"],
    "ENS PAIR+MARKCANON":  SINGLE["PAIR"] + SINGLE["MARKCANON canon+markers"],
    "ENS PAIR+CANONRAW":   SINGLE["PAIR"] + SINGLE["CANONRAW  canon, no fix"],
    "ENS4FMT (V4)":        SINGLE["PAIR"] + SINGLE["PAIRMARK"] + SINGLE["CANONRAW  canon, no fix"]
                           + SINGLE["MARKCANON canon+markers"],
    "ENS3FMT lean":        SINGLE["PAIR"] + SINGLE["MARKCANON canon+markers"]
                           + SINGLE["CANONRAW  canon, no fix"],
}


def signed_best(S, y, hv, v1):
    """Best chi2 among thresholds where the model is AHEAD of V1 (k>m). Oracle -- diagnostic only."""
    bb = None
    for t in np.arange(0.005, 1.0, 0.005):
        p = (S >= t).astype(int)
        a, b = (v1[hv] == y[hv]), (p[hv] == y[hv])
        k, m = int((~a & b).sum()), int((a & ~b).sum())
        if k <= m:
            continue
        c2 = (abs(k - m) - 1) ** 2 / max(k + m, 1)
        if bb is None or c2 > bb[0]:
            bb = (c2, float(t), k, m)
    return bb


def main():
    d = E.load_clean()
    y = d.label.to_numpy()
    strat = (d.source.astype(str) + "_" + d.label.astype(str)).to_numpy()
    hv = d.v1.notna().to_numpy()
    v1 = d.v1.fillna(0).to_numpy().astype(int)

    allcfg = dict(SINGLE); allcfg.update(ENSEMBLES)
    SC, OUT = {}, {}
    for tag, mds in allcfg.items():
        miss = [m for m in mds if not (REPO/m).exists()]
        if miss:
            print(f"[skip] {tag}: missing {len(miss)} ckpt"); continue
        o, S, P = E.summarise(tag, mds, d, R=R)
        SC[tag] = S
        st = [E.mcnemar(v1[hv], p[hv], y[hv]) for p in P]
        o["cf_vs_v1"] = dict(
            p_med=float(np.median([s[0] for s in st])),
            p_lo=float(min(s[0] for s in st)), p_hi=float(max(s[0] for s in st)),
            k=float(np.median([s[1] for s in st])), m=float(np.median([s[2] for s in st])),
            f1=float(np.median([f1_score(y[hv], p[hv], zero_division=0) for p in P])),
            prec=float(np.median([precision_score(y[hv], p[hv], zero_division=0) for p in P])),
            rec=float(np.median([recall_score(y[hv], p[hv], zero_division=0) for p in P])),
            n_sig=int(sum(1 for s in st if s[0] < 0.05 and s[1] > s[2])))
        o["oracle_signed_vs_v1"] = signed_best(S, y, hv, v1)
        o["_P"] = P
        OUT[tag] = o

    v1ref = dict(n=int(hv.sum()), precision=float(precision_score(y[hv], v1[hv])),
                 recall=float(recall_score(y[hv], v1[hv])), f1=float(f1_score(y[hv], v1[hv])))
    print(f"\nV1 on its {v1ref['n']} clean rows:  P {v1ref['precision']:.4f}  "
          f"R {v1ref['recall']:.4f}  F1 {v1ref['f1']:.4f}\n")

    hdr = (f"{'arm':<24}{'AUPRC':>8}{'cfF1':>8}{'devF1':>7}{'orcF1':>7} | "
           f"{'k':>3}{'m':>3}{'p(V1)':>7}{'sig':>4} | {'bx':>6}{'rej':>6}{'t299':>6}")
    print(hdr); print("-" * len(hdr))
    for tag, o in OUT.items():
        c = o["cf_vs_v1"]; ps = o["per_source_auprc"]
        print(f"{tag:<24}{o['auprc']:>8.4f}{o['cf_f1_med']:>8.4f}"
              f"{o.get('dev_f1_440', float('nan')):>7.3f}{o['oracle_f1']:>7.3f} | "
              f"{c['k']:>3.0f}{c['m']:>3.0f}{c['p_med']:>7.3f}{c['n_sig']:>4d} | "
              f"{ps.get('biotx100', 0):>6.3f}{ps.get('reject50', 0):>6.3f}{ps.get('test299', 0):>6.3f}")

    # paired vs V2 / V3 on all 440, cross-fitted decisions
    print("\nPaired vs V2 / V3 (all 440, cross-fitted decisions; AUPRC by paired bootstrap)")
    print(f"{'arm':<24}{'k/m vs V2':>12}{'p':>7}{'dAUPRC':>9}{'P(<=0)':>8} | "
          f"{'k/m vs V3':>12}{'p':>7}{'dAUPRC':>9}{'P(<=0)':>8}")
    for tag, o in OUT.items():
        if tag.startswith(("V2", "V3")): continue
        cells = []
        for base in ("V2 (triple)", "V3 (triple)"):
            if base not in OUT: cells.append("-"); continue
            st = [E.mcnemar(pb, pa, y) for pb, pa in zip(OUT[base]["_P"], o["_P"])]
            bt = E.boot_auprc(y, SC[base], SC[tag])
            o.setdefault("paired", {})[base[:2]] = dict(
                p_med=float(np.median([s[0] for s in st])),
                k=float(np.median([s[1] for s in st])), m=float(np.median([s[2] for s in st])),
                auprc=bt)
            cells.append(f"{np.median([s[1] for s in st]):>5.0f}/{np.median([s[2] for s in st]):<5.0f}"
                         f"{np.median([s[0] for s in st]):>7.3f}{bt['delta']:>+9.4f}{bt['p_le0']:>8.3f}")
        print(f"{tag:<24}{cells[0]} | {cells[1]}")

    # nested cross-fit over every config
    print("\nNested cross-fit (config AND threshold chosen inside the fold)")
    cfgs = {t: SC[t] for t in SC if not t.startswith(("V2", "V3"))}
    P, picks = N.nested(cfgs, y, strat, R=R)
    f1s = [f1_score(y, p, zero_division=0) for p in P]
    st = [E.mcnemar(v1[hv], p[hv], y[hv]) for p in P]
    sel = pd.Series(picks).value_counts()
    print(f"  out-of-fold F1 (440) = {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
    print(f"  vs V1: k={np.median([s[1] for s in st]):.0f} m={np.median([s[2] for s in st]):.0f} "
          f"p_med={np.median([s[0] for s in st]):.3f}  folds significant: "
          f"{sum(1 for s in st if s[0]<0.05 and s[1]>s[2])}/{len(st)}")
    print(f"  configs chosen: {dict(sel.head(6))}")
    nested_out = dict(f1=float(np.mean(f1s)), f1_sd=float(np.std(f1s)),
                      k=float(np.median([s[1] for s in st])), m=float(np.median([s[2] for s in st])),
                      p_med=float(np.median([s[0] for s in st])), picks=dict(sel.astype(int)))

    for t in OUT: OUT[t].pop("_P", None)
    outdir = REPO/"results/final_v4"; outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"summary.json").write_text(json.dumps(
        dict(v1=v1ref, arms=OUT, nested=nested_out), indent=2, default=float))
    np.savez(outdir/"scores.npz", **{t.replace(" ", "_"): s for t, s in SC.items()})
    print(f"\nwrote {outdir}/summary.json")
    return OUT, SC, d


if __name__ == "__main__":
    main()
