#!/usr/bin/env python3
"""Aggregate the loss-shaping sweep into one table, with the seed-variance band.

Every number comes from results/loss_shaping/*.json, which scripts/frontier_unified.py
wrote by calling scripts/eval_unified.evaluate() -- so all of it is on the same 440
clean rows of the unified test set, under species-level labels.

The point of the table is the CONTROL BAND: models/student_v3/xenc_s{1,2,3} and
models/loss_shaping/ce_s{1,2,3} are six runs of the *same* recipe differing only in
RNG. Their spread is the noise floor. A loss variant that lands inside that band has
not been shown to do anything.
"""
import json, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/loss_shaping"
# six runs of the unmodified cross-entropy recipe on v3_combined_train.csv
CE_REPLICATES = ["V3_seed1", "V3_seed2", "V3_seed3", "ce_s1", "ce_s2", "ce_s3"]
V1 = dict(precision=0.8351648351648352, recall=0.8941176470588236, f1=0.8636363636363636)


def load():
    rows = []
    for p in sorted(glob.glob(str(RES / "*.json"))):
        o = json.load(open(p))
        tag = os.path.basename(p)[:-5]
        bm = o.get("best_mcnemar") or {}
        hr = o.get("high_recall_region") or {}
        hm = o.get("best_mcnemar_at_v1_recall") or {}
        rows.append(dict(
            tag=tag,
            auprc=o["auprc"],
            best_f1=o.get("best_f1_evalunified", o["best_f1"]),
            best_f1_fine=o["best_f1"],
            p_at_v1_recall=hr.get("max_precision_at_recall_ge_v1"),
            dead_pos=(o.get("dead_positive_rate") or {}).get("lt_0.05"),
            auprc_biotx=o["per_source"]["biotx100"]["auprc"],
            auprc_reject=o["per_source"]["reject50"]["auprc"],
            auprc_t299=o["per_source"]["test299"]["auprc"],
            fixes=bm.get("fixes"), breaks=bm.get("breaks"),
            mcnemar_p=bm.get("p"), model_wins=bm.get("model_wins"),
            p_at_v1R=hm.get("p"), fixes_at_v1R=hm.get("fixes"), breaks_at_v1R=hm.get("breaks"),
        ))
    return pd.DataFrame(rows)


def band(df, cols):
    ce = df[df.tag.isin(CE_REPLICATES)]
    out = {}
    for c in cols:
        v = ce[c].dropna().to_numpy()
        out[c] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                      lo=float(v.min()), hi=float(v.max()))
    return out


def main():
    df = load()
    if df.empty:
        print("no results yet"); return
    METRICS = ["auprc", "best_f1", "p_at_v1_recall", "dead_pos"]
    B = band(df, METRICS)

    print("=" * 108)
    print("CONTROL BAND -- six runs of the unmodified CE recipe, RNG the only difference")
    print("=" * 108)
    for c in METRICS:
        b = B[c]
        print(f"  {c:16s} n={b['n']}  mean {b['mean']:.4f}  sd {b['sd']:.4f}   "
              f"observed range [{b['lo']:.4f}, {b['hi']:.4f}]  (width {b['hi']-b['lo']:.4f})")
    print()
    print("V1 on its 141 rows: precision %.4f  recall %.4f  F1 %.4f"
          % (V1["precision"], V1["recall"], V1["f1"]))
    print()

    def flag(r, c):
        v = r[c]
        if pd.isna(v) or r.tag in CE_REPLICATES or r.tag in ("V2", "V3"):
            return " "
        return "+" if v > B[c]["hi"] else ("-" if v < B[c]["lo"] else ".")

    print("=" * 108)
    print("ALL RUNS.  +/-/. = above / below / inside the CE control band")
    print("=" * 108)
    hdr = (f"{'tag':22s} {'AUPRC':>8s}  {'bestF1':>8s}  {'P@R>=V1':>9s}  {'dead<.05':>9s}  "
           f"{'fix':>3s} {'brk':>3s} {'p(McN)':>7s}")
    print(hdr); print("-" * len(hdr))
    order = [t for t in CE_REPLICATES if t in set(df.tag)] + ["V2", "V3"]
    rest = sorted(set(df.tag) - set(order))
    for tag in order + rest:
        r = df[df.tag == tag]
        if r.empty: continue
        r = r.iloc[0]
        g = lambda c, n=4: "    -   " if pd.isna(r[c]) else f"{r[c]:.{n}f}"
        print(f"{tag:22s} {g('auprc')}{flag(r,'auprc')} {g('best_f1')}{flag(r,'best_f1')} "
              f"{g('p_at_v1_recall')}{flag(r,'p_at_v1_recall')}  {g('dead_pos')}{flag(r,'dead_pos')}  "
              f"{str(r['fixes']):>3s} {str(r['breaks']):>3s} "
              f"{'   -   ' if pd.isna(r['mcnemar_p']) else f'{r.mcnemar_p:7.3f}'}"
              f"{'' if r['model_wins'] in (True, np.True_) else '  (V1 ahead)'}")
    print()
    print("Significance bar: McNemar on the 141 rows needs (|k-m|-1)^2/(k+m) > 3.84.")
    print("None of the above reaches it unless p < 0.05 in the last column AND fix > brk.")
    df.to_csv(RES / "summary.csv", index=False)
    print(f"\nwrote {RES/'summary.csv'}")


if __name__ == "__main__":
    main()
