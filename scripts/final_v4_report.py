#!/usr/bin/env python3
"""Consolidated comparison of every arm against V1, V2 and V3.

Thresholds are cross-fitted (see eval_final_v4); the V1 comparison uses the 141
clean rows carrying a recorded V1 decision; V2/V3 comparisons use all 440.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import eval_final_v4 as E

REPO = Path(__file__).resolve().parents[1]
R = 20

ARMS = {
    "V2  (triple, deployed-lineage)": [f"models/student/xenc_s{i}" for i in (1, 2, 3)],
    "V3  (triple, +contrast data)":   [f"models/student_v3/xenc_s{i}" for i in (1, 2, 3)],
    "PAIR (arch lever)":              [f"models/lever_arch/fmt_pair_s{i}" for i in (1, 2, 3)],
    "PAIRMARK (arch lever)":          [f"models/lever_arch/fmt_pair_mark_s{i}" for i in (1, 2, 3)],
    "CTLW (pair, prior control)":     [f"models/final_v4/ctlw_s{i}" for i in (1, 2, 3)],
    "CMAX (pair, contra->1)":         [f"models/final_v4/cmax_s{i}" for i in (1, 2, 3)],
    "CDROP (pair, contra dropped)":   [f"models/final_v4/cdrop_s{i}" for i in (1, 2, 3)],
    "CCOLL (pair, collapsed)":        [f"models/final_v4/ccoll_s{i}" for i in (1, 2, 3)],
    "CCOLLTGT (collapsed+targeted)":  [f"models/final_v4/ccolltgt_s{i}" for i in (1, 2, 3)],
}


def main(extra=None):
    arms = dict(ARMS)
    if extra:
        arms.update(extra)
    d = E.load_clean()
    y = d.label.to_numpy()
    hv = d.v1.notna().to_numpy()
    v1 = d.v1.fillna(0).to_numpy().astype(int)
    out, SS, PP = {}, {}, {}
    for tag, mds in arms.items():
        if not all((REPO/m).exists() for m in mds):
            print(f"  [skip] {tag}: missing checkpoints"); continue
        o, S, P = E.summarise(tag, mds, d, R=R)
        SS[tag], PP[tag] = S, P
        # vs V1 on its 141, per cross-fit repeat
        st = [E.mcnemar(v1[hv], p[hv], y[hv]) for p in P]
        o["vs_v1"] = dict(
            p_med=float(np.median([s[0] for s in st])),
            p_min=float(min(s[0] for s in st)), p_max=float(max(s[0] for s in st)),
            k_med=float(np.median([s[1] for s in st])), m_med=float(np.median([s[2] for s in st])),
            chi2_med=float(np.median([s[3] for s in st])),
            f1_med=float(np.median([f1_score(y[hv], p[hv], zero_division=0) for p in P])),
            prec_med=float(np.median([precision_score(y[hv], p[hv], zero_division=0) for p in P])),
            rec_med=float(np.median([recall_score(y[hv], p[hv], zero_division=0) for p in P])),
            frac_sig=float(np.mean([s[0] < 0.05 and s[1] > s[2] for s in st])))
        out[tag] = o
    # V1 reference
    out["_V1"] = dict(tag="V1 (deployed)", n=int(hv.sum()),
                      precision=float(precision_score(y[hv], v1[hv])),
                      recall=float(recall_score(y[hv], v1[hv])),
                      f1=float(f1_score(y[hv], v1[hv])))
    # paired vs V2 / V3 on all 440
    for base in ("V2  (triple, deployed-lineage)", "V3  (triple, +contrast data)"):
        if base not in SS: continue
        for tag in SS:
            if tag == base: continue
            st = [E.mcnemar(pb, pa, y) for pb, pa in zip(PP[base], PP[tag])]
            out[tag].setdefault("paired", {})[base.split()[0]] = dict(
                p_med=float(np.median([s[0] for s in st])),
                k_med=float(np.median([s[1] for s in st])), m_med=float(np.median([s[2] for s in st])),
                frac_sig=float(np.mean([s[0] < 0.05 and s[1] > s[2] for s in st])),
                auprc=E.boot_auprc(y, SS[base], SS[tag]))
    Path(REPO/"results/final_v4").mkdir(parents=True, exist_ok=True)
    (REPO/"results/final_v4/summary.json").write_text(json.dumps(out, indent=2, default=float))
    np.save(REPO/"results/final_v4/scores_by_arm.npy", {k: v for k, v in SS.items()}, allow_pickle=True)

    v1r = out["_V1"]
    print(f"\nV1 on its {v1r['n']} rows: P {v1r['precision']:.4f}  R {v1r['recall']:.4f}  F1 {v1r['f1']:.4f}\n")
    hdr = f"{'arm':<32}{'AUPRC':>8}{'cfF1':>8}{'devF1':>8}{'orcF1':>8} | {'k':>4}{'m':>4}{'p(V1)':>8} | {'p(V2)':>8}{'p(V3)':>8}"
    print(hdr); print("-"*len(hdr))
    for tag, o in out.items():
        if tag.startswith("_"): continue
        v = o["vs_v1"]; pr = o.get("paired", {})
        print(f"{tag:<32}{o['auprc']:>8.4f}{o['cf_f1_med']:>8.4f}"
              f"{o.get('dev_f1_440', float('nan')):>8.4f}{o['oracle_f1']:>8.4f} | "
              f"{v['k_med']:>4.0f}{v['m_med']:>4.0f}{v['p_med']:>8.3f} | "
              f"{pr.get('V2',{}).get('p_med', float('nan')):>8.3f}{pr.get('V3',{}).get('p_med', float('nan')):>8.3f}")
    return out


if __name__ == "__main__":
    main()
