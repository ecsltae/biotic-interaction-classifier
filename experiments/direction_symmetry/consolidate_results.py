#!/usr/bin/env python3
"""Single source of truth for every direction-arm number quoted in the paper.

Values are transcribed from the evaluator runs of 2026-09-27; re-running
eval_direction_general.py / eval_ordersens.py / eval_ours_on_biored.py reproduces them.
"""
import json
import numpy as np

R = {
 "biored": {
   "n_directed": 1045,
   "ours_canonical_sym": {"acc": [0.9091, 0.9100, 0.9129], "swap": [1.0, 1.0, 1.0],
                          "auc": [0.8208, 0.8495, 0.8663], "maxdev": 2.98e-8},
   "text_uncon":         {"acc": [0.9120, 0.9129, 0.9100], "swap": [0.9809, 0.9694, 0.9569],
                          "auc": [0.8650, 0.8671, 0.8616], "maxdev": 9.75e-1},
 },
 "semeval": {
   "n_directed": 2260,
   "ours_canonical_sym": {"acc": [0.9650, 0.9615, 0.9611], "swap": [1.0, 1.0, 1.0],
                          "auc": [0.9823, 0.9245, 0.9354], "maxdev": 2.98e-8},
   "text_uncon":         {"acc": [0.9642, 0.9673, 0.9659], "swap": [0.9588, 0.9066, 0.9496],
                          "auc": [0.9298, 0.8952, 0.8962], "maxdev": 9.78e-1},
   "canonical_uncon":    {"acc": [0.9642], "swap": [1.0], "auc": [0.9297], "maxdev": 2.98e-8},
   "text_sym":           {"acc": [0.9668], "swap": [0.9894], "auc": [0.9794], "maxdev": 9.99e-1},
 },
 "biodiversity": {
   "n_directed": 84,
   "ours_canonical_sym": {"acc": [0.869], "swap": [1.0],   "auc": [0.748], "maxdev": 2.98e-8},
   "canonical_uncon":    {"acc": [0.857], "swap": [1.0],   "auc": [0.787], "maxdev": 2.98e-8},
   "text_sym":           {"acc": [0.810], "swap": [0.969], "auc": [0.718], "maxdev": None},
   "text_uncon":         {"acc": [0.786], "swap": [0.473], "auc": [0.811], "maxdev": None},
 },
}
PI = {"biodiversity": 0.708, "semeval": 0.544, "biored": 0.486}


def ms(v):
    a = np.array(v, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else float("nan")), len(a)


if __name__ == "__main__":
    print(f"{'corpus':13} {'arm':20} {'k':>2} {'dir acc':>16} {'swap':>16} {'und AUC':>16}")
    print("-" * 88)
    out = {}
    for c, arms in R.items():
        for a, d in arms.items():
            if a == "n_directed":
                continue
            row = {}
            for m in ("acc", "swap", "auc"):
                mu, sd, k = ms(d[m])
                row[m] = {"mean": round(mu, 4), "sd": None if np.isnan(sd) else round(sd, 4), "k": k}
            row["maxdev"] = d["maxdev"]
            out[f"{c}/{a}"] = row
            f = lambda m: (f"{row[m]['mean']:.4f}" +
                           (f" ±{row[m]['sd']:.4f}" if row[m]["sd"] is not None else "        "))
            print(f"{c:13} {a:20} {row['acc']['k']:2d} {f('acc'):>16} {f('swap'):>16} {f('auc'):>16}")
    json.dump({"pi": PI, "arms": out}, open("RESULTS_CONSOLIDATED.json", "w"), indent=2)
    print("\nwrote RESULTS_CONSOLIDATED.json")
