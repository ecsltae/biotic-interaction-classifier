#!/usr/bin/env python3
"""Why the pair-max AUPRCs differ from the sandbox's in the 5th-6th decimal.

Two candidate causes, both tested on seed 1 of the pair arm over the same 6,253 pairs:
  order      the sandbox enumerated each passage's pairs as a Python set (hash-seed dependent
             order); `biodiv_sentence.py` fixes the order, so GPU batches differ
  precision  `biodiv_sentence.py` scores with torch.set_float32_matmul_precision("high") (TF32), as
             `scripts/paperA_tables.py` does for the paper's tables; the sandbox script left the
             default ("highest", full FP32)
The result says which one reproduces the sandbox's per-seed AUPRC (all three seeds and the
ensemble are re-scored in FP32 for the record).

Usage
  python3 scripts/sentence_level/order_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as ap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from biodiv_sentence import benchmark, enum_pairs, silver  # noqa: E402
from paperA_tables import order_free  # noqa: E402


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--seed", type=int, default=1, help="pair-arm checkpoint seed")
    a = ap_.parse_args()
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    d = benchmark(); P = enum_pairs(d); y = silver(d)["majority3"]
    fixed = np.load(C.CACHE / f"{C.model_key('biodiv_enum', C.REPO / f'models/pair_baseline/xenc_s{a.seed}')}.npy")
    perm = np.random.default_rng(0).permutation(len(P))
    sh = np.empty(len(P)); sh[perm] = order_free(C.REPO / f"models/pair_baseline/xenc_s{a.seed}", P.iloc[perm], dev)
    mx = lambda e: P.assign(s=e).groupby("i").s.max().reindex(range(len(d))).to_numpy()  # noqa: E731
    torch.set_float32_matmul_precision("highest")
    fp32_all = {k: order_free(C.REPO / f"models/pair_baseline/xenc_s{k}", P, dev) for k in (1, 2, 3)}
    fp32 = fp32_all[a.seed]
    sb = json.loads((C.REPO / "sandbox/sentence_level/results/biodiv_sentence_silver.json").read_text())
    res = {"seed": a.seed, "pairs": int(len(P)),
           "order": {"max_abs_score_diff": float(np.abs(sh - fixed).max()),
                     "auprc_fixed_order": float(ap(y, mx(fixed))), "auprc_shuffled_order": float(ap(y, mx(sh)))},
           "precision": {"max_abs_score_diff_tf32_vs_fp32": float(np.abs(fp32 - fixed).max()),
                         "auprc_tf32": float(ap(y, mx(fixed))), "auprc_fp32": float(ap(y, mx(fp32)))},
           "sandbox_auprc": sb["pair_max_auprc_per_seed"][a.seed - 1]}
    res["fp32_reproduces_sandbox_exactly"] = res["precision"]["auprc_fp32"] == res["sandbox_auprc"]
    per = [mx(fp32_all[k]) for k in (1, 2, 3)]
    res["fp32_all_seeds"] = {"auprc_per_seed": [float(ap(y, p)) for p in per],
                             "sandbox_per_seed": sb["pair_max_auprc_per_seed"],
                             "auprc_ensemble": float(ap(y, np.mean(per, axis=0))),
                             "sandbox_ensemble": sb["pair-max"]["auprc"]}
    res["fp32_all_identical"] = (res["fp32_all_seeds"]["auprc_per_seed"] == sb["pair_max_auprc_per_seed"]
                                 and res["fp32_all_seeds"]["auprc_ensemble"] == sb["pair-max"]["auprc"])
    print(res)
    C.dump(res, "order_check.json")


if __name__ == "__main__":
    main()
