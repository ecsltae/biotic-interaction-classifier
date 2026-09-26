#!/usr/bin/env python3
"""CPU throughput and parameter count for the shipped configuration.

The collaborator runs this on CPU, so pairs/s and parameter count are hard
constraints, not footnotes. Measured with dynamic per-batch padding (what the
eval harness and any sane deployment use) on the 440 evaluation rows.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xenc_format, eval_unified as EU

REPO = Path(__file__).resolve().parents[1]


def bench(md, d, threads=8, bs=16, reps=3, warmup=1):
    torch.set_num_threads(threads)
    fmt = EU.model_format(md)
    q, p = xenc_format.build_many(fmt, d.species1, d.relation, d.species2, d.sentence.astype(str))
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(md, local_files_only=True).eval()
    n_par = sum(x.numel() for x in m.parameters())
    ts = []
    with torch.no_grad():
        for r in range(reps + warmup):
            t0 = time.perf_counter()
            for i in range(0, len(q), bs):
                kw = dict(truncation=True, max_length=256, padding=True, return_tensors="pt")
                e = tok(q[i:i+bs], **kw) if p is None else \
                    tok(q[i:i+bs], p[i:i+bs], truncation="only_second", max_length=256,
                        padding=True, return_tensors="pt")
                m(**e)
            dt = time.perf_counter() - t0
            if r >= warmup:
                ts.append(len(q)/dt)
    return dict(model=str(md), format=fmt, params=int(n_par),
                pairs_per_s=float(np.median(ts)), runs=[float(x) for x in ts], threads=threads)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = pd.read_csv(REPO/"data/evaluation/unified_test_set.csv")
    d = d[~d.in_train].reset_index(drop=True)
    res = [bench(REPO/m, d, a.threads) for m in a.models]
    for r in res:
        print(f"{r['model'].split('/')[-1]:<24} fmt={r['format']:<10} "
              f"params={r['params']/1e6:6.1f}M  {r['pairs_per_s']:7.1f} pairs/s (1 ckpt)")
    if a.out:
        Path(REPO/a.out).write_text(json.dumps(res, indent=2))
