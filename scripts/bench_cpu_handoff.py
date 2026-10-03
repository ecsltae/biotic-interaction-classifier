#!/usr/bin/env python3
"""CPU throughput of the shipped verifier, measured through the handoff package's own predict().

Both heads, candidate rules on (the deployed configuration), eight threads, batch 16, the 437
benchmark rows; the median of three timed passes after one warm-up, plus the one-off load time.
The load average at the start is recorded, since other jobs on the machine slow the measurement.

Usage
  python3 scripts/bench_cpu_handoff.py      # -> results/paperA_v2/cpu_bench_handoff.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "handoff/biotic_verifier"))
from eval.core import clean_benchmark  # noqa: E402
import predict as PR  # noqa: E402


def main() -> None:
    torch.set_num_threads(8)
    load_avg = os.getloadavg()[0]
    t0 = time.perf_counter()
    m, tok, cfg = PR.load(REPO / "handoff/biotic_verifier/model", device="cpu")
    load_s = time.perf_counter() - t0
    d = clean_benchmark()
    run = lambda: PR.predict(m, tok, d.species1, d.species2, d.relation, d.sentence, device="cpu",  # noqa: E731
                             bs=16, max_len=int(cfg.get("max_len", 256)), rules=True)
    run()                                                 # warm-up
    ts = []
    for _ in range(3):
        t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
    out = {"rows": int(len(d)), "threads": 8, "batch": 16, "rules": True,
           "candidates_per_s_median": float(len(d) / np.median(ts)), "per_pass_s": ts,
           "load_s": load_s, "load_average_at_start": load_avg}
    (REPO / "results/paperA_v2/cpu_bench_handoff.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
