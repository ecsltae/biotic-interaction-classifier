#!/usr/bin/env python3
"""The deployed sentence-level filter's (V1's) accept/reject decision for every benchmark row.

Internal artefact for the model card and the thesis, never a paper baseline. Each of the 449
benchmark rows gets V1's decision from, in order of preference,

  1. the benchmark's own `v1` column, where the decision was recorded with the row;
  2. V1's stored probabilities on test299 (results/v2/base299_V1_champion.json), thresholded at
     the threshold stored with them, matched on normalised sentence + species1 + species2;
  3. otherwise -1 (no decision).

The shipped results/v1_decisions_449.npy was written by this logic on 2026-09-25; `--check`
recomputes it and asserts the result is byte-identical.

Usage
  python3 scripts/build_v1_decisions.py            # write results/v1_decisions_449.npy
  python3 scripts/build_v1_decisions.py --check    # recompute and compare with the shipped file
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data/evaluation/unified_test_set.csv"
V1_RUN = REPO / "results/v2/base299_V1_champion.json"
OUT = REPO / "results/v1_decisions_449.npy"


def key(sentence, s1, s2) -> str:
    n = lambda s: re.sub(r"\W+", " ", str(s)).lower().strip()  # noqa: E731
    return f"{n(sentence)}||{n(s1)}||{n(s2)}"


def build() -> np.ndarray:
    j = json.loads(V1_RUN.read_text())["test299"]
    probs, thr = np.array(j["_probs"]), j["overall"]["threshold"]
    src = pd.read_csv(REPO / j["benchmark_path"] if not Path(j["benchmark_path"]).is_absolute()
                      else j["benchmark_path"])
    stored = dict(zip((key(*r) for r in zip(src.sentence, src.species1, src.species2)),
                      (probs >= thr).astype(int)))
    d = pd.read_csv(BENCH)
    k = [key(*r) for r in zip(d.sentence, d.species1, d.species2)]
    V = np.full(len(d), -1, dtype=int)
    rec = d.v1.notna().to_numpy()
    V[rec] = d.v1.fillna(0).to_numpy()[rec].astype(int)
    from_store = (~rec) & np.array([x in stored for x in k])
    V[from_store] = [stored[x] for x, m in zip(k, from_store) if m]
    print(f"{len(V)} rows: {int(rec.sum())} recorded + {int(from_store.sum())} from stored probabilities, "
          f"{int((V < 0).sum())} without a decision")
    return V


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    V = build()
    if a.check:
        buf = io.BytesIO()
        np.save(buf, V)
        same = buf.getvalue() == OUT.read_bytes()
        print(f"recomputed decisions are byte-identical to {OUT.relative_to(REPO)}: {same}")
        if not same:
            raise SystemExit(1)
        return
    np.save(OUT, V)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
