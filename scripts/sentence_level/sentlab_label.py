#!/usr/bin/env python3
"""The teacher's answer to the SENTENCE question, for the passages of the paper's corpus.

The paper's biodiversity sentence arm learned from candidate labels (one row per candidate, the
teacher asked about *that* pair). The fair sentence-level competitor learns from the teacher's
answer to the sentence question itself. This script asks it: the paper's teacher (qwen3:32b on the
system Ollama, the server that labelled the corpus), the zero-shot sentence prompt of
`scripts/llm_baseline.py` (`BIODIV["sentence"]`), scored through its `ask()` (P(YES) from the first
answer token, temperature 0, thinking off). Prompt and scorer are imported, not copied.

Order. The 34,242 unique passages of `data/training/distill/v3_combined_train.csv` are visited in
one fixed random permutation (numpy RandomState(0)). A run limited to N passages therefore labels
a uniform random sample (the first N of the permutation), and an interrupted run leaves a random
sample too.

Output. `data/training/distill/v3_sentence_teacher_qwen3-32b.csv` (text, p_yes, answer, label;
label = 1 iff the greedy answer starts with YES). Appended chunk by chunk and flushed; a re-run
skips the passages already present. Never overwritten.

Usage
  python3 scripts/sentence_level/sentlab_label.py --measure 200          # throughput, 1 vs 4 workers
  python3 scripts/sentence_level/sentlab_label.py --limit 12000 --workers 4 --stop-at "2026-10-05 05:00"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import llm_baseline as LB  # noqa: E402  (URL from OLLAMA_URL, default the system Ollama :11434)

CORPUS = REPO / "data/training/distill/v3_combined_train.csv"
OUT = REPO / "data/training/distill/v3_sentence_teacher_qwen3-32b.csv"
MODEL = "qwen3:32b"
COLUMNS = ["text", "p_yes", "answer", "label"]


def passages() -> list[str]:
    """Unique corpus passages in the fixed seed-0 random order."""
    u = pd.read_csv(CORPUS).text.astype(str).drop_duplicates().tolist()
    perm = np.random.RandomState(0).permutation(len(u))
    return [u[i] for i in perm]


def label_one(text: str) -> tuple[float, str, int]:
    """P(YES), raw greedy answer and its binary label for one passage."""
    p, t = LB.ask(MODEL, LB.BIODIV["sentence"].format(sent=text))
    return p, t, int(t.upper().startswith("YES"))


def measure(texts: list[str], workers: tuple[int, ...], out: Path) -> dict:
    """Seconds per passage at each concurrency, on the same passages (first call warms the model)."""
    label_one(texts[0])
    res = {"n": len(texts), "model": MODEL, "url": LB.URL}
    for w in workers:
        t0 = time.time()
        with ThreadPoolExecutor(w) as ex:
            r = list(ex.map(label_one, texts))
        el = time.time() - t0
        res[f"workers_{w}"] = {"seconds": el, "sec_per_passage": el / len(texts),
                               "yes_rate": float(np.mean([x[2] for x in r]))}
        print(f"workers={w}: {el:.1f}s for {len(texts)} passages = {el / len(texts):.3f} s/passage, "
              f"YES {res[f'workers_{w}']['yes_rate']:.3f}", flush=True)
    out.write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", type=int, default=0, help="time this many random passages at 1 and 4 workers")
    ap.add_argument("--measure-out", default="results/paperA_v2/sentence_level/sentlab_throughput.json")
    ap.add_argument("--limit", type=int, default=0, help="label the first N passages of the permutation (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--stop-at", default=None,
                    help="'YYYY-mm-dd HH:MM': stop labelling at this time (the labelled prefix is still random)")
    a = ap.parse_args()
    texts = passages()
    if a.measure:
        rng = np.random.RandomState(12345)                 # a timing sample, not the labelling order
        sample = [texts[i] for i in rng.choice(len(texts), a.measure, replace=False)]
        out = REPO / a.measure_out
        out.parent.mkdir(parents=True, exist_ok=True)
        measure(sample, (1, 4), out)
        return
    target = texts[: a.limit] if a.limit else texts
    done = set(pd.read_csv(OUT).text.astype(str)) if OUT.exists() else set()
    todo = [t for t in target if t not in done]
    stop = datetime.strptime(a.stop_at, "%Y-%m-%d %H:%M").timestamp() if a.stop_at else None
    print(f"{len(target)} passages targeted, {len(target) - len(todo)} already labelled, {len(todo)} to go", flush=True)
    new = not OUT.exists()
    t0, n = time.time(), 0
    with OUT.open("a", newline="") as fh, ThreadPoolExecutor(a.workers) as ex:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS); fh.flush()
        for i in range(0, len(todo), a.chunk):
            if stop and time.time() > stop:
                print(f"stop time reached after {n} new passages", flush=True)
                break
            part = todo[i: i + a.chunk]
            for txt, (p, t, y) in zip(part, ex.map(label_one, part)):
                w.writerow([txt, p, t, y])
            fh.flush()
            n += len(part)
            el = time.time() - t0
            print(f"{n}/{len(todo)} new passages, {el / n:.3f} s/passage, eta "
                  f"{(len(todo) - n) * el / n / 60:.0f} min", flush=True)
    print(f"labelled file: {OUT} ({len(pd.read_csv(OUT))} rows)", flush=True)


if __name__ == "__main__":
    main()
