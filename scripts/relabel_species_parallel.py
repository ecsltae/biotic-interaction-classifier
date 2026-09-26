#!/usr/bin/env python3
"""Parallel species-level re-label. Same prompt/semantics as relabel_species_level.py,
but issues concurrent requests to Ollama. Resumable: skips rows already in --out."""
import argparse, re, threading, time
from pathlib import Path
import concurrent.futures as cf
import pandas as pd, requests

REPO = Path(__file__).resolve().parents[1]
PROMPT = """You are judging whether a scientific sentence states that two organisms interact.

Sentence: {sent}

Organism 1: "{s1}"
Organism 2: "{s2}"

Answer YES if the sentence asserts ANY direct biological interaction between these two
organisms -- for example one infecting, parasitising, preying on, pollinating, hosting,
transmitting or feeding on the other, in either direction.

Ignore whether any particular relation word is the technically correct one. The question is
only whether the sentence says these two organisms interact with each other.

Answer NO if the sentence merely mentions both organisms without asserting an interaction
between them, or if each interacts only with some third organism, or if either name does not
refer to an organism here.

Reply with exactly one word: YES or NO."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--out", default="data/training/distill/species_relabel.csv")
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--endpoint", default="http://localhost:11434")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--part", type=int, default=0, help="this worker's index")
    ap.add_argument("--nparts", type=int, default=1, help="number of workers sharing the queue")
    ap.add_argument("--done-from", nargs="*", default=[],
                    help="extra already-done files to skip (union of all workers' outputs)")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomise queue order so every worker covers all kinds uniformly")
    a = ap.parse_args()

    df = pd.read_csv(REPO/a.data)
    m = df.reset_index().rename(columns={"index": "row"})
    no = m[m.label == 0].copy()
    out = REPO/a.out
    done = set()
    for f in [out] + [REPO/x for x in a.done_from]:
        if Path(f).exists():
            done |= set(pd.read_csv(f)["row"].tolist())
    if a.shuffle:
        no = no.sample(frac=1.0, random_state=20260923).reset_index(drop=True)
    if a.nparts > 1:
        no = no.iloc[a.part::a.nparts]
    todo = no[~no.row.isin(done)]
    print(f"{len(no):,} NO rows; {len(done):,} already done; {len(todo):,} to go", flush=True)

    lock, buf, t0, n = threading.Lock(), [], time.time(), [0]
    sess = threading.local()
    def judge(r):
        if not hasattr(sess, "s"): sess.s = requests.Session()
        try:
            resp = sess.s.post(f"{a.endpoint}/api/generate", json={
                "model": a.model,
                "prompt": PROMPT.format(sent=r.text, s1=r.source_species, s2=r.target_species),
                "stream": False, "options": {"temperature": 0, "num_predict": 4, "seed": 0},
                "think": False}, timeout=600).json().get("response", "")
        except Exception:
            resp = "ERR"
        rec = {"row": int(r.row), "label_species": 1 if re.search(r"\byes\b", resp, re.I) else 0,
               "raw": resp.strip()[:16]}
        with lock:
            buf.append(rec); n[0] += 1
            if len(buf) >= 200:
                pd.DataFrame(buf).to_csv(out, mode="a" if done else "w", header=not done, index=False)
                done.update(x["row"] for x in buf); buf.clear()
                el = time.time()-t0
                print(f"  {len(done)}/{len(no)}  {n[0]/el:.2f} it/s  "
                      f"eta {(len(todo)-n[0])/max(n[0]/el,1e-9)/60:.0f} min", flush=True)
        return None

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(judge, [r for _, r in todo.iterrows()]))
    if buf:
        pd.DataFrame(buf).to_csv(out, mode="a" if done else "w", header=not done, index=False)
    print(f"done {round(time.time()-t0)}s", flush=True)

if __name__ == "__main__":
    main()
