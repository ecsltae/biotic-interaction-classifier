#!/usr/bin/env python3
"""Species-level teacher labelling, N items per request.

Same rubric and decoding temperature as scripts/relabel_species_level.py. Packing
several items into one request multiplies throughput per scheduler slot, which is
what matters when the Ollama queue is contended. Agreement against one-at-a-time
labelling is measured separately before this is trusted.

Input parquet needs columns: uid, sent, s1, s2.
"""
import argparse, itertools, re, sys, threading, time
from pathlib import Path
import pandas as pd, requests, concurrent.futures as cf

RUBRIC = """You are judging whether scientific sentences state that two organisms interact.

For each numbered item, answer YES if the sentence asserts ANY direct biological
interaction between the two named organisms -- for example one infecting, parasitising,
preying on, pollinating, hosting, transmitting or feeding on the other, in either
direction. Ignore whether any particular relation word is the technically correct one.
The question is only whether the sentence says these two organisms interact with each other.

Answer NO if the sentence merely mentions both organisms without asserting an interaction
between them, or if each interacts only with some third organism, or if either name does
not refer to an organism here.

{items}

Reply with exactly {n} lines, one per item, in order, each of the form
<number>: YES
or
<number>: NO
Nothing else."""

ITEM = """Item {i}
Sentence: {sent}
Organism 1: "{s1}"
Organism 2: "{s2}"
"""

def build(chunk):
    items = "\n".join(ITEM.format(i=j+1, sent=str(r.sent).replace("\n", " "), s1=r.s1, s2=r.s2)
                      for j, r in enumerate(chunk))
    return RUBRIC.format(items=items, n=len(chunk))

def parse(resp, k):
    out = {}
    for line in str(resp).splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)-]\s*(YES|NO)\b", line.strip(), re.I)
        if m:
            i = int(m.group(1))
            if 1 <= i <= k: out[i] = 1 if m.group(2).upper() == "YES" else 0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints", default="http://127.0.0.1:11434")
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--flush", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_parquet(a.inp)
    out = Path(a.out); done = set()
    if out.exists():
        try: done = set(pd.read_csv(out).uid.astype(str))
        except Exception: pass
    todo = df[~df.uid.astype(str).isin(done)]
    if a.limit: todo = todo.head(a.limit)
    rows = [r for _, r in todo.iterrows()]
    chunks = [rows[i:i+a.batch] for i in range(0, len(rows), a.batch)]
    eps = itertools.cycle(a.endpoints.split(","))
    print(f"{len(df)} rows, {len(done)} done, {len(rows)} to label in {len(chunks)} requests "
          f"(batch={a.batch}, conc={a.conc})", flush=True)

    lock = threading.Lock(); buf = []; n = [0]; miss = [0]; t0 = time.time(); hdr = [not out.exists()]
    tl = threading.local()

    def run(arg):
        chunk, ep = arg
        if not hasattr(tl, "s"): tl.s = requests.Session()
        try:
            resp = tl.s.post(f"{ep}/api/generate", json={
                "model": "qwen3:32b", "prompt": build(chunk), "stream": False,
                "options": {"temperature": 0, "num_predict": 12*len(chunk)+16, "seed": 0},
                "think": False}, timeout=900).json().get("response", "")
        except Exception:
            resp = ""
        got = parse(resp, len(chunk))
        recs = [{"uid": str(r.uid), "label_species": got[j+1]}
                for j, r in enumerate(chunk) if (j+1) in got]
        with lock:
            buf.extend(recs); n[0] += len(recs); miss[0] += len(chunk)-len(recs)
            if len(buf) >= a.flush:
                pd.DataFrame(buf).to_csv(out, mode="w" if hdr[0] else "a", header=hdr[0], index=False)
                hdr[0] = False; buf.clear()
                el = time.time()-t0
                print(f"  {n[0]}/{len(rows)}  {n[0]/el:.2f} it/s  miss {miss[0]}  "
                      f"eta {(len(rows)-n[0])/max(n[0]/el,1e-9)/60:.0f}m", flush=True)
        return None

    with cf.ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(run, ((c, next(eps)) for c in chunks)))
    if buf:
        pd.DataFrame(buf).to_csv(out, mode="w" if hdr[0] else "a", header=hdr[0], index=False)
    print(f"done {round(time.time()-t0)}s  labelled={n[0]}  unparsed={miss[0]}", flush=True)

if __name__ == "__main__": main()
