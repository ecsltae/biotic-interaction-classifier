#!/usr/bin/env python3
"""Species-level teacher labelling, concurrent.

Same prompt and same decoding settings as relabel_species_level.py, so labels
produced here and there are interchangeable. The only change is that requests are
issued from a thread pool instead of one at a time -- Ollama batches concurrent
requests against the resident model, so throughput scales with --workers until the
GPU saturates, and a sequential client leaves most of that on the table.

Resumable on the `row` key; re-running skips whatever the output already contains.
"""
import argparse, json, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-label", type=int, default=None,
                    help="restrict to rows whose existing `label` equals this")
    ap.add_argument("--rows-from", default=None, help="csv of `row` values to do (in order)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--shuffle", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(REPO/a.data if not Path(a.data).is_absolute() else a.data)
    df = df.reset_index().rename(columns={"index": "row"})
    if a.only_label is not None and "label" in df.columns:
        df = df[df.label == a.only_label]
    if a.rows_from:
        order = pd.read_csv(REPO/a.rows_from)["row"].tolist()
        df = df.set_index("row").loc[[r for r in order if r in set(df.row)]].reset_index()
    if a.shuffle: df = df.sample(frac=1.0, random_state=20260923).reset_index(drop=True)
    if a.limit: df = df.head(a.limit)

    out = Path(a.out) if Path(a.out).is_absolute() else REPO/a.out
    done = set()
    if out.exists():
        try: done = set(pd.read_csv(out)["row"].tolist())
        except Exception: done = set()
    todo = df[~df.row.isin(done)]
    print(f"{len(df):,} candidate rows, {len(done):,} already done, {len(todo):,} to do, "
          f"{a.workers} workers", flush=True)
    if not len(todo): return

    lock = threading.Lock()
    buf, state = [], {"n": 0, "t0": time.time()}
    sess = threading.local()

    def ask(r):
        if not hasattr(sess, "s"): sess.s = requests.Session()
        s1 = r.get("source_species", r.get("species1_form"))
        s2 = r.get("target_species", r.get("species2_form"))
        txt = r.get("text", r.get("passage"))
        try:
            resp = sess.s.post(f"{a.host}/api/generate", json={
                "model": a.model, "prompt": PROMPT.format(sent=txt, s1=s1, s2=s2),
                "stream": False, "options": {"temperature": 0, "num_predict": 4, "seed": 0},
                "think": False}, timeout=600).json().get("response", "")
        except Exception as e:
            resp = f"ERR:{type(e).__name__}"
        rec = {"row": int(r["row"]),
               "label_species": 1 if re.search(r"\byes\b", resp, re.I) else 0,
               "raw": resp.strip()[:16]}
        with lock:
            buf.append(rec); state["n"] += 1
            if len(buf) >= 100:
                hdr = not out.exists()
                pd.DataFrame(buf).to_csv(out, mode="a", header=hdr, index=False)
                buf.clear()
                el = time.time() - state["t0"]; rate = state["n"]/el
                print(f"  {state['n']:,}/{len(todo):,}  {rate:.2f} it/s  "
                      f"eta {(len(todo)-state['n'])/max(rate,1e-9)/60:.0f} min", flush=True)
        return rec

    recs = [r for _, r in todo.iterrows()]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(ask, recs))
    with lock:
        if buf:
            pd.DataFrame(buf).to_csv(out, mode="a", header=not out.exists(), index=False)
    el = time.time() - state["t0"]
    print(f"done {state['n']:,} in {el/60:.1f} min ({state['n']/el:.2f} it/s)", flush=True)

if __name__ == "__main__": main()
