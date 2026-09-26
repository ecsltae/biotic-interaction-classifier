#!/usr/bin/env python3
"""Re-label with SPECIES-LEVEL semantics — the task the classifier is actually judged on.

The original teacher prompt required the stated relation to be the correct type and
direction (condition 3). That is a FULL-TRIPLE question. The evaluation label
triples_ok_species asks only whether the two taxa interact at all, regardless of
whether the retrieved relation term is the right one. Rows where the pair interacts
but the relation term is wrong are therefore labelled NO in training and YES in
evaluation -- a systematic label flip.

Monotonicity: a YES under full-triple semantics is necessarily YES under species
semantics (species-level is the weaker condition), so only the NO rows can flip.
We re-label only those.
"""
import argparse, re, sys, time
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
    ap.add_argument("--labels", required=True, help="existing full-triple labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(REPO/a.data)
    if "label" in df.columns:
        m = df.reset_index().rename(columns={"index": "row"}); lc = "label"
        return_early = True
    else:
        return_early = False
    lab = pd.read_csv(REPO/a.labels) if not return_early else None
    if not return_early:
        m = df.reset_index().rename(columns={"index":"row"}).merge(lab, on="row", how="inner",
                                                                  suffixes=("", "_old"))
        lc = "label" if "label" in lab.columns else "label_old"
    no = m[m[lc] == 0].copy()
    if a.limit: no = no.sample(min(a.limit, len(no)), random_state=1)
    print(f"{len(m):,} labelled rows; {len(no):,} are NO and can flip. Re-judging {len(no):,}.", flush=True)

    out = REPO/a.out
    done = set()
    if out.exists():
        done = set(pd.read_csv(out)["row"].tolist()); print(f"resuming, {len(done)} done", flush=True)
    rows, t0, n = [], time.time(), 0
    for _, r in no.iterrows():
        if r.row in done: continue
        s1 = r.get("source_species", r.get("species1_form"))
        s2 = r.get("target_species", r.get("species2_form"))
        txt = r.get("text", r.get("passage"))
        try:
            resp = requests.post("http://localhost:11434/api/generate", json={
                "model": a.model,
                "prompt": PROMPT.format(sent=txt, s1=s1, s2=s2),
                "stream": False, "options": {"temperature": 0, "num_predict": 4, "seed": 0},
                "think": False}, timeout=300).json().get("response", "")
        except Exception as e:
            resp = "ERR"
        rows.append({"row": int(r.row), "label_species": 1 if re.search(r"\byes\b", resp, re.I) else 0,
                     "raw": resp.strip()[:16]})
        n += 1
        if n % 200 == 0:
            pd.DataFrame(rows).to_csv(out, mode="a" if done else "w", header=not done, index=False)
            done |= {x["row"] for x in rows}; rows = []
            el = time.time()-t0
            print(f"  {len(done)}/{len(no)}  {n/el:.2f} it/s  eta {(len(no)-len(done))/(n/el)/3600:.1f}h", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out, mode="a" if done else "w", header=not done, index=False)
    print("done", round(time.time()-t0), "s", flush=True)

if __name__ == "__main__":
    main()
