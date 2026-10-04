#!/usr/bin/env python3
"""Optional second opinion: let a local LLM re-decide the candidates the verifier is least sure of.

The verifier decides every candidate. The fraction --band of candidates whose score lies nearest
the threshold is then re-decided by a local LLM (served by Ollama) asked whether the passage
asserts an interaction between these two taxa. A candidate rejected by a candidate rule stays
rejected. Precision goes up, recall comes down a little (README, section 4).

    python predict.py  --in candidates.csv --out scored.csv
    python escalate.py --in candidates.csv --scored scored.csv --out final.csv --band 0.3

Needs a running Ollama with the model pulled (`ollama pull qwen3.8:27b`, Ollama >= 0.35; on an
older Ollama use `--model qwen3:32b`). Nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

PROMPT = """You verify candidate biotic-interaction pairs extracted from scientific literature.

Sentence: {sent}

Candidate pair:
  Entity 1: "{s1}"
  Entity 2: "{s2}"

Answer YES only if ALL of the following hold:
 1. Both entities are genuinely the organisms referred to by those surface strings in this sentence (not a gene, protein, chemical, author name, or a different species).
 2. The sentence asserts a direct biological interaction between these two organisms (not merely a co-mention, and not an interaction each has with some third organism), and the interaction is not negated.

Reply with exactly one word: YES or NO."""


def ask(url: str, model: str, prompt: str) -> str:
    r = requests.post(f"{url}/api/generate", timeout=600, json={
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 2, "seed": 0}})
    r.raise_for_status()
    return r.json().get("response", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="the CSV given to predict.py (needs `sentence`)")
    ap.add_argument("--scored", required=True, help="predict.py's output for that CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", type=float, default=0.3, help="fraction of candidates to escalate (0-1)")
    ap.add_argument("--threshold", type=float, default=0.5, help="the threshold predict.py used")
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    d, s = pd.read_csv(a.inp), pd.read_csv(a.scored)
    if len(d) != len(s) or not ((d.species1.astype(str).values == s.species1.astype(str).values).all()
                                and (d.species2.astype(str).values == s.species2.astype(str).values).all()):
        sys.exit("--in and --scored do not line up row by row; pass the CSV that produced --scored")
    p = s.p_interact.to_numpy()
    ruled_out = s.rejected_by_rule.fillna("").astype(str).str.len().to_numpy() > 0
    k = int(round(a.band * len(s)))
    band = np.zeros(len(s), bool)
    band[np.argsort(np.abs(p - a.threshold), kind="stable")[:k]] = True
    ask_rows = np.flatnonzero(band & ~ruled_out)            # a rule rejection is final: no call
    print(f"{len(s)} candidates; escalating {len(ask_rows)} ({a.band:.0%} band, "
          f"{int((band & ruled_out).sum())} rule-rejected skipped) to {a.model}", file=sys.stderr)

    def job(i):
        try:
            return ask(a.ollama, a.model, PROMPT.format(sent=d.sentence.iloc[i], s1=d.species1.iloc[i],
                                                        s2=d.species2.iloc[i]))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}"

    with ThreadPoolExecutor(a.workers) as ex:
        answers = dict(zip(ask_rows, ex.map(job, ask_rows)))
    llm = pd.Series([answers.get(i, "") for i in range(len(s))])
    yes = llm.str.upper().str.startswith("YES").to_numpy()
    failed = llm.str.startswith("ERROR").to_numpy()
    final = np.where(band & ~ruled_out & ~failed, yes.astype(int), s.interacts.to_numpy())
    out = s.assign(escalated=(band & ~ruled_out).astype(int), llm_answer=llm.values, interacts_final=final)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: accepted {int(s.interacts.sum())} -> {int(final.sum())}"
          + (f"; {int(failed.sum())} LLM calls failed (kept the verifier's decision)" if failed.any() else ""),
          file=sys.stderr)


if __name__ == "__main__":
    main()
