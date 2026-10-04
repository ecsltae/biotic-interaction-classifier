#!/usr/bin/env python3
"""Species-level labels from a local LLM teacher, the way v4_species_train.csv was built.

The deployed verifier is trained on species-level labels: does the passage assert that these two
taxa interact, whatever the retrieved relation term? A YES under the full-triple prompt is
necessarily a YES at species level, so only the triple-level NO rows are re-judged, with the
species prompt of scripts/relabel_species_level.py (unchanged), and a row's species label is
1 if its triple label is 1 or the teacher now answers YES.

Differences from relabel_species_level.py, which built the qwen3:32b version: the server is
configurable (OLLAMA_URL, for the user-level Ollama that serves newer models), calls run in
parallel, and P(yes) is kept alongside the verdict.

Usage
  OLLAMA_URL=http://127.0.0.1:11435 python3 scripts/species_relabel_llm.py --model qwen3.8:27b \\
      --triple data/training/distill/v3_combined_train_qwen38.csv \\
      --out data/training/distill/v4_species_train_qwen38.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from llm_baseline import ask  # noqa: E402
from relabel_species_level import PROMPT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--triple", required=True, help="corpus with this teacher's triple-level labels")
    ap.add_argument("--out", required=True, help="corpus with species-level labels (new file)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = pd.read_csv(REPO / a.triple)
    log = REPO / a.out.replace(".csv", "_rejudged.csv")
    done = set(pd.read_csv(log).row) if log.exists() else set()
    todo = [i for i in d.index[d.label == 0] if i not in done]
    print(f"{len(d)} rows; {int((d.label == 0).sum())} triple-level NO to re-judge; {len(done)} done, "
          f"{len(todo)} to go", flush=True)

    def job(i):
        r = d.loc[i]
        prompt = PROMPT.format(sent=r.text, s1=r.source_species, s2=r.target_species)
        for _ in range(3):
            try:
                p, t = ask(a.model, prompt)
                return {"row": int(i), "p_yes": p, "raw": t[:12]}
            except Exception as e:  # noqa: BLE001
                err = type(e).__name__
                time.sleep(5)
        return {"row": int(i), "p_yes": float("nan"), "raw": "ERR:" + err}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k in range(0, len(todo), 500):
            chunk = list(ex.map(job, todo[k:k + 500]))
            pd.DataFrame(chunk).to_csv(log, mode="a", header=not log.exists(), index=False)
            n = k + len(chunk); rate = n / (time.time() - t0)
            print(f"  {len(done) + n}/{len(done) + len(todo)}  {rate:.2f} rows/s  "
                  f"eta {(len(todo) - n) / rate / 3600:.1f} h", flush=True)
    j = pd.read_csv(log).drop_duplicates("row").set_index("row")
    flip = j.p_yes.reindex(d.index).fillna(0) >= 0.5
    out = d.assign(label=((d.label == 1) | flip).astype(int))
    out.to_csv(REPO / a.out, index=False)
    print(f"done: {int(j.p_yes.isna().sum())} failed calls (kept NO); {int(flip.sum())} rows flipped to YES; "
          f"positive rate {d.label.mean():.3f} -> {out.label.mean():.3f}; wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
