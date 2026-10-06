#!/usr/bin/env python3
"""Zero-shot pair question over every enumerated pair of the 437 biodiversity passages.

The LLM counterpart of the trained pair-max arm, needed only for the human-label evaluation
(`biodiv_sentence.py --human`; LLMs are never scored against the silver labels, which partly come
from LLM answers). Each pair that pair-max scores (`biodiv_sentence.enum_pairs`: the candidate plus
every pair of TaxoNERD mentions) is asked with the paper's zero-shot pair prompt
(`llm_baseline.BIODIV["pair"]`, one argument order, as in the paper's LLM pair rows) through
`llm_baseline.ask` (P(YES), temperature 0, thinking off) on the system Ollama.

Output: results/paperA_v2/sentence_level/llm/biodiv_<model>_enum_pair.csv (i, species1, species2,
is_candidate, p_yes, raw, verdict), appended in chunks; a re-run skips pairs already scored.

Usage
  python3 scripts/sentence_level/biodiv_llm_enum.py --model qwen3:32b --workers 4
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import llm_baseline as LB  # noqa: E402
from biodiv_sentence import benchmark, enum_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()
    out = C.OUT / "llm" / f"biodiv_{a.model.replace(':', '-')}_enum_pair.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    P = enum_pairs(benchmark()).reset_index(names="pair_row")
    done = set(pd.read_csv(out).pair_row) if out.exists() else set()
    todo = P[~P.pair_row.isin(done)]
    print(f"{len(P)} pairs, {len(P) - len(todo)} done, {len(todo)} to go ({LB.URL}, {a.model})", flush=True)

    def job(r):
        p, t = LB.ask(a.model, LB.BIODIV["pair"].format(sent=r.sentence, s1=r.species1, s2=r.species2))
        return p, t, int(t.upper().startswith("YES"))
    cols = ["pair_row", "i", "species1", "species2", "is_candidate"]
    with ThreadPoolExecutor(a.workers) as ex:
        for i in range(0, len(todo), a.chunk):
            part = todo.iloc[i: i + a.chunk]
            res = list(ex.map(job, part.itertuples(index=False)))
            part[cols].assign(p_yes=[x[0] for x in res], raw=[x[1] for x in res], verdict=[x[2] for x in res]) \
                .to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"  {min(i + a.chunk, len(todo))}/{len(todo)}", flush=True)
    o = pd.read_csv(out)
    print(f"-> {out} ({len(o)} of {len(P)} pairs)", flush=True)


if __name__ == "__main__":
    main()
