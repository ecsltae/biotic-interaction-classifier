#!/usr/bin/env python3
"""Complete the zero-shot qwen3:32b scores of a random sample of BioRED test sentences.

The paper's BioRED LLM runs scored a 3,000-candidate random sample (`llm_baseline.py --n 3000`), so
few sentences have every candidate scored (149 of 2,582), and those few are mostly sentences with a
single candidate: no pair competes there. The sentence-level comparison needs every candidate of a
sentence. This script takes a random sample of 500 test sentences (pandas `sample(500,
random_state=0)` over the sorted sent_ids) and scores, with the same model, server, prompts
(`llm_baseline.BIORED`) and scorer (`llm_baseline.ask`):
  pair      every candidate of a sampled sentence that the paper's run did not score (one argument
            order, as in the paper's run)
  sentence  the sentence question, once per sampled sentence that has no sentence score yet
Rows already in the paper's files are reused, not re-asked. Outputs are NEW files under
`results/paperA_v2/sentence_level/llm/` (outside `results/paperA_v2/llm/`, so the paper's tables
script never sees them); appended in chunks, resumable.

Usage
  python3 scripts/sentence_level/biored_llm_fill.py --workers 2
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import llm_baseline as LB  # noqa: E402

MODEL = "qwen3:32b"
TEST = REPO / "data/benchmarks/biored_bc8/test.csv"
PAPER = REPO / "results/paperA_v2/llm"
OUT = REPO / "results/paperA_v2/sentence_level/llm"
N_SENT = 500


def sample_sentences(t: pd.DataFrame, n: int = N_SENT) -> list[str]:
    """The fixed random sample of test sentences (seed 0)."""
    return sorted(pd.Series(sorted(t.sent_id.unique())).sample(n, random_state=0))


def run(jobs: pd.DataFrame, out: Path, prompt: str, workers: int, chunk: int = 100) -> None:
    """Score `jobs` (columns sentence, species1, species2, ...) and append to `out`, skipping done rows."""
    done = set(pd.read_csv(out).key) if out.exists() else set()
    todo = jobs[~jobs.key.isin(done)]
    print(f"{out.name}: {len(jobs)} needed, {len(jobs) - len(todo)} done, {len(todo)} to go", flush=True)

    def job(r):
        p, t = LB.ask(MODEL, prompt.format(sent=r.sentence, s1=r.species1, s2=r.species2))
        return p, t, int(t.upper().startswith("YES"))
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(todo), chunk):
            part = todo.iloc[i: i + chunk]
            res = list(ex.map(job, part.itertuples(index=False)))
            part.assign(p_yes=[x[0] for x in res], raw=[x[1] for x in res], verdict=[x[2] for x in res]) \
                .to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"  {min(i + chunk, len(todo))}/{len(todo)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    assert LB.URL.startswith("http://localhost:11434"), f"expected the system Ollama, got {LB.URL}"
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(TEST, dtype={"pmid": str})
    t = t.rename(columns={"text": "sentence", "source_species": "species1", "target_species": "species2"})
    t["row"] = t.index
    samp = set(sample_sentences(t))
    sub = t[t.sent_id.isin(samp)]
    pair = pd.read_csv(PAPER / f"biored_{MODEL.replace(':', '-')}_pair.csv")
    sent = pd.read_csv(PAPER / f"biored_{MODEL.replace(':', '-')}_sentence.csv")
    cols = ["sentence", "species1", "species2", "label", "pmid", "n_concepts", "row", "sent_id"]
    pj = sub[~sub.row.isin(pair.row)][cols].assign(key=lambda x: x.row)
    have = set(t.loc[sent.row, "sent_id"])
    sj = (sub[~sub.sent_id.isin(have)].sort_values("row").groupby("sent_id", as_index=False).first()
          .assign(species1="", species2="", label=lambda x: x.sent_id.map(sub.groupby("sent_id").label.max())))
    sj = sj[cols].assign(key=lambda x: x.sent_id)
    run(sj, OUT / "biored_qwen3-32b_sent500_sentence.csv", LB.BIORED["sentence"], a.workers)
    run(pj, OUT / "biored_qwen3-32b_sent500_pair.csv", LB.BIORED["pair"], a.workers)
    print("done", flush=True)


if __name__ == "__main__":
    main()
