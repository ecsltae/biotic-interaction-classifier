#!/usr/bin/env python3
"""Every sentence-level number the reframe draft may cite, regenerated with one command.

Runs, in order (each step is its own script; see its docstring):
  1. biored_label_semantics.py --split test dev   what BioRED's derived sentence label means (CPU)
  2. biored_sentence.py                           BioRED sentence level: arms, bootstrap, McNemar,
                                                  strata, ceiling, zero-shot LLM rows
  3. biodiv_sentence.py                           biodiversity, three silver label definitions
  4. biodiv_sentence.py --sentlab                 + the sentence-label students (if trained)
  5. biodiv_sentence.py --human                   the user's SENTENCE answers ("no human labels yet"
                                                  until the sheets are filled)
  6. gold_review_report.py                        results/reframe_2026-10-05/gold_review_22.md, only
                                                  once all 22 gold-review items have both answers
optional:
  --timing      taxonerd_cost.py (all parts): enumeration cost; timings depend on the machine's load
  --fp32-check  order_check.py and biored_sentence.py --fp32: the sandbox's FP32 numbers, bit for bit
Then it flattens every result JSON into results/paperA_v2/sentence_level/sentence_tables.json:
`numbers` maps "<file>:<dotted.path>" to each value, and `headline` lists the main numbers with
their sources. Model scores are cached (results/paperA_v2/sentence_level/scores/), so a re-run
needs the GPU only for checkpoints it has not scored yet.

Usage
  python3 scripts/sentence_level/make_sentence_tables.py
  python3 scripts/sentence_level/make_sentence_tables.py --timing --fp32-check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
SENTLAB = [C.REPO / f"models/sentence_level/biodiv_sentlab_s{k}/student_config.json" for k in (1, 2, 3)]
FILES = ["biored_sentence.json", "biored_label_semantics.json", "biodiv_sentence.json", "biodiv_sentlab.json",
         "biodiv_human.json", "biodiv_enum_cost.json", "order_check.json", "biored_sentence_fp32.json",
         "sentlab_throughput.json", "sentlab_split.json", "sentlab_label_stats.json"]

# (label, file, dotted path) of the numbers most likely to be cited
HEADLINE = [
    ("BioRED sentences, positive rate", "biored_sentence.json", "positive_rate"),
    ("BioRED accept-all F1", "biored_sentence.json", "trivial_accept_all.F1"),
    *[(f"BioRED {a} {k}", "biored_sentence.json", f"arms.{a}.{k}") for a in ("pair-max", "sentlab", "sentence")
      for k in ("auprc_mean", "auprc_sd", "auprc_ensemble", "P", "R", "F1")],
    *[(f"BioRED AUPRC {d} ({u})", "biored_sentence.json", f"bootstrap_auprc_{u}.diff.{d}")
      for d in ("pair-max - sentlab", "pair-max - sentence") for u in ("sentences", "abstracts")],
    *[(f"BioRED McNemar {d}", "biored_sentence.json", f"mcnemar_exact.{d}") for d in ("pair-max vs sentlab", "pair-max vs sentence")],
    ("BioRED perfect sentence filter, pair precision", "biored_sentence.json", "perfect_filter_pair_precision.precision"),
    ("BioRED LLM qwen3-32b random500", "biored_sentence.json", "llm.qwen3-32b random500.bootstrap_auprc.diff"),
    ("BioRED share of positive sentences possibly stating nothing (max, stated assumption)", "biored_label_semantics.json",
     "splits.test.concept_key_unit.positive_sentences.non_stating_bounds.max_if_each_pair_stated_in_at_least_one_co_mentioning_sentence.share"),
    *[(f"biodiv [{s}] {a} {k}", "biodiv_sentence.json", f"by_silver.{s}.arms.{a}.{k}")
      for s in ("majority3", "nonteacher2", "q122b") for a in ("pair-max", "pair-own", "sentence") for k in ("auprc", "F1")],
    *[(f"biodiv [{s}] AUPRC pair-max - sentence", "biodiv_sentence.json", f"by_silver.{s}.bootstrap_auprc.diff.pair-max - sentence")
      for s in ("majority3", "nonteacher2", "q122b")],
    *[(f"biodiv [{s}] accept-all F1", "biodiv_sentence.json", f"by_silver.{s}.trivial_accept_all.F1")
      for s in ("majority3", "nonteacher2", "q122b")],
    ("biodiv perfect sentence filter, pair precision (majority3)", "biodiv_sentence.json",
     "by_silver.majority3.perfect_filter_pair_precision"),
    ("biodiv pairs per passage", "biodiv_sentence.json", "enumeration.mean_pairs_per_passage"),
    *[(f"biodiv sentlab [{s}] {a} {k}", "biodiv_sentlab.json", f"by_silver.{s}.arms.{a}.{k}")
      for s in ("majority3", "nonteacher2", "q122b") for a in ("sentlab", "pair-max", "pair-own") for k in ("auprc", "F1")],
    *[(f"biodiv sentlab [{s}] AUPRC {d}", "biodiv_sentlab.json", f"by_silver.{s}.bootstrap_auprc.diff.{d}")
      for s in ("majority3", "nonteacher2", "q122b") for d in ("pair-max - sentlab", "pair-own - sentlab")],
    ("human labels status", "biodiv_human.json", "status"),
]


def run(args: list[str]) -> None:
    print(f"$ {' '.join(args)}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *args], check=True, cwd=C.REPO)
    print(f"  ({time.time() - t0:.0f}s)", flush=True)


def flatten(x, pre: str = "") -> dict:
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flatten(v, f"{pre}.{k}" if pre else str(k)))
        return out
    return {pre: x}


def get(obj: dict, path: str):
    """Follow a dotted path whose keys may themselves contain dots-free spaces and dashes."""
    cur = obj
    parts = path.split(".")
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):      # longest key first (keys like "pair-max - sentlab")
            k = ".".join(parts[i:j])
            if isinstance(cur, dict) and k in cur:
                cur, i = cur[k], j
                break
        else:
            return None
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing", action="store_true", help="re-measure the pair-enumeration cost")
    ap.add_argument("--fp32-check", action="store_true", help="re-run the FP32 reproduction of the sandbox")
    ap.add_argument("--boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    s = str(HERE)
    b = ["--boot", str(a.boot)]
    run([f"{s}/biored_label_semantics.py", "--split", "test", "dev"])
    run([f"{s}/biored_sentence.py", *b])
    run([f"{s}/biodiv_sentence.py", *b])
    if (C.REPO / "data/training/distill/v3_sentence_teacher_qwen3-32b.csv").exists():
        run([f"{s}/sentlab_label_stats.py"])
    if all(f.exists() for f in SENTLAB):
        run([f"{s}/biodiv_sentence.py", "--sentlab", *b])
    else:
        print("sentence-label students not trained yet: biodiv_sentlab.json not (re)generated", flush=True)
    run([f"{s}/biodiv_sentence.py", "--human", *b])
    run([f"{s}/gold_review_report.py"])          # never replaces the parent's gold_review_22.md
    if a.timing:
        run([f"{s}/taxonerd_cost.py", "--parts", "stats,taxonerd,gpu,cpu,check"])
    if a.fp32_check:
        run([f"{s}/order_check.py"])
        run([f"{s}/biored_sentence.py", "--fp32", *b])
    numbers, have = {}, {}
    for f in FILES:
        p = C.OUT / f
        if p.exists():
            have[f] = json.loads(p.read_text())
            numbers.update({f"{f}:{k}": v for k, v in flatten(have[f]).items()})
    head = [{"label": lab, "source": f"{f}:{path}", "value": get(have[f], path) if f in have else None}
            for lab, f, path in HEADLINE]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "command": "python3 scripts/sentence_level/make_sentence_tables.py",
           "files": sorted(have), "headline": head, "numbers": numbers}
    f = C.dump(out, "sentence_tables.json")
    missing = [h["label"] for h in head if h["value"] is None]
    print(f"-> {f} ({len(numbers)} numbers; headline entries without a value yet: {len(missing)})")


if __name__ == "__main__":
    main()
