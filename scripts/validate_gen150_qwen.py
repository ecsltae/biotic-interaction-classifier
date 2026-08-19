#!/usr/bin/env python3
"""
Independently validate the 150 newly-generated test sentences via Qwen3.5-122B,
the same validator used elsewhere in this project (validate_eval_sets.py).
Keeps only sentences where Qwen agrees with the intended gold label —
deliberately NOT filtered by any classifier model, to avoid biasing the
test set toward any model under evaluation.

Usage:
    python classifier/scripts/validate_gen150_qwen.py
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "classifier/data/evaluation/gen_set_150_extension_raw.csv"
OUT_FILE = ROOT / "classifier/data/evaluation/gen_set_150_extension_qwen_validated.csv"

MODEL = "qwen3.5:122b"
OLLAMA_URL = "http://localhost:11434/api/chat"
N_KEEP = 97

PROMPT_TEMPLATE = (
    "Does this sentence describe a direct biotic interaction between two named organisms? "
    "Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, "
    "symbiosis, seed dispersal, competition, pathogen infection, or disease transmission. "
    "The sentence must describe an actual interaction occurring, not just mention organisms. "
    "Answer YES or NO only.\n\n"
    "Sentence: {sentence}"
)


def ask_qwen(sentence: str, timeout: int = 300) -> int:
    """Returns 1=YES, 0=NO, -1=unclear."""
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {"temperature": 0, "num_predict": 10, "num_ctx": 2048},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip().upper()
        if re.search(r"\bYES\b", text):
            return 1
        if re.search(r"\bNO\b", text):
            return 0
        return -1
    except Exception as e:
        print(f"  ERROR: {e}")
        return -1


def main():
    df = pd.read_csv(IN_FILE)
    print(f"Loaded {len(df)} candidate sentences ({df['label'].sum()} intended positive)")

    qwen_labels = []
    t0 = time.time()
    for i, row in df.iterrows():
        ql = ask_qwen(row["sentence"])
        qwen_labels.append(ql)
        agree = "✓" if ql == row["label"] else ("? " if ql == -1 else "✗ DISAGREE")
        print(f"  [{i+1}/{len(df)}] intended={row['label']} qwen={ql}  {agree}")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    ({elapsed:.0f}s elapsed, {elapsed/(i+1):.1f}s/sentence)")

    df["qwen_label"] = qwen_labels
    df["agrees"] = df["qwen_label"] == df["label"]

    n_agree = df["agrees"].sum()
    print(f"\nQwen agreement: {n_agree}/{len(df)} ({100*n_agree/len(df):.1f}%)")

    passed = df[df["agrees"]].sample(frac=1, random_state=42).reset_index(drop=True)
    if len(passed) < N_KEEP:
        print(f"WARNING: only {len(passed)} passed, need {N_KEEP}. Keeping all that passed.")
        kept = passed
    else:
        kept = passed.head(N_KEEP)

    print(f"\nKeeping first {len(kept)} sentences (by gold-label/Qwen agreement, "
          f"NOT filtered by any classifier model):")
    print(f"  Positives: {kept['label'].sum()}  Negatives: {(kept['label']==0).sum()}")
    print(kept["category"].value_counts())

    df.to_csv(OUT_FILE, index=False)
    kept[["sentence", "label", "category", "difficulty"]].to_csv(
        OUT_FILE.with_name("gen_set_150_extension_kept97.csv"), index=False)
    print(f"\nFull validation log -> {OUT_FILE}")
    print(f"Kept {len(kept)} -> {OUT_FILE.with_name('gen_set_150_extension_kept97.csv')}")


if __name__ == "__main__":
    main()
