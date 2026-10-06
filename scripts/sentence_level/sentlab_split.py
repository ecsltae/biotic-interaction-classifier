#!/usr/bin/env python3
"""Training and development files for the biodiversity sentence-label students (sentlab).

One row per teacher-labelled passage (`sentlab_label.py`). The development set is 10% of the
passages (numpy RandomState(0) over the passages sorted by text), held out by passage; it selects
the epoch and the development threshold in `experiments/multitask/train_student.py`. The split is
built once: if its files exist they are reused unchanged.

If only a sample of the corpus passages was labelled, the matched pair arm is built too: the corpus
rows (`v3_combined_train.csv`, candidate labels) whose passage is in the labelled training passages,
with the corpus rows of the same development passages as its development set. Then the sentence and
pair questions are compared on the same passages, and the comparison does not confound the question
with the amount of data.

Usage
  python3 scripts/sentence_level/sentlab_split.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/training/distill"
LABELS = D / "v3_sentence_teacher_qwen3-32b.csv"
CORPUS = D / "v3_combined_train.csv"
TRAIN, DEV = D / "v3_sentence_teacher_qwen3-32b_train.csv", D / "v3_sentence_teacher_qwen3-32b_dev.csv"
PTRAIN, PDEV = D / "v3_pair_matched_train.csv", D / "v3_pair_matched_dev.csv"
N_PASSAGES = 34242


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--summary", default="results/paperA_v2/sentence_level/sentlab_split.json")
    a = ap.parse_args()
    lab = pd.read_csv(LABELS).drop_duplicates("text")
    if TRAIN.exists() and DEV.exists():
        tr, dv = pd.read_csv(TRAIN), pd.read_csv(DEV)
        print(f"split exists, reused: train {len(tr)} / dev {len(dv)}")
    else:
        texts = np.array(sorted(lab.text.astype(str)))
        perm = np.random.RandomState(0).permutation(len(texts))
        dev_set = set(texts[perm[: int(len(texts) * a.dev_frac)]])
        lab = lab.assign(source_species="", target_species="", interaction_type="")
        cols = ["text", "source_species", "target_species", "interaction_type", "label", "p_yes"]
        tr, dv = lab[~lab.text.isin(dev_set)][cols], lab[lab.text.isin(dev_set)][cols]
        tr.to_csv(TRAIN, index=False); dv.to_csv(DEV, index=False)
        print(f"wrote {TRAIN.name} ({len(tr)}) and {DEV.name} ({len(dv)})")
    n_split = int(len(tr) + len(dv))           # what the students train on (a reused split may predate later labels)
    summ = {"labelled_passages": n_split, "labels_file_passages": int(len(lab)), "corpus_passages": N_PASSAGES,
            "train_passages": int(len(tr)), "dev_passages": int(len(dv)),
            "train_pos_rate": float(tr.label.mean()), "dev_pos_rate": float(dv.label.mean()),
            "sample_only": bool(n_split < N_PASSAGES)}
    if n_split < N_PASSAGES:
        c = pd.read_csv(CORPUS)
        trs, dvs = set(tr.text.astype(str)), set(dv.text.astype(str))
        if not (PTRAIN.exists() and PDEV.exists()):
            c[c.text.isin(trs)].to_csv(PTRAIN, index=False); c[c.text.isin(dvs)].to_csv(PDEV, index=False)
        summ["pair_matched_train_rows"] = int(len(pd.read_csv(PTRAIN)))
        summ["pair_matched_dev_rows"] = int(len(pd.read_csv(PDEV)))
        print(f"matched pair arm: {PTRAIN.name} ({summ['pair_matched_train_rows']}) / "
              f"{PDEV.name} ({summ['pair_matched_dev_rows']})")
    out = REPO / a.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
