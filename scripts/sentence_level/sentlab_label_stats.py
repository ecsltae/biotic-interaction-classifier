#!/usr/bin/env python3
"""How the teacher's SENTENCE answers relate to the corpus's candidate (pair) labels.

For every passage the teacher labelled (`data/training/distill/v3_sentence_teacher_qwen3-32b.csv`):
its sentence answer versus "at least one of the passage's candidate rows in
`v3_combined_train.csv` is positive" (the passage-level max of the pair-question labels the paper's
arms learned from). Tells what the sentence-label students (sentlab) learn that the candidate-trained
arms do not.

Usage
  python3 scripts/sentence_level/sentlab_label_stats.py   # -> results/paperA_v2/sentence_level/sentlab_label_stats.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

D = C.REPO / "data/training/distill"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    lab = pd.read_csv(D / "v3_sentence_teacher_qwen3-32b.csv").drop_duplicates("text").set_index("text")
    c = pd.read_csv(D / "v3_combined_train.csv")
    pos = c.groupby("text").label.max().reindex(lab.index)
    ncand = c.groupby("text").size().reindex(lab.index)
    s, p = lab.label.astype(int), pos.astype(int)
    a = lab.answer.astype(str).str.strip().str.upper()
    out = {"labelled_passages": int(len(lab)), "corpus_passages": int(c.text.nunique()),
           "sentence_yes_rate": float(s.mean()), "any_positive_candidate_rate": float(p.mean()),
           "well_formed_answers": float(a.str.startswith(("YES", "NO")).mean()),
           "p_yes_between_0.1_and_0.9": float(((lab.p_yes > 0.1) & (lab.p_yes < 0.9)).mean()),
           "crosstab": {"sentence_yes_candidate_pos": int(((s == 1) & (p == 1)).sum()),
                        "sentence_yes_no_candidate_pos": int(((s == 1) & (p == 0)).sum()),
                        "sentence_no_candidate_pos": int(((s == 0) & (p == 1)).sum()),
                        "sentence_no_no_candidate_pos": int(((s == 0) & (p == 0)).sum())},
           "sentence_yes_rate_by_candidates": {k: float(s[(ncand == 1) if k == "1" else (ncand >= 2)].mean())
                                               for k in ("1", ">=2")},
           "note": "sentence_no_candidate_pos counts passages where the teacher said a candidate pair interacts "
                   "but answered NO to 'does the passage describe a biotic interaction?' (inconsistent answers)"}
    out["kappa_sentence_vs_any_candidate"] = C.kappa(s.to_numpy(), p.to_numpy())
    f = C.dump(out, "sentlab_label_stats.json")
    print(f"{out['labelled_passages']} passages: sentence YES {out['sentence_yes_rate']:.3f} vs any positive "
          f"candidate {out['any_positive_candidate_rate']:.3f}; {out['crosstab']} -> {f}")


if __name__ == "__main__":
    main()
