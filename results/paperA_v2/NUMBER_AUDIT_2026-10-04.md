# Paper A number audit, 2026-10-04

`scripts/audit_paper_numbers.py` extracts every number in `paperA/paperA.tex` (from the abstract on)
and looks for it among all values in the result files (`results/paperA_v2/*.json`, the shipping and
rebuild JSONs; ×100 for percentages and EP F1; differences between same-level arm values for gaps),
at the printed precision. Value matches were then checked semantically for Tables 1, 2 and 3
(every cell against its own key).

Final pass: **785 numbers, 746 explained by a result file.** The other 39 were checked by hand:

| Number(s) | Source / check |
|---|---|
| +9.6 EP F1 | 67.08 − 57.46 (`biored_ep.json`) |
| 0.542 | Test299 prevalence, 162/299 |
| 48,338 rows, 0.343 positive; 20,000 / 16,546 / 6,201 / 5,591 | `v3_combined_train.csv`, `kind` column |
| 2,017 passages | pair-contrast passages with several rows and both labels, text normalised (raw text: 2,018; two passages differ only in case/spacing) |
| 22/150 = 14.7%, CP [9.4%, 21.4%] | recomputed (scipy beta) |
| 227, 0.628 (Table 2 cells) | `arg_recognition.csv` × `S_biodiv_*.npy`; all four cells recomputed |
| 5.1 / 2.6 / 1.7 taxa | `n_taxa_per_row.csv` by block (5.11 / 2.58 / 1.66) |
| 500 abstracts, 15,640 candidates | BC8 protocol; converter output |
| 0.918 ± 0.003 (base-sample model) | `models/student/xenc_s{1,2,3}`, order-free: 0.9175 ± 0.0028 |
| 42,747 | 48,338 − 5,591 reversal rows |
| 99.7%, 24% | soft relabel log (0.997); soft labels in (0.05, 0.95): 23.5% |
| 0.442, 0.527 / 0.796, 0.714 / 0.841, +0.321 | recomputed on the 122B's 1,000 BioRED rows |
| 53,000 | 30% of 175,588 |
| 6,000-row sample | fixed sample (seed 20261002) of the co_listed validation, reject precision 0.931 |
| 15/32 = 0.47, CP [0.29, 0.65]; bound 0.81, [0.76, 0.88] | recomputed |
| 0.765–0.786 | whole-abstract 3-epoch seeds' dev AUPRC (student_config.json) |

## Corrections made
- "Intervals are 20,000-sample bootstraps or Clopper–Pearson": the paper has no bootstrap interval
  any more → "Intervals are Clopper–Pearson".
- GloBI vocabulary share 99.82% → **99.83%** (165,037 / 165,325; `scripts/pool_lexical_scan.py`,
  `pool_scan.json`). The old figure had no traceable source.
- CPU throughput 32.7 → **32.3** candidates/s (abstract and teacher paragraph: 33 → 32), re-measured
  through the handoff `predict()` (`scripts/bench_cpu_handoff.py`, `cpu_bench_handoff.json`; load
  average 4.9 during the run).

## Made traceable (no change in value)
- Pool statistics: `pool_scan.json` (175,588 rows, 165,325 passages, 115,404 triplet keys).
- Fresh-sample AUPRC 0.665 / 0.785 / 0.765: `scripts/fresh_sample.py` → `fresh_sample.json`
  (the 100 rows of `eval_sets_qwen_validated.csv` not in the benchmark).

## Added today
- BioLinkBERT-large, all arms scored once (`*_linkbertL` keys); encoder paragraph and appendix updated.
- Escalation to the teacher: `scripts/cascade_tables.py` → `cascade.json`, Appendix
  "Escalating uncertain candidates to the teacher" and one body sentence.
