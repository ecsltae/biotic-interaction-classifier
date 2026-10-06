# Regression check before the 2026-10-06 gold flips

Date: 2026-10-06. Purpose: show that the code changes made for the gold revision (label-version
switch in `src/eval/core.py`, benchmark loaders, LLM rows scored against the benchmark labels, new
stored fields, `scripts/paperA_derived_numbers.py`) leave every existing number unchanged on the old
labels, and that every number the paper printed by hand is now reproduced by a script.

**Verdict: identical.** Every pre-existing key of every regenerated JSON is bit-identical to the
backup (maximum absolute difference 0), every score vector and every CSV is byte-identical, both
PNG figures are byte-identical and the PDFs differ only in their CreationDate. Every derived number
reproduces the value printed in the pre-flip paper, except two printed roundings that were already
off before this change (l.290/l.300 `+0.084`, l.502/l.1065 `p = 0.067`; see the table).

## How it was run

All runs used `--labels pre_review_2026-10-06`
(`data/evaluation/unified_test_set.pre_review_2026-10-06.csv`, sha256
`057416074764bdb1c6697d95fa58b68b9755176f6b2d2ad2e78b4ae3656b547b`). At the time of the run
`data/evaluation/unified_test_set.csv` had the same SHA (nothing flipped yet), so this is also the
`current` label set of that moment. Every output went to
`results/paperA_v2/regression_2026-10-06_pre_review/` (no file in `results/` was overwritten), and
was compared with `results/paperA_v2/backup_2026-10-06_pre_flips/`. GPU: one A100, CUDA_VISIBLE_DEVICES=0.

```
R=results/paperA_v2/regression_2026-10-06_pre_review; L="--labels pre_review_2026-10-06"
python3 scripts/paperA_tables.py --bench biodiv $L --out-dir $R                 # base arms, 45 s
python3 scripts/paperA_tables.py --bench biodiv --arms soft      $L --out-dir $R
python3 scripts/paperA_tables.py --bench biodiv --arms linkbert  $L --out-dir $R
python3 scripts/paperA_tables.py --bench biodiv --arms linkbertL $L --out-dir $R
python3 scripts/paperA_tables.py --bench biodiv --arms qwen38    $L --out-dir $R
python3 scripts/paperA_tables.py --bench biodiv --llm-only       $L --out-dir $R   # as the current llm block was made
python3 scripts/eval_shipping.py --model models/dirhead/joint_a05_s1 --name a05 $L --out-dir $R/shipping
python3 scripts/eval_shipping.py --model models/dirhead/joint_v3_s1  --name v3  $L --out-dir $R/shipping
python3 scripts/cascade_tables.py          $L --out-dir $R --bench-scores $R/shipping/bench_a05.csv
python3 scripts/cascade_tables.py --deploy $L --out-dir $R --bench-scores $R/shipping/bench_a05.csv
python3 paperA/fig/make_threshold_figure.py $L --results-dir $R --out-dir $R/fig
python3 scripts/paperA_derived_numbers.py  $L --results-dir $R --bench-scores $R/shipping/bench_a05.csv
python3 scripts/model_card_tables.py       $L      # -> results/model_card_tables_pre_review_2026-10-06.json (new file)
```

## Pre-existing keys against the backup

| file | pre-existing leaves | identical | missing | different | new leaves |
|---|---|---|---|---|---|
| tables_biodiv.json | 1635 | 1635 | 0 | none | 477 (accept_everything, base_rate, benchmark_sha256, benchmark_sha256_linkbert, benchmark_sha256_linkbertL, benchmark_sha256_qwen38, benchmark_sha256_soft, labels, llm, positives_by_block) |
| cascade.json | 127 | 127 | 0 | none | 73 (benchmark_sha256, co_listed_on_accept_set, labels, llm, rules_vs_no_rules, student_no_rules) |
| cascade_deploy.json | 127 | 127 | 0 | none | 78 (benchmark_sha256, co_listed_on_accept_set, labels, llm, rules_vs_no_rules, student_no_rules) |
| eval_a05.json | 70 | 70 | 0 | none | 2 (benchmark_sha256, labels) |
| eval_v3.json | 70 | 70 | 0 | none | 2 (benchmark_sha256, labels) |
| model_card_tables (current, 2026-10-02) vs --labels pre_review_2026-10-06 | 304 | 303 | 0 | `labels` 'current' -> 'pre_review_2026-10-06' | 0 () |

The only "different" leaf is the `labels` field of the model-card JSON, which names the label version
by design (`current` then, `pre_review_2026-10-06` now). New leaves are the added provenance and
summary fields (`labels`, `benchmark_sha256[_suffix]`, `base_rate`, `accept_everything`,
`positives_by_block`; in the cascade files `student_no_rules`, `rules_vs_no_rules`,
`co_listed_on_accept_set` and a `model_alone_reaches_P` entry per band), and in `tables_biodiv.json`
the LLM rows of eight newer answer files (medgemma-27b, qwen3-32b-v035, qwen3.8-27b-v035) plus the
trained arms `*_linkbert`, `*_linkbertL`, `*_qwen38` in `trained_arms_same_rows` (the stored block was
last refreshed when only the base and `_soft` vectors existed).

Byte-identical files: the 15 `S_biodiv_*.npy` (base, `_soft`, `_linkbert`, `_linkbertL`, `_qwen38`);
`bench_a05.csv`, `direction_a05.csv`, `bench_v3.csv`, `direction_v3.csv`; `false_positives_pair.csv`
(39 rows, now written by `scripts/paperA_derived_numbers.py`; before it had no writer);
`fig4_threshold.png` and `fig4_threshold_wide.png`. `fig4_threshold{,_wide}.pdf`: identical
content, different CreationDate. The base-sample model's per-seed scores
(`per_seed/S_biodiv_triple_basesample.npy`, new) give exactly the hand-computed 0.9175 +- 0.0028.

BioRED is unaffected by the flips, but `llm_rows` changed for both benchmarks, so its LLM block was
also recomputed with `--bench biored --llm-only` in a scratch directory: all 224 pre-existing leaves
identical (the new leaves are trained arms added after the block was made).

## Numbers the paper printed by hand, now scripted

Printed values from the pre-flip paper (`results/paperA_v2/backup_2026-10-06_pre_flips/paperA/paperA.tex`,
2026-10-04 12:35; line numbers refer to that copy). Computed values from the regression run
(`derived_biodiv.json`, `tables_biodiv.json`, `cascade.json`).

| paper line | number | printed | regenerated | status |
|---|---|---|---|---|
| 108 | positives in passages naming >= 3 taxa | 154 of 246 (63%) | 154 of 246 (63%) | same |
| 128 | Biotx100 GloBI / random positives | 50 with 43, 47 with 32 | 50 with 43, 47 with 32 | same |
| 132 | Reject50 clean rows / positives | 41, 9 | 41, 9 | same |
| 134 | Test299 positive rate | 0.542 | 0.542 | same |
| 220 | Accept everything AUPRC / P / R / F1 | 0.563 0.563 1.000 0.720 | 0.563 0.563 1.000 0.720 | same |
| 252 | both taxa, <=2: n, sentence, pair | 33, 0.945, 0.970 | 33, 0.945, 0.970 | same |
| 253 | both taxa, >=3: n, sentence, pair | 227, 0.878, 0.947 | 227, 0.878, 0.947 | same |
| 254 | not a taxon, <=2: n, sentence, pair | 128, 0.856, 0.908 | 128, 0.856, 0.908 | same |
| 255 | not a taxon, >=3: n, sentence, pair | 49, 0.628, 0.808 | 49, 0.628, 0.808 | same |
| 275-276 | pair ahead of sentence (of 99 thresholds), smallest F1 gap | 99, 0.058 | 99, 0.058 | same |
| 288 | teacher gap, pair - sentence question | +0.155 | +0.155 | same |
| 289-290 | teacher gap on >= 3 / <= 2 taxa | +0.180, +0.104 | +0.180, +0.104 | same |
| 290 | student gap on >= 3 / <= 2 taxa (ensembles) | +0.084, +0.042 | +0.085, +0.042 | DIFFERS: printed as the difference of the rounded Table 2 cells (0.930 - 0.846 = 0.084); the unrounded difference is 0.0847. Not a code issue. |
| 291 | student pair (per-seed mean) minus teacher sentence question | 0.13 | 0.13 | same |
| 292 | teacher triple vs pair question | 0.946 against 0.943 | 0.946 against 0.943 | same |
| 295 | teacher (triple question) minus pair student, AUPRC | 0.027 | 0.027 | same |
| 297 | pair student F1 vs teacher triple-question F1 | 0.871 against 0.870 | 0.871 against 0.870 | same |
| 300-301 | pair query gain, >= 3 / <= 2 taxa (ensembles) | +0.084, +0.042 | +0.085, +0.042 | DIFFERS: same as l.290: 0.930 - 0.846 of the rounded cells; unrounded 0.0847. |
| 302 | candidates with a non-taxon argument | 41% | 41% | same |
| 305 | coinciding stratum: gain, n | +0.026 on 33 | +0.026 on 33 | same |
| 305-306 | not-a-taxon gains, <= 2 / >= 3 | +0.052, +0.180 | +0.052, +0.180 | same |
| (patch a) | negatives in the coinciding stratum | 13 of 33 | 13 of 33 | same |
| 407-408 | base-sample triple model: per-seed AUPRC, block-held-out F1 | 0.918 +- 0.003, 0.871 | 0.918 +- 0.003, 0.871 | same |
| 462-463 | pair FPs: total, >= 3 taxa, non-taxon argument | 39, 27, 17 | 39, 27, 17 | same |
| 500 | pair FPs caught by the 8 rules | 6 | 6 | same |
| 502 | triple + 8 rules: P, fixed / broken, p | 0.820 -> 0.862, 14 / 5, p = 0.067 | 0.820 -> 0.862, 14 / 5, p = 0.066 | DIFFERS: the stored p is 0.066457 (14 fixed / 5 broken), which rounds to 0.066; 0.067 comes from rounding twice (0.0665 -> 0.067). Pre-existing; the triple rows of tables_biodiv.json hold the same value. |
| 535-536 | deployed model, rules vs none: fixed / broken, p | 6 / 2, p = 0.29 | 6 / 2, p = 0.29 | same |
| 536-537 | co_listed on the deployed accept set: FP removed / TP removed | 1 / 3 | 1 / 3 | same |
| 541-542, 1007 | model + rules alone first reaches the 30% (122B) band's precision | tau 0.99, recall 0.740 | tau 0.99, recall 0.740 | same |
| 705-707 | per-seed AUPRC, seeds 1-3 (sentence; pair; triple) | .858 .851 .845; .922 .921 .912; .912 .905 .909 | .858 .851 .845; .922 .921 .912; .912 .905 .909 | same |
| 705-707 | per-seed F1 at tau = 0.5, seeds 1-3 | .632 .698 .663; .790 .758 .788; .842 .847 .834 | .632 .698 .663; .790 .758 .788; .842 .847 .834 | same |
| 709-710 | per-seed F1 at tau = 0.5, mean / sd | .664 .779 .841 / .033 .018 .006 | .664 .779 .841 / .033 .018 .006 | same |
| 821-825 | Reject50: positives; recovered / readmitted / F1 (sentence, pair, triple) | 9; 7/15/0.452, 4/7/0.400, 6/5/0.600 | 9; 7/15/0.452, 4/7/0.400, 6/5/0.600 | same |
| 962 | 0.6B AUPRCs | 0.64 | 0.64 | same |
| 964 | pair - sentence question at 1.7B | +0.040 | +0.040 | same |
| 964-965 | pair - sentence question from 4B on (min, max) | +0.10 to +0.20 | +0.10 to +0.20 | same |
| 966 | sentence question from 4B on (min, max) | 0.74-0.80 | 0.74-0.80 | same |
| 968 | pair - sentence question at 122B | +0.195 | +0.195 | same |
| 973 | 122B sentence question | 0.773 | 0.773 | same |
| 1054-1056 | pair: P R F1; +7 rules P R F1 fixed/broken; +8 rules | 0.849 0.894 0.871; 0.858 0.886 0.872 3/2; 0.867 0.874 0.870 6/5 | 0.849 0.894 0.871; 0.858 0.886 0.872 3/2; 0.867 0.874 0.870 6/5 | same |
| 1059-1060 | triple: +7 rules / +8 rules P R F1 fixed/broken | 0.840 0.878 0.859 7/2; 0.862 0.866 0.864 14/5 | 0.840 0.878 0.859 7/2; 0.862 0.866 0.864 14/5 | same |
| 1065 | McNemar p, pair +7/+8, triple +7/+8 | 1.00 1.00 0.18 0.067 | 1.00 1.00 0.18 0.066 | DIFFERS: as l.502: 0.0665 rounded twice; the value is 0.066. |

38 of 42 rows reproduce exactly. The 4 that do not come from two
roundings that were already in the paper: the code reproduces the underlying values, and the printed
digits were derived from them by rounding differently. For the new labels the same conventions matter: see FLIPS_2026-10-06.md.

Also reproduced from pre-existing keys (all identical, so not repeated here): Table 1, Table 2
(`<= 2` / `>= 3` rows and blocks), the curve table, the ablation table, the soft, encoder and scaling
tables, the cascade table, and the deployed model's 0.934 / 0.851 / 0.955 / 0.900 / 0.869 / 0.947 / 0.907.
