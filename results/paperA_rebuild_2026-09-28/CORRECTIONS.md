# Paper A: what changed when the numbers were rebuilt (2026-09-28)

paperA.tex was a 22 Sep build. docs/EVAL_AUDIT_2026-09-25.md landed three days later and
showed the empirical core was computed on the wrong label column with a threshold fitted on
the reporting set. The paper was never rebuilt. These are the corrected numbers, produced by
`scripts/rebuild_paperA_numbers.py` from `scripts/eval_unified.py` (the non-superseded harness).

## Benchmark actually used now

437 rows, 246 positive. That is 449 minus 9 `in_train` minus **3 exact duplicates of training
passages** (5-gram Jaccard 1.000) found by `results/test_contamination_scan.csv`. All three are
in biotx100, so biotx100 contributes 97 rows, not 100.

Labels are **species-level** throughout ("do these two taxa interact in this passage"), which is
what `BENCHMARKS.json` pins. The whole `sentence` column is the passage segment; only 1 of 437
rows exceeds 240 wordpieces, so 256-token truncation is not a confound here.

| block | n | positive |
|---|---|---|
| biotx100 | 97 | 75 |
| reject50 | 41 | 9 |
| test299 | 299 | 162 |

## Threshold policy

Three policies, reported side by side, because the old τ=0.219 was a function of the reporting
set's own gold prior:

- **pre-specified** τ=0.5, fixed in advance
- **block-held-out** τ fitted on two source blocks and applied to the third, so every row is
  scored by a threshold never fitted on it. This is the policy the paper should report.
- **oracle** τ maximising F1 on the reporting set. An upper bound, labelled as such.

## Corrections

| claim as printed | corrected |
|---|---|
| filter accepts 91/100, P=0.714 vs base 0.650, binomial p=0.119 | clean 97 rows, species labels: accepts 89/97, **P=0.843** vs base **0.773**, one-sided binomial **p=0.071** |
| verifier P=0.934, F1=0.905, McNemar **p=0.018** | 437 rows, block-held-out τ: P=0.825, R=0.898, **F1=0.860** against V1's 0.816; 62 fixes / 47 breaks, McNemar **p=0.180** |
| "Biotx100 shares no passage with training data" (0/100) | **3 of 100 are exact duplicates** of training passages |
| "model figures are means over three seeds" | the reported vector is a **probability ensemble** of three checkpoints. Honest per-seed figures: AUPRC **0.9099 ± 0.0060**, F1 at τ=0.5 **0.7868 ± 0.0136** |
| "thresholds ... never selected on the set being reported" | was false for τ=0.219; true only under the block-held-out policy introduced here |

## What survives

- The audit's core claim. The deployed filter is still **not** significantly better than
  accepting everything (p=0.071), so the "near-inert by construction" argument holds.
- **Recall-side recovery is the strongest result and it is unchanged in kind.** On reject50 the
  deployed filter scores F1 0.000 by construction; the verifier reaches **0.700**.
- Ranking. AUPRC 0.9099 ± 0.0060 per seed over the full benchmark.

## What does not survive

The headline significance. On the full clean benchmark with a leak-free threshold the verifier
improves F1 from 0.816 to 0.860, and that difference is **not significant** (p=0.180). Per block
it is better on test299 (0.787 -> 0.859, p=0.115), far better on reject50 (0.000 -> 0.700), and
**worse on biotx100** (0.915 -> 0.884). The paper must claim a calibrated, adjustable operating
point that reaches a recall class the deployed filter cannot reach at all -- not a significant
accuracy win.

---

# 2026-10-02: the deployed filter is removed as a baseline, and a controlled one replaces it

The deployed filter is an artefact nobody outside the project can reproduce, so it is no longer a
baseline anywhere. Its role is taken by a baseline trained here and fully described: the same
BiomedBERT-base encoder, the same 48,338 teacher-labelled rows (`data/training/distill/
v3_combined_train.csv`), the same recipe, three seeds -- with the passage alone as input
(`--input-format sentence`, added to `scripts/xenc_format.py`). Only the input differs.

| system | per-seed AUPRC | P | R | F1 (block-held-out tau) |
|---|---|---|---|---|
| sentence-only baseline | 0.8513 +/- 0.0064 | 0.730 | 0.825 | 0.775 |
| triple-query verifier  | 0.9099 +/- 0.0060 | 0.825 | 0.898 | 0.860 |

76 items fixed, 30 broken, McNemar p = 1.24e-05. Per block AUPRC: test299 0.819 -> 0.925,
biotx100 0.954 -> 0.939, reject50 0.465 -> 0.497.

## Candidate rules (scripts/candidate_rules.py)

Eight deterministic rejection rules. Seven were frozen (sha256 4d953f3f...) before evaluation;
`co_listed` was added afterwards and frozen separately (8e9e8fc3...). All eight were validated on
training data first: each rule's rejections are teacher-negative at least 88% of the time
(42,747 rows; co_listed on a fixed 6,000-row sample at 93.1%). Four rules (non_biotic_relation,
negated, not_an_organism, pathogen_modifier) were written after error analysis of this benchmark's
false positives, so improvements they produce here are an upper estimate.

| system | P | R | F1 | vs. without rules |
|---|---|---|---|---|
| triple-query, block-held-out | 0.825 | 0.898 | 0.860 | |
| + 7 rules | 0.859 | 0.890 | 0.874 | 11 fixed / 2 broken, p = 0.027 |
| + 8 rules | 0.878 | 0.878 | 0.878 | 17 fixed / 5 broken, p = 0.019 |

## The shipped two-head model

`dirhead/joint_a05_s1` (verification head + antisymmetric direction head), threshold 0.5
pre-specified on its own dev split (`threshold_dev: 0.5`):

| system | P | R | F1 |
|---|---|---|---|
| joint, tau = 0.5 | 0.852 | 0.959 | 0.902 |
| + 7 rules | 0.870 | 0.951 | 0.909 (6 fixed / 2 broken, p = 0.29) |

co_listed is net-negative on the joint model (1 FP removed, 3 TP removed) and is not used there.
The joint model already rejects most of what the rules catch.

## Open: contestable gold

Several of the most confident false positives read as gold errors (e.g. a bat tick reported to
feed on humans, labelled negative). They are NOT changed. A blind review sheet mixing 12 of them
with 12 random controls, model scores hidden, is at
`data/evaluation/gold_review_2026-10-02_BLIND.csv`; the key is kept separately.
