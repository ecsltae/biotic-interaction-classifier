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
