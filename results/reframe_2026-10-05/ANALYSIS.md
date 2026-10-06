# Sentence-level analysis for the Paper A reframe (analysis stage, 2026-10-05)

The claim under test: for passages retrieved by co-occurrence, **a sentence-level decision read off
pair-level verification (the max over the passage's candidate pairs) is at least as good as a model
trained to answer the sentence question directly, and only pair-level answers can populate a pair
database.**

`SL/` below means `results/paperA_v2/sentence_level/`. Every number in this file comes from a JSON file
there, except the gold-review breakdown in §2 (from the gold-review reports it names); `SL/sentence_tables.json` flattens all of them (key `numbers`, "file:dotted.path" → value) and
lists the main ones under `headline`. Unless stated otherwise: three seeds per arm, ensemble = mean of
the three seeds' scores, 95% intervals are paired percentile bootstraps (10,000 replicates, seed 0),
McNemar tests are exact (binomial).

Status: A0–A6 done. A1 (the fair biodiversity competitor) finished at 04:26
(`state/gpu_sentlab.done`). Its results are in `SL/biodiv_sentlab.json` and change the conclusion
for biodiversity (§6, §9).

---

## 1. Regenerating everything

```bash
source ~/MetaP/MPvenv/bin/activate
cd ~/MetaP/classifier
python3 scripts/sentence_level/make_sentence_tables.py                        # ~2 min, cached model scores
python3 scripts/sentence_level/make_sentence_tables.py --timing --fp32-check  # + cost timings, FP32 reproduction
```

The same first command regenerates the human-label evaluation (`SL/biodiv_human.json`) as the
437-row sheet gets answers. It also regenerates the gold-review comparison: since the 22-item review
is complete, that goes to `results/reframe_2026-10-05/gold_review_22.script.md`. The parent session's
`gold_review_22.md` is never replaced. Model scores are cached in `SL/scores/` under names keyed by checkpoint path
and a hash of its `student_config.json`. The GPU is needed only for checkpoints not scored yet.

| script (`scripts/sentence_level/`) | writes (`SL/` unless stated) | what |
|---|---|---|
| `biored_sentence.py` | `biored_sentence.json` (`--fp32`: `biored_sentence_fp32.json`) | BioRED arms, bootstrap, McNemar, strata, ceiling, LLM rows |
| `biored_label_semantics.py --split test dev` | `biored_label_semantics.json` | what the derived BioRED sentence label means |
| `biored_llm_fill.py` (+ `.sh`) | `llm/biored_qwen3-32b_sent500_{sentence,pair}.csv` | completes qwen3:32b on 500 random test sentences |
| `biodiv_sentence.py` | `biodiv_sentence.json` | biodiversity, three silver label definitions |
| `biodiv_sentence.py --sentlab` | `biodiv_sentlab.json` | + the sentence-label students (A1) |
| `biodiv_sentence.py --human` | `biodiv_human.json` | the user's SENTENCE answers (A5): 437-row sheet = benchmark; 22-item review = spot check |
| `biodiv_llm_enum.py` (+ `.sh`) | `llm/biodiv_qwen3-32b_enum_pair.csv` | qwen3:32b on every enumerated pair (for A5's LLM pair-max row) |
| `taxonerd_cost.py --parts stats,taxonerd,gpu,cpu,check` | `biodiv_enum_cost.json` | cost of the pair enumeration |
| `order_check.py` | `order_check.json` | why the new numbers differ from the sandbox's in the 5th decimal |
| `sentlab_label.py`, `sentlab_split.py`, `gpu_sentlab.sh` | `data/training/distill/v3_sentence_teacher_qwen3-32b*.csv`, `sentlab_throughput.json`, `sentlab_split.json`, `biodiv_sentlab.json` | A1, one detached job |
| `gold_review_report.py` | `results/reframe_2026-10-05/gold_review_22.script.md` (or `gold_review_22.md` if absent) | A0 comparison, only when the 22-item sheet is complete |
| `make_sentence_tables.py` | `sentence_tables.json` | runs the above, flattens every number |

A1 relaunch (it resumes: labelling appends, finished models are skipped):
`systemd-run --user --unit=reframe-gpu-sentlab-2 --working-directory=~/MetaP/classifier /bin/bash ~/MetaP/classifier/scripts/sentence_level/gpu_sentlab.sh`

---

## 2. A0: the annotation sheets

**At 01:50** both server copies were empty:
- `gold_review_2026-10-02_v2_BLIND.xlsx`: 0 answers of either kind.
- `second_annotation_2026-10-04_BLIND.xlsx`: 437 rows, 20 SKIP; 0 answers of either kind.

The key's labels were not read.

**At 02:32 the user uploaded the completed review** as `data/evaluation/gold_review_2026-10-02_v2_eg_curated.xlsx`
(CONTEXT.md, update 02:45). The parent session did the comparison with the sealed key and switched
`common.GOLD_SHEET` to that file:
- Outputs: `results/reframe_2026-10-05/gold_review_22.md`, the same text in `gold_review_22.parent.md`,
  plus `_comparison.csv` and `_whatif.json`. Read those: they carry the provenance and the per-item
  reasons.
- Cross-check: `gold_review_report.py` regenerates the comparison independently into
  `gold_review_22.script.md`, with the same numbers.

The numbers:
- **PAIR vs current gold:** 15/22 agree (68.2%), κ 0.364. By selection (`_kind`):
  - controls: 11/11 agree;
  - "both models say yes": 3/9;
  - "both models say no": 1/2.
- **Disagreements:** all 7 (items 2, 3, 4, 10, 12, 16, 18) side with both models. They are *proposed*
  flips; the gold is unchanged until the user decides.
- **SENTENCE:** YES 15, NO 4, UNSURE 3 (items 3, 4, 5; dropped). Sentence-YES with pair-NO: 5.
- **Provenance:**
  - The answers come from the user adjudicating a model's pre-filled sheet, so they are **not blind**.
  - 11 of the 22 items were selected by model-vs-gold disagreement, so they are **not a random
    sample**.
  - Their sentence answers are used only as a spot check of the silver labels (§7), never as a
    benchmark result.
- **The 437-row sheet** still has no answers (checked 02:48).

Mapping facts (from the key's `item`/`bench_row` columns):
- The gold-review key's `bench_row` indexes the 449-row `unified_test_set.csv`. All 22 items are among
  the 437 clean rows.
- Gold-review items 12 and 14 (clean rows 66 and 87) appear *unskipped* in the 437-row sheet, as items
  45 and 159. The scripts keep the two sheets separate:
  - benchmark results come from the 437-row sheet only;
  - the review is a spot check;
  - rows answered in both are reported as a consistency count.

---

## 3. A2: the scripts reproduce the sandbox

- **Biodiversity** (`SL/biodiv_sentence.json` → `reproduction_of_sandbox`):
  - Identical to the sandbox: every P, R, F1, accept count, block threshold, the pair count (6,253), the
    sentence-positive rate and the ceiling.
  - The pair-max AUPRCs differ by at most 1.8e-5 per seed and 2.3e-6 for the ensemble.
- **BioRED** (`SL/biored_sentence.json` → `difference_from_sandbox_fp32`): dev thresholds and F1 are
  identical; the AUPRCs differ by at most 1.8e-5.
  - Re-scored in FP32 (`biored_sentence.py --fp32` → `SL/biored_sentence_fp32.json`), every per-seed
    AUPRC, threshold, P, R and F1 equals the sandbox's.
  - Two ensemble AUPRCs differ in the 7th decimal. That comes from averaging the seeds in float64
    instead of float32: with a float32 mean they are equal too (checked at 04:33).
- **Cause, tested** (`SL/order_check.json`):
  - The new scripts score with TF32 matmuls (`torch.set_float32_matmul_precision("high")`), as
    `scripts/paperA_tables.py` does for every number in the paper. The sandbox scripts left PyTorch's
    full-FP32 default.
  - Re-scoring the enumerated pairs in FP32 reproduces the sandbox's per-seed and ensemble pair-max
    AUPRCs bit for bit (`fp32_all_identical: true`).
  - Row order has no effect: a shuffled order gives identical scores, max difference 0.0.
  - The paper's convention is kept for the canonical numbers. `--fp32-check` re-runs the bit-for-bit
    reproduction.
- **Consistency with the paper's saved scores:** the per-seed re-scoring of the paper's sentence and pair
  arms (TF32) matches `results/paperA_v2/S_biodiv_{sentence,pair}.npy` to 8e-8
  (`biodiv_sentence.json` → `score_consistency`).
- **Correction to the cost subagent's note:** `biodiv_enum_cost.json` reports differences up to 1.1e-3
  from those arrays. Its timing scorer ran in FP32, so that is the same TF32-versus-FP32 difference,
  not run-to-run noise.

---

## 4. A3: BioRED at the sentence level (`SL/biored_sentence.json`)

**Data.**
- 2,582 BC8 test sentences with at least one candidate pair; 0.838 are positive.
- Label = the sentence co-mentions at least one entity pair that BioRED relates in the abstract.
- Thresholds: each arm's F1-maximising threshold on the development split (BioRED's own test split, the
  BC8 protocol's development set), applied once.

**Arms.**
- `sentlab`: sentence input, trained on these sentence labels, one row per sentence. This is the fair
  sentence-level competitor.
- `sentence`: the paper's sentence arm, trained on candidate rows.
- `pair-max`: the paper's pair arm, max over the sentence's candidates.

### 4.1 Main result

| arm | AUPRC, 3 seeds (mean ± sd) | ensemble AUPRC [95% CI] | dev thr | P | R | F1 | AUROC | AP, negative class |
|---|---|---|---|---|---|---|---|---|
| pair-max | 0.966 ± 0.003 | 0.965 [0.954, 0.975] | 0.41 | 0.930 | 0.980 | **0.954** | 0.895 | 0.759 |
| sentlab | 0.952 ± 0.003 | 0.955 [0.945, 0.964] | 0.27 | 0.878 | 0.989 | 0.930 | 0.836 | 0.615 |
| sentence | 0.929 ± 0.002 | 0.930 [0.917, 0.942] | 0.13 | 0.872 | 0.976 | 0.922 | 0.762 | 0.492 |
| accept everything |, | (0.838) |, | 0.838 | 1.000 | 0.912 | 0.5 | 0.162 |

Keys: `arms.*`, `bootstrap_auprc_sentences.auprc_ci95`, `trivial_accept_all`. At this base rate F1 is
compressed near the accept-all value of 0.912. The negative-class AP (finding the sentences without a
related pair) separates the arms more clearly: 0.759, 0.615, 0.492.

### 4.2 Paired tests

| difference (ensemble AUPRC) | observed | 95% CI, sentences resampled | 95% CI, abstracts resampled as clusters |
|---|---|---|---|
| pair-max − sentlab | +0.0099 | [+0.0002, +0.0196] | **[−0.0013, +0.0208]** |
| pair-max − sentence | +0.0351 | [+0.0250, +0.0457] | [+0.0237, +0.0475] |
| sentlab − sentence | +0.0252 | [+0.0156, +0.0348] | [+0.0163, +0.0342] |

- **Per seed:** the AUPRC difference pair-max − sentlab is +0.019, +0.009 and +0.012 (seeds 1–3;
  `per_seed_auprc_differences`).
- **F1 at the development thresholds** (`bootstrap_f1_*`):
  - pair-max − sentlab: +0.024 [+0.018, +0.030] (abstract clusters [+0.017, +0.031]).
  - pair-max − sentence: +0.033 [+0.027, +0.039].
  - sentlab − sentence: +0.009 [+0.005, +0.013].
- **Exact McNemar** at the development thresholds (`mcnemar_exact`):
  - pair-max vs sentlab: 158 fixes, 41 breaks, p = 2.1e-17.
  - pair-max vs sentence: 193 / 36, p = 3.8e-27.
  - sentlab vs sentence: 75 / 35, p = 1.7e-4.

Reading:
- **Against the paper's sentence arm,** pair-max is better on every measure, with wide margins.
- **Against the fair competitor:**
  - Pair-max is clearly better at the operating point: F1, McNemar.
  - Its threshold-free advantage is small. The AUPRC gain is +0.010. Its CI excludes zero only when
    sentences are treated as independent; with abstracts as clusters it includes zero. Sentences of
    one abstract share entities and relations, so the clustered interval is the honest one.

### 4.3 Strata (`strata.*`, ensemble AUPRC, CI with sentences resampled)

| stratum | n | positive | sentlab | sentence | pair-max | pair-max − sentlab | pair-max − sentence |
|---|---|---|---|---|---|---|---|
| sentence names 2 concepts (paper's definition) | 538 | 0.775 | 0.892 | 0.909 | 0.902 | +0.010 [−0.021, +0.045] | −0.007 [−0.036, +0.023] |
| sentence names ≥ 3 concepts | 2,044 | 0.855 | 0.966 | 0.945 | 0.978 | +0.013 [+0.004, +0.021] | +0.033 [+0.023, +0.044] |
| 1 candidate pair | 777 | 0.740 | 0.852 | 0.892 | 0.896 | **+0.044 [+0.016, +0.073]** | +0.005 [−0.020, +0.030] |
| ≥ 2 candidate pairs | 1,805 | 0.880 | 0.981 | 0.972 | 0.986 | +0.006 [−0.003, +0.014] | +0.014 [+0.007, +0.022] |

Notes on the strata:
- **Two of the requested partitions coincide.** On this test set "2 vs ≥3 entities in the candidate
  pairs" and "1 vs ≥2 candidate pairs" are the same partition (777 / 1,805): two candidate entities
  always give one pair, three or more always give two or more. The JSON keeps both keys. The paper's
  "concepts named in the sentence" (`n_concepts`) counts concepts outside any candidate too, so it
  differs.
- **Difference-in-differences** (`difference_in_differences`, multi-pair stratum minus the other):

  | difference | by concepts (≥3 minus 2) | by candidate pairs (≥2 minus 1) |
  |---|---|---|
  | pair-max − sentence | +0.040 [+0.008, +0.071] | +0.010 [−0.017, +0.036] |
  | pair-max − sentlab | +0.002 [−0.032, +0.035] | **−0.038 [−0.069, −0.008]** |

What the strata show:
- **Against the paper's sentence arm,** the advantage sits in sentences that name several concepts, as
  at the pair level in the paper.
- **Against sentlab, the advantage is *not* where several pairs compete.** It sits in single-pair
  sentences (+0.044), where the sentence label and the pair label coincide. In multi-pair sentences
  pair-max and sentlab are within +0.006 [−0.003, +0.014].
- **Gold entity information does not explain this alone.** On BioRED the candidate pair comes from
  BioRED's gold entity annotations, so pair-max is told which two spans are annotated entities, and
  sentlab is not. But the paper's sentence arm is also text-only, and in single-pair sentences it
  matches pair-max (0.892 vs 0.896).
- **The pattern is that each sentence-input model is weak in one stratum:**
  - sentlab, trained one row per sentence, ranks single-pair sentences poorly (0.852). Most of its
    training sentences are multi-pair, and those are mostly positive. Of its 3,277 training sentences, 61.4% have
    several candidate pairs and 91.7% of those are positive, against 79.4% of single-pair ones
    (`sentlab_training_sentences`).
  - The paper's sentence arm, trained on candidate labels, ranks multi-pair sentences poorly (0.972 vs
    0.986).
  - Pair-max is best or tied in both strata.
- **So BioRED supports "pair-max is the robust choice across sentence types".** It does not support "the
  max over competing pairs is what beats a fair sentence model".

### 4.4 Precision ceiling of a perfect sentence filter (`perfect_filter_pair_precision`)

- **The ceiling:** accepting every candidate of every positive sentence gives a pair-level precision of
  7,684 / 14,348 = **0.536** (Wilson 95% [0.527, 0.544]).
- **By stratum:** 0.516 within sentences with ≥ 2 candidate pairs; 1.0 in single-pair sentences, by
  construction.
- **For comparison:** the paper's pair arm at its pre-specified pair-level threshold has P 0.741, R 0.911,
  F1 0.817 (`results/paperA_v2/tables_biored.json`, copied under `paper_pair_level_reference`).

### 4.5 What the BioRED sentence label means (`SL/biored_label_semantics.json`, by a subagent)

- **No evidence annotations.** No local file holds sentence- or evidence-level annotations. Files
  checked: the BioREDirect PubTator files (`~/biored_baseline/bioredirect/`), the processed TSVs, every
  `data/benchmarks/biored*` table and the converters. Relations are per document (concept pair, type,
  novelty). The label is therefore "co-mentions a related pair", never "states a relation".
- **Test split, co-mention counts:**
  - 65.1% of the 4,496 related pairs that a sentence co-mentions are co-mentioned in exactly one
    sentence.
  - 44.7% of the 2,164 positive sentences hold such a pair. The other 55.3% are positive only through
    pairs that two or more sentences co-mention.
- **How many positive sentences might state nothing:**
  - Suppose each related pair is stated in at least one sentence that co-mentions it. Then up to 1,017
    positive sentences (47.0%) could still state none of their pairs (exact per-abstract set cover).
  - With no assumption the bound is 100%.
  - A uniform-choice illustration gives 26.9%; this is not an estimate.
- **Titles:** 95.8% of title sentences are positive, against 82.3% of the others.
- **Cross-sentence relations:** 22.2% of the 6,031 test relations are never co-mentioned in one
  sentence. Sentence-scope candidates cannot see them.

### 4.6 Zero-shot LLMs at the sentence level (`llm.*`)

- **Scores:**
  - Sentence question: one call per sentence, giving P(YES).
  - Pair-max: the max P(YES) over every candidate of the sentence, with the paper's pair prompt and one
    argument order. Its greedy verdict is YES if any candidate's is.
- **Coverage:**
  - The paper's runs scored a 3,000-candidate random sample, so only 149 test sentences are complete
    (49 for Qwen3.5-122B).
  - Those sentences are biased: 143 of the 149 have a single candidate, because a sentence with many
    candidates is rarely complete.
  - So qwen3:32b was completed on a random sample of 500 test sentences (`biored_llm_fill.py`, 2,598
    new pair calls and 250 new sentence calls, same server, prompts and scorer).

| model | sentences | LLM sentence question: AUPRC / greedy F1 | LLM pair-max: AUPRC / greedy F1 | pair-max − sentence AUPRC [95% CI] |
|---|---|---|---|---|
| **qwen3:32b, random 500** (0.854 positive; accept-all F1 0.921) | 500 | 0.908 / 0.914 | 0.959 / 0.942 | **+0.051 [+0.025, +0.079]** |
| qwen3:0.6b | 149* | 0.765 / 0.867 | 0.732 / 0.867 | −0.032 [−0.102, +0.040] |
| qwen3:1.7b | 149* | 0.798 / 0.867 | 0.873 / 0.866 | +0.075 [+0.018, +0.133] |
| qwen3:4b (q4_K_M) | 149* | 0.823 / 0.863 | 0.894 / 0.838 | +0.070 [+0.006, +0.138] |
| qwen3:8b | 149* | 0.815 / 0.831 | 0.951 / 0.895 | +0.136 [+0.073, +0.200] |
| qwen3:14b | 149* | 0.829 / 0.848 | 0.936 / 0.873 | +0.108 [+0.042, +0.177] |
| qwen3:30b-a3b (q4_K_M) | 149* | 0.864 / 0.864 | 0.917 / 0.889 | +0.053 [+0.006, +0.104] |
| qwen3:32b | 149* | 0.811 / 0.857 | 0.921 / 0.892 | +0.110 [+0.042, +0.180] |
| qwen3.5:122b | 49* | 0.777 / 0.810 | 0.920 / 0.868 | +0.143 [+0.010, +0.275] |

\* Biased subset (mostly single-pair sentences). Its numbers compare sizes, not the population.

- **On the random 500**, the trained arms on the same sentences reach (AUPRC / F1 at the development
  thresholds; `trained_arms_same_sentences`): pair-max 0.973 / 0.959, sentlab 0.958 / 0.936, sentence
  0.941 / 0.932.
- **By candidate pairs** (`by_n_pairs`), within the random 500:
  - 1 pair (n = 140): LLM pair-max 0.921 vs LLM sentence 0.823.
  - ≥ 2 pairs (n = 360): 0.967 vs 0.933.
- **For the LLMs, the pair question beats the sentence question at the sentence level** at every size
  from 1.7B, as at the pair level in the paper.

---

## 5. A4: biodiversity at the sentence level (silver labels, provisional) (`SL/biodiv_sentence.json`)

**Labels.** 437 passages. A pair-positive candidate (246) always makes its passage positive. For the
other 191 (`by_silver.*`):

| definition | rule for the 191 pair-negative passages | n | positive rate | accept-all F1 |
|---|---|---|---|---|
| `majority3` (the sandbox's) | majority of three LLMs' sentence answers: Qwen3-32B (the teacher), Qwen3.5-122B, Qwen3.8-27B | 437 | 0.778 | 0.875 |
| `nonteacher2` | Qwen3.5-122B and Qwen3.8-27B: both YES → 1, both NO → 0, disagreement dropped | 394 (43 dropped) | 0.787 | 0.881 |
| `q122b` | Qwen3.5-122B alone | 437 | 0.805 | 0.892 |

- **LLM sentence-YES votes on the 191 pair-negative passages:** teacher 101, Qwen3.5-122B 106,
  Qwen3.8-27B 65 (`llm_sentence_votes_on_pair_negative_passages`).
- **LLMs are not scored against these labels** (circular); only against human labels (§7).

**Arms.**
- `sentence`: the paper's sentence arm, trained on candidate rows; the ensemble is the paper's saved
  scores.
- `pair-own`: the pair arm on the candidate only.
- `pair-max`: the pair arm, max over the candidate plus every pair of TaxoNERD mentions, order-free.

Thresholds are block-held-out (fitted on two source blocks, applied to the third), as in the paper.

### 5.1 Results

| labels | arm | AUPRC [95% CI] | seeds mean ± sd | P | R | F1 | AP, negative class |
|---|---|---|---|---|---|---|---|
| majority3 | pair-max | 0.966 [0.952, 0.977] | 0.963 ± 0.003 | 0.864 | 0.932 | 0.897 | 0.703 |
| | pair-own | 0.962 [0.947, 0.975] | 0.961 ± 0.006 | 0.901 | 0.853 | 0.876 | 0.666 |
| | sentence | 0.957 [0.939, 0.972] | 0.955 ± 0.001 | 0.915 | 0.791 | 0.849 | 0.606 |
| nonteacher2 | pair-max | 0.977 [0.966, 0.987] | 0.975 ± 0.004 | 0.884 | 0.935 | 0.909 | 0.762 |
| | pair-own | 0.974 [0.961, 0.984] | 0.972 ± 0.006 | 0.934 | 0.865 | 0.898 | 0.726 |
| | sentence | 0.963 [0.945, 0.978] | 0.962 ± 0.002 | 0.920 | 0.813 | 0.863 | 0.714 |
| q122b | pair-max | 0.974 [0.962, 0.984] | 0.972 ± 0.004 | 0.891 | 0.929 | 0.910 | 0.700 |
| | pair-own | 0.968 [0.954, 0.979] | 0.966 ± 0.006 | 0.916 | 0.838 | 0.875 | 0.647 |
| | sentence | 0.958 [0.939, 0.973] | 0.956 ± 0.002 | 0.905 | 0.838 | 0.870 | 0.552 |

| paired test | majority3 | nonteacher2 | q122b |
|---|---|---|---|
| AUPRC pair-max − sentence | +0.008 [−0.005, +0.021] | **+0.014 [+0.002, +0.027]** | **+0.016 [+0.005, +0.029]** |
| AUPRC pair-own − sentence | +0.005 [−0.008, +0.018] | +0.010 [−0.001, +0.023] | +0.010 [−0.002, +0.022] |
| AUPRC pair-max − pair-own | +0.003 [−0.006, +0.012] | +0.004 [−0.004, +0.011] | +0.006 [−0.001, +0.014] |
| F1 pair-max − sentence | +0.048 [+0.014, +0.083] | +0.046 [+0.013, +0.079] | +0.039 [+0.011, +0.069] |
| exact McNemar, pair-max vs sentence (fixes/breaks, p) | 72/49, p = 0.045 | 60/38, p = 0.033 | 60/37, p = 0.025 |

- **The order of the arms does not change** under any definition: pair-max > pair-own > sentence by
  AUPRC and by F1 (`order_changes_across_silver`).
- **The size and significance of the gap do depend on the definition.** The majority including the
  teacher gives the smallest, non-significant AUPRC gap.
- **Two cautions on F1 at these base rates:**
  - The block-held-out thresholds are mostly 0.01–0.04, near accept-all.
  - The paper's sentence arm is *below* the accept-all F1 under all three definitions (0.849 vs 0.875;
    0.863 vs 0.881; 0.870 vs 0.892). Pair-max is above it by 0.018–0.028.
  - AUPRC and the negative-class AP are the better summaries here.

**By pairs scored** (`by_silver.*.strata_by_pairs_scored`; 100 passages have only the candidate pair):

| labels | 1 pair: pair-max − sentence | ≥ 2 pairs: pair-max − sentence |
|---|---|---|
| majority3 | −0.005 [−0.030, +0.017] | +0.015 [+0.000, +0.031] |
| nonteacher2 | +0.006 [−0.008, +0.022] | +0.017 [+0.004, +0.032] |
| q122b | +0.009 [−0.006, +0.026] | +0.021 [+0.007, +0.036] |

The biodiversity gain over the paper's sentence arm comes from passages with several pairs, unlike the
BioRED gain over sentlab (§4.3).

### 5.2 The precision ceiling, with its interval (`perfect_filter_pair_precision`)

A perfect sentence filter accepts every sentence-positive candidate. Its pair-level precision:

| labels | precision | Wilson 95% | bootstrap 95% |
|---|---|---|---|
| majority3 | 246 / 340 = **0.724** | [0.674, 0.768] | [0.675, 0.769] |
| nonteacher2 | 246 / 310 = 0.794 | [0.745, 0.835] |, |
| q122b | 246 / 352 = 0.699 | [0.649, 0.744] |, |

The nonteacher2 figure is biased upward: the 43 dropped passages are all pair-negative.

### 5.3 What pair enumeration costs, and what the candidate alone gives (`SL/biodiv_enum_cost.json`)

- **Mentions and pairs:**
  - TaxoNERD finds 4.23 mentions per passage (median 3, max 29).
  - Pair-max scores 6,253 pairs: 14.3 per passage, median 4, p90 29, max 407.
  - 100 passages (22.9%) have only the candidate pair.
  - The busiest 10% of passages hold 56.9% of the pairs.
  - In 71 passages the candidate reappears as a reversed mention pair (scored twice; harmless for a max).
- **TaxoNERD finds the candidate's own strings only partly:** both strings in 35.0% of passages (41.2%
  counting "|" alternatives). That is why the candidate pair is always added.
- **Forward passes per passage:** pair-max 85.9 (2 orders × pairs × 3 seeds), sentence 3, pair-own 6.
- **GPU time** (shared A100, Ollama at 74–99% utilisation, so these are upper bounds):

  | arm | ms per passage, 3 seeds |
  |---|---|
  | pair-max | 444 (about 39× the sentence arm) |
  | sentence | 11.3 |
  | pair-own | 24 |
- **CPU time** (8 threads, 50-passage sample):

  | arm | ms per passage, 3 seeds, estimated | how it was estimated |
  |---|---|---|
  | pair-max | 3,224 | length-adjusted |
  | sentence | 118 | |
  | pair-own | 248 | |
  | TaxoNERD extraction (CPU) | 49 | plus a 15.5 s model load; GPU mode needs CuPy, not installed |
- **The candidate alone (pair-own)** gives most of the gain at 2× the sentence arm's cost:
  - AUPRC 0.962 / 0.974 / 0.968 against pair-max's 0.966 / 0.977 / 0.974.
  - pair-max − pair-own is never significant in AUPRC (table above).
  - At the operating point, pair-max − pair-own F1 is significant only under q122b (+0.034, McNemar
    p = 0.015).

### 5.4 A sentence score is not a pair decision (`pair_gold_view`)

Against the benchmark's PAIR gold:
- pair-own scores 0.923 AUPRC. The max over pairs drops to 0.840, below even the sentence arm's 0.855.
- Pair-max is a sentence decision. Populating a pair database needs the per-pair scores, not their max.

---

## 6. A1: the fair biodiversity competitor (`sentlab`) (`SL/biodiv_sentlab.json`)

**What was done** (`scripts/sentence_level/gpu_sentlab.sh`, unit `reframe-gpu-sentlab`, 01:53–04:26):
- **Teacher:** the paper's teacher (qwen3:32b, system Ollama :11434) answered the sentence question
  (`llm_baseline.BIODIV["sentence"]`, via `llm_baseline.ask`) for **all 34,242 unique corpus
  passages**.
- **Throughput** (`SL/sentlab_throughput.json`): 0.203 s per passage at 4 workers, so all passages fit
  the 2.5 h budget. Actual time: 2 h 15 min, so no sample and no matched pair arm.
- **Labels:** `data/training/distill/v3_sentence_teacher_qwen3-32b.csv`. Split 30,818 / 3,424 by
  passage (seed 0; `SL/sentlab_split.json`).
- **Students:** the paper's recipe (BiomedBERT-base, 3 epochs, seeds 1–3, `--input-format sentence`)
  in `models/sentence_level/biodiv_sentlab_s{1,2,3}`. Development AUPRC 0.985 / 0.986 / 0.986 (their
  `student_config.json`).
- **The teacher's sentence answers are not its pair answers** (`SL/sentlab_label_stats.json`):

  | teacher's sentence answer | ≥ 1 positive candidate | no positive candidate |
  |---|---|---|
  | YES (65.5% of passages) | 9,909 | 12,503 |
  | NO | 1,422 | 10,408 |

  - 33.1% of passages have at least one positive candidate; κ between the two labels is 0.26.
  - 12,503 passages are sentence-YES with no positive candidate.
  - 1,422 are sentence-NO although the teacher accepted one of their candidate pairs (inconsistent
    answers).

### 6.1 Results (block-held-out thresholds; 10,000 paired bootstrap replicates over passages)

| labels | sentlab AUPRC [95% CI] | seeds mean ± sd | P | R | F1 | AP, negative class |
|---|---|---|---|---|---|---|
| majority3 | **0.985** [0.977, 0.991] | 0.984 ± 0.001 | 0.912 | 0.947 | **0.929** | 0.849 |
| nonteacher2 | **0.987** [0.980, 0.993] | 0.987 ± 0.001 | 0.919 | 0.955 | **0.937** | 0.862 |
| q122b | **0.986** [0.978, 0.992] | 0.986 ± 0.001 | 0.926 | 0.957 | **0.941** | 0.835 |

Compare pair-max: 0.966 / 0.977 / 0.974 AUPRC, F1 0.897 / 0.909 / 0.910, negative-class AP
0.703 / 0.762 / 0.700 (§5.1).

| paired test (pair-max or pair-own vs sentlab) | majority3 | nonteacher2 | q122b |
|---|---|---|---|
| AUPRC pair-max − sentlab | −0.020 [−0.032, −0.010] | −0.010 [−0.019, −0.002] | −0.012 [−0.022, −0.004] |
| AUPRC pair-own − sentlab | −0.023 [−0.037, −0.010] | −0.014 [−0.026, −0.003] | −0.019 [−0.031, −0.007] |
| F1 pair-max − sentlab | −0.033 [−0.056, −0.010] | −0.028 [−0.052, −0.004] | −0.032 [−0.054, −0.010] |
| exact McNemar, pair-max vs sentlab (pair-max fixes / breaks, p) | 22/46, p = 0.005 | 21/39, p = 0.027 | 21/44, p = 0.006 |

**Order of the arms under every silver definition, by AUPRC and by F1:** sentlab > pair-max > pair-own
> sentence (`order_changes_across_silver`). **The fair sentence-level competitor beats pair-max,
significantly, under all three definitions.**

### 6.2 Where sentlab wins (`by_silver.*.decomposition`)

The silver label is decided by gold for the 246 pair-positive passages and by LLM sentence answers for
the 191 pair-negative ones. Split accordingly (ensemble AUPRC):

| subset | labels | n (pos) | sentlab | pair-max | pair-own | sentence | pair-max − sentlab | pair-own − sentlab |
|---|---|---|---|---|---|---|---|---|
| pair-negative passages only: is there an interaction outside the candidate pair? (label = LLM sentence answers) | majority3 | 191 (94) | 0.953 | 0.788 | 0.658 | 0.754 | −0.165 [−0.238, −0.101] | −0.296 [−0.390, −0.201] |
| | nonteacher2 | 148 (64) | 0.963 | 0.827 | 0.658 | 0.738 | −0.136 [−0.222, −0.066] | −0.305 [−0.422, −0.194] |
| | q122b | 191 (106) | 0.947 | 0.841 | 0.719 | 0.759 | −0.106 [−0.165, −0.053] | −0.228 [−0.315, −0.142] |
| gold pair-positives against silver negatives (positives decided by gold) | majority3 | 343 (246) | 0.980 | 0.971 | 0.980 | 0.962 | −0.008 [−0.020, +0.002] | +0.001 [−0.012, +0.013] |
| | nonteacher2 | 330 (246) | 0.983 | 0.980 | 0.985 | 0.965 | −0.004 [−0.014, +0.005] | +0.002 [−0.009, +0.013] |
| | q122b | 331 (246) | 0.983 | 0.979 | 0.984 | 0.964 | −0.004 [−0.013, +0.005] | +0.002 [−0.010, +0.013] |

**Reading.**
- **The whole advantage of sentlab comes from the pair-negative passages:** passages that the LLMs
  judge to describe an interaction although the candidate pair does not interact.
- **There, the silver label is an LLM's answer to exactly the prompt sentlab was distilled from:** the
  teacher's answer under majority3, two other models' answers under nonteacher2 and q122b. So part of
  the advantage is circular by construction.
- **Where the positives are decided by gold,** sentlab, pair-max and pair-own are statistically
  indistinguishable, and all beat the paper's candidate-trained sentence arm.
- **Pair-max recovers part of the "interaction outside the candidate" passages through its enumerated
  pairs:** 0.79–0.84 against 0.66–0.72 for pair-own. It misses those whose interacting organisms
  TaxoNERD does not name as a pair.
- **Whether sentlab's extra sentence-positives are real is a question for human labels.** The only
  human evidence so far, the 19-item spot check (§7), agrees with the non-teacher LLM sentence
  answers at κ 0.81 (n = 14). If that held on the random 437-row sheet, sentlab's advantage would be
  largely real.
- **A sentence score is a poor pair filter** (`pair_gold_view`): against the benchmark's PAIR gold,
  sentlab reaches 0.794 AUPRC, pair-max 0.840, the candidate-trained sentence arm 0.855, pair-own
  0.923.

---

## 7. A5: human sentence labels (`SL/biodiv_human.json`)

**Benchmark source: the 437-row sheet only** (block-balanced random order).
- **Status:** "no human labels yet (437-row sheet)".
- **What one command computes once it has answers** (`make_sentence_tables.py`, or
  `biodiv_sentence.py --human`):
  - Every arm on the labelled passages: AUPRC, and P/R/F1 at block-held-out thresholds fitted on the
    majority3 silver labels over all 437. Pre-specified, never fitted on human labels.
  - A paired bootstrap when there are ≥ 20 labelled passages. Resamples with one class only are dropped
    and counted.
  - The zero-shot LLMs on the same passages: the sentence question; pair-own; and pair-max for qwen3:32b
    once `llm/biodiv_qwen3-32b_enum_pair.csv` is complete. Unit `reframe-biodiv-llm-enum` has been
    writing it since 04:08 (3,400 of 6,253 pairs at 04:31; the row appears only when all pairs are
    scored).
  - Each row is tagged `paper_comparison_model`: only the Qwen3 3.0 family and Qwen3.5-122B are the
    paper's comparison models.
  - Human-vs-silver Cohen's kappa for each definition.
  - The PAIR answers against the benchmark's pair gold, as agreement only.
  - The count of sentence-YES / pair-NO answers.
- **Rules:**
  - UNSURE and SKIP rows are dropped; rows map through item → bench_row only.
  - The 20 SKIP rows (gold-review items) never get answers in this sheet, so the human benchmark subset
    can cover at most 417 passages, not all 437.
- **Code path:** tested with synthetic labels in memory.

**Spot check of the silver labels: the 22-item review** (`gold_review_spot_check`; not a benchmark result: not
random, not blind). 19 items have a SENTENCE YES/NO answer.

| silver definition | items compared | agreement with the user's SENTENCE answers | κ | user NO, silver YES | user YES, silver NO |
|---|---|---|---|---|---|
| majority3 | 19 | 15/19 (0.789) | 0.465 | 1 | 3 |
| nonteacher2 | 14 (5 dropped by the definition) | 13/14 (0.929) | 0.811 | 0 | 1 |
| q122b | 19 | 17/19 (0.895) | 0.683 | 1 | 1 |

- **Pair-positive items:** 5 of the 19 are pair-positive in the current gold, so their silver label is
  YES by construction.
- **Small and biased, but in one direction:** on this tiny, selected set, the two definitions without
  the teacher agree better with the user than the teacher-including majority does. Under those two
  definitions, pair-max's AUPRC gain over the paper's sentence arm is significant (§5.1).
- **Pair answers:** PAIR vs current pair gold is 15/22, κ 0.364. Sentence-YES with pair-NO: 5 of 19.

---

## 8. Caveats

1. **Biodiversity labels are silver.** Every LLM vote decides only the 191 pair-negative passages. The
   pair-positive 246 are positive by construction, which also inflates agreement between definitions
   (kappa 0.90 between majority3 and q122b). Human SENTENCE labels from the 437-row sheet may change
   the picture. The only human sentence answers so far are the 19 non-random, non-blind gold-review
   items (§7); they agree best with the non-teacher definitions.
2. **Circularity, the main threat to §6.**
   - For the 191 pair-negative passages, every silver definition is an LLM's answer to the very sentence
     prompt sentlab was distilled from: the teacher's under majority3, other models' under nonteacher2
     and q122b.
   - sentlab's whole advantage lies there (§6.2), so silver labels cannot settle sentlab vs pair-max.
   - The paper's sentence arm learned from the same teacher's pair-level answers.
3. **The BioRED label is "co-mentions a related pair".** Up to 47% of positive sentences could state
   nothing under a mild assumption; the annotations cannot settle it (§4.5).
4. **Information asymmetry on BioRED:** pair-max receives the gold entity pair, sentlab only the text.
   The text-only sentence arm matches pair-max on single-pair sentences, so this is not the whole story
   (§4.3), but it remains a difference between the arms. On biodiversity, pair-max uses automatic
   mentions (TaxoNERD) plus the retrieved candidate, which a deployed pipeline has too.
5. **High base rates** (0.84 BioRED, 0.78–0.81 biodiversity) compress F1 towards accept-all.
6. **Thresholds:** biodiversity thresholds are block-held-out on the evaluation labels of the other two
   blocks (the paper's protocol, now with silver labels); BioRED thresholds are from the development
   split.
7. **Sentence clustering:** BioRED sentences cluster within abstracts. The clustered bootstrap is
   reported next to the sentence-level one and is the more conservative; it removes significance from
   the AUPRC gain over sentlab.
8. **LLM coverage:** the LLM rows other than qwen3:32b rest on 149 (or 49) mostly single-pair sentences.
9. **Timings** were taken on a shared GPU and CPU (upper bounds); the CPU figures extrapolate from 50
   passages.
10. **Precision:** TF32 vs FP32 changes AUPRCs at the 1e-5 level and nothing else (§3).

---

## 9. What the evidence supports and what it does not

Claim: *"Ask about the pair, even if you want the sentence."*

### BioRED (exact derived labels; 2,582 test sentences)

- **Against the paper's own sentence arm** (same recipe, gold candidate labels, trained on candidate
  rows), the claim holds with margin and significance:
  - AUPRC +0.035 [+0.024, +0.047] with abstract clusters.
  - F1 +0.033.
  - McNemar p = 4e-27.
- **Against the fair competitor sentlab** (trained on the sentence labels themselves):
  - **"At least as good" holds.**
  - **"Better" holds at the operating point:** F1 +0.024 [+0.017, +0.031] with abstract clusters;
    McNemar 158 vs 41, p = 2e-17.
  - **"Better" is only marginal threshold-free:** AUPRC +0.010, CI [+0.000, +0.020] over sentences,
    [−0.001, +0.021] over abstracts.
  - **The mechanism is not what the reframe suggests.** The edge over sentlab comes from single-pair
    sentences (+0.044), not from sentences where several pairs compete (+0.006, n.s.;
    difference-in-differences −0.038 [−0.069, −0.008]).
  - What BioRED does show: each sentence-input model is weak in one stratum (sentlab on single-pair
    sentences, the paper's sentence arm on multi-pair ones), while pair-max is best or tied in both.
- **Zero-shot LLMs:** the pair question with a max beats the sentence question at the sentence level
  (qwen3:32b on 500 random sentences: +0.051 [+0.025, +0.079]; same direction for every size from 1.7B
  on the biased 149-sentence subset).

### Biodiversity (437 passages, silver sentence labels)

- **Against the paper's sentence arm** (trained on candidate labels, not a fair sentence competitor):
  - The order pair-max > pair-own > sentence holds under all three silver definitions.
  - The AUPRC gap is significant under the two definitions that exclude the teacher (+0.014; +0.016),
    not under the teacher-including majority (+0.008 [−0.005, +0.021]).
  - F1 gaps are significant under all three (+0.039 to +0.048; McNemar p 0.025–0.045), but the sentence
    arm's F1 is below accept-all under every definition, so this is partly a calibration effect.
- **Against the fair competitor (A1, sentlab), the claim fails:**
  - sentlab beats pair-max under every definition: AUPRC −0.020 / −0.010 / −0.012, all CIs below 0;
    F1 −0.028 to −0.033; McNemar p 0.005–0.027.
  - It beats pair-own by more.
- **Partly circular.** The advantage lies entirely in passages whose silver label is an LLM's answer to
  the prompt sentlab imitates (§6.2). Where the positives are decided by gold, sentlab, pair-max and
  pair-own are indistinguishable (pair-max − sentlab −0.004 [−0.014, +0.005] under nonteacher2).
- **The human spot check** (19 selected, non-blind items) agrees with the non-teacher LLM sentence
  answers at κ 0.81 (n = 14). That suggests the silver labels, and hence sentlab's advantage, may be
  largely right, but it is too small and too selected to decide.

### Pair database

- **This part is robust on both benchmarks:** a perfect sentence filter has pair-level precision 0.724
  [0.674, 0.768] on biodiversity and 0.536 [0.527, 0.544] on BioRED (0.516 in multi-pair sentences).
- **The max over pairs is a sentence decision:** as a pair filter it falls to 0.840 AUPRC against
  pair-own's 0.923.
- **Only per-pair answers populate a pair database,** and they come for free with pair verification.

### What can be claimed now

The claim as worded, "ask about the pair, even if you want the sentence", is **not supported** as a
general statement. A model trained on the teacher's answer to the sentence question is the best
sentence filter on biodiversity under every silver definition. On BioRED, pair-max beats the fair
sentence model at the operating point but only marginally threshold-free.

What the evidence does support, from strongest to weakest:
1. **Only pair-level answers can populate a pair database.**
   - A perfect sentence filter leaves pair precision at 0.724 [0.674, 0.768] on biodiversity and 0.536
     on BioRED.
   - Sentence scores are poor pair filters: as a pair filter, sentlab reaches 0.794 AUPRC and pair-max
     0.840, against pair-own's 0.923.
   - This holds on both benchmarks and does not depend on silver labels.
2. **A sentence decision read off pair verification beats a sentence model trained on candidate
   labels.** This is the paper's own sentence arm. It holds on BioRED (AUPRC +0.035, clustered CI above
   0) and on biodiversity under the two non-teacher silver definitions.
3. **For zero-shot LLMs,** the pair question with a max beats the sentence question at the sentence
   level on BioRED (qwen3:32b, 500 random sentences: +0.051 [+0.025, +0.079]).
4. **Against a sentence model trained on sentence labels:**
   - On BioRED, whose sentence label is by definition "some annotated pair is related", pair-max is at
     least as good and better at the operating point.
   - On biodiversity, whose sentence label also counts interactions outside the candidate pair, the
     sentence model is better. Its edge is exactly those interactions, which pair-max over TaxoNERD
     pairs only partly recovers.

A framing the evidence supports: *ask the pair question for the pairs and the sentence question for
the sentence.*
- The same teacher can label both. The two labels differ on a third of the corpus (κ 0.26).
- A sentence filter cannot replace pair verification for the database.
- Pair verification is a competitive but not the best sentence filter.
- No combination of the two was tested; choosing one now would need development data.

Not supported:
- pair-max beating a fairly trained sentence model on biodiversity: the opposite holds under silver
  labels;
- pair-max beating it threshold-free on BioRED: marginal, not significant with abstract clusters;
- the gain coming from pairs competing within a sentence;
- any benchmark claim on human sentence labels: none on the 437-row sheet yet.

### What would change the conclusion

1. **Human SENTENCE labels on the 437-row sheet** decide the biodiversity question.
   - If the user often says NO where the LLMs say YES on pair-negative passages, sentlab's advantage
     shrinks or reverses and pair-max could come out ahead.
   - If the user agrees with the LLMs, as on the spot check, sentlab's advantage is real.
   - The command in §1 recomputes everything.
2. **A labelled sample of pair-negative passages only** (the 191 where the arms disagree most) would
   settle the circularity question at the least annotation cost.
3. **A sentence model given the entity mentions** (e.g. all candidate entities marked, trained on
   sentence labels). If it closes the BioRED gap, the BioRED advantage is entity information, not the
   pair question.
4. **Sentence-level evidence annotation for BioRED** (or a human-checked sample of positive sentences),
   which would turn "co-mentions" into "states".
