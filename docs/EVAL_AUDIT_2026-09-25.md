# Evaluation audit — 2026-09-25

Three independent adversarial passes over the evaluation path (leakage, train/eval format
drift, statistical validity), plus the items I verified by hand afterwards. Findings are
marked **verified** only where I reproduced them myself; the rest are reported as claims.

---

## 1. Every standing "vs V1" p-value was oracle-selected — **verified**

`scripts/summarise_arms.py:32`, `scripts/confusion_delta.py:38`, `scripts/eval_unified.py:91`,
`scripts/oracle_bounds.py:72` and `scripts/frontier_unified.py:79` all choose the challenger's
threshold by maximising F1 on the same rows the McNemar test then runs on, while V1's decisions
stay fixed. `frontier_unified.py` reports the *minimum* p over a 205-point sweep as
`best_mcnemar`.

This is not a p-value. The selection can only raise k and lower m.

I hit the same bug myself while scoring the direction models: the minimum over 99 thresholds
gave p=0.0072, but the only thresholds reaching p<0.05 were 0.01 and 0.02 — predict-everything-
positive, F1 0.80, worse than V1 and the opposite of the precision-priority policy. At the
pre-specified 0.5 the same comparison gives p=1.0.

**Fix applied:** `experiments/multitask/eval_direction.py` reports McNemar only at a
pre-specified threshold; the grid minimum is retained under
`_vs_v1_min_p_oracle_selected` so it cannot be mistaken for a p-value.

## 2. The V1 comparison was running on the wrong rows — **verified, and fixed**

The 141-row comparison subset is biotx100 + reject50, and both blocks were *defined by V1's
output*: biotx100 is what V1 accepted, reject50 is what it rejected. V1 therefore has no false
negatives in one and no false positives in the other, by construction.

Worse, it was unwinnable. V1 makes exactly **24 errors** on those 141 rows, so k ≤ 24 while m
draws on 117. Reaching p<0.05 needs |k−m| ≥ 12; every arm sat at k≈15–19, m≈13–18. I checked
this across six thresholds per arm: χ² never exceeded 0.28 against a bar of 3.84. **Any arm
breaking 11+ of V1's correct answers could not reach significance at any threshold, whatever
its true quality.**

**Fix applied:** V1's decisions on the remaining 299 rows already existed on disk
(`results/v2/base299_V1_champion.json`, threshold 0.28). They reproduce that file's confusion
matrix (TP 119 / FP 19 / FN 44 / TN 117) exactly, and the two benchmark files agree on all 299
labels. `scripts/compare_vs_v1_full.py` now runs the comparison on all 439 clean rows using
V1's actual decisions throughout — no re-run reconstruction.

## 3. The "species × pair" arm was never a pair model — **verified**

`scripts/build_and_train_species.sh:24` loops `for fmt in triple pair` and uses `$fmt` only in
the output directory name. It never passes `--input-format`. All six checkpoints record
`input_format: triple`:

```
pair_s1 triple   pair_s2 triple   pair_s3 triple
triple_s1 triple triple_s2 triple triple_s3 triple
```

So the standing table's "SPECIES × pair 0.9283 vs SPECIES × triple 0.9401" compares two sets of
three triple-format seeds. The difference is seed noise (measured spread below). **The
pair-format experiment on species-relabelled data has not been run.**

## 4. Degenerate arms reported with spectacular p-values — **verified**

`results/loss_shaping/summary.csv` reports `fixes=15, breaks=76, mcnemar_p=3.18e-10` for
`focal_g1_s1` and `focal_g2_s1`, and `fixes=9, breaks=41, p=1.16e-05` for `V3_seed2`,
`lsneg035_s1` and `wt200_s3`. Those counts are an all-positive and an all-negative prediction.
The `model_wins` column is correctly `False`, but the p-values are the smallest in the repo and
any automated scan for extreme p would surface them first. There is no degenerate-output guard.

## 5. Seed noise exceeds every encoder difference — **verified**

BiomedBERT on the identical recipe, three seeds: **0.9464 / 0.9332 / 0.9401**, sd 0.0066.
Any single-seed AUPRC gap below ~0.013 is noise. Several published arm comparisons are inside
that band.

## 6. Benchmark has no pinned hash — **verified**

`data/evaluation/BENCHMARKS.json` pins `ep_relax`, `test299` and `test500`, but not
`unified_test_set.csv` — the primary benchmark, read by nine scripts with a bare `read_csv`.
`src/eval/core.load_benchmark` exists specifically to refuse a mutated benchmark and is
imported by none of them.

Not pinned yet, deliberately: pinning now would freeze a known-bad row (see §8).

## 7. `scripts/eval_v3.py` carries three known-bad behaviours and writes results — **verified, guarded**

  1. `threshold_from_prior(tpr, y.mean())` on all three benchmarks — target prior taken from
     the reporting set's own gold labels.
  2. biotx100 scored against `triples_ok_full`, not the species-level `triples_ok_species`.
  3. Queries built from canonical `*_term` while the students were trained on surface `*_form`.

The script now refuses to run without `--i-know-this-is-superseded`.

## 8. One confirmed gold error, not yet corrected

`unified_test_set.csv` index 400 (`test299`):
`[Acanthamoeba sp] --infection--> [Pseudomonas sp]`, label **1**.

The passage says the protozoan's visceral involvement occurred *"in association with
Aspergillus sp. and/or Escherichia coli and Pseudomonas sp."* — co-occurrence in the same horse
host, with no interaction asserted between the two organisms. Confirmed by the curator's note
(*"sp1 infects horses, sp2 I would say also interacts with horses directly"*).

Should be **0**. Not changed — awaiting sign-off.

## 9. Claims checked and found not to matter

* **Pipe-joined alternates.** 64/439 benchmark rows carry them (`Gobio gobio|gudgeon`); 0 of
  48,338 training rows do. Real mismatch. Measured cost of resolving them: **+0.0000 AUPRC**
  across three seeds. Harmless.
* **Silent marker no-op** in `xenc_format.mark_passage`. The failure mode is real (returns the
  passage unchanged if neither taxon is located, with no error and no counter) but has not
  fired: 0 no-ops across 48,338 training and 439 benchmark rows.
* **Relation noise as the explanation for the test299 win.** I expected test299 to carry more
  wrong relation terms than biotx100. It does not — the relation term appears in the sentence
  in 96% of test299 rows against 100% elsewhere. The hypothesis is not supported; the plainer
  reading (test299 is simply the block V1 was never tuned on) stands.

## 10. Open, not acted on

* `scripts/eval_unified_tta.py:119,125` hard-codes the triple query and ignores each
  checkpoint's recorded `input_format`. It has not corrupted a published number (every TTA arm
  used checkpoints that legitimately default to triple) but it cannot be pointed at the V4 or
  `mark_canon` lineage as written. Note that relation-probe TTA is a **mathematical no-op** on
  `pair`, `pair_canon` and `mark_canon`, because those formats never put the relation in the
  query — verified: three different relations produce one distinct encoder input.
* Ensemble operating points average per-checkpoint thresholds and apply the mean threshold to
  the mean probability — two aggregations that do not compose. Affects ensemble arms only; a
  single checkpoint reduces to its own recorded threshold.
* ~105 arms have been scored against this benchmark with no multiplicity control. Bonferroni
  would put α at 4.8e-04.

---

## 11. Addendum — what the V1 win actually is

After fixing the comparison frame (§2), the aggregate result is p=5.0e-04 on 437 rows. Two
further checks were run before trusting it, and both changed how it should be stated.

**Block decomposition.** The significance is not uniform. Of five blocks, exactly one carries it:

| block | n | prevalence | V1 F1 | model F1 | k | m | p |
|---|---|---|---|---|---|---|---|
| biotx100 | 97 | 0.77 | 0.915 | 0.919 | 6 | 5 | 1.00 |
| reject50 | 41 | 0.22 | 0.000 | 0.692 | 9 | 8 | 1.00 |
| EP-A | 99 | 0.47 | 0.860 | 0.889 | 10 | 7 | 0.63 |
| **EP-passage** | 100 | 0.85 | 0.781 | 0.936 | 23 | 2 | **6.3e-05** |
| eval-100/BioTx-random | 100 | 0.31 | 0.691 | 0.866 | 14 | 6 | 0.12 |

Hand V1 a **per-block oracle threshold** and EP-passage collapses from p=6.3e-05 to p=0.27;
no single block stays significant.

**So the claim is prevalence robustness, not accuracy.** One fixed threshold across three
blocks spanning prevalence 0.31–0.85:

| system | threshold | prev 0.31 | prev 0.47 | prev 0.85 | spread |
|---|---|---|---|---|---|
| V1 | 0.28 | 0.691 | 0.860 | 0.781 | **0.169** |
| joint_a05_s1 | 0.50 | 0.866 | 0.889 | 0.936 | **0.070** |
| detach_s1 | 0.50 | 0.853 | 0.920 | 0.930 | 0.077 |
| species-relabel, 1 seed | 0.50 | 0.817 | 0.885 | 0.926 | 0.109 |

Per-block retuning is not available in deployment — the pipeline sees one stream — so the
fixed-threshold column is the operative one. Held to a single operating point across the whole
distribution, V1's own gold-fitted ceiling on test299 is F1 0.8418 and it still loses at
p=1.2e-03.

**Do not write "we beat the deployed filter."** That invites the per-block objection and loses
it. Write: "one operating point holds across a prevalence range where the deployed filter's
does not."

## 12. What the gain is attributable to

| arm | AUPRC | F1@0.5 | p@0.5 | oracle F1 | p@oracle |
|---|---|---|---|---|---|
| V1 | — | 0.8186 | — | — | — |
| V2 cross-encoder | 0.9163 | 0.7685 | 0.235 | 0.8600 | 0.159 |
| V3 cross-encoder | 0.9121 | 0.7804 | 0.523 | 0.8639 | 0.115 |
| V4 12-ckpt ensemble | 0.9390 | 0.7340 | 0.051 | 0.9010 | **2.97e-04** |
| + species-level relabel | 0.9460 | 0.8835 | 0.026 | 0.8876 | 9.15e-03 |
| + joint mark_canon | 0.9333 | **0.9008** | **5.04e-04** | 0.9029 | 3.65e-04 |
| + joint, frozen trunk | 0.9402 | 0.9036 | 4.86e-04 | 0.9057 | 3.54e-04 |

**V4 already beat V1** at its oracle threshold (p=2.97e-04). The earlier "we tie with V1"
finding was the measurement frame, not the model. What the joint model adds is reaching that
at a **pre-specified 0.5**, where V4 needs 0.09 and sits at p=0.051 at 0.5. AUPRC is flat from
V4 onward (0.933–0.946, inside the ±0.013 seed band), so the edge is calibration, single-model
cost and the direction output — not separability.

The largest single jump in usable F1 (0.768 → 0.884 at a fixed threshold) came from the
**label-semantics fix**, `triples_ok_species` rather than `triples_ok_full` — not from a
modelling change.

## 13. Encoder sweep — final

Nine encoders on an identical recipe; three families taken to three seeds.

| family | seeds | mean AUPRC | sd | vs BiomedBERT |
|---|---|---|---|---|
| BiomedBERT | 3 | 0.9396 | 0.0066 | — |
| DeBERTa-v3 | 3 | 0.9389 | 0.0072 | p=0.920 |
| BioLinkBERT | 3 | 0.9339 | 0.0104 | p=0.474 |
| ModernBERT / bio-DistilBERT / BiodivBERT / SapBERT / BioBERT-1.2 / SciBERT | 1 each | 0.9024–0.9297 | — | equal or worse |

**The backbone is not a lever on this task.** BiodivBERT does not win despite biodiversity
pretraining. Encoder-diversity ensembling was also tested: cross-encoder score correlation is
r=0.933 against r=0.947 for same-encoder seeds, and the best 2-checkpoint arm loses to the
single model at twice the cost.

---

## 14. Paper A — the draft's empirical core uses the wrong label column

**Verified by direct recomputation.** Every headline number in `paperA/paperA.tex` is computed
against `triples_ok_full` (is the stated relation right, in the stated direction?) rather than
`triples_ok_species` (do these two taxa interact?), which is the project's label semantics and
the column the benchmark uses.

On `results/audit_v1_v3/biotx100_audit.csv`, 100 rows:

| | triples_ok_full | triples_ok_species |
|---|---|---|
| positives | 65 | 76 |
| base rate | 0.650 | 0.760 |
| deployed filter precision (91 accepts) | 0.714 | **0.835** |
| its false positives | 26 | **15** |
| one-sided binomial vs accept-everything | p=0.1188 | p=0.0556 |

The paper states 65 / 0.650 / 0.714 / 26 / p=0.119 — the `full` column throughout.

**The significance does not survive the correction.** Recomputing the paper's Table 3 from
`paperA/fig/make_figures.py`'s own score vector:

| τ | P (full) | F1 (full) | McNemar p (full) | P (species) | F1 (species) | McNemar p (species) |
|---|---|---|---|---|---|---|
| 0.001 | 0.747 | 0.855 | 0.386 | 0.816 | 0.871 | 0.149 |
| 0.100 | 0.894 | 0.901 | 0.016 | 0.924 | 0.859 | 0.424 |
| **0.219** | **0.934** | **0.905** | **0.018** | 0.967 | 0.861 | **0.584** |
| 0.600 | 0.932 | 0.887 | 0.052 | 0.966 | 0.844 | 0.377 |

**No threshold reaches p<0.05 under species labels** (best 0.149). The verifier's *precision*
is in fact higher under species labels (0.967 vs 0.934) — but so is the deployed filter's
(0.835 vs 0.714), and the gap is no longer significant on 100 items.

**This does not sink the paper; it relocates its evidence.** The claim it was trying to make is
supported — just not by a 100-item full-triple audit. It is supported by the 437-row
species-level benchmark against V1's actual decisions: F1 0.9008 vs 0.8186, p=5.04e-04 (§11–12).
The fix is to retire the Biotx100 audit as the empirical core and rebuild the results section on
the 437-row benchmark.

### Other verified defects in the draft

* "Biotx100 shares no passage with training data; 0 of 100" — the 5-gram scan finds **3 rows at
  Jaccard 1.000** against training passages.
* "Reject50 … 11 of them true positives" — `gold_label` sums to 11, `gold_species_pair` to 12.
  Same full-vs-species split.
* "A 193-item human direction annotation is in progress" — 20 rows are annotated, 17 decidable.
* "we make no direction claim" — a direction head now scores 13/17, p=0.025.
* "24.2 pairs/s on CPU" — measured 33.9 through the shipped entry point.
* `hinton`, `bucilua`, `gu2021domain` and `mcnemar1947note` are **in the bib but never cited**,
  in a paper whose method is distillation, whose backbone is BiomedBERT, and which reports
  McNemar tests throughout.
* Two duplicate bib entries with year-contradicting keys (`mottin2021sibils`/`gobeill2020sibils`,
  `sibils2023biotxplorer`/`ruch2024biotxplorer`); `biodiversitypmc2024` lists the submitting
  author and would deanonymise an ARR submission if cited.
* Several numbers have **no artifact on disk at all**: the 99.82% lexical-scan rate, the
  teacher's precision/F1 on Biotx100, the paired-bootstrap interval [0.137, 0.308] over 20,000
  resamples, the three heuristic-ranker AUPRCs, and the "5,591 teacher-labelled reversals"
  (the reversal file has 1,761 rows).

### Not a defect, checked and cleared

`Test287` is real: `test299_with_relation.csv` filtered to non-null relation is exactly 287 rows
at 0.544 positive, matching the draft. It is a naming convention, not a phantom set. Its AUPRC
figures (0.905 / 0.921) reproduce `results/v2/v3_eval.json` and use surface forms, so they are
unaffected by the `eval_v3.py` defects — though the Biotx100 AUPRCs quoted beside them
(0.966 / 0.935) match nothing on disk.

### 14b. What the label correction does to the paper's argument — verified

Recomputed on `results/audit_v1_v3/biotx100_audit.csv`, all figures checked directly.

**V1's errors, decomposed by mechanism.** The draft's three-way taxonomy collapses to two,
because a wrong *relation term* is not a species-level error at all — 8 of the 9 predicate-class
rows have `triples_ok_species = 1`:

| | pair-binding | predicate | entity | total |
|---|---|---|---|---|
| V1's accepted-and-wrong, `triples_ok_full` | 11 | 8 | 7 | 26 |
| V1's accepted-and-wrong, `triples_ok_species` | **8** | **0** | 7 | **15** |
| all unsupported candidates, `full` | 16 | 9 | 10 | 35 |
| all unsupported candidates, `species` | **13** | **1** | 10 | **24** |

**This strengthens the thesis rather than weakening it.** Pair-binding rises from 42% to 53% of
V1's errors, and it is exactly the class query conditioning addresses. The other half is entity
resolution, which the cross-encoder does *not* fix — an upstream NER problem, and the paper
should say so.

**The component-wise bound survives, with new numbers.** Any checker that verifies components
independently cannot exceed `76r/(76r+13)` at recall `r` (was `65r/(65r+16)` under `full`), i.e.
0.854 at full recall rather than 0.802 — a harder bar. The verifier still clears it:

| | precision | recall | bound at that recall | margin |
|---|---|---|---|---|
| τ=0.219, `triples_ok_species` | 0.9672 | 0.7763 | 0.8194 | **+0.148** |

The draft claims +0.154 against the `full` bound. The argument holds at +0.148 against the
correct one. **This is the one load-bearing claim in the paper that survives the label
correction intact** — it should become the spine of the method section.

### 14c. Other corrections that go the paper's way

* Reject50: the draft says "recovers 6 of the 9 unleaked true positives". At the pre-specified
  0.5 the shipped model recovers **9 of 9**, re-admitting 8 of 32 correct rejections
  (block McNemar p=1.00).
* V1's precision on biotx100 is 0.835 under species labels, not 0.714 — the deployed filter is
  better than the draft says. The honest framing is not "the filter is near-inert" but "the
  filter is good on the distribution it was tuned for, and does not hold that operating point
  elsewhere".

---

## 15. Encoder sweep, resolved at five seeds

The single-seed conclusion in §13 ("no encoder beats BiomedBERT") was too thin for BiodivBERT,
which had exactly one seed against a ±0.013 noise band. Both contenders were taken to five.

| family | seeds | mean AUPRC | sd | vs BiomedBERT |
|---|---|---|---|---|
| BiomedBERT | 5 | **0.9395** | 0.0051 | — |
| BioLinkBERT | 5 | 0.9307 | 0.0089 | t p=0.092, Mann-Whitney p=0.151 |
| **BiodivBERT** | 5 | **0.9247** | 0.0027 | t **p=0.0005**, Mann-Whitney **p=0.0079**, Δ −0.0148 |

The conclusion changes: BiodivBERT is not merely "no better", it is **significantly worse**.
The one encoder pretrained on biodiversity literature loses on a biodiversity task. Its
variance is also unusually tight (0.0027 against BiomedBERT's 0.0067) — it converges
consistently to a slightly worse solution rather than being noisy.

All three families now stand at five seeds; the comparison is 5-vs-5.

## 16. Direction: a larger, weaker check

`direction_train.parquet` holds a 4,722-row dev split that direction training never reads
(`DirectionDS` filters `split == "train"`), taxon-disjoint from training.

| set | n | joint_a05_s1 | detach_s1 |
|---|---|---|---|
| all held-out dev | 4,644 | **0.9244** | 0.7776 |
| two independent sources agree (`BOTH`) | 1,530 | **0.9869** | 0.9444 |

**Not a replacement for human gold.** These labels come from the same dependency-parse +
GloBI rules the head was trained on, so a high score partly measures "did it learn the rule".
What it does establish, on 4,644 held-out taxon-disjoint rows instead of 17:

1. the head generalises the rule to unseen pairs rather than memorising it, and
2. the joint-vs-frozen-trunk ordering measured on human gold (13/17 vs 11/17) reproduces at
   scale (0.924 vs 0.778) — so that ordering is not a 17-item accident.

Whether the *rule itself* is right is still only answerable on human gold, still n=17.

## 17. Paper tables are now generated, not transcribed

`scripts/paper_tables.py` emits the paper's three core tables from current artifacts under
`triples_ok_species`, with `--latex` for the manuscript. Transcribed numbers drift; generated
ones cannot. Current output:

* **Table 1** — error decomposition under both label columns, side by side, so the
  `full` → `species` change is explicit rather than silent.
* **Table 2** — operating points against V1's real recorded decisions (0.50: P 0.852 /
  R 0.959 / p=2.3e-04 … 0.99: P 0.924 / R 0.744 / p=0.41).
* **Table 3** — per-block decomposition, the table a reviewer will ask for
  (test299 p=2.8e-05; biotx100 and reject50 both p=1.00).

## 18. Fixed this session

| defect | where | status |
|---|---|---|
| gold label error | `unified_test_set.csv` row 400 | corrected 1→0; p 5.0e-04 → **2.3e-04** |
| benchmark had no pinned hash | `BENCHMARKS.json` | pinned |
| TTA hardcoded the triple query | `eval_unified_tta.py:119` | reads each checkpoint's format; refuses probes on relation-free formats |
| degenerate arms ranked to the top | `eval_unified.py` | guard added, plus `core.degenerate()` |
| ensemble threshold composition | `eval_final_v4.py` | `centred_scores()`; verified identical to own threshold for one checkpoint, +0.003 F1 on a 10-checkpoint heterogeneous ensemble |
| score cache ignored the benchmark | `eval_final_v4.py:44` | key now binds benchmark hash + row count |
| hardcoded `/home/egaillac/...` | `polarity.py`, `train_direction.py` | resolved relative to `__file__` |
| `from pathlib import Path, re` | `polarity.py` | worked only because pathlib re-exports `re`; fixed |
| **silent polarity failure** | `predict_joint.py` | a missing lexicon gave every relation the same default, degrading direction to near-chance with exit code 0. Now fails loudly; verified by hiding the file |
| `eval_v3.py` still had all three historic bugs | `eval_v3.py` | guarded behind `--i-know-this-is-superseded` |

---

## 19. The dev-holdout experiment: succeeded as an experiment, failed as a fix

`train_direction.py` hardcoded `threshold_dev = 0.5`, so the shipped model's operating point
was pre-specified but never *chosen* — which is why the precision-first policy could not be
expressed honestly. The trainer now holds out a pair-grouped 10% dev split (4,722 rows, 2,881
dev pairs) and records two thresholds fitted on it: max-F1, and the lowest threshold meeting a
precision floor.

**Result 1 — the hardcoded 0.5 was right.** Dev's max-F1 threshold is **0.49**, and it gives
bit-identical benchmark numbers to 0.5 (P 0.8434, R 0.9634, F1 0.8994, p=7.6e-04). Three
months of using 0.5 was not luck, and nothing needs to change.

**Result 2 — the precision floor does not transfer, badly.**

| | |
|---|---|
| dev split AUPRC | **0.9830** |
| benchmark AUPRC | **0.9436** |
| gap | +0.0394 |

Dev says "threshold 0.19 gives precision ≥ 0.90". On the benchmark, threshold 0.19 gives
precision **0.8219**. To actually reach 0.90 on the benchmark you need **0.98** — a five-fold
miss.

| threshold | benchmark precision |
|---|---|
| 0.19 (what dev prescribed) | 0.8219 |
| 0.50 | 0.8434 |
| 0.90 | 0.8605 |
| 0.95 | 0.8843 |
| **0.98** | **0.9018** ← where P≥0.90 actually starts |

**Root cause, and it is not fixable by holding out more training data.** The dev split is drawn
from the distillation training distribution, which is markedly easier than expert-graded
candidates (AUPRC 0.983 vs 0.944). Any threshold fitted there under-shoots. Holding out 20%
instead of 10% would change nothing; the distribution is the problem, not the sample size.

**What would actually work:** a dev split drawn from the *benchmark* distribution — i.e.
spending expert-graded rows on threshold selection rather than on evaluation. With 437 clean
rows, reserving ~100 would leave ~337 for reporting. That is a real cost and a real decision,
not a free fix.

**Interim recommendation:** ship at 0.5 (validated as max-F1 by dev), and if a precision floor
is required, set it from the benchmark's own P/R curve while stating plainly that the threshold
was chosen on the reporting set and is therefore optimistic. Do not dress a dev-fitted
precision floor up as test-blind when it misses by 0.79 of the threshold range.

**Confirmed across all three seeds.** Dev max-F1 thresholds: 0.49 / 0.53 / 0.47 — all ≈0.5.
Dev P≥0.90 thresholds: 0.19 / 0.27 / 0.19, giving benchmark precision 0.822 / 0.830 / 0.806 —
none reaches the floor. The dev-holdout models are also statistically indistinguishable from
the shipped checkpoint (AUPRC 0.9392 ± 0.0041 vs 0.9336; paired bootstrap ΔAUPRC +0.0063,
CI [−0.0033, +0.0188]; head-to-head McNemar k=3, m=4, p=1.000), so `joint_a05_s1` stays.

---

## 20. Does the direction head read the passage, or recall taxon priors?

The obvious control, which had not been run: mask the passage and give the head only the two
taxon names. If accuracy holds, it is recalling that ticks parasitise mammals rather than
reading the sentence.

**On the 17-item human gold the control is inconclusive**, and that is worth stating plainly:

| input | gold (n=17) |
|---|---|
| full passage | 13/17 = 0.765 |
| taxon names only | 12/17 = 0.706 |
| same words, order destroyed | 11/17 = 0.647 |

One item. At n=17 this cannot distinguish passage-reading from taxon priors, and a direction
claim resting on the gold set alone would not survive review.

**On the 4,644 held-out taxon-disjoint rows it is unambiguous:**

| input | accuracy |
|---|---|
| full passage | **0.9244** |
| taxon names only | **0.4188** |

Paired McNemar on the 4,644 overlapping rows: full-passage-only correct 2,497, names-only-only
correct 149, χ² = 2082, **p ≈ 0**. The passage is worth **+50.6 accuracy points**.

Names-only lands *below* chance, which is what you would expect if the head has no usable signal
and falls back on a bias that is uncorrelated with the label.

**Conclusion: the head reads the passage.** The 17-item gold simply lacks the power to show it —
13 versus 12 is noise. The standing caveat on direction is therefore about the *size of the human
gold set*, not about whether the mechanism works.

One honest qualification: the 4,644 labels are derived from dependency parses of those same
passages, so what is demonstrated is that the head recovers parse-grounded direction from text it
has not seen, for taxon pairs it has not seen. Whether that rule matches human judgement is a
separate question, and the only evidence for it remains n=17.
