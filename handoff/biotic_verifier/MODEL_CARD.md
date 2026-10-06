> **Note for this package (added 2026-10-05).** This is the full model card from the classifier
> repository, copied here for reference. Two things differ from the package you have: run the
> model with `predict.py` in this folder (not `experiments/multitask/predict_joint.py`), and
> `direction` now has a fourth value, `BIDIRECTIONAL` (see README §3). Paths such as
> `results/…`, `scripts/…` and `models/…` refer to the classifier repository and are not
> included in this package.

# Classifier V2: joint interaction + direction model

One model, two outputs, CPU deployment. Replaces the V1 sentence-level classifier for the
BiotXplorer triple-filtering step.

**Checkpoint:** `models/dirhead/joint_a05_s1`
**Run it with:** `experiments/multitask/predict_joint.py`

---

## What it does

Given a candidate triple and the passage it was extracted from, it answers two questions:

| output | meaning | type |
|---|---|---|
| `p_interact` / `interacts` | do these two taxa interact, according to this passage? | symmetric |
| `direction` | which taxon is the subject of the relation? | `FORWARD` / `REVERSE` / `UNCERTAIN` |

V1 answered only the first, and answered it about the *sentence* rather than about the
*pair*: two different candidate triples extracted from the same sentence received the same
score. This model conditions on the pair, so it can separate them.

## Why the direction output is trustworthy by construction

The encoder never sees the stored argument order. Segment A is the two taxa **sorted
alphabetically**; segment B is the passage with the alphabetically-first taxon wrapped in
`@…@` and the second in `#…#`. Swapping `species1` and `species2` therefore produces a
byte-identical input.

The direction head is antisymmetric by construction: `s = g(hA,hB,·) − g(hB,hA,·)`, so
`P(@ is subject) = 1 − P(# is subject)` exactly, at every parameter setting. Decoding back
to FORWARD/REVERSE happens outside the model by comparing the `@` taxon to the stored
`species1`.

Consequence, verified on the shipped inference path: swapping the input order leaves
`p_interact` bit-identical and flips `p_species1_is_subject` exactly (0.9990 ↔ 0.0010).
The model cannot contradict itself about a pair by being handed it the other way round.

The relation string reaches only the head, as a 2-valued polarity embedding. The encoder
never reads it, so the relation-surface shortcut that made earlier attempts degenerate is
unavailable to it.

## Size and speed

| | |
|---|---|
| encoder | BiomedBERT-base, 109,483,778 params |
| direction head | 594,465 params (+0.54%) |
| **total** | **110,078,243 (1.005× V1)** |
| CPU throughput | **33.9 pairs/s**, 8 threads, batch 16, measured through `predict_joint.py` itself (median of 3, loadavg 4). V1 binary-only on the same rig: 33.8. **Direction costs nothing.** ~122,000 pairs/hour. |

The size budget was "about twice V1". It came in at essentially the same size.

## Measured performance

Benchmark: `data/evaluation/unified_test_set.csv`, **437 rows**, 251 positive, after removing
12 near-duplicates of training passages (one loader for every script: `clean_benchmark()` in
`src/eval/core.py`). Unless marked as measured at earlier labels, every table on these 437
rows is produced by `scripts/model_card_tables.py`
(→ `results/model_card_tables_current.json`), at the labels in force since the 2026-10-06 blind
re-annotation, which changed seven rows (six negative → positive, one positive → negative) after
the 2026-09-25 revision of row 400 (an *Acanthamoeba* and *Pseudomonas* candidate, positive →
negative). The same tables at the labels before the 2026-10-06 revision are in
`results/model_card_tables_pre_review_2026-10-06.json`; `results/vs_v1_full.json`
(`scripts/compare_vs_v1_full.py`) has not been rerun and is also at those labels. The version of
this card before the row-400 revision is `MODEL_CARD_joint_v2.pre_goldfix.md`. The shipped `in_train` column flags 9 by exact match;
a 5-gram Jaccard scan against all 48,338 training passages finds 3 more, all in biotx100 and
all at Jaccard 1.000: exact duplicates differing only in mojibake (`na√Øve` vs `naive`).

**test299 is clean**: 0 rows above Jaccard 0.5, maximum 0.176, mean 0.008. That matters more
than the total, because the whole V1 result lives in test299.

### Interaction classification

| | AUPRC | F1 | precision | recall |
|---|---|---|---|---|
| **joint_a05_s1 @ 0.5** | 0.9705 | 0.9167 | 0.874 | 0.964 |
| V1 (actual decisions) | n/a | 0.8201 | 0.863 | 0.781 |

Threshold 0.5 is pre-specified. No threshold anywhere in this card was chosen by
maximising a metric on the reporting set.

### Against V1

V1's decisions are used as V1 actually made them, with no re-run reconstruction anywhere:

* **150 rows** (biotx100 + reject50; 138 of them in the clean set): the decisions the deployed
  pipeline recorded.
* **299 rows** (test299): V1's own stored probabilities from
  `results/v2/base299_V1_champion.json`, at the threshold 0.28 recorded with them. They
  reproduce that file's confusion matrix (TP 119 / FP 19 / FN 44 / TN 117) exactly against its
  own copy of the labels. The two benchmark files differ at row 400 (revised 2026-09-25) and at
  the four test299 rows revised on 2026-10-06; on current gold V1's test299 confusion is
  TP 120 / FP 18 / FN 44 / TN 117.

Both sources are assembled into `results/v1_decisions_449.npy` by
`scripts/build_v1_decisions.py` (`--check` verifies the shipped file byte for byte).

That covers every clean row, so the comparison runs on 437, not on the 141-row overlap that
every previous comparison was stuck with.

| | F1 | precision | recall |
|---|---|---|---|
| **joint_a05_s1 @ 0.5** | **0.9167** | 0.874 | 0.964 |
| V1 | 0.8201 | 0.863 | 0.781 |

| block | n | V1 F1 | model F1 | k (fixes) | m (breaks) | McNemar p |
|---|---|---|---|---|---|---|
| biotx100 | 97 | 0.921 | 0.926 | 6 | 5 | 1.00 |
| reject50 | 41 | 0.000 | 0.786 | 11 | 6 | 0.33 |
| **test299** | **299** | **0.795** | **0.923** | **49** | **13** | **8.8e-06** |
| **all** | **437** | **0.820** | **0.917** | **66** | **24** | **1.5e-05** |

Read this honestly: **the win comes entirely from test299.** On biotx100 and reject50 it is a
tie, as expected, because those two blocks were *defined* by V1's own decisions (biotx100 is
what V1 accepted, reject50 is what it rejected), so V1 has no false negatives in one and no
false positives in the other by construction. test299 is the only block V1 was neither tuned
nor selected on.

Why the earlier comparisons said "tie": they ran only on the 141-row biotx100+reject50
overlap, where V1 made just 24 errors at the labels of the time, so any challenger breaking 11+ of V1's correct
answers could not reach p<0.05 at all, whatever its true quality.

**Objections tested, not assumed.**

* *"It is just V1's 0.28 threshold under-recalling on a new distribution."* Given V1 its
  gold-fitted **oracle** threshold on test299 (maximally generous, an upper bound it could
  not achieve in deployment), V1 reaches F1 0.8512 and still loses: k=33, m=9, p=3.9e-04.
* *"The challengers were distilled on corpora overlapping test299; V1 was not."* A 5-gram
  Jaccard scan puts test299's maximum overlap with any training passage at 0.176. No
  contamination to attribute the result to.
* *"Precision-first policy is violated."* Not violated, but barely improved: at 0.5 the
  model's precision is only slightly above V1's (0.874 vs 0.863) and it spends almost its whole gain on recall. See the operating-point section below: it is fixable, and
  the fix needs a dev split the current trainer does not hold out.

**Multiplicity.** p=1.5e-05 clears 0.05, Bonferroni over the five arms tested here (α=0.01),
and Bonferroni over the ~105 arms historically scored against this benchmark
(α=4.8e-04). At earlier labels it was 2.2e-04 (before the 2026-10-06 revision) and 5.0e-04, just
above that line (before the row-400 revision). Treat it as one pre-registered comparison at a pre-specified
threshold, not as the survivor of a 105-arm search.

## Why it wins, and the honest limit of the claim

The aggregate p=1.5e-05 is not a uniform superiority. Decomposed by block:

| block | n | prevalence | V1 F1 | model F1 | k | m | p |
|---|---|---|---|---|---|---|---|
| biotx100 | 97 | 0.78 | 0.921 | 0.926 | 6 | 5 | 1.00 |
| reject50 | 41 | 0.27 | 0.000 | 0.786 | 11 | 6 | 0.33 |
| EP-A | 99 | 0.51 | 0.874 | 0.922 | 11 | 6 | 0.33 |
| **EP-passage** | 100 | 0.85 | 0.781 | 0.936 | 23 | 2 | **6.3e-05** |
| eval-100/BioTx-random | 100 | 0.29 | 0.679 | 0.892 | 15 | 5 | 0.044 |

**One block carries most of the significance** (EP-passage, p=6.3e-05; eval-100/BioTx-random
reaches p=0.044 now that row 400, which it holds, is a negative). If V1 is handed a *per-block
oracle* threshold, EP-passage collapses to p=0.27 and eval-100/BioTx-random to p=0.15, and no
individual block remains significant.

That is not a refutation: it is the mechanism. Look at what one fixed threshold does across
three blocks spanning prevalence 0.29 to 0.85:

| system | threshold | prev 0.29 | prev 0.51 | prev 0.85 | spread |
|---|---|---|---|---|---|
| V1 | 0.28 | 0.679 | 0.874 | 0.781 | **0.195** |
| **joint_a05_s1** | 0.50 | 0.892 | 0.922 | 0.936 | **0.043** |
| detach_s1 | 0.50 | 0.879 | 0.951 | 0.930 | 0.073 |
| species-relabel, 1 seed | 0.50 | 0.812 | 0.916 | 0.926 | 0.114 |

**The claim is prevalence robustness, not raw accuracy.** V1's F1 swings 0.195 across
prevalence at its single fixed threshold; this model swings 0.043. Retuning per block is not
available in deployment (the pipeline sees one stream), so the fixed-threshold column is the
one that matters. Held to a single operating point across the whole distribution, V1's own
gold-fitted ceiling on test299 is F1 0.8512 and it still loses at p=3.9e-04.

State it that way in any write-up. "We beat the deployed filter" invites the per-block
objection and loses; "we hold one operating point across a prevalence range where the
deployed filter cannot" is what the data actually supports.

## Where the gain actually came from

Every generation on the same 437 rows, against V1's actual decisions. `oracle` sweeps the
threshold on this set: an upper bound, not an achievable operating point.

| arm | AUPRC | F1@0.5 | P | R | p@0.5 | oracle F1 | p@oracle |
|---|---|---|---|---|---|---|---|
| V1 deployed (sentence-level) | n/a | 0.8201 | 0.863 | 0.781 | n/a | n/a | n/a |
| V2 cross-encoder, triple | 0.9364 | 0.7890 | 0.930 | 0.685 | 0.648 | 0.8767 | 0.027 |
| V3 cross-encoder, triple | 0.9425 | 0.7963 | 0.950 | 0.685 | 0.927 | 0.8806 | 0.018 |
| V4 12-checkpoint ensemble | 0.9590 | 0.7512 | 0.969 | 0.614 | 0.164 | 0.9138 | **2.43e-05** |
| + species-level relabel | 0.9647 | 0.8954 | 0.830 | 0.972 | 4.47e-03 | 0.8996 | 1.23e-03 |
| **+ joint mark_canon (shipped)** | 0.9705 | **0.9167** | 0.874 | 0.964 | **1.55e-05** | 0.9187 | 1.07e-05 |
| + joint, frozen trunk | 0.9663 | 0.9193 | 0.869 | 0.976 | 1.64e-05 | 0.9213 | 1.14e-05 |

Three things this says that are easy to get wrong:

1. **V4 already beat V1**: at its oracle threshold, p=2.4e-05. The earlier "we tie with V1"
   conclusion was a measurement artifact (the 141-row frame), not a property of the model.
   What the joint model adds is that it reaches the same place at a **pre-specified 0.5**,
   where V4 needs a threshold of 0.09 to get there and sits at p=0.16 at 0.5.
2. **So the joint model's edge is mostly calibration, size and capability, not separability.**
   AUPRC moves little from V4 onward (0.959–0.970, a spread of 0.011 against seed noise of about
   ±0.008). It wins because it is well-calibrated at a fixed threshold, is one 110M checkpoint
   rather than twelve, and emits direction.
3. **The species-level relabel is the largest single jump in usable F1** (0.789 → 0.895 at a
   fixed 0.5). That was a label-semantics fix (`triples_ok_species`, not `triples_ok_full`),
   not a modelling change.

## Operating point, and the precision-first policy

At the pre-specified 0.5 the model wins on F1 and recall but is only slightly above V1 on
precision. The score ranking does contain points that beat V1 more clearly on **both** axes:

| threshold | precision | recall | F1 | McNemar p |
|---|---|---|---|---|
| 0.50 (shipped) | 0.874 | 0.964 | 0.917 | 1.5e-05 |
| 0.90 | 0.894 | 0.908 | 0.901 | 1.6e-04 |
| **0.95** | **0.906** | **0.884** | 0.895 | 4.4e-04 |
| V1 | 0.863 | 0.781 | 0.820 | n/a |

0.95 beats V1 on precision *and* recall. **But it was found by looking at this table.**

### The dev-holdout retrain: 0.5 validated, the precision floor not solved

`train_direction.py` now holds out a pair-grouped 10% dev split and fits thresholds on it.
Three seeds, three independent dev splits. The benchmark figures in this subsection (the last
column of the table, the 0.806–0.830 range, the 0.98 threshold and the 0.944 AUPRC) were
measured at earlier labels and have not been regenerated:

| seed | dev max-F1 threshold | dev P≥0.90 threshold | benchmark P at that threshold |
|---|---|---|---|
| 1 | 0.49 | 0.19 | 0.822 |
| 2 | 0.53 | 0.27 | 0.830 |
| 3 | 0.47 | 0.19 | 0.806 |

**Result 1: 0.5 is validated.** All three dev splits independently choose ≈0.5 for max F1.
The hardcoded value was not luck, and the shipped operating point needs no change.

**Result 2: the precision floor does not transfer.** Dev prescribes ~0.19–0.27 for P≥0.90;
on the benchmark those give 0.806–0.830, never 0.90. Reaching 0.90 on the benchmark needs
**0.98**. The cause is distributional, not statistical: the dev split comes from the
distillation training data (AUPRC 0.983) which is far easier than expert-graded candidates
(0.944). Holding out more training data cannot fix this.

Setting a precision floor honestly requires a dev split from the **benchmark** distribution,
i.e. spending expert-graded rows on threshold selection rather than evaluation (~100 of the
437, leaving ~337 to report on). That is a real cost and an open decision.

**Until then:** ship at 0.5, and if a precision floor is needed, read it off the benchmark's
own curve while saying plainly that it was chosen on the reporting set and is optimistic.

### The shipped checkpoint does not change

The three dev-holdout models are statistically indistinguishable from `joint_a05_s1` (figures
measured at earlier labels, not regenerated): AUPRC 0.9392 ± 0.0041 against 0.9336, paired-bootstrap ΔAUPRC +0.0063 with 95% CI
[−0.0033, +0.0188], head-to-head McNemar at 0.5 giving k=3, m=4, p=1.000. `joint_a05_s1`
stays; the retrain's value was validating its threshold, not replacing it.

**Note which arms clear it.** The BiomedBERT `triple` controls, the V2/V3/V4 lineage, do
not (p=0.0045 / 0.0094 / 0.060 across three seeds, straddling 0.05, none near the multiplicity line). Only the two joint
`mark_canon` models do. Several things differ between them at once (input format, joint
direction objective, no dev holdout), so this is not an attribution to any single cause.

### Direction

Human gold: **131 curated items, 84 decidable** (`data/evaluation/direction_gold_v2.csv`):
20 from the first curation round and 112 from a second sheet covering 79 distinct relation forms.
One curator-flagged duplicate removed (a double-encoded copy of its neighbour).

| | accuracy (n=84) | |
|---|---|---|
| **direction head** | **66/84 = 0.786** | p = 6.7e-08 vs chance |
| same head, passage hidden (names only) | 39/84 = 0.464 | chance; the passage is worth **+32 points** |
| stored order | 9/17 = 0.529 (first batch) | |

Consistent across batches: 13/17 on the first 20, 53/67 on the second sheet.

**By relation form**: the head is strong where the relation word carries direction and weak
where it does not, which is exactly the split the curator flagged in their own notes:

| relation form | n | accuracy | mean confidence |
|---|---|---|---|
| prep-headed (*pathogen of*, *ectoparasite of*) | 33 | **0.909** | 0.956 |
| present participle (*infecting*) | 7 | 0.857 | 0.877 |
| bare noun (*host*, *pathogen*, *infection*) | 40 | **0.725** | 0.615 |
| past participle (*infested with*) | 4 | 0.250 | 0.806 |

The past-participle cell does **not** replicate on 490 held-out distant-labelled rows (0.898),
so the 1/4 is either noise or a human-vs-parse disagreement about passives. Four items; do not
build a rule on it.

With abstention on head confidence:

| confidence cutoff | coverage | accuracy | 95% CI |
|---|---|---|---|
| none | 100% | 0.786 | [0.690, 0.869] |
| 0.40 | 83% | 0.871 | [0.789, 0.944] |
| **0.60 (shipped)** | **75%** | **0.889** | [0.810, 0.957] |
| 0.71 | 74% | 0.887 | [0.803, 0.956] |
| 0.90 | 57% | 0.896 | [0.800, 0.978] |

0.60 weakly dominates 0.71 (one more item answered at the same accuracy), and 0.40 is
statistically indistinguishable from 0.60 (Δ +0.018, CI [−0.012, +0.059]).

The gold set is now 84 decidable items, not 17, and the passage-ablation control is decisive at
that size. What it still cannot do is separate this head from a *good dependency-parse rule*:
that needs roughly 2,700 items. 23 rows of the second sheet remain unannotated, and
`direction_curation_SHORT.xlsx` (35 rows) is untouched.

### A larger, weaker check: held-out silver

`direction_train.parquet` carries a 4,722-row dev split that direction training never saw
(`DirectionDS` reads `split == "train"` only), taxon-disjoint from training:

| set | n | joint_a05_s1 | detach_s1 |
|---|---|---|---|
| all held-out dev | 4,644 | **0.9244** | 0.7776 |
| rows where two independent sources agree (`BOTH`) | 1,530 | **0.9869** | 0.9444 |

**Read this as a consistency check, not as accuracy.** These labels come from the same
dependency-parse + GloBI rules the head was trained on, so a high score partly measures
"did it learn the rule", not "is the rule right". What it does establish, on 4,644 held-out
taxon-disjoint rows rather than 17, is that (a) the head generalises the rule to unseen
pairs rather than memorising, and (b) the joint/frozen-trunk ordering seen on human gold
(13/17 vs 11/17) reproduces at scale (0.924 vs 0.778), so that ordering is not a
17-item accident.

Human gold remains the only measure of whether the *rule itself* is right, and it is n=17.

### Does it read the passage, or recall taxon priors?

Masking the passage and giving the head only the two taxon names:

| input | gold (n=17) | held-out dev (n=4,644) |
|---|---|---|
| full passage | 13/17 = 0.765 | **0.9244** |
| taxon names only | 12/17 = 0.706 | **0.4188** |

On the gold set the control is **inconclusive**: one item, no power. On the held-out rows the
passage is worth **+50.6 points** (paired McNemar χ²=2082, p≈0). The head reads the passage; the
17-item gold simply cannot show it.

The annotation convention, confirmed by the curator's own notes: **direction is judged
against the canonical relation, not the surface form.** "Leptodora --prey--> Bosmina" is
REVERSE because the canonical is *preyed upon by*.

## Known limits

* `reject50` is near chance for both this model and V1 (AUPRC 0.50 vs 0.54, measured at
  earlier labels and not regenerated). It is a deliberately adversarial recall set; nothing here
  solves it.
* Direction depends on a relation-polarity lexicon. All 17 gold relations were covered; an
  unlisted relation falls back to agent-side polarity and is flagged `unknown_polarity=1`, and a
  symmetric one is answered BIDIRECTIONAL. The four-way evaluation on the current 97-item
  expert gold is `results/shipping_2026-10-02/eval_a05.json` (handoff README §5).
* 14.5% of benchmark rows carry pipe-joined alternates (`Gobio gobio|gudgeon`) and 0 of
  48,338 training rows do. Measured cost (at earlier labels): +0.0000 AUPRC. Harmless, but it is a real
  train/eval mismatch and should not be allowed to grow.
* Seed noise on this benchmark is about ±0.008 AUPRC (two sd; BiomedBERT over 5 seeds:
  0.9583–0.9681, sd 0.0039). Do not read any single-seed gap smaller than that as real.

## Encoder choice

Eight encoders were trained on the identical recipe (same data, format, epochs, lr, seed)
and scored on the same benchmark. Raw table in `models/encoder_sweep/sweep_results.json`.

All benchmark figures in this section are on the 437 clean rows at current labels
(`scripts/model_card_tables.py`, `encoders` block) unless marked otherwise; `sweep_results.json`
holds the original 440-row run.

**Seed noise first**: BiomedBERT over three seeds gives AUPRC 0.9647 / 0.9648 / 0.9681
(sd 0.0019), but its five seeds span 0.9583–0.9681 (sd 0.0039, table below). Any single-seed
gap under ~0.008 is noise, which covers the three rows nearest the incumbent in this table.

| encoder | params | test AUPRC | F1 @ its own dev threshold |
|---|---|---|---|
| BiomedBERT (incumbent) | 109.5M | 0.9647 | 0.8958 |
| BioLinkBERT | 109.5M | 0.9660 | 0.8971 |
| DeBERTa-v3 | 183.8M | **0.9676** | **0.9193** |
| ModernBERT | 149M | 0.9598 | 0.9022 |
| bio-DistilBERT | 66M | 0.9540 | 0.8856 |
| **BiodivBERT** | 109.5M | 0.9549 | 0.8851 |
| SapBERT | 109.5M | 0.9509 | 0.8973 |
| BioBERT-1.2 | 108M | 0.9458 | 0.8931 |
| SciBERT | 109.9M | 0.9370 | 0.8816 |

**BiodivBERT does not win, and with enough seeds it is significantly worse.** The
single-seed result was too thin to call, so both contenders were taken to five seeds:

| family | seeds | mean AUPRC | sd | vs BiomedBERT |
|---|---|---|---|---|
| BiomedBERT | 5 | **0.9647** | 0.0039 | reference |
| BioLinkBERT | 5 | 0.9627 | 0.0044 | t p=0.480, Mann-Whitney p=0.548: no difference |
| **BiodivBERT** | 5 | **0.9525** | 0.0020 | t **p=0.0002**, Mann-Whitney **p=0.0079**, Δ −0.0122 |

So the one encoder pretrained on biodiversity literature is the one that measurably *loses*
on a biodiversity task. Its variance is also unusually tight (sd 0.0020 against BiomedBERT's
0.0039): it converges consistently to a slightly worse solution rather than being noisy.

**BioLinkBERT was run to three seeds** because it led on the distillation dev split
(0.9872 vs BiomedBERT's 0.9835). It does not transfer: 0.9660 / 0.9578 / 0.9677,
mean 0.9638 against BiomedBERT's 0.9658, t-test p=0.574, paired bootstrap CI spanning zero
(the bootstrap was run at earlier labels). No difference.

**Dev thresholds do not transfer to this benchmark.** `biolinkbert_s3` has an AUPRC above
every row of the table (0.9677) and the worst F1 at its own dev threshold (0.8702, threshold
0.18). The distillation dev split's distribution is not the benchmark's, which is the same
reason the ensembles' dev-F1 sits ~0.13 below their oracle F1 (at earlier labels). This is the
strongest argument for building a benchmark-distribution dev split.

**Encoder diversity was tested and does not help.** Cross-encoder score correlation is
r=0.933 (BiomedBERT vs BioLinkBERT) against r=0.947 for two seeds of the same encoder,
barely more decorrelated. The best 2-checkpoint arm reaches AUPRC 0.9497 but F1 0.8897 and
McNemar p=1.2e-02 (measured before the row-400 revision and not regenerated), losing to this
single model at the labels before that revision (F1 0.9008, p=5.0e-04) at twice the cost.

**DeBERTa-v3 was taken to three seeds** because on one seed it had the best F1 at its own dev
threshold. Result: AUPRC 0.9676 / 0.9694 / 0.9619, mean **0.9663** against BiomedBERT's
**0.9647**, t-test p=0.586. Indistinguishable on separability. Its apparent edge was calibration
(F1@dev 0.900–0.920 vs 0.896), not discrimination, and it costs 1.68× the parameters with no
direction head.

**Conclusion: BiomedBERT is retained.** Three encoder families (BiomedBERT, five seeds,
0.9647; DeBERTa-v3, three, 0.9663; BioLinkBERT, five, 0.9627) are statistically
indistinguishable; the six single-seed encoders are equal or worse. **The backbone is not a
lever on this task.** What moved the numbers was the label semantics and the input
formulation, not the pretrained weights.
