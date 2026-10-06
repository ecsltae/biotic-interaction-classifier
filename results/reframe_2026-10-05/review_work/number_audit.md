# Number audit: paperA/paperA_reframe.tex (sentence-level reframe) against paperA/paperA.tex

- Audited file: `paperA/paperA_reframe.tex`, mtime 2026-10-05 05:12:25, md5 `b481aa00fdf5768bae589a6c92da3c26`. Line numbers refer to this version.
- Scope: every number that is new, or that moved into a different sentence or claim. Numbers printed identically in paperA.tex in the same claim are skipped (already covered by NUMBER_AUDIT_2026-10-04.md). Citation years, LaTeX lengths and model names are ignored.
- Method:
  - I diffed both files sentence by sentence and token by token.
  - Every value below was read from the per-file JSON. I did not take values from `sentence_tables.json`, REFRAME_NOTES.md or writer_work/numbers.md.
  - Rounding is half-up at the printed precision. A helper written for this audit is at `review_work/audit_reframe_numbers.py`. It is read-only and lists the JSON keys that round to a given printed value.
- Abbreviations:
  - SL = `results/paperA_v2/sentence_level/`
  - BS = `SL/biored_sentence.json`
  - BL = `SL/biodiv_sentlab.json`. Its `by_silver.*` values are identical to the ones in `biodiv_sentence.json`.
  - maj = majority3, nt = nonteacher2, 122 = q122b
  - "abs-cl" = `bootstrap_auprc_abstracts` (abstract-cluster resampling)

## (a) Numbers checked

| tex line | printed | claim context | source file : key | source value | verdict |
|---|---|---|---|---|---|
| 31 | 48k-row | corpus size | `data/training/distill/v3_combined_train.csv` row count | 48,338 rows | OK |
| 32-33 | 0.851→0.918, 0.775→0.871, p=1.3e-7 | moved within the abstract; F1 now sits next to McNemar again | paperA.tex tab:main (already audited) | same | OK |
| 34 | 0.738→0.843, "gaining only on sentences naming >2 entities" | BioRED pair level | paperA.tex tab:biored and strata (0.909 vs 0.902 on 2-entity sentences, +0.120 on more) | same | OK |
| 35 | 0.965 | BioRED pair-max sentence AUPRC (ensemble) | BS `arms.pair-max.auprc_ensemble` | 0.96496 | OK |
| 36 | 0.930 | BioRED candidate-trained sentence arm AUPRC | BS `arms.sentence.auprc_ensemble` | 0.92982 | OK |
| 37 | 0.955 | BioRED sentence-label model AUPRC | BS `arms.sentlab.auprc_ensemble` | 0.95502 | OK |
| 37 | F1 0.954 vs 0.930 | BioRED F1, pair-max vs sentlab | BS `arms.pair-max.F1`, `arms.sentlab.F1` | 0.95433, 0.93047 | OK |
| 39 | 0.985–0.987 | biodiv sentlab AUPRC over the 3 silver definitions | BL `by_silver.{maj,nt,122}.arms.sentlab.auprc` | 0.98503, 0.98730, 0.98613 | OK |
| 39 | 0.966–0.977 | biodiv pair-max AUPRC over the 3 definitions | BL `by_silver.{maj,nt,122}.arms.pair-max.auprc` | 0.96551, 0.97736, 0.97386 | OK |
| 40, 83, 410, 480, 618 | 0.536 | BioRED perfect-filter pair precision | BS `perfect_filter_pair_precision.precision` | 0.53555 (7684/14348) | OK |
| 40, 83, 410, 481, 618, 649 | 0.724 | biodiv perfect-filter pair precision, majority labels, marked provisional | BL `by_silver.majority3.perfect_filter_pair_precision.precision` | 0.72353 (246/340) | OK. Silver-dependent; the text says so (l.481, l.649). ANALYSIS.md's "does not depend on silver labels" was not carried over. |
| 61 | 175,588 | retrieval pool | paperA.tex l.60 | same | OK (moved) |
| 71, 613 | 1.7B–122B, "every size from 1.7B" | zero-shot gain | paperA.tex l.293, l.964 | same | OK (moved) |
| 132 | 34.5% | teacher answers "no" on corpus passages | SL/sentlab_label_stats.json `1 - sentence_yes_rate` | 1 − 0.65452 = 0.34548 | OK |
| 173 | 10,000 replicates | sentence-level bootstraps | BS/BL `bootstrap_*.B` | 10000 | OK |
| 173-174 | exact McNemar, Wilson | sentence-level protocol | BS/BL `mcnemar_exact.*.p_exact`, `perfect_filter_pair_precision.wilson95` | present | OK |
| 176 | 48,338 "training passages" | 5-gram scan (inherited) | v3_combined_train.csv: 48,338 rows, 34,242 distinct `text` | rows ≠ passages | CONTEXT-WRONG. Inherited wording that now contradicts l.446/l.1179 ("34,242 corpus passages"). See MUST FIX 1. |
| 302-306 | 0.789, 0.943, +0.155, 0.027 | teacher gap | paperA.tex l.285-296 | same | OK (moved) |
| 309-315 | +0.084, +0.042, 41%, +0.026/33, +0.052, +0.180 | tab:where | paperA.tex l.299-306 | same | OK |
| 322-326 | 0.909→0.677, 0.563; +0.103..+0.113; +0.052..+0.069; 0.745→0.963 | ablation and encoders summary, moved from §4 subsections | paperA.tex l.399, 417-418, 451; reframe tab:encoders (gaps 0.067/0.052/0.069 ours) | same | OK (moved) |
| 377-379 | 0.909/0.902, 0.899–0.919/0.892–0.914, 97%, +0.120, p=0.08 | BioRED strata | paperA.tex l.358-361 | same | OK |
| 381-383 | 67.1/57.5, 74.1–75.3, "a quarter" | BioRED EP F1; now "in part because" | paperA.tex l.366-369 | same | OK (writer FLAG 9 fixed) |
| 399 | .838 / .912 | BioRED accept-all AUPRC (base rate) / F1 | BS `positive_rate`, `trivial_accept_all.F1` | 0.83811, 0.91193 | OK |
| 399 | .778 / .787 / .805 / F1 .875 | biodiv accept-all by maj/nt/122; F1 under maj | BL `by_silver.*.positive_rate`; `majority3.trivial_accept_all.F1` | 0.77803, 0.78680, 0.80549; 0.87516 | OK |
| 400 | .930 / .922 | BioRED sentence arm AUPRC / F1 | BS `arms.sentence.auprc_ensemble`, `.F1` | 0.92982, 0.92150 | OK |
| 400 | .957 / .963 / .958 / .849 | biodiv sentence arm | BL `by_silver.*.arms.sentence.auprc`; maj `.F1` | 0.95719, 0.96344, 0.95760; 0.84862 | OK |
| 401 | .955 / .930 | BioRED sentlab | BS `arms.sentlab` | 0.95502, 0.93047 | OK |
| 401 | .985 / .987 / .986 / .929 | biodiv sentlab | BL `by_silver.*.arms.sentlab` | 0.98503, 0.98730, 0.98613; 0.92926 | OK |
| 402 | .962 / .974 / .968 / .876 | biodiv pair-own | BL `by_silver.*.arms.pair-own` | 0.96212, 0.97355, 0.96755; 0.87610 | OK |
| 403 | .965 / .954 | BioRED pair-max | BS `arms.pair-max` | 0.96496, 0.95433 | OK |
| 403 | .966 / .977 / .974 / .897 | biodiv pair-max | BL `by_silver.*.arms.pair-max` | 0.96551, 0.97736, 0.97386; 0.89668 | OK |
| 405 | 500 random sentences | zero-shot subset | BS `llm."qwen3-32b random500".n_sentences` | 500 | OK |
| 406 | .908 / .914 | 32B sentence question, AUPRC / greedy F1 | BS `llm."qwen3-32b random500"."llm sentence"` | 0.9081, 0.9138 | OK |
| 407 | .959 / .942 | 32B pair question max, AUPRC / greedy F1 | same, `"llm pair-max"` | 0.9591, 0.9418 | OK |
| 410 | .794 / .699 | perfect-filter precision, nt / 122 | BL `by_silver.{nt,122}.perfect_filter_pair_precision.precision` | 0.79355, 0.69886 | OK |
| 413-414 | per-seed SD < .007 | tab:sentence caption | max of BL `by_silver.*.arms.*.auprc_seed_sd` and BS `arms.*.auprc_sd` (ddof=1) | 0.00626 (122 pair-own) | OK |
| 414, 438 | 2,582 | BioRED BC8 test sentences with a candidate | BS `n_sentences` | 2582 | OK |
| 416-417, 443, 470 | 437 / 246 / 191 | passages, pair-positive, others | BL `by_silver.majority3.n`, `perfect_filter...pair_positive`, `n_pair_negative` | 437, 246, 191 | OK |
| 419 | 394 passages; 43 dropped, all pair-negative | nt definition | BL `nonteacher2.n`, `.dropped`; `decomposition.pair_negative_only.n` = 148 = 191 − 43 | 394, 43 | OK |
| 422 | base rate 0.854 | zero-shot subset | BS random500 `positive_rate` | 0.854 | OK |
| 423 | accept-all 0.921 | zero-shot subset | BS random500 `trivial_accept_all_F1` | 0.9213 | OK |
| 424 | 0.941 / 0.958 / 0.973 | trained arms on the same 500 sentences | BS random500 `trained_arms_same_sentences.{sentence,sentlab,pair-max}.auprc_ensemble` | 0.9413, 0.9579, 0.9734 | OK |
| 438 | 0.838 positive | BioRED base rate | BS `positive_rate` | 0.83811 | OK |
| 446, 1179 | 34,242 corpus passages | teacher sentence labels | SL/sentlab_label_stats.json `labelled_passages`; distinct `text` in v3_combined_train.csv | 34242 | OK. The model trains on 30,818 and uses 3,424 for development (SL/sentlab_split.json). All passages are used, so "on all" stands. |
| 446 | 65.5% | teacher calls positive | sentlab_label_stats `sentence_yes_rate` | 0.65452 | OK |
| 447, 651 | 14.3 pairs per passage | pair-max enumeration, includes the candidate | SL/biodiv_enum_cost.json `enumeration.pairs_per_passage_incl_candidate.mean`; BL `enumeration.mean_pairs_per_passage` | 14.3089 | OK |
| 451 | +0.035 [+0.024, +0.047] | BioRED pair-max − sentence arm AUPRC, abs-cl | BS `bootstrap_auprc_abstracts.diff."pair-max - sentence"` | 0.03513 [0.02365, 0.04749] | OK |
| 452-453 | F1 +0.024 [+0.017, +0.031] | pair-max − sentlab F1, abs-cl | BS `bootstrap_f1_abstracts.diff."pair-max - sentlab"` | 0.02387 [0.01673, 0.03143] | OK |
| 453 | 158 fixed, 41 broken, p = 2.1e-17 | exact McNemar, pair-max vs sentlab | BS `mcnemar_exact."pair-max vs sentlab"` | 158, 41, 2.126e-17 | OK |
| 454 | +0.010 [−0.001, +0.021], "not significant" | AUPRC pair-max − sentlab, abs-cl | BS `bootstrap_auprc_abstracts.diff."pair-max - sentlab"` | 0.00993 [−0.00130, 0.02083] | OK. Sentence resampling would exclude 0 ([+0.0002, +0.0196]); the text names the abstract scheme at l.450. |
| 455 | 777 single-pair sentences | stratum | BS `strata.candidate_pairs."1".n` | 777 | OK |
| 455-456 | +0.044 [+0.012, +0.079] | single-pair, pair-max − sentlab, abs-cl | BS `strata.candidate_pairs."1".bootstrap_auprc_abstracts.diff."pair-max - sentlab"` | 0.04407 [0.01181, 0.07889] | OK |
| 456 | +0.006 [−0.003, +0.015] | multi-pair, same diff | BS `strata.candidate_pairs.">=2"...` | 0.00589 [−0.00318, 0.01464] | OK |
| 457-458 | "each weak in one stratum; pair-max best or tied" | stratum AUPRCs | BS strata `1`: sentlab 0.852, sentence 0.892, pair-max 0.896; `>=2`: 0.981, 0.972, 0.986 |, | OK |
| 460-461 | 0.959 vs 0.908, +0.051 [+0.025, +0.079] | 32B zero-shot, items resampled | BS random500 `bootstrap_auprc.diff."llm pair-max - llm sentence"` | 0.0511 [0.0251, 0.0794] | OK |
| 464 | pair-max vs pair-own "never significantly" | 3 definitions | BL `by_silver.*.bootstrap_auprc.diff."pair-max - pair-own".ci95` | maj [−0.0056, 0.0120]; nt [−0.0037, 0.0112]; 122 [−0.0011, 0.0137] | OK |
| 466 | +0.014 [+0.002, +0.027] | nt, pair-max − sentence | BL `nonteacher2.bootstrap_auprc.diff."pair-max - sentence"` | 0.01392 [0.00214, 0.02651] | OK |
| 466 | +0.016 [+0.005, +0.029] | 122, same | BL `q122b...` | 0.01626 [0.00469, 0.02895] | OK |
| 467 | +0.008 [−0.005, +0.021] | maj, same | BL `majority3...` | 0.00833 [−0.00476, 0.02150] | OK |
| 467-468 | 0.020, 0.010, 0.012, "every interval excluding zero" | sentlab − pair-max AUPRC (maj, nt, 122) | BL `by_silver.*.bootstrap_auprc.diff."pair-max - sentlab"` | −0.01952 [−0.0315, −0.0095]; −0.00993 [−0.0194, −0.0016]; −0.01227 [−0.0222, −0.0040] | OK |
| 468 | F1 0.929 vs 0.897 | maj | BL `majority3.arms.{sentlab,pair-max}.F1` | 0.92926, 0.89668 | OK |
| 469-470 | 46 vs 22, p = 0.0049 | exact McNemar, maj | BL `majority3.mcnemar_exact."pair-max vs sentlab"` | fixes 22, breaks 46, p 0.004903 | OK |
| 472 | 0.953 / 0.788 / 0.658 | 191 pair-negative passages, maj | BL `majority3.decomposition.pair_negative_only.auprc.{sentlab,pair-max,pair-own}` | 0.9534, 0.7880, 0.6576 | OK |
| 475-476 | −0.004 [−0.014, +0.005], "the three indistinguishable" | gold-positive vs silver-negative, nt | BL `nonteacher2.decomposition.gold_pos_vs_silver_neg.bootstrap_auprc.diff."pair-max - sentlab"` | −0.00357 [−0.01351, 0.00531] | OK under nt and 122. Under majority, pair-max − pair-own excludes 0 (−0.0089 [−0.0184, −0.0008]); the text names nt, so OK (note N2). |
| 480 | [0.527, 0.544] | BioRED filter Wilson CI | BS `perfect_filter_pair_precision.wilson95` | [0.52738, 0.54370] | OK |
| 480 | 0.516 | multi-candidate sentences | BS `...by_candidate_pairs.">=2".precision` | 0.51615 | OK |
| 481 | [0.674, 0.768] | biodiv maj Wilson CI | BL `majority3.perfect_filter_pair_precision.wilson95` | [0.67369, 0.76837] | OK |
| 481-482, 649 | 0.699 to 0.794 | across definitions | BL 122 / nt precision | 0.69886, 0.79355 | OK |
| 485 | 0.794 / 0.840 / 0.923 | AUPRC against pair gold, ensembles | BL `pair_gold_view.{sentlab,pair-max,pair-own}.auprc` | 0.79370, 0.83974, 0.92289 | OK |
| 489 | 41% | teacher sentence answers vs its candidate (triple-prompt) labels | sentlab_label_stats `crosstab`: (12,503 + 1,422) / 34,242 | 0.40669 | OK. The wording "candidate labels" is correct; ANALYSIS.md's "a third" was not carried over. |
| 490 | κ = 0.26 | same | sentlab_label_stats `kappa_sentence_vs_any_candidate` | 0.26363 | OK |
| 490-491 | "mostly ... without accepting any of their candidates" | same | 12,503 / 13,925 | 0.898 | OK |
| 497-498 | 39, 27, 17 | FP decomposition | paperA.tex (same) | same | OK (moved) |
| 511-514 | 97, 15 of 32, 0.81 | component-wise bound | paperA.tex l.479-484; reframe App. l.1378-1384 | same | OK (moved) |
| 528 | 0.5, 0.934, 0.900, 437, 32.3 | CPU configuration summary | paperA.tex l.534-538 | same | OK (moved) |
| 529-530 | 74%, 0.887 | direction head | reframe tab:direction l.1322: 62 of 84, 55 of 62 | 0.738, 0.887 | OK |
| 591 | AUC 0.97, 25% precision | lim2016minter | results/reframe_2026-10-05/LITERATURE.md rows at l.101 and l.171 | AUC 0.97; interaction-level precision 25% | OK (literature) |
| 611-613 | 0.851→0.918, 0.738→0.843, 1.7B | conclusion | paperA.tex | same | OK |
| 614-616 | "beats ... on ours under two of three label definitions" | pair-max vs sentence arm | BL diffs above (point estimates above zero under all 3, significant under nt and 122) |, | OK, read as "significantly" (note N1) |
| 640 | 191, or 148 | LLM-decided passages | BL `n_pair_negative`; `nonteacher2.decomposition.pair_negative_only.n` | 191, 148 | OK |
| 643 | 19 items, 3 UNSURE dropped | spot check | SL/biodiv_human.json `gold_review_spot_check.sentence_labels.{labelled_rows,unsure_dropped}` | 19, 3 | OK |
| 644 | "half of them selected by model–gold disagreement" | spot check | biodiv_human.json `gold_review_spot_check.use` ("22 items, 11 of them selected") | 11/22 | OK |
| 645 | 5 pair-positive | gold-fixed silver | biodiv_human.json `silver_yes_forced_by_pair_gold` | 5 | OK |
| 646 | 13 of 14, κ = 0.81 | vs nt | biodiv_human.json `sentence_vs_silver.nonteacher2` | n 14, agreement 0.92857, κ 0.81081 | OK |
| 646 | 17 of 19, κ = 0.68 | vs 122 | `...q122b` | n 19, 0.89474, κ 0.68333 | OK |
| 647 | 15 of 19, κ = 0.46 | vs maj | `...majority3` | n 19, 0.78947, κ 0.46479 | OK |
| 651 | 85.9 vs 3 forward passes | per passage | biodiv_enum_cost.json `enumeration.forward_passes_per_passage.{pair_max,sentence}.mean` | 85.8535, 3.0 | OK |
| 652 | 6 passes; within 0.003 to 0.006 | pair-own vs pair-max | enum_cost `pair_own.mean` = 6.0; BL `by_silver.*.bootstrap_auprc.diff."pair-max - pair-own".observed` | 0.00339, 0.00382, 0.00631 | OK |
| 692 | 15,640 candidates vs 3,277 sentences | BioRED training sets (train+dev) | paperA.tex l.350; BS `sentlab_training_sentences.n` | 15,640; 3,277 | OK |
| 984-993 | 92%, 0.566, 0.48, 0.813, 0.860, +0.155, +0.180, +0.104, 0.946/0.943, 0.027, 0.871/0.870 | scaling appendix | paperA.tex l.285-297, l.361-364 | same | OK (moved) |
| 1173 | Qwen3 0.6B–32B, Qwen3.5-122B, Qwen3.8-27B | artifacts | models used in BS `llm.coverage`, BL `labels` |, | OK |
| 1179-1180 | 2 h 15 min | teacher sentence labelling of 34,242 passages | logs/gpu_sentlab_label.log: 0.236 s/passage × 34,242 = 8,081 s; ANALYSIS.md l.419 | 2 h 14.7 min | OK |

Notes (wording only, no number is wrong):
- **N1 (l.614-615):** "beats a sentence classifier trained on the same candidates (... on ours under two of three label definitions)". Pair-max is above the arm under all three definitions and significantly above it under two. Optional: "significantly beats".
- **N2 (l.474-476):** the label definition switches from majority (l.472) to non-teacher. The claim "the three are indistinguishable" does not hold under majority labels, where pair-max − pair-own is −0.009 [−0.018, −0.001]. The text names its definition, so it is not wrong. Optional: add "(under the majority labels pair-own edges pair-max)".
- **N3 (l.37-38):** "under provisional LLM-derived labels". Only the 191 pair-negative passages are LLM-decided; gold fixes the 246 positives. The precise version is in §4.3 and the Limitations.

## (b) MUST FIX

1. **l.176.** "48,338 training passages" counts rows: the corpus has 48,338 rows over 34,242 distinct passages (`data/training/distill/v3_combined_train.csv`). The reframe now prints "34,242 corpus passages" at l.446 and l.1179, so the inherited wording contradicts it.
   - Current: `a 5-gram Jaccard scan against all 48{,}338 training passages`
   - Replace: `a 5-gram Jaccard scan against the passages of all 48{,}338 training rows`

No other number is wrong, untraceable or attached to the wrong context. The writer's earlier flags are resolved in this version:
- FLAG 3/4: the abstract again gives the F1 before the McNemar p, and BioRED's "gaining only on" is attached correctly.
- FLAG 5/6: "< .007"; "0.003 to 0.006".
- FLAG 9: "in part because".
- 41% is worded as candidate labels.

## (c) Counts

Each row of table (a) counts as one check. A row that bundles several values from one source and one claim counts once.
- Checked: 92 table rows (about 190 individual printed values).
- OK: 91.
- WRONG: 0.
- CONTEXT-WRONG: 1 (l.176, inherited wording that conflicts with a new number).
- UNTRACEABLE: 0.
- Wording notes, optional: 3 (N1–N3).
