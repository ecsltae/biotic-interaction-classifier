# Paper A number audit, 2026-10-06 (after the gold revision)

Seven benchmark labels changed after the blind re-annotation (`FLIPS_2026-10-06.md`); positives went
246 → 251 of 437 and every result file was regenerated. `paperA/paperA.tex` was then updated from
FLIPS §5 (144 rows, with the §6 corrections) and the regenerated files. The audit has three parts.

1. `scripts/audit_paper_numbers.py` extracts every number in `paperA/paperA.tex` from the abstract
   on, including leading-dot cells (`.858`, `_{\pm .006}`). It looks each one up among the values in
   `results/paperA_v2/*.json` (except the superseded snapshots), `eval_a05.json`, `eval_v3.json`,
   `derived_biodiv.json` and `cascade.json`, at the printed precision.
2. A cell-by-cell check. Every label-dependent table row and sentence was rebuilt from the
   regenerated files and matched against the tex: the JSON values were formatted at the paper's
   precision, not copied from the edit.
3. `number_update.py` (scratch; output `scratchpad/flips/after_edit.tsv`) compares old values (backup) with new values (regenerated files) for
   every number of the edited tex.

## Totals

**913 numbers, 865 explained by a result file.** The other 48 were checked by hand (table below).

**Cell-by-cell check: 116 rows and phrases, 0 mismatches.** It covers every row of:
- Table 1 (incl. the teacher block);
- Table 2;
- the ablation, curve, per-seed, soft-label, encoder (biodiversity columns), scaling, cascade and
  rules tables.

It also covers every `\pm` subscript and leading-dot cell, and every body, caption and appendix
sentence with a changed number. It covers the 30 numbers inside `\textbf{}`/`\mathbf{}` too. Both
scripts miss those, because their number pattern excludes a number preceded by `{`. It also asserts
the claims behind the reworded sentences:

| Claim in the paper | Value |
|---|---|
| bold positions | best per column in each block, or the table's existing convention |
| triple model ahead at strict operating points | at τ = 0.5: P 0.942 / R 0.777 against 0.940 / 0.689; at τ = 0.9: 0.982 / 0.637 against 0.981 / 0.422 |
| teacher gap is "twice" the student's | ratio 2.05 |
| soft labels recover "about a quarter" of the distance to the teacher | 0.24 |
| soft triple model gains "on every seed" | +0.004 / +0.005 / +0.009 |
| zero-shot pair question ahead at every size from 1.7B | every gap positive |
| 122B pair question ranks better than any trained model | 0.985, above every trained model; best is the recommended model at 0.970 |
| sentence-level model last on the fresh sample | 0.665 < 0.765 < 0.785 |

**number_update.py on the edited tex:**

| Status | Rows |
|---|---|
| unchanged | 143 |
| unchanged? | 550 |
| unexplained | 145 |
| AMBIG | 40 |
| CHANGED | 35 |

The 145 unexplained rows are new values that equal no old value. All 75 CHANGED and AMBIG rows are
coincidences. The tool looks up each printed number among the *old* values, so a correctly updated
number that happens to equal an unrelated old key is flagged. Example: 0.948 equals the old
`eval_v3.auprc`, but in the paper it is the triple ensemble AUPRC, recall at τ = 0.04, or the
recommended model's recall.
- 68 of the 75 rows lie inside a row or phrase verified by the cell-by-cell check.
- The other 7 are label-independent, and the tool flagged them the same way before the edit:
  - "1.7" in Qwen3-1.7B (4×);
  - the Biotx100 and Reject50 mean taxa counts, 2.6 and 1.7 (`n_taxa_per_row.csv`);
  - the CP lower bound 0.29 for 15/32 (triple-level grades);
  - the BioRED subsample base rate, 0.442.

The deliberate old-label sentence and the historical rows come out "unchanged?" or "unexplained",
not CHANGED.

## Checked by hand (the 48 unexplained)

| Number(s) | Where | Source / check |
|---|---|---|
| +9.6 EP F1 | abstract | 67.08 − 57.46 (`biored_ep.json`, pair/sentence `test_F1_mean`) |
| 48,338 (3×), 0.343; 20,000 (2×) / 16,546 / 6,201 / 5,591 (2×) | §2, §3, §4.3, App. soft, App. "Where the reformulation stops" | `v3_combined_train.csv`, `kind` column (audit 2026-10-04); training corpus, label-independent |
| 2,017 passages | §3 | audit 2026-10-04 (pair-contrast passages with several rows and both labels) |
| 22/150 = 14.7%, CP [9.4%, 21.4%] | §3 | recomputed (scipy beta): 0.1467, [0.0943, 0.2136]; kept by decision (see below) |
| 5.1 taxa | Table 2 caption | `n_taxa_per_row.csv`: Test299 5.107 (Biotx100 2.577, Reject50 1.659) |
| 500 abstracts, 15,640 candidates | §4.2 | BC8 protocol; converter output (audit 2026-10-04) |
| 42,747 (2×) | §6, App. rules | 48,338 − 5,591 |
| 99.7% | App. soft | soft relabel log (audit 2026-10-04) |
| 0.442, 0.527 | App. scaling caption | recomputed on the 122B's 1,000 BioRED rows (audit 2026-10-04); BioRED, label-independent |
| 53,000 | App. cascade | 30% of 175,588 = 52,676 |
| 1.4 s per call | App. cascade | `results/overnight_2026-10-03/REPORT_s2.md` §2 ("~1.4 s per call") |
| 6,000-row sample | App. rules | fixed sample (seed 20261002) of the co_listed validation (audit 2026-10-04) |
| 122 (11×) | throughout | the "122B" of Qwen3.5-122B, not a quantity |
| 2018 (3×), 2019 (3×), 2020 (2×), 2023, 2024 | §8 | years inside `\citep` keys, not quantities |
| 1.9× | Table negative, row 3 | historical row, kept unchanged by decision. **No source found** in any results/ JSON, report or log (`logs/selfdistill/6L.log` records no throughput). Listed, not changed. |

Also recomputed: Limitations 11/11, CP [0.72, 1.00] (0.715). §5: 15/32 = 0.47, CP [0.29, 0.65]
(0.291, 0.653), and 65/(65 + 15) = 0.81. Limitations: Reject50 32 / 17 / 14, from
`biotx_rejected_50_testset.csv` (`interaction_eval`, `species1_eval`, `species2_eval`; not touched by
the flips).

## Deliberate old-label numbers (not updated)

- **Limitations, "One benchmark, mostly singly annotated."** "On the gold before re-annotation the
  pair query raises AUPRC from 0.851 to 0.918 and F1 from 0.775 to 0.871." These are the pre-revision
  values (`regression_2026-10-06_pre_review/tables_biodiv.json`: 0.8513, 0.9185, 0.7748, 0.8713).
- **Table negative, rows 2–5**, measured under an earlier evaluation and labelled as such:
  - −0.001 P, −0.008 F1;
  - 1.9×, −0.088 precision;
  - −0.059 AUPRC;
  - +0.005 to −0.037 AUPRC.

  The paragraph below the table repeats 0.088, 0.059 and 0.037.
- **Teacher mislabel rate**: 22/150 = 14.7% (10 FP / 12 FN), and "one time in seven". The figure
  rests on triple-level grades, which the PAIR-only re-annotation does not touch (FLIPS §6.0).

## Rounding and choices

- p-values that changed are printed to two significant digits: 0.26, 0.17, 0.50, 0.72, 0.66, 0.11,
  0.48 and 1.7×10⁻⁵, 3.8×10⁻⁷.
  - In the encoder caption this gives p = 0.093 (0.0929) and p = 0.0077 (0.00765). The FLIPS previews
    had 0.09 and 0.008.
  - Bounds follow FLIPS §6: p ≥ 0.23 (min 0.2353) and per-seed sd ≤ 0.032 (max 0.0311).
- Gaps are exact differences, as in FLIPS §6.4:
  - BioLinkBERT-large: +0.090 / +0.026.
  - Student gap on ≥ 3 taxa: +0.098, where the paper used to print a rounded-cell difference
    (+0.084); the exact and rounded-cell values now agree.
- Two FLIPS §5 rows are not printed in the current paper, so nothing changed:
  - the coinciding stratum, 13 → 12 negatives of 33;
  - the Biotx100 positive rate, 77% → 78% (the text says only "far above the 31%", still true).

## Addendum: fixes after the independent verification (2026-10-06)

A five-checker verification workflow (numbers recomputed from raw scores, every table cell, prose,
format, reviewer lens) found no wrong regenerated value. 28 of its 30 proposed fixes were applied;
the two not applied concern describing the 22-item re-annotation, where the user's decision stands.
Final state: 922 numbers checked, 49 not explained by a result file, all of the hand-checked kinds
listed above (corpus sizes, model names, citation years, 22/150 and its interval, BioRED protocol
counts, compute figures, the historical rows of Table negative). Abstract 198 words.

New or changed numbers from these fixes, with their sources:
| Number(s) | Source / check |
|---|---|
| 68.9 (BioLinkBERT-large EP F1 seeds 67.7--68.9) | `biored_ep.json` pair_linkbertL per-seed test F1 0.67816 / 0.68949 / 0.67737 (69.0 was double-rounded) |
| 1 of 32 unsupported (taxon-validity gate, appendix and Table negative row 1) | Biotx100 triple-level grades on the 97 clean rows (`biotx_retrieval_eval_100.csv` triples_ok_full; the 3 excluded duplicates are all unsupported); the one caught is the norovirus/oysters row |
| 41 / 27 / 12 / 10 (Reject50 axes, Limitations) | `biotx_rejected_50_testset.csv` joined to the 41 clean Reject50 rows: interaction_eval == 0 → 27; species1_eval or species2_eval == 0 → 12; both → 10 (the 50-row counts 32/17/14 included the 9 excluded rows) |
| +0.036 / +0.024 / +0.015 (Limitations, effect of the revision) | per-seed AUPRC, `regression_2026-10-06_pre_review/tables_biodiv.json` vs `tables_biodiv.json`: triple +0.0362, recommended model 0.9345 → 0.9701 (+0.0356), pair +0.0238, sentence +0.0152 |
| 0.976 / 0.982 / 0.915 / 0.839 (Table 2, triple column of the argument strata) | `derived_biodiv.json` strata.*.auprc.triple |
| confidence 0.9 (re-annotation selection) | `gold_review_v2_KEY` p_triple / p_shipped: both ≥ 0.9 against a 0 label or both ≤ 0.1 against a 1 label selects exactly the 11 contested items |
| 0.970 (recommended model, appendix) | `results/shipping_2026-10-02/eval_a05.json` auprc |
