# Number audit: paperA_reframe.snapshot_0445.tex against paperA/paperA.tex (2026-10-05)

Scope: every number in the draft's text, tables, captions and abstract that is absent from `paperA/paperA.tex`, or present there with another meaning. Line numbers, section numbers, citation years, model names (Qwen3.8, 32B, ...) and LaTeX lengths are ignored. Values come from `results/paperA_v2/sentence_level/sentence_tables.json` (`numbers`, "file:dotted.path"), checked against the per-file JSONs. Roundings were checked by script (half-up, 3 decimals for AUPRC/P/R/F1 and differences): no plain value mismatch was found. Every flag below is about wording, rounding convention, or a bound that holds only after rounding.

## FLAGS

**(a) Numbers that cannot be traced:** none. The only numbers not from results/ are literature values ("AUC of 0.97", "25% precision", lim2016minter). They trace to `results/reframe_2026-10-05/LITERATURE.md` (rows at about lines 101 and 171).

**FLAG 1 (c): "two answers" are not sentence vs pair answers.**
Draft: "Asking each question for its own decision costs one more teacher prompt per passage, and the teacher's two answers differ on $41\%$ of corpus passages (Cohen's $\kappa = 0.26$)."
- The values are right: (12,503 + 1,422) / 34,242 = 0.4067, and κ = 0.2636 (`sentlab_label_stats.json`).
- But the second "answer" is not an answer to a pair question. It is "at least one of the passage's candidate rows in `data/training/distill/v3_combined_train.csv` is labelled positive". Those rows are (source_species, interaction_type, target_species) triples labelled with the triple question: the draft says so in Sec. 3 ("asked, per candidate, whether the passage supports *this* triple") and in App. prompts ("The triple question is the prompt that labelled the training corpus").
- So 41% / κ 0.26 compare the sentence answer with the passage-level OR of the triple labels, not with pair-question answers.
- The script docstring (`scripts/sentence_level/sentlab_label_stats.py`) calls them "pair-question labels", which contradicts the paper. One of the two needs correcting.

**FLAG 2 (c): 0.724 is presented as silver-free, but it depends on the silver labels.**
- Draft, Sec. sentence: "The converse holds on both benchmarks and needs no silver label. A perfect sentence filter accepts every candidate of every positive passage, at a pair-level precision of $0.724$ $[0.674, 0.768]$ on ours and $0.536$ $[0.527, 0.544]$ on BioRED".
- Same unqualified 0.724 in the Intro ("The converse is unconditional: ... at a pair precision of $0.724$ on our benchmark"), the Abstract ("Conversely, a perfect sentence filter passes candidates at a pair precision of only $0.724$") and the Conclusion.

0.724 = 246/340 (`by_silver.majority3.perfect_filter_pair_precision`). The denominator counts the passages that the **majority3 silver** labels call sentence-positive. Under the other definitions it is 0.794 (nonteacher2) or 0.699 (q122b), as the draft's own Limitations say ("moves with the definition ($0.699$ to $0.794$)"). Only BioRED's 0.536 needs no silver label. The qualitative claim (precision well below 1 under every definition) does hold. Suggested fix: say "under majority labels; 0.699–0.794 across definitions".

**FLAG 3 (c): the abstract's BioRED numbers are attached to the wrong subset.**
Draft: "telling the model which pair it judges raises AUPRC on 437 expert-graded candidates from $0.851$ to $0.918$ (McNemar $p = 1.3\times10^{-7}$), and on BioRED from $0.738$ to $0.843$, on sentences naming more than two entities."
- 0.738 → 0.843 is the per-seed mean AUPRC over **all** 16,606 BioRED candidates (Table tab:biored, AUPRC column).
- On candidates from sentences naming ≥ 3 concepts, the values are 0.727 → 0.847 (+0.120).
- paperA's abstract kept the two apart ("from 0.738 to 0.843 ..., with no gain on sentences naming two entities and +0.120 on sentences naming more").
- Fix: "from 0.738 to 0.843, with the gain on sentences naming more than two entities".

**FLAG 4 (c, minor): the abstract's McNemar p is now attached to AUPRC.**
Same sentence as FLAG 3. The p-value tests decisions at block-held-out thresholds: 75 fixed / 22 broken, as in the Table tab:main caption. paperA's abstract placed it after "F1 from 0.775 to 0.871". The draft dropped the F1, so the p-value now appears to test the AUPRC gain.

**FLAG 5 (b): the caption's "≤ .006" holds only after rounding.**
Draft: "\caption{Sentence-level decisions; AUPRC of three-checkpoint ensembles (per-seed standard deviation $\le .006$)."
- **Ensemble claim: verified.** Every AUPRC cell equals the ensemble key: BioRED `arms.<arm>.auprc_ensemble`; biodiversity `by_silver.<def>.arms.<arm>.auprc`, which is `ap(y, ens)` in `biodiv_sentence.py`. The per-seed means would differ, e.g. BioRED pair-max .966 vs .965 and majority3 pair-max .963 vs .966.
- **SD bound: holds only after rounding.** The largest per-seed SD is 0.006264 (`by_silver.q122b.arms.pair-own.auprc_seed_sd`, ddof=1). The next two are 0.00587 (majority3 pair-own) and 0.00582 (nonteacher2 pair-own). The BioRED maximum is 0.00344.
- Fix: "≤ .0063" or "< .007". With ddof=0 the maximum would be 0.0051, but the JSON uses ddof=1.

**FLAG 6 (b): "within 0.006" holds only after rounding, and "most of its gain" does not hold for F1.**
- Draft: "Pair-max is also the costly route, $85.9$ forward passes per passage against 3, while pair-own, at 6, is within $0.006$ AUPRC of it."
  - pair-max − pair-own (ensemble AUPRC, `by_silver.<def>.bootstrap_auprc.diff.pair-max - pair-own.observed`) is +0.00339 (majority3), +0.00382 (nonteacher2) and **+0.00631 (q122b)**.
  - The q122b gap exceeds 0.006. Fix: "within 0.007", or "by 0.003–0.006".
  - At the operating point under q122b, pair-max − pair-own **is** significant: F1 +0.034 [+0.013, +0.057]; McNemar 37/18, p = 0.014.
- Related, Limitations: "the candidate pair alone gives most of its gain".
  - In AUPRC this holds: pair-own reaches 59–73% of pair-max's gain over the sentence arm.
  - In F1 under q122b it does not: pair-own − sentence is +0.005, against +0.039 for pair-max − sentence.

**FLAG 7 (b): the p-value has one significant digit, and the definition is not named.**
Draft: "The sentence-label model beats pair-max under all three: by $0.020$, $0.010$ and $0.012$ AUPRC, every interval excluding zero, and in F1, $0.929$ against $0.897$ (it is right on 46 passages where pair-max is wrong and wrong on 22, $p = 0.005$)."
- **Direction of the counts: correct.** `majority3.mcnemar_exact."pair-max vs sentlab"` has fixes = 22 (pair-max right, sentlab wrong) and breaks = 46. This follows from `common.mcnemar_exact(preds[b], preds[a])`.
- **Rounding.** p_exact = 0.004903, so the 2-significant-digit convention gives **0.0049**, not 0.005.
- **Definition.** The F1 pair (0.929/0.897) and the McNemar test are majority3 only, but the sentence opens with "under all three". The other definitions give p = 0.027 (nonteacher2) and 0.0059 (q122b).
- Fix: add "(majority labels)".

**FLAG 8 (c, minor): two bootstrap schemes appear in one paragraph without labels.**
Draft: "but its AUPRC gain is marginal, $+0.010$ $[-0.001, +0.021]$. Nor does the edge come from pairs competing within a sentence: it lies in the 777 single-pair sentences ($+0.044$ $[+0.016, +0.073]$), not in those with several candidates ($+0.006$ $[-0.003, +0.014]$)."
- The +0.010 interval and the F1 +0.024 [+0.017, +0.031] are **abstract-cluster** bootstraps (`bootstrap_*_abstracts`).
- The stratum intervals are **sentence-resampled** (`strata.candidate_pairs.*.bootstrap_auprc_sentences`).
- The clustered stratum intervals exist: [+0.012, +0.079] and [−0.003, +0.015]. The conclusions are unchanged either way. Name the scheme, or use the clustered intervals throughout.
- Note: the sentence-resampled interval for +0.010 is [+0.000, +0.020], which excludes 0. The draft rightly reports the clustered one, but it should say so.

**FLAG 9 (c): unchanged numbers 74.1–75.3 carry a stronger causal claim.**
Draft, Sec. biored: "systems that read the whole abstract reach $74.1$ to $75.3$, mostly because a quarter of the related pairs are never co-mentioned in one sentence (Appendix~\ref{app:bioredep})."
- paperA said "Much of that gap is structural".
- The cited appendix (unchanged) adds that the same model given the whole abstract reaches only 67.0, and that "what the published systems add is document-level modelling".
- "Mostly because" is stronger than this evidence and in tension with that appendix.

**FLAG 10 (c, minor): quantity words in the Limitations.**
- **"all rows."** Draft: "A second annotation of all rows, with a sentence and a pair answer for each, is in progress." The 437-row sheet has 20 SKIP rows (the gold-review items) that never get answers there, so it covers at most 417 of 437 (ANALYSIS.md §7).
- **"no human sentence labels."** Draft: "No human sentence labels exist yet for our benchmark." 19 sentence answers exist from the 22-item gold review. They are a non-random, non-blind spot check, not a benchmark, so "no benchmark-grade human sentence labels" would be accurate.
- **"the 191 others."** Draft: "LLM answers to the sentence question decide the 191 others. Under every definition we tried, an LLM's answer ... decides those 191". Under nonteacher2, 43 of the 191 are dropped, so LLM answers decide only 148.

**Checked and correct (no flag):**
- **158 / 41 direction.** "158 sentences fixed, 41 broken" is from pair-max's side, which is correct.
- **Biodiversity F1 column.** It is majority3, at block-held-out thresholds refit on that definition's labels (`evaluate()` in `biodiv_sentence.py`), as the caption says.
- **"every interval excluding zero".** Holds: the upper bounds are −0.0095, −0.0016 and −0.0040.
- **Arm order.** sentlab > pair-max > pair-own > sentence under every definition, by AUPRC and by F1.
- **"Pair-max is best or tied in both" BioRED strata.** Holds.
- **Wilson intervals and 10,000 replicates.** Both as stated.

**Softer notes (not flagged):**
- **"Its whole advantage lies on the 191 passages".** On the gold-positive subset under majority3, sentlab is still ahead by 0.008, though not significantly (pair-max − sentlab −0.008 [−0.020, +0.002]).
- **"the three are indistinguishable".** True under nonteacher2 (the definition cited) and under q122b. Under majority3, pair-own beats pair-max on that subset: −0.009 [−0.018, −0.001].
- **Editorial, no numbers involved:**
  - Lines 1065 and 1293 cite their own appendix.
  - App. deploy contains two near-duplicate paragraphs (lines 1290–1299 and 1320–1334).
  - The candidate-rules paragraph appears both in Sec. rules and App. rules.
  - "the seven alternative recipes of \S\ref{sec:res-ablation}" (line 803) and "the three recipes of \S\ref{sec:res-ablation}" (line 1189) point to a body section that no longer lists them; they are now in App. ablation.

## Traced numbers

Rows with an empty first cell continue the row above (its CI). "(OK)" means the printed value matches the exact value at the paper's rounding. "derived" rows show the arithmetic.

| number as printed | where in the draft | source file | JSON key (or derivation) | exact value |
|---|---|---|---|---|
| 0.965 | Abstract: "BioRED AUPRC 0.965 against 0.930" | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.pair-max.auprc_ensemble` | 0.964956 (OK) |
| 0.930 (AUPRC) | Abstract: same | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.sentence.auprc_ensemble` | 0.929823 (OK) |
| 0.955 | Abstract: "at least as good as one trained on sentence labels (0.955; ..." | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.sentlab.auprc_ensemble` | 0.955024 (OK) |
| 0.954 | Abstract: "F1 0.954 against 0.930" | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.pair-max.F1` | 0.954331 (OK) |
| 0.930 (F1) | Abstract: same (sentlab F1; not the AUPRC 0.930 above) | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.sentlab.F1` | 0.930465 (OK) |
| 0.985 | Abstract: "is better (0.985 against 0.966)" -- majority3, not stated | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.sentlab.auprc` | 0.98503 (OK) |
| 0.966 | Abstract: same | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.pair-max.auprc` | 0.965508 (OK) |
| 0.724 | Abstract ("pair precision of only 0.724"); also Intro, Sec. sentence, Conclusion, Limitations -- see FLAG 2 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.perfect_filter_pair_precision.precision` | 0.723529 (OK); = 246/340 (`...pair_positive` / `...sentence_positive`) |
| 0.536 | Abstract ("BioRED 0.536"); also Intro, Sec. sentence, Conclusion | `results/paperA_v2/sentence_level/biored_sentence.json` | `perfect_filter_pair_precision.precision` | 0.535545 (OK); = 7,684/14,348 |
| 0.738 -> 0.843 (context changed) | Abstract: "and on BioRED from 0.738 to 0.843, on sentences naming more than two entities" -- see FLAG 3 | `paperA/paperA.tex` Table tab:biored | per-seed mean AUPRC over ALL candidates; the >=3-concept column is 0.727 -> 0.847 | 0.738 / 0.843 (all); 0.727 / 0.847 (>=3) |
| p = 1.3e-7 (context changed) | Abstract: "raises AUPRC ... from 0.851 to 0.918 (McNemar p = 1.3e-7)" -- see FLAG 4 | `paperA/paperA.tex` Table tab:main caption | McNemar on decisions at block-held-out thresholds (75 fixed / 22 broken), not on AUPRC | unchanged value; meaning shifted |
| 34.5% | Sec. Setting: "the teacher ... answers no on 34.5% of the corpus passages" | `results/paperA_v2/sentence_level/sentlab_label_stats.json` | derived: 1 - `sentence_yes_rate` = 1 - 0.654518 | 0.345482 (OK) |
| 10,000 | Sec. Setting, Evaluation protocol: "paired percentile bootstraps (10,000 replicates)" | `results/paperA_v2/sentence_level/biored_sentence.json` | `bootstrap_auprc_sentences.B` | 10000; also `bootstrap_auprc_abstracts.B`, `biodiv_sentlab.json:by_silver.*.bootstrap_auprc.B` = 10000; percentile per `common.ci`; exact McNemar per `common.mcnemar_exact`; Wilson per `common.wilson` (OK) |
| .838 | Table tab:sentence, Accept everything, BioRED AUPRC (= base rate) | `results/paperA_v2/sentence_level/biored_sentence.json` | `positive_rate` | 0.83811 (OK) |
| .912 | Table tab:sentence, Accept everything, BioRED F1 | `results/paperA_v2/sentence_level/biored_sentence.json` | `trivial_accept_all.F1` | 0.911926 (OK) |
| .778 | Table tab:sentence, Accept everything, maj. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.positive_rate` | 0.778032 (OK) |
| .787 | Table tab:sentence, Accept everything, non-t. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.positive_rate` | 0.786802 (OK) |
| .805 | Table tab:sentence, Accept everything, 122B | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.positive_rate` | 0.805492 (OK) |
| .875 | Table tab:sentence, Accept everything, biodiv F1 (majority3) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.trivial_accept_all.F1` | 0.875161 (OK) |
| .930 | Table tab:sentence, Sentence arm, BioRED AUPRC | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.sentence.auprc_ensemble` | 0.929823 (OK) |
| .922 | Table tab:sentence, Sentence arm, BioRED F1 | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.sentence.F1` | 0.9215 (OK) |
| .957 | Table tab:sentence, Sentence arm, maj. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.sentence.auprc` | 0.957181 (OK) |
| .963 | Table tab:sentence, Sentence arm, non-t. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.arms.sentence.auprc` | 0.963446 (OK) |
| .958 | Table tab:sentence, Sentence arm, 122B | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.arms.sentence.auprc` | 0.9576 (OK) |
| .849 | Table tab:sentence, Sentence arm, biodiv F1 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.sentence.F1` | 0.84858 (OK) |
| .955 | Table tab:sentence, Sentence-label model, BioRED AUPRC | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.sentlab.auprc_ensemble` | 0.955024 (OK) |
| .930 | Table tab:sentence, Sentence-label model, BioRED F1 | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.sentlab.F1` | 0.930465 (OK) |
| .985 | Table tab:sentence, Sentence-label model, maj. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.sentlab.auprc` | 0.98503 (OK) |
| .987 | Table tab:sentence, Sentence-label model, non-t. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.arms.sentlab.auprc` | 0.987298 (OK) |
| .986 | Table tab:sentence, Sentence-label model, 122B | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.arms.sentlab.auprc` | 0.98613 (OK) |
| .929 | Table tab:sentence, Sentence-label model, biodiv F1 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.sentlab.F1` | 0.929293 (OK) |
| .962 | Table tab:sentence, Pair verifier own candidate, maj. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.pair-own.auprc` | 0.962118 (OK) |
| .974 | Table tab:sentence, Pair verifier own candidate, non-t. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.arms.pair-own.auprc` | 0.973549 (OK); exact 0.973549 -> .974 (close to the boundary, correct) |
| .968 | Table tab:sentence, Pair verifier own candidate, 122B | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.arms.pair-own.auprc` | 0.967545 (OK); exact 0.967545 -> .968 (close to the boundary, correct) |
| .876 | Table tab:sentence, Pair verifier own candidate, biodiv F1 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.pair-own.F1` | 0.876133 (OK) |
| .965 | Table tab:sentence, Pair-max, BioRED AUPRC | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.pair-max.auprc_ensemble` | 0.964956 (OK); per-seed mean would be .966 (`auprc_mean`); ensemble confirmed |
| .954 | Table tab:sentence, Pair-max, BioRED F1 | `results/paperA_v2/sentence_level/biored_sentence.json` | `arms.pair-max.F1` | 0.954331 (OK) |
| .966 | Table tab:sentence, Pair-max, maj. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.pair-max.auprc` | 0.965508 (OK); seed mean .963 (`auprc_seed_mean`); ensemble confirmed |
| .977 | Table tab:sentence, Pair-max, non-t. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.arms.pair-max.auprc` | 0.977365 (OK) |
| .974 | Table tab:sentence, Pair-max, 122B | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.arms.pair-max.auprc` | 0.973858 (OK) |
| .897 | Table tab:sentence, Pair-max, biodiv F1 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.pair-max.F1` | 0.896747 (OK) |
| .908 | Table tab:sentence, zero-shot Sentence question, AUPRC | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.llm sentence.auprc` | 0.908064 (OK) |
| .914 | Table tab:sentence, zero-shot Sentence question, F1 (greedy) | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.llm sentence.greedy.F1` | 0.913774 (OK) |
| .959 | Table tab:sentence, zero-shot Pair question max, AUPRC | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.llm pair-max.auprc` | 0.959135 (OK) |
| .942 | Table tab:sentence, zero-shot Pair question max, F1 (greedy) | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.llm pair-max.greedy.F1` | 0.941834 (OK) |
| .536 | Table tab:sentence, perfect-filter precision, BioRED | `results/paperA_v2/sentence_level/biored_sentence.json` | `perfect_filter_pair_precision.precision` | 0.535545 (OK) |
| .724 | Table tab:sentence, perfect-filter precision, maj. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.perfect_filter_pair_precision.precision` | 0.723529 (OK) |
| .794 | Table tab:sentence, perfect-filter precision, non-t. | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.perfect_filter_pair_precision.precision` | 0.793548 (OK); = 246/310 |
| .699 | Table tab:sentence, perfect-filter precision, 122B | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.perfect_filter_pair_precision.precision` | 0.698864 (OK); = 246/352 |
| <= .006 | Table tab:sentence caption: "AUPRC of three-checkpoint ensembles (per-seed standard deviation <= .006)" -- see FLAG 5 | `biored_sentence.json`, `biodiv_sentlab.json` | max over `arms.*.auprc_sd` (BioRED, 3 arms) and `by_silver.*.arms.*.auprc_seed_sd` (biodiv, 4 arms x 3 defs), ddof=1 | max = 0.006264 at `by_silver.q122b.arms.pair-own.auprc_seed_sd` (> .006 before rounding); BioRED max 0.003442; others: arms.pair-max=0.0034, arms.sentlab=0.0031, arms.sentence=0.0018, majority3.sentence=0.0013, majority3.pair-own=0.0059, majority3.pair-max=0.0029, majority3.sentlab=0.0014, nonteacher2.sentence=0.0016, nonteacher2.pair-own=0.0058, nonteacher2.pair-max=0.0044, nonteacher2.sentlab=0.0010, q122b.sentence=0.0020, q122b.pair-own=0.0063, q122b.pair-max=0.0039, q122b.sentlab=0.0011 |
| 2,582 | Table tab:sentence caption and Sec. sentence Labels: "2,582 BC8 test sentences" | `results/paperA_v2/sentence_level/biored_sentence.json` | `n_sentences` | 2582 |
| 191 | Table tab:sentence caption, Sec. sentence (x2), Limitations (x2): "the 191 others" | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `n_pair_negative` | 191; under nonteacher2 only 148 of them are kept (`by_silver.nonteacher2.decomposition.pair_negative_only.n`) |
| 394 | Table tab:sentence caption: "(non-t., 394 passages)" | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.n` | 394 |
| 0.854 | Table tab:sentence caption: "Zero-shot rows: base rate 0.854" | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.positive_rate` | 0.854 (OK) |
| 0.921 | Table tab:sentence caption: "(accept-all 0.921)" | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.trivial_accept_all_F1` | 0.921251 (OK) |
| 0.941 | Table tab:sentence caption: "the trained models score 0.941, 0.958 and 0.973" (sentence arm) | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.trained_arms_same_sentences.sentence.auprc_ensemble` | 0.941346 (OK) |
| 0.958 | Table tab:sentence caption: same (sentence-label model) | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.trained_arms_same_sentences.sentlab.auprc_ensemble` | 0.957926 (OK) |
| 0.973 | Table tab:sentence caption: same (pair-max) | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.trained_arms_same_sentences.pair-max.auprc_ensemble` | 0.973371 (OK) |
| 500 | Table tab:sentence caption and Sec. sentence BioRED: "500 random (test) sentences" | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.n_sentences` | 500 |
| 0.838 | Sec. sentence, Labels: "(2,582 test sentences, 0.838 positive)" | `results/paperA_v2/sentence_level/biored_sentence.json` | `positive_rate` | 0.83811 (OK) |
| 246 | Sec. sentence, Labels: "pair-positive (246)"; Limitations: "gold decides the 246" | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.perfect_filter_pair_precision.pair_positive` | 246; = 437 x `pair_positive_rate` 0.562929 |
| 34,242 | Sec. sentence, Labels: "on all 34,242 corpus passages" | `results/paperA_v2/sentence_level/sentlab_label_stats.json` | `labelled_passages` | 34242; students train on 30,818 / dev 3,424 (`sentlab_split.json`) |
| 14.3 | Sec. sentence, Labels: "14.3 pairs per passage"; Limitations: "scores 14.3 pairs per passage" | `results/paperA_v2/sentence_level/biodiv_enum_cost.json` | `enumeration.pairs_per_passage_incl_candidate.mean` | 14.3089 (OK); = 6,253/437 |
| +0.035 [+0.024, +0.047] | Sec. sentence, BioRED: "survives resampling abstracts as clusters (AUPRC +0.035, 95% interval [+0.024, +0.047])" | `results/paperA_v2/sentence_level/biored_sentence.json` | `bootstrap_auprc_abstracts.diff.pair-max - sentence.observed` | 0.0351337 (OK) |
|  | (CI of the row above) | `results/paperA_v2/sentence_level/biored_sentence.json` | `bootstrap_auprc_abstracts.diff.pair-max - sentence.ci95` | [0.0236535, 0.0474944] (OK); upper 0.047494 -> .047 (correct) |
| +0.024 | Sec. sentence, BioRED: "better at the operating point (F1 +0.024 [+0.017, +0.031]" | `results/paperA_v2/sentence_level/biored_sentence.json` | `bootstrap_f1_abstracts.diff.pair-max - sentlab.observed` | 0.0238657 (OK) |
| [+0.017, +0.031] | (CI of the row above; abstract-cluster bootstrap, not labelled in the text) | `results/paperA_v2/sentence_level/biored_sentence.json` | `bootstrap_f1_abstracts.diff.pair-max - sentlab.ci95` | [0.016734, 0.0314298] (OK); sentence-resampled would be [+0.018, +0.030] |
| 158 / 41 / 2.1e-17 | Sec. sentence, BioRED: "158 sentences fixed, 41 broken, exact McNemar p = 2.1e-17" | `results/paperA_v2/sentence_level/biored_sentence.json` | `mcnemar_exact.pair-max vs sentlab.fixes` | 158; fixes=158, breaks=41, p_exact=2.126e-17; "a vs b" fixes = b wrong & a right (`common.mcnemar_exact` called as (preds[b], preds[a])), so fixed = by pair-max (OK) |
| +0.010 | Sec. sentence, BioRED: "its AUPRC gain is marginal, +0.010 [-0.001, +0.021]" | `results/paperA_v2/sentence_level/biored_sentence.json` | `bootstrap_auprc_abstracts.diff.pair-max - sentlab.observed` | 0.00993279 (OK) |
| [-0.001, +0.021] | (CI of the row above; abstract-cluster bootstrap, not labelled) | `results/paperA_v2/sentence_level/biored_sentence.json` | `bootstrap_auprc_abstracts.diff.pair-max - sentlab.ci95` | [-0.00130189, 0.0208277] (OK); sentence-resampled CI [+0.000, +0.020] would exclude 0 |
| 777 | Sec. sentence, BioRED: "the 777 single-pair sentences" | `results/paperA_v2/sentence_level/biored_sentence.json` | `strata.candidate_pairs.1.n` | 777 |
| +0.044 [+0.016, +0.073] | Sec. sentence, BioRED: "(+0.044 [+0.016, +0.073])" | `results/paperA_v2/sentence_level/biored_sentence.json` | `strata.candidate_pairs.1.bootstrap_auprc_sentences.diff.pair-max - sentlab.observed` | 0.0440677 (OK) |
|  | (CI of the row above; SENTENCE-resampled -- see FLAG 8) | `results/paperA_v2/sentence_level/biored_sentence.json` | `strata.candidate_pairs.1.bootstrap_auprc_sentences.diff.pair-max - sentlab.ci95` | [0.0156506, 0.0734064] (OK); clustered: [0.0118, 0.0789] |
| +0.006 [-0.003, +0.014] | Sec. sentence, BioRED: "not in those with several candidates (+0.006 [-0.003, +0.014])" | `results/paperA_v2/sentence_level/biored_sentence.json` | `strata.candidate_pairs.>=2.bootstrap_auprc_sentences.diff.pair-max - sentlab.observed` | 0.00588736 (OK) |
|  | (CI of the row above; sentence-resampled) | `results/paperA_v2/sentence_level/biored_sentence.json` | `strata.candidate_pairs.>=2.bootstrap_auprc_sentences.diff.pair-max - sentlab.ci95` | [-0.0027294, 0.0137483] (OK); clustered: [-0.0032, 0.0146] |
| 0.959 vs 0.908 | Sec. sentence, BioRED: "the teacher reaches 0.959 against 0.908" | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.llm pair-max.auprc` | 0.959135 (OK); vs `llm.qwen3-32b random500.llm sentence.auprc` = 0.908064 (OK) |
| +0.051 [+0.025, +0.079] | Sec. sentence, BioRED: "(+0.051 [+0.025, +0.079])" | `results/paperA_v2/sentence_level/biored_sentence.json` | `llm.qwen3-32b random500.bootstrap_auprc.diff.llm pair-max - llm sentence.observed` | 0.0510712 (OK); ci95 = [0.02506, 0.0794] (OK) |
| +0.014 [+0.002, +0.027] | Sec. sentence, Biodiversity: "(+0.014 [+0.002, +0.027]" (pair-max - candidate-trained arm, nonteacher2) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.bootstrap_auprc.diff.pair-max - sentence.observed` | 0.0139184 (OK); ci95 = [0.002137, 0.026508] (OK) |
| +0.016 [+0.005, +0.029] | Sec. sentence, Biodiversity: "and +0.016 [+0.005, +0.029]" (pair-max - candidate-trained arm, q122b) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.bootstrap_auprc.diff.pair-max - sentence.observed` | 0.0162578 (OK); ci95 = [0.004695, 0.028953] (OK) |
| +0.008 [-0.005, +0.021] | Sec. sentence, Biodiversity: "majority +0.008 [-0.005, +0.021]" (pair-max - candidate-trained arm, majority3) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.bootstrap_auprc.diff.pair-max - sentence.observed` | 0.00832741 (OK); ci95 = [-0.004757, 0.021499] (OK) |
| 0.020 | Sec. sentence, Biodiversity: "beats pair-max under all three: by 0.020, 0.010 and 0.012 AUPRC, every interval excluding zero" (majority3) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.bootstrap_auprc.diff.pair-max - sentlab.observed` | -0.0195224 (OK); ci95 = [-0.03146, -0.00951], excludes 0 (OK) |
| 0.010 | Sec. sentence, Biodiversity: "beats pair-max under all three: by 0.020, 0.010 and 0.012 AUPRC, every interval excluding zero" (nonteacher2) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.bootstrap_auprc.diff.pair-max - sentlab.observed` | -0.00993327 (OK); ci95 = [-0.01939, -0.00159], excludes 0 (OK) |
| 0.012 | Sec. sentence, Biodiversity: "beats pair-max under all three: by 0.020, 0.010 and 0.012 AUPRC, every interval excluding zero" (q122b) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.bootstrap_auprc.diff.pair-max - sentlab.observed` | -0.0122715 (OK); ci95 = [-0.0222, -0.00399], excludes 0 (OK) |
| 0.929 vs 0.897 | Sec. sentence, Biodiversity: "and in F1, 0.929 against 0.897" (majority3, not stated) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.arms.sentlab.F1` | 0.929293 (OK); pair-max `majority3.arms.pair-max.F1` = 0.896747 (OK) |
| 46 / 22 / p = 0.005 | Sec. sentence, Biodiversity: "it is right on 46 passages where pair-max is wrong and wrong on 22, p = 0.005" -- see FLAG 7 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.mcnemar_exact.pair-max vs sentlab.breaks` | 46; breaks=46 (pair-max wrong, sentlab right), fixes=22; p_exact=0.00490338 -> 2 s.f. 0.0049 (ROUNDING) |
| 0.953 | Sec. sentence, Biodiversity: "there it reaches AUPRC 0.953, pair-max 0.788 ..., and pair-own 0.658 (majority labels)" (sentlab) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.decomposition.pair_negative_only.auprc.sentlab` | 0.953402 (OK) |
| 0.788 | Sec. sentence, Biodiversity: "there it reaches AUPRC 0.953, pair-max 0.788 ..., and pair-own 0.658 (majority labels)" (pair-max) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.decomposition.pair_negative_only.auprc.pair-max` | 0.788032 (OK) |
| 0.658 | Sec. sentence, Biodiversity: "there it reaches AUPRC 0.953, pair-max 0.788 ..., and pair-own 0.658 (majority labels)" (pair-own) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.decomposition.pair_negative_only.auprc.pair-own` | 0.65759 (OK) |
| -0.004 | Sec. sentence, Biodiversity: "(pair-max minus sentence-label model -0.004 [-0.014, +0.005], non-teacher labels)" | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.decomposition.gold_pos_vs_silver_neg.bootstrap_auprc.diff.pair-max - sentlab.observed` | -0.00356572 (OK) |
| [-0.014, +0.005] | (CI of the row above) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.decomposition.gold_pos_vs_silver_neg.bootstrap_auprc.diff.pair-max - sentlab.ci95` | [-0.0135082, 0.00531164] (OK); lower -0.013508 -> -0.014 (correct) |
| 85.9 | Sec. sentence, Biodiversity: "85.9 forward passes per passage against 3, while pair-own, at 6" | `results/paperA_v2/sentence_level/biodiv_enum_cost.json` | `enumeration.forward_passes_per_passage.pair_max.mean` | 85.8535 (OK); 2 orders x 14.31 pairs x 3 seeds |
| 3 | Sec. sentence, same (sentence arm; sentlab is also 1 x 3 seeds) | `results/paperA_v2/sentence_level/biodiv_enum_cost.json` | `enumeration.forward_passes_per_passage.sentence.mean` | 3 (OK) |
| 6 | Sec. sentence, same (pair-own) | `results/paperA_v2/sentence_level/biodiv_enum_cost.json` | `enumeration.forward_passes_per_passage.pair_own.mean` | 6 (OK) |
| 0.006 ("within") | Sec. sentence, Biodiversity: "pair-own, at 6, is within 0.006 AUPRC of it" (majority3) -- see FLAG 6 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.bootstrap_auprc.diff.pair-max - pair-own.observed` | 0.00338963 |
| 0.006 ("within") | Sec. sentence, Biodiversity: "pair-own, at 6, is within 0.006 AUPRC of it" (nonteacher2) -- see FLAG 6 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.nonteacher2.bootstrap_auprc.diff.pair-max - pair-own.observed` | 0.00381524 |
| 0.006 ("within") | Sec. sentence, Biodiversity: "pair-own, at 6, is within 0.006 AUPRC of it" (q122b) -- see FLAG 6 | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.bootstrap_auprc.diff.pair-max - pair-own.observed` | 0.00631298 |
| 0.724 [0.674, 0.768] | Sec. sentence, "The pair cannot be read off ...": "pair-level precision of 0.724 [0.674, 0.768] on ours" (majority3; Wilson) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.majority3.perfect_filter_pair_precision.wilson95` | [0.673692, 0.768372] (OK); bootstrap95 would be [0.675, 0.769] |
| 0.536 [0.527, 0.544] | Sec. sentence, same: "0.536 [0.527, 0.544] on BioRED" (Wilson) | `results/paperA_v2/sentence_level/biored_sentence.json` | `perfect_filter_pair_precision.wilson95` | [0.527376, 0.543695] (OK) |
| 0.516 | Sec. sentence, same: "(0.516 in sentences with several candidates)" | `results/paperA_v2/sentence_level/biored_sentence.json` | `perfect_filter_pair_precision.by_candidate_pairs.>=2.precision` | 0.516155 (OK) |
| 0.794 | Sec. sentence, same: "against our pair gold the sentence-label model reaches AUPRC 0.794" | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `pair_gold_view.sentlab.auprc` | 0.793655 (OK) |
| 0.840 | Sec. sentence, same: "and pair-max 0.840" | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `pair_gold_view.pair-max.auprc` | 0.839691 (OK) |
| 0.923 | Sec. sentence, same: "the per-pair scores 0.923" (pair-own ensemble; = paperA scaling-table "trained, same rows" pair 0.923) | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `pair_gold_view.pair-own.auprc` | 0.922928 (OK) |
| 41% | Sec. sentence, same: "the teacher's two answers differ on 41% of corpus passages" -- see FLAG 1 | `results/paperA_v2/sentence_level/sentlab_label_stats.json` | derived: (`crosstab.sentence_yes_no_candidate_pos` + `crosstab.sentence_no_candidate_pos`) / `labelled_passages` = (12503 + 1422) / 34242 | 13925/34242 = 0.406664 -> 41% (OK) |
| 0.26 | Sec. sentence, same: "(Cohen's kappa = 0.26)" | `results/paperA_v2/sentence_level/sentlab_label_stats.json` | `kappa_sentence_vs_any_candidate` | 0.263629 (OK) |
| 0.699 to 0.794 | Limitations, "biodiversity sentence labels are provisional": "the precision ceiling of 0.724 moves with the definition (0.699 to 0.794)" | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | `by_silver.q122b.perfect_filter_pair_precision.precision` / `by_silver.nonteacher2.perfect_filter_pair_precision.precision` | 0.698864 / 0.793548 (OK) |
| "most of its gain" | Limitations, "biodiversity sentence labels are provisional": "the candidate pair alone gives most of its gain" | `results/paperA_v2/sentence_level/biodiv_sentlab.json` | pair-own - sentence vs pair-max - sentence, `by_silver.*.bootstrap_auprc.diff` and `bootstrap_f1.diff` | AUPRC: 0.0049/0.0083, 0.0101/0.0139, 0.0099/0.0163 (59-73%, OK); F1 q122b: 0.0052/0.0394 (13%, does not hold for F1) |
| 0.97 / 25% | Related work: "an abstract filter for microbial interactions with an AUC of 0.97 gives 25% precision on the species pairs read off it" | `results/reframe_2026-10-05/LITERATURE.md` (lim2016minter rows, lines ~101, ~171) | literature value, not a results/ JSON (by design) | AUC 0.97; species-pair precision 25% (abstract of Lim et al. 2016) |

Count: 104 printed numbers traced (rows with a printed value; CI-continuation rows excluded). That includes 2 context-changed numbers from paperA.tex, 1 bound (≤ .006), 1 wording quantity ("most of its gain") and 1 literature pair. No value mismatches; issues are listed under FLAGS.
