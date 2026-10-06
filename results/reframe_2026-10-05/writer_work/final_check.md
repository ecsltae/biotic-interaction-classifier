# Final check of `paperA_reframe.snapshot_0501.tex` (against `snapshot_0445.tex`), 2026-10-05

Method: line diff of the two snapshots; every number in changed hunks extracted and traced; labels/refs
checked programmatically (no dangling `\ref`; all citation keys resolve in `references.bib` +
`reframe_extra.bib`); near-duplicate sentence scan; abstract word count 198 including the Code line
(was 197; limit 200). Line numbers below refer to `snapshot_0501.tex`.

## NEW PROBLEMS (most important first)

**P1. The spot check is reported only under its best-agreeing silver definition (l.642-644).**
- Quote: "they agree with the non-teacher labels on 13 of 14 ($\kappa = 0.81$), too few to decide, but in
  the silver labels' favour."
- Why: every biodiversity headline in §4.3 (F1 0.929 vs 0.897, $p = 0.0049$, 0.953/0.788/0.658, the 0.724
  ceiling) uses the *majority* labels. Against those the same 19 answers agree 15 of 19, $\kappa = 0.46$
  (`biodiv_human.json: gold_review_spot_check.sentence_vs_silver.majority3`; Qwen3.5-122B: 17/19,
  $\kappa = 0.68$). Quoting only $\kappa = 0.81$ makes the caveat one-sided in the other direction. Also,
  5 of the 19 are pair-positive, so their silver label is YES by construction (ANALYSIS §7), and 3 UNSURE
  answers were dropped to reach 19 (LITERATURE "must not" item 12: dropping UNSURE is not harmless; it is
  not disclosed). "Selected rather than random" is slightly off: half the 22 items are random controls
  (Limitations l.629-630 says so); the set as a whole is not a random sample.
- Fix: "The only human sentence answers so far are 19 items (3 UNSURE answers dropped) of an expert
  adjudication of a pre-filled sheet, half selected by model-gold disagreement and not blind; 5 are
  pair-positive, so their silver label is fixed by gold. They agree with the non-teacher labels on 13 of
  14 ($\kappa = 0.81$), with Qwen3.5-122B on 17 of 19 ($\kappa = 0.68$) and with the majority labels on
  15 of 19 ($\kappa = 0.46$): too few to decide."

**P2. "The relation term adds nothing" is now stated unqualified in the conclusion, and the rules appendix
says the opposite.**
- Quotes: Conclusion l.611-612 (new): "it lifts zero-shot LLMs at every size from 1.7B, and the relation
  term adds nothing." App. rules l.1094 (new): "On the triple-conditioned model, which leans on its
  relation term, they do more".
- Why: §4.1 l.297-299 says "The one place the relation term earns its keep is at strict operating points"
  and Limitations l.675-678 leaves that open; the new §4.1 paragraph (l.322-323) and App. ablation
  (l.1254-1255) say replacing or shuffling the relation "barely moves" the triple model's ranking and
  "the model leans on the pair". So the conclusion overstates and the rules appendix contradicts the
  ablation.
- Fix: conclusion "and the relation term adds nothing to ranking"; rules appendix delete ", which leans
  on its relation term," (or write "whose thresholded decisions shift when the relation changes,
  Table~\ref{tab:ablation}").

**P3. Self-referencing `\ref` and a duplicated paragraph in Appendix "Zero-shot scaling" (l.981-993).**
- The paragraph moved out of §4.1 now sits inside `app:scaling` and cites itself: "From Qwen3-1.7B to
  Qwen3.5-122B the pair question leads, by $+0.04$ to $+0.20$ AUPRC on our benchmark and $+0.11$ to
  $+0.32$ on BioRED (Appendix~\ref{app:scaling})." The next paragraph (l.996-1000) gives the same range
  again ("by $+0.040$ AUPRC at 1.7B and by $+0.10$ to $+0.20$ from 4B on ($+0.11$, then $+0.20$ to
  $+0.32$, on BioRED)").
- Its first sentences are also a near-verbatim copy of §4.1 l.301-304 ("Asked the sentence-level
  question zero-shot, the 32B teacher scores AUPRC $0.789$ ... Its gap, $+0.155$, is twice the
  student's and sits in the same place").
- It also says "On BioRED the teacher" (l.981), which S8(e) asked to call "the 32B model" (done in §4.3).
- Fix: in the appendix, delete the sentence "From Qwen3-1.7B ... (Appendix~\ref{app:scaling})." and start
  the paragraph at "Its gap on our benchmark, $+0.155$, sits ... $+0.180$ on passages naming three or more
  taxa against $+0.104$ on the rest, and the relation term adds nothing to it ($0.946$ against
  $0.943$)."; write "On BioRED the 32B model, asked ...".

**P4. Duplicated paragraph inside Appendix "Candidate rules in detail".**
- The rewritten opening paragraph (l.1086-1091: "No rule has a parameter fitted to evaluation labels, and
  since four were written after reading the false positives above, each had to pass a criterion fixed in
  advance on the 42{,}747 training rows (at least 85\% ...); they were then frozen, recorded by hash and
  evaluated once.") repeats the same appendix's paragraph "How they were validated" (l.1134-1144: "No rule
  has a parameter fitted to evaluation labels, but rules can still be fitted by \emph{choice}, and four
  ... 42{,}747 training rows, at least 85\% ... The rules were then frozen, recorded by hash, and evaluated
  once."). "The false positives above" is also vague in an appendix.
- Fix: shorten the opening to "We answer them with eight deterministic rejection rules applied before the
  verifier decides, validated as described below."

**P5. "Deployed" wording left from M7 (anonymity; inherited text, but M7 listed it).**
- Quotes: l.1030 "The deployed model with its candidate rules", l.1032 "Band $0\%$ is the deployed model
  alone", l.1037 "The deployed model is cheap and recall-oriented", l.1330 "The deployed model's four-way
  direction output", l.1345 "on the deployed model's accept set"; also l.1208-1209 "buys throughput the
  pipeline does not need" and l.1209 "the deployment signal distribution" (the pipeline = BiotXplorer's,
  per l.60).
- Fix: "the single-model configuration" for "the deployed model"; "buys throughput this setting does not
  need"; "toward the signal distribution of the retrieval pool". New text itself passes anonymity (no
  names, affiliations, URLs, or BiotXplorer deployment claims).

**P6. The spot-check sentence and §4.3 now use "blind" correctly** ("it is not blind", l.630; "not
blind", l.642). No other occurrence. PASS, listed for completeness.

**P7. Minor.**
- Abstract l.34-36: "(BioRED AUPRC $0.965$ against $0.930$) ... ($0.955$; F1 $0.954$ against $0.930$)":
  0.930 means two different things in one sentence, and 0.955 is the competitor's AUPRC with pair-max's
  0.965 left implicit. Fix: "($0.965$ against $0.955$ AUPRC; F1 $0.954$ against $0.930$)". The numbers
  themselves are correct (the claims review's proposed "F1 $0.930$ against $0.954$" had the order reversed;
  the draft is right).
- Abstract l.33-34: "beats a sentence classifier trained on the same candidates" is scoped to BioRED only
  by the parenthesis; the conclusion says "on ours under two of three label definitions". Optional: "on
  BioRED (AUPRC ...)".
- l.444: "$65.5\%$ of which it calls positive": "it" reads as the sentence-label model; the 65.5% is the
  teacher's YES rate. Fix: "of which the teacher calls $65.5\%$ positive".
- Intro l.75-76: "the maximum of its scores over the passage's candidate pairs"; on our benchmark pair-max
  runs over the candidate and all TaxoNERD pairs. Fix: "over the passage's pairs".
- "Pair-max is also told which spans are BioRED's gold entities" appears twice (§4.3 l.458; Limitations
  l.688). Acceptable; could cut one.
- App. artifacts l.1176: Qwen3.8-27B has no licence stated (S13 asked for one).
- Related work l.536-537 (l.537): "a claim-only classifier matches evidence-aware models": Schuster et al. say it
  "performs competitively with" them; "matches" slightly overstates.
- Inherited, not new: App. rules gives $p = 0.067$ for 14 fixed / 5 broken; the exact binomial McNemar is
  $0.064$ (continuity-corrected chi-square gives 0.067). Elsewhere the paper says "exact McNemar".

No contradiction remains between abstract, introduction, §4.3, conclusion and Limitations on the
sentence-level results, the 0.724/0.536 ceilings, or the provisional status; nothing new goes beyond
ANALYSIS §9; none of the 20 "claims we must not make" is made in new text (li2021dual and lim2016minter
are cited where item 3/4 require).

## MUST-FIX STATUS

| Item | Status | Revised text |
|---|---|---|
| M1 | RESOLVED | Abstract: "and on BioRED from $0.738$ to $0.843$, gaining only on sentences naming more than two entities." |
| M2 | RESOLVED | Abstract: "A perfect sentence filter has a pair precision of $0.536$ on BioRED (provisionally $0.724$ on ours)."; intro: "at a pair precision of $0.536$ on BioRED and, provisionally, $0.724$ on ours"; §4.3: "and, on our silver labels, $0.724$ $[0.674, 0.768]$ ($0.699$ to $0.794$ across definitions)"; conclusion: "and, provisionally, $0.724$ here". "needs no silver label" is gone. |
| M3 | RESOLVED | Limitations: "On these labels the sentence-label model is the better sentence filter under every definition, but its whole advantage lies where the labels are an LLM's answer to the very prompt it imitates, so only human sentence labels can confirm it." (see P1 for the spot-check sentence that follows) |
| M4 | RESOLVED | Abstract: "Ask each question for its own decision; one teacher can label both."; intro: "the pair decision needs the pair question, pair verification is a competitive but, provisionally, not the best sentence filter". |
| M5 | RESOLVED | §4.3: "If every interacting pair of a passage is among the pairs scored, the passage describes an interaction exactly when one of them interacts ... is the natural sentence score; how far the scored pairs fall short of a passage's interactions is what the two benchmarks measure." Intro made conditional too. |
| M6 | RESOLVED | Limitations: "have gone through an expert adjudication of a pre-filled sheet; it is not blind, and no label in this paper is changed on its strength or on a model's disagreement." |
| M7 | PARTLY | Done: §6 "Candidate rules and a CPU configuration"; appendix "The single-model CPU configuration in detail"; "In the pipeline" removed; "faster than the pipeline retrieves them" deleted. Left: "the deployed model" at l.1030, 1032, 1037, 1330, 1345, and "throughput the pipeline does not need" l.1207 (P5). |
| M8 | RESOLVED | Abstract: "one trained on BioRED's derived sentence labels ($0.955$; F1 $0.954$ against $0.930$)"; conclusion: "one trained on BioRED's derived sentence labels". Readability of the parenthesis: P7. |
| M9 | RESOLVED | Abstract: "is better under all three definitions (AUPRC $0.985$--$0.987$ against $0.966$--$0.977$)"; the mechanism clause is deleted. |

## NEW NUMBERS

Numbers in changed hunks that are not in `snapshot_0445` (or are in it only elsewhere/with another role).
Moved-only text (§4.1 "The query, not the encoder", App. "A bound on component-wise verification",
App. "A gold label changed after freezing", the teacher paragraphs moved to App. scaling) carries the same
numbers as 0445 and is not re-traced. SL = `results/paperA_v2/sentence_level/`.

| Number as printed | Where | Source file | JSON key or derivation | Exact value | OK? |
|---|---|---|---|---|---|
| F1 $0.775$ to $0.871$ | Abstract l.32 | 0445 §4.1 / `paperA/paperA.tex` | unchanged from body (§4.1 l.288-289) | 0.775, 0.871 | OK |
| $0.985$--$0.987$ | Abstract l.39 | SL/biodiv_sentlab.json | `by_silver.{majority3,nonteacher2,q122b}.arms.sentlab.auprc` | 0.98503, 0.98730, 0.98613 | OK |
| $0.966$--$0.977$ | Abstract l.39 | SL/biodiv_sentlab.json | `by_silver.{majority3,nonteacher2,q122b}.arms.pair-max.auprc` | 0.96551, 0.97736, 0.97386 | OK |
| 31, 27 and 54 | §2 l.162 | `paperA/paperA.tex` l.147 | unchanged from paperA.tex | 31, 27, 54 | OK |
| $< .007$ | tab:sentence caption l.412 | SL/biodiv_sentlab.json, SL/biored_sentence.json | max of `by_silver.*.arms.*.auprc_seed_sd`, `arms.*.auprc_sd` (max = q122b pair-own) | 0.006264 | OK ("$\le .006$" was wrong) |
| 43 | tab:sentence caption l.418 | SL/biodiv_sentlab.json | 437 - 394; = 191 - `by_silver.nonteacher2.decomposition.pair_negative_only.n` (148) | 43 | OK |
| $65.5\%$ | §4.3 l.444 | SL/sentlab_label_stats.json | `sentence_yes_rate` | 0.654518 | OK (wording: P7) |
| $+0.012$, $+0.079$ | §4.3 l.454 | SL/biored_sentence.json | `strata.candidate_pairs.1.bootstrap_auprc_abstracts.diff.pair-max - sentlab.ci95` | [0.011809, 0.078892] | OK |
| $+0.044$ | §4.3 l.453 | SL/biored_sentence.json | same `.observed` | 0.044068 | OK (unchanged) |
| $-0.003$, $+0.015$ | §4.3 l.454 | SL/biored_sentence.json | `strata.candidate_pairs.>=2.bootstrap_auprc_abstracts.diff.pair-max - sentlab.ci95` | [-0.003181, 0.014638] | OK |
| $+0.006$ | §4.3 l.454 | SL/biored_sentence.json | same `.observed` | 0.005887 | OK (unchanged) |
| $p = 0.0049$ | §4.3 l.468 | SL/biodiv_sentlab.json | `by_silver.majority3.mcnemar_exact.pair-max vs sentlab.p_exact` (22 vs 46) | 0.0049034 | OK |
| $0.699$ to $0.794$ | §4.3 l.479 | SL/biodiv_sentlab.json | `by_silver.{q122b,nonteacher2}.perfect_filter_pair_precision.precision` | 0.69886, 0.79355 | OK (moved from Limitations) |
| $74\%$ | §6 l.527 | `paperA/paperA.tex` tab:direction | 62 / 84 | 0.7381 | OK |
| 148 | Limitations l.638 | SL/biodiv_sentlab.json | `by_silver.nonteacher2.decomposition.pair_negative_only.n` | 148 | OK |
| 19 items | Limitations l.641 | SL/biodiv_human.json | `gold_review_spot_check.sentence_labels.labelled_rows` (22 answers - 3 UNSURE) | 19 | OK (UNSURE drop undisclosed: P1) |
| 13 of 14 | Limitations l.642 | SL/biodiv_human.json | `gold_review_spot_check.sentence_vs_silver.nonteacher2.n` = 14, `.agreement` = 0.92857 | 13/14 | OK (selective: P1) |
| $\kappa = 0.81$ | Limitations l.643 | SL/biodiv_human.json; ANALYSIS §7 | `gold_review_spot_check.sentence_vs_silver.nonteacher2.kappa` | 0.81081 | OK (selective: P1) |
| $0.003$ to $0.006$ | Limitations l.648 | SL/biodiv_sentlab.json | `by_silver.{majority3,nonteacher2,q122b}.bootstrap_auprc.diff.pair-max - pair-own.observed` | 0.00339, 0.00382, 0.00631 | OK |
| $85.9$, 3, 6 passes | Limitations l.647-648 | 0445 §4.3 | moved, unchanged | - | OK |
| 15{,}640 | Limitations l.689 | `paperA/paperA.tex` l.350 | BioRED train+dev candidates; = sum of `n_candidates` over the sentlab training sentences | 15,640 | OK |
| 3{,}277 | Limitations l.689 | SL/biored_sentence.json | `sentlab_training_sentences.n` | 3277 | OK |
| 14 fixed and 5 broken | App. rules l.1095 | `paperA/paperA.tex` tab:rules, l.502 | unchanged from paperA.tex | 14, 5 | OK (inherited $p = 0.067$ is chi-square with continuity correction; exact = 0.064) |
| 2\,h 15\,min | App. artifacts l.1183 | ANALYSIS.md §6 ("Actual time: 2 h 15 min") | no JSON key; SL/sentlab_throughput.json gives 0.203 s/passage at 4 workers (about 1 h 56 min for 34,242) | 2 h 15 min | OK |
| 34{,}242 | App. artifacts l.1182 | ANALYSIS §6 | unique corpus passages (already in 0445 §4.3) | 34,242 | OK |
| $57$--$95\%$ | App. curve l.1313 | `paperA/paperA.tex` l.599-600 | unchanged from paperA.tex (swaminathan2024selective) | 57-95% | OK |
| six negative results | Contributions l.98 | tab:negative | six table rows | 6 | OK |
| four-way direction | Contributions l.98 | App. deploy | four categories (Forward, Reverse, Bidirectional, Uncertain) | 4 | OK |

Every new number traces to its source with correct rounding. Two are worded or selected in a misleading
way, not wrong: 65.5% (whose answers) and $\kappa = 0.81$ (only the best definition reported; P1).
