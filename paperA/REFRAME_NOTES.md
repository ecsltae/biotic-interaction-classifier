# Paper A: notes on the sentence-level reframe draft

Writer stage, 2026-10-05 (unattended, 04:35–06:30). Paths are relative to `~/MetaP/classifier`.

## 0. What and where

- **The draft:** `paperA/paperA_reframe.tex` → `paperA/paperA_reframe.pdf` (21 pages). It is a separate draft for
  you to compare with the submission version.
  - `paperA/paperA.tex`, `paperA.pdf`, `references.bib` and `SUBMISSION.md` were not touched.
  - Bibliography: `\bibliography{references,reframe_extra}`. `reframe_extra.bib` comes from the literature stage.
- **Format** (as for the submission):
  - Review mode with line numbers.
  - 0 LaTeX errors and 0 overfull boxes (`grep -a` on `paperA_reframe.log`).
  - The body ends on page 8: the Conclusion closes page 8, and the Limitations start on page 9.
  - The abstract is 194 words, counted as for the submission's 196: math = 1 word, the code line included.
  - The bibtex message `Illegal, another \bibstyle command` also appears in the submission build. `acl.sty` causes
    it and it is harmless.
- **Rebuild:**
  `cd paperA && timeout 120 pdflatex paperA_reframe.tex && bibtex paperA_reframe && pdflatex paperA_reframe.tex && pdflatex paperA_reframe.tex`.
- **Checks on the draft:** four independent read-only checkers. Their reports are in
  `results/reframe_2026-10-05/writer_work/`:
  - `numbers.md`: every new number in the first draft, traced to its JSON key.
  - `claims_review.md`: scope against ANALYSIS.md §9, the 20 "claims we must not make", anonymity.
  - `dropped_content.md`: what the first draft dropped from `paperA.tex`.
  - `final_check.md`: the 05:01 revision, the new numbers and the status of every must-fix.

  How their findings were handled is in §7 below.

## 1. The result that shaped the draft

The reframe set out to test "ask about the pair, even if you want the sentence". The analysis stage's fair
biodiversity competitor (A1) was added for this test. It is BiomedBERT trained with the paper's recipe on the
teacher's (Qwen3-32B) answers to the sentence question for all 34,242 corpus passages. It **beats** the pair route
(pair-max) at the sentence level under all three silver label definitions, significantly. ANALYSIS.md §6 and §9
have the details. The draft therefore does **not** claim that pair verification is the better sentence filter.
What it claims is what ANALYSIS.md §9 supports, scoped per benchmark:

1. **Only pair answers can populate a pair database.**
   - A perfect sentence filter has a pair precision of 0.536 on BioRED. On ours it is 0.724 *provisionally*: it
     depends on the silver sentence labels, 0.699 to 0.794 across definitions.
   - Sentence scores rank pairs poorly against pair gold. This needs no silver label.
2. **Pair-max beats a sentence classifier trained on the same candidates.**
   - On BioRED it holds robustly.
   - On ours it holds under two of the three silver definitions.
3. **For zero-shot LLMs, the pair question read through a max beats the sentence question at the sentence
   level** (BioRED, Qwen3-32B, 500 random sentences).
4. **Against a classifier trained on sentence labels, the result depends on the benchmark.**
   - BioRED: pair-max is at least as good and better at the operating point. The AUPRC gain is not significant
     with abstract clusters, and the edge lies in single-pair sentences, not where pairs compete.
   - Ours: the sentence-label model is better on provisional silver labels. Its whole advantage is on the 191
     pair-negative passages, where the labels are LLM answers to the prompt it imitates, so it is partly circular.
5. **The recommendation:** "ask each question for its own decision; one teacher can label both", not "ask about
   the pair even if you want the sentence".

The A1 result is stated in the abstract, the introduction, §4.3, the conclusion and the Limitations. It is not
buried.

## 2. What changed, section by section, and why

| Section | Change | Why |
|---|---|---|
| Title | "Which Pair Is It About? Sentence and Pair Decisions for Literature-Mined Biotic Interactions". The submission's subtitle was "Pair-Conditioned Verification of Literature-Mined Biotic Interactions". | Two decisions now; the pair question is still the core. |
| Abstract | Rewritten around the two decisions. It keeps the controlled comparison: AUPRC, F1, McNemar, and BioRED with its localisation. It adds the sentence-level result: the BioRED win against both sentence models, the provisional biodiversity loss to the sentence-label model, and the precision ceiling. The deployed-model sentence moved to §6. | Your sentence-level target (2026-10-05). |
| §1 Introduction | Rewritten. Two decisions: the passage, which curation triage asks for (cited to public curation challenges, not to the application's owners), and the pair, which the database records. Two reasons a sentence filter alone is fragile under co-occurrence retrieval. A pair-level paragraph with unchanged results. A sentence-level paragraph, with the outcome per benchmark and the multi-instance assumption cited. Contributions: 3 bullets. The second is new: the sentence-level evaluation. The third merges the error bound, the rules, the CPU configuration with direction output, and the negative results. | Story. Space. |
| §2 Setting | Five changes. (1) BiotXplorer: "This is the application we build for" becomes what its public abstract supports: it shows each interaction with supporting passages, and its authors' evaluations are pair-in-passage judgements. (2) "Why the sentence-level question is constant here" becomes "What retrieval decides, and what it leaves": a keyword test would accept every passage, but the semantic sentence question is not constant (the teacher says no on 34.5% of corpus passages). (3) The query representation is shortened. (4) The evaluation protocol adds the sentence-level statistics (paired bootstrap, exact McNemar, Wilson). (5) The Acanthamoeba relabel moves to Appendix "A gold label changed after freezing", with a pointer kept. | The submission's "answered before it is asked" is false of the semantic sentence question. Anonymity guidance (LITERATURE.md): no wording implying a working relationship. |
| §3 | Unchanged, except that west2022symbolic and gekhman2023trueteacher join the distillation citation (moved from Related work). | Space. |
| §4.1 | Results unchanged. Four edits. (1) The teacher paragraph is shortened; its detail now follows Table `tab:scaling` in its appendix. (2) "What the decision needs is the pair" becomes "What the pair decision needs ...". (3) "Coincide" becomes "nearly coincide". (4) New paragraph "The query, not the encoder" replaces the submission's §4.3–4.4 subsections, whose tables and figure are now in appendices. | Space. Wording made false by the reframe. |
| §4.2 BioRED | Results unchanged. The EP-F1 paragraph is shortened, with its citations kept; the full text is in Appendix "BioRED at the document level". "Much of" the gap is written "in part", not "mostly". The zero-shot teacher sentence moves to the scaling appendix. | Space. Accuracy. |
| **§4.3 From pairs to sentences (new)** | Table `tab:sentence` has four blocks. (1) BioRED rows from exact gold-derived labels. (2) Biodiversity rows under three silver definitions, marked *provisional*. (3) Zero-shot Qwen3-32B rows. (4) The pair precision of a perfect sentence filter. The text has five paragraphs. (1) Pair-max as the multi-instance rule, exact only if the scored pairs cover the passage's interactions. (2) Labels: BioRED = "co-mentions a related pair", where a pair win is expected by construction; biodiversity = silver. (3) BioRED, with abstract-cluster intervals. (4) Biodiversity, provisional: the A1 loss, where it lies, the circularity. (5) The pair cannot be read off a sentence filter. | The added result. |
| §5 Errors | The mechanism paragraph is unchanged. The bound paragraph is shortened to its result; the full derivation is in Appendix "A bound on component-wise verification". | Space. |
| §6 Candidate rules and a CPU configuration | The submission's §6 and §7 merge into one paragraph. It says that four rules were written after reading the false positives. It calls the CPU model a variant of the pair-conditioned model. It gives the direction head's 74% coverage and the expert-chosen abstention. Full texts are in Appendices "Candidate rules in detail" (the submission's §6 paragraph, verbatim except one clause) and "The single-model CPU configuration in detail". | Space. Anonymity: the new text avoids "in the pipeline" / "deployed". |
| §7 Related work | "Partial-input classifiers": the heading is renamed, all citations are kept, and the citation roles are corrected (FEVER for the task, Schuster for the finding; annotator-artefact contrast restored). "Conditioning on the pair": two paragraphs merged; adds lee2020biobert, zhong2021frustratingly and mraz2026fewshot; keeps the "what is ours" statement, now including the sentence-level comparison. "Sentence decisions from pair decisions" and "Sentence filters do not give pairs": new, from LITERATURE.md. "Text mining for interaction databases": compressed. | Prior art for the max over pairs and for the precision ceiling. LITERATURE.md "claims we must not make". |
| §8 Conclusion | Rewritten to the narrowed claim. It keeps the relation-term null and the non-organism locus. |, |
| Limitations | "One benchmark": the review is now "an expert adjudication of a pre-filled sheet; it is not blind, and no label ... is changed", and a second annotation is in progress. New paragraph "The biodiversity sentence labels are provisional" covers: the A1 result stated as a result, with its circularity; the 19-item spot check; the definition-dependent ceiling; pair-max's out-of-distribution pairs and cost; the passage-level split; and the two untested controls. "BioRED's labels are document-level" adds: the derived sentence label's meaning, the expected pair win, gold entity information, training sizes, and the zero-shot prompts' "states". | Required by the stage. CONTEXT.md §4: the review is not blind. |
| Appendices | New, all moved material: "Query degradation and encoder controls", "An operating curve, not a verdict" (now with the swaminathan2024selective detail), "The single-model CPU configuration in detail", "BioRED at the document level", "A bound on component-wise verification", "A gold label changed after freezing". Edited: the scaling appendix ("the pair question" instead of "the right question"; it now holds the teacher detail); the Artifacts appendix (Qwen3.8-27B as a silver voter only; sentence-level compute); references to the old §4.3 now point to Appendix `app:ablation`. | Space. Consistency. |

Not in the draft:
- the 22-item review's results;
- the 7 proposed flips;
- the what-if (§5 below).

The 19-item spot check is mentioned once in the Limitations, as an expert adjudication of a pre-filled sheet,
selected and not blind. The gold is unchanged.

## 3. The abstracts, side by side

| Submission (`paperA.tex`, 196 words) | Reframe (`paperA_reframe.tex`, 194 words) |
|---|---|
| Literature-mining pipelines that populate biotic-interaction databases propose candidate (taxon, relation, taxon) triples by co-occurrence, then filter them with a \emph{sentence-level} classifier: does this passage describe an interaction? Candidates are retrieved precisely because their passage names two taxa near an interaction term, so that question is answered before it is asked; what needs deciding is whether the passage asserts an interaction between \emph{these two}. In a controlled comparison --- one encoder, one 48k-row teacher-labelled corpus, three seeds, only the input differing --- telling the model which pair it is judging raises AUPRC on 437 expert-graded candidates from $0.851$ to $0.918$ and F1 from $0.775$ to $0.871$ (McNemar $p = 1.3\times10^{-7}$), at every threshold; adding the relation term buys nothing. Asked the same questions zero-shot, the 32B teacher shows the same gap at twice the size. On the public BioRED corpus, with candidates built the same way, conditioning on the pair raises AUPRC from $0.738$ to $0.843$ and entity-pair F1 by $9.6$ points, with no gain on sentences naming two entities and $+0.120$ on sentences naming more. The deployed verifier, 110M parameters, also labels direction in four categories and runs at 32 candidates per second on a CPU. Code: \url{https://anonymous.4open.science/r/XXXX}. | Literature-mining pipelines for biotic-interaction databases retrieve passages naming two taxa near an interaction term, and must decide whether a passage describes an interaction, what curators triage, and between which organisms, what the database records. In a controlled comparison (one encoder, one 48k-row corpus labelled by a 32B teacher, three seeds, only the input differing), telling the model which pair it judges raises AUPRC on 437 expert-graded candidates from $0.851$ to $0.918$ and F1 from $0.775$ to $0.871$ (McNemar $p = 1.3\times10^{-7}$), and on BioRED from $0.738$ to $0.843$, gaining only on sentences naming more than two entities. Read at the sentence level as a maximum over pairs, the pair verifier reaches BioRED AUPRC $0.965$, beating a sentence classifier trained on the same candidates ($0.930$) and at least matching one trained on BioRED's derived sentence labels ($0.955$; F1 $0.954$ against $0.930$). On our benchmark, under provisional LLM-derived labels, a classifier distilled from the teacher's sentence answers is better under all three label definitions (AUPRC $0.985$--$0.987$ against $0.966$--$0.977$). A perfect sentence filter has a pair precision of $0.536$ on BioRED (provisionally $0.724$ on ours). Ask each question for its own decision; one teacher can label both. Code: \url{https://anonymous.4open.science/r/XXXX}. |

## 4. Every new number, with its source

All new numbers come from files under `results/` (mostly `results/paperA_v2/sentence_level/*.json`). Each table
row below gives the number as printed, where it appears, its source file and its JSON key or derivation.

Two derived numbers:
- **34.5%:** 1 − `sentence_yes_rate` in `sentlab_label_stats.json`.
- **41%:** (12,503 + 1,422) / 34,242 from the same file's `crosstab`.
  - The draft words it as "the teacher's sentence answers and its candidate labels disagree".
  - The candidate labels come from the triple (labelling) prompt, not the pair prompt.

Literature values (0.97 AUC, 25% precision) come from LITERATURE.md.

### 4a. Numbers new in the first draft (from `writer_work/numbers.md`, 104 rows)

The table was audited on the 04:45 snapshot. Later edits changed some wording around these numbers, but not their values. Exceptions:
- the sentence-level table's SD bound is now "< .007";
- "within 0.006" is now "0.003 to 0.006";
- the McNemar p is now 0.0049;
- the BioRED stratum intervals are now the abstract-cluster ones.

These are re-traced in 4b.

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

### 4b. Numbers added by the revision (from `writer_work/final_check.md`)

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


## 5. The 22-item gold review: what-if (reported here only, not in the draft)

Source: `results/reframe_2026-10-05/gold_review_22_whatif.json`.
- The 7 proposed flips (items 2, 3, 4, 10, 12, 16, 18) are where your PAIR answers disagree with the current gold.
  All of them side with both models.
- The scores below are on 437 rows, three-checkpoint ensembles at the paper's block-held-out thresholds.

| arm | AUPRC, current gold | AUPRC, with the 7 flips | F1, current | F1, with flips |
|---|---|---|---|---|
| sentence | 0.855 | 0.871 | 0.775 | 0.790 |
| pair | 0.923 | 0.947 | 0.871 | 0.882 |
| triple | 0.909 | 0.948 | 0.852 | 0.867 |

What the flips would do:
- All three arms gain.
- The pair-vs-sentence gap stays about the same: AUPRC +0.067 now, +0.076 with the flips.
- The triple arm would move level with the pair arm in AUPRC (0.948 vs 0.947).

The review is an expert adjudication of a pre-filled sheet. It is not blind, and it is not a random sample (11 of
the 22 items were selected by model-vs-gold disagreement). The draft keeps the current gold.

## 6. Open decisions for you

1. **Adopt the reframe, or keep the submission version.**
   - The submission version's claims all remain true. The reframe adds the sentence-level evaluation, and its
     honest result is mixed: a win on BioRED, a provisional loss on ours to a fair sentence model.
   - My reading: the reframe is the more useful paper for curation and the more exposed one. A reviewer will note
     that the paper's own target, the biodiversity sentence decision, is evaluated on silver labels, and that the
     fair competitor wins there.
   - If you must submit before human sentence labels exist, the submission version is the safer text. The
     reframe can follow once the labels exist.
2. **Title.** The draft uses "Which Pair Is It About? Sentence and Pair Decisions for Literature-Mined Biotic
   Interactions". Alternatives:
   - "Two Questions, One Teacher: Sentence- and Pair-Level Verification of Literature-Mined Biotic Interactions"
   - "Does It Describe an Interaction, and Between Whom? Verifying Literature-Mined Biotic Interactions"
   - Or keep the submission title; the pair question is still the core.
3. **Wait for human sentence labels?** The biodiversity sentence-level comparison cannot be settled without them.
   - Where it is decided: on the passages whose candidate pair is negative in the gold, at most 191 (fewer if some
     are among the 20 SKIP rows). There the silver label is an LLM's answer to the sentence question, and the
     sentence-label model's whole advantage lies there (ANALYSIS.md §6.2).
   - How large the gap is there: 0.106 to 0.165 AUPRC on current silver labels, with 95% intervals 0.11 to 0.16
     wide (`biodiv_sentlab.json` → `by_silver.*.decomposition`).
   - **Minimum: those pair-negative passages of the 437-row sheet, at most 191.** With fewer, the intervals widen
     roughly as 1/√n: half of them gives intervals about 1.4 times as wide.
   - The 246 pair-positive passages are sentence-positive by definition whenever your PAIR answer agrees with the
     gold. Labelling them mostly checks agreement.
   - To compare the routes on the whole benchmark (overall AUPRC gaps of 0.010 to 0.020, intervals about 0.02 wide
     now), all 417 answerable rows are needed.
   - One command recomputes everything: `python3 scripts/sentence_level/make_sentence_tables.py` (ANALYSIS.md §1).
     It writes `results/paperA_v2/sentence_level/biodiv_human.json`.
   - UNSURE answers are dropped, as you decided.
   - The 0.724 ceiling will also be recomputed from your labels.
4. **Qwen3.8-27B as a silver voter.** It appears only as one of three voters (majority definition, Table
   `tab:sentence`, and the Artifacts appendix). It is not a comparison model.
   - If you prefer no newer Qwen anywhere in the paper, keep only the Qwen3.5-122B-alone column. The conclusion
     does not change: the sentence-label model wins under every definition.
   - If you keep it, add its licence to the Artifacts appendix. I did not verify it.
5. **The 7 proposed gold flips** (§5). They are your decision. The draft keeps the current gold.
6. **Whether to run the two untested controls named in the Limitations.** Both would pre-empt reviewer questions:
   - (a) A sentence model given all entity mentions, trained on BioRED's sentence labels. It tests whether the
     BioRED edge is entity information rather than the pair question.
   - (b) A combination of the sentence and pair routes, chosen on development data.
7. **Wording inherited from the submission that a checker flagged for anonymity** (`claims_review.md` M7/S12).
   These sentences are unchanged in both versions, so the decision is yours:
   - "the pipeline we study";
   - "We pulled the retrieval pool in full: 175,588 candidate rows";
   - "We take the diagnosis as given and question only the remedy";
   - Reject50 as "candidates the retrieval pipeline discarded";
   - "The deployed model" in the escalation appendix.

   The draft's NEW text avoids "deployed" and "in the pipeline". The anonymous-repository URL is still the `XXXX`
   placeholder.

## 7. How the checkers' findings were handled

- **numbers.md** (first draft): 104 numbers traced, 0 untraceable, 10 flags. All 10 were applied:
  - the abstract's BioRED subset;
  - the McNemar p now follows F1;
  - the SD bound is now "< .007";
  - "within 0.006" became "0.003 to 0.006";
  - p = 0.0049, with the definition named;
  - the bootstrap scheme is named, with abstract-cluster stratum intervals;
  - the 41% wording;
  - "in part" instead of "mostly";
  - the Limitations' quantities: 148 under nonteacher2, "benchmark-grade", "the benchmark" rather than "all rows".
- **claims_review.md:** M1–M9 were applied as proposed or in equivalent wording; `final_check.md` gives the status
  of each.
  - Applied: S1–S6, S8 (in the text: the gold-entity caveat and "the 32B model"; in the Limitations: training sizes, out-of-distribution pairs, the zero-shot prompts' "states", the passage-level split), S9, S10, S11, S14, S15, and S16
    (as "as found for document-level filters of microbial interactions \citep{lim2016minter}").
  - Partly applied: S7. The table labels and caption were fixed, but the zero-shot rows stay in the same table as
    a labelled block.
  - Partly applied: S13. The voter is listed, but its licence is left for you.
  - Not applied: S12 (inherited wording; §6.7 above).
- **dropped_content.md:** the restorations were applied:
  - the canonical-string clause;
  - the teacher's relation-term null;
  - the co-occurrence mechanism sentence in the introduction;
  - the BioRED null half;
  - direction output and negative results in the contributions;
  - the direction coverage and its optimism caveat;
  - the "variant" clause;
  - the "what is ours" statement;
  - the conclusion's two loci and the relation-term null;
  - the taxonomy-gate pointer and the `tab:rules` reference (via the restored rules paragraph);
  - the swaminathan2024selective detail;
  - the EP-F1 citations in the body.

  Also fixed: the self-references and duplicated paragraphs in the appendices, and the surviving "false"
  wordings F1–F8. Dropped as low value: D16.
- Remaining issues found by `final_check.md`, if any, are listed in §8.

## 8. Known issues for the reviewer stage

- **A count in ANALYSIS.md is wrong.** §6 and §9 say the teacher's sentence and pair labels "differ on a third of
  the corpus". The crosstab gives 0.407, and the "pair" side is the passage-level OR of triple-prompt candidate
  labels. The draft says 41% with that wording.
- **ANALYSIS.md overstates the ceiling.** §9 says the precision ceiling "does not depend on silver labels". That
  is true for BioRED and for the per-pair ranking point. On ours the 0.724 depends on the silver sentence labels,
  and the draft calls it provisional.
- **The page budget is tight.** The Conclusion ends on the last line of page 8, so a body addition needs a cut
  or a move to the appendices.
- **The new appendices sit at the end, after "Negative results and cost".** Reorder them if you prefer.

- **Final check** (`writer_work/final_check.md`, run on the 05:01 snapshot):
  - M1–M6, M8 and M9 are resolved.
  - M7 was partly resolved at that point. The remaining inherited "deployed model" and "pipeline" wordings in the
    escalation, direction and negative-results appendices were then changed to "the single-model configuration",
    "this setting" and "the retrieval pool".
  - P1 applied: the spot check now gives all three agreements, the 3 dropped UNSURE answers and the 5 gold-fixed
    items.
  - P2 applied: "adds nothing to ranking"; the rules appendix no longer says the triple model leans on its
    relation term.
  - P3 applied: the self-reference and repetition in the scaling appendix were removed; "the 32B model".
  - P4 applied: the rules appendix opening now defers to its validation paragraph.
  - P7 applied: the abstract sentence was rewritten; "of which the teacher calls 65.5% positive"; "over the
    passage's pairs"; the duplicate gold-entity sentence was removed from the Limitations.
- **Not changed, inherited from the submission:**
  - The rules appendix's McNemar p = 0.067 (14 fixed / 5 broken) uses the continuity-corrected chi-square, as
    §2's protocol states for the pair-level results. An exact test gives 0.064.
  - The related-work phrase "a claim-only classifier matches evidence-aware models" (Schuster et al.) is
    unchanged from the submission and was not re-checked against the paper.
- **Abstract count:** 194 by the submission's convention (math = 1 word, code line included). A whitespace count
  that splits math into tokens gives 209, but the submission scores 196 by the first method and higher by the
  second, so compare like with like.

