# Claims review of `paperA_reframe.snapshot_0445.tex` (adversarial, 2026-10-05)

Reviewed against `ANALYSIS.md` §9 (authority), §4–§8; `LITERATURE.md` (NOVELTY STATEMENT, CLAIMS WE MUST NOT
MAKE, BiotXplorer section); `CONTEXT.md`; and a diff against `paperA/paperA.tex` (to separate what the draft
adds from what it inherits). Line numbers refer to the snapshot. Every number in §4.3 and in `tab:sentence`
was checked against ANALYSIS.md and matches it, with one exception noted at S3 (where 41% is correct but
mislabelled). All citation keys resolve in `references.bib` + `reframe_extra.bib`. The sentence prompt in
Appendix "Teacher prompts" is the one `llm_baseline.BIODIV["sentence"]` uses, so the reference from §4.3 to it
is correct.

---------------------------------------------------------------------------------------------------------------

## MUST-FIX

**M1. Abstract misattributes the BioRED number (rule 8).**
- Quote (l.32–33): "and on BioRED from $0.738$ to $0.843$, on sentences naming more than two entities."
- Why: 0.738 → 0.843 is the AUPRC over all 16,606 candidates. On sentences naming more than two concepts it is
  0.727 → 0.847, and on two-concept sentences there is no gain (0.909 vs 0.902; body l.362–366, `tab:biored`).
  As written, the abstract attaches the overall number to the subset.
- Replace with: "and on BioRED from $0.738$ to $0.843$, gaining only on sentences naming more than two entities."

**M2. The biodiversity precision ceiling 0.724 is presented as needing no silver labels (rules 3, 8).**
- Quotes:
  - l.466: "The converse holds on both benchmarks and needs no silver label."
  - l.79–80: "The converse is unconditional: a perfect sentence filter still passes non-interacting pairs,
    at a pair precision of $0.724$ on our benchmark and $0.536$ on BioRED."
  - Abstract l.39–40: "Conversely, a perfect sentence filter passes candidates at a pair precision of only
    $0.724$ (BioRED $0.536$)."
  - Conclusion l.617–618: "a perfect one passes candidates at a pair precision of $0.724$ here".
- Why:
  - Each benchmark passage has one candidate, so the ceiling is 246 / (246 + the pair-negative passages
    labelled sentence-positive). The silver labels on the 191 pair-negative passages fix it completely
    (ANALYSIS §5.2: 0.724 / 0.794 / 0.699 by definition; the 0.794 is biased upward).
  - The draft's own Limitations (l.643–644) says the ceiling "moves with the definition ($0.699$ to $0.794$)",
    so §4.3 contradicts the Limitations.
  - Only BioRED's 0.536 and the pair-gold AUPRCs (0.794 / 0.840 / 0.923) need no silver label. ANALYSIS §9's
    "does not depend on silver labels" is true of those, and of "the ceiling is below 1", but not of 0.724.
- Replace with:
  - l.466–468: "On BioRED the converse needs no silver label: a perfect sentence filter accepts every
    candidate of every positive sentence at a pair-level precision of $0.536$ $[0.527, 0.544]$ ($0.516$ in
    sentences with several candidates). On ours the same ceiling is provisional: $0.724$ $[0.674, 0.768]$
    under the majority labels, $0.699$ to $0.794$ across definitions. It falls below 1 as soon as any passage
    describes an interaction its candidate does not. Sentence scores also rank pairs poorly, and this needs no
    silver label: against our pair gold …"
  - l.79–80: "The converse holds on both benchmarks: a perfect sentence filter still passes non-interacting
    pairs, at a pair precision of $0.536$ on BioRED and, provisionally, $0.724$ on ours."
  - Abstract: "A perfect sentence filter has a pair precision of $0.536$ on BioRED (provisionally $0.724$ on
    ours)."
  - Conclusion: "a perfect one passes candidates at a pair precision of $0.536$ on BioRED and, provisionally,
    $0.724$ here."

**M3. The Limitations hedges away the sentence-label model's win and contradicts the abstract (rules 1, 8).**
- Quote (l.642–643): "Our benchmark therefore cannot yet say which sentence filter is better".
- Why:
  - Rule 1 says the result that sentlab beats pair-max under all three silver definitions must be stated
    plainly, not hedged away. ANALYSIS §9: "Against the fair competitor (A1, sentlab), the claim fails".
  - The abstract (l.38) and the introduction (l.79) say the sentence-label model "is better"; this sentence
    says the benchmark cannot tell.
  - The circularity caveat (§8.2) is legitimate. It must read as a caveat on a result, not as erasing it.
  - The hedge is also one-sided. The only human evidence, the spot check, favours the silver labels
    (κ 0.81 with the non-teacher labels, n = 14), and the draft omits it (see S6).
- Replace with: "On silver labels the sentence-label model is the better sentence filter under every
  definition. Its whole advantage lies where those labels are an LLM's answer to the prompt it imitates, so
  only human sentence labels can confirm it."

**M4. "Each decision needs its own question" claims more than §9 supports (rules 1, 8).**
- Quotes:
  - Abstract l.40: "Each decision needs its own question; one teacher can label both."
  - Intro l.80–81: "Each decision therefore needs its own question, and the teacher that labels one can label
    the other."
- Why:
  - The claim is necessity for the sentence decision. On BioRED pair-max is at least as good as the
    sentence-label model (§9). On ours the opposite result is provisional and partly circular (Limitations
    l.639–643).
  - §9 supports only the recommendation "ask the pair question for the pairs and the sentence question for the
    sentence", together with "Pair verification is a competitive but not the best sentence filter".
  - The conclusion (l.619) already uses the supported wording.
- Replace with:
  - Abstract: "Ask each question for its own decision; one teacher can label both."
  - Intro: "The pair decision therefore needs the pair question. For the sentence decision, pair verification
    is competitive but, on our provisional labels, not the best, so we ask each question for its own decision;
    the teacher that labels one can label the other."

**M5. §4.3 presents the max as exact, which the draft's own biodiversity result refutes (rule 9; LITERATURE
item 6).**
- Quote (l.414–418): "A passage describes an interaction if at least one of its pairs interacts, … so the same
  verifier also answers the curator's: … The maximum is not an approximation here but the meaning of ``any
  interaction''."
- Why:
  - The max is exact only if the scored pairs cover every interaction in the passage.
  - On ours they do not. The sentence-label model's whole advantage is on passages whose interaction lies
    outside the candidate, and pair-max over TaxoNERD pairs only partly recovers those (0.788 vs 0.953).
    TaxoNERD even finds both candidate strings in only 35% of passages (ANALYSIS §5.3, §6.2).
  - LITERATURE item 6 allows justifying the max by the semantics of "any interaction", not presenting it as
    obviously right (jia2019document lost 3.8 AUC with max).
  - "so the same verifier also answers the curator's" is then contradicted two paragraphs later.
- Replace with: "If every interacting pair of a passage is among the pairs scored, the passage describes an
  interaction exactly when one of them interacts: the multi-instance assumption
  \citep{dietterich1997solving,hoffmann2011knowledge}. The maximum of the verifier's scores over those pairs
  (\emph{pair-max}) is then the natural sentence score. The two benchmarks measure how far the scored pairs
  fall short of covering a passage's interactions."

**M6. The gold review is not described as rule 6 requires.**
- Quote (l.630–632): "Candidates on which two independent models confidently contradict the gold are under
  expert adjudication, mixed with an equal number of random controls".
- Why:
  - Rule 6: if mentioned, the review is "an expert adjudication of a pre-filled sheet".
  - "Are under expert adjudication" suggests an independent, ongoing re-annotation. In fact the 22 items were
    adjudicated on a model-pre-filled sheet and are not blind (ANALYSIS §2). The submission's "blind … with
    labels and scores hidden" was correctly removed, but the replacement is still incomplete.
  - No gold label is changed: this passes.
- Replace with: "Candidates on which two independent models confidently contradict the gold, mixed with an
  equal number of random controls, have gone through an expert adjudication of a pre-filled sheet. It is not
  blind, and no label in this paper is changed on its strength or on a model's disagreement."

**M7. Deployment wording implies the verifier runs inside the application (rule 5).**
- Quotes:
  - l.519: "\section{Candidate rules and deployment}"
  - l.527–528: "In the pipeline, verification runs as one 110M-parameter model with a fixed threshold of $0.5$"
  - Appendix l.1287: "The deployed configuration in detail"
  - l.1290 and l.1320: "In the pipeline, verification runs as …"
  - l.1330–1331: "faster than the pipeline retrieves them"
  - l.1009, l.1016: "The deployed model"
- Why:
  - §2 defines "the pipeline" as BiotXplorer's retrieval ("in the pipeline we study"). These sentences
    therefore say the authors' model is deployed in BiotXplorer.
  - That is the "deployed in" / working-relationship wording LITERATURE forbids, and a route to de-anonymising
    the authors.
  - The §6 title and paragraph are new text in the draft. The appendix wording is inherited from the submission,
    which has the same problem (the user's call there).
- Replace with:
  - Retitle §6 "Candidate rules and a CPU configuration".
  - "In our recommended configuration, verification runs as one 110M-parameter model …"
  - Appendix title: "The CPU configuration in detail".
  - "the single-model configuration" for "the deployed model".
  - Delete "faster than the pipeline retrieves them".

**M8. BioRED's sentence labels go unqualified in the abstract (rule 2).**
- Quote (l.35–36): "and is at least as good as one trained on sentence labels ($0.955$; F1 $0.954$ against
  $0.930$)".
- Why:
  - The abstract has just defined the sentence decision as "whether a passage describes a biotic interaction".
    Read that way, BioRED's labels would mean "states a relation", which is what rule 2 forbids. They mean
    "co-mentions a related pair".
  - Because they are the OR of the pair labels, the pair route is the expected winner (LITERATURE item 4;
    the draft's own l.683–685).
  - The parenthesis is also hard to read: 0.955 is the competitor's AUPRC, compared with an unstated 0.965.
  - Same gap in the conclusion, l.616: "one trained on BioRED's sentence labels".
- Replace with:
  - Abstract: "and is at least as good as one trained on BioRED's derived sentence labels ($0.955$; F1 $0.930$
    against $0.954$)".
  - Conclusion: "one trained on BioRED's derived sentence labels (a related pair co-mentioned)".

**M9. The abstract states a provisional, LLM-judged mechanism as fact (rules 3, 8).**
- Quote (l.38): "through interactions outside the candidate pairs."
- Why: ANALYSIS §6.2 says these are "passages that the LLMs judge to describe an interaction although the
  candidate pair does not interact", and "Whether sentlab's extra sentence-positives are real is a question for
  human labels". The abstract asserts that the interactions exist.
- Replace with: delete the clause, and state the result under all three definitions: "is better under all
  three label definitions (AUPRC $0.985$ against $0.966$)". If space allows: "on passages the LLMs judge to
  describe an interaction outside the candidate pair".

**Proposed abstract incorporating M1, M2, M4, M8 and M9: 199 words, Code line included.**

```latex
Literature-mining pipelines for biotic-interaction databases retrieve passages that name two taxa
near an interaction term, and must decide two things: whether a passage describes a biotic
interaction, what curators triage, and between which organisms, what the database records. In a
controlled comparison (one encoder, one 48k-row corpus labelled by a 32B teacher, three seeds, only
the input differing), telling the model which pair it judges raises AUPRC on 437 expert-graded
candidates from $0.851$ to $0.918$ (McNemar $p = 1.3\times10^{-7}$), and on BioRED from $0.738$ to
$0.843$, gaining only on sentences naming more than two entities. Read at the sentence level as a
maximum over candidate pairs, the pair verifier beats a sentence classifier trained on the same
candidates (BioRED AUPRC $0.965$ against $0.930$) and is at least as good as one trained on BioRED's
derived sentence labels ($0.955$; F1 $0.930$ against $0.954$). On our benchmark, under provisional
LLM-derived sentence labels, a classifier distilled from the teacher's sentence answers is better
under all three label definitions (AUPRC $0.985$ against $0.966$). A perfect sentence filter has a
pair precision of $0.536$ on BioRED (provisionally $0.724$ on ours). Ask each question for its own
decision; one teacher can label both.
Code: \url{https://anonymous.4open.science/r/XXXX}.
```

---------------------------------------------------------------------------------------------------------------

## SHOULD-FIX

**S1. The BioRED paragraph of §4.3 overstates against the sentence-label model (rule 1).**
- Quote (l.436): "Pair-max is the best sentence filter". Write instead: "Pair-max scores highest of the three".
- l.440: say which interval is quoted and that it includes zero: "$+0.010$, whose interval includes zero when
  abstracts are resampled as clusters ($[-0.001, +0.021]$; $[+0.000, +0.020]$ over sentences)".
- Optional: the difference-in-differences, $-0.038$ $[-0.069, -0.008]$, makes "not from pairs competing"
  airtight.

**S2. Say in §4.3 that a pair win is expected on BioRED by construction.** At present the point appears only in
the Limitations (l.683–685). Add to the BioRED paragraph: "Since this label is the maximum of the pair labels,
a pair-route win is expected here \citep{li2021dual}." A reviewer will otherwise discount the BioRED
"at least as good" against the paper.

**S3. The "41%" sentence is mislabelled (l.472–473).**
- Quote: "the teacher's two answers differ on $41\%$ of corpus passages (Cohen's $\kappa = 0.26$)".
- The "pair" side is not a pair answer. It is "any of the passage's corpus candidates accepted under the
  *triple* labelling prompt", over 1.41 candidates per passage.
- 12,503 of the 13,925 disagreements are sentence-YES with no accepted candidate, which is expected whenever
  other organisms interact. The remaining 1,422 (4%) are inconsistent answers.
- Write instead: "the teacher's sentence answer and its candidate labels (any candidate accepted) disagree on
  $41\%$ of corpus passages (Cohen's $\kappa = 0.26$), mostly passages it says describe an interaction without
  accepting any of their candidates; on $4\%$ it is the reverse."
- Related: the Setting (l.125–126) cites "$34.5\%$" with a pointer to §4.3, which never states that figure.
  Add "(it answers yes on $65.5\%$)" in §4.3.

**S4. The conclusion invites the conflation rule 1 forbids.**
- l.614: "most where passages name several candidate pairs". The pair-level strata are by taxa or concepts
  named, not candidate pairs. At the sentence level, BioRED's edge over the sentence-label model lies in
  single-pair sentences. Write instead: "most where passages name more than two taxa or entities".
- l.615–616: "the pair verifier beats a sentence classifier trained on the same candidates". On ours this holds
  only under two of three definitions (majority: $+0.008$ $[-0.005, +0.021]$). Add: "on BioRED, and on ours
  under two of three label definitions".

**S5. The headline biodiversity numbers come from the most circular definition.**
- The abstract and F1 use the majority vote, which includes the teacher, the very model the sentence-label
  model imitates. It also gives the largest gap ($-0.020$).
- A hostile reader will ask why. Either state the range (AUPRC $0.985$–$0.987$ against $0.966$–$0.977$) or lead
  with the non-teacher labels ($0.987$ against $0.977$).

**S6. Add the spot check, so that the circularity caveat is not one-sided (rules 1, 6).**
- One sentence in §4.3 or the Limitations: "On 19 items of an expert adjudication of a pre-filled sheet
  (selected, not random, not blind), the expert's sentence answers agree with the non-teacher labels on 13 of
  14 ($\kappa = 0.81$): too few to decide, but in the silver labels' favour."
- Never call it blind and never use it as a benchmark result.

**S7. Fixes to the `tab:sentence` table.**
- Label the biodiversity columns as AUPRC: "AUPRC: maj. / non-t. / 122B".
- Caption: the sentence-label model is trained on BioRED's derived labels there and on the teacher's sentence
  answers on ours.
- Caption: pair-max on ours maxes over the candidate plus TaxoNERD pairs.
- Name the order of "$0.941$, $0.958$ and $0.973$": candidate-trained arm, sentence-label model, pair-max.
- Note that the non-t. ceiling of $.794$ is biased upward: its 43 dropped passages are all pair-negative.
- Note that F1 at these base rates is compressed toward accept-all, and that on ours the candidate-trained arm
  falls below accept-all ($.849 < .875$).
- The zero-shot rows sit on a 500-sentence subset (base rate $0.854$) but share columns with full-test numbers.
  Consider a separate block.

**S8. Confounds a hostile reviewer will raise; state them in §4.3 or the Limitations.**
- (a) BioRED's sentence-label model trains on 3,277 sentences, against 15,640 candidates for the pair arm
  (ANALYSIS §4.3). This plausibly explains its weakness on single-pair sentences.
- (b) On ours, pair-max applies a verifier trained on retrieved candidates to TaxoNERD-enumerated pairs, which
  lie outside its training distribution.
- (c) Both zero-shot BioRED prompts ask whether the sentence "states a relation", but the label only requires a
  co-mention (up to 47% of positives may state nothing, ANALYSIS §4.5). Pair-max also gets one YES chance per
  candidate. Say this next to the $+0.051$.
- (d) The biodiversity sentence-label model was split by passage, not grouped by taxon pair as the recipe in
  §3 states.
- (e) On BioRED "the teacher" (l.446–447) is just the 32B model, since there is no teacher there. Write
  "the 32B model".

**S9. Two small precision issues in §4.3.**
- l.450, "Pair-max ranks above … pair-own": add "never significantly".
- l.470, "the per-pair scores $0.923$": add "(three-checkpoint ensemble; Table 1's $0.918$ is the per-seed
  mean)". Otherwise it looks like an inconsistency.

**S10. Intro wording that the draft's own results contradict.**
- l.57–58: "Every candidate arrives already satisfying, lexically, what a sentence classifier learns to
  detect". A sentence classifier trained on sentence labels reaches 0.985 AUPRC (provisional) here, so it
  learns more than lexical cues. Scope it: "what a sentence classifier trained on candidate labels can lean
  on".
- l.71–72: "A passage describes an interaction if at least one of its pairs interacts" is uncited and
  unconditional in the introduction. Cite the multi-instance assumption there and make it conditional, as in
  M5.

**S11. Related work, l.581–583: "change only the question".**
- On ours the arms also differ in their training rows (34,242 passages against 48,338 candidates), in the
  teacher prompt (sentence against triple) and in pair-max's taxon recogniser.
- Write instead: "change the question, and with it the training rows".

**S12. Anonymity issues inherited from the submission (the user's call; rule 5).**
- l.119: "We pulled the retrieval pool in full: $175{,}588$ candidate rows …". Together with \textsc{Reject50}
  ("candidates the retrieval pipeline discarded"), this exposes the system's internals and usage beyond its
  BISS abstract.
- l.113: "We take the diagnosis as given and question only the remedy" argues with the system authors' work,
  which LITERATURE asks the paper not to do.
- l.43: the `XXXX` placeholder must be filled before submission.

**S13. List Qwen3.8-27B in the Artifacts appendix (rule 7).** The appendix (l.1151–1156) lists only Qwen3 and
Qwen3.5-122B. Add Qwen3.8-27B as "used only as one of the silver-label voters", with its licence (ACL
checklist), and keep it out of every comparison. It currently appears only in the `tab:sentence` caption, as a
voter, which passes rule 7.

**S14. Leftover editing artifacts.**
- Appendix `app:rules` (l.1062–1073) duplicates the old §6 paragraph and cites itself
  ("Appendix~\ref{app:rules}" inside `app:rules`).
- `app:deploy` repeats l.1290–1299 at l.1320–1334 and cites itself (l.1293).
- The appendix "The query, not the encoder" repeats the title of §4.4.

**S15. Cite public practice for the curator motivation.** At l.52, "That is the decision a curator triages",
cite public curation practice (krallinger2008overview, islamajdogan2019overview). The LITERATURE BiotXplorer
section asks that the sentence target be motivated by the task and public practice, not by the application's
owners.

**S16. Present the precision ceiling as a known limit (LITERATURE item 3).** Contribution 2 lists "the pair
precision of a perfect sentence filter". That limit is known (pyysalo2008comparative, lim2016minter,
polajnar2011protein). In §4.3, add "as known for protein interactions \citep{pyysalo2008comparative}" so that
it reads as a measurement on these benchmarks, not a finding.

---------------------------------------------------------------------------------------------------------------

## The 20 "claims we must not make": pass/fail

| # | Claim | Verdict | Evidence in the draft |
|---|---|---|---|
| 1 | "We introduce max-over-pairs" / first to derive a sentence decision from pairs | PASS | §4.3 cites the multi-instance assumption (l.415–416). Related work: "We claim none of these" (l.581). Contributions do not claim the max. (Intro l.71–73 should cite it too; see S10.) |
| 2 | Pair-conditioned input is our contribution | PASS | "We claim neither that diagnosis nor the pair-conditioned input" (l.555–556); cites zhang2017position, lee2020biobert, soares2019matching, zhong2021frustratingly. |
| 3 | First to show a sentence or document filter has limited pair precision | PASS (weak) | No "first" claim, and related work cites pyysalo, tikk, lim2016minter and polajnar. But contribution 2 lists the ceiling as a contribution; see S16. |
| 4 | Pair route winning is surprising / known to win | PASS | Limitations l.683–685: on BioRED it is "the expected winner … by construction \citep{li2021dual}". No "surprising" anywhere. Move the point into §4.3 (S2). |
| 5 | Asking an LLM one pair at a time is known to be better | PASS | l.563–564: "not known to beat listing all pairs at once \citep{mraz2026fewshot}". Labonte et al. is not cited. |
| 6 | Max is obviously the right aggregator | PASS (borderline) | The max is justified by the semantics of "any interaction", which item 6 allows. But "not an approximation here" overreaches and is refuted by the draft's own biodiversity result (M5). No soft aggregator is reported. |
| 7 | The gain on passages naming >2 entities is a new phenomenon | PASS | No novelty claim; jiang2019challenge, pyysalo and tikk are cited. |
| 8 | "Distant supervision is ~30% noisy" | PASS | No number is given (l.548–549). |
| 9 | No biotic-interaction corpus has pair labels | PASS | Not claimed; cuzick2023interspecies is cited for the pair as the annotation unit. |
| 10 | Ecological LLM pipelines skip the sentence question | PASS | l.603: "\citet{zou2026llm} first ask whether a text contains an interaction". |
| 11 | Humans agree more on pair than sentence labels (or the reverse) | PASS | Not made. |
| 12 | Dropping UNSURE is harmless | PASS | UNSURE is not mentioned. If human labels are added later, say that UNSURE rows are dropped and cite plank2022problem. |
| 13 | keck2025extracting reports 89.5% accuracy | PASS | Cited without a figure. |
| 14 | Mixing PPI co-occurrence precisions across papers | PASS | Only pyysalo2008comparative and tikk2013detailed, with no Airola figure. |
| 15 | A published statistic gives the share of negative pairs in positive sentences | PASS | Not made. |
| 16 | "Finer-grained supervision helps the coarse decision" as our finding | PASS | rei2019jointly and dasanmartino2019fine are cited with "We claim none of these"; the measured result is scoped and reports where it fails. |
| 17 | Nobody compared "fine detector + at least one" with a direct sentence classifier in biomedical text | PASS | farkas2010conll is cited (l.577–579). |
| 18 | Fine-grained models read at the coarse level always win | PASS | magge2021deepademiner is cited, and the biodiversity loss is reported. |
| 19 | Only a pair model can say which pair interacts | PASS | l.470–472 uses the required wording, with pavlopoulos2022from. (The conclusion's "No sentence filter can stand in for the pair" concerns a passage-level filter, which is acceptable.) |
| 20 | Asking finer questions is better, unconditionally | PASS | The draft reports where the sentence question wins. The remaining overreach is "Each decision needs its own question" (M4), which concerns necessity, not "finer is better". |

The literature conditions in NOVELTY STATEMENT are respected: "same teacher" is stated only for ours
(l.582); "better on >2 entities" is not claimed against the sentence-label model; and TaxoNERD enumeration and
its cost are reported. The multi-instance "at least one" rule is acknowledged as prior work (rule 4, PASS).

## Rules 1–9 in brief

1. Scope to §9: the sentlab win is stated plainly in §4.3 and the abstract (PASS), but the Limitations hedges
   it away (M3). The BioRED operating-point and marginal threshold-free result: PASS (clarify the clustered
   interval, S1). Edge in single-pair sentences, not where pairs compete: PASS in §4.3. The conclusion's
   wording risks the conflation (S4).
2. BioRED label semantics: PASS in §4.3, the table caption, the intro and the Limitations. Missing in the
   abstract and conclusion (M8).
3. Provisional labels: PASS in the table header, §4.3 and the Limitations, and circularity is acknowledged
   (l.458–459, l.639–642). FAIL for the 0.724 in the abstract, intro, §4.3 and conclusion (M2), and for the
   abstract's mechanism (M9).
4. Prior work acknowledged, all 20 items pass: see the table above.
5. Anonymity: the new BiotXplorer sentences (l.104–107) stay within the public abstract (PASS). The rewritten
   deployment section implies a working relationship (M7). Inherited items are in S12.
6. Gold review: never called blind (PASS), and no gold label is changed (PASS). It is not yet described as
   "an expert adjudication of a pre-filled sheet" (M6).
7. Models: Qwen3.8-27B appears only as a silver voter (PASS). Add it to the Artifacts appendix (S13).
8. Consistency: M1, M2, M3, M4, M8 and M9.
9. Hostile-reviewer points on §4.3 and its table: M5, S1–S3 and S5–S9.

## Abstract word count

- Draft (l.28–43): 197 whitespace-delimited words, counting the line "Code: \url{…}" and the "=" in
  "$p = 1.3\times10^{-7}$". That is 196 without the "=", and 195 without the Code line.
- It is within the 200-word limit, with 3 words of headroom.
- For calibration, the same method gives the submission abstract 198 words with the Code line and 196 without.
  The 196 in CONTEXT.md excludes the Code line.
- The proposed abstract above counts 199 with the Code line, by the same method.
