# What the reframe dropped from the source

Source: `paperA/paperA.tex` (S = source line numbers).
Draft: `results/reframe_2026-10-05/writer_work/paperA_reframe.snapshot_0445.tex` (D = draft line numbers).
Evidence for the reframe: `results/reframe_2026-10-05/ANALYSIS.md` (34.5% = 11,830 / 34,242 sentence-NO
answers from the teacher, §6 table; this checks out).

## Summary counts (KEPT / MOVED / REWORDED)

- **KEPT, same place, verbatim or nearly:**
  - Setting: 5 of 6 paragraphs.
  - §3 (teacher, corpus, three models): all of it.
  - Tables 1–3.
  - §4.1: the main-result paragraph, the relation-term paragraph and the Table 2 paragraph.
  - §4.2: the first two paragraphs.
  - §5: error mechanisms and the component-wise bound.
  - Limitations: 7 of 8 paragraphs (the 8th is changed as required, see C16).
  - Appendices: all 12 source appendices, verbatim (seeds, soft labels, encoders, recall, prompts,
    scaling, cascade, examples, rules detail, where-it-stops, artifacts, negative results).
- **MOVED to appendices, each leaving a 1–3 sentence summary in the body:** 5 blocks.
  - §4.3 query ablation: Table `tab:ablation` plus 4 paragraphs, now in App "The query, not the
    encoder" (`app:ablation`).
  - §4.4 operating curve: `fig:threshold`, `tab:curve` and 1 paragraph, now in `app:curve`.
  - §6 rules paragraph, now in `app:rules`.
  - §7 deployment: `tab:direction`, 2 paragraphs and a footnote, now in `app:deploy`.
  - §4.2 BioRED entity-pair F1 paragraph, now in `app:bioredep`.
- **REWORDED, meaning kept:**
  - the abstract's main result;
  - the intro's first and third paragraphs;
  - contributions 1 and 2;
  - the teacher paragraph;
  - query representation;
  - all four related-work paragraphs (condensed);
  - the conclusion's numbers.
- **Citation keys:**
  - All 58 source keys are still in the draft.
  - The draft adds 21 keys, and all of them resolve in `paperA/reframe_extra.bib`.

---

## DROPPED

Items absent from the draft, or removed from the body while the body still makes the claim they
qualify.

**D1. Query representation (S146–147).**
- Quote: "and the canonical string is absent from the passage on 31, 27 and 54."
- Why it matters: this is the actual reason the model must be queried with the surface form, because
  the canonical string often does not occur in the passage at all. Without it, the choice looks
  arbitrary.
- Suggestion: **restore in body** (one clause).

**D2. Teacher paragraph (S291–292).**
- Quote: "As with the student, the relation term adds nothing ($0.946$ against $0.943$)."
- Why it matters: this is the only sentence saying the triple question does not help the teacher.
  - The numbers remain in Table 1, but the claim is now unstated.
  - App. scaling shows triple > pair for Qwen3 1.7B–14B (e.g. 4B: 0.903 vs 0.864), so the null
    really is specific to the 32B teacher and should be stated as such.
- Suggestion: **restore in body**, scoped to the 32B teacher.

**D3. Teacher paragraph (S290–291).**
- Quote: "A model three hundred times smaller that is asked the right question beats the teacher asked
  the wrong one by $0.13$ AUPRC."
- Why it matters: the result is real (pair student 0.918 vs teacher sentence question 0.789, on pair
  gold), but "right/wrong question" contradicts the reframe.
- Suggestion: **fine to drop**. If restored, reword: "...asked about the pair, beats the teacher asked
  the sentence question by 0.13 AUPRC on pair gold."

**D4. Intro (S56–58).**
- Quote: "Candidates are proposed by co-occurrence: a passage naming two taxa near an interaction term
  yields a candidate whether or not it asserts that interaction between those two."
- Why it matters: it states the mechanism behind the pair argument, and it stays true under the
  reframe. The draft keeps the "three organisms" sentence but not this one.
- Suggestion: **restore in body** (intro).

**D5. Intro (S72–74), the null half of the BioRED prediction.**
- Quote: "absent on sentences that name only two entities, large on sentences that name more."
- Why it matters: the predicted null is what makes the localisation a test rather than a post-hoc
  story. The draft intro (D68) keeps only the positive half, and the draft abstract garbles it (C3).
  The body (D363–366) still has it.
- Suggestion: **restore in body** (intro, a few words).

**D6. Contributions (S82–90).** Several elements are gone:
- "disambiguation between the pairs a passage names, and a check on candidates whose arguments are
  not organisms";
- "validated on 42k training rows before evaluation, measured as a small precision gain rather than a
  large one";
- "a fixed threshold, four-way direction output";
- "six negative results (… Appendix~\ref{sec:negative})".

Why it matters:
- The honest "small gain" framing of the rules is lost.
- The direction output appears nowhere in the abstract, contributions or conclusion.
- Appendix `sec:negative` is now reachable only from other appendices.

Suggestion: **restore in body**, at least "four-way direction output" and "six negative results
(Appendix~\ref{sec:negative})".

**D7. Abstract (S35–41).** Missing phrases:
- "and F1 from $0.775$ to $0.871$"
- "at every threshold; adding the relation term buys nothing"
- "Asked the same questions zero-shot, the 32B teacher shows the same gap at twice the size"
- "and entity-pair F1 by $9.6$ points"
- "The deployed verifier, 110M parameters, also labels direction in four categories and runs at 32
  candidates per second on a CPU."

Why it matters: the body keeps all of these. But together with D11, the null result for the relation
term is now in neither the abstract nor the conclusion, and the deployment facts are in neither.

Suggestion: **fine to drop for space**, but restore "the relation term adds nothing" in the abstract
or the conclusion.

**D8. Deployment, body summary (D528–531 vs S551–553 and the caption at S523–524).**
- Quotes:
  - "the model answers 74\% of the directional ones, correctly on $0.887$ of those"
  - "The abstention level ($0.60$) was chosen by the expert after seeing this set's coverage curve,
    so the figures are optimistic."
- Why it matters: the body now reports "correct on $0.887$ of the directional candidates it answers"
  with no coverage and no optimism caveat. Both are only in the appendix.
- Suggestion: **restore in body** ("on the 74% it answers; abstention level chosen on this set").

**D9. Deployment, body summary (S528–533).**
- Quote: "It differs from the pair-conditioned model of \S\ref{sec:verifier} in three respects …
  alphabetical order … marked in place … pair-level judgments."
- Why it matters: the body now gives AUPRC 0.934 and F1 0.900 for "one 110M-parameter model" right
  after the 0.918 / 0.871 pair model. A reader will take them to be the same model and see a
  contradiction.
- Suggestion: **restore in body**, one clause ("a variant of the pair-conditioned model with
  in-passage markers, alphabetical order and pair-level labels").

**D10. Related work (S580–582).**
- Quote: "What is ours is a controlled measurement of what it costs in a verification setting built by
  co-occurrence retrieval, located by block; a bound on component-wise remedies; and the candidate
  rules."
- Why it matters: this was the only positive novelty statement in related work. The draft now
  disclaims both the diagnosis and the pair-conditioned input (C10), so nothing says what is new.
- Suggestion: **restore in body**, updated to add the sentence-level comparison and the
  perfect-sentence-filter precision bound.

**D11. Conclusion (S621–624).** Missing phrases:
- "and the relation term adds nothing on top"
- "and, where candidates come from automatic matching, on candidates whose arguments are not
  organisms"
- "The result is a calibrated score where a verdict stood, with a four-way direction label beside it,
  cheap enough to run on every candidate."

Why it matters:
- These are the second locus of the gain (half of "what the pair query buys").
- They state the null for the relation term.
- They carry the practical deliverable.
- None of them is made false by the reframe.

Suggestion: **restore in body** (conclusion; about two short sentences).

**D12. Rules (S504–505).**
- Quote: "A taxonomy gate is no substitute: surnames, clades and transposon families all resolve in a
  taxonomy (Appendix~\ref{sec:notfixed})."
- Why it matters: it was the only pointer to the entity-gate analysis, which is now unreachable from
  the body or from `app:rules`.
- Suggestion: **restore in appendix** (`app:rules`), with the reference.

**D13. Related work (S598–600).**
- Quote: "which is how extraction is made useful to human abstractors: \citet{swaminathan2024selective}
  abstain under calibrated thresholds to automate $57$--$95\%$ of clinical chart abstraction at
  controlled error."
- Why it matters: it was the concrete precedent for the calibrated-score argument. The key survives
  only as a generic "established practice" citation (C11).
- Suggestion: **restore in appendix** (`app:curve`, where the operating-curve argument now lives), or
  fine to drop.

**D14. BioRED entity-pair F1, body (D372–374 vs S366–372).**
- Quote: "(PubMedBERT, BioREx and BioREDirect; \citealp{lai2023biorex,lai2025bioredirect}) and
  zero-shot GPT-4 $42.9$ \citep{islamaj2024overview}"
- Why it matters: the body still states "$74.1$ to $75.3$" but now without any citation; the citation
  survives only in `app:bioredep`.
- Suggestion: **restore in body** (the citation only).

**D15. Rules (S497–498).**
- Quote: "The result (Table~\ref{tab:rules}) is smaller than the motivation suggested, and the reason
  is the headline of this paper."
- Why it matters: "headline" is no longer accurate, but this was the only `\ref{tab:rules}`, which is
  now unreferenced.
- Suggestion: **fine to drop**, but re-add a `Table~\ref{tab:rules}` reference in `app:rules`.

**D16. Low-value drops, all fine to drop:**
- "We make this argument testable rather than rhetorical." (S65)
- "Everything except the input is held fixed, so differences between them are attributable to the
  formulation and to nothing else." (S68–70; §3 keeps "They differ only in their input")
- "This is the application we build for" (S100)
- "The formulation effect is not a property of small encoders." (S286)
- "where the student's is $+0.084$ against $+0.042$" (S290; the numbers are in the next paragraph)
- "That a classifier can score well by exploiting how its data were assembled is among the
  best-established results in NLP." (S559–560)
- "such rules hold in distribution while" (S563)
- "rather than the sentence" (S577, TACRED)
- "standard and" (S585)
- "and the question is whether that passage, and no other, supports it" (S595–596)
- "independently …, the unit our pair-binding class isolates" (S606–608, cuzick)
- "since half the rules were chosen with this benchmark's errors in view" (S502–503; covered in
  `app:rules` and the Limitations)

---

## CHANGED

**C1. Title (S19–20 → D19–20).**
- Source: "Pair-Conditioned Verification of Literature-Mined Biotic Interactions"
- Draft: "Sentence and Pair Decisions for Literature-Mined Biotic Interactions"
- Intended.

**C2. Abstract framing (S28–32 → D28–30).**
- Source: "Candidates are retrieved precisely because their passage names two taxa near an
  interaction term, so that question is answered before it is asked; what needs deciding is whether
  the passage asserts an interaction between \emph{these two}."
- Draft: "…and must decide two things: whether a passage describes a biotic interaction, what curators
  triage, and between which organisms, what the database records."
- This is the required change and was made correctly.

**C3. Abstract, BioRED clause (S38–40 → D32–33). This is an error.**
- Source: "conditioning on the pair raises AUPRC from $0.738$ to $0.843$ and entity-pair F1 by $9.6$
  points, with no gain on sentences naming two entities and $+0.120$ on sentences naming more."
- Draft: "and on BioRED from $0.738$ to $0.843$, on sentences naming more than two entities."
- Problem: as written, 0.738 → 0.843 reads as the figure on sentences naming more than two entities.
  - That figure is over all candidates. On ≥3-concept sentences it is 0.727 → 0.847 (+0.120,
    Table `tab:biored`).
  - The null on two-concept sentences is lost.
- Fix: "and on BioRED from 0.738 to 0.843, with no gain on sentences naming two entities."

**C4. Intro, second paragraph (S56–63 → D56–61).**
- Source: "That question is answered before it is asked. … What actually needs deciding is narrower:
  whether the passage asserts an interaction between \emph{these two} taxa."
- Draft: "The usual filter asks only the first question and takes the pair from co-occurrence.
  Retrieval makes that fragile in two ways."
- The required change was made. Residue: see F2.

**C5. Setting heading and paragraph (S112–119 → D118–127).**
- Source heading: "Why the sentence-level question is constant here."
- Draft heading: "What retrieval decides, and what it leaves."
- The draft adds: "Whether a passage describes an interaction is not constant: asked that question,
  the teacher … answers no on $34.5\%$ of the corpus passages."
- Intended and correct, but the "being asked about a constant" sentence survives verbatim (F1).

**C6. Setting, last sentence (S118–119 → D126–127).**
- Source: "The missing input is not an annotator artefact but the query the pipeline itself issued."
- Draft: "What is missing from the sentence-level input is the query the pipeline itself issued: which
  pair."
- The contrast with annotator artefacts is gone (see C7 and F7).

**C7. Related work, citation role of wiegand / parmar (S566–569 → D541–543).**
- Source: "the missing input is the query the pipeline itself issued rather than an annotator artefact
  \citep{wiegand-etal-2019-detection,parmar-etal-2023-dont}"
- Draft: "a partial-input model missing the query the pipeline itself issued
  \citep{wiegand-etal-2019-detection,parmar-etal-2023-dont}"
- Problem: both papers are about annotator or instruction artefacts and were cited as the contrast
  case. The draft now cites them as if they supported the claim they were contrasted with.
- Fix: restore "rather than an annotator artefact" before the citation.

**C8. Related work, thorne2018fever attribution (S564–566 → D537–540).**
- Source: "the closest task-level precedent is fact verification \citep{thorne2018fever}, where a
  claim-only classifier matches evidence-aware models because of how the corpus was built
  \citep{schuster-etal-2019-towards}"
- Draft: "Partial-input baselines recover much of the label from part of the input in … fact
  verification \citep{thorne2018fever,schuster-etal-2019-towards}"
- Problem: FEVER (thorne) introduced the task and corpus, not the partial-input finding.
- Fix: cite thorne for the task and schuster alone for the finding.

**C9. Related work, levy / obamuyide / sainz merged (S589–591 → D559–561).**
- Source: "Casting a relation as a question about a named entity \citep{levy2017zero}, or as an
  entailment hypothesis verbalising the relation between two of them
  \citep{obamuyide2018zero,sainz2021label}"
- Draft: "Casting a relation as a question about named entities
  \citep{levy2017zero,obamuyide2018zero,sainz2021label}"
- Problem: obamuyide and sainz use entailment hypotheses, not questions. This is a minor
  mis-attribution.

**C10. Related work disclaimer broadened (S580 → D555–556).**
- Source: "\emph{We claim no part of that diagnosis.}"
- Draft: "\emph{We claim neither that diagnosis nor the pair-conditioned input.}"
- The broader disclaimer is defensible because of the new `lee2020biobert` and `zhong2021` citations.
  But together with D10, nothing in that paragraph says what is new.

**C11. Related work, swaminathan's role (S597–600 → D566–567).**
- Source: specific claim, "abstain under calibrated thresholds to automate 57–95% of clinical chart
  abstraction".
- Draft: "as is thresholding a score instead of issuing a verdict
  \citep{geifman2017selective,swaminathan2024selective}"
- Now generic (see D13).

**C12. Related work, zou2026llm (S610–611 → D603–604).**
- Source: "standardise names and categories post hoc"
- Draft: "first ask whether a text contains an interaction and standardise names post hoc"
- The added clause is supported by LITERATURE.md (l.108, l.247), and "and categories" was dropped.
  Acceptable.

**C13. BioRED entity-pair F1 gap, a stronger causal claim (S369–372 → D373–374).**
- Source: "Much of that gap is structural"
- Draft: "systems that read the whole abstract reach $74.1$ to $75.3$, mostly because a quarter of
  the related pairs are never co-mentioned in one sentence"
- Problem: the source's own evidence does not support "mostly":
  - the same model given the whole abstract reaches only 67.0;
  - the source attributes the rest to document-level modelling.
- Fix: restore "much of".

**C14. Rules, body summary (S493–497 → D524–527).**
- Source: "No rule has a parameter fitted to evaluation labels, and since four were written after
  reading the false positives above, each had to pass a criterion fixed in advance on the 42{,}747
  training rows"
- Draft: "eight deterministic rejection rules, validated on training data before evaluation"
- Problem: the body loses the caveat that four rules were written after reading this benchmark's
  false positives, so it reads more favourably. The caveat survives in the Limitations and in
  `app:rules`.
- Fix: add "four of them written after reading the benchmark's false positives".

**C15. Contribution 1 (S80 → D87).**
- Source: "in a zero-shot 32B model"
- Draft: "in zero-shot LLMs"
- Supported by App. scaling. Fine.

**C16. Limitations, re-annotation (S635–637 → D630–633).**
- Source: "go to blind expert re-annotation, mixed with an equal number of random controls and with
  labels and scores hidden; no label is changed on the strength of a model's disagreement."
- Draft: "are under expert adjudication, mixed with an equal number of random controls; no label in
  this paper is changed on the strength of a model's disagreement. A second annotation of all rows,
  with a sentence and a pair answer for each, is in progress."
- The required change was made: "blind" and "labels and scores hidden" are gone.
- This is consistent with ANALYSIS.md §2: the answers came from a pre-filled sheet, so they are not
  blind, and the 7 flips are proposed only, with the gold unchanged.
- Optional: say the review sheet was model-pre-filled.

**C17. Conclusion, first sentence (S617–618 → D611–612).**
- Source: "the sentence-level question is answered before it is asked, and a classifier built for it
  is left with little to decide."
- Draft: "A literature-mining pipeline for biotic interactions needs two decisions …"
- The required change was made correctly.

---

## MISSING CITATIONS

- **Keys:** none. All 58 `\citep`/`\citet`/`\citealp` keys of the source appear in the draft.
- **Citations kept but lost from the body or changed in role:**
  - `lai2023biorex`, `lai2025bioredirect`, `islamaj2024overview` (entity-pair F1 comparison): the
    body still gives 74.1–75.3 (D373) but cites nothing there; the citations are only in
    `app:bioredep` (D14).
  - `wiegand-etal-2019-detection`, `parmar-etal-2023-dont`: role inverted (C7).
  - `thorne2018fever`: now attributed a partial-input finding (C8).
  - `obamuyide2018zero`, `sainz2021label`: attributed the "question" framing (C9).
  - `swaminathan2024selective`: reduced to a generic citation; its specific finding is dropped (C11,
    D13).
- **New keys:** the 21 new keys (dietterich1997solving, hoffmann2011knowledge, li2021dual,
  lee2020biobert, zhong2021frustratingly, mraz2026fewshot, etc.) all resolve in
  `paperA/reframe_extra.bib`.

---

## REFERENCE PROBLEMS

**R1. D125–126.** "answers no on $34.5\%$ of the corpus passages (\S\ref{sec:sentence})".
- §sentence never states 34.5%. It gives 34,242 passages, 41% disagreement and κ 0.26.
- Fix: add the 34.5% to the §sentence "Labels" paragraph, or drop the pointer.

**R2. D1065.** Self-reference: `app:rules` cites "(Appendix~\ref{app:rules})" inside itself.

**R3. D1293.** Self-reference: `app:deploy` cites "(Appendix~\ref{app:deploy})" inside itself.

**R4. D803 (`app:encoders`).** "Three of the seven alternative recipes of \S\ref{sec:res-ablation}".
- The seven recipes and the development screen are now listed only in `app:ablation` (D1237–1241).
  Body §4.3 no longer lists them.
- Fix: point to Appendix~\ref{app:ablation}.

**R5. D1189 (`sec:negative`).** "one of the three recipes of \S\ref{sec:res-ablation} that passed the
development screen". Same problem as R4: point to Appendix~\ref{app:ablation}.

**R6. D1196 (`sec:negative`).** "For the whole-abstract BioRED model of \S\ref{sec:biored}".
- The whole-abstract model (67.0) is now only in `app:bioredep`.
- Fix: point to Appendix~\ref{app:bioredep}.

**R7. D658 (Limitations).** "Four of the eight rules were selected after reading this benchmark's
false positives (\S\ref{sec:rules})".
- The body section no longer says this (C14).
- Fix: point to Appendix~\ref{app:rules}.

**R8. D1133 (`sec:notfixed`).** "The \emph{authority} and \emph{same\_organism} rules
(\S\ref{sec:rules})".
- The rules are named only in `app:rules`.
- Fix: point to Appendix~\ref{app:rules}.

**R9. D1140 (`sec:notfixed`).** "direction is scored on its own annotation (\S\ref{sec:deploy})".
- `sec:deploy` is now a label on the merged section "Candidate rules and deployment", which contains
  one clause on direction. The annotation and `tab:direction` are in `app:deploy`.
- Fix: point to Appendix~\ref{app:deploy}.

**R10. `tab:rules`.** Now unreferenced anywhere. Its only reference was in the source's §6 (D15).
- Fix: cite it from `app:rules`.

**R11. D286, D293.** `Figure~\ref{fig:threshold}` and `Table~\ref{tab:curve}` are cited from body
§4.1 but now live in `app:curve`.
- The body's "ahead at every one of 99 thresholds" now rests on an appendix figure.
- Fix: write "(Figure~\ref{fig:threshold}, Appendix~\ref{app:curve})".

**R12. Duplicate titles.** Body §4.3 and Appendix `app:ablation` are both titled "The query, not the
encoder".
- Fix: rename the appendix, e.g. "Query ablation and encoder screen".

**R13. Duplicated text in `app:deploy`.** The summary paragraph D1290–1299 repeats D1320–1334 nearly
word for word: 110M with two heads, three respects, threshold 0.5, 0.934, 32.3/s, escalation
0.941/0.911.
- Fix: delete D1290–1299.

**R14. Duplicated, dangling text in `app:rules`.** D1062–1073 duplicates the "How they were
validated" paragraph.
- It opens "Several of these mechanisms …" and refers to "the false positives above", with no
  antecedent in the appendix (the preceding appendix is Error examples).
- Fix: merge it into the existing paragraphs and point to §\ref{sec:errors}.

**R15. `sec:rules` and `sec:deploy`.** Both labels are on one section (D520–521).
- This compiles, but every `\S\ref{sec:deploy}` now lands on a section whose deployment content is
  one sentence (see R9).

**R16. D846–849 (`app:prompts`).** "The three questions put to the 32B teacher in
Table~\ref{tab:main} … The triple question is the prompt that labelled the training corpus."
- §sentence (D431) now sends readers here for the sentence question, which also labelled the
  34,242-passage sentence corpus and the LLM silver labels (including Qwen3.5-122B and Qwen3.8-27B).
- The appendix does not say so.

**R17. Appendix order.** `app:ablation`, `app:curve`, `app:deploy` and `app:bioredep` are appended
after `sec:negative`.
- `app:encoders`, which depends on the `app:ablation` screen, comes before it.
- This is cosmetic, but reordering would let R4 and R5 read naturally.

Unreferenced labels that were already unreferenced in the source (not new): `app:artifacts`,
`sec:conclusion`, `sec:intro`, `sec:related`, `sec:res-main`, `tab:perseed`.

---

## SURVIVING WORDING THAT THE REFRAME MAKES FALSE

**Checked and clean:**
- "answered before it is asked": absent.
- "little to decide": absent.
- "blind": absent.
- "labels and scores hidden": absent.
- D184–185, "annotating direction without model scores": concerns the separate Acanthamoeba
  direction relabel, not the gold re-annotation. Not a blindness claim about the review; keep only if
  it is true of that event.

**F1. D123–124 (Setting).** Survives verbatim from S117–118: "A classifier trained to separate passages
with interaction language from passages without it is, on this distribution, being asked about a
constant."
- The draft follows it with "Whether a passage describes an interaction is not constant…".
- But no model in the paper is a lexical interaction-language classifier, so readers will take the
  sentence as describing the paper's sentence-level models, which is what the source meant.
- The constant / not-constant juxtaposition also invites confusion.
- Suggestion: "A keyword test for interaction language would accept every passage in the pool.
  Whether a passage describes an interaction is a different question, and it is not constant: …"

**F2. D57–58 (Intro).** "Every candidate arrives already satisfying, lexically, what a sentence
classifier learns to detect".
- This implies sentence classifiers learn a property every candidate already has, i.e. that the
  sentence question is uninformative.
- But the reframe's sentence-label model learns a question the teacher answers NO on 34.5% of
  passages, and it is the best biodiversity sentence filter (AUPRC 0.985).
- Suggestion: "already satisfying the lexical trigger a keyword filter checks".

**F3. D536 (Related work heading).** "Classifiers that answer an easier question."
- Under the reframe the sentence question is a different decision, not an easier or degenerate one.
- Suggestion: "Partial-input classifiers".

**F4. D981 (App. scaling).** "scale mostly improves the answer to the right question".
- This calls the sentence question the wrong one.
- Suggestion: "the answer to the pair question, the one this pair-level gold scores".

**F5. D310–311 (§4.1).** "the case in which the sentence-level question and the pair question
coincide".
- Under the reframe's semantic sentence question they need not coincide even with two recognised
  taxa: a passage can describe an interaction with an organism outside the candidate pair (an
  unrecognised organism, or "host").
- The reframe finds the sentence-label model's whole biodiversity advantage in exactly such
  out-of-pair interactions (D455–457).
- Suggestion: "nearly coincide", or "the case closest to a single-pair passage".

**F6. D126–127 (Setting).** "What is missing from the sentence-level input is the query the pipeline
itself issued: which pair."
- Nothing is missing for the sentence decision, only for the pair decision.
- Suggestion: "What the sentence-level input lacks for the pair decision is …".

**F7. D290–291 (§4.1).** "What the decision needs is the pair." (Low priority.)
- The paper now argues there are two decisions.
- Suggestion: "What the pair decision needs is the pair, not the relation."

**F8. D367–369 (§4.2).** "The teacher, asked the sentence-level question, accepts 92\% of its sample
(AUPRC $0.566$ at a base rate of $0.48$)". (Low priority.)
- This is scored against pair labels, yet the same question reaches sentence-level AUPRC 0.908 in
  §sentence (Table `tab:sentence`). Read alone, it suggests the sentence question is useless.
- Suggestion: add "scored against pair labels".

**F9. Stale after the reframe (not false, but now incomplete):**
- D1151–1156 (App. artifacts):
  - The zero-shot model list "Qwen3 0.6B--32B and Qwen3.5-122B" omits Qwen3.8-27B, which is used for
    the silver labels in Table `tab:sentence`.
  - "about 50 GPU-hours" predates the 34,242-passage sentence labelling (about 2 h 15 min), the
    sentence-label students and the enumerated-pair scoring.
- D846–849 (App. prompts): see R16.
