# Critical review of `paperA/paperA_reframe.tex` (sentence-level reframe), 2026-10-05

Reviewer role: demanding ACL area chair, and co-author. Read-only review. The draft reviewed is the file as of
05:12 (PDF built at 05:12:34; 21 pages; Conclusion on page 8, Limitations from page 9; abstract about 194
words). Line numbers below are for `paperA/paperA_reframe.tex` at that time.

The .tex was edited again at 05:24:59, by another stage, not me. The changes I saw:
- abstract l.38: "provisional LLM-derived labels" → "provisional, partly LLM-derived labels";
- l.474-475: "indistinguishable" → "close";
- "\citep[cf.][]{li2021dual}" at l.441/l.691;
- a reworded BioRED EP sentence (l.383);
- some appendix edits.

I re-checked every MUST FIX anchor below against the 05:24:59 file. All are still present, at the same line
numbers. The PDF had not been rebuilt at 05:25. Every number I quote was checked
against `results/paperA_v2/sentence_level/{biodiv_sentlab,biored_sentence,biodiv_human,sentlab_label_stats}.json`.
I did not open the sealed gold-review key.

Decisions I took without being able to ask anyone:
- I review the draft as a submission to ARR and also as a candidate to replace `paperA.tex`.
- I treat the 22-item review only as a spot check, as instructed. I did not use it to re-score anything.
- Writer issues that are already in `writer_work/{claims_review,final_check}.md` and already fixed in the
  draft are not repeated. Writer issues still open are mentioned only where my verdict depends on them.
- I list as MUST FIX only things I consider clearly wrong. Everything arguable is a JUDGEMENT CALL.

---------------------------------------------------------------------------------------------------------

## 1. ACL-style review

### Summary

The paper studies the filter step of co-occurrence-based literature mining for biotic interactions. That step
involves two decisions:
- a passage-level decision: does the passage describe an interaction?
- a pair-level decision: do these two taxa interact, as written?

Pair-level results (unchanged from the submission version):
- One BiomedBERT-base cross-encoder is distilled three times from one 48k-row corpus labelled by a Qwen3-32B
  teacher, with different inputs: passage only, passage + pair, passage + triple.
- On a 437-row expert-graded benchmark, AUPRC rises from 0.851 to 0.918 and F1 from 0.775 to 0.871.
- On a BioRED rebuild under the BC8 protocol, AUPRC rises from 0.738 to 0.843.
- The gain sits in passages that name more than two entities.
- The relation term adds nothing to ranking.

New in this draft, §4.3 "From pairs to sentences":
- The pair verifier is read as a sentence classifier by taking the maximum over pairs ("pair-max").
- It is compared with two sentence classifiers: the candidate-trained sentence arm and a "sentence-label model".
- On BioRED, with gold-derived "co-mentions a related pair" labels: pair-max beats the candidate-trained arm and
  at least matches the sentence-label model (F1 +0.024, significant; AUPRC +0.010, not significant).
- On the biodiversity benchmark, with silver LLM-derived labels: the sentence-label model, distilled from the
  teacher's sentence answers, beats pair-max under all three label definitions. The paper itself notes that
  this edge is partly circular.
- A perfect sentence filter has pair precision 0.536 on BioRED and, provisionally, 0.724 on the biodiversity
  benchmark.
- The paper concludes: "Ask each question for its own decision; one teacher can label both."

### Strengths

1. **The pair-level comparison is controlled and well reported.**
   - Same encoder, same corpus, three seeds; only the input differs.
   - Thresholds are pre-specified or block-held-out, and paired tests are used throughout.
   - The benchmark's history is disclosed: about 100 configurations were scored on it, and one label was changed.
   - The BioRED replication follows a published protocol and puts the gain in the predicted stratum.
2. **The sentence-level section is unusually honest.** It reports a result that goes against the authors' own
   method on their own benchmark. It locates that result (only in the 191 pair-negative passages), calls it
   partly circular, and reports the BioRED mechanism test against the reframe's intuition: the edge sits in
   single-pair sentences, not where pairs compete.
3. **The multi-instance link is cited, and no novelty claim is made for it.** Dietterich, Hoffmann, Ilse,
   Li 2021, Rei & Søgaard, Magge and Farkas are all cited, with "We claim none of these" (l.581). Writing that
   a pair-route win is "the expected outcome" on BioRED (l.441, l.690-691) is the correct and defensible stance.
4. **A strong result that does not depend on silver labels** (l.484-486). A sentence classifier trained on
   sentence labels is a worse pair filter (AUPRC 0.794) than the candidate-trained sentence arm (0.855) and
   much worse than the pair verifier (0.923).
   - This pre-empts the obvious objection to Table 1 ("your sentence baseline was trained on the wrong
     labels").
   - Its BioRED analogue, the perfect-filter precision of 0.536, is gold-derived.
5. **The reframe removes a claim of the submission version that the new evidence refutes.** The submission
   says the sentence question "is answered before it is asked" (paperA.tex abstract l.31, l.56 and the
   Conclusion at l.617-618). Asked the sentence question, however, the teacher says no on 34.5% of the corpus passages.
   The reframe's Setting (l.124-133) states this correctly.

### Weaknesses

1. **The sentence-level evidence supports no positive claim for the method on the authors' own benchmark.**
   - On biodiversity, the best sentence filter is the sentence-label model, not pair-max.
   - On BioRED, pair-max ≥ sentence-label model is expected by construction, because the label is the OR of
     the pair labels. The paper says so itself (l.441, l.690-691).
   - The BioRED task is also near ceiling: base rate 0.838, accept-all F1 0.912.
   - Pair-max also receives gold entity spans and 4.8 times as many training rows (15,640 against 3,277).
   - So the new section adds little evidence that the reader can act on. "Ask each question for its own
     decision" rests, for its sentence half, on provisional, partly circular labels.
2. **The biodiversity sentence labels are silver and circular.**
   - 191 of 437 labels are LLM answers to the sentence prompt.
   - Under the majority definition, one of the three voters is the teacher whose answers the winning model
     imitates.
   - The "non-teacher" definition still uses two Qwen-family models (Qwen3.5-122B and Qwen3.8-27B). Errors
     correlated within the model family are therefore not excluded, and the draft does not say so.
   - The F1 operating points are fitted on these silver labels (block-held-out).
   - Under the majority labels, F1 is weakly informative (base rate 0.778): pair-own's F1, 0.876, equals
     accept-all's 0.875.
3. **The explanation of the reversal is stated as a finding: "depends on what the sentence label counts"
   (l.77, l.583).** The two benchmarks differ in at least five ways at once:
   - who decides the label (gold or LLM);
   - who trained the sentence model (gold labels or the teacher);
   - where the pairs come from (gold spans or TaxoNERD);
   - the domain;
   - the base rate.
   The circular share of the edge is not separated out. The causal attribution is not supported (MUST FIX
   M1/M2).
4. **The story is two papers.**
   - Title, §1 (first half), §2-§4.2, §5 and §6 are the submission's pair paper, nearly verbatim.
   - §4.3 and the reframed intro/conclusion add a sentence study whose outcome on the authors' benchmark goes
     against the method of the title ("Which Pair Is It About?").
   - A reader comes away asking which result matters to the authors. The user's stated goal is sentence
     level, and on that goal the paper's provisional answer is "use a sentence classifier".
5. **Human sentence labels are absent.**
   - The 437-row sheet has 0 of 417 rows answered.
   - The only human sentence answers are 19 items from an adjudication of a pre-filled sheet: non-blind,
     selected, 5 of them forced positive by pair gold.
   - Reviewers will read §4.3's biodiversity column as "results pending" and the BioRED column as "by
     construction".
6. **Anonymity and overlap risk is inherited from the submission and still open** (details in §5).
   - "We pulled the retrieval pool in full", Reject50 ("candidates the retrieval pipeline discarded") and
     "We take the diagnosis as given and question only the remedy" are still in the draft.
   - New in this review: the Biotx100 design echoes the system authors' own published evaluation.
7. **Smaller issues.**
   - Two different models are called "sentence-level": Table 1's "Sentence-level" and Table 5's "Sentence
     arm, candidate labels" are the same model, while the "sentence-label model" is a different one.
   - The body cites an appendix figure and an appendix table as if they were in the body (l.292, l.299).
   - The Table 5 caption's "its ceiling is biased upward" (l.419) does not say which ceiling.
   - Qwen3.8-27B's licence is not stated (l.1173-1174).
   - The anonymous code URL is still `XXXX` (l.44).

### Questions to the authors

1. Does the human sentence label agree with pair-max or with the sentence-label model on the 191 pair-negative
   passages? This single question decides §4.3. Without it, what should a reader conclude for biodiversity?
2. Who adjudicated the 22 items: the expert who made the gold, or a second person? "Expert adjudication"
   (l.632, l.644) is ambiguous. One reading measures self-consistency; the other measures a second opinion.
3. On BioRED, does a sentence model that is given all entity mentions (marked) close the gap to pair-max? If it
   does, the BioRED edge is entity information rather than the pair question (the draft lists this as untested,
   l.653-655).
4. Why should "what the label counts" explain the reversal, rather than "who produced the label"? Under the
   gold-positive-only comparison (l.474-476) the arms are tied. Is there any setting where the sentence-label
   model wins on labels not produced by an LLM?
5. How were the Biotx100 rows obtained and graded? Is any part of the benchmark derived from the system
   authors' own evaluation (100 triples; GloBI-validated against random)? The local `data/evaluation/eval_100.tsv`
   has exactly 31/100 positives, the same figure as the published "31%".
6. The 22-item adjudication sided with the models against the gold on 7 of the 11 selected items and with the
   gold on all 11 controls. Why is this not reported, given that its sentence answers are?
7. Is a simple combination of the two routes (for example, accept if either is confident), chosen on
   development data, better than both? The Limitations list it as untested.

### Scores (ARR)

- **Overall assessment: 2.5 / 5 (borderline Findings).**
  - The pair-level contribution is solid and would by itself sit at about 3 (Findings).
  - The added sentence-level section lowers it. Its biodiversity half is provisional and partly circular. Its
    BioRED half is expected by construction and near ceiling. Its take-home ("ask each question for its own
    decision") is unsurprising.
  - The section also blurs the paper's message.
  - A reviewer will very likely write "come back with human sentence labels".
- **Soundness: 3 / 5 (acceptable).** The pair-level claims are well supported (about 4). The sentence-level
  claims are honestly qualified in most places, but two causal sentences overreach (M1/M2), and one sentence
  contradicts the BioRED construction (M3).
- **Excitement: 2.5 / 5.** The sentence-level reframe was meant to be the exciting part. As the evidence
  stands, it ends in "it depends", and the provisional answer favours the baseline formulation.
- **Reproducibility: 4 / 5.** Recipes, seeds, thresholds and the code placeholder are given. The silver-label
  pipeline depends on a local Qwen3.8-27B build.
- **Comparison: the submission version** (`paperA.tex`) would get about Overall 3.0, Soundness 3.5,
  Excitement 3.0. Its one known overclaim ("the sentence-level question is answered before it is asked") is
  invisible to reviewers but now known to be false in its semantic reading; see §8.

---------------------------------------------------------------------------------------------------------

## 2. Is the sentence-level claim supported AS STATED?

### BioRED (gold-derived "co-mentions a related pair" labels; 2,582 sentences; base rate 0.838)

| Line | Exact claim | Verdict |
|---|---|---|
| l.34-37 (abstract) | "Read at the sentence level as a maximum over pairs, the pair verifier reaches BioRED AUPRC $0.965$, beating a sentence classifier trained on the same candidates ($0.930$) and at least matching one trained on BioRED's derived sentence labels ($0.955$; F1 $0.954$ against $0.930$)." | **Supported.** pair-max − sentence arm: AUPRC +0.035 [+0.024, +0.047] (abstract clusters). pair-max − sentence-label model: AUPRC +0.010 [−0.001, +0.021]; F1 +0.024, p = 2.1e-17. "At least matching" is the right verb. What the abstract omits: the label is the OR of the pair labels, so the result is expected by construction. Judgement call J1 suggests "pair-derived". |
| l.78-79 (intro) | "On BioRED, where a sentence is positive when it co-mentions a related pair, the pair route is at least as good as the sentence-label classifier and better at its operating point." | **Supported.** "Better at its operating point" = F1 at development thresholds, significant. |
| l.450-454 (§4.3) | "Pair-max scores highest of the three … Against the sentence-label model pair-max is better at the operating point (F1 $+0.024$ …) but its AUPRC gain, $+0.010$ $[-0.001, +0.021]$, is not significant." | **Supported**; the numbers match `biored_sentence.json`. |
| l.454-458 | "Nor does the edge come from pairs competing within a sentence: it lies in the 777 single-pair sentences …" | **Supported**, and commendably candid. |
| l.458-459 | "Pair-max is also told which spans are BioRED's gold entities, which the sentence models are not." | Correct disclosure. |
| l.459-461 | Zero-shot: "the 32B model reaches $0.959$ against $0.908$ ($+0.051$ …, sentences resampled)" | **Supported.** The prompt/label mismatch ("states" against "co-mentions") is disclosed at l.693-694. |
| l.614-617 (conclusion) | "the pair verifier beats a sentence classifier trained on the same candidates (on BioRED, …) and is at least as good as one trained on BioRED's derived sentence labels" | **Supported.** |
| l.432 | "how far the scored pairs fall short of a passage's interactions is what the two benchmarks measure." | **Not supported for BioRED.** There the sentence label is the maximum of the scored candidates' own labels (l.440-441, l.690-691), so the scored pairs cannot fall short of the label by construction. **MUST FIX M3.** |

**BioRED verdict: supported as stated, apart from l.432.** The claim is weak evidence, which the draft partly
concedes: it is expected by construction, near ceiling, and pair-max has gold spans and 4.8 times as many
training rows.

### Biodiversity (437 passages; PAIR gold; 191 sentence labels from LLMs; provisional)

| Line | Exact claim | Verdict |
|---|---|---|
| l.37-39 (abstract; since 05:24:59 it reads "under provisional, partly LLM-derived labels") | "On our benchmark, under provisional LLM-derived labels, a classifier distilled from the teacher's sentence answers is better under all three label definitions (AUPRC $0.985$--$0.987$ against $0.966$--$0.977$)." | **Supported, and flagged as provisional.** All CIs exclude 0. The abstract does not mention circularity; that is judgement call J2. |
| l.39-40 | "A perfect sentence filter has a pair precision of $0.536$ on BioRED (provisionally $0.724$ on ours)." | **Supported**: 246/340 = 0.7235. The qualitative direction (< 1) survives any labelling in which at least one pair-negative passage is sentence-positive. The human spot check has 5 such cases. |
| l.41 and l.618-619 | "Ask each question for its own decision; one teacher can label both." | **Reads as final.** The pair half is robust. The sentence half, asking the sentence question for the sentence decision, rests only on the provisional, partly circular biodiversity result; BioRED points the other way. **MUST FIX M4.** |
| l.76-78 (intro) | "The outcome depends on what the sentence label counts." | **Not supported as a causal statement** (weakness 3). **MUST FIX M1.** |
| l.80-82 | "On our benchmark, whose provisional, LLM-decided sentence labels also count interactions outside the candidate pair, a classifier distilled from the teacher's answers to the sentence question is better." | **Supported, and flagged as provisional.** |
| l.84-86 | "… pair verification is a competitive but, provisionally, not the best sentence filter …" | **Supported.** "Competitive" is fair: −0.010 to −0.020 AUPRC. |
| l.464-467 | "Pair-max ranks above … the candidate-trained arm under all three label definitions, significantly so only under the two without the teacher …" | **Supported.** |
| l.467-470 | "The sentence-label model beats pair-max under all three: by $0.020$, $0.010$ and $0.012$ AUPRC … $0.929$ against $0.897$ … 46 … 22, $p = 0.0049$" | **Supported.** 0.0195 rounds to 0.020; the table's rounded entries give 0.019. That difference is cosmetic. |
| l.470-476 | "Its whole advantage lies on the 191 passages … part of its advantage is circular. Where positives are decided by gold the three are indistinguishable …" (since 05:24:59 "close") | **Supported.** "Close" is the safer word. Note that the "gold positives" comparison still uses 97 LLM-decided negatives. |
| l.583 (related work) | "… and find that the answer depends on what the sentence label counts." | **Not supported as stated** (as for l.77). **MUST FIX M2.** |
| l.614-616 | "… beats a sentence classifier trained on the same candidates (on BioRED, and on ours under two of three label definitions)" | The "on ours" clause is silver and not flagged at this point. **MUST FIX M7** (short). |
| l.641-643 (Limitations) | "On these labels the sentence-label model is the better sentence filter under every definition, … so only human sentence labels can confirm it." | **Supported and honest.** |

**Biodiversity verdict:** the sentence-label model's win is stated correctly and labelled provisional in the
abstract, intro, §4.3 and Limitations. The overreach is in the interpretation ("depends on what the label
counts") and in the unqualified closing recommendation.

---------------------------------------------------------------------------------------------------------

## 3. Multi-instance relation and novelty

**Acknowledged, three times, correctly:**
- l.73-75 (intro): "If the pairs scored cover a passage's interactions, the passage describes one exactly when
  one of its pairs interacts \citep{dietterich1997solving}".
- l.428-432 (§4.3): "the multi-instance assumption \citep{dietterich1997solving,hoffmann2011knowledge}".
- l.569-583 (related work): MI, OR/max aggregation, Ilse, Li 2021 (instance supervision as the upper bound),
  Rei & Søgaard, Da San Martino, Magge (a counterexample), Farkas (mixed results), Chowdhury and Xie, ending
  "We claim none of these."

**Novelty is stated defensibly:**
- There is no "first", "novel" or "we introduce" anywhere in the body (grep checked).
- The positive claim (l.555-559) is "a controlled measurement …; a bound …; the candidate rules; and the
  sentence-level comparison of \S\ref{sec:sentence}".
- l.603-604 ("We are not aware of published work that verifies literature-mined biotic-interaction triples
  against their source passage before ingestion") is hedged and inherited. It is acceptable.

**Remaining issues:**
- All of LITERATURE.md's must-not items are respected (#1-#4, #17-#20). Pavlopoulos is cited for #19 (l.487),
  and "expected outcome" covers #4.
- The one residual overreach is the causal "depends on what the label counts" (M1/M2). It turns a two-benchmark
  observation into a mechanism.
- After the honest negative on biodiversity, the sentence-level novelty reduces to "a controlled
  sentence-vs-pair-route comparison under one teacher". That is thin, but it is stated without overclaiming.

---------------------------------------------------------------------------------------------------------

## 4. Is the provisional/silver biodiversity evidence flagged everywhere?

| Place | Flagged? | Note |
|---|---|---|
| Abstract l.37-40 | Yes: "under provisional LLM-derived labels", "(provisionally $0.724$ on ours)" | The closing line l.41 reads as final (M4). Circularity is not mentioned (J2). |
| Intro l.80-85 | Yes: "provisional, LLM-decided", "provisionally, $0.724$", "provisionally, not the best" | l.77 overreaches (M1). |
| Contributions l.93-95 | **No**: "on two benchmarks, stating where each wins" | **M6.** |
| Setting l.148-149 | **No**: "its sentence labels are described in \S\ref{sec:sentence}" makes the benchmark sound as if it ships sentence labels | **M5.** |
| Setting l.130-132 | n/a | "the teacher … answers no on $34.5\%$" is correctly attributed to the teacher. |
| §4.3 Labels l.441-447 | Yes: "\emph{provisional}, on silver labels" | |
| §4.3 heading l.463 | Yes: "Biodiversity, provisional." | |
| §4.3 l.481 | Yes: "on our silver labels, $0.724$" | |
| Table 5 header l.393 | Yes: "Biodiversity, \emph{provisional}" | The caption explains the LLM labels. "its ceiling is biased upward" (l.419) is ambiguous (J7). |
| Related work l.583 | **No**: "find that the answer depends on what the sentence label counts" | **M2.** |
| Conclusion l.614-619 | Partly: "on our provisional labels", "provisionally, $0.724$ here" | "on ours under two of three label definitions" is unflagged (M7). The closing line reads as final (M4). |
| Limitations l.633-634, l.637-655 | Yes, thoroughly | "A second annotation … is in progress" (l.633-634) is true only in that the sheet exists: 0 of 417 rows are answered (J8). l.690-692 "our benchmark … is the harder test" presumes labels it does not yet have (J9). |

The spot check is described honestly (l.643-647): non-blind, selected, 3 UNSURE dropped, 5 forced by gold, and
all three κ values given. It is never presented as a benchmark result. Good.

---------------------------------------------------------------------------------------------------------

## 5. BiotXplorer framing

### Every direct mention

- l.106-113: "BiotXplorer \citep{ruch2024biotxplorer} proposes candidate biotic-interaction triples from the
  literature indexed by SIB Literature Services \citep{gobeill2020sibils}. For each passage it matches taxon
  surface forms against a taxonomy and interaction surface forms against a vocabulary derived from the Relation
  Ontology, and emits every resulting $(s_1, r, s_2)$ combination with the passage, the matched surface strings,
  and their canonical forms, and it shows each interaction with its supporting passages. A passage it shows
  should therefore both describe an interaction and concern the pair it is shown for: its authors' own
  evaluations are judgements of that kind, a pair in a passage. We use it only as the source of candidates: no
  component of it is a baseline in this paper."
- l.115-122: "The problem this paper addresses was diagnosed by BiotXplorer's own authors.
  \citet{ruch2024biotxplorer} graded 100 random BiotXplorer triples, found that the system identified the
  interacting species with a precision of 31\%, and named the main cause of error: ``instances where passages
  listed multiple species, which can be automatically filtered out.'' We take the diagnosis as given and
  question only the remedy. …"
- l.143-145: "… which is why its positive rate is far above the 31\% \citet{ruch2024biotxplorer} measured on
  random triples at any rank."

### Indirect references (the system's data and pipeline, written as "ours")

- l.61: "in the pipeline we study, all $175{,}588$ candidates of its retrieval pool".
- l.125: "We pulled the retrieval pool in full".
- l.145-146: "\textsc{Reject50} is candidates the retrieval pipeline discarded".
- l.203: "drawn from the same retrieval pool".
- The block name \textsc{Biotx100}.

### Assessment

- **Application only: yes.**
  - The phrase "This is the application we build for" is gone from the submission text.
  - l.110-113 motivate both decisions from the tool's public behaviour: it shows passages as evidence for a
    pair. This is the argument LITERATURE.md allows.
  - "no component of it is a baseline" is correct.
  - The intro does not name it. That is good for anonymity, but the application motivation is weaker than the
    user hoped (see J10).
- **No names or affiliations: yes**, apart from the citation itself. "Ruch et al. (2024)" in third person is
  ordinary third-party citation and is allowed.
- **Overlap or internals: two inherited risks remain.**
  1. *Inherited, already flagged by the writer (S12):* "We pulled the retrieval pool in full", Reject50
     (discarded candidates) and "We take the diagnosis as given and question only the remedy".
     - Full-pool access and access to discarded candidates imply privileged access to the system. That leaks
       a working relationship.
     - Arguing against the authors' stated remedy pre-empts their forthcoming paper if that paper implements
       the remedy.
  2. *New, not in the writer's checks:* the Biotx100 block echoes the system authors' published evaluation
     design. That evaluation used 100 triples, compared GloBI-validated with random, and reported 31%. The
     local `data/evaluation/eval_100.tsv` (100 rows, 31 positive on `evaluation_pair_interacting`) looks like
     that very set.
     - If Biotx100 (or its grading) derives from the system team's evaluation, the paper uses that team's
       unpublished grading data. That is both overlap and an anonymity leak.
     - The explanation at l.143-145, "far above the 31% … because top-ranked", would then also need
       re-checking.
     - I could not verify the provenance read-only. This is question 5 to the authors and judgement call J11.
  3. l.108-110 ("emits every resulting combination with … the matched surface strings, and their canonical
     forms") is output-format detail beyond the public abstract. It is inherited. It is acceptable only if the
     public tool visibly shows it.

---------------------------------------------------------------------------------------------------------

## 6. Better or worse with reviewers than the submission version? Coherence

**Worse on balance, though more honest.**

What the reframe gains:
- It drops a refuted claim. The submission's "That question is answered before it is asked" (paperA.tex
  abstract and l.56) and "a classifier built for it is left with little to decide" (Conclusion) are false
  when read semantically: the teacher says no on 34.5% of corpus passages, and the sentence-label model reaches
  0.985 AUPRC against a 0.778 base rate.
- It pre-empts the "unfair sentence baseline" objection with a robust result: the sentence-label model ranks
  pairs at 0.794, against 0.923 for the pair verifier.
- It engages the curation framing the user cares about.

What the reframe loses:
- **A sharp thesis becomes a hedge.** The submission says "ask about the pair". The reframe says "ask each
  question", and on its own benchmark the sentence half favours the baseline formulation.
- **It adds a section reviewers will mark as incomplete.** The biodiversity results are provisional, with an
  admitted circular component and no human labels. ARR reviewers routinely score "results pending human
  labels" at 2-2.5.
- **The BioRED sentence result is pre-announced as the expected outcome** (l.441). Reviewers give little credit
  for confirming what follows by construction.
- **Space.** §4.3 displaced the operating-curve, deployment and ablation material to appendices. Some reviewers
  valued the 32 candidates/s CPU configuration and the calibrated operating curve as practical contributions.

**Coherence: it reads as a pair paper with a sentence section bolted on, though the bolting is skilful.**
- The intro does set up two decisions from its first paragraph (l.50-56), and the conclusion tells the same
  "two decisions, two questions" story.
- However:
  - The title is still the pair question ("Which Pair Is It About?").
  - Three of four results subsections, §5 and §6 are the pair paper verbatim.
  - The "Sentence-level" row of Table 1 and the "sentence-label model" of Table 5 are different models with
    near-identical names.
  - The sentence section's outcome on the authors' benchmark runs against the title's method.
- A reviewer will reasonably ask whether this is a paper about pair verification, with a negative sentence
  appendix, or a paper about sentence filtering that lacks sentence labels.
- The user's stated intent ("the whole goal is to assess whether or not a sentence has a biotic interaction")
  is not what the evidence currently rewards.

---------------------------------------------------------------------------------------------------------

## 7. MUST FIX (clear errors only)

Every change below is length-neutral or within ±2 words.

**M1. l.77-78 (intro). Causal overreach.**
- Current: `The outcome depends on what the sentence label` / `counts.`
- Replace with: `The outcome differs between the two benchmarks.`
- Why: the benchmarks differ in label source (gold against LLMs, including the teacher), in the sentence
  model's training labels (gold against the teacher), in pair source (gold spans against TaxoNERD), in domain
  and in base rate. §4.3 says the edge is partly circular. The following two sentences still describe what each
  label counts, so no information is lost.

**M2. l.582-583 (related work). Same overreach, stated as a finding.**
- Current: `training rows, and find that the answer depends on what the sentence label counts.`
- Replace with: `training rows, and find that the answer differs between the two benchmarks.`

**M3. l.432 (§4.3). Contradicts l.440-441 and l.690-691.**
- Current: `scored pairs fall short of a passage's interactions is what the two benchmarks measure. We compare`
- Replace with: `scored pairs fall short of a passage's interactions is what our benchmark measures. We compare`
- Why: on BioRED the sentence label is the maximum of the scored candidates' own labels, so no shortfall is
  possible by construction.

**M4. l.41 (abstract) and l.618-619 (conclusion). A final-sounding recommendation on provisional evidence.**
- Current: `Ask each question for its own decision; one teacher can label both.`
- Replace with (12 words → 12 words): `Pending human sentence labels, ask each question separately; one teacher labels both.`
- Alternative, if "for its own decision" must stay (+2 words; the abstract is at about 194/200):
  `For now, ask each question for its own decision; one teacher can label both.`
- Why: on BioRED, pair-max is at least as good as the sentence-label model. The only evidence for asking the
  sentence question separately is the provisional, partly circular biodiversity result.

**M5. l.148-149 (Setting). Reads as if the benchmark ships sentence labels.**
- Current: `is pair-level; its sentence labels are described in \S\ref{sec:sentence}.`
- Replace with: `is pair-level; its provisional sentence labels are in \S\ref{sec:sentence}.`

**M6. l.94 (contributions). Unflagged provisional result.**
- Current: `labels and on sentence labels, on two benchmarks, stating where each wins, with the pair`
- Replace with: `labels and on sentence labels, on two benchmarks (ours provisionally), stating where each wins, with the pair`
- This adds 2 words. If the line overflows, drop "each" from "on each" at l.95.

**M7. l.615 (conclusion). Unflagged silver result.**
- Current: `same candidates (on BioRED, and on ours under two of three label definitions) and is at least as`
- Replace with: `same candidates (on BioRED, and provisionally on ours under two of three definitions) and is at least as`
- Word count unchanged.

**M8. l.44. Pre-submission blocker, also present in paperA.tex.**
- Current: `Code: \url{https://anonymous.4open.science/r/XXXX}.`
- Replace `XXXX` with the real anonymous repository ID before any upload. If there is none, delete the line.

No numeric disagreements between sections were found. I checked:
- abstract, intro, §4.3, Table 5, conclusion and Limitations against each other;
- all of them against the JSONs: 0.965/0.955/0.930; F1 0.954/0.930; 0.985-0.987 and 0.966-0.977; 0.020/0.010/0.012; 0.929/0.897 with 46/22, p = 0.0049; 0.724, 0.536 and 0.699-0.794; 394/148/191/43; 41%, κ 0.26 (13,925/34,242 = 0.407); 34.5%/65.5%; the spot check 13/14 (κ 0.81), 17/19 (0.68) and 15/19 (0.46); 85.9 against 3 against 6 passes.

The only cosmetic rounding point: "0.020" at l.467, against .985 − .966 = .019 from the table.

## JUDGEMENT CALLS for the user (not clearly wrong)

- **J1. Abstract l.36:** "BioRED's derived sentence labels" → "BioRED's pair-derived sentence labels". This
  costs no words and tells the reader the BioRED result is expected by construction. Recommended.
- **J2. The abstract and intro do not mention that the biodiversity edge is partly circular.** The 05:24:59 text,
  "provisional, partly LLM-derived labels", is accurate (246 of the labels come from gold). It still does not
  say that the winning model imitates one of the labellers. If space allows, write "under provisional labels
  partly decided by the teacher's own family of LLMs", or add "partly circular" in the intro (l.80).
- **J3. Limitations:** add that the two "non-teacher" voters share the teacher's model family (Qwen), so the
  non-teacher definition does not remove correlated errors.
- **J4. Report the pair outcome of the 22-item adjudication** in one clause of the Limitations: it sided with
  the models on 7 of 11 selected items and with the gold on 11 of 11 controls; the gold is unchanged. It is
  more consistent to report this than to report only its sentence answers. Never call it blind.
- **J5. Say who adjudicated** (l.632, l.644): the original expert, or a second annotator.
- **J6. Naming:** rename Table 1's "Sentence-level" row to "Sentence-level (candidate labels)", or rename Table 5's
  row to match it, so that readers do not confuse it with the sentence-label model.
- **J7. Table 5 caption l.419:** "its ceiling is biased upward" → "its perfect-filter precision is biased upward".
- **J8. l.633-634:** "A second annotation … is in progress". It is not started (0 of 417 answered). Write
  "is planned", or keep "in progress" only once rows exist.
- **J9. l.690-692:** "our benchmark … is the harder test" presumes human sentence labels. Consider "will be the
  harder test once its sentence labels are human".
- **J10. BiotXplorer as the application:** the reframe's intro does not use it. This is safer for anonymity and
  overlap, and I recommend keeping it that way. Motivate the sentence decision by curation triage, as done
  (l.53-56).
- **J11. Biotx100 provenance** (new; see §5). Confirm that no part of the benchmark or its grading comes from the
  system team's evaluation. If it does, decide with the team how to describe it without implying a working
  relationship, and re-check l.143-145.
- **J12. Inherited anonymity wording** (writer's S12): "We pulled the retrieval pool in full", Reject50 as
  "candidates the retrieval pipeline discarded", "We take the diagnosis as given and question only the remedy".
  A neutral alternative for the last one: "We test an alternative to filtering such passages out."
- **J13. Qwen3.8-27B:** state its licence (l.1173-1174) or drop the majority definition in favour of the two
  others. The conclusion is the same under all three.
- **J14. l.292, l.299:** say "(Appendix~\ref{app:curve}, Figure~\ref{fig:threshold})" so the body is visibly
  self-contained.
- **J15. Run the two untested controls before submission** if GPU time allows: (a) a BioRED sentence model with
  the entity mentions marked; (b) a development-tuned combination of the two routes. Each is a few
  four-minute runs. Run (a) first: it decides whether the BioRED sentence result is about the pair question at
  all.

---------------------------------------------------------------------------------------------------------

## 8. Recommendation

**Keep the submission version for this cycle, with a small patch. Adopt the sentence reframe only after human
SENTENCE labels exist for the biodiversity benchmark.**

Why not adopt the reframe now:
- Its new section's main result on the authors' own benchmark is provisional, partly circular, and against the
  title's method.
- Its BioRED result is expected by construction.
- With reviewers it scores below the submission (about 2.5 against 3.0), and it reads as two papers.
- The user's goal (sentence level) is not served by submitting a draft whose sentence-level conclusion may flip
  when the human labels arrive.

The patch to the submission (the user's decision; I did not edit anything):
1. Soften the claim that the new evidence refutes:
   - abstract: "so that question is answered before it is asked" → "so that question, read lexically, is
     answered before it is asked";
   - intro l.56: "That question is answered before it is asked." → "Lexically, that question is answered
     before it is asked.";
   - Conclusion l.617-618: "the sentence-level question is answered before it is asked, and a classifier built for it
     is left with little to decide" → "the sentence-level question is, lexically, answered before it is
     asked, and a classifier built for it cannot say which pair".
2. Optionally add two sentences built only on the robust, silver-free results:
   - a sentence classifier trained on the teacher's sentence answers ranks pairs at AUPRC 0.794, against 0.923
     for the pair verifier on the 437 rows;
   - on BioRED a perfect sentence filter has pair precision 0.536 [0.527, 0.544].
   These strengthen the submission's thesis and pre-empt the "unfair sentence baseline" objection without any
   provisional number.

Which human labels the reframe needs:
- **Minimum, and decisive: the 191 pair-negative passages.** Only these rows have LLM-decided sentence labels,
  and the sentence-label model's whole advantage lies on them.
  - Annotate them blind, inside the existing randomised 437-row sheet
    (`data/evaluation/second_annotation_2026-10-04_BLIND.xlsx`), so that the annotator cannot tell which rows
    are pair-negative.
  - In practice that means answering the sheet's rows in order. That covers 177 of the 191. The other 14 are
    among the 20 SKIP rows, which hold the non-blind gold-review items. This count is computed from the
    rows-only key and `clean_benchmark()` labels, assuming `bench_row` is the 0-based benchmark index.
- **Recommended: the full 417 non-SKIP rows** (SENTENCE and PAIR).
  - This also checks the assumption "pair-positive ⇒ sentence-positive" on the 246. Item 4 of the spot check
    already shows the annotator can be inconsistent here.
  - It gives an independent pair re-grading.
  - At about 30 s a row this is roughly 3.5 hours.
  - Keep the 19 non-blind spot-check answers out of the benchmark, or report them separately.
- **For credibility of sentence labels:** a second annotator on a random 100 of the 417, reporting κ.
  Sentence-level judgements are known to vary between annotators; the user said so ("several annotators blindly
  have different responses").
- **Then:** re-run the analysis (ANALYSIS §1 command), and decide the paper by the outcome:
  - **If the human labels reverse or erase the sentence-label model's edge on the pair-negative passages**,
    adopt the reframe, re-titled around the sentence decision, e.g. "Ask about the pair, even for the
    sentence". It would then be a stronger paper than the submission.
  - **If they confirm the edge**, the honest paper is the submission, plus the short "one teacher, two
    questions" result as a section or appendix. The sentence-label model is then itself the deliverable for
    the sentence-level goal, and may deserve its own short paper.
