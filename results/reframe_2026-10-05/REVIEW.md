# Paper A, sentence-level reframe: review (reviewer stage, 2026-10-05 05:14-06:30)

Draft reviewed: `paperA/paperA_reframe.tex` / `.pdf`, as committed by the writer in b078ccf
(md5 `b481aa00…`, backed up at `review_work/paperA_reframe.before_review.{tex,pdf}`). The submission version,
`paperA/paperA.tex`, was read for comparison and not touched.

The detailed reports behind this file are in `results/reframe_2026-10-05/review_work/`. Three read-only subagents
wrote them independently, and I checked their main points before acting on them:
- `number_audit.md`: every new or changed number, with source file, JSON key and value;
- `citations.md` (+ `citations_refsbib_part.md`, `check_references_*.txt`): key resolution, the checker, and
  whether each new citation supports its sentence;
- `critical_review.md`: an area-chair review, with must-fix items and judgement calls;
- `lead_notes.md`: my own format, anonymity and row-count checks.

Bottom line: the draft is careful and its numbers are right. It is honest about a result that goes against the
pair route on our own benchmark. Its weakness is that the result it adds rests on provisional, partly circular
labels. **Recommendation: adopt the reframe only after the human sentence labels exist (section 6).**

---

## 1. GPU result (A1, fair sentence competitor)

`state/gpu_sentlab.done` exists (04:26). The writer used `biodiv_sentlab.json` from the start:
- the "Sentence-label model" row of Table `tab:sentence`;
- the abstract, the intro, §4.3 "Biodiversity, provisional", the conclusion and the Limitations.

The claims were already narrowed to match: the sentence-label model beats pair-max on biodiversity under all three
silver definitions, and the draft says so everywhere. No action was needed.

## 2. Number audit

- **Scope:** every number in `paperA_reframe.tex` that is new, or that moved into a different claim, relative to
  `paperA.tex`. Numbers unchanged in the same claim were already audited (`NUMBER_AUDIT_2026-10-04.md`).
- **Method:** each value was read from the per-file JSON, never from REFRAME_NOTES or `sentence_tables.json`.
  Rounding was checked by script (`review_work/audit_reframe_numbers.py`). The context was also checked:
  benchmark, arm, silver definition, metric, kind of interval, n.
- **Result:** 92 claims (about 190 printed values). 91 OK, 0 wrong, 0 untraceable, 1 in the wrong context.
  - The wrong context: "a 5-gram Jaccard scan against all 48,338 training passages". These are rows; the corpus has
    34,242 distinct passages, a number the draft now prints elsewhere. **Fixed.**
- **What was checked**, by group (full table in `number_audit.md`):
  - Abstract and intro:
    - 0.965 / 0.930 / 0.955, F1 0.954 vs 0.930 (BioRED);
    - 0.985-0.987 vs 0.966-0.977 (biodiversity);
    - 0.536 and 0.724 (perfect-filter pair precision);
    - 34.5% (the teacher says no to the sentence question).
  - §4.3 and Table `tab:sentence`:
    - every cell;
    - the "< .007" SD bound (largest per-seed SD 0.0063);
    - BioRED +0.035 [+0.024, +0.047], F1 +0.024 [+0.017, +0.031], 158/41, p = 2.1e-17,
      +0.010 [-0.001, +0.021] (abstract clusters), 777 single-pair sentences +0.044, multi-pair +0.006;
    - zero-shot 0.959 vs 0.908, +0.051 [+0.025, +0.079];
    - biodiversity +0.014 / +0.016 / +0.008 with their intervals; 0.020 / 0.010 / 0.012; 0.929 vs 0.897;
      46/22, p = 0.0049;
    - the decomposition 0.953 / 0.788 / 0.658;
    - gold-positive-only -0.004 [-0.014, +0.005];
    - 0.536 [0.527, 0.544], 0.516, 0.724 [0.674, 0.768], 0.699-0.794;
    - pair-gold AUPRC 0.794 / 0.840 / 0.923;
    - 41% and kappa 0.26.
  - Labels: 2,582 sentences, 0.838; 246 / 191 / 394 / 43 / 148; 34,242 passages, 65.5%; 14.3 pairs per passage.
  - Limitations:
    - the spot check: 13/14 (kappa 0.81), 17/19 (0.68), 15/19 (0.46), with 5 pair-positive;
    - 85.9 vs 3 forward passes; 6 passes; 0.003-0.006.
  - BioRED limitations: 15,640 vs 3,277 training rows.
  - Appendices: 2 h 15 min labelling time (8,081 s in the log); moved scaling and encoder numbers.
- **Known ANALYSIS.md errors:** neither reached the draft. The draft prints 41% (0.407), not "a third", and it
  calls 0.724 provisional, with its 0.699-0.794 range.
- **Cosmetic:** "by 0.020" AUPRC (0.0195 in the JSON) against the table's rounded .985 - .966 = .019. Left as is.

## 3. Citations

- **Resolution:** all 80 cited keys resolve (58 from `references.bib`, 22 from `reframe_extra.bib` after the fix
  below). There is no undefined citation.
  - The `.blg` "1 error message" is `Illegal, another \bibstyle`: acl.sty and the tex both set acl_natbib.
    `paperA.blg` (submission) has the same message. It is harmless.
- **`check_references.py`** (`review_work/check_references_final.txt`, final aux): 20 OK, 0 CHECK, 2 NOT FOUND.
  - The two NOT FOUND entries, `zhong2021frustratingly` (NAACL 2021) and `dasanmartino2019fine` (EMNLP 2019),
    were verified by hand on the ACL Anthology and Crossref. The fields match.
  - No entry had to be removed.
- **Support:** every new citation was checked against its abstract or text (table in `citations.md`).
  - The draft breaks none of the 20 "claims we must not make" in LITERATURE.md.
  - Three citations did not support their sentence. **Fixed:**
    1. BioRED EP F1, `lai2023biorex,lai2025bioredirect`: "in part because a quarter of the related pairs are
       never co-mentioned" read as the cited papers' explanation of their own scores. It is our measurement.
       Now: "...; part of our gap is structural, as a quarter of the related pairs are never co-mentioned in one
       sentence (Appendix)".
    2. Related work, LLMs: `dsouza2025mining` extracts species, locations and habitats, not interactions, and
       `farrell2024landscape` is a review. Now: "Large language models mine ecological text [Farrell; D'Souza] and
       extract interactions at scale [Keck; Zou]".
    3. `farkas2010conll`: CoNLL-2010 is not "biomedical hedge detection" (it has a Wikipedia track), and only the
       Wikipedia winner was a sentence classifier. Now: "in the CoNLL-2010 hedge task ... a direct sentence
       classifier won on Wikipedia".
  - Also applied:
    - `\citep[cf.][]{li2021dual}` at both places. That paper is about whole-slide images; the BioRED expectation is
      our analogy, not its finding.
    - `zeng2015distant` added next to `hoffmann2011knowledge` for "a maximum over instances". It is verified and
      was already in the bib.
- **Not changed (inherited from the submission; optional):** "\citet{zou2026llm} ... without checking extractions
  against their text" could read as "never validated". They do score the LLM on hand-labelled sets.

## 4. Format

Checked on the final build:
- **Compilation:** 0 LaTeX errors, 0 overfull boxes (`grep -a`), no undefined references or citations.
- **Pages:** 21 in all. The Conclusion ends on the last line of page 8 and the Limitations start on page 9.
  - My first fixes pushed the Conclusion 4 lines onto page 9.
  - I recovered the space by trimming five paragraph-final short lines, without dropping any content: contribution
    1; the end of the Benchmark paragraph; the blocks paragraph ("everywhere", "whole", "rather than"); the error
    paragraph ("A few more read as gold errors (Limitations)."); §4.3 Labels.
  - Any further body edit must be length-neutral, or the Conclusion moves to page 9.
- **Abstract:** 197 words. The count uses the submission's convention: LaTeX stripped, a math group is one word,
  "--" ranges are not counted as words, the code line is included. The same counter gives 194 for the submission.
- **Review mode:** `\usepackage[review]{acl}`, so line numbers are on.
- **Anonymity:**
  - The tex (non-comment lines) and the PDF text were grepped for author names, affiliations, e-mail addresses and
    URLs. The only names are authors of three third-party references, all already cited by the submission. The
    only URL is the `anonymous.4open.science/r/XXXX` placeholder.
  - `pdfinfo` and the raw Info dictionary: Author, Title, Subject and Keywords are empty; the only custom key is
    PTEX.Fullbanner.
- **BiotXplorer** appears in three places (Setting, "Where candidates come from"; the 31% sentence; the Biotx100
  description). These paragraphs are word for word the submission's.
  - It is the application and the source of candidates, not a baseline and not the novelty.
  - Nothing about its internals goes beyond the public description, and no person or working relationship is named.
  - The inherited wording that hints at privileged access is listed under judgement calls (J5).

## 5. Critical review (ACL style)

### Summary
The paper studies the filter step of co-occurrence literature mining for biotic interactions. That step involves two
decisions: does the passage describe an interaction, and do these two organisms interact.

Pair level (unchanged from the submission): one BiomedBERT-base cross-encoder is distilled three times from one
48k-row corpus labelled by Qwen3-32B, and only the input differs.
- Conditioning on the pair raises AUPRC from 0.851 to 0.918 on a 437-row expert benchmark, and from 0.738 to 0.843
  on BioRED (BC8 protocol).
- The gain sits where passages name more than two entities.
- The relation term adds nothing to ranking.

New, §4.3: the pair verifier is read at the sentence level as a maximum over pairs ("pair-max") and compared with
two sentence classifiers.
- On BioRED, with gold-derived "co-mentions a related pair" labels:
  - pair-max beats the candidate-trained sentence arm;
  - it at least matches a model trained on sentence labels (F1 +0.024, significant; AUPRC +0.010, not significant).
- On the biodiversity benchmark, with provisional silver labels: a model distilled from the teacher's sentence
  answers beats pair-max under all three definitions. The edge lies entirely on LLM-labelled passages, so it is
  partly circular.
- A perfect sentence filter has pair precision 0.536 on BioRED and, provisionally, 0.724 here.
- The paper concludes that each question should be asked for its own decision, and that one teacher can label both.

### Strengths
1. The pair-level comparison is controlled and carefully reported. Thresholds are pre-specified or block-held-out,
   tests are paired, and the benchmark's history is disclosed. The BioRED replication follows a published protocol
   and finds the gain in the predicted stratum.
2. The sentence-level section is unusually honest.
   - It reports a loss to a fair competitor on the authors' own benchmark and locates it: all of it on the 191
     pair-negative passages.
   - It calls the loss partly circular.
   - It reports that the BioRED edge is in single-pair sentences, against the reframe's own intuition.
3. Two results do not depend on silver labels, and they pre-empt the "unfair sentence baseline" objection to the
   pair-level table:
   - sentence scores are poor pair filters (pair-gold AUPRC 0.794 for the sentence-label model, against 0.923);
   - on BioRED a perfect sentence filter has pair precision 0.536.
4. The multi-instance link is credited to prior work three times (Dietterich; Hoffmann and Zeng; Ilse, Li, Rei,
   Da San Martino, Magge, Farkas), with "We claim none of these". No "first" or "novel" claim appears anywhere.

### Weaknesses
1. **On the authors' own benchmark, the new evidence supports no positive claim for the method.** The best sentence
   filter there is the sentence-label model. On BioRED, a pair-route win is expected by construction (the label is
   the OR of the pair labels, as the draft says), the task is near ceiling (base rate 0.838), and pair-max also gets
   gold entity spans and 4.8 times the training rows.
2. **The biodiversity sentence labels are silver and partly circular.**
   - 191 of the 437 are LLM answers to the sentence prompt.
   - The majority definition includes the teacher that the winning model imitates.
   - The "non-teacher" voters are Qwen models too. This is now stated in the Limitations.
3. **It reads as two papers.** The title, §2 to §4.2, §5 and §6 are the pair paper. §4.3 adds a sentence study
   whose provisional outcome runs against the title's method. The user's goal is the sentence decision, and the
   evidence so far favours a sentence classifier for it.
4. **There are no human sentence labels.** The 437-row sheet has 0 of 417 rows answered. The 19 human sentence
   answers come from a non-blind adjudication of a pre-filled sheet, and 11 of its 22 items were selected by
   model-gold disagreement.
5. **Inherited:** the anonymity-sensitive wording about the retrieval pool, and the `XXXX` code URL.

### Questions to the authors
1. On the 191 pair-negative passages, do human sentence labels agree with the LLMs (and so with the sentence-label
   model) or with pair-max? This decides §4.3.
2. On BioRED, does a sentence model given all entity mentions close the gap to pair-max? If it does, the BioRED
   edge is entity information, not the pair question.
3. Is a development-tuned combination of the two routes better than either?
4. Who adjudicated the 22 items: the original grader or a second person?

### Score
- **Overall 2.5/5 (borderline Findings).** Soundness 3, Excitement 2.5, Reproducibility 4.
- The pair-level part alone would sit at about 3. The sentence part lowers the score, for three reasons:
  - its own-benchmark half is provisional and partly circular;
  - its BioRED half is expected by construction;
  - its take-home message ("ask each question") is unsurprising and, on current evidence, favours the baseline
    formulation for the sentence decision.
- With human sentence labels on the pair-negative passages, whichever way they fall, the section would become a
  real result. I would expect about 3 to 3.5.
- The critical-review subagent reached the same score independently (2.5 for the reframe, about 3.0 for the
  submission).

### Is the sentence-level claim supported as stated?
- **BioRED: yes, as now worded.**
  - "Beats the candidate-trained arm": AUPRC +0.035, CI excludes 0 with abstract clusters.
  - "At least as good as the sentence-label model, and better at its operating point": F1 +0.024, McNemar
    p = 2.1e-17; AUPRC +0.010, not significant, and the draft says so.
  - The labels are now called "pair-derived" in the abstract and conclusion.
  - One sentence said the "two benchmarks" measure how far the scored pairs fall short of a passage's interactions.
    On BioRED they cannot fall short by construction. **Fixed** to "our benchmark".
- **Biodiversity: yes, as a provisional statement.**
  - The loss to the sentence-label model is stated correctly and flagged "provisional" in the abstract, intro,
    §4.3 heading, table header and Limitations.
  - Three places lacked the flag; **fixed**:
    - the contributions ("(ours provisionally)");
    - the Setting ("its provisional sentence labels");
    - the conclusion ("provisionally on ours").
  - Two causal sentences turned a two-benchmark observation into a mechanism ("the outcome depends on what the
    sentence label counts"). The benchmarks differ in label source, training labels, pair source, domain and base
    rate. **Fixed** to "differs between the two benchmarks".
  - The closing recommendation read as final, but its sentence half rests only on the provisional result (BioRED
    points the other way). **Fixed** in the abstract and the conclusion: "Pending human sentence labels, ask each
    question separately; one teacher labels both."

### Multi-instance relation and novelty
- The multi-instance relation is acknowledged in the intro, §4.3 and related work.
- The novelty is stated defensibly: a controlled measurement with the passages and encoder fixed (and, on our
  benchmark, the teacher), only the question changed. There is no novelty claim for the max, the pair input or
  sentence filtering.
- After the honest negative, the sentence-level novelty is thin but not overclaimed.

### Is the provisional biodiversity evidence flagged honestly?
Yes, after the three flag fixes above:
- the Limitations paragraph "The biodiversity sentence labels are provisional" is thorough;
- the spot check is described as non-blind and selected, and is never used as a benchmark result;
- the 7 proposed gold flips appear only in REFRAME_NOTES, not in the draft.

### Is the BiotXplorer framing safe?
Mostly, and no worse than the submission:
- application and candidate source only;
- public facts only (31%, the stated error cause);
- third-person citation; no names, no internals.

Remaining risks are inherited and listed as judgement calls J5 and J6:
- privileged-access wording;
- the provenance of the benchmark grading;
- a curator's initials in a file name in the anonymous mirror.

### Better or worse with reviewers than the submission?
**Worse on balance today, though more honest.**

What the reframe gains:
- It drops a claim that the new evidence contradicts.
  - The submission says the sentence question "is answered before it is asked" (abstract), and that a sentence
    classifier "is left with little to decide" (conclusion).
  - In fact the teacher answers that question "no" on 34.5% of the corpus passages, and a sentence-label model
    reaches AUPRC 0.985 against a base rate of 0.778. (The submission's intro qualifies "lexically"; its abstract
    and conclusion do not.)
- It pre-empts the unfair-baseline objection.
- It speaks to the curation (sentence) goal.

What it loses:
- A sharp thesis becomes "it depends".
- It adds a section that reviewers will read as "results pending human labels".
- Its BioRED half is announced as expected by construction.
- The operating curve and CPU configuration move to appendices.

---

## 6. RECOMMENDATION

**Adopt the reframe after the human sentence labels exist, not before. Until then, the submission version is the
one to submit, with its "answered before it is asked" sentences softened (your decision; I did not touch
paperA.tex).**

Reasons:
- The reframe's new result on our own benchmark is provisional and partly circular. Its BioRED result is expected
  by construction. Submitted now, it would likely score below the submission (about 2.5 against 3.0).
- Human sentence labels settle it either way, and either outcome makes the reframe a sound paper:
  - **If they erase or reverse the sentence-label model's edge:** the pair route is the best sentence filter as
    well, and the reframe can say "ask about the pair, even for the sentence".
  - **If they confirm the edge:** "ask each question for its own decision; one teacher labels both" becomes a
    supported conclusion, and the sentence-label model is the deliverable for your sentence-level goal.
- The reframe matches your stated intention (the sentence decision) better than the submission does. The
  submission also carries a sentence that the new data contradict.

Which labels are needed:
- **Minimum, and decisive: SENTENCE answers (column F) on the pair-negative passages.**
  - 191 passages have an LLM-decided silver label. 14 of them are SKIP rows (gold-review items, already answered,
    non-blind).
  - That leaves **177 rows** of `data/evaluation/second_annotation_2026-10-04_BLIND.xlsx` to answer.
  - Answer the sheet in its randomised order, without looking up which rows these are, so the labels stay blind.
    The 177 are mixed among the 417 non-SKIP rows.
  - Computed from the benchmark's pair gold and the rows-only key; the sealed key was not used.
- **Recommended: all 417 non-SKIP rows, SENTENCE and PAIR** (about 3.5 h at 30 s a row).
  - This also checks "pair-positive implies sentence-positive" on the 246 positives (item 4 of the gold review,
    PAIR YES with SENTENCE UNSURE, shows that answers can disagree) and gives an independent pair re-grading.
- **For credibility:** a second annotator on a random 100 rows, with kappa.
- **Then:** run `python3 scripts/sentence_level/make_sentence_tables.py`. It writes `biodiv_human.json` and
  recomputes the 0.724 ceiling.

If a submission deadline comes before the labels:
- submit `paperA.tex`, with the softening patch (J1);
- optionally add the two robust, silver-free sentences: pair-gold AUPRC 0.794 for the sentence-label model against
  0.923, and the BioRED perfect-filter precision 0.536 [0.527, 0.544].

---

## 7. Fixes applied to the draft (local commit, no push)

All fixes are in `paperA/paperA_reframe.tex`. The PDF was rebuilt, and the full word diff is
`git diff b078ccf -- paperA/paperA_reframe.tex`.

| # | where | change | source |
|---|---|---|---|
| 1 | Evaluation protocol | "48,338 training passages" → "the passages of all 48,338 training rows" | number audit |
| 2 | §4.2, EP F1 | the gap explanation is now ours, not the cited systems' | citations M1 |
| 3 | Related work | LLMs "mine ecological text [Farrell; D'Souza] and extract interactions at scale [Keck; Zou]" | citations M2 |
| 4 | Related work | CoNLL-2010 hedge task; "a direct sentence classifier" | citations M3 |
| 5 | §4.3 Labels; Limitations | `\citep[cf.][]{li2021dual}` | citations R1 |
| 6 | Related work | + `zeng2015distant` for the maximum over instances | citations R2 |
| 7 | §4.3 | gold-positive-only comparison: "indistinguishable" → "close" (pair-own edges pair-max under majority labels) | number audit N2 |
| 8 | Abstract | "provisional, partly LLM-derived labels" | number audit N3 |
| 9 | Intro; Related work | "depends on what the sentence label counts" → "differs between the two benchmarks" | review M1, M2 |
| 10 | §4.3 | "what the two benchmarks measure" → "what our benchmark measures" | review M3 |
| 11 | Abstract; Conclusion | closing line → "Pending human sentence labels, ask each question separately; one teacher labels both." | review M4 |
| 12 | Setting; Contributions; Conclusion | "provisional" flags | review M5-M7 |
| 13 | Abstract; Conclusion | BioRED's "pair-derived" sentence labels | review J1 |
| 14 | Table `tab:sentence` caption | "its ceiling" → "its perfect-filter precision" | review J7 |
| 15 | Limitations | the non-teacher voters are Qwen models too, so shared family errors are not excluded | review J3 |
| 16 | five paragraphs | widow trims to keep the Conclusion on page 8 (no content removed) | format |

Not applied:
- the `XXXX` anonymous URL (needs the real repository ID; the submission has the same placeholder);
- everything in section 8.

## 8. Judgement calls for you

1. **J1, the submission version** (`paperA.tex`, untouched). Its abstract ("so that question is answered before it
   is asked") and its conclusion ("answered before it is asked, and a classifier built for it is left with little
   to decide") are contradicted semantically by the new data: the teacher says no on 34.5% of corpus passages, and
   the sentence-label model reaches AUPRC 0.985. Suggested, the review subagent's wording:
   - abstract: "so that question, read lexically, is answered before it is asked";
   - conclusion: "...is, lexically, answered before it is asked, and a classifier built for it cannot say which
     pair".
2. **J2, adopt or not, and the title**: see section 6, and the alternatives in REFRAME_NOTES.md section 6.
3. **J3, the closing line.** I applied "Pending human sentence labels, ask each question separately; one teacher
   labels both." Once the labels exist, change it to whichever conclusion they support.
4. **J4, the 22-item gold review in the paper.** The draft reports its 19 SENTENCE answers (spot check) but not its
   PAIR outcome: it sides with the models on 7 of 11 selected items and with the gold on 11 of 11 controls. Report
   that consistently, or report neither. Say who adjudicated. The 7 flips remain your decision.
5. **J5, inherited wording that hints at privileged access** (same in the submission):
   - "the pipeline we study";
   - "We pulled the retrieval pool in full";
   - Reject50 as "candidates the retrieval pipeline discarded";
   - "We take the diagnosis as given and question only the remedy". A neutral alternative: "We test an alternative
     to filtering such passages out."
6. **J6, provenance and anonymity of the benchmark grading.**
   - Biotx100's design (100 triples, half GloBI-matched, half random) echoes the system authors' published
     evaluation. Make sure the text implies no working relationship.
   - The anonymous mirror ships `data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv`, whose
     name appears to carry a curator's initials. Consider renaming it in the mirror. I did not touch the mirror.
7. **J7, Qwen3.8-27B as a silver voter.** State its licence in the Artifacts appendix, or drop the majority
   definition. The conclusion is the same under the other two.
8. **J8, "A second annotation ... is in progress"** (Limitations). True only in that the sheet exists: 0 of 417 rows
   are answered.
9. **J9, naming.** Table 1's "Sentence-level" row and Table `tab:sentence`'s "Sentence arm, candidate labels" are
   the same model, and the "sentence-label model" is a different one. Consider "Sentence-level (candidate labels)"
   in Table 1.
10. **J10, two untested controls** (Limitations), each a few four-minute runs:
    - a BioRED sentence model with all entity mentions marked: it decides whether the BioRED result is about the
      pair question at all;
    - a development-tuned combination of the two routes.
11. **J11, the `XXXX` code URL**, in both versions: fill it before any upload.
12. **J12, page budget.** The Conclusion ends on the last line of page 8. Every further body edit must be
    length-neutral.
