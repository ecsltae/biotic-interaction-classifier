# Stage: reviewer (last)

Goal: check the draft independently before the user reads it, fix what is clearly wrong, give a
recommendation, and e-mail the report. You did not write the draft. Read it as a demanding ACL area
chair would, and as the user's careful co-author.

Read: CONTEXT.md, every `.done` file, ANALYSIS.md, LITERATURE.md, `paperA/REFRAME_NOTES.md`,
`paperA/paperA_reframe.tex` and its PDF, and `paperA/paperA.tex` (the submission version) for
comparison. Up to 3 subagents may work in parallel, for example on the number audit, the citation
check and the critical review. You own the final judgement.

1. GPU result. If `state/gpu_sentlab.done` exists and the draft does not yet use
   `results/paperA_v2/sentence_level/biodiv_sentlab.json`, integrate it first: the table, the
   claims, the notes. If the fair competitor changes the conclusion, narrow the claims to match. If
   the job failed or is still missing, the draft must say that this comparison is pending.
2. Number audit. Trace every number in `paperA_reframe.tex` that is new or differs from
   `paperA.tex` to a result file (adapt `scripts/audit_paper_numbers.py` if that is practical).
   Fix or remove every number you cannot trace. List what you checked in REVIEW.md.
3. Citations.
   - Every `\cite` key must resolve.
   - Run `python3 scripts/check_references.py paperA/reframe_extra.bib paperA/paperA_reframe.aux`.
   - Remove any reference that cannot be verified.
   - Check that each new citation supports the sentence it is attached to, by reading its abstract.
4. Format.
   - Compiles with 0 errors and 0 overfull boxes (`grep -a`).
   - The body ends on page 8; the abstract has at most 200 words.
   - Anonymity: grep the tex and the PDF text for names, affiliations, e-mail addresses and URLs,
     and `pdfinfo` the metadata.
   - Line numbers are on (review mode).
5. Critical review. Write `results/reframe_2026-10-05/REVIEW.md` with:
   - an ACL-style review (summary, strengths, weaknesses, questions, a score with its reason);
   - is the sentence-level claim supported, as stated, on each benchmark?
   - is the multi-instance relation acknowledged, and is the novelty stated defensibly?
   - is the provisional biodiversity evidence flagged honestly?
   - is the BiotXplorer framing safe (application only, no overlap, anonymous)?
   - would this draft fare better or worse with reviewers than the submission version, and why?
   - a clear RECOMMENDATION: adopt the reframe now / adopt it after the human sentence labels (say
     how many rows and which) / keep the submission version. Give reasons.
6. Fix what is clearly wrong in the draft: numbers, citations, overclaims, format, anonymity.
   Recompile. Leave judgement calls to the user, listed in REVIEW.md. Commit the fixes locally with
   explicit paths; do not push, and add no attribution.
7. Memory. Add a short paragraph to
   `~/.claude-hesso/projects/-home-USER-MetaP/memory/project_paper_a.md`: the
   reframe draft exists, where it is, the key sentence-level numbers, the recommendation, and what
   waits on the user. Keep the file's style, and do not delete other content.
8. E-mail the report with `bash ~/MetaP/classifier/scripts/notify.sh "Paper A: sentence-level reframe draft ready for your review" "<body>"`.
   The body is plain text, 20–40 lines:
   - what was done overnight;
   - the key sentence-level numbers (BioRED with CIs; biodiversity provisional; the fair
     competitor);
   - the recommendation, in two sentences;
   - the files to open (absolute paths: the draft PDF, REFRAME_NOTES.md, REVIEW.md, ANALYSIS.md,
     gold_review_22.md);
   - one line on the gold review: PAIR agreement with gold, the 7 proposed flips (applied: no)
     and what they would change;
   - what needs the user: the 22-item review, SENTENCE labels in the second annotation, and the
     decisions listed in REVIEW.md.
   
   Then write `state/reviewer.done`.
