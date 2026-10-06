# Stage: writer (after analysis and literature)

Goal: write the reframed paper as a SEPARATE draft, for the user to compare with the submission
version and decide. The reframe changes the emphasis and adds a result. It is not a new paper:
keep everything in `paperA.tex` that remains true.

Read: CONTEXT.md; `state/analysis.done`; `results/reframe_2026-10-05/ANALYSIS.md`;
`state/literature.done`; `results/reframe_2026-10-05/LITERATURE.md`; `paperA/paperA.tex` in full;
`state/gpu_sentlab.done` or `.failed`, and `results/paperA_v2/sentence_level/biodiv_sentlab.json` if
it exists. If neither GPU marker exists yet, write the draft without it, and look again before
finishing (see step 4).

## 1. The draft
Write `paperA/paperA_reframe.tex`, starting from a copy of `paperA.tex` in the same directory so that
figures and style files resolve. Use `\bibliography{references,reframe_extra}`. Compile it to
`paperA/paperA_reframe.pdf`: `timeout 120 pdflatex`, then bibtex, then pdflatex twice.
- Story: the application needs a SENTENCE-level decision: does this passage describe a biotic
  interaction? That is what curators triage, and the user's curation was done in that spirit. The
  database also needs the pair. The finding: ask about the pair, even if you want the sentence.
  Pair-conditioned verification, aggregated over the passage's candidate pairs, answers the
  sentence question at least as well as a model trained on it, and it also yields the pair, which
  a sentence filter cannot (the precision ceiling). The existing controlled comparison stays as the
  mechanism, the evidence that conditioning on the pair is what helps.
- Scope every claim to the evidence in ANALYSIS.md's section "What the evidence supports and what
  it does not".
  - BioRED's sentence labels are derived exactly from gold, but mean "co-mentions a related pair":
    say exactly that.
  - The biodiversity sentence-level numbers are PROVISIONAL silver labels unless human labels exist:
    say so where they appear, and in the Limitations.
  - If the fair biodiversity competitor (A1) matches or beats pair-max, the claim must be narrowed
    to match. Do not bury it.
  - Acknowledge the multi-instance "at least one" assumption as prior work (LITERATURE.md), and
    state the novelty the way LITERATURE.md's novelty statement and its "claims we must not make"
    allow.
- Title: pick one, and list two alternatives in the notes. "Which Pair Is It About?" may stay if it
  still fits.
- Abstract of at most 200 words (count it). Rewrite the introduction around the sentence-level
  target. Add a sentence-level results subsection with one compact table: BioRED rows from exact
  gold, and the biodiversity rows, provisional, including the fair competitor if available. Fold
  the precision-ceiling argument into it. Update the related work from LITERATURE.md's draft
  paragraphs, the conclusion and the Limitations.
- BiotXplorer is the application only: where verified output goes and why the sentence decision
  matters, said from public sources. It is not a contribution, and nothing about its internals.
  Preserve anonymity.
- Every new or changed number must come from a file under `results/` (mostly
  `results/paperA_v2/sentence_level/*.json`), with the same rounding conventions as the paper.
- Format, as for the submission version: the body ends on page 8 (Conclusion on page 8; the
  Limitations, references and appendices follow); 0 LaTeX errors; 0 overfull boxes (use `grep -a`
  on the log); review mode with line numbers. To make room, move material to the appendices rather
  than shrinking fonts or spacing.

Before writing, check `scripts/sentence_level/common.py`. If GOLD_SHEET still points at the BLIND
copy, point it at `data/evaluation/gold_review_2026-10-02_v2_eg_curated.xlsx` and re-run the
human-label evaluation (the command is in ANALYSIS.md).

Gold review (CONTEXT.md §4, UPDATE 02:45). The draft keeps the current gold: the 7 proposed flips
are the user's decision. If the draft mentions the 22-item review at all, call it an expert
adjudication of a pre-filled sheet, never blind. Report the what-if result
(`results/reframe_2026-10-05/gold_review_22_whatif.json`) only in REFRAME_NOTES.md.

## 2. `paperA/REFRAME_NOTES.md`
- What changed, section by section, and why.
- The old and the new abstract, side by side.
- Every new number, with its source file and JSON key.
- The open decisions for the user:
  - adopt the reframe or keep the submission version;
  - the title;
  - whether to wait for human sentence labels, and how many second-annotation rows the
    biodiversity sentence-level claim needs;
  - anything else a human must decide.

## 3. Commit (local only, no push)
Add by explicit path:
- `scripts/sentence_level/`;
- the sandbox scripts;
- `results/paperA_v2/sentence_level/*.json`;
- the reframe folder's markdown, prompts and shell scripts, excluding `logs/` and `state/*.lock`;
- `paperA/paperA_reframe.tex`, `paperA_reframe.pdf`, `reframe_extra.bib` and `REFRAME_NOTES.md`.

Check every file's size first (none over 10 MB). Do not add data CSVs or models. The subject is
something like "Paper A: sentence-level reframe draft (separate from the submission version)".
The rules forbid any attribution line.

## 4. Before writing `state/writer.done`
Look at the GPU markers again. If `gpu_sentlab.done` appeared after you wrote the draft, integrate
its numbers now: the table, the claims, the notes. If it is still missing, say so in
`writer.done` so that the reviewer integrates it.
