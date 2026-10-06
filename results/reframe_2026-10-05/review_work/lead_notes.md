# Lead reviewer's own notes (05:15-05:20)

Format (rebuilt in /tmp/rv_build from the writer's tex, unchanged):
- 0 LaTeX errors, 0 overfull boxes, no undefined citations/references after 4 passes.
- BibTeX reports "1 error message": "Illegal, another \bibstyle command" (acl.sty and the tex both set acl_natbib).
  The submission's paperA.blg has the same message. Harmless.
- Conclusion ends at the bottom of page 8; Limitations start page 9; 21 pages.
- Abstract: 196 words by the submission's counter (LaTeX stripped, math = 1 word, pure punctuation tokens such as
  "--" excluded, code line included); the same counter gives 194 for the submission. 208 if "--" range dashes
  are counted as words (199 for the submission). Under 200 by the convention the submission used.
- Review mode: \usepackage[review]{acl}; line numbers present in the PDF.
- PDF Info: Author/Title/Subject/Keywords empty; custom key only PTEX.Fullbanner.
- Anonymity grep (tex non-comment lines + pdftotext): no author names, affiliations, e-mails; names only as authors
  of third-party references (gobeill2020sibils, ruch2024biotxplorer, britan2018nexta5), all three already cited by
  the submission. Only URL: https://anonymous.4open.science/r/XXXX (placeholder inherited from the submission).
- BiotXplorer: 3 mentions (Setting, lines 106, 115-116, 144); the paragraphs are identical to the submission's.

Content (own read of intro, §4.3, conclusion, Limitations):
- A1 (biodiv_sentlab.json) is integrated: table row, abstract, intro, §4.3, conclusion, Limitations.
- Claims are narrowed to "ask each question for its own decision; one teacher can label both".
- Rows needed for the biodiversity sentence question: 191 pair-negative passages; 14 of them are gold-review
  (SKIP) rows already answered (non-blind); 177 non-SKIP pair-negative rows of the second-annotation sheet need a
  SENTENCE answer (column F). Computed from benchmark pair gold + the rows-only key; sealed key not opened.
