# Paper A: submission sheet (ACL Rolling Review), 2026-10-04

## State of the PDF (`paperA/paperA.pdf`)

- Restructured on 2026-10-06 to the redesign blueprint (new title, abstract, section order, one body
  table and four body figures). The four figures are in, each built by a script in `paperA/fig/`
  (`make_{concept,curves,where,scaling}_figure.py`) that reads only the result files, asserts the
  values the paper prints and writes them to a JSON next to the figure. 19 pages: 8-page body (the
  Conclusion ends on page 8, about 7 lines short of the limit), Limitations, references,
  appendices. Figure 1 is on page 1, Table 1 and Figure 2 on page 5, Figures 3 and 4 on page 6; no
  body float comes after the Limitations heading.
  Review mode with line numbers, author "Anonymous ACL submission", no title or author in the PDF
  metadata (the figures carry none either).
- ACL pubcheck (`~/opt/aclpubcheck-venv`), rerun on the integrated PDF: **All Clear** (page size,
  page limit, fonts). Its margin test renders through ImageMagick, which is not installed, and
  without it the run crashes, so it was run with only the margin test stubbed out. Margins measured
  instead from glyph and graphics positions on all 19 pages: 2.43–2.50 cm at the sides (microtype's
  protrusion of hyphens and punctuation), 2.57–2.77 cm at the top, the four figures exactly on the
  2.5 cm text edges, nothing outside the text area except line and page numbers.
- 0 LaTeX errors, 0 overfull boxes, no undefined references, all fonts embedded, no Type 3 fonts
  (the figures embed Liberation Serif as TrueType), A4.
- Abstract: 193 words (ACL limit 200); no URL (the code link is a footnote on the Contributions list).
- Numbers: `scripts/audit_paper_numbers.py` on the final paper (2026-10-06, after the verification
  fixes): 957 numbers checked, 52 not explained by a result file, all of the hand-checked kinds
  (corpus sizes, model names, citation years, the 22/150 interval, BioRED protocol counts, compute
  figures, the historical rows of Table negative); see `results/paperA_v2/NUMBER_AUDIT_2026-10-06.md`.
- References: all 58 cited references verified to exist (Semantic Scholar, Crossref) with the cited
  first author and year (`scripts/check_references.py`). The three year differences are
  preprint vs. published versions; the bib cites the published ones.

## To do before uploading (you)

1. **Anonymous code link.** Publish the mirror through anonymous.4open.science (steps in
   `results/overnight_2026-10-03/ANONYMOUS_4OPEN_SCIENCE.md`), then replace `XXXX` in `paperA.tex`
   (search `ANONYMOUS LINK`: a footnote at the end of the Contributions list) and rebuild:
   `pdflatex paperA; bibtex paperA; pdflatex paperA; pdflatex paperA; pdflatex paperA` (with `;`,
   not `&&`: bibtex exits 2 on a harmless duplicate-\bibstyle message, which would stop an `&&`
   chain). Keep the fourth pdflatex pass: from a clean directory the third pass still leaves a
   "Rerun" warning in `paperA.log` and misplaces some review line numbers. When syncing the
   mirror, copy from `paperA/fig/` only the four `make_{concept,curves,where,scaling}_figure.py`
   scripts and their `fig_*` outputs (plus the `make_threshold_figure.py` and `fig4_threshold*`
   already there); leave out the September `make_figures.py`, `fig1_operating_curve.*`,
   `fig2_mechanism.*` and `fig3_two_sided.*`, which call the earlier filter "deployed" (an
   anonymity risk) and contain em dashes.
2. **Make the GitHub mirror private** for the review period: its git history still contains old
   manuscripts with names (anonymous.4open.science shows only the current tree), and its public
   tag `paperA-ipmc` holds identifying files (delete it: `git push origin :refs/tags/paperA-ipmc`).
   The non-anonymous main repo is public too (paper, your name, the benchmark): make it private for
   the review period, or keep it public knowingly and answer the preprint field accordingly.
3. **OpenReview profiles** of every author complete (ORCID, affiliation, DBLP, conflicts), author
   order final (it cannot change after submission), the preprint field answered. **Every author must
   register as a reviewer by 14 October** (ARR: otherwise the paper may be desk-rejected).
4. **Responsible NLP checklist**: draft answers below. The question on AI assistance (E) is yours to
   answer according to ACL's policy.
5. Done 2026-10-06: the 22-item blind gold review (`gold_review_2026-10-02_v2_eg_curated.xlsx`),
   7 labels changed (benchmark now 251/437 positive; `data/evaluation/BENCHMARKS.json`), every table
   regenerated, agreement on the controls in the Limitations. The 437-row second annotation
   (`second_annotation_2026-10-04_BLIND.xlsx`) is follow-up work, not needed for this submission.

## OpenReview fields

- **Title:** Ask About the Pair, Not the Passage: Verifying Literature-Mined Interactions
- **Type:** long paper
- **Area (suggested):** Information Extraction (alternatives: NLP Applications; Resources and Evaluation)
- **Keywords:** relation extraction; candidate verification; knowledge base population; biotic
  interactions; biodiversity informatics; text mining; BioRED; zero-shot LLMs
- **TL;DR** (247 characters): When literature-mining pipelines propose candidates by co-occurrence, ask
  whether the passage supports this pair, not whether it describes an interaction: a controlled
  comparison, a public benchmark and zero-shot LLMs all favour the pair question.
- **Abstract (plain text, 193 words; 197 if hyphenated words are split; same text as the PDF):**

Species-interaction databases (who eats, infects or pollinates whom), like many biomedical knowledge bases, are partly built by mining the literature. A typical pipeline proposes a candidate wherever a passage names two organisms near an interaction word, and keeps it if a classifier says the passage describes an interaction. That check concerns the passage, not the pair: a passage naming several organisms can support one pair but not another, yet the classifier answers the same whichever pair was proposed. We instead ask whether the passage states that this pair interacts, keeping the encoder and the training data fixed. On 437 expert-graded candidates, asking about the pair raises the area under the precision-recall curve from 0.867 to 0.942 and wins at every decision threshold; adding the interaction word helps only where high precision is needed. The gain is largest on passages naming several entities. It carries over to BioRED, a public biomedical benchmark, and to zero-shot Qwen language models beyond the smallest, where scale helps the pair question far more than the original one. The verifier runs on a CPU and returns a score, not a verdict, letting each database balance missed against wrong entries.

## Responsible NLP checklist: draft answers

**A. Every submission**
- A1 Limitations: Yes, section "Limitations".
- A2 Potential risks: Yes. Verified candidates would feed a public database, so errors propagate; the
  paper reports precision across thresholds and separates ingestion from expert review as operating
  points (§5 "A score, not a verdict", Figure 2, Appendix C); the Limitations name the upstream
  errors no verifier can catch.

**B. Scientific artifacts**
- B1 Cited the creators: Yes (§3.1 BiotXplorer and SIBiLS, §3.4 and §4.2 BioRED and BioREDirect,
  §3.2 models, Appendix O "Artifacts and compute").
- B2 Licences: Yes, Appendix O "Artifacts and compute" (BiomedBERT MIT; BioLinkBERT, Qwen3, Qwen3.5
  Apache 2.0; TaxoNERD MIT; BioRED and BioREDirect public NCBI releases without a licence file).
- B3 Use consistent with intended use: Yes; research use of research artifacts.
- B4 Personal data / offensive content: Not applicable: the data are sentences from published
  scientific literature about organisms; no personal information.
- B5 Documentation: Yes (§3.3 benchmark blocks, §3.2 corpus composition; the code README).
- B6 Statistics of the data: Yes (§3, Table captions: sizes, positive rates, splits).

**C. Computational experiments**
- C1 Parameters and compute: Yes (§3.2: 110M parameters, about four minutes per run on one A100;
  Appendix O "Artifacts and compute": about 50 GPU-hours in total).
- C2 Experimental setup and hyperparameters: Yes (§3.2 recipe, with the optimiser settings in
  Appendix O; §3.5 protocol; §3.3 records that about 100 configurations were scored during
  development; Appendix H "Encoders" for the recipes tried).
- C3 Descriptive statistics: Yes (three seeds, mean and standard deviation; McNemar tests;
  Clopper–Pearson intervals).
- C4 Existing packages: Yes for TaxoNERD (`en_ner_eco_biobert`, Appendix C, Table 4 caption) and
  Ollama (Appendix O "Artifacts and compute"); Hugging Face transformers, scikit-learn and spaCy are named
  only in the code's requirements.txt, which pins their versions. One sentence naming them in that
  appendix makes this a full Yes.

**D. Human annotators**: needs your input
- D1 Instructions: two annotation efforts are reported. (i) The benchmark's original labels: Biotx100
  is a domain expert's four-axis grading; Reject50 has its own curator field in
  `biotx_rejected_50_testset.csv` (keep names and initials out of OpenReview); Test299 comes from
  expert-graded pair-level sets (keep internal set names and curators' initials out of OpenReview). Say who
  graded each and with which guideline. (ii) The 22-item blind re-annotation (Limitations): its
  instructions are the "how to" tab of `gold_review_2026-10-02_v2_BLIND.xlsx` (SENTENCE / PAIR questions; UNSURE allowed), which can be
  quoted or added as a short appendix to make D1 a Yes. "Not applicable" is not an option: the paper
  reports annotation made for this work (also the 97 direction labels and the 150 teacher checks).
- D2 Recruitment and payment; D3 consent; D4 ethics review; D5 demographics: answer for the
  benchmark's graders and for the second annotator (say if they are authors).

**E. AI assistants**: yours to answer under ACL's policy on AI writing and coding assistance.
