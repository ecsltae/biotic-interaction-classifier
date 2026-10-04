# Paper A — submission sheet (ACL Rolling Review), 2026-10-04

## State of the PDF (`paperA/paperA.pdf`)

- 17 pages: 8-page body (the Conclusion ends on page 8), Limitations, references, appendices.
  Review mode with line numbers, author "Anonymous ACL submission", empty PDF metadata.
- ACL pubcheck (`~/opt/aclpubcheck-venv`): **All Clear**. Its margin test needs ImageMagick and was
  skipped; margins measured from glyph positions instead: 2.42–2.50 cm at the sides, 2.55–2.77 cm
  at the top, nothing outside the text area except line and page numbers.
- 0 LaTeX errors, 0 overfull boxes, all fonts embedded, no Type 3 fonts, A4.
- Abstract: 196 words (ACL limit 200).
- Numbers: all 785 numbers in the PDF checked against the result files
  (`results/paperA_v2/NUMBER_AUDIT_2026-10-04.md`).
- References: all 58 cited references verified to exist (Semantic Scholar, Crossref) with the cited
  first author and year (`scripts/check_references.py`). The three year differences are
  preprint vs. published versions; the bib cites the published ones.

## To do before uploading (you)

1. **Anonymous code link.** Publish the mirror through anonymous.4open.science (steps in
   `results/overnight_2026-10-03/ANONYMOUS_4OPEN_SCIENCE.md`), then replace `XXXX` in `paperA.tex`
   (search `ANONYMOUS LINK`) and rebuild:
   `pdflatex paperA && bibtex paperA && pdflatex paperA && pdflatex paperA`.
2. **Make the GitHub mirror private** for the review period: its git history still contains old
   manuscripts with names (anonymous.4open.science shows only the current tree).
3. **OpenReview profiles** of every author complete (ORCID, affiliation, DBLP, conflicts), author
   order final (it cannot change after submission), the preprint field answered.
4. **Responsible NLP checklist**: draft answers below. The question on AI assistance (E) is yours to
   answer according to ACL's policy.
5. Optional, if done before the deadline: the 22-item gold review and the second annotation
   (`data/evaluation/*_BLIND.xlsx`). With them I regenerate every table (one command per benchmark)
   and add the agreement figure to the Limitations.

## OpenReview fields

- **Title:** Which Pair Is It About? Pair-Conditioned Verification of Literature-Mined Biotic Interactions
- **Type:** long paper
- **Area (suggested):** Information Extraction (alternatives: NLP Applications; Resources and Evaluation)
- **Keywords:** relation extraction; candidate verification; biotic interactions; biodiversity
  informatics; knowledge base population; knowledge distillation; BioRED; zero-shot LLMs
- **TL;DR:** For candidates retrieved by co-occurrence, ask whether the passage supports *this pair*,
  not whether it describes an interaction: a controlled comparison and a public benchmark show the
  pair question wins for fine-tuned encoders and zero-shot LLMs alike.
- **Abstract (plain text, 196 words):**

Literature-mining pipelines that populate biotic-interaction databases propose candidate (taxon,
relation, taxon) triples by co-occurrence, then filter them with a sentence-level classifier: does
this passage describe an interaction? Candidates are retrieved precisely because their passage names
two taxa near an interaction term, so that question is answered before it is asked; what needs
deciding is whether the passage asserts an interaction between these two. In a controlled
comparison - one encoder, one 48k-row teacher-labelled corpus, three seeds, only the input differing
- telling the model which pair it is judging raises AUPRC on 437 expert-graded candidates from 0.851
to 0.918 and F1 from 0.775 to 0.871 (McNemar p = 1.3e-7), at every threshold; adding the relation
term buys nothing. Asked the same questions zero-shot, the 32B teacher shows the same gap at twice
the size. On the public BioRED corpus, with candidates built the same way, conditioning on the pair
raises AUPRC from 0.738 to 0.843 and entity-pair F1 by 9.6 points, with no gain on sentences naming
two entities and +0.120 on sentences naming more. The deployed verifier, 110M parameters, also
labels direction in four categories and runs at 32 candidates per second on a CPU.

## Responsible NLP checklist — draft answers

**A. Every submission**
- A1 Limitations: Yes, section "Limitations".
- A2 Potential risks: Yes. Verified candidates feed a public database, so errors propagate; the
  paper reports precision at every threshold and recommends curator review (§7, Limitations).

**B. Scientific artifacts**
- B1 Cited the creators: Yes (§2 BiotXplorer and SIBiLS, §4.2 BioRED and BioREDirect, §3 models,
  Appendix "Artifacts and compute").
- B2 Licences: Yes, Appendix "Artifacts and compute" (BiomedBERT MIT; BioLinkBERT, Qwen3, Qwen3.5
  Apache 2.0; TaxoNERD MIT; BioRED and BioREDirect public NCBI releases without a licence file).
- B3 Use consistent with intended use: Yes; research use of research artifacts.
- B4 Personal data / offensive content: Not applicable: the data are sentences from published
  scientific literature about organisms; no personal information.
- B5 Documentation: Yes (§2 benchmark blocks, §3 corpus composition; the code README).
- B6 Statistics of the data: Yes (§2, §3, Table captions: sizes, positive rates, splits).

**C. Computational experiments**
- C1 Parameters and compute: Yes (§3: 110M parameters, about four minutes per run on one A100;
  Appendix "Artifacts and compute": about 50 GPU-hours in total).
- C2 Experimental setup and hyperparameters: Yes (§3 recipe; §2 protocol, including that about 100
  configurations were scored during development; Appendix "Encoders" for the recipes tried).
- C3 Descriptive statistics: Yes (three seeds, mean and standard deviation; McNemar tests;
  Clopper–Pearson intervals).
- C4 Existing packages: Yes (Hugging Face transformers, scikit-learn, spaCy, TaxoNERD, Ollama; the
  code pins versions).

**D. Human annotators** — needs your input
- D1 Instructions: the benchmark labels are the domain expert's curation (§2). Say who graded it and
  with which guideline, or "not applicable: existing expert curation".
- D2 Recruitment and payment; D3 consent; D4 ethics review; D5 demographics: answer for the
  benchmark's grader, and for the second annotator if that figure goes in.

**E. AI assistants** — yours to answer under ACL's policy on AI writing and coding assistance.
