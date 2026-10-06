# Stage: literature (runs in parallel with the analysis stage)

Goal: place the reframe ("ask about the pair, even if you want the sentence") in prior work, using
only references that exist. ARR desk-rejects papers with invented or inaccurate references. Do not
use any reference you have not verified.

First read the Introduction, the Related work section and the bibliography of `paperA/paperA.tex`
and `paperA/references.bib`. The paper already cites 58 verified works: reuse their keys, and look
only for what the new angle needs. Use WebSearch and WebFetch, the Semantic Scholar API, DBLP, the
ACL Anthology and Crossref. Up to 3 subagents may search in parallel. Give each one the rules above
and a precise question.

Questions:
1. Is deriving a sentence-level decision from pair-level verification known? Look at:
   - multi-instance learning and the "at least one" assumption in distant supervision (Riedel et
     al. 2010; Hoffmann et al. 2011; Surdeanu et al. 2012; Zeng et al. 2015; Lin et al. 2016);
   - aggregation of mention-level predictions into sentence-level or document-level decisions;
   - relation detection or "relation existence" as a filter.

   What exactly is new in our claim, which prior work must we cite, and which claims would a
   reviewer call known?
2. Sentence-level triage versus pair- or relation-conditioned filters in curation and
   database-population pipelines. On the biomedical side: BioCreative triage tasks, PPI and
   gene–disease sentence filtering. On the biodiversity side: text mining of biotic and species
   interactions, GloBI-related work, and ecological literature-mining pipelines. What did those
   pipelines use as their filter question?
3. Pair-conditioned inputs: entity markers and typed markers (e.g. Baldini Soares et al. 2019; Zhou
   and Chen 2022), QA-style relation extraction (Levy et al. 2017; Li et al. 2019), and, for LLMs,
   question specificity or decomposition (asking about a specific pair versus a general question).
4. Annotation semantics: sentence-level versus pair-level labels for interaction or relation
   detection, and the inter-annotator agreement reported for each, as context for the user's dual
   labels and the UNSURE removal.
5. Evidence that a perfect sentence filter cannot populate a pair database. For example, distant
   supervision noise analyses report the share of sentences with a related pair that do not express
   the relation (Riedel et al. 2010), and work reports how often sentences that mention an
   interaction concern a different pair.

Deliverables:
- `results/reframe_2026-10-05/LITERATURE.md`. Per question, 2–6 key works, each with one line on
  what it shows and one line on how our claim relates. Then:
  - a NOVELTY STATEMENT draft: 3–5 sentences, accurate and defensible against the strongest
    related work;
  - a list of CLAIMS WE MUST NOT MAKE, with the work that refutes each;
  - draft related-work paragraphs, about 150–250 words of LaTeX, using the citation keys, ready for
    the writer;
  - a short note on BiotXplorer as the application: what the paper can say about it from public
    sources only, so that nothing overlaps an upcoming paper by its own authors and anonymity holds.
- `paperA/reframe_extra.bib`: BibTeX for every NEW reference. Do not duplicate keys or works already
  in `paperA/references.bib`. Copy entries from DBLP, the ACL Anthology or Crossref where possible,
  and never invent a DOI, venue, page range or author.
- Verification: run `python3 scripts/check_references.py paperA/reframe_extra.bib`. Check every
  entry it marks CHECK or NOT FOUND by hand against the publisher or DOI page. Record each entry's
  status in LITERATURE.md, and delete any entry you cannot verify, from both files.

Do not commit; the writer stage commits.
