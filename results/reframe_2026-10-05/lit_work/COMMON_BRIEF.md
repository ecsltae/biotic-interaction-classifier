# Common brief for literature-search subagents (2026-10-05)

## Context
An ACL Rolling Review paper (double-blind, under review) argues: when literature-mining pipelines propose
candidate (taxon, relation, taxon) triples by co-occurrence, the filter should ask "does the passage assert an
interaction between THESE TWO organisms?" (pair-conditioned verification) instead of "does the passage describe
an interaction?" (sentence-level). Same encoder (BiomedBERT-base), same teacher-labelled corpus (labels from a
local Qwen3-32B), only the input differs (sentence / pair / triple). Results: biodiversity benchmark AUPRC
0.851 -> 0.918; BioRED (BC8 protocol, sentence-scope candidate pairs) AUPRC 0.738 -> 0.843, gain only on
sentences naming more than two entities.

The NEW angle being drafted ("ask about the pair, even if you want the sentence"): the end user wants a
SENTENCE-level decision (does this passage describe any biotic interaction?). Claim to place in prior work: a
sentence-level decision read off pair-level verification (max over the passage's candidate pairs = the
multi-instance "at least one" assumption) is at least as good as a model trained to answer the sentence question
directly, with the same teacher, data and encoder; and only pair-level answers can populate a pair database (a
perfect sentence filter has limited pair precision: 0.724 on our benchmark). The novelty is NOT the max
aggregation; it is the controlled measurement.

## Hard rules (the user's; they override everything else)
- Never use the Anthropic API key or the `anthropic` package. Do not use the Workflow tool. Do not spawn
  subagents yourself.
- `source ~/MetaP/MPvenv/bin/activate` before Python. Never pip-install into MPvenv.
- Write ONLY to the two files assigned to you under
  ~/MetaP/classifier/results/reframe_2026-10-05/lit_work/. Do not modify any other file.
  Do not touch git. Do not use the GPU.
- Never invent a reference, DOI, venue, page range, author or number. A work is reported only after you have
  confirmed it exists on a primary bibliographic source: the ACL Anthology (https://aclanthology.org/<ID>.bib
  works with curl), Crossref (`curl -sLH "Accept: application/x-bibtex" https://doi.org/<DOI>` or
  https://api.crossref.org/works?query.bibliographic=...), the Semantic Scholar API
  (https://api.semanticscholar.org/graph/v1/paper/search/match?query=...&fields=title,year,authors,venue,externalIds),
  the publisher page, PubMed/PMC, OpenReview, PMLR (proceedings.mlr.press), NeurIPS proceedings or arXiv.
  DBLP blocks curl from this server (bot check); WebFetch on dblp.org may or may not work.
- Every statement of what a work shows must come from text you actually read (abstract at minimum; full text
  for any specific number or quoted phrase). Mark each claim as [abstract] or [full text, section/page X]. If
  you quote a number (e.g. a share of noisy distant-supervision sentences), give the exact sentence and where it
  is. If you cannot read the text, say so; do not paraphrase from memory.
- Do NOT return works already in the paper's bibliography (keys below); you may mention them by key when they
  matter to your question (e.g. "riedel2010modeling already cited; its noise figure is X, quote: ...").
- Anonymity: never search for or mention the paper's authors; BiotXplorer and SIBiLS are public third-party
  systems only.
- Prefer peer-reviewed versions over arXiv. Use arXiv only when no peer-reviewed version exists; say so.
- Time: finish and return by 03:45 (check `date`). Write your files incrementally so partial work survives.

## Existing bibliography keys (do not duplicate these works)
poelen2014global mintz2009distant qwen2025technical gururangan-etal-2018-annotation poliak-etal-2018-hypothesis
kaushik-lipton-2018-much feng-etal-2019-misleading schuster-etal-2019-towards wiegand-etal-2019-detection
parmar-etal-2023-dont geirhos-etal-2020-shortcut du-etal-2024-shortcut riedel2010modeling zhang2017position
soares2019matching wu2019enriching zhou2022improved rosenman2020exposing peng2020learning mtumbuka2024entity
levy2017zero obamuyide2018zero sainz2021label wadden2020fact west2022symbolic gekhman2023trueteacher
nogueira2019passage lu2025feeding gobeill2020sibils ruch2024biotxplorer dimitrova2020semantic
rodriguezesteban2006imitating hirschman2012biocuration cuzick2023interspecies farrell2024landscape
keck2025extracting taille-etal-2020-lets ai2023endtoend britan2018nexta5 lee2018scaling geifman2017selective
swaminathan2024selective rees2017opentree gerner2010linnaeus zou2026llm dsouza2025mining hinton2015distilling
bucilua2006model mcnemar1947note gu2021domain luo2022biored thorne2018fever lehmann2012defacto
leguillarme2022taxonerd lai2025bioredirect lai2023biorex islamaj2024overview yasunaga2022linkbert
(Titles of note: Riedel 2010 "Modeling Relations and Their Mentions without Labeled Text"; Baldini Soares 2019
"Matching the Blanks"; Zhou & Chen 2022 "An Improved Baseline for Sentence-level RE"; Levy 2017 "Zero-Shot RE
via Reading Comprehension"; Rosenman 2020 "Exposing Shallow Heuristics of RE Models"; Cuzick 2023 "A framework
for community curation of interspecies interactions literature" (PHI-Canto); Hirschman 2012 "Text mining for the
biocuration workflow"; Lee 2018 "Scaling up data curation using deep learning: literature triage in genomic
variation resources"; Britan 2018 neXtA5; Dimitrova 2020 "Semantic Publishing Enables Text Mining of Biotic
Interactions"; Keck 2025 "Extracting massive ecological data on state and interactions of species using LLMs";
Farrell 2024 text-mining review for ecology; Zou 2026 "LLMs unlock the ecology of species interactions";
D'Souza 2025 invasion biology mining; Luo 2022 BioRED; Islamaj 2024 BioRED track overview; Lai 2023 BioREx;
Lai 2025 BioREDirect.)

## Output format
1. `<agent>.md`: per sub-question, the key works (aim for 2-6 per sub-question, quality over quantity). For each:
   - proposed BibTeX key (style: firstauthorsurname + year + first content word of title, lowercase, e.g.
     hoffmann2011knowledge), full title, venue, year, DOI or Anthology ID, the URL you verified it at;
   - WHAT IT SHOWS (one line, with [abstract]/[full text, where] tag; exact quotes for numbers);
   - HOW OUR CLAIM RELATES (one line: does it pre-empt us, support us, or contrast with us?).
   Then: a short list "claims a reviewer would call known, and the citation that makes them known", and
   "claims we must not make, and the work that refutes each". End with anything you could not verify.
2. `<agent>.bib`: one BibTeX entry per reported work, copied from the ACL Anthology .bib or Crossref
   content negotiation (then normalised to this style: key as above, fields title, author ("Surname, Given and
   ..."), booktitle or journal, volume/number/pages when the source gives them, year, address/publisher when the
   source gives them, doi when one exists; no url/abstract/editor/month fields). Never fill a field the source
   did not give. Add a comment line above each entry: `% source: <URL used>`.
