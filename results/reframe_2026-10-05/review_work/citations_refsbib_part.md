# Citation-support check: assigned references.bib keys in paperA_reframe.tex

Checked 2026-10-05 05:15-05:25. COMPLETE (not partial). Read-only; no other file touched.
Sources: lit_work notes (agentB.md, LITERATURE.md "CLAIMS WE MUST NOT MAKE"), plus abstracts fetched live:
PubMed (BioREx), Crossref (BioREDirect, D'Souza 2025, Farrell 2024), bioRxiv API (Zou 2026).
"OLD" = the same sentence with the same citation is in paperA.tex (line given) and was skipped as instructed.
Lines are paperA_reframe.tex lines.

| tex line | key | sentence (short) | OLD/NEW | what the source says (1 line) | verdict |
|---|---|---|---|---|---|
| L70 | luo2022biored | "The result replicates on BioRED" | OLD (paperA L71) | Dataset citation | skip (SUPPORTS) |
| L172 | mcnemar1947note | "McNemar tests carry a continuity correction" | OLD (paperA L157) | The new sentence-level "exact McNemar" clause after it has no citation and needs none | skip |
| L213 | bucilua2006model, hinton2015distilling | "distilled from the teacher's labels" | OLD (paperA L197) | Model compression: train a small model on a teacher's labels / soft targets | skip (SUPPORTS) |
| L213 | west2022symbolic | same | NEW position (paperA L597 cited it for "Distilling a large model's judgments into a small student is established practice") | GPT-3 generates a commonsense corpus, a critic filters it, a smaller COMET student is trained on it | SUPPORTS (teacher-generated data, not teacher labels on given inputs, but this is still distillation into a student) |
| L213 | gekhman2023trueteacher | same | NEW position (paperA L597) | An LLM labels model-generated summaries for factual consistency; a T5 student is trained on those labels | SUPPORTS (closest match: an LLM teacher labels the data, then a small classifier is trained) |
| L213 | all four | "...\citep{...} with the same recipe: AdamW at 2e-5..." | NEW | None of the four papers uses this recipe; the citation attaches to "distilled", and "same recipe" means the three arms share it | SUPPORTS; optional: move "with the same recipe" before the citation so the hyperparameters are not read as taken from those papers |
| L382-383 | lai2023biorex | "systems that read the whole abstract reach 74.1 to 75.3 \citep{...}" | Number is OLD (paperA L366-368). Body wording is NEW: it drops the system names "(PubMedBERT, BioREx and BioREDirect; ...)" | The BioREx abstract (PubMed 37673376) gives "a new SOTA from 74.4% to 79.6% in F-1 measure on BioRED", a different number and setting. The 74.1-75.3 range must come from a re-evaluation table, presumably in BioREDirect | PARTIAL: with the system names removed, a reader checking BioREx finds 79.6, not 74.1-75.3 |
| L382-383 | lai2025bioredirect | same | same | The abstract (Bioinformatics, 10.1093/bioinformatics/btaf226) covers directionality annotations and a multi-task soft-prompt model that beats GPT-4 and Llama-3. The abstract gives no F1, so 74.1-75.3 can only be checked against the full-text table | PARTIAL: check the full text before submission |
| L383 | lai2023biorex, lai2025bioredirect | "..., in part because a quarter of the related pairs are never co-mentioned in one sentence (Appendix)" | NEW (paperA put this in a separate sentence, "Much of that gap is structural") | Neither paper makes this claim. It is our own measurement (app:bioredep, recall cap 0.749). The clause directly follows the citation, so it reads as an explanation the cited papers give. It also states a cause for *their* score, while the appendix itself says "what the published systems add is document-level modelling" (the same model given the whole abstract reaches only 67.0) | DOES NOT SUPPORT (wrong attribution through placement, plus a causal-direction problem) |
| L537-544 | gururangan, poliak, kaushik-lipton, thorne2018fever, schuster, geirhos, du, wiegand, parmar, feng | Partial-input paragraph | OLD (paperA L560-571; compressed, same claims) | -- | skip |
| L550 | zhang2017position | "Since TACRED an instance has been the triple (sentence, e1, e2)" | OLD (paperA L577) | -- | skip |
| L551 | lee2020biobert (not assigned, but NEW) | "biomedical RE marks the target pair in one copy of the sentence per pair" | NEW | BioBERT RE anonymises the target entities with tags (@GENE$, @DISEASE$), one instance per pair | SUPPORTS; "masks" would be more exact than "marks" |
| L552 | soares2019matching, wu2019enriching, zhou2022improved | "how the pair is injected is worth several F1 points" | OLD (paperA L578) | -- | skip |
| L552 | zhong2021frustratingly (not assigned, but NEW) | "pair-specific encodings beat a shared one at the cost of a pass per pair" | NEW | Typed markers inserted per pair beat the shared encoding; their batched approximation is faster with a small accuracy loss | SUPPORTS |
| L562-563 | taille-etal-2020-lets, ai2023endtoend | "error attribution across pipeline stages is easy to get wrong" | OLD (paperA L584-586) | -- | skip |
| L563-565 | levy2017zero, obamuyide2018zero, sainz2021label, wadden2020fact, lehmann2012defacto | relation as question / entailment, claim verification, KG triple validation | OLD (paperA L589-594, condensed) | -- | skip |
| L599 | hirschman2012biocuration, britan2018nexta5, lee2018scaling | "evaluating such tools by whether they help the curator" | OLD (paperA L606) | -- | skip |
| L599-600 | cuzick2023interspecies | "take the interacting pair as the annotation unit" | OLD (paperA L606) | -- | skip |
| L600-601 | keck2025extracting, zou2026llm | "Large language models extract interactions at scale" | OLD (paperA L608-609: "are applied to interaction extraction at scale") | Keck: GPT-4o extracts pairwise species interactions from paragraphs. Zou: an LLM workflow extracts interaction type and species from citizen-science comments | SUPPORTS |
| L600-601 | dsouza2025mining | same | OLD sentence (wording now stronger: "extract interactions") | Abstract: LLMs "mine key ecological entities ... species names, their locations, associated habitats, and ecosystems". It extracts entities, not interactions | DOES NOT SUPPORT (carried over from paperA; flagged because it is a mis-attribution) |
| L600-601 | farrell2024landscape | same | OLD sentence | A review of text-mining approaches for ecology and evolution (frequency-based, traditional NLP, deep language models, LLMs). It does not itself extract interactions | DOES NOT SUPPORT "extract interactions at scale" (carried over; fine as a general text-mining review) |
| L602 | zou2026llm | "first ask whether a text contains an interaction" | NEW | Step 1 asks whether a COMMENT contains an interaction, then which species and what type (agentB.md, Fig. 1 caption; LITERATURE.md MUST-NOT #10) | SUPPORTS ("a citizen-science comment" is more exact than "a text") |
| L602 | zou2026llm | "standardise names post hoc, without checking extractions against their text" | OLD (paperA L610) | The abstract adds "With appropriate validation, expert review ..." and the paper scores the LLM on manually labelled test sets (76% / 96% on eBird). It has no step that checks each extraction | PARTIAL: "without checking" can read as "never validated". Recommended rewording below |
| L1308 | geifman2017selective | "A calibrated score with an adjustable threshold leaves it to them" | OLD (paperA L456; the following Swaminathan clause is moved from paperA L597-600) | -- | skip |

## MUST FIX

1. **L382-383** (lai2023biorex / lai2025bioredirect: the system names are gone, and our own "quarter" finding sits on their citation).
   - Current: `$57.5$; systems that read the whole abstract reach $74.1$ to $75.3$
\citep{lai2023biorex,lai2025bioredirect}, in part because a quarter of the related pairs are never co-mentioned in one sentence (Appendix~\ref{app:bioredep}).`
   - Replacement: `$57.5$; systems that read the whole abstract reach $74.1$ to $75.3$
(PubMedBERT, BioREx and BioREDirect; \citealp{lai2023biorex,lai2025bioredirect}). Part of that gap is structural: a quarter of the related pairs are never co-mentioned in one sentence, which no sentence-scope verifier can recover (Appendix~\ref{app:bioredep}).`
   - Also check the BioREDirect full-text table for 74.1 / 75.3 before submission. BioREx's own abstract reports 79.6 on BioRED in a different setting.

2. **L600-601** (OLD sentence carried over from paperA L608-609, but a mis-attribution: dsouza2025mining extracts entities, not interactions; farrell2024landscape is a review).
   - Current: `Large language models extract interactions at
scale \citep{keck2025extracting,farrell2024landscape,dsouza2025mining,zou2026llm};`
   - Replacement: `Large language models extract interactions at
scale \citep{keck2025extracting,zou2026llm} and are applied to ecological text mining more broadly \citep{farrell2024landscape,dsouza2025mining};`

## SHOULD FIX (wording, optional)

- **L602**. Current: `\citet{zou2026llm} first ask whether a text contains an interaction and standardise names post hoc,
without checking extractions against their text.` Replacement: `\citet{zou2026llm} first ask whether a citizen-science comment contains an interaction and standardise names post hoc; their workflow has no step that checks each extraction against its text.`
- **L551**. Current: `marks the target pair in one copy of the
sentence per pair \citep{lee2020biobert}`. Replacement: `masks the target pair in one copy of the sentence per pair \citep{lee2020biobert}`.
- **L212-213**. Optional reorder: `distilled from the teacher's labels \citep{...}, all with the same recipe:`. This makes clear the hyperparameters are not taken from the distillation papers.
