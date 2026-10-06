# Citation check: paperA/paperA_reframe.tex (reframe draft)

Read-only review, 2026-10-05, 05:15-05:30 CEST. COMPLETE (not partial).
I edited no existing files. Files I wrote in this directory:
- `check_references_out.txt`: full checker run
- `retry_notfound.bib` and `check_references_retry.txt`: retry of the 2 NOT FOUND entries
- `cite_contexts.txt`: every cite command with its tex line
- `citations_refsbib_part.md`: detailed sub-check of the references.bib keys that appear in new sentences, run by a helper; its findings are merged below

Draft state checked: tex 05:12:25, bbl 05:12:26, aux/log 05:12:34. The log is newer than the tex, so the build reflects the current text.

---------------------------------------------------------------------------------------------------

## (a) Resolution check: PASS

- 79 distinct keys are cited (`\cite`, `\citet`, `\citep`, `\citealp`, including `\citep[..][]` forms).
  - 58 are in `references.bib` and 21 in `reframe_extra.bib`.
  - No key is missing from both files, and no key is defined in both.
- `paperA_reframe.bbl` has 79 `\bibitem`s, one per cited key.
- `paperA_reframe.log` has no "undefined", "Citation ... undefined", "multiply defined", "Rerun" or natbib warnings.
- `paperA_reframe.blg` reports `warning$ -- 0`. Its one "error message" is `Illegal, another \bibstyle command`.
  - Cause: `acl.sty:195` already sets `\bibliographystyle{acl_natbib}`, and `paperA_reframe.tex:713` sets it again.
  - It is harmless and appears identically in the submission build (`paperA.blg`).
  - Optional: drop tex line 713 to silence it.
- Harmless, for information only:
  - 29 of the 50 `reframe_extra.bib` entries are not cited. BibTeX omits them, so they are not printed.
  - Uncited: andrews2002support, bravo2015extraction, dumitrache2018crowdsourcing, elkhettari2023building, gao2021manual, gurulingappa2012development, herrerozazo2013ddi, hershey2021benefit, hu2025decomposition, jia2019document, jiang2011target, jimenezgutierrez2022thinking, krallinger2011protein, laban2022summac, li2019entity, liu2019event, min2023factscore, peng2019transfer, plank2022problem, thessen2014knowledge, thieu2012literature, verga2018simultaneously, wadden2022multivers, wiegers2009text, zaidan2007using, zeng2015distant, zhang2016rationale, zhang2023aligning, zhang2025entity.
  - Every `references.bib` entry is cited.

## (b) check_references.py (Semantic Scholar title match), cited reframe_extra entries only

Command: `python3 scripts/check_references.py paperA/reframe_extra.bib paperA/paperA_reframe.aux`
Result: **19 OK, 0 CHECK, 2 NOT FOUND, of 21 cited entries.**

| key | result |
|---|---|
| dietterich1997solving | OK |
| hoffmann2011knowledge | OK |
| ilse2018attention | OK |
| chowdhury2013exploiting | OK |
| xie2021revisiting | OK |
| krallinger2008overview | OK |
| islamajdogan2019overview | OK |
| polajnar2011protein | OK |
| lim2016minter | OK |
| lee2020biobert | OK |
| zhong2021frustratingly | NOT FOUND (and NOT FOUND again on retry) |
| mraz2026fewshot | OK |
| jiang2019challenge | OK |
| pyysalo2008comparative | OK |
| tikk2013detailed | OK |
| rei2019jointly | OK |
| dasanmartino2019fine | NOT FOUND (and NOT FOUND again on retry) |
| farkas2010conll | OK |
| li2021dual | OK |
| magge2021deepademiner | OK |
| pavlopoulos2022from | OK |

Hand verification of the two NOT FOUND entries (both fetched live today):

- **zhong2021frustratingly: REAL, MATCHES.**
  - ACL Anthology `2021.naacl-main.5.bib`: "A Frustratingly Easy Approach for Entity and Relation Extraction"; Zhong, Zexuan and Chen, Danqi; NAACL-HLT 2021; Online; ACL; pp. 50--61; DOI 10.18653/v1/2021.naacl-main.5.
  - Crossref gives the same title, authors (Zhong, Chen), pages 50-61 and year 2021.
  - The .bib entry agrees on every field. The S2 miss is a lookup failure (LITERATURE.md run 2 had it OK).
- **dasanmartino2019fine: REAL, MATCHES.**
  - ACL Anthology `D19-1565.bib`: "Fine-Grained Analysis of Propaganda in News Articles"; Da San Martino, Yu, Barrón-Cedeño, Petrov, Nakov; EMNLP-IJCNLP 2019; Hong Kong; pp. 5636--5646; DOI 10.18653/v1/D19-1565.
  - The .bib entry agrees on every field.
  - Crossref prints "...News Article" and pp. 5635-5645, a known Crossref metadata variant. The Anthology and the PDF agree with the .bib, so keep it as is.

Extra spot check: **mraz2026fewshot** (2026 preprint) was confirmed live on arxiv.org/abs/2606.15412.
- Title, the three authors, v1 13 Jun 2026 and v2 3 Aug 2026 all match.
- The .bbl renders it as "Preprint, arXiv:2606.15412".
- Before camera-ready, re-check whether a peer-reviewed version exists.

**No bib entry needs removing for lack of verification.**

## (c) Per-citation support

Verdicts are based on:
- the verifier reports (`lit_work/verify1-3.md`, quoting full text);
- abstracts fetched live today: PURE and mraz on Anthology/arXiv; BioREx, BioREDirect, D'Souza, Farrell and Zou in the helper check;
- `references.bib`.

"OLD" means the same sentence and citation already appear in `paperA.tex`. Those were skipped, as instructed, unless the wording changed.

### c1. reframe_extra.bib keys

| tex line | key | sentence (short) | what the source says | verdict |
|---|---|---|---|---|
| 54 | krallinger2008overview | "does it describe a biotic interaction at all? That is the decision curation triage asks" | BioCreative II PPI: IAS = classify PubMed abstracts as relevant to PPI annotation, separate from IPS (pair extraction) | SUPPORTS. Triage there is abstract level and about proteins; the analogy is fair |
| 54 | islamajdogan2019overview | same | BC VI PM track: (i) document triage, (ii) relation extraction of protein pairs | SUPPORTS |
| 74 | dietterich1997solving | "the passage describes one exactly when one of its pairs interacts" | MI: a bag is positive iff at least one instance is positive | SUPPORTS. The "if the pairs scored cover..." condition is ours and stated as such |
| 430 | dietterich1997solving, hoffmann2011knowledge | "the multi-instance assumption" | Dietterich defines MI. Hoffmann Sec. 1: weak supervision cast as MI, "at least one of the sentences..." | SUPPORTS |
| 441 | li2021dual | "as the maximum of the pair labels it makes a pair-route win the expected outcome" | DSMIL (whole-slide images): a patch-label "upper-bound fully-supervised model" (AUC 0.936 vs 0.894 single-scale DSMIL; DSMIL-LC is within 2% accuracy) | PARTIAL. It supports "instance supervision is the usual upper bound". The BioRED expectation is our analogy. LITERATURE.md Q-conditions endorse the sentence, so this is not a violation; "cf." would be more precise (recommended R1) |
| 483 | lim2016minter | perfect-filter pair precision 0.536/0.724, "as found for document-level filters of microbial interactions" | @MInter: species-level precision is low "partly because it identifies abstracts containing interactions and reports all pairs of species names in such abstracts"; the All-Positives control has precision 7% | SUPPORTS (phenomenon match) |
| 487 | pavlopoulos2022from | "attribution from a sentence classifier ... is not tested here" | Rationale extraction on top of post-level toxicity classifiers comes close to a BiLSTM span tagger | SUPPORTS. Matches MUST-NOT #19 wording |
| 551 | lee2020biobert | "biomedical RE marks the target pair in one copy of the sentence per pair" | BioBERT Sec. 3.3: "We anonymized target named entities in a sentence using pre-defined tags such as @GENE$" | SUPPORTS ("masks" is more exact; R4) |
| 553 | zhong2021frustratingly | "pair-specific encodings beat a shared one at the cost of a pass per pair" | Abstract: the relation model is run once per pair with markers; a one-pass approximation gives an "8-16x speedup with a slight reduction in accuracy" | SUPPORTS |
| 567 | mraz2026fewshot | "Asking an LLM about one pair at a time is not known to beat listing all pairs at once" | Abstract: "Pairwise classification achieves higher recall, whereas joint generation is more precise and computationally efficient"; Sec. 5: "broadly comparable F1" | SUPPORTS. Complies with MUST-NOT #5; Labonte is not cited |
| 571 | dietterich1997solving | "standard multi-instance assumption" | as above | SUPPORTS |
| 572 | hoffmann2011knowledge | "a deterministic OR or a maximum over instances is how distant supervision aggregates them; there the instances are the sentences of one pair and their labels are unknown" | MultiR: deterministic-OR factors over latent sentence-level labels (Sec. 3, p. 543) | SUPPORTS for the OR and the latent labels. The "maximum" half is zeng2015distant (verified, in reframe_extra.bib, uncited). OR over binary labels equals max, so not wrong; adding Zeng is recommended (R2) |
| 574 | ilse2018attention | "which favours classifying the bag directly" | Sec. 2.1: the embedding-level approach is preferable because "the individual labels are unknown" (citing Wang et al. 2016) | SUPPORTS |
| 575 | rei2019jointly | "Trained jointly with finer labels, models make better coarse 'contains any' decisions" | Table 3: adding the token objective raises sentence F1 (83.87 to 85.90 on CoNLL-10; FCE only +0.20) | SUPPORTS |
| 575 | dasanmartino2019fine | same | SLC ("contains at least one propaganda technique"): MGN 60.98 vs BERT 57.74 F1 | SUPPORTS |
| 576 | li2021dual | "instance-supervised models serve as the upper bound for multiple-instance ones" | Authors' own words: "upper-bound fully-supervised model" | SUPPORTS (one WSI study; no numbers quoted) |
| 577 | magge2021deepademiner | "an adverse-event span tagger used to flag tweets loses to a tweet classifier" | "the ADE classifier ... (F1 = 0.63) outperforms the NER (F1 = 0.41) in identifying tweets containing ADEs" | SUPPORTS. Uncontrolled (the classifier was undersampled and threshold-tuned), but the draft makes no controlled claim |
| 578-579 | farkas2010conll | "in biomedical hedge detection cue taggers read as 'at least one cue' won on biological text while direct sentence classifiers won on Wikipedia" | CoNLL-2010 Task 1 covers biological AND Wikipedia text. Top Task1B systems were sequence labellers (a sentence is uncertain if it has at least one cue); the Task1W winner was a BoW sentence classifier, while #2 and #4 on Wikipedia were token classifiers | PARTIAL. "biomedical hedge detection" mislabels a shared task that includes Wikipedia, and "direct sentence classifiers won on Wikipedia" holds only for the winner. **MUST FIX 3** |
| 581 | chowdhury2013exploiting | "Sentence-level relation detection also serves as a filter before pair classification" | Stage 1: "any sentence that contains at least one DDI is considered ... positive"; sentences are filtered before pair classification | SUPPORTS |
| 581 | xie2021revisiting | same | RERE: "first performs sentence classification with relational labels and then extracts the subjects/objects" | SUPPORTS (multi-label relation-type detection; fine) |
| 587 | krallinger2008overview, islamajdogan2019overview | "Curation challenges score article triage and pair extraction separately" | IAS vs IPS; BC6 triage vs RE (RE labels on a subset of the triage documents) | SUPPORTS. "document triage" is the BC6 term; IAS is on abstracts (optional, R5) |
| 588 | polajnar2011protein | "sentence-level interaction detectors find the sentence but not the pair" | "only locates the sentence that describes the PPI and not the exact interacting pair" | SUPPORTS |
| 590 | pyysalo2008comparative | "accepting every co-occurring pair has a precision equal to the number of interactions per entity pair" | "the average number of interactions divided by the average number of entity pairs per sentence ... equals the precision of the co-occurrence method" | SUPPORTS |
| 590 | tikk2013detailed | "which falls as sentences name more entities" | "the more protein mentions a sentence exhibits, the lower the ratio of positive pairs" | SUPPORTS (no Fig. 7 numbers quoted; good) |
| 592 | lim2016minter | "an abstract filter ... with an AUC of 0.97 gives 25% precision on the species pairs read off it" | Abstract: AUC 0.97 (abstract level); interaction-level recall 95%, precision 25%; Table 1 species-level SVM precision 25 | SUPPORTS. Levels are kept apart correctly (no 95% specificity paired with 25%) |
| 593 | jiang2019challenge | "Aspect-level sentiment likewise reduces to the sentence-level task when sentences name a single target" | Abstract: such datasets make "ABSA task degenerate to sentence-level sentiment analysis" | SUPPORTS |
| 691 | li2021dual | "Because it is the maximum of the pair labels by construction, the pair route is the expected winner there" | as L441 | PARTIAL, same as L441 (R1) |

### c2. references.bib keys in new or changed sentences

Details are in `citations_refsbib_part.md`.

| tex line | key | sentence (short) | what the source says | verdict |
|---|---|---|---|---|
| 70 | luo2022biored | "The result replicates on BioRED" | dataset citation | OLD, SUPPORTS |
| 172 | mcnemar1947note | McNemar continuity correction | -- | OLD |
| 213 | bucilua2006model, hinton2015distilling | "distilled from the teacher's labels" | model compression / distillation | OLD, SUPPORTS |
| 213 | west2022symbolic, gekhman2023trueteacher | same; moved from paperA L597 | LLM teacher, small student (TrueTeacher: LLM-labelled data, T5 student) | SUPPORTS. Optional reorder so "with the same recipe" is not read as theirs (R6) |
| 382-383 | lai2023biorex, lai2025bioredirect | "systems that read the whole abstract reach 74.1 to 75.3 \citep{..}, in part because a quarter of the related pairs are never co-mentioned in one sentence" | The numbers are from the submission. But the system names "(PubMedBERT, BioREx and BioREDirect)" were dropped, and BioREx's own abstract gives 79.6 on BioRED in another setting. The "quarter never co-mentioned" figure is OUR measurement (App. app:bioredep), yet it now follows their citation as a cause of their scores. Our own appendix says "what the published systems add is document-level modelling" | DOES NOT SUPPORT as placed. **MUST FIX 1** |
| 537-544 | gururangan..., poliak..., kaushik-lipton, thorne2018fever, schuster..., geirhos..., du..., wiegand..., parmar..., feng... | partial-input paragraph | -- | OLD (compressed, same claims) |
| 550, 552 | zhang2017position, soares2019matching, wu2019enriching, zhou2022improved | instance = triple; pair injection worth F1 points | -- | OLD |
| 562-565 | taille..., ai2023endtoend, levy2017zero, obamuyide2018zero, sainz2021label, wadden2020fact, lehmann2012defacto | -- | -- | OLD (condensed) |
| 599 | hirschman2012biocuration, britan2018nexta5, lee2018scaling, cuzick2023interspecies | -- | -- | OLD |
| 600-601 | keck2025extracting, zou2026llm | "Large language models extract interactions at scale" | Keck: GPT-4o extracts species-interaction pairs. Zou: LLM workflow extracts interactions from citizen-science comments | SUPPORTS |
| 600-601 | dsouza2025mining | same | Title and abstract: mining "Species, Locations, Habitats, and Ecosystems". Entities, not interactions | DOES NOT SUPPORT "extract interactions". The sentence is carried over but strengthened from paperA's "are applied to interaction extraction". **MUST FIX 2** |
| 600-601 | farrell2024landscape | same | A review of text-mining approaches for ecology and evolution | DOES NOT SUPPORT "extract interactions" (fine as a review). **MUST FIX 2** |
| 602 | zou2026llm | "first ask whether a text contains an interaction and standardise names post hoc, without checking extractions against their text" | Step 1 asks whether a comment contains an interaction. They do score the LLM against hand-labelled test sets, but have no per-extraction check | SUPPORTS / PARTIAL ("without checking" can read as "never validated"; R3). Complies with MUST-NOT #10 |
| 1308 | geifman2017selective | "a calibrated score with an adjustable threshold leaves it to them" | -- | OLD |

### c3. LITERATURE.md "CLAIMS WE MUST NOT MAKE": compliance

I checked all 20 items against the draft (grep plus reading the Intro, Contributions, Sec. 6, Related work and Limitations). **No violation found.**

- #1 (max-over-pairs is new): never claimed. L74, L430 and L570-571 cite Dietterich/Hoffmann, and L581 says "We claim none of these".
- #2 (pair-conditioned input is ours): L556 says "We claim neither that diagnosis nor the pair-conditioned input".
- #3 and #7: prior work is cited (L483, L586-593).
- #4: no "surprising" or unconditional "known to win". The "expected winner" wording on BioRED (L441, L691) is scoped to BioRED's OR-construction, as LITERATURE.md itself prescribes.
- #5: L567 matches.
- #6: "pair-max is the natural sentence score" (L431) is justified by the "any interaction" semantics in the same sentence, as #6 requires.
- #8, #13 and #14 (30% DS noise, Keck 89.5%, mixed PPI precisions): none of these appear.
- #10: L602 complies.
- #11 and #15: no agreement-comparison claim and no negative-pair-share statistic.
- #12: L643 reports "3 UNSURE answers dropped" without calling it harmless.
- #16-#18: L575-581 cite Rei, Da San Martino, Li, Magge and Farkas and disclaim novelty.
- #19: L487 uses the prescribed wording.
- #20: the conclusion is conditional ("depends on what the sentence label counts").
- The one "not aware of published work" claim (L603) is carried over unchanged from paperA.tex L611.

## Anonymity of the bibliography: PASS

- No entry carries the author's name, affiliation (the authors' institution), GitHub/repository URL, "in preparation", "submitted" or "personal communication" notes.
- ruch2024biotxplorer and gobeill2020sibils (the system under study and its search engine) are cited in the third person ("BiotXplorer's own authors", L115), as in the submission version.
- The code link is the anonymous.4open.science placeholder.

---------------------------------------------------------------------------------------------------

## (d) MUST FIX

1. **tex L382-383** (lai2023biorex, lai2025bioredirect). The system names were dropped, and our own "quarter never co-mentioned" measurement now follows their citation as if it explained their scores.
   - Current text:
     `$57.5$; systems that read the whole abstract reach $74.1$ to $75.3$`
     `\citep{lai2023biorex,lai2025bioredirect}, in part because a quarter of the related pairs are never co-mentioned in one sentence (Appendix~\ref{app:bioredep}).`
   - Replacement:
     `$57.5$; systems that read the whole abstract reach $74.1$ to $75.3$`
     `(PubMedBERT, BioREx and BioREDirect; \citealp{lai2023biorex,lai2025bioredirect}). A quarter of the related pairs are never co-mentioned in one sentence, which caps any sentence-scope verifier at recall $0.749$ (Appendix~\ref{app:bioredep}).`
   - This mirrors the appendix wording.

2. **tex L600-601** (dsouza2025mining, farrell2024landscape). Mis-attribution: D'Souza mines species, locations, habitats and ecosystems, not interactions, and Farrell is a review. The wording is stronger than paperA's.
   - Current text:
     `take the interacting pair as the annotation unit. Large language models extract interactions at`
     `scale \citep{keck2025extracting,farrell2024landscape,dsouza2025mining,zou2026llm};`
   - Replacement:
     `take the interacting pair as the annotation unit. Large language models extract interactions at`
     `scale \citep{keck2025extracting,zou2026llm} and are applied to ecological text mining more broadly \citep{farrell2024landscape,dsouza2025mining};`

3. **tex L578-579** (farkas2010conll). The CoNLL-2010 task covers biological text AND Wikipedia, so calling it "biomedical hedge detection" is wrong. On Wikipedia only the winner was a bag-of-words sentence classifier.
   - Current text:
     `span tagger used to flag tweets loses to a tweet classifier \citep{magge2021deepademiner}, and in`
     `biomedical hedge detection cue taggers read as ``at least one cue'' won on biological text while`
     `direct sentence classifiers won on Wikipedia \citep{farkas2010conll}.`
   - Replacement:
     `span tagger used to flag tweets loses to a tweet classifier \citep{magge2021deepademiner}, and in`
     `the CoNLL-2010 hedge-detection task cue taggers read as ``at least one cue'' won on biological text while`
     `a direct sentence classifier won on Wikipedia \citep{farkas2010conll}.`

**Bib entries to remove for lack of verification: none.** All 21 cited reframe_extra entries are verified. The 2 NOT FOUND entries are confirmed on the ACL Anthology and Crossref.

### Recommended (non-blocking)

- **R1, L441 and L691:** `\citep{li2021dual}` becomes `\citep[cf.][]{li2021dual}`. The paper supports "instance supervision is an upper bound" in whole-slide images; the BioRED expectation is our analogy.
- **R2, L572:** `\citep{hoffmann2011knowledge}` becomes `\citep{hoffmann2011knowledge,zeng2015distant}`. Zeng (verified, already in reframe_extra.bib) is the "maximum over instances" source.
- **R3, L602:** `\citet{zou2026llm} first ask whether a text contains an interaction and standardise names post hoc, without checking extractions against their text.` becomes `\citet{zou2026llm} first ask whether a citizen-science comment contains an interaction and standardise names post hoc; their workflow has no step that checks each extraction against its text.`
- **R4, L551:** "marks the target pair" becomes "masks the target pair". BioBERT replaces the entities with @GENE$-style tags.
- **R5, L587:** "article triage" becomes "document triage" (BC6 term; BC II IAS used abstracts).
- **R6, L212-213:** put "with the same recipe" before the distillation citations so the hyperparameters do not read as taken from them.
- **R7:** the 74.1-75.3 range (OLD number) can only be checked in BioREDirect's full-text table; its abstract gives no F1. Re-confirm before camera-ready.
- **R8:** mraz2026fewshot is an arXiv preprint. Re-check for a peer-reviewed version before camera-ready.
- **R9:** the duplicate `\bibliographystyle` (tex L713 plus acl.sty) causes the BibTeX "Illegal, another \bibstyle" message. It is harmless; drop L713 for a clean .blg.
