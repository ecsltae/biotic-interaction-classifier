# agentB: Q2 (triage unit in curation pipelines) and Q4 (sentence vs pair annotation semantics, IAA)

Status: complete (started 01:50, finished 02:13 CEST, 2026-10-05). Sources read are marked per claim; 22 works in agentB.bib.
Decisions taken without the user (unattended run) are flagged "DECISION".

## Q2. Which filter question do curation / database-population pipelines ask?

Legend for "filter question": DOC = "is this document relevant / does it describe an interaction?";
SENT = "does this sentence describe an interaction?"; PAIR = "do these two entities interact (here)?".

### Q2-A. Biomedical side

**krallinger2008overview** -- Krallinger, Leitner, Rodriguez-Penagos, Valencia. "Overview of the protein-protein
interaction annotation extraction task of BioCreative II." Genome Biology 9(Suppl 2):S4, 2008.
DOI 10.1186/gb-2008-9-s2-s4. Verified: Crossref content negotiation (https://doi.org/10.1186/gb-2008-9-s2-s4);
full text read from PMC2559988 (E-utilities efetch).
- Filter questions: four subtasks mirror a curation pipeline: IAS = DOC ("classification and ranking of PubMed
  abstracts, based on whether they are relevant to protein interaction annotation or not"); IPS = PAIR
  ("extraction of binary protein-protein interaction pairs from full-text articles"); ISS = pair-conditioned
  passage retrieval, NOT a free-standing sentence classifier: "For the ISS, participants had to provide, for each
  protein interaction pair, a ranked list of a maximum of five evidence passages describing their interaction."
  [full text, "Protein-protein interaction task" and "Interaction sentences subtask" sections].
- WHAT IT SHOWS: the curation workflow is decomposed as DOC triage -> PAIR extraction -> pair-anchored evidence
  sentences; the sentence-level resources offered as auxiliary training data were pair-agnostic (Prodisen:
  "Each sentence from a given abstract was manually classified regarding whether it contained interaction
  descriptions of genes and proteins.") and the organisers note they "differ from the passages extracted by the
  interaction database curators" [full text, Methods, "Interaction pair subtask, interaction method subtask,
  and interaction sentences subtask datasets"]. Among listed obstacles: "5. Difficulties in extracting the
  associations and in the handling of coordination (multiple interaction pairs) from a single sentence."
  [full text, Discussion]. Numbers [abstract]: "In the interaction article detection subtask, the top scoring team
  reached an F-score of 0.78. In the interaction pair extraction and mapping to SwissProt, a precision of 0.37
  (with recall of 0.33) was obtained."
- HOW OUR CLAIM RELATES: supports the framing (curation databases store pairs; sentence evidence is wanted per
  pair; multi-pair sentences are a named difficulty). Does not pre-empt the controlled sentence-vs-pair comparison.

**leitner2010overview** -- Leitner, Mardis, Krallinger, Cesareni, Hirschman, Valencia. "An Overview of
BioCreative II.5." IEEE/ACM Transactions on Computational Biology and Bioinformatics 7(3):385-399, 2010.
DOI 10.1109/TCBB.2010.61. Verified: Crossref content negotiation; abstract read via Europe PMC (PMID 20704011).
Full text not read (paywalled).
- Filter questions: DOC then entity then PAIR [abstract]: "The tasks were to rank articles for curation based on
  curatable protein-protein interactions; to identify the interacting proteins (using UniProt identifiers) in the
  positive articles (61); and to identify interacting protein pairs."
- WHAT IT SHOWS [abstract]: "For article classification, the best AUC iP/R was 0.70; ... for interacting protein
  pairs, the top (filtered, mapped) recall was 0.42 and AUC iP/R was 0.29."
- HOW OUR CLAIM RELATES: context only; shows the document-level decision is a separate, easier task than the pair
  decision a database needs. Optional citation (krallinger2008overview covers the same point with full text).

**krallinger2011protein** -- Krallinger, Vazquez, Leitner, et al. "The Protein-Protein Interaction tasks of
BioCreative III: classification/ranking of articles and linking bio-ontology concepts to full text." BMC
Bioinformatics 12(Suppl 8):S3, 2011. DOI 10.1186/1471-2105-12-S8-S3. Verified: Crossref; full text read from
PMC3269938.
- Filter question: DOC only (Article Classification Task, ACT): "Detecting articles describing complex biological
  events like PPIs was addressed in the Article Classification Task (ACT), where participants were asked to
  implement tools for detecting PPI-describing abstracts." [abstract]
- WHAT IT SHOWS: document-level triage reaches "the highest Matthew's Correlation Coefficient (MCC) score measured
  was 0.55 at an accuracy of 89% and the best AUC iP/R was 68%" [abstract]. Also gives document-level IAA (see Q4).
- Precedent for reading the document decision off a pair-extraction pipeline [full text, "Individual system
  descriptions", Team 65 (OntoGene)]: "Features include lexical items, MeSH annotations, plus crucially a score
  delivered by their PPI detection pipeline. Two runs used only results of their protein-protein interaction
  detection pipeline (as developed for BC II.5), for comparison." and "These results prove that an NLP-based
  pipeline for PPI extraction definitely provides a positive contribution towards the solution of the ACT task."
  In the ACT results table, Team 65 runs 1, 2, 5 have AUC iP/R 63.85, 63.89, 62.39 and runs 3, 4 both have 41.74
  (F 43.34 and 45.08 vs 50.83-59.82). INFERENCE, not stated in the text: runs 3 and 4 are probably the two
  pipeline-only runs, i.e. the document decision read off pair extraction alone was clearly worse than the
  trained document classifier in that (uncontrolled) setting.
- HOW OUR CLAIM RELATES: example of a triage filter that asks the DOC question; contrast with our PAIR filter.
  Also the closest pre-existing test of "document decision from pair evidence": positive as a feature, apparently
  weak on its own, with a rule-based PPI pipeline and no shared training data -- the controlled comparison we run
  (same teacher, data, encoder) is what is missing.

**islamajdogan2019overview** -- Islamaj Dogan, Kim, Chatr-aryamontri, et al. "Overview of the BioCreative VI
Precision Medicine Track: mining protein interactions and mutations for precision medicine." Database 2019:bay147,
2019. DOI 10.1093/database/bay147. Verified: Crossref; full text read from PMC6348314.
DECISION: key uses the compound surname "Islamaj Dogan"; rename to dogan2019overview if the bib style prefers.
- Filter questions: DOC triage then PAIR extraction [abstract]: "(i) document triage task, focused on identifying
  scientific literature containing experimentally verified protein-protein interactions (PPIs) affected by genetic
  mutations and (ii) relation extraction task, focused on extracting the affected interactions (protein pairs)."
  Annotation was nested: "curators marked 'relevant' if it described PPIs affected by mutations, 'not relevant'
  otherwise, and if 'relevant', they identified the interacting pair of proteins with their Entrez Gene ID"
  [full text, corpus section].
- WHAT IT SHOWS [abstract]: "for the triage task, the best F-score was 69.06% ... For the relation extraction task,
  when taking homologous genes into account, the best F-score was 37.73%". The RE baseline is pure sentence
  co-occurrence: "a relation was predicted if two gene entities were found in the same sentence" [full text,
  baseline paragraph]. One team's pair extractor leaned on the triage positive: sentence co-occurrence "was quite
  effective for the task, given that all documents analyzed could be assumed to describe at least one protein
  interaction in the context of a mutation." [full text, Melbourne READ-Biomed team description].
- Partial precedent for reading the document decision off pair-level predictions: one triage system (Team 433,
  Florida State University) fed a pair/triplet model's outputs into its document classifier: "Features were also
  extracted based on a model previously developed by our group for predicting PPI triplets. A triplet consists of
  two protein names and an interaction word that are all contained in the same sentence. ... The predicted
  probabilities generated by this model were incorporated by taking normalized counts of the number of predicted
  triplet probabilities lying in equally spaced bins." [full text, "Individual system descriptions", Team 433].
  These are features among n-grams in a gradient-boosted model, not a max over pair verifications, and no
  ablation against a directly trained triage model with the same data is reported in the overview.
- HOW OUR CLAIM RELATES: closest biomedical analogue of our setting (document-level "describes at least one
  interaction" label plus pair labels on the same documents), but the two tasks are evaluated separately; no
  controlled comparison of reading the document label off pair predictions. Cite Team 433's design as "pair
  evidence has been used as triage features" so a reviewer cannot say we missed it. Also supports the known point
  that triage and pair extraction are distinct and pair extraction is much harder.

**wiegers2009text** -- Wiegers, Davis, Cohen, Hirschman, Mattingly. "Text mining and manual curation of
chemical-gene-disease networks for the Comparative Toxicogenomics Database (CTD)." BMC Bioinformatics 10:326,
2009. DOI 10.1186/1471-2105-10-326. Verified: Crossref; full text read from PMC2768719.
- Filter question: DOC ranking (prioritise abstracts for curation), with a sentence co-occurrence feature inside the
  document score: ranking criteria include "Co-occurrence of action terms and actors in the same sentence."
  [full text, "Document ranking"].
- WHAT IT SHOWS [abstract]: re-ranking documents "resulting in increases in mean average precision (63% for the
  baseline vs. 73% for a rule-based re-ranking)". Also gives a two-level curator agreement (document disposition
  vs extracted interactions; see Q4).
- HOW OUR CLAIM RELATES: example of a DOC-level filter in a pair-populated database (CTD stores chemical-gene
  interactions); contrast.

**bravo2015extraction** -- Bravo, Pinero, Queralt-Rosinach, Rautschka, Furlong. "Extraction of relations between
genes and diseases from text and large-scale data analysis: implications for translational research." BMC
Bioinformatics 16:55, 2015 (article number from PMC record; Crossref gives volume 16, no pages). DOI
10.1186/s12859-015-0472-9. Verified: Crossref; full text read from PMC4466840. (BeFree system; GAD corpus.)
- Filter question: PAIR within a sentence: "TRUE sentences contain real relationship between the entities
  analysed, in contrast with FALSE sentences where the two entities co-occur, but there is no semantic relationship
  between them." [full text, Methods, evaluation paragraph]. GAD-corpus negatives are co-occurring pairs not curated
  as associations: "a gene and a disease that co-occur in a sentence but are semantically not associated, we
  selected the sentences with co-occurrences between a disease and a gene found by the BioNER system that were not
  annotated by GAD curators as gene-disease associations." [full text, "GAD corpus"].
- WHAT IT SHOWS [abstract]: pair-level extraction populates a database (DisGeNET); "only a small proportion of this
  dataset is actually recorded in curated resources (2%)".
- HOW OUR CLAIM RELATES: precedent for a pair-conditioned filter over co-occurrence candidates (our design);
  distantly-labelled negatives = "co-occur but not curated", same noise caveat as riedel2010modeling.

**ding2001mining** -- Ding, Berleant, Nettleton, Wurtele. "Mining MEDLINE: abstracts, sentences, or phrases?" In
Biocomputing 2002 (Pacific Symposium on Biocomputing 7), pp. 326-337. World Scientific. DOI
10.1142/9789812799623_0031. Verified: Crossref (gives year 2001, booktitle "Biocomputing 2002"; the PDF header
reads "Pacific Symposium on Biocomputing 7:326-337 (2002)"; DECISION: bib keeps Crossref's year 2001, change to
2002 if the paper's style cites PSB by meeting year); full text read from
https://psb.stanford.edu/psb-online/proceedings/psb02/ding.pdf.
- Filter question: co-occurrence of a given PAIR of query terms inside a text unit (abstract, sentence, phrase,
  sentence pair); compares which unit to use.
- WHAT IT SHOWS [full text, Sec. 4 "Data Analysis"]: "Table 2 suggests a trend of increasing precision for smaller
  text units, except for sentence pairs which rated poorly overall. Phrases, the smallest unit, had the highest
  precision. Precision differences were significant at the 0.05 level except in the case of abstracts vs.
  sentences"; "Abstracts measured about equal to sentences in effectiveness." [Discussion and Conclusion]: for
  term pairs split across two sentences "the precision was a mere 0.05".
- HOW OUR CLAIM RELATES: earliest controlled comparison of the text unit for interaction mining; it varies the
  window, not the question asked of a classifier, so it does not pre-empt us. Cite for "the unit of decision
  matters" being known.

**pyysalo2008comparative** (also Q4) -- see Q4 entry; relevant to Q2 because it formalises how a perfect sentence
filter bounds pair precision (I/EP).

### Q2-B. Biodiversity / interspecies side

**lim2016minter** -- Lim, Li, Chng, Nagarajan. "@MInter: automated text-mining of microbial interactions."
Bioinformatics 32(19):2981-2987, 2016. DOI 10.1093/bioinformatics/btw357. Verified: Crossref content negotiation;
abstract via Europe PMC (PMID 27312413); full text read with WebFetch on
https://academic.oup.com/bioinformatics/article/32/19/2981/2196283, and every quote below re-checked word for word
against the publisher PDF text (PDF retrieved through WebFetch, converted with pdftotext).
- Filter question: DOC ("does this abstract report a microbial interaction?"), then every co-mentioned species pair
  in a positive abstract becomes a candidate: "the SVM classifier in @MInter can only identify abstracts that are
  likely to report a microbial interaction. ... Once an abstract is identified, all pairs of species reported in
  it then need to be considered as candidate microbial interactions, as abstracts frequently report multiple
  interactions." [full text, Sec. 2.2 "Training of the @MInter classifier"]. Gold standard annotated "on two
  levels": species-pair level and abstract level [full text, Sec. 2.1.3]; single annotator ("abstracts were
  manually scanned by a single annotator"), so no IAA.
- WHAT IT SHOWS: the document filter is good but the pairs read off it are not: [abstract] "@MInter was able to
  detect abstracts pertaining to one or more microbial interactions with high specificity (specificity = 95%,
  AUC = 0.97). Despite challenges in identifying specific microbial interactions in an abstract (interaction level
  recall = 95%, precision = 25%)". Explanation [full text, Sec. 3.2]: "@MInter's species-level precision was found
  to be low despite being the best across all methods. This is partly because it identifies abstracts containing
  interactions and reports all pairs of species names in such abstracts."
- Species level means corpus-level pairs: "species level annotation considered whether any of the abstracts
  containing a species pair described a true interaction" [Sec. 2.1.3]; "Overall, 62 interacting pairs were
  identified (out of 735)". Table 1 ("Performance summary for different methods at the species level"):
  precision 7% for 'All Positives' (every co-mentioned pair), 25% for the @MInter SVM, 15% pattern scanner,
  10% naive Fisher co-occurrence test (Freilich et al.), 15% Fisher with inhibition terms; recall 100/95/24/51/50.
- HOW OUR CLAIM RELATES: strongest prior support, in the biotic domain, for "a sentence/document filter has
  limited pair precision" (their 25% species-pair precision from an abstract-level filter, 7% for all co-mentioned
  pairs, vs our 0.724 pair precision for a perfect sentence-level filter). It does NOT test reading the document decision off a pair verifier, so it does not pre-empt the
  controlled comparison. Must be cited if we claim the pair-precision ceiling as a finding: it is known.

**thessen2014knowledge** -- Thessen, Parr. "Knowledge Extraction and Semantic Annotation of Text from the
Encyclopedia of Life." PLoS ONE 9(3):e89550, 2014. DOI 10.1371/journal.pone.0089550. Verified: Crossref; full text
read from PMC3940440.
- Filter question: section-level DOC filter (text under EOL "Associations"/"Trophic Strategy"/"General
  Ecology"/"Habitat" subchapters; "We focused on these subchapters so we could extract information from text
  specifically describing ecological interactions" [full text, Methods, "Information extraction and network building"]); every taxon name found there is linked
  to the page's taxon, i.e. the pair label is inherited from the passage label.
- WHAT IT SHOWS [full text, Results, "Association network", and Table 4 for the GNRD-only baseline]: "For the task of identifying ecological interactions
  in text, GNRD alone had a precision of 0.477, a recall of 0.957 and an F1 Score of 0.636"; with the section
  filter "Our automated methods had an overall precision of 0.844, a recall of 0.930 and an F1 Score of 0.885".
  Errors: "The largest source of false positives was the inclusion of higher taxon names for species that were
  mentioned (89%)." Discussion: "One cannot assume that all of the taxa mentioned on an EOL taxon page have an
  ecological relationship with the topic of that page. Taxa are mentioned for several reasons including
  comparison, taxonomic relationships and in discussion of a common phenomenon". Also a pair-level IAA (Q4).
- HOW OUR CLAIM RELATES: early biodiversity precedent for passage-level filtering with pair labels inherited from
  the passage; supports that co-mention is not interaction; GloBI-adjacent (EOL interaction data later fed GloBI;
  ref. 40 of the paper is a GloBI talk). Does not test pair-conditioned verification.

**elkhettari2023building** -- El Khettari, Quiniou, Chaffron. "Building a Corpus for Biomedical Relation Extraction
of Species Mentions." Proceedings of the 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared
Tasks (BioNLP 2023), pp. 248-254, Toronto. ACL Anthology 2023.bionlp-1.21, DOI 10.18653/v1/2023.bionlp-1.21.
Verified: https://aclanthology.org/2023.bionlp-1.21.bib; full text read from the Anthology PDF.
- Filter question: PAIR = SENT by construction. Species-Species Interaction (SSI) corpus, binary label per
  sentence with masked species: "We focused on sentences where exactly two species are mentioned to maximize the
  probability that a binary relation is indeed expressed." [full text, Sec. 3.2]; label 1 for "sentences containing
  a relation between at least two species mentions" [Sec. 3.2]; guideline "Multiple relations: Only consider the
  potential presence or absence of a relation between the masked species." 442 LINNAEUS + 557 PubTator sentences.
  No IAA reported (none found in the paper).
- HOW OUR CLAIM RELATES: the only species-species RE corpus found sidesteps the sentence-vs-pair distinction by
  restricting to two-species sentences, exactly the regime where our pair gain vanishes (BioRED gain only on
  sentences naming >2 entities). Supports the claim that the distinction matters only in multi-taxon passages;
  does not pre-empt us.

**abdelmageed2022biodivnere** -- Abdelmageed, Loeffler, Feddoul, Algergawy, Samuel, Gaikwad, Kazem, Koenig-Ries.
"BiodivNERE: Gold standard corpora for named entity recognition and relation extraction in the biodiversity
domain." Biodiversity Data Journal 10:e89481, 2022. DOI 10.3897/BDJ.10.e89481. Verified: Crossref; full text read
from PMC9836593.
- Filter question: PAIR in sentence (one entity pair per sentence copy): "an RE corpus should be designed in a way
  that each sentence contains exactly two tags. We generated all possible combinations for sentences with more
  than two tags, including exactly two tags." [full text, "BiodivRE construction pipeline", Initial
  Construction]. Pilot IAA: "Two of the authors annotated the same 50 sentences that were randomly picked.
  Afterwards, we calculated the inter-rater agreement (Kappa's score), which resulted in 0.94." [full text,
  Annotation Process].
- Caveat: it contains no organism-organism relations: "Other category pairs that the BiodivOnto support do not
  appear in the text used for creating the RE corpus. For example, ORG-ORG and ORG-LOC." [full text, "BiodivRE
  Characteristics"].
- HOW OUR CLAIM RELATES: shows that biodiversity RE corpora pose the pair question by duplicating multi-entity
  sentences, but none covers biotic interactions; supports "no biotic pair benchmark existed". Minor.

**thieu2012literature** -- Thieu, Joshi, Warren, Korkin. "Literature mining of host-pathogen interactions:
comparing feature-based supervised learning and language-based approaches." Bioinformatics 28(6):867-875, 2012.
DOI 10.1093/bioinformatics/bts042. Verified: Crossref; abstract via Europe PMC (PMID 22285561); full text read
from the publisher PDF (retrieved through WebFetch, converted with pdftotext).
- Filter question: DOC, with sentence-derived features [abstract]: "we introduce and compare two new approaches to
  automatically detect whether the title or abstract of a PubMed publication contains HPI data, and extract the
  information about organisms and proteins involved in the interaction. ... The SVM models are trained on the
  features derived from the individual sentences." Result [abstract]: "The most accurate, feature-based, approach
  achieved 66-73% accuracy, depending on the test protocol."
- Cascade DOC -> SENT with one classifier [full text, Sec. 2.1 "Feature-based approach"]: "the same classifier is
  used in our method twice: first, to classify whether an abstract is HPI-relevant, and secondly, if it is
  relevant, to determine which sentences of the abstract are most likely to contain the HPI-relevant data. For
  the latter, we generate a feature vector for each sentence and use it as an input to the SVM classifier."
  The baseline is also unit-stacked [Sec. 2.3 "Assessment"]: "in the naive approach, an HPI-containing abstract
  is classified by (i) determining whether the abstract contains any PPIs; and (ii) determining whether it has at
  least one host and one pathogen organism." Sentences are scored as "complete" (host and pathogen proteins and
  organisms "in one sentence") or "partial" (information "split into multiple sentences").
- HOW OUR CLAIM RELATES: host-pathogen precedent where the question asked is DOC then SENT, never "do THESE
  host and pathogen interact"; the pair is extracted afterwards from positive sentences. Contrast; it shows the
  sentence-question pipeline our proposal replaces.

**scheepens2024large** -- Scheepens, Millard, Farrell, Newbold. "Large language models help facilitate the automated
synthesis of information on potential pest controllers." Methods in Ecology and Evolution 15(7):1261-1273, 2024.
DOI 10.1111/2041-210X.14341. Verified: Crossref (metadata + abstract). Full text NOT read (Wiley returns 403 to
this server). Not already cited (farrell2024landscape is a different work by an overlapping author).
- Filter question [abstract]: per-abstract entity-and-ROLE extraction on a topic-preselected corpus, not pair
  verification: "we analyse the ability of GPT-4 to extract information about invertebrate pests and pest
  controllers from abstracts of articles on biological pest control, using a bespoke, zero-shot prompt." Result
  [abstract]: "species and geographic locations are extracted with F1-scores of 99.8% and 95.3%, respectively, and
  highlight that the model can effectively distinguish between ecological roles of interest such as predators,
  parasitoids and pests."
- HOW OUR CLAIM RELATES: example of LLM ecology extraction evaluated on entities and roles; no pair-level
  (controller-pest link) score is given in the abstract. Contrast; optional.

### Q2-C. Filter question used by works ALREADY in the bibliography (described by key; not in agentB.bib)

| Key | Unit and filter question | Evidence read |
|---|---|---|
| keck2025extracting | Pre-filter by co-occurrence + keyword on PARAGRAPHS ("We refined this dataset by retaining only the paragraphs that included at least two distinct taxonomic names and one keyword associated with species interactions."), then GPT-4o generates PAIRS: "The prompt template was specifically designed to extract all pairwise species interactions present within the paragraph ... The output consisted of a four-column table where the first two columns contained the names of the two species involved in the interaction". Validation: "The manual annotation of 500 paragraphs in the validation set identified 327 biological interactions. Our automated approach managed to identify 229 of these (true positives), while making 27 errors (false positives). This corresponds to an accuracy of 89.5% and a recall of 70.0%." The abstract calls the same number a precision: "high sensitivity (70.0%) and excellent precision (89.5%)" (229/256 = 0.895). | bioRxiv v1 (posted 27 Jan 2025) PDF text, verified word for word: abstract; Results (validation paragraph); Methods "Data extraction" and corpus-filtering paragraph |
| zou2026llm | Unit = a citizen-science COMMENT (eBird, GBIF). Step 1 is a comment-level binary decision, then species and type: LLMs "extract the three key types of information ...: whether a comment contains interaction, what species were involved in the interaction, and what type of interaction was recorded." eBird: "we first trained a GPT-4o model on 544 eBird comments, half of which contained information on species interactions and half of which did not. Within the test set (Supplemental Material), the LLM correctly identified 76% of all comments containing interactions and 96% of all comments that do not contain interactions." | bioRxiv v1 (posted 10 Feb 2026) PDF text, verified word for word: Figure 1 caption; Box 1 "Case study 1: eBird comments" |
| dimitrova2020semantic | Unit = TABLE (document-component triage by ontology term matching): "Using the Pensoft Annotator, a text-to-ontology mapping tool, we were able to detect tables that could contain biotic interactions." "Annotation of biotic interactions via the Pensoft Annotator helped to identify 233 tables possibly containing biotic interactions out of the 6993 tables that were processed." "Currently, GloBI has indexed 2378 interactions, extracted from a subset of 46 of the 233 tables." Pairs then come from table rows, untyped and undirected: "The exact interaction types between the species were not determined, instead the general term labelled "interacts with" was used."; "we can not automatically detect which species is the host and which is the parasite." | BISS 4:e59036 abstract, DOI 10.3897/biss.4.59036, text from Crossref metadata (Methods, Results, Discussion paragraphs) |
| cuzick2023interspecies | Manual curation unit is inherently a PAIR: "we developed the concept of a 'metagenotype,' which represents the combination of a pathogen genotype and a host genotype" | full text PMC10319440, section "Developing the metagenotype to capture interspecies interactions" |
| luo2022biored | BioRED: relations annotated at document level between concept pairs; IAA in Q4-B. | full text PMC9487702 |

Takeaway for the paper: the biodiversity pipelines found ask either a passage/table/document question
(Dimitrova, Thessen & Parr, @MInter, Keck's keyword pre-filter) or ask a generative model to list pairs (Keck,
Zou); none trains a pair-conditioned verifier and compares it with a sentence-level classifier on the same data.

Checked, not reported as key works (DECISION):
- Thessen, Cui, Mozzherin 2012, "Applications of Natural Language Processing in Biodiversity Science", Advances in
  Bioinformatics 2012:391574, DOI 10.1155/2012/391574 (verified, full text PMC3364545): a general review of NLP for
  taxonomic names and morphological characters; no species-interaction pipeline, no filter question to report.
  Usable only as a generic background citation; not put in the .bib.
- Le Guillarme and Thuiller 2023, "A practical approach to constructing a knowledge graph for soil ecological
  research", European Journal of Soil Biology 117:103497, DOI 10.1016/j.ejsobi.2023.103497 (verified on Crossref;
  preprint abstract read via Europe PMC PPR624901): integrates "(semi-)structured data sources" into a trophic
  knowledge graph; it is not a text-mining pipeline, so there is no filter question. Not put in the .bib.
- Castro et al. 2024 (Ecological Informatics 82:102742, DOI 10.1016/j.ecoinf.2024.102742): LLM extraction of
  species DISTRIBUTION data (document classification + region extraction), not interactions. Not reported.
- An eBird "bird interactions" LLM case study surfaced in search is an EcoEvoRxiv preprint (Gallois 2025, per the
  Crossref reference list of Mammides et al. 2026, Conservation Biology); not peer-reviewed, not read. Not reported.
- Pafilis et al. 2013 (SPECIES/ORGANISMS): NER only; not relevant to the filter question; not checked further.

## Q4. Sentence-level vs pair-level labels; inter-annotator agreement (IAA); dropping UNSURE

### Q4-A. What counts as an interaction, and at what unit (pair vs sentence vs document)

**pyysalo2008comparative** -- Pyysalo, Airola, Heimonen, Bjoerne, Ginter, Salakoski. "Comparative analysis of five
protein-protein interaction corpora." BMC Bioinformatics 9(Suppl 3):S6, 2008. DOI 10.1186/1471-2105-9-S3-S6.
Verified: Crossref; full text read from PMC2349296.
- WHAT IT SHOWS: no agreed definition of a PPI across AIMed, BioInfer, HPRD50, IEPA, LLL: "there is no general
  consensus regarding PPI annotation and consequently resources are largely incompatible" [abstract]; "the F-score
  performance of a state-of-the-art PPI extraction method varies on average 19 percentage units and in some cases
  over 30 percentage units between the different evaluated corpora" [abstract]. Corpora differ in how much
  sentence-level pre-filtering they embed: "the fraction of sentences with no annotated interactions varies from
  zero to more than two thirds. This may reflect the corpus authors' different views regarding the appropriate
  starting point for extraction, in particular on how aggressively non-relevant sentences can be filtered out."
  [full text, Results, corpus statistics]. Key formula for our "perfect sentence filter" argument: "the average
  number of interactions divided by the average number of entity pairs per sentence (below abbreviated I/EP)
  equals the precision of the co-occurrence method ... As more proteins are annotated, we would not expect I to
  grow more than linearly, while EP grows quadratically." [same section]. Table 3 gives per-sentence entity pairs
  (AIMed 3.0, BioInfer 9.4, HPRD50 3.0, IEPA 1.7, LLL 4.3) and interactions (0.5, 1.3, 1.1, 0.7, 2.1).
  Directness differs widely: "the corpora do not aim to separate direct interactions from indirect ones" [full
  text, qualitative analysis]. IAA: none reported for the corpora; their own 2-annotator analysis of 50
  interactions per corpus counts disagreements "as half a point" (Table 5 caption), no kappa.
- HOW OUR CLAIM RELATES: makes "co-occurrence precision = interactions / candidate pairs, falling quadratically
  with entities per sentence" a KNOWN quantity (cite it next to our 0.724). Supports that the gain should
  concentrate in sentences with more entities (our BioRED >2-entity finding is consistent with it).

**pyysalo2007bioinfer** -- Pyysalo, Ginter, Heimonen, Bjoerne, Boberg, Jaervinen, Salakoski. "BioInfer: a corpus for
information extraction in the biomedical domain." BMC Bioinformatics 8:50, 2007. DOI 10.1186/1471-2105-8-50.
Verified: Crossref (see .bib); full text read from PMC1808065.
- WHAT IT SHOWS: a major PPI corpus released without IAA: "For the current initial release of the corpus, we did
  not undertake the effort to measure inter-annotator agreement, a measure of the stability of the annotation
  scheme." [full text, "Quality of the annotation"]; disagreements were resolved by
  discussion ("the matter was discussed until an agreement was reached" [full text, Methods]).
- HOW OUR CLAIM RELATES: context: classic pair corpora often report no IAA, so a dual-label set with reported
  agreement per label type is not the norm. Optional citation.

**gurulingappa2012development** -- Gurulingappa, Rajput, Roberts, Fluck, Hofmann-Apitius, Toldo. "Development of a
benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports."
Journal of Biomedical Informatics 45(5):885-892, 2012. DOI 10.1016/j.jbi.2012.04.008. Verified: Crossref; abstract
via Europe PMC (PMID 22554702). Full text NOT read (ScienceDirect blocks this server), so no IAA number is given.
- WHAT IT SHOWS [abstract]: double annotation and harmonisation ("The documents are systematically double annotated
  in various rounds to ensure consistent annotations. The annotated documents are finally harmonized to generate
  representative consensus annotations."); its demonstration task is the SENTENCE question: "the corpus was
  employed to train and validate models for the classification of informative against the non-informative
  sentences. A Maximum Entropy classifier ... resulted in the F1 score of 0.70". The corpus also carries
  drug-adverse-effect relation annotations at sentence level (stated by luo2022biored [full text, related-work
  table]: "ADE annotates the relations (i.e. drug-ADE and drug-dosage relations) at the sentence level").
- HOW OUR CLAIM RELATES: the best-known biomedical corpus carrying both a sentence label and pair labels; it was
  used for the sentence question and (by later work) for pair RE, but the paper does not compare the two. If
  cited for IAA, someone must read the full text first.

**vanmulligen2012euadr** -- van Mulligen, Fourrier-Reglat, Gurwitz, Molokhia, Nieto, Trifiro, Kors, Furlong.
"The EU-ADR corpus: Annotated drugs, diseases, targets, and their relationships." Journal of Biomedical
Informatics 45(5):879-884, 2012. DOI 10.1016/j.jbi.2012.04.004. Verified: Crossref; abstract via Europe PMC (PMID
22554700). Full text NOT read (paywalled; no OA copy reachable), so no IAA number is given.
- WHAT IT SHOWS [abstract]: pair-level relation annotation by three experts on 100 abstracts per relation type;
  "The agreement figures achieved show that the inter-annotator agreement is much better than the agreement with
  the system provided annotations." bravo2015extraction [full text, Methods] adds that each relation "is classified
  according to its level of certainty as: positive association (PA), negative association (NA), speculative
  association (SA) and false association (FA)" and that they "considered the relationships that result from the
  consensus annotation of two experts".
- HOW OUR CLAIM RELATES: precedent for keeping an explicit "speculative" class rather than dropping uncertain
  pairs (contrast with our UNSURE removal). IAA numbers unverified.

### Q4-B. Reported IAA by unit of decision (all numbers quoted from text read)

| Unit | Work | Measure and value | Location |
|---|---|---|---|
| Document ("PPI-relevant abstract?") | krallinger2011protein | MINT vs BioGrid "overlap 96%, Cohen's Kappa = 0.85"; "for MINT vs. expert, 92% overlap, Kappa = 0.69; for BioGrid vs. expert, 91%, Kappa = 0.69"; all three groups agree on "85.5% of all abstracts" | full text, "ACT inter-annotator and manual classification time analysis" |
| Document disposition AND extracted interactions | wiegers2009text (CTD) | Step 1 (curate this article or not): "the curators agreed on the disposition of 86/112 articles (77%) and had an average pair-wise agreement of 85%"; Step 2 (interactions vs adjudicated gold): "91% of the interactions extracted by CTD biocurators were judged by the lead curator to be correct (average precision = 0.91). Average recall was 0.71." | full text, "Inter-Biocurator Agreement" |
| Taxon-taxon association (pair) from passages | thessen2014knowledge | "Agreement between annotators was 0.840 (Fleiss' Kappa ...)" (3 annotators) | full text, Results, "Association network"; method in Methods, "Associations network" |
| Entity pair in sentence (biodiversity, non-biotic) | abdelmageed2022biodivnere | "Kappa's score ... resulted in 0.94" on 50 pilot sentences, 2 annotators | full text, BiodivRE "Annotation Process" |
| Drug-drug pair in sentence | herrerozazo2013ddi | "The agreement was almost perfect (Kappa up to 0.96 and generally over 0.80), except for the DDIs in the MedLine database (0.55-0.72)." | abstract only |
| Document-level relation (BioRED, already cited) | luo2022biored | "we computed the inter-annotator-agreement (IAA) for the entity, relation and novelty annotations, where we achieved 97.01, 77.91 and 85.01%, respectively." Every abstract annotated by three annotators; unresolved cases "reviewed by another senior annotator" | full text, "Data characteristics" and Table 4; "Annotation process" |
| Sentence-only (species-species, 2 species/sentence) | elkhettari2023building | none reported | full text |
| Abstract + species-pair (microbial) | lim2016minter | none: "a single annotator" | full text, Sec. 2.1.3 |

Reading of the table (for the paper): agreement is reported per unit, never as a matched sentence-vs-pair
comparison on the same items in any work found. The only same-items two-level figure is CTD (document disposition:
85% average pairwise agreement; interaction level: individual curators reach recall 0.71 and precision 0.91 against
an adjudicated gold). The two levels use different measures, so this at most suggests that interaction-level
decisions are the less reliable ones for humans too; one 112-article study, not a general finding.

**herrerozazo2013ddi** -- Herrero-Zazo, Segura-Bedmar, Martinez, Declerck. "The DDI corpus: An annotated corpus with
pharmacological substances and drug-drug interactions." Journal of Biomedical Informatics 46(5):914-920, 2013.
DOI 10.1016/j.jbi.2013.07.011. Verified: Crossref; abstract via Europe PMC (PMID 23906817). Full text not read.
- WHAT IT SHOWS: pair-level DDI annotation within sentences (luo2022biored [full text, Introduction]: "Herrero-Zazo
  et al. [8] developed a drug-drug interaction (DDI) corpus by annotating relations only if both drug names appear
  in the same single sentence."); kappa as in the table [abstract].
- HOW OUR CLAIM RELATES: reference point for pair-level IAA in a well-known corpus; the lower MedLine-abstract kappa
  (0.55-0.72) vs DrugBank texts suggests pair decisions are harder on research prose.

**hripcsak2005agreement** -- Hripcsak, Rothschild. "Agreement, the F-Measure, and Reliability in Information
Retrieval." Journal of the American Medical Informatics Association 12(3):296-298, 2005. DOI 10.1197/jamia.M1733.
Verified: Crossref (which lists only Hripcsak as author) and PMC1090460 / PubMed 15684123 (which list Hripcsak
and Rothschild; DECISION: bib uses the PMC author list). Abstract read (PMC holds only the abstract).
- WHAT IT SHOWS [abstract]: "Information retrieval studies that involve searching the Internet or marking phrases
  usually lack a well-defined number of negative cases. This prevents the use of traditional interrater
  reliability metrics like the kappa statistic ... It can be shown that the average F-measure among pairs of
  experts is numerically identical to the average positive specific agreement among experts and that kappa
  approaches these measures as the number of negative cases grows large."
- HOW OUR CLAIM RELATES: methodological note for reporting IAA on the two label types: the SENTENCE label and the
  PAIR label over enumerated co-occurrence candidates both have well-defined negatives, so Cohen's kappa is
  applicable to both; F1-agreement is the fallback for open-ended pair marking (as in BioRED/EU-ADR-style
  annotation). Cite only if the paper reports or discusses IAA.

### Q4-C. Dropping UNSURE / disagreement items is a choice with known consequences

**dumitrache2018crowdsourcing** -- Dumitrache, Aroyo, Welty. "Crowdsourcing Ground Truth for Medical Relation
Extraction." ACM Transactions on Interactive Intelligent Systems 8(2):1-20 (article pages as given by Crossref),
2018. DOI 10.1145/3152889. Verified: Crossref; full text read from the arXiv version (arXiv:1701.02185v2), which the
Semantic Scholar record links to the same DOI.
- Unit: PAIR in sentence ("a binary classifier ... that takes as input a set of sentences and two terms from the
  sentence, and returns a score reflecting the confidence of the model that a specific relation is expressed in
  the sentence between the terms" [full text, Sec. 3 "Experimental setup"]).
- WHAT IT SHOWS: [abstract] "disagreement between annotators provides a useful signal for phenomena such as
  ambiguity in the text ... by modeling ambiguity, labeled data gathered from crowd workers can (1) reach the level
  of quality of domain experts for this task while reducing the cost, and (2) provide better training data at scale
  than distant supervision." They too dropped undecidable items from the test set and state the cost: "The
  sentences where no decision could be reached were subsequently removed from the evaluation. There were 32 such
  sentences for cause ... and 15 for treat"; "Often these sentences contained a vague association between the two
  terms, but the relation was too broad to label it as a positive classification example. However, because a
  relation is nevertheless present, these sentences cannot be labeled as negative examples either. Eliminating
  these sentences is a disadvantage to a system like ours which was motivated specifically by the need to handle
  such cases" [full text, Sec. 3.4 "Evaluation data"].
- HOW OUR CLAIM RELATES: direct precedent for removing undecidable pair items from a relation test set, and for
  naming what it hides (vague associations). Cite when stating that UNSURE items are dropped.

**plank2022problem** -- Plank. "The 'Problem' of Human Label Variation: On Ground Truth in Data, Modeling and
Evaluation." Proceedings of EMNLP 2022, pp. 10671-10682, Abu Dhabi. ACL Anthology 2022.emnlp-main.731, DOI
10.18653/v1/2022.emnlp-main.731. Verified: Anthology .bib; full text read (Anthology PDF).
- WHAT IT SHOWS: [full text, Sec. 3 "Modeling and Human Label Variation", p. 10673] "Filtering methods are advocated
  by some with the idea to remove data instances with low agreement ... However, only using high-agreement
  instances can yield worse performance (Jamison and Gurevych, 2015) and it wastes data." [abstract] "this
  conventional practice assumes that there exists a ground truth, and neglects that there exists genuine human
  variation in labeling due to disagreement, subjectivity in annotation or multiple plausible answers."
- HOW OUR CLAIM RELATES: the standard citation for "dropping UNSURE is filtering, a known choice that removes the
  hard cases"; our evaluation therefore covers the decidable subset only (limitation).

**pavlick2019inherent** -- Pavlick, Kwiatkowski. "Inherent Disagreements in Human Textual Inferences." Transactions
of the Association for Computational Linguistics 7:677-694, 2019. ACL Anthology Q19-1043, DOI
10.1162/tacl_a_00293. Verified: Anthology .bib; full text read (Anthology PDF).
- WHAT IT SHOWS: [abstract] "We show that, very often, disagreements are not dismissible as annotation 'noise', but
  rather persist as we collect more ratings and as we vary the amount of context provided to raters." [full text,
  Sec. 2] "current practices-- in which we aggregate human judgments through majority vote/averaging and evaluate
  models on their ability to predict this aggregated label--are only appropriate if humans all tend to use the
  same process for resolving uncertainties in practice."
- HOW OUR CLAIM RELATES: supports treating UNSURE as signal (possible future work: soft labels / keep UNSURE as a
  third class); optional second citation alongside plank2022problem.

## Claims a reviewer would call KNOWN, and the citation that makes them known

1. Curation pipelines decompose into document triage, then pair extraction, with pair-anchored evidence
   sentences; pair extraction is much harder than triage: krallinger2008overview, leitner2010overview,
   islamajdogan2019overview (triage best F 69.06% vs pair RE best F 37.73%).
2. A document/sentence-level "describes an interaction" decision gives low precision on the pairs read off it:
   lim2016minter (abstract-level specificity 95%, AUC 0.97, but interaction-level precision 25%, "because it
   identifies abstracts containing interactions and reports all pairs of species names in such abstracts");
   thessen2014knowledge (taxa on a page are mentioned "for several reasons including comparison, taxonomic
   relationships ..."). Our 0.724 pair precision of a perfect sentence filter is a new measurement of a known
   effect, not a discovery.
3. Co-occurrence precision = interactions / candidate pairs and falls as entities per sentence grow
   (pyysalo2008comparative, I/EP; EP grows quadratically); the text unit changes co-occurrence precision
   (ding2001mining). Our ">2 entities" result should be framed as consistent with this.
4. Pair-in-sentence classification over co-occurring candidates, with "co-occur but not related" negatives, is
   standard (bravo2015extraction GAD/EU-ADR; abdelmageed2022biodivnere; herrerozazo2013ddi; elkhettari2023building
   for species with exactly two mentions).
5. PPI corpora disagree on what counts as an interaction and on how much sentence filtering they embed
   (pyysalo2008comparative).
6. Dropping undecidable items is a known filtering choice that removes hard/ambiguous cases and can bias
   evaluation (plank2022problem; dumitrache2018crowdsourcing did exactly this for a pair-in-sentence medical RE
   test set and named the cost; pavlick2019inherent: disagreements are signal).

## Claims we must NOT make, and the work that refutes or qualifies each

1. "Prior biotic-interaction mining works only at the sentence/document level" -- false: keck2025extracting and
   zou2026llm output species pairs; elkhettari2023building labels pairs (in two-species sentences);
   cuzick2023interspecies curates pathogen-host pairs (metagenotypes). Say instead: none compares a pair-conditioned verifier with a
   sentence-level classifier under the same teacher, data and encoder.
1b. "Reading the passage decision off pair-level predictions is new" -- BC III ACT Team 65 (krallinger2011protein)
   ranked articles with a PPI-detection pipeline score (alone and as a feature) and BC VI PM Team 433
   (islamajdogan2019overview) used PPI-triplet probabilities as triage features. What is new is the controlled
   comparison with the same teacher, data and encoder, and the finding that it is at least as good.
2. "We are the first to show that sentence-level filters give poor pair precision" -- lim2016minter showed it for
   microbial interactions (25% interaction-level precision); pyysalo2008comparative formalised it for PPI.
3. "Biotic interaction corpora have no pair labels" -- elkhettari2023building (SSI) is a species-pair corpus
   (restricted to exactly two species per sentence; no IAA reported).
4. "Human agreement is higher on pair labels than on sentence labels" (or the reverse) as a general fact -- no
   work found measures both on the same items; CTD (wiegers2009text) is the only two-level figure found and it
   compares document disposition with interaction extraction, on 112 articles.
5. "Dropping UNSURE is harmless" -- plank2022problem: "only using high-agreement instances can yield worse
   performance ... and it wastes data"; dumitrache2018crowdsourcing: eliminating such sentences "is a disadvantage
   to a system ... motivated specifically by the need to handle such cases".
6. Quoting keck2025extracting's "accuracy of 89.5%" as accuracy -- it is 229 TP / (229 TP + 27 FP), i.e. a
   precision (their own abstract says "excellent precision (89.5%)"); recall 70.0% on 327 gold interactions in
   500 paragraphs (bioRxiv v1). Check whether the version in our bib (keck2025extracting) reports the same.
7. "Recent LLM pipelines skip the sentence-level question" -- zou2026llm's first step is exactly a comment-level
   "contains interaction?" classifier (fine-tuned GPT-4o; 76% / 96% per-class accuracy on eBird).

## Could not verify / not read (do not cite these details without checking)

- EU-ADR (vanmulligen2012euadr), DDI (herrerozazo2013ddi), ADE (gurulingappa2012development): full texts paywalled
  (ScienceDirect blocks this server); only abstracts read. EU-ADR and ADE IAA NUMBERS NOT OBTAINED. DDI kappa is
  from the abstract only.
- leitner2010overview, scheepens2024large: abstracts only.
- (lim2016minter, keck2025extracting v1 and zou2026llm v1 quotes were cross-checked word for word against the PDF
  text; dimitrova2020semantic quotes against the Crossref-deposited abstract text.)
- ChemProt IAA: not pursued (the BioCreative VI ChemProt overview is workshop proceedings; no peer-reviewed
  journal version located in the time box).
- BioCreative VI Precision Medicine corpus paper (ref. 44 of islamajdogan2019overview, "The BioCreative VI
  Precision Medicine Track corpus: selection, annotation and curation ...", listed there as Database 2018): not
  found on Crossref; its BioNLP 2017 precursor (W17-2321) was read and reports no agreement figures. The overview
  says curators "met regularly to discuss and resolve their differences so that the final corpus is produced with
  complete consensus", without numbers.
- No corpus was found that reports IAA separately for a sentence-level "any interaction" label and a pair-level
  label on the same items (searched: BioCreative II/II.5/III/VI PPI, CTD, EU-ADR, GAD, DDI, ADE, BioInfer, the five
  PPI corpora in pyysalo2008comparative, BioRED, BiodivNERE, SSI, @MInter). Our dual-label set with per-label
  agreement would be unusual; state that as "we did not find", not as "none exists".
- Excluded as irrelevant after checking: Thessen et al. 2012 review; Le Guillarme and Thuiller 2023 (structured
  data integration, no text mining); Castro et al. 2024 (species distribution, not interactions); Alshawi et al.
  2023 WIESP virus-host dataset (NER only, no relation labels); Spillias et al. SPELL (database synthesis, not
  literature filtering); the eBird interaction case in Gallois 2025 (EcoEvoRxiv preprint).
