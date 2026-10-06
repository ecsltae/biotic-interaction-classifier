# verify2 -- adversarial re-verification, Set 2 (17 keys: biomedical / biodiversity journals)

Verifier 2, run 2026-10-05 from 02:26 CEST. Unattended; decisions noted inline.
Method: (a) every bib field compared with Crossref REST (`https://api.crossref.org/works/<DOI>`),
plus ACL Anthology `.bib` for Anthology papers and PubMed efetch for the abstract-only entries;
(b) claims located in full text: Europe PMC fullTextXML (flattened to text), ACL Anthology PDF and
arXiv PDF via `pdftotext -layout`. Scratch downloads were kept in /tmp/verify2_scratch (not in the repo).
Quotes below are copied from those texts; "[...]" marks my elisions.

Verdict scale: VERIFIED / FIX-BIB / CLAIM-WRONG / UNVERIFIABLE. "Nuance" = not an error, but wording
the paper should respect.

---

## krallinger2008overview -- VERIFIED (claim nuance on "article classification")

Bib: all fields match Crossref https://api.crossref.org/works/10.1186/gb-2008-9-s2-s4 (title; 4 authors in
order Krallinger, Leitner, Rodriguez-Penagos, Valencia; Genome Biology 9(S2), article-number S4; issued 2008-09-01;
Springer Science and Business Media LLC). No fixes.

Claim (full text, Europe PMC PMC2559988, "Protein-protein interaction task" section and Discussion list):
- Sub-task list, verbatim: "1. Interaction article subtask (IAS): classification and ranking of PubMed abstracts,
  based on whether they are relevant to protein interaction annotation or not. 2. Interaction pair subtask (IPS):
  extraction of binary protein-protein interaction pairs from full-text articles. [...] 4. Interaction sentences
  subtask (ISS): retrieval of the textual evidence passages that describe/summarize the interaction."
  (There are four subtasks; the third, IMS, is interaction detection method.)
- ISS per pair, verbatim: "For the ISS, participants had to provide, for each protein interaction pair, a ranked
  list of a maximum of five evidence passages describing their interaction. Each submitted evidence passage could
  comprise up to three consecutive sentences." -> "up to five evidence passages" per pair confirmed.
- Difficulty, verbatim, in the list introduced by "Participating groups encountered several obstacles that
  increased the difficulty of detecting normalized interaction pairs from full-text articles. Some of these are
  listed below.": "5. Difficulties in extracting the associations and in the handling of coordination (multiple
  interaction pairs) from a single sentence." -> quote exact, context fair.
- Nuance: IAS classifies PubMed titles+abstracts ("based on PubMed titles and abstracts only"), not full
  articles. If the paper writes "article classification", say "abstract (article-relevance) classification".

## islamajdogan2019overview -- VERIFIED, but the 37.73% needs a qualifier (source-internal inconsistency)

Bib: all fields match Crossref https://api.crossref.org/works/10.1093/database/bay147 (title; 27 authors, complete
and in the same order, Islamaj Dogan ... Lu; Database vol 2019; OUP). Crossref has no page/article-number; PMC XML
`<elocation-id>bay147</elocation-id>` confirms `pages = {bay147}`. No fixes.

Claims (full text PMC6348314):
- Two tasks, verbatim (abstract): "(i) document triage task, focused on identifying scientific literature containing
  experimentally verified protein-protein interactions (PPIs) affected by genetic mutations and (ii) relation
  extraction task, focused on extracting the affected interactions (protein pairs)."
- "same documents": NOT exactly. Corpus section: "each of these PubMed documents was first manually labeled for
  relevance for the triage task, and next, for the relation extraction task, the subset of PubMed documents that
  had been previously curated by IntAct/Mint for PPI relations was annotated with those interacting protein pairs".
  -> RE labels exist on a SUBSET of the triage documents. Write "on (a subset of) the same PubMed abstracts".
- 69.06%: verbatim in abstract ("for the triage task, the best F-score was 69.06%"), and Table 3 has a run at
  0.6906. Confirmed.
- 37.73%: abstract, verbatim: "For the relation extraction task, when taking homologous genes into account, the best
  F-score was 37.73%". Two problems if quoted bare:
  (1) it is the HomoloGene (relaxed) evaluation, not exact match; Table 4 (exact match) best F1 = 0.3483 (team 375 run 3);
  (2) the paper's own Table 5 (HomoloGene) maximum F1 is 0.3767 (team 375 run 3; next 0.3727, team 420 run 2), and the
      Results text says "The best F-score, precision and recall were 37.7%, 46.5% and 54.1%". 37.73 does not appear in
      the table. Safe wording: "best relation-extraction F 37.7% under HomoloGene-aware matching (34.8% exact match)".
- Team 433: their system description says triage features "were also extracted based on a model previously developed
  by our group for predicting PPI triplets. A triplet consists of two protein names and an interaction word that are
  all contained in the same sentence. [...] The predicted probabilities generated by this model were incorporated by
  taking normalized counts of the number of predicted triplet probabilities lying in equally spaced bins."
  -> Claim fair if worded "binned probabilities from a sentence-level PPI-triplet (protein pair + interaction word)
  model". Do not imply it helped much: team 433 triage F1 = 0.6713 (Table 3), below the best 0.6906, and its RE F1 was
  0.0900 exact / 0.1241 HomoloGene.

## krallinger2011protein -- VERIFIED

Bib: all fields match Crossref https://api.crossref.org/works/10.1186/1471-2105-12-S8-S3 (title; 34 authors complete
and in order, checked programmatically, incl. "Dogan, Rezarta Islamaj" as Crossref spells it; BMC Bioinformatics
12(S8), article-number S3; issued 2011). No fixes.

Claims (full text PMC3269938):
- Kappa, verbatim: "[...] as can also be seen by the very strong agreement on labels between the two databases
  (overlap 96%, Cohen's Kappa = 0.85). Agreement with the expert curators was lower, as expected, but within acceptable
  ranges (for MINT vs. expert, 92% overlap, Kappa = 0.69; for BioGrid vs. expert, 91%, Kappa = 0.69)."
  "the two databases" = MINT and BioGrid (preceding sentences discuss the MINT and BioGrid curators). Labels are
  ACT abstract-level relevance labels (true/false). Confirmed: 0.85 MINT vs BioGrid; 0.69 each vs expert.
  Nuance: these were the database curators' labels on the subset of ACT abstracts they classified (per-curator df in
  the timing test: BG 248, MINT 22-51), not the whole ACT corpus. Say "document-level agreement".
- Team 65, verbatim (team summary): "Three of the Team 65 runs apply Maximum Entropy optimization [...]. Features
  include lexical items, MeSH annotations, plus crucially a score delivered by their PPI detection pipeline. Two runs
  used only results of their protein-protein interaction detection pipeline". Confirmed. Note the claim "These results
  prove that an NLP-based pipeline for PPI extraction definitely provides a positive contribution" is the team's own
  summary text, and they ranked "3rd or 4th" per measure (best ACT run was team 73, F 63.16 vs T65 best 59.82, Table 4).

## wiegers2009text -- VERIFIED

Bib: all fields match Crossref https://api.crossref.org/works/10.1186/1471-2105-10-326 (title; 5 authors in order;
BMC Bioinformatics 10(1), article-number 326; 2009; Springer). No fixes.

Claims (full text PMC2768719):
- Ranking for curation, abstract: "These terms were used to re-rank documents for curation, resulting in increases in
  mean average precision (63% for the baseline vs. 73% for a rule-based re-ranking)".
- Step 1 (article disposition), verbatim: "Table 2 demonstrates that the curators agreed on the disposition of 86/112
  articles (77%) and had an average pair-wise agreement of 85%." Confirmed (3 curators; Table 2 "Consensus" row 0.77
  all three, pair-wise average 0.85).
- Step 2 (interactions), verbatim: "91% of the interactions extracted by CTD biocurators were judged by the lead curator
  to be correct (average precision = 0.91). Average recall was 0.71." Confirmed (Table 3; F1 0.77).
  Nuance: P/R are against a lead-curator-adjudicated pooled "gold standard", chemical-gene interactions only
  (Table 3 footnote a: "disease interactions were not considered"), and per curator over 53-68 articles.

## bravo2015extraction -- VERIFIED (with a labelling nuance)

Bib: all fields match Crossref https://api.crossref.org/works/10.1186/s12859-015-0472-9 (title; 5 authors in order,
Bravo, Pinero, Queralt-Rosinach, Rautschka, Furlong; BMC Bioinformatics 16(1), article-number 55 (PMC
`<elocation-id>55`); 2015). No fixes.

Claims (full text PMC4466840, Methods "GAD corpus" and "Evaluation of Kernel based RE"):
- Negatives, verbatim: "In order to create a dataset containing false associations (FALSE) between a gene and a
  disease, that is, a gene and a disease that co-occur in a sentence but are semantically not associated, we selected
  the sentences with co-occurrences between a disease and a gene found by the BioNER system that were not annotated by
  GAD curators as gene-disease associations." Confirmed.
- Pair-in-sentence instance, verbatim: "TRUE sentences contain real relationship between the entities analysed, in
  contrast with FALSE sentences where the two entities co-occur, but there is no semantic relationship between them."
  Evaluation "by sentence-level 10-fold cross validation". Confirmed.
- Nuance (do not misstate): GAD associations curated as *negative* ("no association found") are labelled TRUE
  ("annotated by GAD curators as positive or negative, were labelled as TRUE"); FALSE = uncurated co-occurrences, i.e.
  negatives are inferred from absence of curation (distant/silver), not human-judged. Entity spans come from their
  own BioNER, not curators.

## polajnar2011protein -- VERIFIED

Bib: all fields match Crossref https://api.crossref.org/works/10.1186/2041-1480-2-1 (title; Polajnar, Damoulas,
Girolami; J Biomed Semantics 2(1), article-number 1; 2011). No fixes.

Claims (full text PMC3116455, Methods, data description):
- Task definition, verbatim: "The other approach, and one which is employed in this paper, is to consider the sentences
  that contain interactions as positive examples, and the ones that do not, as negative." (contrasted with the pair
  approach: "Interacting pairs are then used as positive training examples, while any two proteins, that occur in the
  same sentence and do not interact, constitute the negative data.")
- Quote, verbatim: "The simpler classification leads to higher precision and recall, but only locates the sentence that
  describes the PPI and not the exact interacting pair." Confirmed; context fair (authors' own motivation, followed by
  "Thus while it is not fully automated, it might be more useful in a curation pipeline where the results need to be
  checked by humans").
- Counts, verbatim: "Using the provided sentence segmentation, the data set contains 614 positive and 1355 negative
  sentences. All sentences are included regardless of the number of annotated proteins contained within." Confirmed
  (AImed). Nuance: "higher precision and recall" is the authors' assertion, not a controlled comparison in the paper.

## thessen2014knowledge -- VERIFIED

Bib: all fields match Crossref https://api.crossref.org/works/10.1371/journal.pone.0089550 (title; Thessen, Anne E.;
Parr, Cynthia Sims; PLoS ONE 9(3), e89550; 2014; PLoS). No fixes.

Claims (full text PMC3940440):
- Section filter, verbatim (Methods): "retrieve the text objects under the "Associations", "Trophic Strategy" "General
  Ecology" and "Habitat" subchapters [...] We focused on these subchapters so we could extract information from text
  specifically describing ecological interactions".
- Kappa, verbatim (Results): "Agreement between annotators was 0.840 (Fleiss' Kappa; See Appendix S1,
  annotator_agreement.xlsx)." Confirmed.
- Precision, verbatim: "Our automated methods had an overall precision of 0.844, a recall of 0.930 and an F1 Score of
  0.885 (Table 3)"; "For the task of identifying ecological interactions in text, GNRD alone had a precision of 0.477,
  a recall of 0.957 and an F1 Score of 0.636 (Table 4)". Baseline = "GNRD API working directly on the EOL taxon page
  without the intervention of our workflow". Confirmed (Table 3 TOTAL 0.844/0.930/0.885; Table 4 TOTAL 0.477/0.957/0.636).
- Quote, verbatim (Discussion): "One cannot assume that all of the taxa mentioned on an EOL taxon page have an
  ecological relationship with the topic of that page." Confirmed.
- Nuance: evaluation is on 21 test species pages; a "relationship" is page-topic taxon <-> mentioned taxon, so these are
  page-level co-mention pairs, not sentence-level pairs.

## lee2020biobert -- VERIFIED

Bib: all fields match Crossref https://api.crossref.org/works/10.1093/bioinformatics/btz682 (title; 7 authors in order
Lee, Yoon, Kim S., Kim D., Kim S., So, Kang; Bioinformatics 36(4) 1234-1240; published-print 2020-02-15, online
2019-09-10; OUP). year=2020 is correct for the issue citation. No fixes.

Claim (full text PMC7703786, Sec. "3.3 Fine-tuning BioBERT", RE paragraph), verbatim: "We utilized the sentence
classifier of the original version of BERT, which uses a [CLS] token for the classification of relations. Sentence
classification is performed using a single output layer based on a [CLS] token representation from BERT. We
anonymized target named entities in a sentence using pre-defined tags such as @GENE$ or @DISEASE$." Section number 3.3
confirmed from the XML section titles.

## pyysalo2008comparative -- VERIFIED (one wording trap on Table 4)

Bib: all fields match Crossref https://api.crossref.org/works/10.1186/1471-2105-9-S3-S6 (title; 6 authors in order;
BMC Bioinformatics 9(S3), article-number S6; 2008). No fixes.

Claims (full text PMC2349296; tables read from the XML):
- Table 2 "PPI extraction performance", co-occurrence P: AIMed 0.17, BioInfer 0.13, HPRD50 0.38, IEPA 0.41, LLL 0.50
  (R 0.95/0.99/1.0/1.0/1.0). Confirmed.
- Table 3 "Corpus statistics", "Fraction of sentences with ... No interactions": 69%, 48%, 38%, 37%, 0%. Confirmed.
- I/EP, verbatim: "the average number of interactions divided by the average number of entity pairs per sentence
  (below abbreviated I/EP) equals the precision of the co-occurrence method evaluated above." Confirmed.
- Table 4 "PPI extraction performance on filtered corpora", co-occurrence P: 0.53, 0.53, 0.64, 0.88, 0.50. Confirmed.
  TRAP: the filter is NOT "keep only positive sentences". Footnote: "corpora with only entities that participate in an
  interaction preserved"; text: "removing from the annotation proteins for which there is no interaction, thus
  reducing EP". So Table 4 = co-occurrence precision when every non-interacting protein is deleted (this also removes
  non-interacting proteins inside positive sentences). It must not be described as "co-occurrence precision within
  interaction-bearing sentences"; that quantity is not reported (it is >= Table 2 and <= Table 4).
- Quote, verbatim (abstract): "However, there is no general consensus regarding PPI annotation and consequently
  resources are largely incompatible and methods are difficult to evaluate." Confirmed.
- Nuance: all numbers are on the authors' unified transformed versions of the corpora ("All results and discussion
  below concern these transformed versions of the corpora").

## tikk2013detailed -- VERIFIED

Bib: all fields match Crossref https://api.crossref.org/works/10.1186/1471-2105-14-12 (title; Tikk, Solt, Thomas, Leser;
BMC Bioinformatics 14(1), article-number 12; 2013). No fixes. (PMCID is PMC3680070.)

Claim (full text PMC3680070, "Results and discussion > Relation between sentence length, entity distance and pair
difficulty"), verbatim: "We investigated the class distribution of pairs depending on the number of proteins in the
sentence (see Figure 7). We can see that the more protein mentions a sentence exhibits, the lower the ratio of positive
pairs." Confirmed. Figure 7 itself (per-count values) was not read (image); do not quote per-count numbers.
Context: the five PPI benchmark corpora (AIMed, BioInfer, HPRD50, IEPA, LLL), candidate pairs = all protein pairs in a
sentence.

## herrerozazo2013ddi -- VERIFIED (abstract only)

Bib: all fields match Crossref https://api.crossref.org/works/10.1016/j.jbi.2013.07.011 (title; Herrero-Zazo,
Segura-Bedmar, Martinez, Declerck; J Biomed Inform 46(5) 914-920; 2013-10; Elsevier BV) and PubMed PMID 23906817
("J Biomed Inform. 2013 Oct;46(5):914-20"). No fixes.

Claim: ONLY THE ABSTRACT WAS CHECKED (PubMed efetch rettype=abstract,
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=23906817&rettype=abstract&retmode=text).
Verbatim: "The agreement was almost perfect (Kappa up to 0.96 and generally over 0.80), except for the DDIs in the
MedLine database (0.55-0.72)." Confirmed word for word. Context: inter-annotator agreement "between two annotators";
the 0.55-0.72 range concerns the DDI (relation) annotation in the 233 MedLine abstracts. Which kappa refers to which
entity/DDI type is in the full text (not read).

## gurulingappa2012development -- VERIFIED (abstract only)

Bib: all fields match Crossref https://api.crossref.org/works/10.1016/j.jbi.2012.04.008 (title; 6 authors in order
Gurulingappa, Rajput, Roberts, Fluck, Hofmann-Apitius, Toldo; J Biomed Inform 45(5) 885-892; 2012-10; Elsevier BV) and
PubMed PMID 22554702. No fixes.

Claim: ONLY THE ABSTRACT WAS CHECKED (PubMed efetch, id=22554702). Verbatim: "The documents are systematically double
annotated in various rounds to ensure consistent annotations. The annotated documents are finally harmonized to
generate representative consensus annotations. [...] the corpus was employed to train and validate models for the
classification of informative against the non-informative sentences. A Maximum Entropy classifier trained with simple
features and evaluated by 10-fold cross-validation resulted in the F1 score of 0.70". Confirmed. Nuance: say
"MaxEnt, 10-fold CV"; the abstract does not say which class the F1 is computed on.

## elkhettari2023building -- VERIFIED

Bib: matches ACL Anthology https://aclanthology.org/2023.bionlp-1.21.bib (title; El Khettari, Quiniou, Chaffron;
booktitle "Proceedings of the 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks";
pages 248--254; 2023; Toronto, Canada; ACL; doi 10.18653/v1/2023.bionlp-1.21) and Crossref. Editors/month omitted
(optional). No fixes.

Claim (Anthology PDF https://aclanthology.org/2023.bionlp-1.21.pdf, Sec. 3.2 "Annotating Species-Species Relation"),
verbatim: "We focused on sentences where exactly two species are mentioned to maximize the probability that a binary
relation is indeed expressed." Confirmed. Following context: "Consequently, no sentences with the previous criterion
were found on S800. [...] Therefore, we collected 442 sentences from the LINNAEUS corpus and 557 sentences [...]".
Related fact (Limitations): "only one annotator has been invested in" the annotation -> no IAA reported.

## peng2019transfer -- VERIFIED

Bib: matches ACL Anthology https://aclanthology.org/W19-5006.bib (title, 3 authors, booktitle "Proceedings of the 18th
BioNLP Workshop and Shared Task", pages 58--65, 2019, Florence, Italy, ACL, doi 10.18653/v1/W19-5006). Cosmetic only:
the Anthology protects "{ELM}o", the entry "{ELMo}"; both render "ELMo". No fixes.

Claim (Anthology PDF https://aclanthology.org/W19-5006.pdf, Sec. 4.1.2 "Fine-tuning with BERT", printed p. 61, right
column), verbatim: "We treated the relation extraction task as a sentence classification by replacing two named entity
mentions of interest in the sentence with predefined tags (e.g., @GENE$, @DRUG$) (Lee et al., 2019)." Confirmed
(page located by the printed page footers in the pdftotext output). Note the source credits Lee et al. 2019 (BioBERT).

## dumitrache2018crowdsourcing -- VERIFIED against arXiv v2 (published TiiS text not accessible)

Bib: all fields match Crossref https://api.crossref.org/works/10.1145/3152889 (title; Dumitrache, Aroyo, Welty; ACM
TiiS 8(2), pages 1-20; issued 2018-06-30; ACM). No fixes. (ACM also assigns an article number, not given by Crossref;
its absence is not an error.)

Claim (arXiv:1701.02185v2, 3 Oct 2017, https://arxiv.org/pdf/1701.02185v2, Sec. "3.4. Evaluation data", p. A:8-A:9),
verbatim: "To ensure a fair comparison, our team adjudicated each of them to decide whether or not the relation is
present in the sentence. The sentences where no decision could be reached were subsequently removed from the
evaluation. There were 32 such sentences for cause (18 with negative expert labels, and 14 with positive), and 15 for
treat (all for positive expert labels)." and "Eliminating these sentences is a disadvantage to a system like ours which
was motivated specifically by the need to handle such cases, however the scientific community still only recognizes
discrete measures such as precision and recall, and we felt it only fair to eliminate the cases where we could not
agree on the correct way to map ambiguity into a discrete score." Confirmed; context fair (removed sentences were
those where crowd and expert disagreed and adjudication failed).
Residual risk: the arXiv v2 header says "18 pages" while the TiiS record is pp. 1-20, so the published text may differ
in wording/section numbering; I could not open the ACM full text. If the paper quotes it, cite "Sec. 3.4" only if
the published version is checked, or quote without a section number.

## lim2016minter -- VERIFIED, but keep "partly" (and do not mix abstract- and species-level specificity)

Bib: all fields match Crossref https://api.crossref.org/works/10.1093/bioinformatics/btw357 (title "@MInter: automated
text-mining of microbial interactions"; Lim, Kun Ming Kenneth; Li, Chenhao; Chng, Kern Rei; Nagarajan, Niranjan;
Bioinformatics 32(19) 2981-2987; 2016; OUP) and PubMed PMID 27312413. No fixes.

Access: not in PMC; OUP PDF returns 403/Cloudflare to curl. Abstract checked verbatim via PubMed efetch. Full text read
via WebFetch of https://academic.oup.com/bioinformatics/article/32/19/2981/2196369 (rendered by a summarising model,
so wording is near-verbatim, not byte-verified; two separate fetches gave the same sentences).
- Abstract, verbatim (PubMed): "@MInter was able to detect abstracts pertaining to one or more microbial interactions
  with high specificity (specificity = 95%, AUC = 0.97). Despite challenges in identifying specific microbial
  interactions in an abstract (interaction level recall = 95%, precision = 25%), @MInter was shown to reduce annotator
  workload 13-fold compared to alternate approaches." Confirmed.
- Sec. 3.2 "@MInter improves performance at the species level": "@MInter's species-level precision was found to be low
  despite being the best across all methods. This is partly because it identifies abstracts containing interactions
  and reports all pairs of species names in such abstracts." -> The claim file quotes it as "because it identifies
  ...". The source says "partly because". If the paper writes "because", that overstates the source; write "partly
  because" or start the quote at "it identifies".
- Table 1 "Performance summary for different methods at the species level": All Positives (AP) recall 100 /
  specificity 0 / precision 7; @MInter SVM 95 / 78 / 25; pattern scanner 24 / 92 / 15; naive Fisher 51 / 63 / 10;
  Fisher w/ inhibition 50 / 77 / 15. "'All Positives' precision 7%" confirmed. AP definition (Table 1 note, per the WebFetch
  rendering; one fetch placed it under a methods subsection, so do not cite a section number): "The 'All Positives'
  approach reports all considered species pairs as being interacting ones and is provided here as a control [...]".
- Nuance: the 95% specificity / AUC 0.97 is ABSTRACT level (Sec. 3.1); at the species-pair level the SVM's
  specificity is 78% (Table 1). Do not pair "specificity 95%" with "precision 25%" as if same level. "Species level"
  = corpus-level species pairs, not per-abstract pairs: Sec. 2.1.3, "First, species level annotation considered
  whether any of the abstracts containing a species pair described a true interaction." (735 pairs, 62 interacting).

## thieu2012literature -- VERIFIED (66-73% is Task-1 abstract-level accuracy)

Bib: all fields match Crossref https://api.crossref.org/works/10.1093/bioinformatics/bts042 (title; Thieu, Joshi,
Warren, Korkin; Bioinformatics 28(6) 867-875; 2012; OUP) and PubMed PMID 22285561. No fixes.

Access: not in PMC; abstract checked verbatim via PubMed efetch; full text via WebFetch of
https://academic.oup.com/bioinformatics/article/28/6/867/311962 (model-rendered, near-verbatim).
- Sec. 2.1 "Feature-based approach": "Once trained, the same classifier is used in our method twice: first, to classify
  whether an abstract is HPI-relevant, and secondly, if it is relevant, to determine which sentences of the abstract
  are most likely to contain the HPI-relevant data." Confirmed (abstract -> sentence cascade).
- Pair afterwards: tasks as rendered: Task 1 "given an expanded abstract [...] determine whether it is HPI-relevant";
  Task 2 "determine specific sentences that include this information"; Task 3 "determine specific pairs of host and
  pathogen proteins/genes participating in the interactions". Confirmed (the pair is protein/gene + organism, extracted
  from positive sentences).
- Abstract, verbatim (PubMed): "The most accurate, feature-based, approach achieved 66-73% accuracy, depending on the
  test protocol." Nuance: this is the CLASSIFICATION task (abstract HPI-relevance, Task 1; abstract: "higher accuracy
  and recall in the classification task"); it is not a sentence- or pair-extraction accuracy. Per-protocol numbers from
  the WebFetch rendering of Table 3 were inconsistent with the 73% upper bound, so do not quote per-protocol values.

---

## Quote locations (section paths from the PMC XML)
- krallinger2008overview: ISS five passages -> Results > Interaction sentences subtask; coordination -> Discussion and conclusion.
- islamajdogan2019overview: RE subset -> Methods and data > Precision Medicine Track corpus development and annotation;
  Team 433 -> Results > Individual system descriptions.
- krallinger2011protein: Kappa -> Results > ACT > ACT inter-annotator and manual classification time analysis;
  Team 65 -> Results > Individual system descriptions > Team 65.
- wiegers2009text: Results > Baseline analysis of CTD manual curation > Inter-Biocurator Agreement.
- bravo2015extraction: Methods > GAD.
- polajnar2011protein: Methods > Experimental Setup.
- thessen2014knowledge: kappa and precisions -> Results > Association network; "One cannot assume" -> Discussion > Association network.

---

## Summary (Set 2, 17 keys)

No FIX-BIB: every bib field (title, all authors in order -- compared programmatically with Crossref -- venue, volume,
number, pages/article number, year, publisher, DOI) matches Crossref / ACL Anthology. No CLAIM-WRONG. Five entries
need wording care (marked "wording" below); two are abstract-only; three rest on non-PMC full text.

| key | verdict | full text checked | wording the paper must respect |
|---|---|---|---|
| krallinger2008overview | VERIFIED | PMC2559988 | IAS classifies PubMed titles+abstracts, not full articles |
| islamajdogan2019overview | VERIFIED (wording) | PMC6348314 | 37.73% is HomoloGene-aware matching (exact match best 34.83%); Table 5 max is 0.3767, text says 37.7% -> write "37.7%"; RE labels on a SUBSET of the triage documents; Team 433 used binned PPI-triplet probabilities (its F1 0.6713, not best) |
| krallinger2011protein | VERIFIED | PMC3269938 | kappa 0.85 MINT vs BioGrid, 0.69 each vs expert, on abstract labels |
| wiegers2009text | VERIFIED | PMC2768719 | P 0.91 / R 0.71 vs lead-curator-adjudicated gold, chemical-gene only |
| bravo2015extraction | VERIFIED | PMC4466840 | FALSE = uncurated co-occurrences (silver); GAD "negative" associations are labelled TRUE |
| polajnar2011protein | VERIFIED | PMC3116455 | -- |
| lim2016minter | VERIFIED (wording) | abstract (PubMed) + OUP HTML via WebFetch | source says "partly because"; 95% specificity is abstract-level, species-level SVM specificity 78% |
| thessen2014knowledge | VERIFIED | PMC3940440 | page-level co-mention pairs, 21 test pages |
| elkhettari2023building | VERIFIED | Anthology PDF | -- |
| thieu2012literature | VERIFIED (wording) | abstract (PubMed) + OUP HTML via WebFetch | 66-73% accuracy = abstract-level HPI classification (Task 1), not pair extraction |
| lee2020biobert | VERIFIED | PMC7703786 | -- |
| peng2019transfer | VERIFIED | Anthology PDF (p. 61, Sec. 4.1.2) | -- |
| pyysalo2008comparative | VERIFIED (wording) | PMC2349296 | Table 4 = non-interacting proteins removed, NOT "positive sentences only" |
| tikk2013detailed | VERIFIED | PMC3680070 | Fig. 7 values not read; quote the sentence only |
| herrerozazo2013ddi | VERIFIED | abstract only | -- |
| gurulingappa2012development | VERIFIED | abstract only | MaxEnt, 10-fold CV |
| dumitrache2018crowdsourcing | VERIFIED (wording) | arXiv v2 only (ACM text 403) | published TiiS wording/section unverified; cite "Sec. 3.4" only for the arXiv version |

Finished 02:35 CEST.
