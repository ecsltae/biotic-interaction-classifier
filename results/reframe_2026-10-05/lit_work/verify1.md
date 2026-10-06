# Verifier 1 report: Set 1 (general NLP / ML), 22 keys

Adversarial re-verification, started 2026-10-05 02:25 CEST. Bib entries from
paperA/reframe_extra.bib compared field by field against the primary record (ACL Anthology .bib,
Crossref, NeurIPS/PMLR pages, arXiv abs page). Claims from CLAIMS_TO_VERIFY.md "Set 1" checked
against full text (PDF -> pdftotext -layout). Scratch downloads in /tmp/v1s (not in the repo).

Verdicts: VERIFIED / FIX-BIB / CLAIM-WRONG / UNVERIFIABLE.

## 1. dietterich1997solving -- VERIFIED

- Bib: matches Crossref (https://api.crossref.org/works/10.1016/S0004-3702(96)00034-3 and doi.org BibTeX
  negotiation) in every field: title, 3 authors and order (Dietterich, Thomas G.; Lathrop, Richard H.;
  Lozano-Pérez, Tomás), Artificial Intelligence 89(1-2):31-71, 1997, Elsevier BV, DOI. No fix.
- Claim: full text (unofficial mirror, same as finder's; ScienceDirect blocks curl):
  https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1997%20-%20DietterichLathropLozano-Perez%20-%20AI.pdf
  Sec. 1 Introduction, p. 34 (text precedes the p. 35 running head): "An hypothesis ĝ is consistent with a set of
  training examples if it classifies every feature vector of every negative example as negative and if it
  classifies at least one feature vector of every positive example as positive." Also p. 34: "object m_i is
  predicted to be a positive example if and only if there exists at least one feature vector for m_i ... that is
  predicted to be positive according to g." Abstract wording ("only one of those feature vectors may be responsible
  for the observed classification") confirmed. Paraphrase "standard MI assumption" is fair. Caveat: the
  mirror is not the publisher copy, but the running heads ("Artificial Intelligence 89 (1997) 31-71") show it is
  a scan of the published version.

## 2. andrews2002support -- VERIFIED

- Bib: matches the NeurIPS proceedings BibTeX
  (https://proceedings.neurips.cc/paper_files/paper/2002/file/3e6260b81898beacda3d16db379ed329-Bibtex.bib):
  title, authors (Andrews, Stuart; Tsochantaridis, Ioannis; Hofmann, Thomas), booktitle "Advances in Neural
  Information Processing Systems", volume 15, year 2002, MIT Press. The source has empty pages; omitting them is
  not an error (editors S. Becker, S. Thrun, K. Obermayer are optional).
- Claim: NeurIPS PDF (…-Paper.pdf), Sec. 1 Introduction, first paragraph, verbatim: "In the important case of
  binary classification, this implies that a bag is "positive" if at least one of its member patterns is a
  positive example. MIL differs from the general set-learning problem in that the set-level classifier is by
  design induced by a pattern-level classifier." Context: the next sentence makes the key challenge "not knowing
  which of the patterns in a positive bag are the actual positive examples", i.e. pattern labels are hidden;
  the paper should not imply Andrews et al. train on pattern labels.

## 3. hoffmann2011knowledge -- VERIFIED (locator fix only)

- Bib: identical to https://aclanthology.org/P11-1055.bib (title, 5 authors in order, booktitle, pages 541--550,
  2011, Portland, Oregon, USA, ACL). PDF author line agrees. No fix.
- Claim, quote 1: https://aclanthology.org/P11-1055.pdf, p. 543: "The factors Φjoin are deterministic OR
  operators ... which are included to ensure that the ground fact r(e) is predicted at the aggregate level for
  the assignment Y^r = y^r only if at least one of the sentence level assignments Z_i = z_i signals a mention of
  r(e)." (p. 543, both columns). LOCATOR: this is in Sec. 3 "Modeling Overlapping Relations", not Sec. 4 (Sec. 4 is "Learning", p. 544).
  Page 543 is correct. The model is named MULTIR in the paper (Sec. 7 onwards); fair.
- Claim, quote 2: Sec. 1, p. 541, right column, verbatim: "they cast weak supervision as a form of multi-instance
  learning, assuming only that at least one of the sentences containing e1 and e2 are expressing r(e1, e2)"
  ("they" = Riedel et al. 2010). Fair. (A second wording of the same assumption is in Sec. 8.2, p. 549.)

## 4. zeng2015distant -- VERIFIED

- Bib: identical to https://aclanthology.org/D15-1203.bib (title, 4 authors, EMNLP 2015 booktitle, pages
  1753--1762, Lisbon, Portugal, ACL, DOI 10.18653/v1/D15-1203). No fix.
- Claim: https://aclanthology.org/D15-1203.pdf, Sec. 1, p. 1754: "The labels of the bags are known; however, the
  labels of the instances in the bags are unknown. We design an objective function at the bag level."
  Sec. 3.5 (Multi-instance Learning), p. 1758 (the claim gives no page): "The objective of multi-instance learning is to discriminate bags
  rather than instances." followed by the bag-level cross-entropy J(θ) = Σ log p(y_i | m_i^j; θ) with
  "j* = arg max_j p(y_i | m_i^j; θ)" (Eq. 9), i.e. only the instance with the highest probability for the bag's
  label is used. "Trains on the highest-scoring instance per bag" is a fair paraphrase (highest-scoring for the
  bag's gold relation, not highest-scoring overall).

## 5. jia2019document -- VERIFIED

- Bib: identical to https://aclanthology.org/N19-1370.bib (title, Jia, Robin; Wong, Cliff; Poon, Hoifung;
  NAACL 2019 Vol. 1 booktitle, pages 3693--3704, Minneapolis, Minnesota, ACL, DOI). No fix.
- Claim: https://aclanthology.org/N19-1370.pdf, Sec. 4.3 (Main Results), p. 3699, verbatim: "We use logsumexp as
  the aggregation operator to combine mention-level representations into an entity-level one. If we replace it
  with max pooling, the performance drops substantially across the board, as shown in Table 5. For example,
  MULTISCALE lost 3.8 absolute points in AUC." The ellipsis in the claim stands for "across the board, as shown
  in Table 5. For example,", which is fair. Table 5 caption: "Results on CKB after replacing logsumexp with max
  (with noisy-or and gene-mutation filter)". Sec. 3.2 (p. 3695) motivates logsumexp as "the smooth version of
  max". Nuance for the paper: the operator pools mention-tuple REPRESENTATIONS (vectors) into an entity-level
  representation, not final scores; and the dataset is CKB (document-level drug-gene-mutation), not sentences.
  Do not describe it as score-level max vs logsumexp over pair probabilities.

## 6. ilse2018attention -- VERIFIED

- Bib: matches the PMLR BibTeX on https://proceedings.mlr.press/v80/ilse18a.html (title; Ilse, Maximilian and
  Tomczak, Jakub and Welling, Max; Proceedings of the 35th International Conference on Machine Learning; volume
  80; pages 2127--2136; 2018; PMLR). The PDF prints "Jakub M. Tomczak"; PMLR metadata omits the initial, so
  "Tomczak, Jakub" is acceptable. Optional: series = {Proceedings of Machine Learning Research}. No fix.
- Claim: https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf, Sec. 2.1 (Multiple instance learning (MIL)),
  p. 2129, verbatim: "It is advocated in (Wang et al., 2016) that the latter approach is preferable in terms of
  the bag level classification performance. Since the individual labels are unknown, there is a threat that the
  instance-level classifier might be trained insufficiently and it introduces additional error to the final
  prediction." "The latter approach" = (ii) the embedding-level approach; (i) is the instance-level approach.
  Fair. Context to keep: the same paragraph continues that the instance-level approach "provides a score that can
  be used to find key instances", and the argument rests on instance labels being UNKNOWN.

## 7. laban2022summac -- CLAIM-WRONG (minor; the number pair mixes two aggregators)

- Bib: identical to https://aclanthology.org/2022.tacl-1.10.bib (title with braces, 4 authors, TACL 10:163--177,
  2022, Cambridge, MA, MIT Press, DOI 10.1162/tacl_a_00453). PDF author line agrees. No fix.
- Quote: https://aclanthology.org/2022.tacl-1.10.pdf, Sec. 5.3.3 (Choice of Granularity), p. 172, verbatim:
  "The MNLI-only trained model achieves lowest performance when used with full text granularity on the document
  level, and performance steadily increases from 56.4% to 73.5% as granularity is made finer both on the document
  and summary side." The words are correct and it is the same MNLI-trained NLI model.
- What is wrong in the paraphrase: the claim attaches the 56.4 -> 73.5 range to "aggregated (max/mean)". In
  Table 5 (p. 172), 56.4 is SummaC-ZS (the zero-shot max-then-mean aggregator) at (full, full), but 73.5 is the
  SummaC-Conv column at (two sent., sentence); SummaC-Conv replaces max/mean by a TRAINED 1-D convolution over
  histograms of the pair scores. With max/mean only (ZS column), the MNLI model goes from 56.4 (full, full) to
  71.2 (two sent., sentence) / 70.3 (sentence, sentence). Metric: balanced accuracy on the SummaC benchmark test
  set. Also "steadily" is the authors' word; Table 5 is not monotone (e.g. paragraph/sentence ZS 65.2 > two
  sent./full ZS 64.0; sentence/full 58.7).
- Also inaccurate in agentA.md (lines 267, 324, 515, 538-539): "same model, different inference granularity" /
  "compares input granularities of one fixed NLI model at inference time". True for the ZS column only; the 73.5
  endpoint uses SummaC-Conv, whose convolutional layer is trained (Sec. 3.3: "a learned convolutional layer";
  "we train the SUMMACCONV model end-to-end with the synthetic training data in FactCC", also Sec. 3.3), so that endpoint is not inference-time aggregation of a fixed model.
- Fix: either quote the authors' sentence without the "(max/mean)" gloss, or write "with the zero-shot max/mean
  aggregation, the same MNLI model rises from 56.4% to 71.2% balanced accuracy (Table 5); 73.5% with the learned
  SummaC-Conv aggregator". The abstract supports the general statement ("segmenting documents into sentence units
  and aggregating scores between pairs of sentences"); the max/mean operators are in Sec. 3.2 SummaC-ZS (Appendix B,
  Table A1: max for operator 1, mean for operator 2).

## 8. verga2018simultaneously -- VERIFIED

- Bib: identical to https://aclanthology.org/N18-1080.bib (title, Verga, Patrick; Strubell, Emma; McCallum,
  Andrew; NAACL 2018 Vol. 1 (Long Papers); pages 872--884; New Orleans, Louisiana; ACL; DOI). No fix.
- Claim: https://aclanthology.org/N18-1080.pdf, Sec. 2.4 (Entity Level Prediction), p. 875, verbatim: "we use the
  LogSumExp function to aggregate the relation scores from A across all pairs of mentions of p_head and p_tail"
  and "The LogSumExp scoring function is a smooth approximation to the max function and has the benefits of
  aggregating information from multiple predictions and propagating dense gradients as opposed to the sparse
  gradient updates of the max". Aggregation is over mention-pair relation SCORES (logits A_ij) to entity-pair
  scores, over a full abstract. Fair. Note the locator: the quote is in Sec. 2.4, not Sec. 1/2 generally.

## 9. min2023factscore -- FIX-BIB (one author's given name truncated; low severity) ; claim VERIFIED

- Bib vs https://aclanthology.org/2023.emnlp-main.741.bib: identical, and the Crossref DOI record also has
  given "Pang", family "Koh". But the paper's title block (https://aclanthology.org/2023.emnlp-main.741.pdf, p. 1)
  prints "Pang Wei Koh", and arXiv metadata (https://arxiv.org/abs/2305.14251) has "Koh, Pang Wei": the given name
  is "Pang Wei", so "Koh, Pang" is a truncation inherited from the Anthology record. FIX: author "Koh, Pang Wei".
  All other fields (title, 9 authors and order, EMNLP 2023, pages 12076--12100, Singapore, ACL, DOI) correct.
- Claim: abstract, verbatim: "generations often contain a mixture of supported and unsupported pieces of
  information, making binary judgments of quality inadequate" and FActScore "breaks a generation into a series of
  atomic facts and computes the percentage of atomic facts supported by a reliable knowledge source." Fair.

## 10. chowdhury2013exploiting -- VERIFIED (with a scope caveat)

- Bib: identical to https://aclanthology.org/N13-1093.bib (title, Chowdhury, Md. Faisal Mahbub; Lavelli, Alberto;
  NAACL-HLT 2013 booktitle; pages 765--771; Atlanta, Georgia; ACL; the Anthology record has no DOI). No fix.
- Claim: https://aclanthology.org/N13-1093.pdf, Sec. 2.1 "Stage 1: Exploiting scope of negation to filter out
  sentences", p. 766, verbatim: "In the Stage 1, any sentence that contains at least one DDI is considered by the
  classifier as a positive (training/test) instance. Other sentences are considered as negative instances."
  Sec. 3 (Results and Discussion), p. 768, verbatim: "Unlike Stage 1, in Stage 2 where we train the hybrid kernel
  based RE classifier and use it for RE (i.e. DDI extraction) from the test data, sentences are not the RE
  training/test instances. Instead, a RE instance corresponds to a candidate mention pair." Both parts confirmed.
- Caveat (do not overstate): Stage 1 is not a general "does this sentence contain an interaction" detector. It
  is trained only on sentences that pass four rules (>= 2 drug mentions; a "no"/"n't"/"not" cue; a drug after the
  cue; none of "not recommended"/"should not be"/"must not be"), uses negation-scope features only, and its job is
  to DISCARD likely-negative sentences ("to reduce the number of candidate mention pairs by discarding
  sentences"). On test it flagged 121 sentences (7.86%), 5 wrongly. Gain: +1.0 F (KHybrid 66.4 -> 67.4, Table 2).
  Describe it as a negation-based sentence filter before a pair classifier.

## 11. xie2021revisiting -- VERIFIED

- Bib: identical to https://aclanthology.org/2021.acl-long.277.bib (title, 6 authors in PDF order, ACL-IJCNLP 2021
  Vol. 1 booktitle, pages 3572--3581, Online, ACL, DOI). No fix.
- Claim: abstract, verbatim: "we propose a pipeline approach, dubbed RERE, that first performs sentence
  classification with relational labels and then extracts the subjects/objects." Sec. 2.2 (Addressing the
  Overwhelming Negative Labels), p. 3574, verbatim: "Suppose a sentence contains m entities, the classifier has to
  decide relation from O(m^2) entity pairs, while in reality, relations are often sparse, i.e., O(m). In other
  words, most entity pairs in P1 do not form valid relation, thus resulting in a low class prior." Fair.
- Nuance: RERE's sentence stage is MULTI-LABEL relation-type detection (one sigmoid per relation type, Sec. 3.1),
  not a single binary "any relation" decision; and the data are distantly supervised NYT/SKE. Table 2 supports
  the class-prior argument numerically (e.g. NYT10-HRL: pair-level pi2 = 0.01421 for P1 vs sentence-level
  pi1 = 0.0390 for P3).

## 12. liu2019event -- VERIFIED (author-order discrepancy noted; no change required)

- Bib vs https://aclanthology.org/N19-1080.bib: identical. Author order in the bib (Liu, Shulin; Li, Yang; Zhang,
  Feng; Yang, Tao; Zhou, Xinpeng) is also the order in the Crossref DOI record
  (https://api.crossref.org/works/10.18653/v1/N19-1080), the Anthology landing-page citation_author tags and
  Semantic Scholar. BUT the paper's own title block (https://aclanthology.org/N19-1080.pdf, p. 735) prints
  "Shulin Liu, Yang Li, Xinpeng Zhou, Tao Yang, Feng Zhang" (Zhou and Zhang swapped). Decision: keep the bib as is
  (it matches every registered metadata record, which is what reference checkers compare against); optionally
  switch to the printed order. Not a desk-reject risk either way. Other fields (title, NAACL 2019 Vol. 1, pages
  735--744, Minneapolis, Minnesota, ACL, DOI) correct.
- Claim: abstract, verbatim: "Remarkably, the proposed approach even achieves competitive performances compared
  with state-of-the-arts that used annotated triggers." Sec. 1, p. 736: "we transform multi-label classification
  to multiple binary classification problems. Specifically, a given sentence s attached each pre-defined event
  type t forms an instance, which is expected to be labeled with 0 or 1 according to whether s contains an event
  of type t." and (Sec. 1, p. 735) "the only annotated information of each sentence is the types of events occurred
  in it."
  Fair. Dataset: ACE 2005.

## 13. zhong2021frustratingly -- VERIFIED

- Bib: identical to https://aclanthology.org/2021.naacl-main.5.bib (title; Zhong, Zexuan; Chen, Danqi; NAACL-HLT
  2021 booktitle; pages 50--61; Online; ACL; DOI). No fix.
- Claim: https://aclanthology.org/2021.naacl-main.5.pdf.
  (i) Sec. 3.2 (Our Approach), p. 53, verbatim: "Our relation model instead processes each pair of spans
  independently and inserts typed markers at the input layer to highlight the subject and object and their types."
  (ii) "we need to run our relation model once for every pair of entities": Sec. 1 (p. 50) and again Sec. 3.3
  (Efficient Batch Computations, p. 54), verbatim.
  (iii) Sec. 5.1 (Importance of Typed Text Markers), p. 57, verbatim: "Compared to TEXT, TYPED MARKERS improved the
  F1 scores dramatically by +5.0% and +7.4% absolute when gold entities are given." Table 4 checks out:
  ACE05 gold 67.6 -> 72.6 (+5.0), SciERC gold 61.7 -> 69.1 (+7.4). Setting to state if quoted: relation F1
  (boundaries) on the DEVELOPMENT sets of ACE05 (BERT-base, single-sentence context) and SciERC (SciBERT,
  cross-sentence context); with predicted entities the gain is smaller (ACE05 61.6 -> 64.2, SciERC 45.3 -> 49.7).

## 14. li2019entity -- VERIFIED

- Bib: identical to https://aclanthology.org/P19-1129.bib (title, 8 authors in PDF order, ACL 2019, pages
  1340--1350, Florence, Italy, ACL, DOI). Minor: the PDF title capitalises "Multi-turn" ("Entity-Relation
  Extraction as Multi-turn Question Answering"); the Anthology metadata has "Multi-Turn"; casing only, not an
  error.
- Claim: https://aclanthology.org/P19-1129.pdf, Sec. 4.2 (Generating Questions using Templates), p. 1344,
  verbatim: "At the relation and the tail-entity joint extraction stage, a question is generated by combing a
  relation-specific template with the extracted head-entity." Sec. 4.1 (p. 1344), verbatim: "A returned NONE token indicates
  that there is no answer in the given sentence." and "If an entity e extracted from the first stage is indeed a
  head-entity of a relation, then the QA model will extract the tail-entity by answering the corresponding
  question. Otherwise, the answer will be NONE and thus ignored." Sec. 1 (p. 1341): "each entity type and
  relation type is characterized by a question answering template". Fair. Note: the question contains only the
  HEAD entity; the tail is the extracted answer span (not a given pair being verified).

## 15. zhang2023aligning -- VERIFIED

- Bib: identical to https://aclanthology.org/2023.findings-acl.50.bib (title; Zhang, Kai; Jimenez Gutierrez,
  Bernal; Su, Yu; Findings of ACL 2023; pages 794--812; Toronto, Canada; ACL; DOI). PDF prints "Bernal Jiménez
  Gutiérrez" (accents); the Anthology form without accents is acceptable. No fix.
- Claim: https://aclanthology.org/2023.findings-acl.50.pdf, Sec. 3.3 (QA4RE Framework), p. 797, verbatim: "By
  integrating the given head and tail RE entities (Eh and Et) into the relation templates and using them as
  multiple-choice options" and "create options composed of pre-defined templates filled with Eh and Et entities"
  (with type constraints removing incompatible relation types). Sec. 1 (p. 795), verbatim: "our framework enables
  text-davinci-003 and FLAN-T5-XXLarge to achieve an average of 8.2% and 8.6% absolute improvements in F1,
  respectively." Table 1 (p. 798): text-003 avg 47.6 -> 55.8 (+8.2); FLAN-T5 XXLarge 46.5 -> 55.1 (+8.6).
  Setting: zero-shot, averaged over TACRED, RETACRED, TACREV, SemEval; improvement over the vanilla RE prompt for
  the same LLM. The numbers are in Sec. 1 and Table 1 (not in the abstract); Sec. 3.3 / Fig. 2 give the format.

## 16. zhang2025entity -- VERIFIED

- Bib: identical to https://aclanthology.org/2025.findings-naacl.224.bib (title with {LLM}s; Zhang, Fu; Yu,
  Hongsen; Cheng, Jingwei; Xu, Huangming; Findings of NAACL 2025; pages 4022--4037; Albuquerque, New Mexico; ACL;
  DOI). PDF author line agrees. (Source also has ISBN 979-8-89176-195-7; optional.) No fix.
- Claim: https://aclanthology.org/2025.findings-naacl.224.pdf, Sec. 4.3 (Ablation Study), p. 4029, verbatim:
  "Replacing the entit-pair-level candidate relations with the document-level candidate relations, the
  performance of our model experiences a significant drop, with F1 and Ign F1 dropping by 6.77 and 5.68
  respectively" (typo "entit-pair" is in the original). Table 3 (p. 4028), "Ablation study on the DocRED":
  53.77/51.25 -> 47.00/45.57. The section says the ablation is "on DocRED dev set". Abstract: "most approaches for
  candidate relation filtering are based on the document level, which results in insufficient correlation between
  candidate relations and entity pairs". Fair. Related number in the same ablation (may be useful): removing entity
  pair selection drops F1 by 31.94 (53.77 -> 21.83), attributed to the NA (no-relation) imbalance.

## 17. jimenezgutierrez2022thinking -- VERIFIED

- Bib: identical to https://aclanthology.org/2022.findings-emnlp.329.bib (title, 7 authors, Findings of EMNLP 2022,
  pages 4497--4512, Abu Dhabi, United Arab Emirates, ACL, DOI). PDF prints "Bernal Jiménez Gutiérrez" and "Clay
  Washington"; the Anthology record has "Jimenez Gutierrez, Bernal" and "Washington, Clayton". Following the
  Anthology record is acceptable (not an error); optionally restore the accents.
- Claim: https://aclanthology.org/2022.findings-emnlp.329.pdf, Sec. 1 (p. 4498), verbatim: "In-depth analyses
  further reveal that in-context learning struggles with the null class, e.g., sentences that contain no named
  entity (for NER) or entity pairs that hold none of the target relations (for RE), which is likely detrimental to
  IE tasks in general." Sec. 4.4.2 (RE Error Analysis, p. 4504), verbatim: "Based on the confusion matrices derived
  from LOOCV (Appendix F.1), the none relation in DDI is rarely predicted by GPT-3. This bias against the none
  class greatly degrades the model's precision given that the DDI dataset is, rightfully so, heavily skewed
  towards this class." Fair. Model: GPT-3 in-context learning (2022), not current LLMs.

## 18. mraz2026fewshot -- VERIFIED (preprint)

- Bib: matches https://arxiv.org/bibtex/2606.15412 and the abs-page citation metadata
  (https://arxiv.org/abs/2606.15412): title, authors Mraz, Jakob; Curk, Tomaž; Zupan, Blaž; year 2026; eprint
  2606.15412; cs.CL. DOI 10.48550/arXiv.2606.15412 resolves (HTTP 302 to the abs page). Versions: v1 13 Jun 2026,
  v2 3 Aug 2026; no journal reference on the abs page. Correctly typed @misc; the paper must present it as a
  preprint. No fix.
- Claim (read on v2 PDF): abstract, verbatim: "Pairwise classification achieves higher recall, whereas joint
  generation is more precise and computationally efficient." Sec. 5 (Conclusion), verbatim: "Across task
  formulations, classification and generation achieved broadly comparable F1 scores, with performance
  differences primarily reflecting a precision-recall trade-off rather than a clear superiority of either
  approach." Dataset BioREDirect confirmed (abstract). Locator: the "comparable F1" wording is in Sec. 5, the
  P/R trade-off also in Sec. 4 (Fig. 4). Caution if restating "F1 comparable": on the entity-pair (EP) task,
  Table 2 shows pairwise > joint for gemma-4-31B (0.68 vs 0.60 without reasoning); "broadly comparable" is the
  authors' own summary, so quote it as theirs.

## 19. jiang2011target -- VERIFIED (locator fix: p. 152, not p. 151)

- Bib: identical to https://aclanthology.org/P11-1016.bib (title with {T}witter; Jiang, Long; Yu, Mo; Zhou, Ming;
  Liu, Xiaohua; Zhao, Tiejun; ACL-HLT 2011; pages 151--160; Portland, Oregon, USA; ACL). PDF author line agrees.
  No fix.
- Claim: https://aclanthology.org/P11-1016.pdf. Sec. 1, p. 152 (left column, after the p. 151 footer): "Based on
  our manual evaluation of Twitter Sentiment output, about 40% of errors are because of this (see Section 6.1 for
  more details)." Sec. 6.1, p. 157, verbatim: "2) sentiments in some tweets are classified correctly but the
  sentiments are not truly about the query. The two types take up about 35% and 40% of the total errors,
  respectively." Fair, with the setting stated: the errors analysed are those of the commercial "Twitter
  Sentiment" (TS) system, on 5 queries x 20 randomly selected TS-labelled tweets, i.e. a small manual sample;
  "target-independent classifiers" is the authors' characterisation of such systems (Sec. 1).

## 20. jiang2019challenge -- VERIFIED

- Bib: identical to https://aclanthology.org/D19-1654.bib (title; Jiang, Qingnan; Chen, Lei; Xu, Ruifeng; Ao,
  Xiang; Yang, Min; EMNLP-IJCNLP 2019; pages 6280--6285; Hong Kong, China; ACL; DOI). No fix.
- Claim: https://aclanthology.org/D19-1654.pdf. Abstract, verbatim: "In existing ABSA datasets, most sentences
  contain only one aspect or multiple aspects with the same sentiment polarity, which makes ABSA task degenerate
  to sentence-level sentiment analysis." Sec. 4.2 (Experimental Results and Analysis), p. 6284, verbatim:
  "sentence-level sentiment classifiers (TextCNN and LSTM) achieve competitive results on SemEval-14 Restaurant
  Review dataset but perform poorly on MAMS datasets." Table 3: TextCNN ATSA Restaurant 75.9 vs MAMS 52.7. Fair.

## 21. plank2022problem -- VERIFIED

- Bib: identical to https://aclanthology.org/2022.emnlp-main.731.bib (title with ``Problem''; Plank, Barbara;
  EMNLP 2022; pages 10671--10682; Abu Dhabi, United Arab Emirates; ACL; DOI). No fix.
- Claim: https://aclanthology.org/2022.emnlp-main.731.pdf, Sec. 3 (Modeling and Human Label Variation), p. 10673,
  verbatim: "However, only using high-agreement instances can yield worse performance (Jamison and Gurevych, 2015)
  and it wastes data." Context: it is Plank's critique of "filtering" methods (removing low-agreement
  instances). Fair; if the paper uses it, it is Plank citing Jamison & Gurevych 2015 (secondary citation).

## 22. gao2021manual -- CLAIM-WRONG (interpretation of the 32%; quotes themselves verbatim)

- Bib: identical to https://aclanthology.org/2021.findings-acl.112.bib (title, 10 authors in PDF order, Findings of
  ACL-IJCNLP 2021, pages 1306--1318, Online, ACL, DOI). No fix.
- Quotes: https://aclanthology.org/2021.findings-acl.112.pdf. Abstract (p. 1306): "as much as 53% wrong labels at
  the entity pair level in the popular NYT10 dataset". Sec. 1 (p. 1307): "After checking 9, 744 sentences in the
  held-out set of NYT10 ..., we found that about 53% of the entity pairs are wrongly labeled". Sec. 3.1 (p. 1309):
  "Finally, we obtain the human-labeled test set with 9, 744 sentences, 32% of which are N/A instances. It
  contains 5, 174 entity pairs and 3, 899 manually-verified relational facts in total. ... at the fact level, the
  DS annotations only have a precision of 69.1% and a recall of 33.9%. At the entity pair level, the accuracy of DS
  labels is only 47.1%." All verbatim.
- What is wrong: the 9,744 sentences are NOT a natural sample of the held-out set and NOT only DS-positive
  sentences. Sec. 3.1 (p. 1308, "Annotated Test Set"): "we manually annotate all sentences that have a positive DS
  label. In addition to that, we also fine-tune a BERT model ... to predict the relations of all sentences
  originally labeled as N/A, and annotate the 5, 000 sentences with the highest predicted scores of non-N/A
  relations." (Same in Sec. 1, p. 1307: "manually label the top 5, 000 sentences scored as having a relation".)
  So (a) "32% N/A" is the N/A share of a mixture of DS-positive sentences and 5,000 BERT-selected DS-N/A
  sentences; it is not "32% of DS-positive sentences express no relation", and it is not comparable to Riedel et
  al.'s 31% (agentA.md's "Reading" and the main.md/agentA "co-mention != assertion: 32%" use are wrong). (b) The
  "53% wrongly labeled" is entity-pair-level accuracy 47.1%, dominated by MISSING relations (DS recall 33.9%),
  not by false positives. The number that does speak to DS false positives is fact-level DS precision 69.1%
  (about 31% of DS-asserted facts not confirmed by annotators), at fact (bag) level, not sentence level.
- Fix: describe the set as "a human-annotated NYT10 test set of 9,744 sentences (all DS-positive test sentences
  plus 5,000 DS-negative sentences selected by a BERT model)"; for the co-mention-is-not-assertion point cite
  "DS labels have fact-level precision 69.1% and recall 33.9%; entity-pair-level accuracy 47.1%" instead of 32%.

## Summary

| # | key | verdict | action |
|---|-----|---------|--------|
| 1 | dietterich1997solving | VERIFIED | none (full text from unofficial mirror of the published scan) |
| 2 | andrews2002support | VERIFIED | none |
| 3 | hoffmann2011knowledge | VERIFIED | locator: Phi-join passage is Sec. 3, p. 543 (not Sec. 4) |
| 4 | zeng2015distant | VERIFIED | none (Sec. 3.5 quote is p. 1758) |
| 5 | jia2019document | VERIFIED | wording: logsumexp/max pool mention REPRESENTATIONS, CKB document-level |
| 6 | ilse2018attention | VERIFIED | none (Sec. 2.1, p. 2129) |
| 7 | laban2022summac | CLAIM-WRONG (minor) | 73.5% is SummaC-Conv (learned aggregator), not max/mean; ZS max/mean: 56.4 -> 71.2 |
| 8 | verga2018simultaneously | VERIFIED | locator: Sec. 2.4, p. 875 |
| 9 | min2023factscore | FIX-BIB | author "Koh, Pang" -> "Koh, Pang Wei" |
| 10 | chowdhury2013exploiting | VERIFIED | caveat: Stage 1 is a negation-scope sentence filter on a rule-selected subset |
| 11 | xie2021revisiting | VERIFIED | note: sentence stage is multi-label relation-type detection |
| 12 | liu2019event | VERIFIED | note: PDF author order differs from Anthology/Crossref/S2 order used in bib; keep |
| 13 | zhong2021frustratingly | VERIFIED | state setting: dev sets, gold entities |
| 14 | li2019entity | VERIFIED | none (Sec. 4.1/4.2 on p. 1344) |
| 15 | zhang2023aligning | VERIFIED | none (+8.2/+8.6 in Sec. 1 p. 795 and Table 1, zero-shot, 4-dataset average) |
| 16 | zhang2025entity | VERIFIED | none (DocRED dev, Sec. 4.3, p. 4029) |
| 17 | jimenezgutierrez2022thinking | VERIFIED | none |
| 18 | mraz2026fewshot | VERIFIED | preprint; "broadly comparable F1" is the authors' own summary (Sec. 5) |
| 19 | jiang2011target | VERIFIED | locator: Intro sentence is p. 152; 40% is of Twitter Sentiment errors on 5x20 tweets |
| 20 | jiang2019challenge | VERIFIED | none |
| 21 | plank2022problem | VERIFIED | none (secondary citation of Jamison & Gurevych 2015) |
| 22 | gao2021manual | CLAIM-WRONG (interpretation) | 9,744-sentence set = all DS-positive + 5,000 BERT-selected DS-N/A sentences; 32% N/A is not a DS false-positive rate and not comparable to Riedel's 31%; 53% is mostly missing relations (DS recall 33.9%); use fact-level DS precision 69.1% for the co-mention != assertion point |

Totals: 19 VERIFIED (several with locator/wording notes), 1 FIX-BIB, 2 CLAIM-WRONG, 0 UNVERIFIABLE.
Bibliographic fields: every entry was compared field by field; apart from min2023factscore no field is wrong.
Process note: at 02:26 eleven scratch downloads (andrews.*, ilse.*, mraz.*) were written by mistake into
~/MetaP/classifier/ (a backgrounded `cd` did not apply). They were new untracked files created by
this verifier, moved to /tmp/v1s within a minute; git status shows no residue. No existing file was modified.
Finished 2026-10-05 ~02:38 CEST.
