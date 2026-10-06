# agentA findings (Q1: pair-level -> sentence-level aggregation; Q5: a sentence filter cannot populate a pair DB)

Run 2026-10-05, started 01:50 CEST. All bibliographic records below were fetched from the ACL Anthology .bib,
Crossref content negotiation or the NeurIPS proceedings site during this run. Tags: [abstract] = read the abstract only;
[full text, X] = read the PDF text at location X (PDF text extracted with pdftotext from the URL given).

Decisions taken (unattended run):
- Page numbers are given only when they are printed in the PDF; otherwise section names are used.
- Dietterich et al. 1997 full text was read from an unofficial mirror PDF (sci2s.ugr.es, KEEL site), because
  ScienceDirect blocks curl; bibliographic data come from Crossref.


INDEX OF NEW KEYS (all in agentA.bib; none duplicates the paper's existing keys):
- Q1(a) MIL / at-least-one: hoffmann2011knowledge, surdeanu2012multi, zeng2015distant, lin2016neural,
  bunescu2007learning, dietterich1997solving, andrews2002support
- Q1(b) aggregation: verga2018simultaneously, jia2019document, laban2022summac, min2023factscore, kamoi2023wice,
  ilse2018attention, wang2018revisiting
- Q1(c) sentence-level relation detection first: chowdhury2013exploiting, xie2021revisiting, zheng2021prgc
- Q1 contrast: liu2019event
- Q5(a) DS noise audits: takamatsu2012reducing, gao2021manual, zhu2020towards, jia2019arnor
- Q5(b) sentence-positive vs pair-negative: pyysalo2008comparative, tikk2013detailed (+ jia2019arnor test-set stats)
- Already-cited works whose text was read for exact figures: riedel2010modeling, rosenman2020exposing

---------------------------------------------------------------------------------------------------------------

## Q1(a) Multi-instance learning for RE and the "at least one" assumption

### riedel2010modeling (ALREADY CITED; reported here only for its text)
"Modeling Relations and Their Mentions without Labeled Text", ECML PKDD 2010, LNCS 6323, pp. 148-163,
DOI 10.1007/978-3-642-15939-8_10. Full text read at https://maroo.cs.umass.edu/getpdf.php?id=931 (UMass author copy).
- Exact statement of the assumption [full text, Sec. 1 Introduction, p. 149]:
  "In this paper we employ the following expressed-at-least-once assumption and show that it leads to more
  accurate results: If two entities participate in a relation, at least one sentence that mentions these two
  entities might express that relation."
  It is contrasted with the distant-supervision assumption [full text, Sec. 1, p. 148]: "If two entities
  participate in a relation, all sentences that mention these two entities express that relation."
- Note the word "might": Riedel's assumption is stated as a soft constraint; at training time "we only know that a
  relation is expressed at least once, but not in which sentences" [full text, Sec. 1, p. 149].
- Direction: bag = all sentences mentioning one entity PAIR; bag label = pair-level (KB) relation; instance =
  relation-mention variable per sentence (Fig. 1 caption, p. 152: "For each pair of entities that are mentioned
  together in at least one sentence we create one relation variable ... For each of the pairs of entity mentions
  that appear in a sentence we create one relation mention variable").
- For its noise figures see Q5(a).

### hoffmann2011knowledge
Hoffmann, Zhang, Ling, Zettlemoyer, Weld. "Knowledge-Based Weak Supervision for Information Extraction of
Overlapping Relations". ACL-HLT 2011, pp. 541-550. Anthology P11-1055 (no DOI in the Anthology record).
Verified: https://aclanthology.org/P11-1055.bib ; text: https://aclanthology.org/P11-1055.pdf
- WHAT IT SHOWS: MultiR; describes Riedel 2010 as casting weak supervision "as a form of multi-instance learning,
  assuming only that at least one of the sentences containing e1 and e2 are expressing r(e1, e2)" [full text,
  Sec. 1]; implements the aggregation as a hard max: "The factors Φjoin are deterministic OR operators ... which are
  included to ensure that the ground fact r(e) is predicted at the aggregate level ... only if at least one of the
  sen[tences]" [full text, Sec. 3 "Modeling Overlapping Relations", p. 543]. It also decodes sentence-level extractions ("MULTIR also produces
  accurate sentence-level predictions", Sec. 1).
- HOW WE RELATE: the deterministic-OR (= max for binary) aggregation of instance decisions into a bag decision is
  exactly the operator we use; MultiR is the canonical NLP citation for it. Direction differs (their bag = sentences
  of one pair, ours = pairs of one sentence), and their instances are latent (no instance labels), ours are
  teacher-labelled. Does not pre-empt the controlled comparison.

### surdeanu2012multi
Surdeanu, Tibshirani, Nallapati, Manning. "Multi-instance Multi-label Learning for Relation Extraction".
EMNLP-CoNLL 2012, pp. 455-465. Anthology D12-1042. Verified: https://aclanthology.org/D12-1042.bib
- WHAT IT SHOWS: MIML-RE jointly models "all the instances of a pair of entities in text and all their labels using a
  graphical model with latent variables" [abstract]; the relation-level classifier includes a feature that "models
  Hoffmann et al.'s at least one heuristic using a single feature, which is set to true if at least one mention in
  zi has the label r" [full text, Sec. 5 Experimental Results, features]. Also: "Bunescu and Mooney (2007) and Riedel et al. (2010) model
  distant supervision for relation extraction as a multi-instance single-label problem" [full text, Sec. 2].
- HOW WE RELATE: background; shows that the at-least-one rule was treated as a heuristic that a learned bag-level
  aggregator can replace. Does not pre-empt us.

### zeng2015distant
Zeng, Liu, Chen, Zhao. "Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks".
EMNLP 2015, pp. 1753-1762. DOI 10.18653/v1/D15-1203. Verified: https://aclanthology.org/D15-1203.bib
- WHAT IT SHOWS: PCNN+MIL; "distant supervised relation extraction is treated as a multi-instance problem ... the
  labels of the bags are known; however, the labels of the instances in the bags are unknown" [full text, Sec. 1];
  "The objective of multi-instance learning is to discriminate bags rather than instances" [full text, Sec. 3.5];
  training uses only the highest-scoring instance per bag (Algorithm 1, Eq. 9; Lin et al. 2016 summarise it as
  "only uses the sentence with the highest probability in each set for training").
- HOW WE RELATE: the neural max-over-instances bag decision; again bag = sentences of one pair. Background.

### lin2016neural
Lin, Shen, Liu, Luan, Sun. "Neural Relation Extraction with Selective Attention over Instances". ACL 2016
(Vol. 1), pp. 2124-2133. DOI 10.18653/v1/P16-1200. Verified: https://aclanthology.org/P16-1200.bib
- WHAT IT SHOWS: criticises the max/at-least-one rule of Zeng 2015: "The method assumes that at least one sentence
  that mentions these two entities will express their relation, and only selects the most likely sentence for each
  entity pair in training and prediction. It's apparent that the method will lose a large amount of rich information
  containing in neglected sentences." [full text, Sec. 1]; replaces max with sentence-level attention [abstract].
- HOW WE RELATE: the standard counter-argument to max aggregation (max discards information from other instances).
  A reviewer could ask why we use max rather than a soft aggregator; cite it when justifying max (simplicity;
  max is the at-least-one semantics of the sentence question).

### bunescu2007learning
Bunescu, Mooney. "Learning to Extract Relations from the Web using Minimal Supervision". ACL 2007, pp. 576-583.
Anthology P07-1073 (no DOI in the Anthology record). Verified: https://aclanthology.org/P07-1073.bib
- WHAT IT SHOWS: the first explicit MIL formulation for RE: "Multiple instance learning (MIL) is a machine learning
  framework that exploits this sort of weak supervision, in which a positive bag is a set of instances which is
  guaranteed to contain at least one positive example, and a negative bag is a set of instances all of which are
  negative." [full text, Sec. 1]. Bags are web sentences retrieved for a positive/negative entity pair.
- HOW WE RELATE: origin of MIL-for-RE; background citation only.

### dietterich1997solving
Dietterich, Lathrop, Lozano-Pérez. "Solving the multiple instance problem with axis-parallel rectangles".
Artificial Intelligence 89(1-2):31-71, 1997. DOI 10.1016/S0004-3702(96)00034-3. Verified via Crossref
(https://doi.org/10.1016/S0004-3702(96)00034-3, BibTeX content negotiation). Text read from mirror
https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1997%20-%20DietterichLathropLozano-Perez%20-%20AI.pdf
- WHAT IT SHOWS: defines the multiple instance problem [abstract]: "a single example object may have many alternative
  feature vectors (instances) that describe it, and yet only one of those feature vectors may be responsible for the
  observed classification of the object." Standard assumption [full text, Sec. 1, p. 34]: "An hypothesis ĝ is
  consistent with a set of training examples if it classifies every feature vector of every negative example as
  negative and if it classifies at least one feature vector of every positive example as positive."
- HOW WE RELATE: origin of the "at least one positive instance <=> positive bag" (standard MI) assumption that our
  sentence = max over pairs rule instantiates. Cite once.

### andrews2002support
Andrews, Tsochantaridis, Hofmann. "Support Vector Machines for Multiple-Instance Learning". Advances in Neural
Information Processing Systems 15 (NIPS 2002), MIT Press. Verified:
https://proceedings.neurips.cc/paper_files/paper/2002/hash/3e6260b81898beacda3d16db379ed329-Abstract.html
(BibTeX from the same site; the site gives no page numbers).
- WHAT IT SHOWS: the clearest one-sentence statement of the standard MI assumption and of "bag classifier induced by
  an instance classifier" [full text, Sec. 1]: "In the important case of binary classification, this implies that a
  bag is "positive" if at least one of its member patterns is a positive example. MIL differs from the general
  set-learning problem in that the set-level classifier is by design induced by a pattern-level classifier."
  Proposes mi-SVM (instance-level margin, imputed instance labels) and MI-SVM (bag-level margin via the most
  positive instance) [abstract; full text, Sec. 2-3].
- HOW WE RELATE: supports our framing that a set-level (sentence) decision can be read off an instance-level (pair)
  classifier. But mi-SVM vs MI-SVM is NOT our comparison: both train from bag labels only. Optional citation.

### Answer to the precise question in Q1(a)
- In every DS-MIL work above (Riedel 2010, Hoffmann 2011, Surdeanu 2012, Zeng 2015, Lin 2016, Bunescu & Mooney
  2007), the bag is the set of sentences mentioning one entity pair, instances are sentences, and the bag decision is
  a PAIR-level (corpus-level) fact. The instance labels are latent; training uses bag labels only.
- Our direction is the transpose: bag = candidate pairs in one sentence, instance = pair, bag decision = sentence
  level; and instance (pair) labels are available (teacher-labelled) at training time.
- None of the DS-MIL papers read here makes a SENTENCE-level "does this sentence express any relation" decision by
  max over pair-level scores, and none compares such a decision with a model trained on sentence labels directly.
  (Search for such a comparison elsewhere: see the end of Q1.)

---------------------------------------------------------------------------------------------------------------

## Q5(a) Distant-supervision noise audits: share of sentences mentioning a related pair that do NOT express it

### riedel2010modeling (ALREADY CITED) -- exact figures
Full text read at https://maroo.cs.umass.edu/getpdf.php?id=931.
- [full text, Sec. 1, p. 149]: "In fact, by manual inspection (see section 2) we find that the distant supervision
  assumption is violated approximately 13% of the time when aligning Freebase to Wikipedia, but 31% when aligning
  to the New York Times Corpus [22]." Footnote 2 (p. 149): "This is the average over three relation types:
  nationality, contains and place_of_birth."
- Protocol [full text, Sec. 2, p. 151]: "we test it for the case of Freebase and two different text corpora:
  Wikipedia articles and the New York Times corpus. To this end we consider three frequent relation types:
  nationality, place_of_birth, and contains. For each type we sample 100 relation mention candidates from both
  corpora, and evaluate whether these candidates are or are not expressing the relation in question."
- Table 1 (p. 151), caption: "Percentage of times a related pair of entities is mentioned in the same sentence, but
  where the sentence does not express the corresponding relation". Values (NYT / Wikipedia):
  nationality 38% / 20%; place_of_birth 35% / 20%; contains 20% / 10%.
- CAUTION (our arithmetic, not stated in the paper): the NYT column averages to (38+35+20)/3 = 31%, matching the
  text; the Wikipedia column averages to (20+20+10)/3 = 16.7%, NOT the 13% stated in the text. If the paper cites
  Riedel, quote the NYT 31% figure (consistent) and either avoid the Wikipedia 13% or cite the Table 1 values.
- Sample size: 100 candidates per relation per corpus (so 300 per corpus).

### takamatsu2012reducing
Takamatsu, Sato, Nakagawa. "Reducing Wrong Labels in Distant Supervision for Relation Extraction". ACL 2012
(Vol. 1), pp. 721-729. Anthology P12-1076 (no DOI in the Anthology record). Verified: https://aclanthology.org/P12-1076.bib
- WHAT IT SHOWS: generative model of the DS labelling process that predicts which DS labels are wrong [abstract].
  Two useful figures [full text, Sec. 1]: "The relaxation is equivalent to the DS assumption when a labeled
  pair of entities is mentioned once in a target corpus (Riedel et al., 2010). In fact, 91.7% of entity pairs appear
  only once in Wikipedia articles (see Section 7)." and [full text, Sec. 4 "Wrong Label Reduction"]: "in our
  Wikipedia corpus, more than 6% of the sentences containing the pattern "[Person] moved to [Location]", which does
  not express place of death, are labeled as place of death". Also [full text, Sec. 7]: "Multi-instance
  learning does not work for wrong labels assigned to entity pairs that appear only once in a corpus. In fact, 72% of
  entity pairs that appeared with this pattern and were wrongly labeled as place of death appeared only once in the
  corpus."
- Takamatsu does NOT give a corpus-wide "share of DS sentences that are wrong" figure (I searched the PDF for all
  percentages); its numbers are pattern-specific.
- HOW WE RELATE: supports the point that co-mention != assertion, and that the at-least-once relaxation collapses
  when a pair occurs in one sentence (the common case in our setting: one passage, one candidate pair).

### gao2021manual
Gao, Han, Bai, Qiu, Xie, Lin, Liu, Li, Sun, Zhou. "Manual Evaluation Matters: Reviewing Test Protocols of Distantly
Supervised Relation Extraction". Findings of ACL-IJCNLP 2021, pp. 1306-1318. DOI 10.18653/v1/2021.findings-acl.112.
Verified: https://aclanthology.org/2021.findings-acl.112.bib
- WHAT IT SHOWS [abstract]: auto-labelled DS test data "produce as much as 53% wrong labels at the entity pair level
  in the popular NYT10 dataset". [full text, Sec. 1]: "After checking 9, 744 sentences in the held-out set of NYT10
  (Riedel et al., 2010), the most popular DS-RE dataset, we found that about 53% of the entity pairs are wrongly
  labeled". [full text, Sec. 3.1 "NYT10 Dataset"]: "For NYT10, we manually annotate sentences with positive DS
  relations in its held-out test set." ... "Finally, we obtain the human-labeled test set with 9, 744 sentences, 32%
  of which are N/A instances. It contains 5, 174 entity pairs and 3, 899 manually-verified relational facts in total.
  After comparing it with the corresponding original DS-generated labels, we found that at the fact level, the DS
  annotations only have a precision of 69.1% and a recall of 33.9%. At the entity pair level, the accuracy of DS
  labels is only 47.1%."
- Reading: of NYT10 held-out sentences that DS labels as expressing a KB relation for their entity pair, 32% express
  no relation at all for that pair (N/A) under human annotation -- the modern, larger-sample replication of Riedel's
  31% on NYT.
- HOW WE RELATE: strongest recent number for "a sentence mentioning a related pair often does not assert it". Note
  that this is about DS (KB-driven) labels, i.e. pair-in-KB -> sentence; our situation is the converse
  (sentence-positive -> pair). Use it for the co-mention != assertion point, not as a direct estimate of our 0.724.

### zhu2020towards
Zhu, Wang, Yu, Zhou, Chen, Zhang, Zhang. "Towards Accurate and Consistent Evaluation: A Dataset for
Distantly-Supervised Relation Extraction". COLING 2020, pp. 6436-6447. DOI 10.18653/v1/2020.coling-main.566.
Verified: https://aclanthology.org/2020.coling-main.566.bib
- WHAT IT SHOWS: NYT-H, a NYT10-derived dataset whose test sentences were annotated "Yes"/"No" for whether "a
  sentence actually expresses the DS-annotated relation for the given entity pair" [full text, Sec. 3.2]. Kappa
  0.753 [Sec. 3.2]. [full text, Sec. 3.4 "Dataset Statistics"]: "In the test set, 5,202 out of 9,955 instances are annotated as
  "Yes", which also indicates the wrong label problem of distant supervision." Table 2: 2,277 of 3,735
  test bag-relation entries have at least one "Yes" instance.
- Derived (our arithmetic from the quoted counts): 4,753/9,955 = 47.7% of DS-positive test sentences do not express
  the DS relation. (Differs from Gao 2021's 32% because NYT-H asks about the specific DS relation, binary Yes/No,
  after their filtering; Gao's 32% is "no relation at all".)
- HOW WE RELATE: second independent manual audit of NYT; supports co-mention != assertion.

### jia2019arnor (for Q5(a) only marginal; see Q5(b))
Jia, Dai, Xiao, Wu. "ARNOR: Attention Regularization based Noise Reduction for Distant Supervision Relation
Classification". ACL 2019, pp. 1399-1408. DOI 10.18653/v1/P19-1135. Verified: https://aclanthology.org/P19-1135.bib
- Its only corpus-level noise figure is second-hand: "Mintz et al. (2009) reports that distant supervision may lead
  to more than 30% noisy instances." [full text, Sec. 1]. I did not verify that this figure appears in Mintz 2009
  (mintz2009distant, already cited); do NOT use it via ARNOR.
- Its own manual test set is useful for Q5(b) (pairs per sentence), see below.

---------------------------------------------------------------------------------------------------------------

## Q1(b) Aggregating finer-grained predictions into coarser decisions

### verga2018simultaneously
Verga, Strubell, McCallum. "Simultaneously Self-Attending to All Mentions for Full-Abstract Biological Relation
Extraction". NAACL-HLT 2018 (Vol. 1), pp. 872-884. DOI 10.18653/v1/N18-1080. Verified: https://aclanthology.org/N18-1080.bib
- WHAT IT SHOWS: scores all mention pairs of an abstract and "aggregate[s] over mention pairs using a soft
  approximation of the max function in order to perform multi-instance learning" [full text, Sec. 1]; "we use the
  LogSumExp function to aggregate the relation scores from A across all pairs of mentions of phead and ptail ... The
  LogSumExp scoring function is a smooth approximation to the max function and has the benefits of aggregating
  information from multiple predictions and propagating dense gradients as opposed to the sparse gradient updates of
  the max" [full text, Sec. 2, subsection "Entity Level Prediction"]. Abstract: "All-pairs mention scores allow us to perform multi-instance learning by
  aggregating over mentions to form entity pair representations" [abstract].
- HOW WE RELATE: biomedical precedent for (smooth-)max aggregation of fine-grained scores into a coarser decision,
  but the coarse unit is the entity pair (mention pair -> entity pair), and training uses only the coarse (entity
  pair) labels. Not a sentence-level decision; no comparison with a coarse-trained model. Background, does not
  pre-empt.

### jia2019document
Jia, Wong, Poon. "Document-Level N-ary Relation Extraction with Multiscale Representation Learning". NAACL-HLT 2019
(Vol. 1), pp. 3693-3704. DOI 10.18653/v1/N19-1370. Verified: https://aclanthology.org/N19-1370.bib
- WHAT IT SHOWS: builds mention-level representations within discourse units and combines them into entity-level
  representations with an aggregation operator [full text, Sec. 3.2]: "A standard choice for C is max pooling, which
  works well if it is pretty clear-cut whether a mention tuple expresses a relation. In practice, however, the mention
  tuples could be ambiguous and less than certain individually, yet collectively express a relation in the document.
  This motivates us to experiment with logsumexp, the smooth version of max". Result [full text, Sec. 4, ablation paragraph on
  Table 5]: "If we replace it with max pooling, the performance drops substantially across the board, as shown in
  Table 5. For example, MULTISCALE lost 3.8 absolute points in AUC. Such difference is also observed in Verga et al.
  (2018)."
- HOW WE RELATE: a reviewer could cite this to ask why hard max rather than logsumexp/noisy-or. Their setting
  (aggregating many weak, noisy mentions of the same tuple across a document) differs from ours (aggregating
  different pairs within one sentence, where the sentence question is a logical OR over pairs). Worth one sentence
  justifying max (or reporting a soft alternative).

### laban2022summac
Laban, Schnabel, Bennett, Hearst. "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in
Summarization". TACL 10:163-177, 2022. DOI 10.1162/tacl_a_00453. Verified: https://aclanthology.org/2022.tacl-1.10.bib
- WHAT IT SHOWS: past NLI-based inconsistency detection "suffered from a mismatch in input granularity between NLI
  datasets (sentence-level), and inconsistency detection (document level)" [abstract]; SummaC-ZS "performs zero-shot
  aggregation by combining sentence-level scores using max and mean operators" [full text, Sec. 1]: max over document
  sentences for each summary sentence, then mean over summary sentences [Sec. 3.2]. Same NLI model at different input
  granularities [full text, Sec. 5.3.3 "Choice of Granularity"]: "The MNLI-only trained model achieves lowest
  performance when used with full text granularity on the document level, and performance steadily increases from
  56.4% to 73.5% as granularity is made finer both on the document and summary side." and "Overall, we find that
  finer granularity for the document and summary is beneficial in terms of performance".
- HOW WE RELATE: closest conceptual analogue for "decide at the fine granularity, aggregate with max, rather than ask
  the coarse question directly", and it is a controlled same-model comparison of granularity. Differences: the NLI
  model is fixed (trained on sentence pairs), so SummaC compares input granularity at inference time, not models
  TRAINED at two granularities on the same data as we do. Supports us; must be cited as precedent for the idea.

### min2023factscore
Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi. "FActScore: Fine-grained Atomic Evaluation of
Factual Precision in Long Form Text Generation". EMNLP 2023, pp. 12076-12100. DOI 10.18653/v1/2023.emnlp-main.741.
Verified: https://aclanthology.org/2023.emnlp-main.741.bib
- WHAT IT SHOWS [abstract]: "generations often contain a mixture of supported and unsupported pieces of information,
  making binary judgments of quality inadequate"; FActScore "breaks a generation into a series of atomic facts and
  computes the percentage of atomic facts supported by a reliable knowledge source."
- HOW WE RELATE: the "decompose, verify each unit, aggregate" pattern and the argument that a single coarse binary
  judgement is inadequate when a text mixes supported and unsupported units -- the same reason a sentence label
  cannot populate a pair database. Aggregator is a proportion, not max. Supporting citation (one sentence).

### kamoi2023wice
Kamoi, Goyal, Rodriguez, Durrett. "WiCE: Real-World Entailment for Claims in Wikipedia". EMNLP 2023, pp. 7561-7583.
DOI 10.18653/v1/2023.emnlp-main.470. Verified: https://aclanthology.org/2023.emnlp-main.470.bib
- WHAT IT SHOWS: entailment judgements for sub-sentence units; "we propose an automatic claim decomposition strategy
  using GPT3.5 which we show is also effective at improving entailment models' performance on multiple datasets at
  test time" [abstract]. Claim labels are derived from subclaim labels [full text, Sec. 2, "Deriving claim labels from
  subclaim labels"]: "if all subclaims are SUPPORTED or NOT-SUPPORTED, we assign that as the claim-level label. Else,
  we assign PARTIALLY-SUPPORTED."
- HOW WE RELATE: another decompose-then-aggregate verification result (AND-like aggregation for claims vs our OR/max
  for "any interaction"). Optional; cite only if space allows.

### ilse2018attention (MIL: instance-level vs embedding-level approaches)
Ilse, Tomczak, Welling. "Attention-based Deep Multiple Instance Learning". ICML 2018, PMLR 80:2127-2136.
Verified: https://proceedings.mlr.press/v80/ilse18a.html (BibTeX from that page; no DOI).
- WHAT IT SHOWS: distinguishes "(i) The instance-level approach: The transformation f is an instance-level classifier
  that returns scores for each instance. Then individual scores are aggregated by MIL pooling to obtain θ(X)" from
  "(ii) The embedding-level approach ... The bag representation is further processed by a bag-level classifier"
  [full text, Sec. 2.1]; then: "It is advocated in (Wang et al., 2016) that the latter approach is preferable in terms
  of the bag level classification performance. Since the individual labels are unknown, there is a threat that the
  instance-level classifier might be trained insufficiently and it introduces additional error to the final
  prediction." [full text, Sec. 2.1].
- HOW WE RELATE: the closest MIL-literature statement of the question we test, and it predicts the OPPOSITE
  (bag-level classifier preferable). The stated reason is that instance labels are unknown; in our design the
  instance (pair) labels are given by the same teacher, which removes that reason. Cite it to frame the
  contribution: "with instance supervision available, does the instance-level route still lose?". Contrast, not
  pre-emption.

### wang2018revisiting
Wang, Yan, Tang, Bai, Liu. "Revisiting multiple instance neural networks". Pattern Recognition 74:15-24, 2018.
DOI 10.1016/j.patcog.2017.08.026. Verified via Crossref (https://doi.org/10.1016/j.patcog.2017.08.026); text read
from the arXiv preprint of the same title, https://arxiv.org/abs/1610.02501 (journal version not accessible).
- WHAT IT SHOWS [arXiv preprint, Sec. I]: "Instance-space paradigm learns instance classifier and performs bag
  classification by aggregating the responses of instance-level classifier." and, comparing mi-Net (instance-space)
  and MI-Net (embedded-space): "Compared with mi-Net, MI-Net can obtain better bag classification accuracy." ...
  "experiments have shown that the MI-Net outperforms mi-Net in more cases."
- HOW WE RELATE: the empirical source behind Ilse's statement; trained from bag labels only on classical MIL
  benchmarks. A reviewer may cite it against us; our answer is the instance-label availability (same teacher).
  Optional if ilse2018attention is cited.

### Q1(b) answer
- Fine-to-coarse aggregation with (smooth) max is standard (Verga 2018, Jia 2019 for mentions -> entity pairs;
  SummaC for sentence NLI -> document consistency; FActScore/WiCE for atomic claims -> text). None of them trains two
  models with the same teacher/data/encoder, one at the fine and one at the coarse granularity, and compares them on
  the coarse decision. SummaC is the nearest (same model, different inference granularity; finer is better).
- The MIL literature (Ilse 2018 citing Wang 2016/2018) states the opposite expectation for bag classification
  (bag-level/embedding-level preferable), explicitly because instance labels are unknown. Our controlled comparison
  with instance labels available is therefore not answered by that literature.

---------------------------------------------------------------------------------------------------------------

## Q1(c) Relation detection / "relation existence" as a filter or first stage

### chowdhury2013exploiting
Chowdhury, Lavelli. "Exploiting the Scope of Negations and Heterogeneous Features for Relation Extraction: A Case
Study for Drug-Drug Interaction Extraction". NAACL-HLT 2013, pp. 765-771. Anthology N13-1093 (no DOI in the Anthology record).
Verified: https://aclanthology.org/N13-1093.bib
- WHAT IT SHOWS: a two-stage biomedical RE system whose Stage 1 is a SENTENCE-level classifier that filters
  sentences unlikely to contain any DDI before Stage 2 pair classification. [full text, Sec. 2.1 "Stage 1: Exploiting
  scope of negation to filter out sentences"]: "In the Stage 1, any sentence that contains at least one DDI is
  considered by the classifier as a positive (training/test) instance. Other sentences are considered as negative
  instances." [full text, Sec. 3 "Results and Discussion"]: "Unlike Stage 1, in Stage 2 where we
  train the hybrid kernel based RE classifier and use it for RE (i.e. DDI extraction) from the test data, sentences
  are not the RE training/test instances. Instead, a RE instance corresponds to a candidate mention pair." Scale of
  filtering [full text, Sec. 3]: "334 sentences (7.83% of the total sentences) containing at least 2 drug mentions
  were identified by our proposed classifier ... as unlikely to have any DDI ... Only 19 of these sentences were
  incorrectly identified."
- HOW WE RELATE: precedent for a sentence-level "contains at least one relation" label derived from pair labels and
  used as a filter in front of a pair classifier (biomedical, DDI). Their sentence classifier only removes negatives;
  the pair decision is still made by the pair classifier. Supports our "only pair answers populate a pair DB".
  Same authors' shared-task version: FBK-irst, SemEval-2013 (Anthology S13-2057), Sec. 2.1 ("Any sentence that
  contains at least one relation of interest is considered by the less informative sentence (LIS) classifier as a
  positive (training/test) instance") -- verified, not added to the .bib to avoid duplication.

### xie2021revisiting
Xie, Liang, Liu, Huang, Huang, Xiao. "Revisiting the Negative Data of Distantly Supervised Relation Extraction".
ACL-IJCNLP 2021 (Vol. 1), pp. 3572-3581. DOI 10.18653/v1/2021.acl-long.277. Verified:
https://aclanthology.org/2021.acl-long.277.bib
- WHAT IT SHOWS [abstract]: "we propose a pipeline approach, dubbed RERE, that first performs sentence classification
  with relational labels and then extracts the subjects/objects." [full text, Sec. 2.2]: three paradigms; "P1 The
  first paradigm is a pipeline that begins with named entity recognition (NER) and then classifies each entity pair
  into different relations, i.e., [s, o then r]" vs "P3 The third paradigm first perform sentence-level relation
  detection (cf. P1, which is at entity pair level.) then extract subjects and entities, i.e., [r then s, o]." The
  argument is class imbalance: "Suppose a sentence contains m entities, the classifier has to decide relation from
  O(m^2) entity pairs, while in reality, relations are often sparse, i.e., O(m). In other words, most entity pairs in
  P1 do not form valid relation, thus resulting in a low class prior. The situation is even worse when the sentence
  contains more entities" [full text, Sec. 2.2; Table 2 gives the class priors].
- HOW WE RELATE: the strongest published argument FOR sentence-level detection first (class-prior argument). It
  contrasts with our "ask about the pair" position and should be cited and answered: in our setting candidate pairs
  are proposed by co-occurrence anyway, so the pair question must be answered; and their own O(m^2) vs O(m) argument
  is exactly why a sentence-positive label does not transfer to its pairs when m > 2 (consistent with our gain being
  confined to sentences with more than two entities).

### zheng2021prgc
Zheng, Wen, Chen, Yang, Zhang, Zhang, Zhang, Qin, Xu, Zheng. "PRGC: Potential Relation and Global Correspondence
Based Joint Relational Triple Extraction". ACL-IJCNLP 2021 (Vol. 1), pp. 6225-6235. DOI 10.18653/v1/2021.acl-long.486.
Verified: https://aclanthology.org/2021.acl-long.486.bib
- WHAT IT SHOWS [abstract]: "we design a component to predict potential relations, which constrains the following
  entity extraction to the predicted relation subset rather than all relations"; [full text, Sec. 1]: "Given a
  sentence, PRGC first predicts a subset of potential relations and a global matrix which contains the correspondence
  score between all subjects and objects; then performs sequence tagging to extract subjects and objects for each
  potential relation in parallel; finally enumerates all predicted entity pairs, which are then pruned by the global
  correspondence matrix." and the potential-relation component "is overall beneficial, even though it introduces the
  exposure bias".
- HOW WE RELATE: in general-domain joint extraction, sentence-level relation existence is used as a first stage,
  but the final output is still decided at the subject-object pair level (global correspondence). Supports "the
  sentence decision alone is insufficient for triples". Background.

### Q1(c) answer
Sentence-level relation detection as a first stage is established (Chowdhury & Lavelli 2013 in biomedical DDI;
RERE and PRGC in general-domain joint extraction; RERE also cites HRL, Takanobu et al. 2019, which I did not verify).
In all of them the sentence detector is a filter or a conditioning step and the triple is still decided per pair.
None reads the sentence decision off the pair classifier and compares it with the dedicated sentence detector.

---------------------------------------------------------------------------------------------------------------

## Q5(b) How often a relation-positive sentence is negative for a given candidate pair

### pyysalo2008comparative  (best external analogue of our "perfect sentence filter -> pair precision 0.724")
Pyysalo, Airola, Heimonen, Björne, Ginter, Salakoski. "Comparative analysis of five protein-protein interaction
corpora". BMC Bioinformatics 9(Suppl 3):S6, 2008. DOI 10.1186/1471-2105-9-S3-S6. Verified via Crossref
(https://doi.org/10.1186/1471-2105-9-S3-S6) and PubMed (PMID 18426551). Full text read from PMC open access via the
NCBI BioC API (https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/18426551/unicode).
- Co-occurrence baseline [full text, Methods, "Co-occurrence method"]: "One very simple PPI extraction method is to
  assign a relationship between all annotated entities co-occurring within a sentence."
- Its precision on the five corpora [full text, Results, Table 2 "PPI extraction performance"], co-occurrence P:
  AIMed 0.17, BioInfer 0.13, HPRD50 0.38, IEPA 0.41, LLL 0.50 (recall ~1).
- Corpus statistics [full text, Results, Table 3 "Corpus statistics"], per-sentence averages (entity pairs /
  interactions) and fraction of sentences with no interactions: AIMed 3.0 / 0.5, 69%; BioInfer 9.4 / 1.3, 48%;
  HPRD50 3.0 / 1.1, 38%; IEPA 1.7 / 0.7, 37%; LLL 4.3 / 2.1, 0%.
- The paper's own identity [full text, Results]: "the average number of interactions divided by the average number
  of entity pairs per sentence (below abbreviated I/EP) equals the precision of the co-occurrence method evaluated
  above." and "As more proteins are annotated, we would not expect I to grow more than linearly, while EP grows
  quadratically."
- Oracle entity filter [full text, Results, Table 4 "PPI extraction performance on filtered corpora"; caption:
  "Precision and F-score for the co-occurrence and RelEx methods on the corpora with only entities that participate
  in an interaction preserved"]: co-occurrence P becomes AIMed 0.53, BioInfer 0.53, HPRD50 0.64, IEPA 0.88, LLL 0.50.
- DERIVED (our reasoning, not stated by the authors; state it as such if used):
  (1) In LLL every sentence contains at least one interaction (0% of sentences with no interactions), so the
      co-occurrence baseline there IS a perfect sentence filter followed by "all pairs positive": pair precision 0.50.
      LLL is tiny (77 sentences, Table 1), so treat this as illustrative.
  (2) Keeping only entities that participate in an interaction removes every pair from interaction-free sentences
      and also removes pairs with non-interacting entities inside positive sentences; it is therefore a stronger
      oracle than a perfect sentence filter, and the pair precision of a perfect sentence filter cannot exceed the
      Table 4 values (0.53 AIMed, 0.53 BioInfer, 0.64 HPRD50, 0.88 IEPA, 0.50 LLL). This assumes the filtering is per
      entity occurrence within sentences, which the caption implies but the text does not spell out.
- HOW WE RELATE: independent biomedical evidence (PPI) that a sentence-level "contains an interaction" oracle leaves
  a large share of negative pairs; our 0.724 sits inside that range. Also supports the >2-entity mechanism (EP grows
  quadratically, I linearly). Strongly recommended citation.

### tikk2013detailed
Tikk, Solt, Thomas, Leser. "A detailed error analysis of 13 kernel methods for protein-protein interaction
extraction". BMC Bioinformatics 14:12, 2013. DOI 10.1186/1471-2105-14-12. Verified via Crossref
(https://doi.org/10.1186/1471-2105-14-12) and PubMed (PMID 23323857); full text read via the NCBI BioC API (PMC OA).
- WHAT IT SHOWS [full text, Results]: "In general, long sentences have more protein mentions, and the number of
  pairs increases quadratically with the number of mentions. We investigated the class distribution of pairs
  depending on the number of proteins in the sentence (see Figure 7). We can see that the more protein mentions a
  sentence exhibits, the lower the ratio of positive pairs." Also: "Positive pairs occur more often in shorter
  sentences with less proteins (see Figures 5 and 6), and most of the analyzed classifiers fail to capture the
  characteristics of rare positive pairs in longer sentences."
- HOW WE RELATE: direct support for the mechanism behind our result (gain only on sentences naming more than two
  entities): the share of negative pairs inside a sentence grows with the number of entities. No single summary
  number is given in the text (the numbers are in Figure 7, which I did not read).

### rosenman2020exposing (ALREADY CITED) -- figures relevant to Q5(b)
Full text read at https://aclanthology.org/2020.emnlp-main.302.pdf
- [full text, Sec. 3 "Challenge Set"]: "in only 17.2% of the sentences in TACRED more than a single pair is annotated
  (and only 3.46% of the sentences have more than one different annotated labels in the sentence)". Separately
  [full text, Sec. 1]: "not all e1, e2 pairs in the dataset are annotated (only 17.2% of the entity pairs whose type
  match a TACRED relation are)". (Both are 17.2% in the paper; they are different quantities.)
- Their challenge set CRE [full text, Sec. 3]: "The resulting challenge set (CRE) has 3,000 distinct sentences, and
  10,844 classification instances." ... "In 57% of the sentences, there are at least two classification instances
  with conflicting labels, indicating the use of the event+type heuristic. On average there are 3.7 candidate
  entity-pairs per sentence."
- HOW WE RELATE: the most direct general-domain number for "sentence expresses the relation, but not for this pair":
  57% of CRE sentences contain both a positive and a negative candidate pair for the same relation. Caveat: CRE was
  sampled from sentences on which a TACRED model predicted the relation, so 57% is not a corpus base rate. Also shows
  that TACRED under-represents multi-pair sentences, which is why standard RE benchmarks hide this effect.

### jia2019arnor (manual sentence-level NYT test set statistics)
Verified: https://aclanthology.org/P19-1135.bib (full entry under Q5(a)).
- [full text, Sec. 1]: "We publish a better manually labeled sentence-level test set for evaluating the performance
  of RC models. This test set contains 1,024 sentences and 4,543 entity pairs, and is carefully annotated to ensure
  accuracy." Table 1 ("Statistics of the dataset in our experiments"): test #Sentences 1,024, #Instances 4,543,
  #Positive instances 671; training 235,253 / 371,461 / 110,518.
- DERIVED: 4.4 candidate pairs per test sentence; 14.8% of test pairs positive. The paper does not report how many
  sentences are positive, so the within-positive-sentence negative share cannot be computed from it.
- HOW WE RELATE: shows that once all entity pairs of a sentence are annotated (rather than one pair per sentence, as
  in the earlier Ren et al. 2017 set they criticise: "this test set was annotated with only one entity pair for one
  sentence" [Sec. 4.1]), most candidate pairs are negative. Weak support; optional.

### Not useful for Q5(b) (checked)
- DocRED (Yao et al. 2019, ACL, P19-1074, verified): gives document-level statistics only ("19.5 entities on
  average" per document; "most entity pairs in a document do not contain relations", Sec. 2.1; "at least 40.7% of the
  relational facts in DocRED can only be extracted from multiple sentences", Sec. 1). No sentence-level positive/
  negative-pair figure. Not added to the .bib.
- SemEval-2013 DDI overview (Segura-Bedmar et al., S13-2056, verified): Table 1 gives sentence and DDI counts per
  split but no negative-pairs-in-positive-sentences figure. Not added.
- RERE (xie2021revisiting) gives pair-level class priors per (pair, relation) in Table 2 (e.g. NYT10-HRL 0.01421)
  but not per sentence; listed under Q1(c).

---------------------------------------------------------------------------------------------------------------

## Additional contrasting analogue found during the search (Q1)

### liu2019event
Liu, Li, Zhang, Yang, Zhou (Anthology author order; the PDF lists Liu, Li, Zhou, Yang, Zhang). "Event Detection
without Triggers". NAACL-HLT 2019 (Vol. 1), pp. 735-744. DOI 10.18653/v1/N19-1080. Verified:
https://aclanthology.org/N19-1080.bib (also Semantic Scholar match).
- WHAT IT SHOWS: drops fine-grained trigger annotation and detects event types at the sentence level, recast as
  binary (sentence, event type) decisions: "a given sentence s attached each pre-defined event type t forms an
  instance, which is expected to be labeled with 0 or 1 according to whether s contains an event of type t" [full
  text, Sec. 1]; result: "the proposed approach even achieves competitive performances compared with
  state-of-the-arts that used annotated triggers" [abstract].
- HOW WE RELATE: the mirror-image claim in event extraction (coarse, sentence-level supervision is competitive with
  fine-grained supervision for a sentence-level decision). Their comparison is against published trigger-supervised
  systems with different architectures (Table 4), i.e. NOT controlled for encoder/data; I did not establish how the
  trigger-based rows were scored at sentence level. A reviewer could cite it; our answer is the controlled design.
  Optional.

---------------------------------------------------------------------------------------------------------------

## CONCLUSIONS FOR THE PAPER

### What is new in our claim (after this search)
1. The operator (sentence = max over its candidate pairs; standard MI "at least one" assumption) is NOT new:
   Dietterich 1997; Andrews 2002 ("the set-level classifier is by design induced by a pattern-level classifier");
   in NLP, Riedel 2010 / Hoffmann 2011 (deterministic OR) / Zeng 2015 (max) / Verga 2018 and Jia 2019 (logsumexp).
2. The direction is transposed relative to DS-MIL: there the bag is the sentences of one pair and the bag decision is
   pair-level; ours is the pairs of one sentence and the bag decision is sentence-level. I found no work that reads a
   "does this sentence express any relation/interaction" decision off a pair-level verifier by max.
3. The controlled comparison is what is new: same teacher, same data, same encoder; instance(pair)-supervised
   classifier + max vs a classifier trained on the bag (sentence) label, evaluated on the bag decision. The closest
   published comparisons are (a) MIL instance-level vs embedding-level (Wang 2016/2018; Ilse 2018), which found the
   bag-level route preferable but trained from bag labels only, and explain that by unknown instance labels; and (b)
   SummaC, which compares input granularities of one fixed NLI model at inference time (finer + max/mean better).
   Neither trains both granularities on the same supervision source.
4. Sentence-level detection as a first stage exists (Chowdhury & Lavelli 2013; RERE 2021; PRGC 2021), always
   followed by a pair-level decision; nobody compared reading the sentence decision off the pair stage.

### Prior work we must cite
- MIL / at-least-one: riedel2010modeling (already cited), hoffmann2011knowledge, dietterich1997solving (or
  andrews2002support); optionally zeng2015distant, lin2016neural, surdeanu2012multi, bunescu2007learning.
- Fine-to-coarse aggregation: laban2022summac (closest analogue), verga2018simultaneously (biomedical, smooth max);
  optionally jia2019document, min2023factscore, kamoi2023wice.
- The opposite expectation from MIL: ilse2018attention (and/or wang2018revisiting).
- Sentence-level detect-then-extract: chowdhury2013exploiting (biomedical), xie2021revisiting (argues for
  sentence-first), zheng2021prgc.
- Sentence-positive != pair-positive: pyysalo2008comparative (PPI; co-occurrence precision = I/EP; LLL 0.50 with
  every sentence positive), tikk2013detailed (positive-pair ratio falls with entities per sentence),
  rosenman2020exposing (already cited; 57% of CRE sentences have conflicting pair labels; TACRED rarely annotates
  more than one pair per sentence), gao2021manual / riedel2010modeling (co-mention != assertion: 32% / 31% on NYT).

### Claims a reviewer would call known, and the citation that makes them known
- "A bag is positive iff at least one instance is positive; aggregate instance scores with max/OR" -- Dietterich
  1997; Andrews 2002; Hoffmann 2011; Zeng 2015.
- "Max is a crude aggregator; smooth max / attention works better" -- Lin 2016; Verga 2018; Jia 2019 (max costs 3.8
  AUC points vs logsumexp in their setting).
- "Deciding at finer granularity and aggregating beats deciding at the coarse granularity" -- SummaC (NLI,
  56.4% -> 73.5% as granularity is made finer, same model).
- "Co-occurrence of two entities in a sentence does not mean the sentence asserts their relation" -- Riedel 2010;
  Gao 2021; Zhu 2020; Takamatsu 2012; Pyysalo 2008.
- "Most candidate pairs in multi-entity sentences are negative; the problem grows quadratically with entity count" --
  Pyysalo 2008; Tikk 2013; Xie 2021 (RERE, O(m^2) vs O(m)).
- "Sentence-level relation detection can be used as a filter before pair classification" -- Chowdhury & Lavelli
  2013; Xie 2021; Zheng 2021.

### Claims we must not make, and what refutes each
- "We introduce max-over-pairs aggregation" / "first to derive a sentence decision from pair decisions" -- refuted by
  the MIL lineage above (Dietterich 1997; Andrews 2002; Hoffmann 2011) and, for sentence-level filters built from
  pair labels, by Chowdhury & Lavelli 2013 (their sentence label is "contains at least one DDI").
- "Instance-level MIL is known to be better for bag classification" -- the MIL literature says the opposite in the
  bag-label-only setting (Wang 2016/2018; Ilse 2018). State our result as holding WITH instance supervision.
- "Finer granularity always helps" -- Liu et al. 2019 (event detection without triggers) report sentence-level
  supervision competitive with trigger-supervised systems; Jia 2019 shows the aggregator choice matters. Keep the
  claim to "at least as good, in our controlled setting".
- "Distant supervision noise is ~30%" as a general fact -- figures vary with corpus and protocol: Riedel 31% NYT
  (300 sampled candidates, 3 relations) and a stated 13% for Wikipedia that does not match its own Table 1 average
  (16.7%); Gao 2021 32% N/A among DS-positive NYT10 sentences but 53% wrong at the entity-pair level; Zhu 2020 47.7%
  "No" (derived). Quote the specific figure with its corpus.
- "Our 0.724 is unusually low/high" -- comparable oracle figures from PPI corpora range 0.50-0.88 (Pyysalo 2008,
  derived bound); do not present 0.724 as surprising without that context.

---------------------------------------------------------------------------------------------------------------

## Could not verify / not done
- HRL (Takanobu et al. 2019, AAAI), cited by RERE as the only prior P3 (relation-first) work: not verified, not
  reported.
- Mintz et al. 2009 "more than 30% noisy instances" (as attributed by ARNOR): not checked in Mintz's text (Mintz is
  already cited as mintz2009distant); do not use via ARNOR.
- Amores 2013 (MIL taxonomy: instance-space / bag-space / embedded-space), referenced by Wang et al.: not verified.
- Tikk 2013 Figure 7 (class distribution vs number of proteins per sentence): figure not read, so no per-count
  numbers are reported.
- No work found (after targeted searches of ACL Anthology papers above, Semantic Scholar and web search) that
  compares a pair-classifier-with-max against a sentence-trained classifier for a sentence-level relation/interaction
  decision under the same supervision. Absence of evidence, not proof: search was limited to ~2 hours.
- Dietterich 1997 full text was read from an unofficial mirror (sci2s.ugr.es); bibliographic data from Crossref.
- Page numbers are given only where the PDF page header/footer placement made them unambiguous (Riedel, Hoffmann,
  Dietterich); elsewhere sections are given.
