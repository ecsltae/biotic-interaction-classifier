# Novelty attack: "ask about the pair, even if you want the sentence" (2026-10-05)

Role: skeptical reviewer-simulator. Goal: refute the controlled-comparison novelty claim by finding prior work.
Status: COMPLETE (first pass 02:43; later additions, if any, are marked "addendum").

Claim under attack: for co-occurrence-retrieved passages, a sentence-level decision read off a pair-conditioned
verifier (max over candidate pairs) is at least as good as a classifier trained on the sentence question directly,
same LLM teacher labels, same passages, same encoder (BiomedBERT); only the pair verifier also yields which pair.

Tags: [abstract] = read in the abstract; [full text, X] = read in the full text at location X.

## A. Controlled comparisons: fine-grained supervision used for the coarse decision (NLP)

### rei2019jointly
- Title: Jointly Learning to Label Sentences and Tokens
- Venue/year: Proceedings of the AAAI Conference on Artificial Intelligence 33 (AAAI 2019); DOI 10.1609/aaai.v33i01.33016916; arXiv 1811.05949
- Verified at: Semantic Scholar search/match (DOI, venue); full text read at https://arxiv.org/pdf/1811.05949
- What it shows: same data, same BiLSTM encoder; a sentence classifier trained on sentence labels only (BiLSTM-ATTN) vs the same architecture also given token-level labels ("+token": token predictions are the attention weights that compose the sentence). Sentence-level F1 rises on all four tasks: "BiLSTM-ATTN 83.87 85.21 89.77 93.96" vs "+token 85.90 85.41 91.97 95.82" (CoNLL-2010 hedge, FCE grammatical errors, SST-neg, SST-pos) [full text, Table 3]. "Integrating the token-labeling objective with the attention gives the biggest improvements overall ... With over 25% error reduction in F1, the token-level objective is particularly beneficial for the sentiment analysis datasets" [full text, Sentence Classification Experiments]. Its attention-range objective explicitly ties "the maximum unnormalized attention weight in a sentence to be equal to the gold label" (a max link token -> sentence) [full text, eq. Lattn].
- Pre-emption: PARTIAL. It already shows, with the same data and encoder, that finer-grained (token) labels improve the coarse (sentence) decision for "does the sentence contain X?" tasks. Differences: joint training (sentence + token labels together), attention composition rather than a pure max over independently scored instances, human labels, no entity pairs, no LLM teacher.
- Forces change to: any wording implying that "fine-grained supervision helps the coarse decision" is a new observation. Our novelty must be limited to the pair-conditioned (entity-pair) version with teacher labels and to the fact that the pair verifier is trained WITHOUT sentence labels.

### pislar2020seeing
- Title: Seeing Both the Forest and the Trees: Multi-head Attention for Joint Classification on Different Compositional Levels
- Venue/year: COLING 2020, pages 3761-3775; ACL Anthology 2020.coling-main.335; DOI 10.18653/v1/2020.coling-main.335
- Verified at: Crossref query (DOI, pages); full text read at https://arxiv.org/pdf/2011.00470
- What it shows: follow-up to rei2019jointly with an explicit same-input control. On CoNLL03 and FCE the sentence labels are existential ("derived automatically from the existing token-level annotation"). Sentence classification: BiLSTM-sent (sentence labels only) vs MHAL-joint (sentence + token labels): SST F1* 73.18 -> 77.30, CoNLL03 98.22 -> 98.50, FCE 84.67 -> 85.13 [full text, Table 2]. "While additional annotation is required to train the multi-task models, the same input sentences are used in all cases, indicating that the benefits are coming directly from the model solving the task on multiple levels, as opposed to just from seeing more data examples ... By teaching the model where to focus at the token level, the architecture is able to make better decisions on the sentence-level classification task" [full text, Sec. 3.4].
- Pre-emption: PARTIAL; same status as rei2019jointly (joint model, token not pair, human labels). The gains on existential sentence labels are small (CoNLL03 +0.28, FCE +0.46 F1*), larger on SST.
- Forces change to: same as rei2019jointly.

### dasanmartino2019fine
- Title: Fine-Grained Analysis of Propaganda in News Articles
- Venue/year: EMNLP-IJCNLP 2019, pages 5636-5646; ACL Anthology D19-1565; DOI 10.18653/v1/D19-1565
- Verified at: https://aclanthology.org/D19-1565.bib ; full text read at https://aclanthology.org/D19-1565.pdf
- What it shows: SLC task "asks to predict whether a sentence contains at least one propaganda technique" (the 'any' sentence question), FLC = fragment spans + technique [full text, Sec. 5]. Same corpus, same BERT: sentence-only BERT F1 57.74 (P 63.20, R 53.16); with fragment-level supervision (BERT-Joint) 58.91; multi-granularity network 60.71 / 60.98 [full text, Table 7]. "Compared to the BERT baseline, it increases the recall by 8.42%, resulting in a 3.24% increase of the F1 score. In this case, the result of token-level classification is used as additional information for the sentence-level task, and it helps to find more positive samples." [full text, Sec. 6.3]. Conclusion: "Going at this fine-grained level can yield more reliable systems and it also makes it possible to explain to the user why an article was judged as propagandistic" [full text, Sec. 8].
- Pre-emption: PARTIAL, and the closest NLP analogue found. Same data/encoder, fine-grained supervision improves the "contains at least one X" sentence decision AND yields the fine-grained localisation (the "only the fine model also says which" half of our claim). Differences: multi-task (sentence labels still used), no fine-only + max arm, spans not entity pairs, human labels, no teacher.
- Forces change to: "only the pair verifier also yields which pair interacts" cannot be framed as a new argument for fine-grained modelling (Da San Martino et al. make the explainability argument); and "fine-grained supervision is at least as good for the coarse decision" must be cited to this and rei2019jointly.

### farkas2010conll (addendum)
- Title: The CoNLL-2010 Shared Task: Learning to Detect Hedges and their Scope in Natural Language Text
- Venue/year: CoNLL-2010 Shared Task, pages 1-12; ACL Anthology W10-3001 (no DOI in the Anthology record)
- Verified at: https://aclanthology.org/W10-3001.bib ; full text read at https://aclanthology.org/W10-3001.pdf
- What it shows: a BIOMEDICAL existential sentence task exactly of our form: "sentences containing at least one cue were considered as uncertain, while sentences with no cues were considered as factual" [full text, task definition section]. Two families of supervised systems: "six systems which handled the task as a classical sentence classification problem and employed essentially a bag-of-words feature representation"; the others "sought to classify every token if it was a part of a cue phrase, then a sentence was predicted as uncertain if it contained at least one recognized cue phrase" [full text, Sec. 6.2]. Outcome: "the top ranked systems of Task1B [biological] followed a sequence labeling approach, while the best systems on Task1W [Wikipedia] applied a bag-of-words sentence classification. This may be due to the fact that biological sentences have relatively simple patterns" [full text, Sec. 6.2]; best Task1B sentence-level F 86.4 (Tang; approach SL = CRF sequence labelling) [full text, Tables 3 and 6], best Task1W 60.2 (Georgescul; approach BoW, SVM) [full text, Tables 1 and 6].
- Pre-emption: PARTIAL, and the oldest biomedical precedent for "decide the sentence question by an 'any' over fine-grained decisions" vs "classify the sentence directly", both fully supervised on the same data. Not controlled (different teams, features, learners), cue tokens not entity pairs, human labels; and the winner depends on the domain.
- Forces change to: we cannot say the comparison "fine + any vs direct sentence classifier" has not been made in biomedical text; we can say it has not been made under a same-encoder, same-teacher control, with entity-pair conditioning.

### schmaltz2016sentence (addendum)
- Title: Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction
- Venue/year: Proceedings of the 11th Workshop on Innovative Use of NLP for Building Educational Applications (BEA), 2016, pages 242-251; ACL Anthology W16-0528; DOI 10.18653/v1/W16-0528
- Verified at: Crossref query (DOI, pages) and BibTeX via https://doi.org/10.18653/v1/w16-0528 ; full text read at https://aclanthology.org/W16-0528.pdf
- What it shows: AESW 2016 asks an existential sentence question ("whether a given source sentence has one or more grammatical errors") [full text, Sec. 1]. Same training data; a model trained on the fine-grained edit annotations (seq2seq predicting the corrected sentence with <ins>/<del> tags; positive if the target differs from the source) vs CNN sentence classifiers trained on the binary label: dev F1 CNN-NONSTATIC 0.6343 vs CHAR+SAMPLE 0.6579; "The encoder-decoder models improve over the CNN classifiers" [full text, Sec. 6, Table 1]; the edit model also yields "intra-sentence error identification and the generation of possible corrections" [full text, Sec. 1].
- Pre-emption: PARTIAL; another same-data precedent where the fine-grained-supervised model, read existentially, beats the direct sentence classifier and also localises. Not same encoder (encoder-decoder vs CNN), human labels, no pairs.
- Forces change to: same as farkas2010conll.

## B. Controlled comparisons outside NLP (instance/strong labels vs bag/weak labels for the bag decision)

### hershey2021benefit
- Title: The Benefit of Temporally-Strong Labels in Audio Event Classification
- Venue/year: ICASSP 2021 (IEEE), pages 366-370; DOI 10.1109/ICASSP39728.2021.9414579; arXiv 2105.07031
- Verified at: Semantic Scholar search/match (DOI, venue); full text read at https://arxiv.org/pdf/2105.07031
- What it shows: the cleanest controlled design found. Same 67k clips, same annotation pass, same ResNet-50; "Diffuse-67k - The clips from Strong-67k with each label expanded to the entire 10 sec" (coarse labels) vs "Strong-67k" (temporally fine labels) [full text, Sec. 4]. On the clip-level (weak) evaluation, d' 0.82 (Diffuse-67k) vs 0.96 (Strong-67k); fine-tuning Weak-1.8M "+ Diffuse" 1.21 vs "+ Strong" 1.28 [full text, Table 1]. "fine-tuning with a mix of weak- and strongly-labeled data can substantially improve classifier performance, even when evaluated using only the original weak labels" [abstract]. "The only difference between +Diffuse and +Strong is the temporal precision" [full text, Sec. 4.1].
- Pre-emption: PARTIAL (cross-domain). Establishes, with a same-data/same-model control, that finer-grained labels improve the coarse (clip "contains X") decision. Not text, not entity pairs, human labels, coarse labels derived from the fine ones (in our setup the sentence and pair labels are separate teacher queries).
- Forces change to: any wording that presents "fine-grained labels help the coarse decision under a same-data, same-model control" as new in general. Restrict novelty to text / entity-pair conditioning / LLM-teacher labels.

### li2021dual (DSMIL)
- Title: Dual-Stream Multiple Instance Learning Network for Whole Slide Image Classification With Self-Supervised Contrastive Learning
- Venue/year: CVPR 2021, pages 14313-14323 (Crossref); DOI 10.1109/CVPR46437.2021.01409
- Verified at: full text read at https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf ; Crossref https://api.crossref.org/works?query.bibliographic=Dual-stream+multiple+instance+learning... -> DOI 10.1109/cvpr46437.2021.01409
- What it shows: on Camelyon16 with the same SimCLR features, an "upper-bound fully-supervised model by making use of the pixel-level annotations, where a patch is labeled positive if it falls within a tumor region and the score of a WSI is then obtained by averaging the scores of all its patches" [full text, Sec. 4.1] beats every slide-label MIL model for the SLIDE decision: fully-supervised AUC 0.9362 vs max-pooling MIL 0.8641, ABMIL 0.8653, DSMIL 0.8944, DSMIL-LC 0.9165 [full text, Table 1].
- Pre-emption: PARTIAL (cross-domain), and important rhetorically: in the MIL literature an instance-supervised model aggregated to the bag is treated as the UPPER BOUND for bag classification. A reviewer can say our "pair >= sentence" result is the expected direction, because pair labels carry strictly more information than "any pair interacts".
- Forces change to: do not frame "at least as good" as surprising. Frame the contribution as measuring the size of the gain (and where it comes from: passages with >2 entities) when both label sets are produced by the same LLM teacher from separate questions, i.e. when the pair labels are NOT a refinement of the sentence labels and may disagree with them.

## C. Evidence AGAINST the claim, or against its second half ("only the pair verifier yields which pair")

### pavlopoulos2022from
- Title: From the Detection of Toxic Spans in Online Discussions to the Analysis of Toxic-to-Civil Transfer
- Venue/year: ACL 2022 (Volume 1: Long Papers), pages 3721-3734; ACL Anthology 2022.acl-long.259; DOI 10.18653/v1/2022.acl-long.259
- Verified at: https://aclanthology.org/2022.acl-long.259.bib ; full text read at https://aclanthology.org/2022.acl-long.259.pdf
- What it shows: a post-level (coarse) classifier plus attention-based rationale extraction is a credible localiser. "sequence labeling models perform best. Moreover, methods that add generic rationale extraction mechanisms on top of classifiers trained to predict if a post is toxic or not are also surprisingly promising" [abstract]. With 80k extra post-level examples, span F1 of BILSTM+ARE "improved from 57.7% to 58.8%, almost reaching the performance of BILSTM-SEQ" [full text, Sec. 6]; "this method can reach the performance of a BILSTM sequence labelling approach that was trained on the more costly toxic spans annotations" [full text, Conclusions].
- Pre-emption: does not pre-empt; it QUALIFIES the second half. Fine-grained supervision wins, but a coarse-trained model with attribution is "close". A reviewer will ask whether attribution from the sentence classifier (e.g. attention or gradient saliency over the two taxon mentions) could rank pairs.
- Forces change to: "only the pair verifier also yields which pair interacts" -> "only the pair verifier is trained and evaluated to say which pair interacts; a sentence classifier can at best localise through post-hoc attribution, which we do not test" (or test it).

### magge2021deepademiner
- Title: DeepADEMiner: a deep learning pharmacovigilance pipeline for extraction and normalization of adverse drug event mentions on Twitter
- Venue/year: Journal of the American Medical Informatics Association 28(10):2184-2192, 2021; DOI 10.1093/jamia/ocab114; PMID 34270701; PMC8449608
- Verified at: Europe PMC REST (core record + fullTextXML for PMC8449608); Crossref BibTeX via https://doi.org/10.1093/jamia/ocab114
- What it shows: the closest BIOMEDICAL same-data comparison found, and it goes AGAINST the fine-grained model. Same corpus (HLP-ADE-v1, ADEs in about 7% of tweets); tweet-level decision "does the tweet contain an ADE?" made by (a) a RoBERTa tweet classifier or (b) the ADE span tagger (NER), positive if it finds any span. "Since the loss function of the classification tasks and NER tasks are defined differently, we naturally expect the NER to perform lower in the classification task ... the ADE classifier fine-tuned using the RoBERTa model (F1-score = 0.63) outperforms the NER (F1-score = 0.41) in identifying tweets containing ADEs by about 22 percentage points" [full text, Results, Experiment 2]. Motivation: "Due to the higher complexity of the NER task, NERs have a lower sensitivity to identifying ADEs in tweets compared to a classifier" and "we find it important to reevaluate if ADE classifiers continue to be an essential step" [full text, Introduction].
- Pre-emption: none, but it is the most likely reviewer counter-example ("fine-grained models aggregated to the coarse decision lose to a coarse classifier; that is why pipelines keep a sentence/tweet filter"). Not a controlled comparison: encoders differ (RoBERTa classifier vs Flair-based taggers), human labels, and the fine model is a span EXTRACTOR that must find the mention, whereas our pair verifier only classifies candidate pairs that the co-occurrence step already supplies.
- Forces change to: the claim must say "pair-CONDITIONED verifier over given candidate pairs", not "pair-level extraction model"; and should state explicitly that the opposite result is reported when the fine-grained model must also detect the spans (cite this).

### wadden2022multivers
- Title: MultiVerS: Improving scientific claim verification with weak supervision and full-document context
- Venue/year: Findings of NAACL 2022; ACL Anthology 2022.findings-naacl.6
- Verified at: https://aclanthology.org/2022.findings-naacl.6.bib ; full text read at https://aclanthology.org/2022.findings-naacl.6.pdf
- What it shows: biomedical claim verification; coarse (abstract-level) label decided from the full document (Multitask) vs from extracted sentence-level rationales (Pipeline, "extract-then-label") vs MT/PI (multitask training, pipeline inference). "the rationales may lack information required to make a prediction when taken out-of-context" [full text, Sec. 1]. Fully supervised abstract-level F1 (HealthVer / COVIDFact / SciFact): Multitask 77.6 / 77.3 / 72.5, Pipeline 78.4 / 77.6 / 70.9, MT/PI 70.6 / 73.3 / 60.3 [full text, Table 3c]; "The Multitask approach performs 14.0% worse on context-dependent instances, while the Pipeline approach performs 22.8% worse ... MT/PI performs 66.4% worse" [full text, Sec. 7.1].
- Pre-emption: none; QUALIFIES. Deciding the coarse label from fine units is only safe if each fine decision still sees the context. Our pair verifier encodes the whole passage with the pair marked, so it is closer to their Multitask model than to their Pipeline; the claim should say so.
- Forces change to: specify "pair-conditioned verifier over the FULL passage (pair marked in context)", not "pair-level classifier"; cite as the reason context must be kept.

### dekok2018review
- Title: Review-aggregated aspect-based sentiment analysis with ontology features
- Venue/year: Progress in Artificial Intelligence 7(4):295-306, 2018; DOI 10.1007/s13748-018-0163-7
- Verified at: Crossref https://api.crossref.org/works/10.1007/s13748-018-0163-7 ; abstract via Semantic Scholar; full text (CC BY) read at https://link.springer.com/content/pdf/10.1007/s13748-018-0163-7.pdf
- What it shows: the one same-data controlled comparison found where the COARSE model wins. SemEval-2016 restaurant reviews annotated at both review and sentence level; "we compare a pure review-level algorithm with aggregating the sentiment values of individual sentences ... the pure review-level algorithm outperforms the sentence aggregation method" [abstract]. Test F1 for review-level aspect polarity: review-level SVM 0.8020 (base) / 0.8119 (final) vs sentence-level SVM + summation 0.6824 (baseSA) / 0.7717 (ontSA) [full text, Tables 4-5]; "The final model has an accuracy that is approximately 4.0% points higher than the accuracy of the ontSA model" [full text, Sec. 5.1]. Aggregation is a SUM of predicted polarities (Eq. 1), not max/OR.
- Pre-emption: none; it is evidence AGAINST the general principle, but for a coarse label that is not a deterministic OR of the fine labels (review polarity is not "any sentence positive"). Useful to bound our claim: the max rule is exact only when the coarse question is existential.
- Forces change to: state that the claim is for existential coarse questions ("does the passage describe ANY interaction"), where OR is the correct link; do not generalise to graded or compositional coarse labels.

### rathore2022pare
- Title: PARE: A Simple and Strong Baseline for Monolingual and Multilingual Distantly Supervised Relation Extraction
- Venue/year: ACL 2022 (Volume 2: Short Papers); ACL Anthology 2022.acl-short.38; DOI 10.18653/v1/2022.acl-short.38
- Verified at: Semantic Scholar search/match (ACL ID, DOI); abstract read there; bib from https://aclanthology.org/2022.acl-short.38.bib
- What it shows: "Neural models for distantly supervised relation extraction (DS-RE) encode each sentence in an entity-pair bag separately. These are then aggregated for bag-level relation prediction ... all sentences of a bag are concatenated into a passage of sentences, and encoded jointly using BERT ... our simple baseline solution outperforms existing state-of-the-art DS-RE models" [abstract].
- Pre-emption: none. Weak evidence AGAINST per-instance encoding + aggregation for the bag decision (joint encoding of the whole bag wins), but with bag labels only (no instance supervision) and the instances are sentences, not pairs.
- Forces change to: nothing in the claim; a reviewer could cite it to argue the sentence encoder sees more context. Our sentence classifier already sees the whole passage, so the comparison is fair on this axis; say so.

## D. More cross-domain partial support (instance supervision helps the bag decision; bag and instance accuracy differ)

### li2018thoracic
- Title: Thoracic Disease Identification and Localization with Limited Supervision
- Venue/year: CVPR 2018, pages 8290-8299; DOI 10.1109/CVPR.2018.00865; arXiv 1711.06373
- Verified at: Crossref BibTeX via https://doi.org/10.1109/CVPR.2018.00865 ; full text read at https://arxiv.org/pdf/1711.06373
- What it shows: patch-level model whose image-level score comes from patch scores (MIL "at least one positive patch" for unannotated images); adding 704 bounding-box-annotated images (patch-level supervision) to image-label training: "Bounding box supervision improves classification performances ... for almost all the disease types, using 80% annotated images to train the model improves the prediction performance ... For some disease types, the absolute improvement is significant (> 5%)" [full text, Sec. 4.1].
- Pre-emption: PARTIAL (cross-domain, mixed supervision). Same message as hershey2021benefit: instance-level supervision helps the image-level decision.
- Forces change to: same as hershey2021benefit (do not claim the general principle).

### vanwinckelen2015instance
- Title: Instance-level accuracy versus bag-level accuracy in multi-instance learning
- Venue/year: Data Mining and Knowledge Discovery 30(2):313-341; Crossref gives year 2015 (online), key uses 2015; DOI 10.1007/s10618-015-0416-z
- Verified at: Crossref BibTeX via https://doi.org/10.1007/s10618-015-0416-z ; abstract via Semantic Scholar
- What it shows: "We show that there is a substantial difference between these two, and better performance on one does not necessarily imply better performance on the other ... always use the evaluation criterion most relevant for the task at hand" [abstract].
- Pre-emption: none; SUPPORTS our second half (a model good at the bag/sentence decision need not be good at the instance/pair decision, and vice versa) and supports reporting both levels.
- Forces change to: nothing; cite when arguing that sentence-level scores do not license pair-level conclusions.

## E. LLM side: asking finer questions and aggregating vs asking the general question (search item 3)

### hu2025decomposition
- Title: Decomposition Dilemmas: Does Claim Decomposition Boost or Burden Fact-Checking Performance?
- Venue/year: NAACL 2025 (Volume 1: Long Papers), pages 6313-6336; ACL Anthology 2025.naacl-long.320
- Verified at: https://aclanthology.org/2025.naacl-long.320.bib ; full text read at https://aclanthology.org/2025.naacl-long.320.pdf
- What it shows: the strongest evidence AGAINST "decompose, ask the finer question, aggregate" as a universal win. "Some studies have reported improvements from decompostition, while others have observed performance declines" [abstract]. "decomposition generally benefits weaker verifiers, while it tends to negatively affect stronger verification systems" [full text, Sec. 4.2]; "Decomposition can improve the handling of complex inputs; however, while increasing the number of sub-claims may initially enhance performance, the additional noise introduced will gradually offset these gains" [full text, Sec. 1 takeaways]; "the effectiveness of decomposition is highly dependent on the input granularity" [full text, Sec. 4.1]. Also reports from prior work that claim-based methods were "advantageous for ChatGPT but detrimental for GPT-4" (FELM) and that MiniCheck saw "no improvement with decomposition" [full text, Sec. 4.2; secondary report, not checked in those papers].
- Pre-emption: none (it is the contrary evidence). It QUALIFIES the claim: gains from finer-grained questions depend on input complexity and on the noise of the decomposition step. Consistent with our finding that the gain is confined to passages naming more than two entities. Key difference to state: our decomposition is deterministic (candidate pairs from the NER/co-occurrence step), so the LLM-decomposition error term analysed by Hu et al. is absent.
- Forces change to: never write "pair-level questions are better" unconditionally; write "no worse overall, better on passages with more than two candidate entities", and cite this as the reason the gain should be conditional.

### tang2024minicheck (addendum)
- Title: MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents
- Venue/year: EMNLP 2024, pages 8818-8847; ACL Anthology 2024.emnlp-main.499; DOI 10.18653/v1/2024.emnlp-main.499
- Verified at: Crossref query (DOI, pages); https://aclanthology.org/2024.emnlp-main.499.bib ; full text read at https://aclanthology.org/2024.emnlp-main.499.pdf
- What it shows: small checkers distilled from LLM-generated synthetic data, evaluated on sentence-level claims. Inference-time decomposition into atomic facts (claim supported iff all atomic facts supported) vs checking the whole claim: GPT-4 75.6 (up 0.3), MiniCheck-FT5 73.3 (down 1.4), MiniCheck-RBTA 73.2 (up 0.5), AlignScore 71.5 (up 1.1), SummaC-CV 58.8 (down 3.3) [full text, Table 5]. "Overall, there is no clear indication that decomposing claims into atomic facts can consistently improve models' performance ... it should not be used until it provides a clear accuracy benefit" [full text, Sec. 7.1]; "Surprisingly, we find that claim decomposition is not needed in our settings, contradicting prior work" [full text, Sec. 7].
- Pre-emption: none; contrary/neutral evidence for search items 2-3. Note the difference: MiniCheck decomposes only at inference (the student is trained on claim-level labels); we train the student on the finer (pair) question. Also the aggregation is "all" (universal), ours "any" (existential).
- Forces change to: supports wording "no worse overall" rather than "better"; cite with hu2025decomposition.

### wanner2024closer
- Title: A Closer Look at Claim Decomposition
- Venue/year: *SEM 2024; ACL Anthology 2024.starsem-1.13
- Verified at: https://aclanthology.org/2024.starsem-1.13.bib ; abstract via Semantic Scholar search/match
- What it shows: FActScore-style decompose-then-verify scores are "sensitive to the decomposition method used ... error can also come from the metric's decomposition step" [abstract].
- Pre-emption: none; supports the point that the decomposition step must be controlled (ours is fixed by the candidate generator). Optional citation.

### xie2023empirical
- Title: Empirical Study of Zero-Shot NER with ChatGPT
- Venue/year: EMNLP 2023 (main), pages 7935-7956; ACL Anthology 2023.emnlp-main.493; DOI 10.18653/v1/2023.emnlp-main.493
- Verified at: https://aclanthology.org/2023.emnlp-main.493.bib ; full text read at https://aclanthology.org/2023.emnlp-main.493.pdf
- What it shows: asking the LLM one narrower question at a time beats asking the whole task at once. "Recognizing entities of all labels at one time may be too challenging for ChatGPT ... Each time, ChatGPT is asked to recognize entities of a single label" [full text, Sec. 3.1]; F1 Vanilla vs Decomposed-QA e.g. ACE05 28.12 -> 34.37, OntoNotes 4 33.74 -> 37.45 [full text, Table 1]; "decomposing by labels makes the NER task much more manageable for ChatGPT" [full text, Sec. 4.2.1].
- Pre-emption: PARTIAL for search item 3 only (decomposition by label type, not by entity pair; zero-shot LLM, no student; output is NER, not a coarse yes/no).
- Forces change to: nothing in the student claim; cite together with the already-known QA4RE / SumAsk / EP-RSR when saying narrower LLM questions are known to help LLM extraction.

## F. Rationale / evidence supervision for the coarse decision (older, partial)

### zhang2016rationale
- Title: Rationale-Augmented Convolutional Neural Networks for Text Classification
- Venue/year: EMNLP 2016, pages 795-804; ACL Anthology D16-1076; DOI 10.18653/v1/D16-1076
- Verified at: Semantic Scholar search/match (ACL ID, DOI, PMC5300751); Crossref BibTeX via https://doi.org/10.18653/v1/D16-1076 ; full text read at https://aclanthology.org/D16-1076.pdf
- What it shows: same documents, same sentence-level CNN backbone: Doc-CNN (document labels only) vs RA-CNN (adds sentence-level rationale labels; sentence scores weight the document representation). Accuracy, risk-of-bias in clinical trial reports: RSG 72.60 -> 77.42, AC 72.92 -> 76.14, BPP 74.24 -> 76.47, BOA 63.64 -> 69.67; movie reviews 87.14 -> 90.43 [full text, Tables 2-3]. "a sentence-level convolutional model that estimates the probability that a given sentence is a rationale ... consistently outperforms strong baselines. Moreover, our model naturally provides explanations for its predictions" [abstract].
- Pre-emption: PARTIAL. Same-data, same-encoder evidence (biomedical documents) that finer-grained (sentence) supervision helps the coarse (document) decision and also localises the evidence. Differences: document + sentence labels used jointly; aggregation is weighted pooling of representations, not max over independent instance decisions; human labels.
- Forces change to: same as rei2019jointly.

### zaidan2007using
- Title: Using "Annotator Rationales" to Improve Machine Learning for Text Categorization
- Venue/year: NAACL-HLT 2007, pages 260-267; ACL Anthology N07-1033 (no DOI)
- Verified at: https://aclanthology.org/N07-1033.bib ; full text (abstract) read at https://aclanthology.org/N07-1033.pdf
- What it shows: annotators highlight evidence for the document label; "a learning method that exploits the rationales during training to boost performance significantly on a sample task, namely sentiment classification of movie reviews ... providing rationales is a more fruitful use of an annotator's time than annotating more examples" [abstract].
- Pre-emption: PARTIAL, historical origin of "ask the annotator for the finer-grained evidence even if you want the coarse label". Our analogue: ask the teacher about the pair even if you want the sentence. Cite as the ancestor; no change beyond A.

## G. Searches with no pre-empting hit (decisions noted; unattended run)

- Search item 2 (LLM-teacher distillation where the granularity of the question asked to the teacher, sentence vs entity pair, is varied and the student is evaluated at the coarse level): NO work found. Queries (WebSearch, Semantic Scholar; S2 keyword search was rate-limited after the first calls): "LLM teacher distillation entity pair level labels versus sentence level labels student classifier"; "LLM-generated labels at different granularity sentence-level vs span-level distillation BERT student"; "LLM annotator relation extraction silver labels entity pair prompt versus sentence prompt annotation granularity"; "LLM pseudo-labels question granularity aspect-level vs sentence-level annotation distillation". Closest non-hits: per-label dichotomic prompting + LLM-to-SLM distillation ("Divide, Cache, Conquer", arXiv 2511.03830, abstract read; per LABEL, not per entity pair; not peer-reviewed; not reported as a work); entity-matching distillation where the teacher labels candidate PAIRS (arXiv 2606.28823, search snippet only; no sentence-level arm; not verified further).
- Search item 1, biomedical RE used for sentence/abstract triage beyond BioCreative III Team 65 and BioCreative VI Team 433: no controlled same-encoder comparison found; the only same-corpus biomedical comparison is magge2021deepademiner (goes against, span extractor).
- Search item 3 beyond the known QA4RE / SumAsk / EP-RSR / Jimenez Gutierrez / Rehana / Mraz: only xie2023empirical (per-label decomposition) and the decomposition-in-fact-checking literature (hu2025decomposition, wanner2024closer), which is mixed-to-negative.
- Not pursued (time): HateXplain (rationales for hate speech; Mathew et al., AAAI 2021) - not verified, not reported; ERASER pipelines; Evidence Inference 2.0 (2020.bionlp-1.13, bib fetched, not read beyond search snippet - not reported); MILNET / Kotzias-style coarse-to-fine sentiment (reverse direction).
- Late checks (02:46, no hit): BioCreative II.5 ACT vs IPT (overview exists; no same-encoder comparison of pair output vs article classifier found in search snippets; not read, not reported); "relation-level supervision improves sentence-level relation existence detection" (no paper found).
- One search result pointed to a public code repository that appears to belong to the paper under review; ignored per the anonymity rule and not used.

## H. Synthesis for the paper

### Verdict: SURVIVES WITH REWORDING (not refuted; the general principle is pre-empted, the specific controlled measurement is not)

What is already known, and the citation that makes it known:
1. Finer-grained supervision improves the coarse "does it contain any X?" decision on the same data with the same encoder, when both levels are trained jointly: rei2019jointly (Table 3), pislar2020seeing (Table 2, existential sentence labels on CoNLL03/FCE), dasanmartino2019fine (SLC = "at least one propaganda technique", BERT 57.74 -> 60.98 F1), zhang2016rationale (biomedical risk-of-bias documents), zaidan2007using (origin of the idea of asking the annotator for finer evidence).
2. In MIL, an instance-supervised model aggregated to the bag is the accepted upper bound for the bag decision: li2021dual (Camelyon16 AUC 0.9362 vs 0.8641-0.9165 for slide-label MIL); hershey2021benefit (same clips, strong vs diffuse labels, clip-level d' 0.82 -> 0.96); li2018thoracic. A reviewer can therefore call "pair >= sentence" the expected direction, because pair answers contain the sentence answer.
2b. The comparison "fine-grained detector + 'at least one' vs direct sentence classifier, both supervised on the same data" was already run informally in shared tasks: CoNLL-2010 hedge detection (farkas2010conll: cue-tagging systems won on biomedical text, bag-of-words sentence classifiers won on Wikipedia) and AESW 2016 (schmaltz2016sentence: edit-supervised encoder-decoder beat CNN sentence classifiers). Neither controls the encoder.
3. Fine-grained output for free from the fine model (localisation/explanation) is an argument already made by dasanmartino2019fine and zhang2016rationale; and a coarse model plus attribution can come close for localisation (pavlopoulos2022from).

What contradicts or bounds the claim (must be cited and the claim scoped around it):
4. A fine-grained EXTRACTOR aggregated to the coarse decision can lose badly: magge2021deepademiner (tweet-level ADE: RoBERTa classifier F1 0.63 vs span NER 0.41, same corpus). Our model differs: it classifies candidate pairs supplied by co-occurrence; it does not have to find them.
5. Non-existential aggregation loses: dekok2018review (review polarity: review-level SVM beats summing sentence predictions by about 4 points). The max rule is exact only for the existential question.
6. Out-of-context fine units lose on context-dependent cases: wadden2022multivers (Pipeline 22.8% worse, MT/PI 66.4% worse on context-dependent instances). Our verifier keeps the full passage with the pair marked.
7. Asking finer questions and aggregating is not uniformly better for LLM verifiers; gains depend on input complexity and verifier strength: hu2025decomposition, tang2024minicheck (no consistent gain from atomic-fact decomposition for distilled checkers), wanner2024closer. Consistent with our gain being confined to passages naming more than two entities.
8. Weak: joint encoding of the whole bag beats per-instance encoding + aggregation in DS-RE (rathore2022pare), with bag labels only.

What remains new (as far as this search can tell):
(a) the controlled variable is the QUESTION put to the LLM teacher (pair vs sentence) when distilling a literature-mining filter, with the same teacher, passages and encoder - no prior work found (search item 2);
(b) the pair student is trained WITHOUT any sentence labels and the sentence decision is a pure max over candidate pairs (all NLP pre-emptions above are joint/multi-task models that still use coarse labels);
(c) the conditioning unit is an entity PAIR marked in the full passage, in relation-extraction-style literature mining, with the gain located on passages with more than two candidate entities;
(d) the pair labels are separate teacher answers, not a refinement of the sentence labels, so the MIL "upper bound" argument (item 2 above) does not apply automatically.

### Sentences of the novelty statement that must change
- "at least as good as a classifier trained to answer the sentence question directly" -> keep, but add "for an existential question, when the verifier sees the whole passage with the pair marked and the candidate pairs are given" (because of magge2021deepademiner, dekok2018review, wadden2022multivers).
- Any phrasing that presents "fine-grained supervision helps the coarse decision" as a finding -> attribute it to rei2019jointly, pislar2020seeing, dasanmartino2019fine, zhang2016rationale (NLP) and hershey2021benefit, li2021dual (MIL with instance labels).
- "only the pair verifier also yields which pair interacts" -> "only the pair verifier is trained, and can be evaluated, to say which pair interacts; post-hoc attribution from a sentence classifier can localise evidence (pavlopoulos2022from) but we do not test it" (or add the attribution baseline).
- Do not write "pair questions are better"; write "no worse overall and better when the passage names more than two taxa" (hu2025decomposition).

### Recommended rewording (drop-in)
"Learning from finer-grained labels is known to help coarse decisions when both levels are trained jointly (Zaidan et al., 2007; Zhang et al., 2016; Rei and Søgaard, 2019; Da San Martino et al., 2019), shared tasks have pitted cue detectors read as "at least one cue" against direct sentence classifiers (Farkas et al., 2010), and instance-supervised models are the usual upper bound in multiple-instance learning (Hershey et al., 2021; Li et al., 2021); conversely, a fine-grained extractor read at the coarse level can lose to a direct classifier (Magge et al., 2021), and decomposing a question for a verifier does not always help (Hu et al., 2025; Tang et al., 2024). What has not been measured, to our knowledge, is which question to put to the LLM teacher when distilling a literature-mining filter. With the same co-occurrence passages, the same teacher and the same encoder, a student trained only on pair-conditioned answers, whose passage decision is the maximum over its given candidate pairs (each marked in the full passage), is no worse at the sentence-level decision than a student trained on the sentence question, is better on passages naming more than two taxa, and is the only one of the two trained to output which pair interacts."

## I. BibTeX (recommended citations; normalised from the source given in each % source line)

Primary: dasanmartino2019fine, rei2019jointly, hershey2021benefit, li2021dual, magge2021deepademiner, wadden2022multivers, hu2025decomposition, pavlopoulos2022from, dekok2018review, zhang2016rationale. Optional: farkas2010conll and schmaltz2016sentence (addenda; recommended if the paper discusses shared-task precedents), pislar2020seeing, zaidan2007using, tang2024minicheck, xie2023empirical, li2018thoracic, vanwinckelen2015instance, rathore2022pare, wanner2024closer.

```bibtex
% source: https://aclanthology.org/D19-1565.bib
@inproceedings{dasanmartino2019fine,
  title = {Fine-Grained Analysis of Propaganda in News Articles},
  author = {Da San Martino, Giovanni and Yu, Seunghak and Barr{\'o}n-Cede{\~n}o, Alberto and Petrov, Rostislav and Nakov, Preslav},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages = {5636--5646},
  year = {2019},
  address = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/D19-1565}
}

% source: https://aclanthology.org/2020.coling-main.335.bib
@inproceedings{pislar2020seeing,
  title = {Seeing Both the Forest and the Trees: Multi-head Attention for Joint Classification on Different Compositional Levels},
  author = {Pislar, Miruna and Rei, Marek},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  pages = {3761--3775},
  year = {2020},
  address = {Barcelona, Spain (Online)},
  publisher = {International Committee on Computational Linguistics},
  doi = {10.18653/v1/2020.coling-main.335}
}

% source: https://aclanthology.org/2022.acl-long.259.bib
@inproceedings{pavlopoulos2022from,
  title = {From the Detection of Toxic Spans in Online Discussions to the Analysis of Toxic-to-Civil Transfer},
  author = {Pavlopoulos, John and Laugier, Leo and Xenos, Alexandros and Sorensen, Jeffrey and Androutsopoulos, Ion},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {3721--3734},
  year = {2022},
  address = {Dublin, Ireland},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2022.acl-long.259}
}

% source: https://aclanthology.org/2025.naacl-long.320.bib
@inproceedings{hu2025decomposition,
  title = {Decomposition Dilemmas: Does Claim Decomposition Boost or Burden Fact-Checking Performance?},
  author = {Hu, Qisheng and Long, Quanyu and Wang, Wenya},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages = {6313--6336},
  year = {2025},
  address = {Albuquerque, New Mexico},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2025.naacl-long.320}
}

% source: https://aclanthology.org/2022.findings-naacl.6.bib
@inproceedings{wadden2022multivers,
  title = {{M}ulti{V}er{S}: Improving scientific claim verification with weak supervision and full-document context},
  author = {Wadden, David and Lo, Kyle and Wang, Lucy Lu and Cohan, Arman and Beltagy, Iz and Hajishirzi, Hannaneh},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2022},
  pages = {61--76},
  year = {2022},
  address = {Seattle, United States},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2022.findings-naacl.6}
}

% source: https://aclanthology.org/2023.emnlp-main.493.bib
@inproceedings{xie2023empirical,
  title = {Empirical Study of Zero-Shot {NER} with {C}hat{GPT}},
  author = {Xie, Tingyu and Li, Qi and Zhang, Jian and Zhang, Yan and Liu, Zuozhu and Wang, Hongwei},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {7935--7956},
  year = {2023},
  address = {Singapore},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2023.emnlp-main.493}
}

% source: https://aclanthology.org/2022.acl-short.38.bib
@inproceedings{rathore2022pare,
  title = {{PARE}: A Simple and Strong Baseline for Monolingual and Multilingual Distantly Supervised Relation Extraction},
  author = {Rathore, Vipul and Badola, Kartikeya and Singla, Parag and Mausam},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages = {340--354},
  year = {2022},
  address = {Dublin, Ireland},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2022.acl-short.38}
}

% source: https://aclanthology.org/D16-1076.bib
@inproceedings{zhang2016rationale,
  title = {Rationale-Augmented Convolutional Neural Networks for Text Classification},
  author = {Zhang, Ye and Marshall, Iain and Wallace, Byron C.},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  pages = {795--804},
  year = {2016},
  address = {Austin, Texas},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/D16-1076}
}

% source: https://aclanthology.org/N07-1033.bib
@inproceedings{zaidan2007using,
  title = {Using ``Annotator Rationales'' to Improve Machine Learning for Text Categorization},
  author = {Zaidan, Omar and Eisner, Jason and Piatko, Christine},
  booktitle = {Human Language Technologies 2007: The Conference of the North {A}merican Chapter of the Association for Computational Linguistics; Proceedings of the Main Conference},
  pages = {260--267},
  year = {2007},
  address = {Rochester, New York},
  publisher = {Association for Computational Linguistics}
}

% source: https://aclanthology.org/2024.starsem-1.13.bib
@inproceedings{wanner2024closer,
  title = {A Closer Look at Claim Decomposition},
  author = {Wanner, Miriam and Ebner, Seth and Jiang, Zhengping and Dredze, Mark and Van Durme, Benjamin},
  booktitle = {Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (*SEM 2024)},
  pages = {153--175},
  year = {2024},
  address = {Mexico City, Mexico},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2024.starsem-1.13}
}

% source: https://doi.org/10.1609/aaai.v33i01.33016916 (Crossref BibTeX via content negotiation)
@article{rei2019jointly,
  title = {Jointly Learning to Label Sentences and Tokens},
  author = {Rei, Marek and Søgaard, Anders},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {33},
  number = {01},
  pages = {6916--6923},
  year = {2019},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  doi = {10.1609/aaai.v33i01.33016916}
}

% source: https://doi.org/10.1109/ICASSP39728.2021.9414579 (Crossref BibTeX via content negotiation)
@inproceedings{hershey2021benefit,
  title = {The Benefit of Temporally-Strong Labels in Audio Event Classification},
  author = {Hershey, Shawn and Ellis, Daniel P W and Fonseca, Eduardo and Jansen, Aren and Liu, Caroline and Channing Moore, R and Plakal, Manoj},
  booktitle = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages = {366--370},
  year = {2021},
  publisher = {IEEE},
  doi = {10.1109/icassp39728.2021.9414579}
}

% source: https://doi.org/10.1109/cvpr46437.2021.01409 (Crossref BibTeX via content negotiation)
@inproceedings{li2021dual,
  title = {Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning},
  author = {Li, Bin and Li, Yin and Eliceiri, Kevin W.},
  booktitle = {2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {14313--14323},
  year = {2021},
  publisher = {IEEE},
  doi = {10.1109/cvpr46437.2021.01409}
}

% source: https://doi.org/10.1109/CVPR.2018.00865 (Crossref BibTeX via content negotiation)
@inproceedings{li2018thoracic,
  title = {Thoracic Disease Identification and Localization with Limited Supervision},
  author = {Li, Zhe and Wang, Chong and Han, Mei and Xue, Yuan and Wei, Wei and Li, Li-Jia and Fei-Fei, Li},
  booktitle = {2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {8290--8299},
  year = {2018},
  publisher = {IEEE},
  doi = {10.1109/cvpr.2018.00865}
}

% source: https://doi.org/10.1007/s10618-015-0416-z (Crossref BibTeX via content negotiation)
@article{vanwinckelen2015instance,
  title = {Instance-level accuracy versus bag-level accuracy in multi-instance learning},
  author = {Vanwinckelen, Gitte and Tragante do O, Vinicius and Fierens, Daan and Blockeel, Hendrik},
  journal = {Data Mining and Knowledge Discovery},
  volume = {30},
  number = {2},
  pages = {313--341},
  year = {2015},
  publisher = {Springer Science and Business Media LLC},
  doi = {10.1007/s10618-015-0416-z}
}

% source: https://doi.org/10.1093/jamia/ocab114 (Crossref BibTeX via content negotiation)
@article{magge2021deepademiner,
  title = {DeepADEMiner: a deep learning pharmacovigilance pipeline for extraction and normalization of adverse drug event mentions on Twitter},
  author = {Magge, Arjun and Tutubalina, Elena and Miftahutdinov, Zulfat and Alimova, Ilseyar and Dirkson, Anne and Verberne, Suzan and Weissenbacher, Davy and Gonzalez-Hernandez, Graciela},
  journal = {Journal of the American Medical Informatics Association},
  volume = {28},
  number = {10},
  pages = {2184--2192},
  year = {2021},
  publisher = {Oxford University Press (OUP)},
  doi = {10.1093/jamia/ocab114}
}

% source: https://doi.org/10.1007/s13748-018-0163-7 (Crossref BibTeX via content negotiation)
@article{dekok2018review,
  title = {Review-aggregated aspect-based sentiment analysis with ontology features},
  author = {de Kok, Sophie and Punt, Linda and van den Puttelaar, Rosita and Ranta, Karoliina and Schouten, Kim and Frasincar, Flavius},
  journal = {Progress in Artificial Intelligence},
  volume = {7},
  number = {4},
  pages = {295--306},
  year = {2018},
  publisher = {Springer Science and Business Media LLC},
  doi = {10.1007/s13748-018-0163-7}
}

% source: https://aclanthology.org/W10-3001.bib
@inproceedings{farkas2010conll,
  title = {The {C}o{NLL}-2010 Shared Task: Learning to Detect Hedges and their Scope in Natural Language Text},
  author = {Farkas, Rich{\'a}rd and Vincze, Veronika and M{\'o}ra, Gy{\"o}rgy and Csirik, J{\'a}nos and Szarvas, Gy{\"o}rgy},
  booktitle = {Proceedings of the Fourteenth Conference on Computational Natural Language Learning {--} Shared Task},
  pages = {1--12},
  year = {2010},
  address = {Uppsala, Sweden},
  publisher = {Association for Computational Linguistics}
}

% source: https://doi.org/10.18653/v1/w16-0528 (Crossref BibTeX via content negotiation)
@inproceedings{schmaltz2016sentence,
  title = {Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction},
  author = {Schmaltz, Allen and Kim, Yoon and Rush, Alexander M. and Shieber, Stuart},
  booktitle = {Proceedings of the 11th Workshop on Innovative Use of NLP for Building Educational Applications},
  pages = {242--251},
  year = {2016},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/w16-0528}
}

% source: https://aclanthology.org/2024.emnlp-main.499.bib
@inproceedings{tang2024minicheck,
  title = {{M}ini{C}heck: Efficient Fact-Checking of {LLM}s on Grounding Documents},
  author = {Tang, Liyan and Laban, Philippe and Durrett, Greg},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages = {8818--8847},
  year = {2024},
  address = {Miami, Florida, USA},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2024.emnlp-main.499}
}
```
