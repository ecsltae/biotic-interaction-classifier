# verify3, adversarial check of novelty_extra_staging.bib (12 keys) and CLAIMS_TO_VERIFY_set3.md

Verifier 3, run 2026-10-05 (started 02:46 CEST). Scratch: /tmp/verify3_scratch (PDF -> `pdftotext -layout`).
Method: every BibTeX entry re-fetched independently (ACL Anthology .bib, Crossref BibTeX + Crossref REST JSON,
publisher landing pages); every quoted phrase / number located in the full text. Decisions made without
human input are marked "Decision:".

General note (applies to all ACL-style builds): acl_natbib.bst lower-cases titles of articles/inproceedings
except brace-protected material. Entries whose titles contain acronyms / proper nouns that are NOT braced are
flagged as FIX-BIB (cosmetic but visible in the PDF).

---

## 1. rei2019jointly, VERIFIED

**Bib.** Independent sources: Crossref REST https://api.crossref.org/works/10.1609/aaai.v33i01.33016916 and the AAAI
OJS landing page https://ojs.aaai.org/index.php/AAAI/article/view/4669 (citation_* meta): title "Jointly Learning to
Label Sentences and Tokens"; authors Marek Rei, Anders Søgaard (matches PDF title block); journal "Proceedings of the
AAAI Conference on Artificial Intelligence"; vol 33, issue 01; pp. 6916-6923; 2019 (online 2019-07-17); DOI matches.
All fields match. Optional: write `S{\o}gaard` instead of raw UTF-8 `Søgaard` if the build does not load
utf8 inputenc (modern LaTeX does by default, so not required).

**Claim.** Checked in the AAAI publisher PDF (https://ojs.aaai.org/index.php/AAAI/article/download/4669/4547,
p. 6919) and arXiv 1811.05949 (identical numbers). Table 3 ("Comparing sentence classification performance when
each of the auxiliary objective functions is added to BiLSTM-ATTN in isolation"), columns C'10 / FCE / SST-neg / SST-pos:
- `BiLSTM-ATTN 83.87 85.21 89.77 93.96`
- `+token 85.90 85.41 91.97 95.82`
All eight numbers match exactly. They are TEST-set F1 (Tables 1-2 report a separate "DEV F1" column; the Table 3
values equal the test "F1" column, e.g. CoNLL 2010 BiLSTM-ATTN DEV F1 89.88 vs F1 83.87). "Same data, same encoder"
is fair: +token = the same BiLSTM-ATTN model with the token-level objective switched on (Λtok), trained on the
same datasets. Caveats for honest paraphrase: (i) single runs, no variance or significance reported; (ii) the FCE
gain is only +0.20 F1 (85.21 -> 85.41), so "raises on all four" is literally true but one gain is marginal;
(iii) SST-neg / SST-pos are two binary subtasks of one corpus (SST), and the SST token labels are derived
automatically from SST phrase-level sentiment annotations, not hand-labelled tokens (p. 6918-6919, "phrases of up to
length 3 ... only the token 'good' will be labeled as positive").

## 2. dasanmartino2019fine, VERIFIED

**Bib.** https://aclanthology.org/D19-1565.bib: title, 5 authors (Da San Martino, Yu, Barrón-Cedeño, Petrov, Nakov;
order and spelling match), booktitle (EMNLP-IJCNLP 2019 full name), pp. 5636-5646, 2019, Hong Kong, China, ACL,
DOI 10.18653/v1/D19-1565, all fields match the staging entry. (Anthology adds editors/month/url; omission is fine.)

**Claim.** Full text https://aclanthology.org/D19-1565.pdf.
- Sec. 5 (p. 5642): "SLC (Sentence-level Classification), which asks to predict whether a sentence contains at least
  one propaganda technique", exact.
- Table 7 (p. 5644) "Sentence-level (SLC) results": `BERT 63.20 / 53.16 / 57.74`; `MGN ReLU 60.41 / 61.58 / 60.98`
  (P/R/F1). Exact. Sec. 6.3: "Compared to the BERT baseline, it increases the recall by 8.42%, resulting in a 3.24%
  increase of the F1 score." Sec. 6.1: TEST partition, "average of three experimental runs with different random seeds".
  Same corpus (train/dev/test 293/57/101 articles) and same pre-trained BERT for all rows, "same corpus and BERT" is fair.
- Sec. 8 (Conclusion, p. 5644): "Going at this fine-grained level can yield more reliable systems and it also makes it
  possible to explain to the user why an article was judged as propagandistic by an automatic system.", exact.
Caveats for a fair paraphrase: 60.98 is the better of two MGN gate variants (MGN Sigmoid = 60.71); the multi-granularity
model's extra signal is fragment-level span+technique (18 classes) supervision; BERT-Joint, which also uses the token
labels, gains only +1.17 (58.91). The "explain" sentence is about explaining an ARTICLE-level judgment.

## 3. zhang2016rationale, VERIFIED (paraphrase caveat: metric is accuracy, 9-fold CV)

**Bib.** https://aclanthology.org/D16-1076.bib: title, authors (Zhang, Ye; Marshall, Iain; Wallace, Byron C.), booktitle,
pp. 795-804, 2016, Austin, Texas, ACL, DOI 10.18653/v1/D16-1076, all match. PDF title block matches.

**Claim.** Full text https://aclanthology.org/D16-1076.pdf.
- Abstract: "annotators explicitly mark sentences (or snippets) that support their overall document categorization,
  i.e., they provide rationales ... Experiments on five classification datasets ... our approach consistently
  outperforms strong baselines. Moreover, our model naturally provides explanations for its predictions.", supports
  "improves ... and provides explanations".
- Table 3 (p. 802) "Accuracies on the movie review dataset": `Doc-CNN 87.14 (86.70, 87.60)`, `RA-CNN 90.43 (90.11, 91.00)`. Exact.
- Table 2 (RoB, 4 datasets): RA-CNN > Doc-CNN on all four (e.g. RSG 72.60 -> 77.42).
- Same data: yes, both trained on the same documents; Doc-CNN is the same hierarchical sentence->document CNN without
  rationale supervision (Sec. 4.2/6.1).
Caveats: the numbers are ACCURACY (not F1), means of 5 replications of 9-fold CV (movies) / 5-fold CV (RoB) (Sec. 7.1),
not a held-out test set. For movie reviews the rationales are sub-sentential snippets that the authors map to
sentences ("sentences containing the marked snippets", Sec. 6.1). If LITERATURE.md writes "F1" anywhere for these
numbers, that would be wrong.

## 4. farkas2010conll, VERIFIED

**Bib.** https://aclanthology.org/W10-3001.bib: title (with {C}o{NLL} protected), 5 authors in order Farkas, Vincze,
Móra, Csirik, Szarvas (LaTeX accents identical), booktitle "Proceedings of the Fourteenth Conference on Computational
Natural Language Learning {--} Shared Task", pp. 1-12, 2010, Uppsala, Sweden, ACL. Matches; no DOI exists for this
volume on the Anthology (none given, correct). PDF title block matches.

**Claim.** Full text https://aclanthology.org/W10-3001.pdf.
- Sec. 4.1 (Detection of Uncertain Sentences, p. 4): "The annotation of weasel/hedge cues was carried out on the phrase
  level, and sentences containing at least one cue were considered as uncertain, while sentences with no cues were
  considered as factual.", exact.
- Sec. 6.2 (Approaches, p. 8): "six systems ... handled the task as a classical sentence classification problem and
  employed essentially a bag-of-words feature representation ... The remaining teams focused on the cue phrases and
  sought to classify every token if it was a part of a cue phrase, then a sentence was predicted as uncertain if it
  contained at least one recognized cue phrase." and "the top ranked systems of Task1B followed a sequence labeling
  approach, while the best systems on Task1W applied a bag-of-words sentence classification."
- Tables 1, 3, 6 corroborate: Task1B top 3 = Tang 86.4, Zhou 85.8, Li 85.4 (all SL in Table 6); Task1W top = Georgescul
  60.2 (BoW).
Caveats: on Wikipedia the margin is small and not uniform, #2 Ji (58.7, cross-domain submission) and #4 Morante
(57.3) are token-classification (TC) systems; #3 Chen (57.4) is BoW. "Bag-of-words classifiers won" is accurate for
the winner and matches the authors' own wording; avoid saying all top Wikipedia systems were BoW.

## 5. li2021dual, VERIFIED (note on page numbers)

**Bib.** Crossref REST https://api.crossref.org/works/10.1109/cvpr46437.2021.01409: title "Dual-stream Multiple Instance
Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning"; authors Bin Li, Yin Li,
Kevin W. Eliceiri (order matches PDF title block); container "2021 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR)"; pp. 14313-14323; IEEE; June 2021 (Nashville, TN, USA). Staging entry matches Crossref exactly.
Note: the CVF open-access page
(https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.html)
gives booktitle "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)" and
pp. 14318-14328, a different pagination of the CVF open-access proceedings. Decision: keep the IEEE/Crossref pages
(14313-14323) because the entry carries the IEEE DOI; do not mix the CVF pages with the IEEE DOI. No fix required.

**Claim.** Full text: CVF open-access PDF (identical to accepted version per CVF). Sec. 4.1 "Results on Camelyon16",
Baselines paragraph (CVF p. 14322): "Moreover, we obtain an upper-bound fully-supervised model by making use of the
pixel-level annotations, where a patch is labeled positive if it falls within a tumor region and the score of a WSI is
then obtained by averaging the scores of all its patches in testing.", "upper bound" wording is the authors' own;
mean-of-patch-scores aggregation confirmed.
Table 1 ("Results on Camelyon16 dataset"), AUC column: `Max-pooling Single 0.8641`, `DSMIL Single 0.8944`,
`Fully-supervised Single 0.9362`. Exact. These are on the Camelyon16 test set (129 testing images; 271 training).
Caveats: 0.8944 is single-scale DSMIL; the authors' best model DSMIL-LC (multiscale) reaches AUC 0.9165 / acc 0.8992 and
the paper's headline is "accuracy gap smaller than 2%" vs fully-supervised (0.9147 acc). A fair sentence should say
"single-scale DSMIL" or mention DSMIL-LC so it does not understate the MIL result. All MIL models in Table 1 use the
same SimCLR features (caption).

## 6. hershey2021benefit, FIX-BIB (minor) + CLAIM-WRONG (minor: numbers and quote come from two different comparisons)

**Bib.** Crossref REST https://api.crossref.org/works/10.1109/ICASSP39728.2021.9414579: title "The Benefit of
Temporally-Strong Labels in Audio Event Classification"; container "ICASSP 2021 - 2021 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP)"; pp. 366-370; IEEE; 2021 (Toronto, ON, Canada, 6-11 June 2021).
Title, booktitle, pages, year, publisher, DOI match. Authors: 7, order matches the arXiv title block
("Shawn Hershey, Daniel P W Ellis, Eduardo Fonseca, Aren Jansen, Caroline Liu, R Channing Moore, Manoj Plakal").
Fix (minor): Crossref deposits the 6th author as given="R", family="Channing Moore", which the staging entry copied as
`Channing Moore, R`. The surname is Moore (first name R., middle name Channing; his e-mail is channingmoore@google.com
and the paper's own ref. [1] cites him as "C Moore"). Use `Moore, R. Channing` (renders identically as "R. Channing Moore"
in ACL full-name style but parses the surname correctly). Optional: `Ellis, Daniel P. W.`.
Decision: treated as a bib fix even though the rendered string is almost the same, because the current form is a
mis-parse of the surname.

**Claim.** Full text arXiv 2105.07031 (IEEE copy is paywalled; arXiv v1 is the ICASSP camera-ready layout, 5 pp.).
- Sec. 4: "Diffuse-67k - The clips from Strong-67k with each label expanded to the entire 10 sec." All models ResNet-50
  ("We trained all models with ResNet-50"). So "same clips, same model; same labels spread over the clip" is correct.
- Table 1 ("Evaluation results"), Weak eval d': `Diffuse-67k 0.82`, `Strong-67k 0.96`. Exact. Weak eval = "the original
  weak-label evaluation which rated 10 sec clips according to the average classifier score over the whole clip"
  (Sec. 3.2), so "clip-level d'" is right. (Strong eval d' for the same pair: 0.88 vs 1.05.)
- Quote, Sec. 4.1: "The only difference between +Diffuse and +Strong is the temporal precision, so this suggests that
  the 0.26 d' improvement from Weak-1.8M to +Strong can be split between 0.11 for the improved temporal precision and
  0.15 for all other factors.", exact, BUT it refers to the FINE-TUNING rows (+Diffuse / +Strong = Weak-1.8M model
  fine-tuned on mixtures; Weak eval d' 1.21 vs 1.28, Strong eval d' 1.28 vs 1.39), not to the Diffuse-67k vs Strong-67k
  rows that supply 0.82 / 0.96. The paper describes that pair as "weak versus strong labels (Diffuse-67k versus
  Strong-67k)" and says Strong-67k "even improves the Weak eval d' by 0.14, despite the label mismatch".
  Also note +Diffuse and +Strong were run with different mixing weights (µ = 0.7 vs 0.8), so "only difference" is the
  authors' simplification.
- The abstract's numbers (d' 1.13 -> 1.39) are Weak-1.8M -> +Strong on the Strong eval, not a diffuse-vs-strong comparison.
Fix the paraphrase: either (a) keep 0.82 vs 0.96 (Diffuse-67k vs Strong-67k, 10-s clip-level eval) and drop the
"+Diffuse/+Strong" quote, or (b) keep the quote and use the fine-tuned pair: clip-level (Weak eval) d' 1.21 (+Diffuse)
vs 1.28 (+Strong). Do not attach the quote to the 0.82/0.96 numbers.

## 7. magge2021deepademiner, FIX-BIB (brace protection) ; claim VERIFIED with one unsupported detail

**Bib.** Crossref REST https://api.crossref.org/works/10.1093/jamia/ocab114 and Europe PMC (PMC8449608, PMID 34270701,
https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:PMC8449608&resultType=core&format=json): title
"DeepADEMiner: a deep learning pharmacovigilance pipeline for extraction and normalization of adverse drug event mentions
on Twitter"; 8 authors Magge A, Tutubalina E, Miftahutdinov Z, Alimova I, Dirkson A, Verberne S, Weissenbacher D,
Gonzalez-Hernandez G (order matches); JAMIA 28(10):2184-2192, 2021; DOI matches. All fields match.
Fix: acl_natbib.bst lower-cases article titles, so the PDF would print "Deepademiner: ... on twitter". Use
`title = {{DeepADEMiner}: a deep learning pharmacovigilance pipeline for extraction and normalization of adverse drug
event mentions on {T}witter}`.

**Claim.** Full text: Europe PMC XML (PMC8449608), section RESULTS > "Experiment 2: Effect of data imbalance":
"Firstly, we compare the performance of the classifier and the NER for identifying tweets that contain ADEs. Since the
loss function of the classification tasks and NER tasks are defined differently, we naturally expect the NER to perform
lower in the classification task. ... Comparing classification performances, we found that the ADE classifier fine-tuned
using the RoBERTa model (F1-score = 0.63) outperforms the NER (F1-score = 0.41) in identifying tweets containing ADEs by
about 22 percentage points." Numbers, model (RoBERTa), task and direction confirmed; same corpus (HLP-ADE-v1, ~7% ADE
tweets); Table 3 gives the classifier's test P/R/F1 = 0.61/0.64/0.63 on HLP-ADE-v1.
Unsupported detail: the paper never states how the NER's tweet-level F1 (0.41) is computed; "read as 'any span found'"
is our inference (the natural one, but not in the text). LITERATURE.md must not present the decision rule as the
authors'. Further caveats: the classifier's 0.63 is after 5:1 undersampling and a threshold lowered from 0.5 to 0.15
(chosen on the development set, Fig. 4); no comparable tuning is reported for the NER, and the NER configuration used
for the 0.41 is not specified. The authors themselves say they "naturally expect the NER to perform lower".

## 8. pavlopoulos2022from, VERIFIED

**Bib.** https://aclanthology.org/2022.acl-long.259.bib: title, 5 authors (Pavlopoulos, Laugier, Xenos, Sorensen,
Androutsopoulos, order matches), booktitle ACL 2022 Vol. 1 Long Papers, pp. 3721-3734, 2022, Dublin, Ireland, ACL,
DOI 10.18653/v1/2022.acl-long.259. All match.

**Claim.** Full text https://aclanthology.org/2022.acl-long.259.pdf.
- Abstract: "methods that add generic rationale extraction mechanisms on top of classifiers trained to predict if a post
  is toxic or not are also surprisingly promising.", exact. Abstract also: "we show that sequence labeling models
  perform best."
- Sec. 6 + Table 3 (5-fold Monte Carlo CV, F1): BILSTM-SEQ 58.9, BILSTM+ARE 57.7 ("close in performance with BILSTM-SEQ,
  despite the fact that the latter is directly trained on toxic span annotations"); with 80k extra post-level-labelled
  posts BILSTM+ARE rises to 58.8, "almost reaching the performance of BILSTM-SEQ".
- Sec. 10 Conclusions: "we showed that this method can reach the performance of a BILSTM sequence labelling approach
  that was trained on the more costly toxic spans annotations."
Caveats: "approach a span-trained sequence labeller" holds for the BiLSTM tagger only; the best tagger SPAN-BERT-SEQ
(63.0) is clearly better, and BERT+ARE is much worse (49.1) even though its classifier has higher ROC AUC (96.1 vs 90.9).
The positive result is specific to the BiLSTM + attention rationale extractor.

## 9. hu2025decomposition, VERIFIED (location correction: the quote is in Sec. 4.2, not the abstract)

**Bib.** https://aclanthology.org/2025.naacl-long.320.bib: title, authors (Hu, Qisheng; Long, Quanyu; Wang, Wenya, 
matches PDF title block), booktitle NAACL 2025 Vol. 1 Long Papers (full name as staged), pp. 6313-6336, 2025,
Albuquerque, New Mexico, ACL, DOI 10.18653/v1/2025.naacl-long.320. All match (Anthology also lists ISBN
979-8-89176-189-6; optional).

**Claim.** Full text https://aclanthology.org/2025.naacl-long.320.pdf.
- Sec. 4.2 "Strong & Weak Verifier" (p. 6317): "Our analysis reveals that decomposition generally benefits weaker
  verifiers, while it tends to negatively affect stronger verification systems.", exact. The ABSTRACT does not contain
  this sentence (it speaks of "a trade-off between accuracy gains and the noise introduced through decomposition");
  Introduction bullet: "Decomposition improves performance on simpler sub-claims by reducing complexity, notably
  benefiting weaker verifiers." Cite Sec. 4.2 for the quote.
- Sec. 4.1 "Input Granularity": "These results indicate that the effectiveness of decomposition is highly dependent on
  the input granularity.", supports the granularity part.
Caveat: "weak/strong" is relative and defined by baseline accuracy (AlignScore < Minicheck < GPT-4o-mini); Minicheck is
hurt at claim level (WICE) but helped at response level (FELM), so the same verifier can sit on either side.

## 10. dekok2018review, VERIFIED (quote exact) but with a substantive caveat the paraphrase must not hide

**Bib.** Crossref REST https://api.crossref.org/works/10.1007/s13748-018-0163-7 and Springer landing page
https://link.springer.com/article/10.1007/s13748-018-0163-7 (citation_* meta): title "Review-aggregated aspect-based
sentiment analysis with ontology features"; 6 authors de Kok, Punt, van den Puttelaar, Ranta, Schouten, Frasincar (order
and spelling match the PDF title block); Progress in Artificial Intelligence 7(4):295-306; online 2018-09-12, issue
2018/12; DOI matches. All fields match. (Springer page names the publisher "Springer Berlin Heidelberg"; Crossref's
"Springer Science and Business Media LLC" is also acceptable, optional.)

**Claim.** Full text: Springer open-access PDF (https://link.springer.com/content/pdf/10.1007/s13748-018-0163-7.pdf).
- Abstract: "we compare a pure review-level algorithm with aggregating the sentiment values of individual sentences. We
  show that ... the pure review-level algorithm outperforms the sentence aggregation method.", exact.
- Same data: SemEval-2016 restaurant reviews (Sec. 3), which carry both review-level and sentence-level aspect labels.
  Aggregation = summing PREDICTED sentence polarities per aspect (Sec. 4, Eq. 1), "summing sentence-level predictions" is right.
- Table 4 (sentence aggregation, test F1): baseSA 0.6824, ontSA 0.7717, "Gold value (upper bound)" 0.9633.
  Table 5 (review level, test F1): Base 0.8020, Final 0.8119. Sec. 5.1: "the review-based algorithm outperforms the
  sentence aggregation algorithm for the test data. The final model has an accuracy that is approximately 4.0% points
  higher than the accuracy of the ontSA model."
Caveats (adversarial findings):
1. The win is on a single test-set run. In ten-fold CV on the training data the order is REVERSED: ontSA 0.8130 (Table 4)
   vs Final 0.8001 (Table 5). The authors state only the test numbers are comparable across tables.
2. Aggregating GOLD sentence labels gives 0.9633 (Table 4 "upper bound"), far above the review-level model (0.8119): the
   aggregation rule itself is not the weak point; the errors of the sentence classifier are.
3. Their "F1" is accuracy ("we calculate the accuracy, which is equal to the F1 score", Sec. 4.4); models are linear SVMs.
If LITERATURE.md uses this paper as evidence that whole-unit classification beats sentence aggregation, it should say
"on the test set" and ideally mention the 0.9633 gold-aggregation ceiling, since that ceiling cuts the other way.

## 11. wadden2022multivers, VERIFIED (location fix: the context-dependence result is Table 4, not Table 3)

**Bib.** https://aclanthology.org/2022.findings-naacl.6.bib: title (with {M}ulti{V}er{S} protected), 6 authors Wadden,
Lo, Wang (Lucy Lu), Cohan, Beltagy, Hajishirzi (order matches PDF), booktitle "Findings of the Association for
Computational Linguistics: NAACL 2022", pp. 61-76, 2022, Seattle, United States, ACL, DOI 10.18653/v1/2022.findings-naacl.6.
All match.

**Claim.** Full text https://aclanthology.org/2022.findings-naacl.6.pdf.
- Sec. 1 (p. 61): "First, the rationales may lack information required to make a prediction when taken out-of-context;
  for instance, they may contain acronyms or unresolved coreferences, or lack qualifiers ...", exact.
- Sec. 7.1 + TABLE 4 (p. 68; not Table 3, which holds the pretraining/encoder/architecture ablations): on 128 annotated
  SciFact test instances (82 self-contained, 46 context-dependent), abstract-level F1 drop from self-contained to
  context-dependent: Multitask -14.0% (84.5 -> 72.7), Pipeline -22.8% (90.7 -> 70.0), MT/PI -66.4% (68.7 -> 23.1).
  "These findings suggest that (1) the Multitask approach is, as expected, best at verifying claims with
  context-dependent evidence". Direction confirmed.
Caveats: (i) small sample (46 context-dependent instances); (ii) in absolute terms Multitask beats Pipeline on
context-dependent instances by only 2.7 F1, and Pipeline is better on self-contained ones and on par overall in the
fully supervised setting (Table 3c); (iii) the authors interpret MT/PI's drop as the model "frequently (and correctly)"
judging out-of-context rationales insufficient, and Pipeline's smaller drop as having "over-fit to context-dependent
rationales and learned to make predictions based on insufficient evidence". Cite Table 4, not Table 3.

## 12. zaidan2007using, CLAIM-WRONG (partial: the "more fruitful" sentence is a hedged hypothesis, not a finding)

**Bib.** https://aclanthology.org/N07-1033.bib: title "Using ``Annotator Rationales'' to Improve Machine Learning for Text
Categorization"; authors Zaidan, Omar; Eisner, Jason; Piatko, Christine (order matches PDF); booktitle "Human Language
Technologies 2007: The Conference of the North {A}merican Chapter of the Association for Computational Linguistics;
Proceedings of the Main Conference"; pp. 260-267; 2007; Rochester, New York; ACL. All match; the Anthology has no DOI
for N07 papers (none given, correct).

**Claim.** Full text https://aclanthology.org/N07-1033.pdf.
- Part 1 VERIFIED. Abstract: "present a learning method that exploits the rationales during training to boost
  performance significantly on a sample task, namely sentiment classification of movie reviews." Sec. 6.1: "At the
  largest training set size, rationales raise the accuracy from 88.5% to 92.2%, a 32% error reduction." (SVM, accuracy,
  9-fold CV; Fig. 2 S1 vs S2).
- Part 2 WRONG AS A FINDING. The full abstract sentence is: "We hypothesize that in some situations, providing rationales
  is a more fruitful use of an annotator's time than annotating more examples." It is a hypothesis restricted to "some
  situations". Sec. 6.2 is explicit that it was not established: "While this fresh benefit was often statistically
  significant, and greater than the benefit from more documents, our experiments did not establish that it was
  significantly greater." The quote may be used only with the hedge, e.g. "Zaidan et al. hypothesize that, in some
  situations, providing rationales is a more fruitful use of annotator time than labelling more examples."
  Note also Sec. 4.3: "For each annotator except A2, providing rationales only took roughly twice the time (Task 3 vs.
  Task 1)", so the cost side is not free (the authors call this a low overhead).

---

## Cross-cutting findings (finder's notes, novelty_attack.md)

Checked the finder's paraphrases and the drop-in sentence in novelty_attack.md (Sec. "Recommended rewording") against the
same full texts:
1. **Hershey cited as MIL "upper bound", not supported.** The drop-in says "instance-supervised models are the usual
   upper bound in multiple-instance learning (Hershey et al., 2021; Li et al., 2021)". Hershey et al. never mention
   multiple-instance learning or an "upper bound" (grep of the full arXiv text: no hits); they compare strong vs weak /
   diffuse labels. Cite Hershey only for "temporally precise labels improve the clip-level decision on the same clips with
   the same model". Also "usual" upper bound rests on a single paper (Li et al. call their own fully supervised model
   "upper-bound"); say "is treated as an upper bound (e.g. Li et al., 2021)".
2. **DSMIL range wrong in the notes.** novelty_attack.md (summary item 2) gives "Camelyon16 AUC 0.9362 vs 0.8641-0.9165 for
   slide-label MIL". Table 1 MIL AUCs actually range 0.7620 (mean-pooling) to 0.9165 (DSMIL-LC); mean-pooling 0.7620,
   MILRNN 0.8064, MS-MILRNN 0.8371 lie below 0.8641. Use "0.76-0.92" or name the specific models.
3. **Magge "positive if it finds any span"** (notes, Sec. C) is the finder's reading; not stated in the paper (see key 7).
   The two Introduction quotes in the notes ("Due to the higher complexity of the NER task, NERs have a lower sensitivity
   to identifying ADEs in tweets compared to a classifier"; "we find it important to reevaluate if ADE classifiers
   continue to be an essential step") are exact.
4. **Zaidan "more fruitful"** is quoted in the notes without "We hypothesize that in some situations" (see key 12).
5. **MultiVerS context-dependence numbers** are Table 4, not Table 3 (see key 11). The notes' Table 3c numbers
   (Multitask 77.6/77.3/72.5; Pipeline 78.4/77.6/70.9; MT/PI 70.6/73.3/60.3, abstract-level F1) are correct.
6. **de Kok**: the notes correctly say aggregation is a SUM of predicted polarities, but omit that the 10-fold CV order is
   reversed and that gold-sentence aggregation reaches 0.9633 (see key 10).
7. Hershey bib copied Crossref's mis-parsed "Channing Moore, R" (see key 6). magge title needs brace protection (key 7).

Residual risk (decision): Hershey and Rei claims were checked on arXiv copies; Rei was additionally confirmed on the AAAI
publisher PDF (identical numbers). The IEEE copy of Hershey et al. was not accessible (paywall); arXiv v1 (posted
14 May 2021, ICASSP layout) is assumed to match the published version.

---

## Summary table

| # | key | verdict | action |
|---|-----|---------|--------|
| 1 | rei2019jointly | VERIFIED | none (numbers are single-run test F1; FCE gain only +0.20) |
| 2 | dasanmartino2019fine | VERIFIED | none (60.98 = best of two MGN variants; test, mean of 3 seeds) |
| 3 | zhang2016rationale | VERIFIED | numbers are accuracy (CV means), not F1 |
| 4 | farkas2010conll | VERIFIED | none (Wikipedia margin small; #2 and #4 were token classifiers) |
| 5 | li2021dual | VERIFIED | keep Crossref/IEEE pages 14313-14323 (CVF OA pages 14318-14328 differ); say "single-scale DSMIL" |
| 6 | hershey2021benefit | FIX-BIB + CLAIM-WRONG (minor) | author `Moore, R. Channing`; do not attach the "+Diffuse/+Strong" quote to the 0.82/0.96 (Diffuse-67k/Strong-67k) numbers |
| 7 | magge2021deepademiner | FIX-BIB; claim VERIFIED w/ caveat | brace `{DeepADEMiner}`, `{T}witter`; "any span found" is our inference, not the paper's |
| 8 | pavlopoulos2022from | VERIFIED | "approaches" holds vs BiLSTM tagger only (SpanBERT tagger 63.0 >> 57.7/58.8) |
| 9 | hu2025decomposition | VERIFIED | quote is in Sec. 4.2, not the abstract |
| 10 | dekok2018review | VERIFIED w/ substantive caveat | test-set only (CV reversed 0.8130 vs 0.8001); gold-sentence aggregation = 0.9633 |
| 11 | wadden2022multivers | VERIFIED | context-dependence result is Table 4, not Table 3 |
| 12 | zaidan2007using | CLAIM-WRONG (partial) | "more fruitful use of an annotator's time" is a hedged hypothesis ("We hypothesize that in some situations"); Sec. 6.2 says it was not established |

Finished 2026-10-05 02:54 CEST.
