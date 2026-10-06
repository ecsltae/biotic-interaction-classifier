# Literature for the sentence-level reframe: "ask about the pair, even if you want the sentence"

Literature stage, 2026-10-05 (unattended). Paths are relative to `~/MetaP/classifier`.

- New BibTeX: `paperA/reframe_extra.bib` (50 entries; no key or work is also in `paperA/references.bib`).
  Keys already in `references.bib` are written `[cited]` below.
- The status of every new entry is in the **Verification** section at the end. Tags: [abstract] = only the abstract
  was read; [full text, X] = the passage was read at location X.
- Working notes with longer quotes: `results/reframe_2026-10-05/lit_work/agent{A,B,C}.md`, `main.md`,
  `verify{1,2}.md`, `novelty_attack.md`, `biotxplorer_note.md`.
- Numbers about our own results come from `CONTEXT.md` §3 (sandbox) and must be re-checked against
  `ANALYSIS.md` before the writer uses them. This file adds no new result of ours.

---------------------------------------------------------------------------------------------------------------

## Q1. Is reading a sentence-level decision off pair-level verification known?

Key works (the rest of the section is supporting): dietterich1997solving, hoffmann2011knowledge,
ilse2018attention, dasanmartino2019fine, chowdhury2013exploiting, magge2021deepademiner.

Short answer: the rule is known, and so is the general principle; the specific controlled comparison is not.
"A passage is positive if at least one of its candidate pairs is positive" is the standard multi-instance (MI)
assumption, and max or deterministic OR is how distant supervision has aggregated instances since 2010. That
finer-grained supervision helps a coarse "contains any X?" decision is also known (tokens, spans, rationales;
table Q1(d)), and with instance labels an instance-supervised model serves as the upper bound for MI classifiers
(li2021dual). Detecting
relations at sentence level and then deciding pairs is an established pipeline. We found no work that varies only
the question put to an LLM teacher (pair vs sentence), with the same passages and encoder, trains the pair student
on no sentence labels at all, and compares the two students on the sentence decision.

| Key | What it shows | How our claim relates |
|---|---|---|
| dietterich1997solving | Defines the MI problem; a hypothesis is consistent if it "classifies at least one feature vector of every positive example as positive" [full text, Sec. 1]. | Origin of the at-least-one rule our max over pairs instantiates. Cite it for the rule; do not present the max as ours. |
| andrews2002support | "a bag is "positive" if at least one of its member patterns is a positive example ... the set-level classifier is by design induced by a pattern-level classifier" [full text, Sec. 1]. | The cleanest statement that a set-level (sentence) decision can be induced from an instance-level (pair) classifier. |
| riedel2010modeling [cited], hoffmann2011knowledge, zeng2015distant | Distant supervision as MI learning: "at least one sentence that mentions these two entities might express that relation" (Riedel, Sec. 1, p. 149); MultiR aggregates sentence-level extractions with "deterministic OR" factors [Hoffmann, full text, Sec. 3, p. 543]; PCNN trains on the top-scoring instance of each bag [Zeng, Sec. 3.5]. | Same operator, transposed bag: there the bag is the sentences of one pair, the bag decision is pair-level and instance labels are latent. Ours: the bag is the candidate pairs of one passage, the bag decision is sentence-level, and the instances are labelled. |
| ilse2018attention | Separates instance-level MIL (score instances, pool) from embedding-level MIL (classify the bag); citing Wang et al., says the bag-level route is preferable "since the individual labels are unknown" [full text, Sec. 2.1]. | Without instance labels the MI literature prefers the bag-level route. With instance labels, an instance-supervised model serves as the upper bound (li2021dual, Q1(d)), so "pair at least as good" is the expected direction when pair labels refine sentence labels (BioRED, by construction). It is not guaranteed when the two are separate answers, as for the biodiversity sentence labels. Do not frame the result as surprising. |
| laban2022summac | One MNLI-trained NLI model applied at finer granularity and aggregated beats document-level use. With the zero-shot max-then-mean aggregator, balanced accuracy goes from 56.4% (full document) to 71.2% (two-sentence chunks) [full text, Sec. 5.3.3, Table 5]. The authors' "from 56.4% to 73.5%" ends on their learned SummaC-Conv aggregator, so do not attach 73.5% to max/mean. | Closest precedent for "decide at the fine level, aggregate, rather than ask the coarse question". The NLI model is fixed and only the inference granularity varies; we train both granularities on the same data. |
| min2023factscore | A single binary judgement is "inadequate" when a text mixes supported and unsupported units; verifies atomic facts one by one [abstract]. | The decompose-verify-aggregate pattern is known for verification; cite in one clause. |
| jia2019document, verga2018simultaneously | Biomedical RE aggregates mention-pair scores into entity-pair scores with LogSumExp, "a smooth approximation to the max function" [Verga, Sec. 2.4]; Jia et al. pool mention representations, and replacing logsumexp pooling with max pooling "lost 3.8 absolute points in AUC" [Jia, Sec. 4]. | Expect "why hard max?". Our OR is the semantics of the sentence question ("any interaction"), not an approximation; a soft aggregator could be reported as a robustness check. |
| chowdhury2013exploiting | DDI extraction: a first-stage sentence classifier where "any sentence that contains at least one DDI is considered by the classifier as a positive (training/test) instance" discards sentences before the pair classifier [full text, Sec. 2.1]. It is a negation-cue filter applied only to sentences selected by rules (including at least two drug mentions and a negation cue); +1.0 F overall [Sec. 3, Table 2]. | Precedent for a sentence label defined as "contains at least one related pair", used as a filter. Narrow in scope (negation-based); the pair decision is still made by the pair classifier, and nobody reads the sentence decision off it. |
| xie2021revisiting | RERE argues for sentence-level relation detection first because pair classification faces O(m²) candidates for O(m) relations [full text, Sec. 2.2]. Its sentence stage is multi-label relation-type detection on distantly supervised data [Sec. 3.1]. | The strongest published argument for the sentence-first route; it must be cited and answered. Its own O(m²) argument is why a sentence label does not transfer to pairs when a passage names more than two taxa. |
| liu2019event | Sentence-level (sentence, event type) decisions without trigger annotation are "competitive" with trigger-supervised systems [abstract]. | Counter-example to "finer supervision always wins" (not a controlled comparison). Keep our claim to our setting. |

### Q1(d). Fine-grained supervision for coarse decisions (found by the novelty-attack search; see Verification for status)

| Key | What it shows | How our claim relates |
|---|---|---|
| zaidan2007using | Annotator rationales used in training raise movie-review sentiment accuracy from 88.5% to 92.2% at the largest training size [full text, Sec. 6.1]. The authors only hypothesise that "in some situations" rationales are a better use of annotator time than more examples, and say their experiments "did not establish" it [abstract; Sec. 6.2]. | Ancestor of "ask the annotator for the finer evidence even if you want the coarse label"; our analogue asks the teacher about the pair. |
| zhang2016rationale | Sentence-level rationale supervision (RA-CNN) improves document classification over a document-label CNN on the same data, including biomedical risk-of-bias documents, and explains predictions [full text, Tables 2-3]. | Fine supervision helps the coarse decision and localises; joint training, weighted pooling, human labels. |
| rei2019jointly | Same data and encoder: adding token-level labels raises sentence-level F1 on all four tasks (e.g. CoNLL-2010 hedges 83.87 -> 85.90) [full text, Table 3]. | Pre-empts "fine-grained supervision helps the coarse decision" as a general observation (joint training, tokens not pairs). |
| dasanmartino2019fine | Sentence task "contains at least one propaganda technique"; same corpus and BERT, sentence-only F1 57.74 vs 60.98 with fragment-level supervision; the fine level also explains the decision [full text, Table 7, Sec. 8]. | Closest NLP precedent for both halves (existential sentence question; fine output for free). Multi-task: it still uses sentence labels. |
| farkas2010conll | CoNLL-2010 hedge detection: "sentences containing at least one cue were considered as uncertain"; on biological text the top systems tagged cues and called a sentence uncertain if any cue was found; on Wikipedia, bag-of-words sentence classifiers won [full text, Sec. 6.2]. | The oldest biomedical comparison of "fine detector + at least one" against direct sentence classification on the same data; uncontrolled (different teams and features), and the winner depended on the domain. |
| hershey2021benefit, li2021dual | Outside text, with a same-data control: the same clips and model trained on temporally strong labels vs the same labels spread over the whole clip, clip-level d' 0.96 vs 0.82 (Strong-67k vs Diffuse-67k) [Hershey, Table 1]. A patch-supervised model is the "upper-bound fully-supervised model" for slide classification, AUC 0.9362 vs 0.8641 for max-pooling MIL and 0.8944 for single-scale DSMIL [Li, Sec. 4.1, Table 1]. | Instance supervision is expected to help the bag decision; our "at least as good" is the expected direction when pair labels refine sentence labels. Hershey does not discuss MIL or upper bounds; cite only li2021dual for the upper-bound point. |
| magge2021deepademiner | Same tweet corpus: for "identifying tweets containing ADEs", the RoBERTa tweet classifier (F1 0.63) "outperforms the NER (F1-score = 0.41)" by about 22 points [full text, Results, Experiment 2]. How the tagger's tweet-level decision is computed is not stated; "positive if any span is found" is our reading. | Counter-example a reviewer will cite. Differences: different encoders, and the tagger must find the spans, while our verifier classifies candidate pairs that co-occurrence already supplies. Scope the claim to pair-conditioned verification of given candidates. |
| wadden2022multivers | Decisions made from rationales taken out of context lose on context-dependent instances; the full-document model degrades least (F1 drop 14.0% vs 22.8% for the pipeline, on 46 context-dependent SciFact instances) [full text, Sec. 7.1, Table 4]. | Our verifier reads the full passage with the pair marked; say so, and cite as the reason context must be kept. |
| hu2025decomposition | Claim decomposition "generally benefits weaker verifiers, while it tends to negatively affect stronger verification systems" [full text, Sec. 4.2]; the effect depends on input granularity [Sec. 4.1]. | Asking finer questions is not uniformly better; consistent with a gain confined to passages naming more than two taxa. Our decomposition is deterministic (given candidates), so its decomposition-error term is absent. |
| pavlopoulos2022from | Rationale extraction on top of a post-level toxicity classifier is "surprisingly promising" for locating toxic spans [abstract]; with more post-level data its span F1 rises from 57.7% to 58.8%, close to a BiLSTM span tagger but well below the best span-trained tagger (SpanBERT, 63.0) [full text, Sec. 6]. | Weakens "only the pair verifier says which pair": write "only the pair verifier is trained to say which pair; post-hoc attribution from a sentence classifier is untested here". |

Prior work we must cite for Q1: dietterich1997solving (or andrews2002support), riedel2010modeling [cited],
hoffmann2011knowledge, ilse2018attention, laban2022summac, chowdhury2013exploiting, xie2021revisiting; from
Q1(d): rei2019jointly, dasanmartino2019fine, farkas2010conll, li2021dual (or hershey2021benefit),
magge2021deepademiner.

What a reviewer would call known: the at-least-one rule and max/OR aggregation (Dietterich; Andrews; Hoffmann;
Zeng); deciding at a finer granularity and aggregating (SummaC; FActScore); finer-grained supervision improving
the coarse decision on the same data, and giving the fine output for free (Zhang 2016; Rei & Søgaard 2019;
Da San Martino 2019; Hershey 2021; Li 2021); "fine detector + at least one" against a direct sentence classifier
in a biomedical shared task (CoNLL-2010); sentence-level relation detection
as a filter or first stage (Chowdhury & Lavelli; RERE); using pair-extraction output for document triage
(BioCreative III Team 65 and BioCreative VI Team 433, reported in krallinger2011protein and
islamajdogan2019overview, as features or runs, without a controlled comparison; see Q2).

What is new, as far as four independent searches found: (i) the controlled variable is the question put to the
LLM teacher (pair vs sentence) when distilling a literature-mining filter, with the same teacher, passages and
encoder (biodiversity); on BioRED the same comparison is run with gold-derived labels instead of a teacher; (ii) the pair student never sees a sentence label, and its sentence decision is a plain max over the given
candidate pairs, each marked in the full passage (the NLP precedents train both levels jointly); (iii) the gain is
located on passages naming more than two candidate entities; (iv) on the biodiversity benchmark the sentence and pair
labels are separate annotations rather than one refining the other (pair: expert gold; sentence: the second human
annotation once filled in, silver LLM labels until then), so the MI upper-bound argument does not
settle the outcome in advance (on BioRED it does: there the sentence label is the OR of the gold pair labels);
(v) we measure how far a perfect sentence filter falls short of populating a pair database (Q5).

---------------------------------------------------------------------------------------------------------------

## Q2. What filter question do curation and database-population pipelines ask?

Key works: krallinger2008overview, islamajdogan2019overview, polajnar2011protein, lim2016minter,
thessen2014knowledge.

DOC = "is this document relevant / does it describe an interaction?"; SENT = the same for a sentence or passage;
PAIR = "do these two entities interact (here)?".

| Key | Domain | Filter question | Note |
|---|---|---|---|
| krallinger2008overview | PPI (BioCreative II) | DOC triage (IAS), then PAIR extraction (IPS); the sentence subtask (ISS) returns up to five evidence passages per protein pair [full text]. | Sentences are wanted as evidence for a pair, not as a free-standing label; "multiple interaction pairs" in one sentence is listed as a difficulty. |
| islamajdogan2019overview | PPI + mutations (BioCreative VI PM) | DOC triage, then PAIR extraction on a subset of the triage documents; best triage F 69.06% [abstract]; best pair F 37.7% with homologous genes counted as matches, 34.8% exact match [full text, Results, Tables 4-5; the abstract's "37.73%" is not in the tables]. | Team 433 added binned probabilities from a sentence-level PPI-triplet model (two proteins + an interaction word) to its triage features [full text, team description]; its triage F1 was 0.6713, below the best. An informal precedent for feeding pair scores into the document decision, not a comparison. |
| krallinger2011protein | PPI (BioCreative III ACT) | DOC only. | Team 65 used "a score delivered by their PPI detection pipeline" as a feature, and two runs "used only results of their protein-protein interaction detection pipeline" [full text, team descriptions]; not the best team. Also document-level kappa (Q4). |
| wiegers2009text | Chemical-gene (CTD) | DOC ranking for curation, using sentence co-occurrence of actors and action terms as a feature [full text]. | Two-level curator agreement (Q4). |
| bravo2015extraction | Gene-disease (BeFree/GAD) | PAIR in a sentence; negatives are co-occurring pairs not curated as associations [full text]. | Precedent for a pair filter over co-occurrence candidates. |
| polajnar2011protein | PPI | SENT, chosen as "simpler", with the stated cost that it "only locates the sentence that describes the PPI and not the exact interacting pair" [full text]. | The sentence-first belief our comparison tests, stated plainly. |
| lim2016minter | Microbial interactions | DOC, then every co-mentioned species pair of a positive abstract becomes a candidate [full text, Sec. 2.2]. | Abstract detection specificity 95%, AUC 0.97, but "interaction level recall = 95%, precision = 25%" [abstract]; low species-level precision is "partly because it identifies abstracts containing interactions and reports all pairs of species names in such abstracts" [full text, Sec. 3.2, read through a rendering tool, near-verbatim]. Keep "partly". The key biotic precedent for Q5. |
| thessen2014knowledge | Ecological associations, Encyclopedia of Life | Section-level filter; every taxon in the section is paired with the page's taxon [full text]. | "One cannot assume that all of the taxa mentioned on an EOL taxon page have an ecological relationship with the topic of that page" [full text, Discussion]. |
| elkhettari2023building | Species-species RE corpus | PAIR = SENT by construction: only sentences "where exactly two species are mentioned" [full text, Sec. 3.2]. | The regime where our pair gain vanishes (BioRED: no gain on two-entity sentences). |
| thieu2012literature | Host-pathogen | DOC, then SENT, with the same classifier; the pair is extracted afterwards [full text, Sec. 2.1]. | The sentence-question pipeline our proposal replaces, in the host-pathogen setting. |

Already cited, filter question read by the finders: keck2025extracting (paragraph pre-filter by co-occurrence +
keyword, then an LLM lists pairs; its "accuracy 89.5%" is 229/(229+27), a precision, as its own abstract says);
zou2026llm (first step: does a comment contain an interaction?); dimitrova2020semantic (table-level triage);
cuzick2023interspecies (the curated unit is the pathogen-host pair).

Takeaway: biomedical challenges split DOC triage from PAIR extraction and score them separately; biodiversity
pipelines ask a passage, section, table or document question, or ask a generative model to list pairs. None of the
works found compares a pair-conditioned verifier with a sentence-level classifier on the same data.

---------------------------------------------------------------------------------------------------------------

## Q3. Pair-conditioned inputs and question specificity

Key works: lee2020biobert, zhong2021frustratingly, zhang2023aligning, zhang2025entity, mraz2026fewshot,
jiang2019challenge (plus the already-cited zhang2017position, soares2019matching, levy2017zero).

| Key | What it shows | How our claim relates |
|---|---|---|
| zhang2017position [cited], soares2019matching [cited], zhou2022improved [cited] | The instance is (sentence, e1, e2); entity markers and typed markers. | Our pair input is in this tradition; already cited. |
| lee2020biobert, peng2019transfer | Biomedical RE anonymises the target pair (@GENE$, @DISEASE$, @DRUG$) in one copy of the sentence per pair [full text, BioBERT Sec. 3.3; BLUE Sec. 4.1.2]. | Pair conditioning is the default biomedical recipe; never claim it. Difference: they hide the names; our pair arm names the two taxa. |
| zhong2021frustratingly | PURE: one encoder pass per pair with typed markers; pair-specific encodings beat one shared sentence encoding by +5.0/+7.4 F1 with gold entities [full text, Sec. 5.1, Table 4]; the cost is "once for every pair of entities" [Sec. 3.3]. | Pre-empts "pair-specific encodings beat a shared encoding" for pair decisions; not the sentence decision. Honest citation for the per-pair cost. |
| li2019entity, zhang2023aligning, levy2017zero [cited], sainz2021label [cited] | RE as a question about named entities: multi-turn QA from a head entity [Li]; zero-shot LLM RE as multiple choice over relation templates filled with both entities, +8.2/+8.6 F1 [QA4RE, Sec. 1]. | Supports "ask about the pair". All arms of these papers are already entity-conditioned, so they say nothing about pair vs sentence questions. |
| zhang2025entity | LLM document-level RE: entity-pair-level candidate relations vs document-level; replacing the pair-level step with the document-level one drops F1 by 6.77 on DocRED dev [full text, Sec. 4.3]. | The closest LLM evidence that a pair-specific query beats the same query posed once for the whole text; pair-level output, general domain. |
| jimenezgutierrez2022thinking | GPT-3 in-context learning "struggles with the null class ... entity pairs that hold none of the target relations" [full text, Sec. 1]. | Co-occurring but non-interacting pairs are what LLMs miss; consistent with our zero-shot results. |
| mraz2026fewshot (arXiv only) | On BioREDirect, per-pair classification has higher recall, joint generation of all pairs is "more precise and computationally efficient", F1 broadly comparable [abstract; full text, Sec. 5]. | Counter-evidence: asking per pair is not known to dominate for LLMs. It compares per-pair with list-all-pairs, not with a sentence yes/no question. |
| jiang2011target, jiang2019challenge | Sentiment analysis: target-independent classifiers give every target in a text the same answer; "sentiments ... not truly about the query" make up about 40% of a public tweet-sentiment service's errors in a small manual check (5 queries x 20 tweets) [Jiang 2011, Sec. 6.1]. When sentences name one aspect, aspect-level sentiment "degenerate[s] to sentence-level sentiment analysis" and sentence classifiers are competitive; they fail on multi-aspect sentences [MAMS, abstract and Sec. 4.2]. | The structural argument ("a whole-text classifier cannot tell which target its answer is about") and our BioRED localisation (no gain with two entities, gain with more) both have a precedent in another field. Cite as precedent, not as our discovery. |

---------------------------------------------------------------------------------------------------------------

## Q4. Annotation semantics: sentence-level vs pair-level labels, agreement, UNSURE

Key works: pyysalo2008comparative, wiegers2009text, thessen2014knowledge, dumitrache2018crowdsourcing,
plank2022problem (plus luo2022biored, already cited).

| Key | Unit | Agreement reported (as read) |
|---|---|---|
| pyysalo2008comparative | PPI pairs in sentences, five corpora | No common definition: "there is no general consensus regarding PPI annotation" [abstract]; corpora differ in how much sentence pre-filtering they embed (0% to 69% of sentences without interactions, Table 3). No IAA for the corpora. |
| krallinger2011protein | Document ("PPI-relevant abstract?") | Cohen's kappa 0.85 between two databases' curators; 0.69 against an expert [full text]. |
| wiegers2009text | Document disposition and extracted interactions, same articles (CTD) | Disposition: 77% of 112 articles, average pairwise agreement 85%; interactions vs adjudicated gold: precision 0.91, recall 0.71 [full text]. Different measures at the two levels. |
| thessen2014knowledge | Taxon-taxon association from a passage | Fleiss' kappa 0.840, three annotators [full text]. |
| herrerozazo2013ddi | Drug-drug pair in a sentence | "Kappa up to 0.96 and generally over 0.80), except for the DDIs in the MedLine database (0.55-0.72)" [abstract only]. |
| luo2022biored [cited] | Document-level relation between concept pairs | IAA 77.91% for relations, 97.01% for entities, 85.01% for novelty [full text, Table 4 and "Data characteristics"; re-read by the main session]. |
| gurulingappa2012development | ADE corpus: drug-effect relations at sentence level, plus a sentence-level informative/non-informative task | Double annotation and harmonisation [abstract only; no IAA number read]. |
| dumitrache2018crowdsourcing | Medical relation, pair in sentence | Removed sentences where no decision was reached (32 cause, 15 treat) and named the cost: "Eliminating these sentences is a disadvantage to a system like ours" [full text, arXiv version, Sec. 3.4]. |
| plank2022problem | Label variation in general | Filtering out low-agreement items "can yield worse performance ... and it wastes data" [full text, Sec. 3]. |

For the dual labels: we found no corpus that reports agreement for a sentence-level "any interaction" label and a
pair-level label on the same items. Write "we did not find", not "none exists". If the user's second annotation
yields agreement figures, Cohen's kappa is well defined for both labels (both have fixed negatives).

For the UNSURE removal: dropping undecidable items is a known filtering choice (dumitrache2018crowdsourcing did it
for pair-in-sentence medical RE; plank2022problem names its cost). The paper should state it as a limitation:
the evaluation covers the decidable subset, and the removed items are likely the vague or indirect interactions.

---------------------------------------------------------------------------------------------------------------

## Q5. A perfect sentence filter cannot populate a pair database

Key works: pyysalo2008comparative, lim2016minter, tikk2013detailed, gao2021manual (plus riedel2010modeling and
rosenman2020exposing, already cited).

| Key | Evidence | Relation to our 0.724 |
|---|---|---|
| pyysalo2008comparative | Precision of proposing every co-occurring pair equals interactions per entity pair (I/EP) [full text, Results]. In LLL every sentence contains an interaction (0% without, Table 3), yet co-occurrence pair precision is 0.50 (Table 2). With only interacting entities kept, co-occurrence precision is 0.53 (AIMed), 0.53 (BioInfer), 0.64 (HPRD50), 0.88 (IEPA), 0.50 (LLL) (Table 4). | Our derivation, not the authors' statement: removing every protein that takes part in no interaction removes all pairs of interaction-free sentences plus some negative pairs inside interaction sentences, so a perfect sentence filter has pair precision at most these values on these corpora (assuming the filter acts on mentions within sentences, which the caption implies but the text does not spell out). Our 0.724 falls within the range of these upper bounds. State it as derived. LLL has 77 sentences. |
| tikk2013detailed | "the more protein mentions a sentence exhibits, the lower the ratio of positive pairs" [full text, Results, about Fig. 7]. | The mechanism behind our gain being confined to passages naming more than two entities. Numbers are only in the figure. |
| lim2016minter | Abstract detection with AUC 0.97 (abstract-level specificity 95%) goes with species-pair precision 25%; accepting all co-mentioned pairs gives 7% [abstract; Table 1]. Do not mix the abstract-level 95% specificity with the species-level figures in Table 1. | Biotic precedent at document level. Our 0.724 is the sentence-level counterpart on our benchmark. |
| polajnar2011protein | Sentence-level PPI detection "only locates the sentence ... and not the exact interacting pair" [full text]. | Qualitative statement of the same limit. |
| riedel2010modeling [cited], gao2021manual | Co-mention is not assertion: the distant-supervision assumption is violated 31% of the time on NYT [Riedel, Sec. 1; Table 1: 38/35/20% for three relations, 100 sampled candidates each]. After human re-annotation of NYT10's held-out set, "at the fact level, the DS annotations only have a precision of 69.1% and a recall of 33.9%" [Gao, Sec. 3.1]. | The converse direction (related pair, sentence silent). Use for "co-mention is not assertion", not as an estimate of our quantity. Do not use Gao's "32% N/A" or "53% wrongly labeled": the 9,744 sentences include 5,000 that a BERT model picked from the N/A pool, and the 53% is mostly missing relations. |
| rosenman2020exposing [cited] | In their challenge set, "In 57% of the sentences, there are at least two classification instances with conflicting labels", 3.7 candidate pairs per sentence; only 17.2% of TACRED sentences annotate more than one pair [full text, Sec. 3]. | General-domain sentence-positive, pair-negative cases. The set was sampled from model-positive sentences, so 57% is not a base rate. |

No work we read reports the share of negative pairs inside sentences that contain at least one positive pair.
It could be computed from the public PPI corpora; that would be a new measurement, not a citation.

---------------------------------------------------------------------------------------------------------------

## NOVELTY STATEMENT (draft; bracketed parts depend on ANALYSIS.md)

> Reading a passage-level decision off pair-level decisions is the standard multi-instance assumption
> \citep{dietterich1997solving,hoffmann2011knowledge}, pair-conditioned inputs are the default in relation
> extraction \citep{zhang2017position,lee2020biobert}, and finer-grained supervision is known to help a coarse
> ``contains any'' decision when both levels are trained \citep{rei2019jointly,dasanmartino2019fine}; we claim none
> of these. What we add is a controlled measurement of which question to ask when the sentence decision is the
> goal: with the same co-occurrence passages, the same encoder [and the same teacher], a verifier trained only on
> pair questions, whose passage decision is the maximum over the given candidate pairs, each marked in the full
> passage, [is no worse than] a classifier trained on the sentence question [and is better on passages that name
> more than two entities], on BioRED and on biotic-interaction passages. The outcome is not fixed in advance: a
> fine-grained extractor read at the coarse level can lose to a direct classifier \citep{magge2021deepademiner},
> and on our biodiversity benchmark the sentence and pair labels are separate annotations. Only the pair route
> is trained to say which pair interacts; a perfect sentence filter would pass our candidates at a pair precision
> of [0.724], and sentence- and document-level filters are known to leave many non-interacting pairs
> \citep{pyysalo2008comparative,lim2016minter}.

Conditions for the writer:
- Fill the brackets only from ANALYSIS.md. "No worse than" needs a paired test on both benchmarks; if the
  sentence-trained classifier wins somewhere, say where. "Better on passages naming more than two entities" is
  shown on BioRED for the paper's sentence arm (+0.120 on sentences naming more than two entities, submission
  abstract); check it for the sentence-label-trained arm
  (sentlab) and for the biodiversity benchmark before keeping it.
- On BioRED the sentence label is the OR of gold pair labels ("co-mentions a related pair"), so the MI assumption
  holds by construction there and the instance-supervised route is the expected winner (li2021dual); say so.
  The human SENTENCE labels on the biodiversity benchmark, where SENTENCE = 1 and PAIR = 0 occurs, are the real
  test.
- "The same teacher" holds only for biodiversity arms that were both trained on teacher answers. On BioRED both
  arms use gold-derived labels (no teacher). Word it per benchmark.
- If the passage decision uses TaxoNERD-enumerated pairs rather than the retrieved candidates, say "candidate pairs
  enumerated by a taxon recogniser" and report its cost (CONTEXT.md weakness 5).
- The 0.724 is from the sandbox (CONTEXT.md §3, silver labels); re-derive it from the human SENTENCE labels when they
  exist.

---------------------------------------------------------------------------------------------------------------

## CLAIMS WE MUST NOT MAKE (and what refutes each)

1. "We introduce max-over-pairs aggregation" / "first to derive a sentence decision from pair decisions."
   Refuted by dietterich1997solving, andrews2002support, hoffmann2011knowledge, zeng2015distant; a sentence label
   defined as "contains at least one related pair": chowdhury2013exploiting; pair-extraction scores used for
   document triage: krallinger2011protein (Team 65), islamajdogan2019overview (Team 433).
2. "Pair-conditioned input is our contribution." Refuted by zhang2017position [cited], soares2019matching [cited],
   lee2020biobert, peng2019transfer, zhong2021frustratingly.
3. "We are the first to show that a sentence or document filter has limited pair precision." Refuted by
   pyysalo2008comparative, lim2016minter, polajnar2011protein, thessen2014knowledge.
4. "The pair route winning is surprising" or "the pair route is known to win". Without instance labels the MI
   literature prefers the bag-level route (ilse2018attention); with instance labels the instance-supervised model
   serves as the upper bound (li2021dual). On BioRED, where the sentence label is the OR of
   the pair labels, a pair win is expected. "Coarse supervision is always worse" is contradicted by liu2019event
   and magge2021deepademiner.
5. "Asking an LLM about one pair at a time is known to be better." Qualified by mraz2026fewshot (per pair: higher
   recall, lower precision, comparable F1, joint generation up to 25x cheaper). A Springer CCIS chapter
   (Labonte et al. 2026, "Structured Multi-extraction Prompting ...") reports batch extraction better, but only its
   abstract was read; it is not in the .bib and must not be cited without reading it.
6. "Max is obviously the right aggregator." jia2019document lost 3.8 AUC points with max instead of logsumexp in
   its setting; justify max by the semantics of "any interaction", or report a soft aggregator.
7. "The gain on passages with more than two entities is a new phenomenon." The same localisation is documented
   for aspect-based sentiment (jiang2019challenge) and implied for PPI by the falling positive-pair ratio
   (tikk2013detailed, pyysalo2008comparative: entity pairs grow quadratically, interactions linearly).
8. "Distant supervision is about 30% noisy" as a general fact. Figures depend on corpus and protocol: Riedel 31%
   on NYT (300 sampled candidates, three relations); Riedel's text says 13% for Wikipedia but its own Table 1
   averages 16.7% (20/20/10), so do not quote the 13%; gao2021manual: fact-level precision of distant labels
   69.1% on NYT10 held-out (its "32% N/A" and "53% wrongly labeled" are not false-positive rates).
9. "No biotic-interaction corpus has pair labels." elkhettari2023building (species pairs, two-species sentences),
   thessen2014knowledge (taxon associations with Fleiss kappa 0.840).
10. "Ecological LLM pipelines skip the sentence question." zou2026llm [cited] first classifies whether a comment
    contains an interaction.
11. "Humans agree more on pair labels than on sentence labels" (or the reverse). No work found measures both on
    the same items; wiegers2009text is the only two-level figure and uses different measures.
12. "Dropping UNSURE items is harmless." plank2022problem; dumitrache2018crowdsourcing.
13. "keck2025extracting reports 89.5% accuracy." It is a precision (229 of 256 predictions; its abstract says
    precision).
14. Mixing PPI co-occurrence precisions across papers: Airola et al. 2008 and Pyysalo et al. 2008 differ (corpus
    versions). Quote Pyysalo only.
15. "A published statistic gives the share of negative pairs inside interaction-positive sentences." None found.
16. "Finer-grained supervision helps the coarse decision" as our finding. Known: zaidan2007using,
    zhang2016rationale, rei2019jointly, dasanmartino2019fine (all train both levels jointly), and outside text
    hershey2021benefit, li2021dual.
17. "Nobody has compared 'fine detector + at least one' with a direct sentence classifier in biomedical text."
    farkas2010conll (CoNLL-2010 hedge detection: cue taggers won on biological text, bag-of-words sentence
    classifiers on Wikipedia; uncontrolled). Say instead: not under a same-teacher, same-encoder control, with
    entity-pair conditioning.
18. "Fine-grained models read at the coarse level always win." magge2021deepademiner (ADE tagger 0.41 vs tweet
    classifier 0.63 F1 for flagging tweets), wadden2022multivers (out-of-context fine units lose on
    context-dependent cases). Scope the claim: an existential question ("any interaction", where OR is the exact
    link), given candidates, and the full passage in view.
19. "Only a pair model can say which pair interacts." Qualified by pavlopoulos2022from (attribution from a coarse
    classifier comes close to a BiLSTM span tagger at localisation, though not to the best span-trained tagger). Say "only the pair verifier is trained and
    evaluated to say which pair; attribution from the sentence classifier is not tested here".
20. "Asking finer questions is better" unconditionally. hu2025decomposition (decomposition helps weak verifiers,
    can hurt strong ones), mraz2026fewshot. Say "no worse overall, better where passages name more than two
    entities" if ANALYSIS.md supports it.

---------------------------------------------------------------------------------------------------------------

## DRAFT RELATED-WORK PARAGRAPHS (LaTeX; keys from both .bib files)

```latex
\paragraph{Sentence decisions from pair decisions.}
Calling a passage positive when at least one of its candidate pairs is positive is the standard
multi-instance assumption \citep{dietterich1997solving}, and a deterministic OR or a max over instances
is how distant supervision aggregates them \citep{riedel2010modeling,hoffmann2011knowledge}; there the
instances are the sentences of one pair and their labels are unknown, which favours classifying the bag
directly \citep{ilse2018attention}. Trained jointly with finer labels, models make better coarse
``contains any'' decisions \citep{rei2019jointly,dasanmartino2019fine}, and instance-supervised models
serve as the upper bound for multiple-instance ones \citep{li2021dual}; yet an adverse-event span tagger
used to flag tweets loses to a tweet classifier \citep{magge2021deepademiner}. In biomedical hedge
detection, cue taggers read as ``at least one cue'' won on biological text and direct sentence
classifiers on Wikipedia \citep{farkas2010conll}. Sentence-level
relation detection also serves as a filter before pair classification
\citep{chowdhury2013exploiting,xie2021revisiting}. We hold the teacher, the passages and the encoder fixed
and change only the question.

\paragraph{Sentence filters do not give pairs.}
Curation challenges score article triage and pair extraction separately
\citep{krallinger2008overview,islamajdogan2019overview}, and sentence-level interaction detectors find
the sentence but not the pair \citep{polajnar2011protein}. In protein-interaction corpora, accepting every
co-occurring pair has a precision equal to the number of interactions per entity pair, which falls as
sentences name more entities \citep{pyysalo2008comparative,tikk2013detailed}; an abstract filter for
microbial interactions with an AUC of 0.97 gives 25\% precision on the species pairs read off it
\citep{lim2016minter}. Aspect-level sentiment likewise reduces to the sentence-level task when sentences
name a single target \citep{jiang2019challenge}.
```

Notes: "We hold the teacher ... fixed" is true of the biodiversity arms only if ANALYSIS.md trains both arms on
teacher answers; on BioRED write "the labels, the passages and the encoder". Optional extra citations, if space
allows: andrews2002support, zeng2015distant (aggregation); laban2022summac, min2023factscore (decompose and
aggregate); hershey2021benefit (same-data control outside text); wadden2022multivers (keep the context);
hu2025decomposition, mraz2026fewshot (finer questions are not always better); jiang2011target (target-dependent
sentiment); zhong2021frustratingly, lee2020biobert (pair-conditioned inputs).

---------------------------------------------------------------------------------------------------------------

## BiotXplorer as the application: what the paper may say (public sources only)

Public sources: ruch2024biotxplorer [cited] (BISS 8:e135453, 2024, a conference abstract, full text read from the
publisher XML) and the system's public resource pages. No other BiotXplorer publication was found (web search,
2026-10-05). Full note: `lit_work/biotxplorer_note.md`.

May say, citing ruch2024biotxplorer only:
- It is a public tool that builds "pairs of species co-occurring in the same sentence together with a biotic
  interaction concept as defined in the Relation Ontology" over a literature collection served by SIB Literature
  Services (gobeill2020sibils [cited]), and shows supporting passages for each interaction.
- Its authors' own evaluations are pair-in-passage judgements: "a precision of 31% when identifying the
  interacting species" on 100 random triples, with multi-species passages as the main cause of error; and, for
  validated GloBI interactions, "85% of the returned passages confirming an interaction between the two species".
  So the application shows passages as evidence for a pair: a passage it shows must describe an interaction
  (sentence level) and concern that pair (pair level). The pair verifier gives both; a sentence filter gives only
  the first. This argument rests on the public description alone.
- Its stated next step is volume reduction by "retaining only top-ranked, and therefore most reliable, triples".
  Quote it at most; do not characterise, compare against or predict that work.

Must not say (anonymity and no overlap with the system authors' forthcoming paper):
- Nothing implying a working relationship with the system's team ("our application", "deployed in", "the team asked
  for sentence-level output", "the curator wants"). The sentence-level target must be motivated by the task and by
  public curation practice (Q2), not by what the application's owners want.
- No internals beyond the abstract: no ranking or scoring details, dictionaries, interface, usage figures, roadmap
  or upcoming publications. Our own measurements on retrieved candidates stay worded as our measurements.
- No URL of the tool pages (the DOI citation suffices). BiotXplorer is novelty-free context here: the
  application, not a contribution.

---------------------------------------------------------------------------------------------------------------

## Verification

Method. (1) Every entry was copied from the ACL Anthology .bib, Crossref (doi.org content negotiation or
api.crossref.org), PMLR, the NeurIPS proceedings or arXiv, then normalised (LaTeX accents, article numbers as pages
when Crossref or PMC gives them). (2) `python3 scripts/check_references.py paperA/reframe_extra.bib` (Semantic
Scholar title match): run 1 on 39 entries gave 38 OK, 1 NOT FOUND; run 2 on all 50 entries gave 48 OK, 0 CHECK, 2 NOT
FOUND (outputs `lit_work/check_references_run{1,2}.txt`). Each NOT FOUND was checked by hand against the publisher
record and the PDF (notes below). (3) An independent adversarial verifier re-fetched every record from its primary
source, compared every field, and re-read each claim used here in the full text (V1 = `lit_work/verify1.md`, V2 =
`verify2.md`, V3 = `verify3.md`; "main" = re-checked by the main session). (4) Test compile in /tmp with
`acl_natbib.bst`: `references.bib` + `reframe_extra.bib` give 108 bibliography items, with no BibTeX error and no
undefined citation for the draft paragraphs above.

Result: 50 entries, all verified; 3 bibliographic fixes applied (min2023factscore, hershey2021benefit,
magge2021deepademiner); 4 claims corrected before use (laban2022summac, gao2021manual, zaidan2007using,
hershey2021benefit); none deleted for lack of verification. Two entries rest on abstracts only
(herrerozazo2013ddi, gurulingappa2012development) and say so where used. Backups of the .bib before each edit:
`lit_work/reframe_extra.bib.bak_0245`, `.bak_0300`.

| Key | Source of the BibTeX | check_references | Verified by | Status | Notes |
|---|---|---|---|---|---|
| dietterich1997solving | https://doi.org/10.1016/S0004-3702(96)00034-3 | OK | V1 | VERIFIED | full text from an unofficial mirror of the published scan; bib from Crossref |
| andrews2002support | https://proceedings.neurips.cc/paper_files/paper/2002/file/3e6260b81898beacda3d16db379ed329-Bibtex.bib | OK | V1 | VERIFIED | NeurIPS proceedings give no pages |
| hoffmann2011knowledge | https://aclanthology.org/P11-1055.bib | OK | V1 | VERIFIED | deterministic-OR passage is Sec. 3, p. 543 |
| zeng2015distant | https://aclanthology.org/D15-1203.bib | OK | V1 | VERIFIED |  |
| jia2019document | https://aclanthology.org/N19-1370.bib | OK | V1 | VERIFIED | pools mention representations, not scores (wording applied) |
| ilse2018attention | https://proceedings.mlr.press/v80/ilse18a.html | OK | V1 | VERIFIED | Sec. 2.1 |
| laban2022summac | https://aclanthology.org/2022.tacl-1.10.bib | OK | V1 | VERIFIED after correction | 73.5% is the learned SummaC-Conv aggregator; claim reworded to 56.4 -> 71.2 with max/mean (Table 5) |
| verga2018simultaneously | https://aclanthology.org/N18-1080.bib | OK | V1 | VERIFIED | Sec. 2.4 |
| min2023factscore | https://aclanthology.org/2023.emnlp-main.741.bib; | OK | V1 | FIXED | author "Koh, Pang" -> "Koh, Pang Wei" (as printed on the paper) |
| chowdhury2013exploiting | https://aclanthology.org/N13-1093.bib | OK | V1 | VERIFIED | scope caveat applied: negation-based filter on rule-selected sentences |
| xie2021revisiting | https://aclanthology.org/2021.acl-long.277.bib | OK | V1 | VERIFIED | sentence stage is multi-label relation-type detection |
| liu2019event | https://aclanthology.org/N19-1080.bib | OK | V1 | VERIFIED | author order follows Anthology/Crossref/S2; the PDF title block prints a different order (Liu, Li, Zhou, Yang, Zhang) |
| krallinger2008overview | https://doi.org/10.1186/gb-2008-9-s2-s4 | OK | V2 | VERIFIED | pages = Crossref article number S4 |
| islamajdogan2019overview | https://doi.org/10.1093/database/bay147 | OK | V2 | VERIFIED, wording | pair F 37.7% (HomoloGene), 34.8% exact; RE labels on a subset of triage docs; Team 433 = binned PPI-triplet probabilities |
| krallinger2011protein | https://doi.org/10.1186/1471-2105-12-S8-S3 | OK | V2 | VERIFIED | 34 authors checked programmatically against Crossref; pages = article number S3 |
| wiegers2009text | https://doi.org/10.1186/1471-2105-10-326 | OK | V2 | VERIFIED | pages = Crossref article number 326 |
| bravo2015extraction | https://doi.org/10.1186/s12859-015-0472-9 | OK | V2 | VERIFIED | pages = Crossref article number 55; negatives are uncurated co-occurrences |
| polajnar2011protein | https://doi.org/10.1186/2041-1480-2-1 | OK | V2 | VERIFIED |  |
| lim2016minter | https://doi.org/10.1093/bioinformatics/btw357 | OK | V2 | VERIFIED, wording | "partly because"; full text read through a page-rendering tool (near-verbatim); abstract byte-checked on PubMed |
| thessen2014knowledge | https://doi.org/10.1371/journal.pone.0089550 | OK | V2 | VERIFIED |  |
| elkhettari2023building | https://aclanthology.org/2023.bionlp-1.21.bib | OK | V2 | VERIFIED |  |
| thieu2012literature | https://doi.org/10.1093/bioinformatics/bts042 | OK | V2 | VERIFIED, wording | 66-73% accuracy is abstract-level classification; full text via rendering tool |
| lee2020biobert | https://doi.org/10.1093/bioinformatics/btz682 | OK | V2 | VERIFIED |  |
| peng2019transfer | https://aclanthology.org/W19-5006.bib | OK | V2 | VERIFIED |  |
| zhong2021frustratingly | https://aclanthology.org/2021.naacl-main.5.bib | NOT FOUND (run 1), OK (run 2) | V1 | VERIFIED | check_references run 1 NOT FOUND (transient); hand-checked on Anthology 2021.naacl-main.5 and Crossref; dev sets, gold entities |
| li2019entity | https://aclanthology.org/P19-1129.bib | OK | V1 | VERIFIED |  |
| zhang2023aligning | https://aclanthology.org/2023.findings-acl.50.bib | OK | V1 | VERIFIED |  |
| zhang2025entity | https://aclanthology.org/2025.findings-naacl.224.bib | OK | V1 | VERIFIED | DocRED dev |
| jimenezgutierrez2022thinking | https://aclanthology.org/2022.findings-emnlp.329.bib | OK | V1 | VERIFIED |  |
| mraz2026fewshot | https://arxiv.org/bibtex/2606.15412 | OK | V1+main | VERIFIED | arXiv preprint only (v2, 3 Aug 2026); abs page re-read by the main session; re-check for a peer-reviewed version before camera-ready |
| jiang2011target | https://aclanthology.org/P11-1016.bib | OK | V1+main | VERIFIED | 40% of errors in a 5 queries x 20 tweets check; intro sentence p. 152 |
| jiang2019challenge | https://aclanthology.org/D19-1654.bib | OK | V1+main | VERIFIED |  |
| pyysalo2008comparative | https://doi.org/10.1186/1471-2105-9-S3-S6 | OK | V2+main | VERIFIED, wording | Tables 1-4 re-read by the main session; the 0.50-0.88 bound is our derivation |
| herrerozazo2013ddi | https://doi.org/10.1016/j.jbi.2013.07.011 | OK | V2 | VERIFIED, abstract only | kappa figures from the abstract |
| gurulingappa2012development | https://doi.org/10.1016/j.jbi.2012.04.008 | OK | V2 | VERIFIED, abstract only | no IAA number read |
| dumitrache2018crowdsourcing | https://doi.org/10.1145/3152889 | OK | V2 | VERIFIED, wording | text read in arXiv 1701.02185v2; the ACM version (TiiS 8(2)) was not readable |
| plank2022problem | https://aclanthology.org/2022.emnlp-main.731.bib | OK | V1 | VERIFIED |  |
| tikk2013detailed | https://doi.org/10.1186/1471-2105-14-12 | OK | V2 | VERIFIED | Fig. 7 values not read; quote the sentence only |
| gao2021manual | https://aclanthology.org/2021.findings-acl.112.bib | OK | V1+main | VERIFIED after correction | use fact-level DS precision 69.1%; "32% N/A" and "53%" are not DS false-positive rates (Sec. 3.1 re-read by the main session) |
| zaidan2007using | https://aclanthology.org/N07-1033.bib | OK | V3 | VERIFIED after correction | "more fruitful use of annotator time" is a hedged hypothesis; use the 88.5 -> 92.2% accuracy result |
| zhang2016rationale | https://aclanthology.org/D16-1076.bib | OK | V3 | VERIFIED | numbers are accuracy (CV means) |
| rei2019jointly | https://doi.org/10.1609/aaai.v33i01.33016916 | OK | V3 | VERIFIED | single-run test F1; FCE gain only +0.20 |
| dasanmartino2019fine | https://aclanthology.org/D19-1565.bib | NOT FOUND (run 2); hand-checked | V3+main | VERIFIED | check_references NOT FOUND: Crossref/S2 list "...News Article" and pp. 5635-5645; the PDF and the Anthology print "Articles", pp. 5636-5646, as in the bib |
| farkas2010conll | https://aclanthology.org/W10-3001.bib | OK | V3 | VERIFIED | uncontrolled shared-task comparison |
| li2021dual | https://doi.org/10.1109/cvpr46437.2021.01409 | OK | V3 | VERIFIED | IEEE pages 14313-14323 (the CVF open-access copy prints 14318-14328) |
| hershey2021benefit | https://doi.org/10.1109/ICASSP39728.2021.9414579 | OK | V3 | FIXED | author "Channing Moore, R" (Crossref) -> "Moore, R. Channing"; claim restricted to Strong-67k vs Diffuse-67k (0.96 vs 0.82) |
| magge2021deepademiner | https://doi.org/10.1093/jamia/ocab114 | OK | V3 | FIXED | title braces {DeepADEMiner}, {T}witter; "any span found" is our reading |
| wadden2022multivers | https://aclanthology.org/2022.findings-naacl.6.bib | NOT FOUND (run 2); hand-checked | V3 | VERIFIED | check_references NOT FOUND (S2 rate limit / braces); Anthology and Crossref match; result in Table 4 |
| hu2025decomposition | https://aclanthology.org/2025.naacl-long.320.bib | OK | V3 | VERIFIED | quote in Sec. 4.2 |
| pavlopoulos2022from | https://aclanthology.org/2022.acl-long.259.bib | OK | V3 | VERIFIED, wording | close to the BiLSTM tagger only, not SpanBERT (63.0) |

Checked and deliberately left out (exist, but not needed or too weak; NOT in `reframe_extra.bib`, and not
independently re-verified, so re-verify before any use): dekok2018review (its review-level win reverses under
cross-validation, V3); foulds2010review (main; Dietterich and Andrews suffice); Labonte et al. 2026 (Springer CCIS,
abstract only); and, from the finders' notes, lin2016neural, surdeanu2012multi, bunescu2007learning, kamoi2023wice,
wang2018revisiting, zheng2021prgc, takamatsu2012reducing, zhu2020towards, jia2019arnor, leitner2010overview,
abdelmageed2022biodivnere, ding2001mining, scheepens2024large, vanmulligen2012euadr, pyysalo2007bioinfer,
hripcsak2005agreement, pavlick2019inherent, airola2008allpaths, liu2016drugdrug, segurabedmar2013semeval,
ye2022packed, chen2022biomedical, li2023revisiting, rehana2024evaluating, chen2025benchmarking,
wadhwa2023revisiting, press2023measuring, pislar2020seeing, schmaltz2016sentence, tang2024minicheck,
xie2023empirical, li2018thoracic, vanwinckelen2015instance, rathore2022pare, wanner2024closer. Their BibTeX and
notes are in `lit_work/agent{A,B,C}.{md,bib}` and `lit_work/novelty_attack.md`.

Remarks on already-cited works, from this stage's reading:
- riedel2010modeling: its text says 13% for Wikipedia, but its own Table 1 averages 16.7% (20/20/10); quote only the
  NYT 31%, which matches the table.
- keck2025extracting: its "accuracy of 89.5%" is 229/(229+27), a precision, as its own abstract says.
- luo2022biored: IAA 77.91% (relations), 97.01% (entities), 85.01% (novelty), Table 4.
- rosenman2020exposing: 57% of its challenge-set sentences hold conflicting pair labels; the set was sampled from
  model-positive sentences, so it is not a base rate.
