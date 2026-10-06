# Main-session finds (outside the three agents' questions), 2026-10-05 02:20

## Target-conditioned vs target-independent classification (analog from sentiment analysis)
- jiang2011target, Jiang, Yu, Zhou, Liu, Zhao. "Target-dependent Twitter Sentiment Classification". ACL-HLT 2011,
  pp. 151-160, Anthology P11-1016 (no DOI). Verified: aclanthology.org/P11-1016.bib and PDF.
  SHOWS [full text, p.151 Intro and Sec. 6.1 p.155]: target-independent classifiers give every target in a tweet
  the same answer; in a manual check of a deployed tweet-sentiment service, the error type where "sentiments in
  some tweets are classified correctly but the sentiments are not truly about the query" makes up "about 35% and
  40% of the total errors, respectively" (35% = neutral tweets classified subjective; 40% = sentiment not about
  the query target). Adding target-dependent features raises subjectivity accuracy 63.8 -> 68.2 (Table 1).
  RELATION: the same structural argument (a whole-text classifier cannot tell which target a judgment is about)
  made and measured in another field in 2011; we must cite it as precedent, not claim the argument.
- jiang2019challenge, Jiang, Chen, Xu, Ao, Yang. "A Challenge Dataset and Effective Models for Aspect-Based
  Sentiment Analysis" (MAMS). EMNLP-IJCNLP 2019, pp. 6280-6285, doi 10.18653/v1/D19-1654. Verified: Anthology .bib
  and PDF.
  SHOWS [full text, abstract; Intro p.6280; Sec. 4.2]: "most sentences contain only one aspect or multiple aspects
  with the same sentiment polarity, which makes ABSA task degenerate to sentence-level sentiment analysis";
  "sentence-level sentiment classifiers (TextCNN and LSTM) achieve competitive results on SemEval-14 Restaurant
  Review dataset but perform poorly on MAMS datasets" (every MAMS sentence has >= 2 aspects with different
  polarities).
  RELATION: direct precedent for our BioRED localisation (no pair gain on sentences naming two entities, gain on
  sentences naming more): a target-conditioned task collapses to the sentence task exactly when sentences have
  one target. Supports our measurement; pre-empts any claim that this localisation is a new phenomenon.

## Multi-instance learning: the standard assumption
- foulds2010review, Foulds, Frank. "A review of multi-instance learning assumptions". The Knowledge Engineering
  Review 25(1):1-25, 2010, doi 10.1017/S026988890999035X. Verified: Crossref content negotiation + Semantic
  Scholar (DBLP journals/ker/FouldsF10); full text read from the first author's preprint PDF
  (jfoulds.informationsystems.umbc.edu/FouldsAndFrankMIreview.pdf).
  SHOWS [full text, Sec. 1 p.2 and Sec. 2.4]: "Under this assumption, each instance has a hidden class label ...
  a bag is ... positive if and only if it contains at least one positive instance" (the standard MI assumption,
  from Dietterich et al. 1997), and that it "is not guaranteed to hold in other domains".
  RELATION: our pair-max sentence decision is the standard MI assumption applied with a sentence as the bag and its
  candidate pairs as instances. Difference: our instance labels are not hidden (the teacher labels pairs), so this
  is supervised instance classification plus the standard aggregation rule, not MI learning. Cite for the rule;
  never present the max as ours. Note the rule holds by construction for our SENTENCE definition only if every
  interacting pair of the passage is among the scored pairs (enumeration recall; see ANALYSIS weakness 5).
