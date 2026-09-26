# BiotXplorer classifier — v2 handoff

**For:** review of the v2 design
**From:** E. Gaillac · 21 Sep 2026
**Status of the numbers:** first measured recall for this pipeline. Two 50-row curated samples;
intervals are wide and stated everywhere.

---

## The two questions I'd like your read on

**01 — Should v2 output direction?**
Today the pipeline emits an unordered pair plus a relation. Nothing downstream knows whether the
tick bit the rabbit or the reverse. The ontology can express direction; our gold data cannot.

**02 — Where does better training data come from?**
Every augmentation tried so far has hurt. We are stuck at F1 0.868 on EP-relax and I think the
data, not the architecture, is the binding constraint.

---

## What the system does

BiotXplorer mines species interaction triples from the biomedical literature. Rule-based extraction
(ROBI) proposes candidate `(taxon1, relation, taxon2)` triples from MEDLINE passages; a BERT-family
classifier then decides which to keep. Output feeds a knowledge graph used for ecological
plausibility checks on eDNA samples.

On the MEDLINE 2025 run (`sibils_eval.med25_r1_v_ep`, 38.2M documents):

| stage | triples |
|---|---|
| ROBI rule extraction — candidates proposed | 103,122 |
| **kept by the classifier** — what BiotXplorer publishes | **43,708** |
| **rejected** — the set we had never looked at until now | **59,414** |

The rejected set exists only as a *difference between two MongoDB collections* — there is no
"rejected" table. That is why recall had never been measured: precision was computable from the
kept side alone, recall was not.

---

## Where it stands

Triplet level, from two 50-row curated samples:

| | value | 95% CI |
|---|---|---|
| **precision** | **0.49** | 0.35 – 0.62 |
| **recall** | **0.62** | 0.48 – 0.75 |
| **F1** | **0.55** | 0.41 – 0.65 |

|  | true interaction | not | row total |
|---|---|---|---|
| **kept** (predicted +) | **TP 20,980** | FP 22,728 | 43,708 |
| **rejected** (predicted −) | **FN 13,071** | TN 46,343 | 59,414 |
| column total | **34,051** | 69,071 | 103,122 |

Precision comes from Émilie's curation of 50 random kept triples (0.48), stratum-weighted with her
50 GloBI-matched triples (0.82 — but those are only 4.1% of the pool, so they barely move it).
Recall comes from curation of 50 uniformly-sampled rejected triples, completed 17 Sep.

### Two caveats that must travel with these numbers

**This measures BiotXplorer end-to-end, not the classifier.** 34% of correct rejections failed
because the entity layer produced a nonsense taxon — the classifier never had a chance at them. A
blinded sentence-level sheet that isolates the model exists but is not yet annotated.

**Recall is relative to the ROBI candidate set.** Interactions the rules never proposed are
invisible to both samples, so 0.62 is an upper bound on recall against the literature.

---

## Why the rejects fail

All 50 curated rejects, broken down by *which layer* failed:

| failure | n | owner | what it looks like |
|---|---|---|---|
| **Pairing** — both taxa correct, but the sentence asserts no relation *between them* | 20 (40%) | rules | Co-listed taxa read as a pair: *"P. infestans, and … A. solani"* — two pathogens compared, neither a pathogen of the other |
| **Entity** — a slot holds something that is not that taxon | 17 (34%) | NER | "policemen" → *Coeliadinae* (a butterfly); "comet" → *Calloplesiops altivelis*; "cerium" → genus *Cerium* |
| **False negative** — the triple is fully correct and was discarded | 11 (22%) | **classifier** | "Alphaviruses … mainly transmitted by arthropods" |
| **Predicate** — right pair, wrong relation label | 2 (4%) | rules | *"M. tuberculosis-infected human monocytes"* → labelled `regulates`, the term scraped from a different clause |

**Three quarters of the rejections are correct for reasons that have nothing to do with the model.**
That reframes where v2 effort pays off: a classifier improvement can only address the 22%, while a
rules or NER fix addresses the 74%.

---

## Question 01 — direction

Short version: **the ontology supports direction, the storage layer destroys it, and no gold data
records it.** Adding a direction head to v2 requires new annotation before it requires new
modelling.

### The vocabulary already has inverse pairs

119 distinct interaction concepts appear in the baseline, several as directed inverses:

| relation | inverse | triples in baseline |
|---|---|---|
| `preys on` | `preyed upon by` | 3,589 / 2,403 |
| `parasite of` | `parasitized by` | 3,051 / 2,377 |
| `host of` | `pathogen of` | 4,161 / 11,258 |
| `pollinates` | `pollinated by` | 1,230 / — |

So direction is expressible today without extending the ontology — a directed edge is just the
choice between a relation and its inverse, given an ordered pair.

### But the order is not preserved

The stored key is `taxon1;taxon2;interaction`, and **taxon order is not stable between pipeline
runs.** Diffing two collections on the raw key produced 29,105 phantom rows that looked "kept but
absent from the baseline" — impossible for a pure filter. Sorting the two taxon IDs before
comparison made the kept set a clean subset. Any direction signal the rules may once have had does
not survive storage.

### And the gold standard explicitly ignores it

Émilie's curation does not penalise reversed triples. Two of her rows settle it:

- *Breinlia booliati* --[host of]--> *Rattus*, where the sentence says *Rattus* are hosts of
  *B. booliati* → scored fully correct.
- *Demidospermus* --[parasitized by]--> *Ageneiosus inermis*, where *Demidospermus* is the parasite
  → scored fully correct.

That was the right call given unstable storage — penalising an artefact of serialisation would only
add noise. But it means **every label we hold is direction-blind**, including the 50 just curated.
A v2 direction head has no supervision to train on and no test set to be measured against.

> **What I'd like your view on.** Is direction worth a re-annotation pass, or is the unordered pair
> sufficient for the knowledge-graph use case? If it is worth it: does direction belong as a third
> output head on the classifier, as a constrained choice between inverse relation terms, or as a
> separate downstream model over an already-accepted triple? My instinct is the middle option — it
> reuses the ontology structure and keeps the label space closed — but I have not thought hard about
> the failure modes.

---

## Question 02 — training data

The model is at **F1 0.868 on EP-relax** (multi-task BiomedBERT, `full_typed_a05_ner2`). Every
attempt to push past it by adding data has regressed:

| attempt | result | read |
|---|---|---|
| v7 template data | 0.756 | Templates match EP-relax surface patterns; real sentences don't. Poisonous in the mix. |
| PMC harvest, pos + neg (178 Qwen-validated) | −0.024 | Domain shift — HIV/HBV-specific phrasing narrows the distribution |
| PMC harvest, positives only | −0.042 | Worse. Threshold collapsed to 0.08. |
| Fine-tuning distilled v2 on v18 | 0.808 → 0.617 | Catastrophic in 3 epochs |

The working conclusion has been that the 44K soft-label distillation set is optimally calibrated for
EP-relax and any domain-specific augmentation disrupts it. The recall study gives that a sharper
reading: **we have been augmenting along the wrong axis.** Every attempt added more sentences of the
kind the model already sees. The false negatives are not a topic gap — they are a *phrasing and
framing* gap.

### What the 11 false negatives have in common

| cluster | rows | character |
|---|---|---|
| **Topic-shifted** — the interaction is mentioned while the sentence is about something else | R003, R011, R016, R032, R033 | A quasicrystal-modelling paper that mentions viruses "contain a number of important human pathogens" |
| **Nominalised** — the relation is a noun, not a finite verb | R006, R022, R030, R036, R044 | "prey capture", "pack-hunting predator", "during infestation", "trophic interactions" |
| **Canonical** — no obvious reason for the miss | R047 | "Alphaviruses … are mainly transmitted by arthropods". Both taxa high-rank. |

The last row is the one that bothers me. R047 is a textbook vector-transmission statement in the
plainest possible phrasing, and it was rejected. A competing explanation for several of these is
**taxonomic rank**: three of the four high-rank pairs scored are false negatives, and the training
data is predominantly species-level. That is cheap to test — score the champion model over the
high-rank rows versus the rest and compare — and it implies a different fix (rank-balanced sampling)
from the linguistic one.

### The reject pool as a data source

The 59,414 rejected triples are the obvious mine, and they are unusual in being *pre-filtered for
difficulty*:

- **~13,000 estimated false negatives** — real positives the model already fails on, i.e. exactly
  the hard cases augmentation has failed to supply.
- **~46,000 true negatives** — hard negatives that survived the rules but are genuinely not
  interactions. The corpus has never had a principled negative source; these come free.
- Mining them needs a cheaper labeller than full human curation. Opus 5 zero-shot reached 55%
  agreement at sentence level but only 28% at triple level, failing systematically on pair
  specificity — the same failure the classifier has. Entity-prepended prompting is the obvious next
  try and is untested.

> **What I'd like your view on.** Is mining the reject pool the right move, or does training on the
> model's own errors risk the same calibration collapse we saw with PMC harvest? And if the false
> negatives really are a phrasing gap rather than a topic gap, is synthetic nominalisation
> augmentation (rewriting existing positives from verbal to nominal form) a defensible way to get
> there, or too artificial given what templates did to us?

---

## Test set

The 50 curated rejects are now a test set: `data/evaluation/biotx_rejected_50_testset.csv`.
11 positive, 39 negative, every row carrying the component labels and an `entity_prepended` field in
the v16 format.

**It measures recall only.** Every row was rejected by the classifier, so the set contains no true
positives and cannot produce a precision figure on its own. Pair it with Émilie's 100 kept rows for
a balanced evaluation, and keep the two provenances distinct — hers are triple-sampled, mine
passage-sampled, though all 50 happened to be distinct triples so the rates are comparable.

---

## Files

| path | what |
|---|---|
| `data/evaluation/biotx_rejected_50_testset.csv` | the new test set |
| `data/evaluation/biotx_retrieval_eval_100.csv` | Émilie's kept-side curation (precision) |
| `data/evaluation/biotx_recall/` | full reject pool, rubric, scoring scripts, convention log |
| `data/evaluation/biotx_recall/sentence_level_blind_200.csv` | blinded classifier-only sheet, unannotated |
| `WORKLOG.md` | running record, append-only |

### Reproducing the pools

MongoDB `sibils_eval` on `sibils-mongodb.lan.text-analytics.ch:27017`.
Baseline `med25_r1_v_ep_robirules_corr_*`, kept `med25_r1_v_ep_classifier_2026_testfull2_*`,
rejected = the difference. **Sort the two taxon IDs in `triplet_key` before diffing** or the result
is wrong by 30%.

---

## Open items

- **Two annotation conventions still unsettled** — whether "the effect of X on Y was studied"
  asserts a relation (currently no), and whether co-infection of a shared host counts (currently yes
  at pair level). Logged in `data/evaluation/biotx_recall/convention_questions_emilie.md`.
- **The blinded sentence-level sheet is unannotated.** Until it is, we cannot separate classifier
  recall from pipeline recall, and the 0.62 will be read as the model's number when it is not.
- **R022 carries an inconsistent label** (`ok_full=1` with `ok_species=0`, which the rubric's nesting
  forbids). Moves nothing material; worth fixing before the set is reused.
- **A same-clade filter would remove 5% of candidates for free** — pairs where one taxon's ancestor
  set is a subset of the other's (*Xanthomonas citri* × *X. arboricola*). All correct rejections, so
  it lifts precision upstream at no recall cost.

---

*Evaluation run 5–17 Sep 2026. Curation by E. Gaillac; kept-side curation and annotation conventions
by Émilie.*
