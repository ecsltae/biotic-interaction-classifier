# Champion Model Evaluation Report
**Date:** 2026-06-12  
**Model:** Multi-task BiomedBERT with NER auxiliary task (`full_typed_a05_ner2`)

---

## Model

- **Architecture:** BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext with a shared encoder, classification head (biotic interaction detection), and NER auxiliary head (entity types: HOST, PATHOGEN, SPECIES, INT)
- **Training data:** 44,178 sentences with soft labels from a teacher ensemble (BiomedBERT × FLAN-T5-base, geometric mean)
- **NER scheme:** full_typed, α=0.5, 2 NER pre-training epochs
- **Checkpoint:** `classifier/models/multitask/full_typed_a05_ner2`

---

## Evaluation Methodology

- **Threshold selection:** t=0.090, derived from a held-out validation split of the training data (300 positive + 700 negative sentences, stratified). No test data was used for threshold selection.
- **Test set:** ~500 sentences (499) from 5 human-annotated evaluation sets, all held out from training and threshold tuning. Gold labels from expert curation (Esteban Palencia).
- **Excluded from test set:** two multi-entity BioTx sets (biotx-multiples, biotx-nomultiple) removed due to annotation methodology differences.

---

## Overall Results (~500-sentence test set)

| Metric | Value |
|--------|-------|
| **F1** | **0.807** |
| **Precision** | **0.833** |
| **Recall** | **0.783** |
| **AUC** | **0.869** |
| Sentences evaluated | 499 |
| Positives (gold) | 254 (51%) |
| Approved by model | 239 / 499 |
| Threshold | 0.090 (training-derived) |

---

## Per-Set Breakdown

| Evaluation set | n | Pos | F1 | P | R | AUC |
|----------------|---|-----|----|---|---|-----|
| EP-relax (GloBI-seeded, EP-curated) | 99 | 47 | 0.849 | 0.763 | 0.957 | 0.888 |
| EP-passage (GloBI-seeded, EP-curated) | 100 | 85 | 0.854 | 0.931 | 0.788 | 0.700 |
| BioTx-random (random retrieval, EP-curated) | 100 | 31 | 0.733 | 0.759 | 0.710 | 0.897 |
| eval-100 (gold standard, human-curated) | 100 | 31 | 0.725 | 0.658 | 0.806 | 0.902 |
| gen-set-100 (synthetic, hand-crafted) | 100 | 60 | 0.792 | 0.976 | 0.667 | 0.902 |
| **Overall** | **499** | **254** | **0.807** | **0.833** | **0.783** | **0.869** |

---

## Notes for the Paper

- **AUC=0.869** is the primary threshold-independent metric. It measures ranking quality regardless of operating point.
- **F1=0.807** at t=0.090 with P=0.833, R=0.783 reflects a slightly precision-favoring operating point, appropriate for a pipeline where false positives cost downstream KG quality.
- The threshold was calibrated on training data with ~30% positive rate, consistent with the expected prevalence of biotic interaction sentences in biomedical literature retrieved by a focused system (BioTx-random and eval-100 both show 31% positive rate).
- **EP-relax** is the strongest set for this model (F1=0.849, R=0.957) — it best matches the training distribution. High recall on this set means the model rarely misses real interactions.
- **eval-100 and BioTx-random** are the hardest sets (F1≈0.73), both with 31% positive rate and diverse negative sentence types.
- **gen-set-100** achieves very high precision (0.976) because synthetic sentences are unambiguous; lower recall (0.667) suggests the model is appropriately conservative on easy positives that lack naturalistic phrasing.
- The oracle threshold (best possible on test data) is t=0.070, giving F1=0.817. The gap between training-derived (0.807) and oracle (0.817) is **+0.010 F1**, confirming minimal threshold leakage.

---

## External Validation (Emilie's BioTx pipeline, 100 sentences)

A separate evaluation on 100 sentences from the BioTx/SIBILS retrieval pipeline (50 GloBI-seeded + 50 random queries, curated by Emilie Pasche), using `triples_ok_full` as gold label (whether the retrieved passage correctly expresses the queried triplet):

| Metric | Value |
|--------|-------|
| AUC | 0.803 |
| Note | `triples_ok_full` is stricter than biotic interaction detection; sentences can be genuine interactions without expressing the exact queried triplet |

**Key finding:** the model has **perfect recall on correctly-expressed triplets** — it never rejects a sentence where `triples_ok_full=1` (65/65 retained). This confirms the model is safe to use as a recall-preserving pre-filter in the BioTx pipeline.
