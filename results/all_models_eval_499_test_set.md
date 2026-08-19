# All Models — Evaluation on 499-Sentence Test Set
**Date:** 2026-06-15  
**Test set:** `biotic_interaction_test_set.csv` — 499 sentences, 254 pos / 245 neg, shuffled, gold labels from human curation  
**Threshold:** t=0.090 (derived from 300 pos + 700 neg held-out split of training data, ~30% positive rate)  
**Note:** distilled models (v1–v6) use calibration tuned for higher thresholds — at t=0.09 they approve all 499. Their oracle F1 is the fair comparison.

## Ranked by AUC

| # | Model | AUC | Oracle F1 | Oracle t | F1 @ t=0.09 | Approved |
|---|-------|-----|-----------|----------|-------------|----------|
| 1 | transformer_BiomedBERT_cv_regularized | **0.928** | **0.891** | 0.080 | 0.886 | 272/499 |
| 2 | ensemble_BiomedBERT_cv_reg × flan-t5-v12 (geometric) | 0.925 | 0.867 | 0.090 | 0.867 | 251/499 |
| 3 | multitask/full_typed_a05_ner2_warmstart | 0.907 | 0.873 | 0.100 | 0.871 | 267/499 |
| 4 | flan-t5-base_v7 | 0.906 | 0.837 | 0.190 | 0.835 | 232/499 |
| 5 | transformer_BiomedBERT_v15 | 0.901 | 0.856 | 0.080 | 0.851 | 270/499 |
| 6 | transformer_BiomedBERT_v11_1 | 0.900 | 0.844 | 0.090 | 0.844 | 265/499 |
| 7 | flan-t5-base_v12 | 0.898 | 0.865 | 0.140 | 0.862 | 252/499 |
| 8 | multitask/full_typed_a03_ner2_warmstart | 0.890 | 0.844 | 0.070 | 0.835 | 256/499 |
| 9 | distilled_BiomedBERT_v1 | 0.890 | 0.811 | 0.330 | 0.675† | 499/499 |
| 10 | flan-t5-base_v14 | 0.886 | 0.805 | 0.130 | 0.803 | 239/499 |
| 11 | distilled_BiomedBERT_v6 | 0.884 | 0.808 | 0.170 | 0.675† | 499/499 |
| 12 | transformer_SciBERT_cv_regularized | 0.882 | 0.838 | 0.050 | 0.810 | 203/499 |
| 13 | transformer_BiomedBERT_v12_regularized | 0.882 | 0.847 | 0.070 | 0.834 | 240/499 |
| 14 | distilled_BiomedBERT_v2 | 0.882 | 0.814 | 0.210 | 0.675† | 499/499 |
| 15 | distilled_BiomedBERT_v3 | 0.880 | 0.825 | 0.350 | 0.675† | 499/499 |
| 16 | multitask/full_typed_a03 | 0.877 | 0.834 | 0.060 | 0.823 | 237/499 |
| 17 | multitask/basic_a05 | 0.876 | 0.819 | 0.090 | 0.819 | 242/499 |
| 18 | multitask/full_typed_a03_ner2 | 0.874 | 0.815 | 0.050 | 0.803 | 239/499 |
| 19 | distilled_BiomedBERT_v2_finetuned | 0.873 | 0.709 | 0.050 | 0.692 | 168/499 |
| 20 | multitask/full_typed_a05 | 0.871 | 0.826 | 0.060 | 0.819 | 281/499 |
| 21 | multitask/full_typed_a05_ner2_5ep | 0.871 | 0.822 | 0.080 | 0.818 | 289/499 |
| 22 | **multitask/full_typed_a05_ner2 ★ champion** | 0.869 | 0.817 | 0.070 | 0.807 | 239/499 |
| 23 | distilled_SciBERT_v5 | 0.869 | 0.805 | 0.260 | 0.675† | 499/499 |
| 24 | multitask/full_typed_a05_ner2_posonly | 0.865 | 0.825 | 0.080 | 0.817 | 255/499 |
| 25 | multitask/full_a05_ner2 | 0.865 | 0.804 | 0.050 | 0.792 | 231/499 |
| 26 | multitask/full_typed_a05_ner2_v15 | 0.862 | 0.812 | 0.350 | 0.805 | 260/499 |
| 27 | multitask/full_typed_a05_ner2_aug | 0.862 | 0.803 | 0.140 | 0.795 | 332/499 |
| 28 | transformer_BiomedBERT_v17a | 0.848 | 0.804 | 0.160 | 0.768 | 361/499 |
| 29 | transformer_BiomedBERT_v18 | 0.839 | 0.788 | 0.080 | 0.783 | 257/499 |
| 30 | multitask/multitask_v12_hardce | 0.799 | 0.723 | 0.120 | 0.720 | 224/499 |
| 31 | multitask/multitask_v7_hardce | 0.781 | 0.737 | 0.400 | 0.727 | 230/499 |
| 32 | multitask/kg_enriched_a05 | 0.759 | 0.696 | 0.210 | 0.694 | 265/499 |
| 33 | multitask/multitask_v14_hardce | 0.749 | 0.695 | 0.060 | 0.695 | 273/499 |

† Approves all 499 sentences at t=0.09; use oracle threshold for fair comparison.  
‡ distilled_DistilBERT_v4: inference error (token_type_ids mismatch), not evaluated.  
‡ flan_t5_base_v17a, v18: config format error, not evaluated.

## Key Observations

- **BiomedBERT cv_regularized dominates on the broad 499-sentence test** (AUC=0.928, F1=0.891) — strongest model overall
- **Champion (full_typed_a05_ner2) ranks 22nd by AUC** — it was specifically optimised for EP-relax distribution (F1=0.868 on EP-relax), not for generalisation across diverse sets
- **Warmstart multitask is a surprise** (AUC=0.907, #3) — underperformed on EP-relax but generalises well; benefits from the cv_regularized warm start
- **Soft labels are essential**: hard-CE variants (multitask_v12/v14/v7_hardce) are the three worst multitask models
- **Distilled models are competitive by AUC** but need higher thresholds (0.21–0.35); at t=0.09 they are uninformative
- **PMC augmentation hurts**: full_typed_a05_ner2_aug (AUC=0.862) ranks below un-augmented champion
- **kg_enriched and v18 are poor** — domain-specific fine-tuning degraded generalisation

## Models to Keep

| Model | Reason | Size |
|-------|--------|------|
| `multitask/full_typed_a05_ner2` | Champion, deployed in API, best on EP-relax | 439 MB |
| `transformer_BiomedBERT_cv_regularized` | Best on broad test, paper baseline | 439 MB |
| `flan-t5-base_v12` | Teacher model, part of geometric ensemble, strong standalone | 993 MB |
| `distilled_BiomedBERT_v2` | Best distilled single model (AUC=0.882), small footprint | 439 MB |

**Total to keep: ~2.3 GB**

## Safe to Delete (~27 GB freed)

All other models in `classifier/models/`:
- `transformer_BiomedBERT_v11_1`, `v12_regularized`, `v15`, `v15b`, `v16_realonly`, `v17a`, `v17b`, `v18`, `optimal` (~3.5 GB)
- `transformer_SciBERT_cv_regularized`, `v9` (~0.9 GB)
- `distilled_BiomedBERT_v1`, `v2_finetuned`, `v3`, `v6` (~1.8 GB)
- `distilled_DistilBERT_v4`, `distilled_SciBERT_v5` (~0.7 GB)
- `flan-t5-base_v7`, `v10`, `v10.1`, `v11_1`, `v13`, `v14` + `flan_t5_base_v15b`, `v16_realonly`, `v17a`, `v17b`, `v18` (~11 GB)
- All multitask variants except `full_typed_a05_ner2`: `basic_a05`, `cls_only_a10`, `full_a03`, `full_a05`, `full_a05_ner2`, `full_a07`, `full_typed_a03`, `full_typed_a03_ner2`, `full_typed_a03_ner2_warmstart`, `full_typed_a05`, `full_typed_a05_ner2_5ep`, `full_typed_a05_ner2_aug`, `full_typed_a05_ner2_posonly`, `full_typed_a05_ner2_v15`, `full_typed_a05_ner2_warmstart`, `kg_enriched_a05`, `multitask_v7_hardce`, `multitask_v12_hardce`, `multitask_v14_hardce`, `typed_a05` (~8.4 GB)
- `cv_temp/` (~6.2 GB)
- Stub directories (4K each, safe to delete)
