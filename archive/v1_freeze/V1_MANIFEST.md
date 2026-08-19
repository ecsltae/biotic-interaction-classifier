# Classifier V1 — frozen architecture of record

Frozen **2026-08-20**. Git commit at freeze: `a8c09b5ca4e0e5b7b18474932ea4f08eae0ddfbb` (tag `classifier-v1`).

This is the architecture audited on 2026-08-19. It is preserved verbatim so every V2 claim can be
compared against it by re-running, not by trusting a recorded number.

Checkpoints are git-ignored; physical copies live in `archive/v1_freeze/models/`.

## Architecture

| | |
|---|---|
| Champion | `models/multitask/full_typed_a05_ner2` — MultiTaskBiomedBERT (cls + NER heads) |
| Encoder | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |
| Recipe | `--ner-scheme full_typed --pretrain-ner-epochs 2 --alpha 0.5 --epochs 3`, T=2 soft-label distillation |
| Training data | `data/training/distillation_soft_labels.csv` |
| Trainer | `experiments/multitask/train.py` (no seed — see V1 defects) |
| Ensemble | `transformer_BiomedBERT_cv_regularized` × `flan-t5-base_v12`, geometric |

## Checkpoints

| model | size | sha256 (pytorch_model.bin) |
|---|---|---|
| `multitask/full_typed_a05_ner2` | 419M | `7c96dcbcfee0a384b41d8d73f2b6bf90…` |
| `distilled_BiomedBERT_v2` | 419M | `d2c95dbf959475b959ee80e8c5b9b6d3…` |
| `transformer_BiomedBERT_cv_regularized` | 419M | `425d41ad6f78645be6ddb40241e37d34…` |
| `transformer_BiomedBERT_v16` | 419M | `81f423ba34b935d2c5943931758bc7c2…` |
| `flan-t5-base_v12` | 947M | `7a6b437ac219f21ed3e8ae86d6718788…` |

## Data files as frozen

| file | rows | sha256 |
|---|---|---|
| `data/training/distillation_soft_labels.csv` | 50194 | `a62df36c6954dd8a546ae3ef4f551e91…` |
| `data/training/distillation_44k.csv` | 44331 | `4e5f24cb71ce0ed4165e942869694a3d…` |
| `data/training/training_data_v14.csv` | 35013 | `94c1cdfcc784aac51686cb908ce71a08…` |
| `data/training/training_data_v16.csv` | 44153 | `4bd1200dc347be022e8e1f5943e8419f…` |
| `data/evaluation/biotic_interaction_test_set.csv` | 500 | `b5747df79c9b4a37e42fb83228c46f1d…` |
| `globi-relax_passages-triplets_2024-02-28_curation_EP.tsv` | 100 | `615a8a2971b93f476fe3669d8e74840e…` |
| `data/evaluation/eval_100.tsv` | 100 | `a23e1f028f00d1702c523e311f2eb764…` |

## V1's claimed performance, and what it is actually worth

Recorded here as claims, not as facts. The audit of 2026-08-19 established that every one of these
was produced by selecting the decision threshold, the CV fold, and the config on the same data the
metric is reported on.

| claim | source | status |
|---|---|---|
| EP-relax F1 = 0.868 (champion) | `results/multitask/full_typed_a05_ner2/ep_relax_eval.json` | threshold fitted on EP-relax; n=100 |
| "beats ensemble by +0.011" | MEMORY.md | McNemar p = 1.00; models disagree on 5/100 sentences |
| "0.868 is the hard ceiling" | MEMORY.md | unsupported |
| 500-row F1 = 0.8743 | `results/new_testset/corrected_testset_results.json` | **ties** `template_trained` 0.8746, McNemar p = 0.7604 |
| AUC rank | `results/all_models_eval_499_test_set.md:56` | champion 22nd of 30 |

## Known defects frozen into V1

Carried here so V2 can be checked against each one.

- No seed in `experiments/multitask/train.py` — two identical runs scored 0.8743 vs 0.8346 (spread 0.040).
- Trained against `softmax(logits/2)`, read at T=1 at all 17 inference sites. EP-relax ECE = 0.156.
- `tier2_triple_extractor.py:56` `CLS_THRESHOLD = 0.13` is the EP-relax F1-argmax — test set leaking into production.
- KD loss: `distill_ensemble.py:193-195` softmaxes a probability pair; `train.py:38-63` applies T to the student only.
- Checkpoint selection on `cls_f1` at threshold 0.5, which is near-degenerate (val range 0.379-0.410, Spearman vs EP F1 = 0.027).
- NER head emits zero HOST and zero PATHOGEN — Tier 2 produced 0 triples over 34,880 sentences.
- `models/multitask/mt_distill_warm_ner0`, `mt_distill_warm_ner2`, `mt_cold_start_champion`, `mt_hardce` were **deleted** before this freeze; `corrected_testset_results.json` is therefore not reproducible from V1 artifacts.

Full audit: 15 validity findings, 22 high-leverage, 23 cleanups — see `docs/AUDIT_2026-08-19.md`.
