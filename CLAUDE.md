# Classifier - Biotic Interaction NLP Pipeline

## Structure
- src/models/     -- Model training (transformer_classifier.py is main entry)
- src/data/       -- Data processing, validation, LLM-based cleaning
- src/utils/      -- Prediction, evaluation helpers
- src/annotation/ -- Interactive annotation tool (annotator.py)
- api/            -- FastAPI endpoints (fastapi_ensemble.py on port 8001)
- validator/      -- Interaction validator (LLM + heuristic fallback)
- tests/          -- pytest suite (test_training_data.py: 5-gate quality checks)
- scripts/        -- Training scripts, dataset builders, analysis tools

## Testing
Run quality gates: python -m pytest tests/test_training_data.py -v
Gates: Species validation, ROBI rules, template validation, negative quality, balance

## Key Training Pipeline
1. Build dataset: scripts/build_v14_dataset.py (latest) or scripts/build_v12_dataset.py
2. Train (discriminative): python scripts/train_cv_regularized.py --model BiomedBERT --epochs 5
3. Train (generative): python src/models/flan_t5_classifier.py --train data/training/training_data_v14.csv --epochs 5
4. Evaluate all EP sets: python scripts/evaluate_all_ep.py
5. Deploy original API: bash start_api.sh (port 8001)
6. Deploy enriched pipeline: bash start_pipeline_generative.sh (port 8002, FLAN-T5 + NER + GloBI)

## Available Models
**Discriminative:** transformer_classifier.py (BiomedBERT/SciBERT), bert_classifier.py, svm_classifier.py,
random_forest_classifier.py, luke_classifier.py, ensemble_classifier.py, optimized_ensemble.py
**Generative:** flan_t5_classifier.py, flan_t5_enriched.py (structured seq2seq)
**Other:** hyperbolic_embeddings.py, relation_extractor.py, biogpt_classifier.py

## Data Validation Chain
taxonomy_validator.py -> robi_validator.py -> quality_filter.py -> llm_validator.py

## Models Directory
Git-ignored. Contains trained model checkpoints. Do not attempt to read or modify directly.
