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
1. Build dataset: scripts/build_hybrid_dataset.py
2. Train model: src/models/transformer_classifier.py --model BiomedBERT --epochs 5
3. Evaluate: src/utils/prediction.py --model biomedbert --input data/evaluation/eval_100.tsv
4. Deploy API: bash start_api.sh (uvicorn on port 8001)

## Available Models
transformer_classifier.py, bert_classifier.py, svm_classifier.py,
random_forest_classifier.py, luke_classifier.py, ensemble_classifier.py,
optimized_ensemble.py, hyperbolic_embeddings.py, relation_extractor.py

## Data Validation Chain
taxonomy_validator.py -> robi_validator.py -> quality_filter.py -> llm_validator.py

## Models Directory
Git-ignored. Contains trained model checkpoints. Do not attempt to read or modify directly.
