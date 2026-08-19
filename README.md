# Biotic Interaction Classifier

A comprehensive machine learning pipeline for detecting and classifying biotic interactions in scientific text using multiple state-of-the-art NLP models.

## Overview

This project implements various classification approaches for identifying biotic interactions (predation, parasitism, pollination, etc.) from scientific literature passages. The system supports multiple model architectures and includes tools for training, evaluation, and inference.

## Features

- **Multiple Model Architectures**:
  - Transformer models (DistilBERT, BioBERT, RoBERTa, PubMedBERT)
  - Support Vector Machines (SVM)
  - Random Forest
  - Logistic Regression
  - BERT variants
  - LLaMA models
  - LUKE (entity-aware transformers)

- **Advanced Training Features**:
  - 5-fold stratified cross-validation
  - Active learning for efficient annotation
  - Precision-recall threshold optimization
  - High precision mode for reduced false positives
  - External validation on curated test sets

- **Production-Ready APIs**:
  - RESTful API endpoints for each model
  - Batch prediction support
  - Real-time inference

## Project Structure

```
classifier/
├── src/                          # Source code
│   ├── models/                   # Model training scripts
│   │   ├── transformer_classifier.py    # Main transformer training pipeline
│   │   ├── svm_classifier.py            # SVM implementation
│   │   ├── random_forest_classifier.py  # Random Forest classifier
│   │   ├── bert_classifier.py           # BERT fine-tuning
│   │   ├── llama_classifier.py          # LLaMA model training
│   │   └── luke_classifier.py           # LUKE entity-aware model
│   ├── api/                      # API endpoints
│   │   ├── biomedbert_api.py
│   │   ├── svm_api.py
│   │   ├── lr_api.py
│   │   └── test_api.py
│   ├── data/                     # Data processing
│   │   ├── preprocessing.py
│   │   ├── data_building.py
│   │   ├── retrieve_positives.py
│   │   ├── retrieve_negatives.py
│   │   └── retrieve_random.py
│   └── utils/                    # Utilities
│       ├── prediction.py
│       ├── evaluation.py
│       └── classification_details.py
├── data/
│   ├── training/                 # Training datasets
│   ├── evaluation/               # Test/validation sets
│   ├── processed/                # Processed data
│   └── taxonomies/               # Species taxonomy data
├── models/                       # Trained models (git-ignored)
├── results/                      # Outputs
│   ├── cv_results/              # Cross-validation metrics
│   ├── predictions/             # Model predictions
│   └── figures/                 # Visualizations
├── notebooks/                    # Jupyter notebooks
└── scripts/                      # Utility scripts
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd classifier
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required models (if needed)

```bash
# Download spaCy language models
python -m spacy download en_core_web_sm
```

## Usage

### Training Transformer Models

Train all transformer models with cross-validation:

```bash
python src/models/transformer_classifier.py --model all --epochs 3 --cv_folds 5
```

Train a specific model:

```bash
python src/models/transformer_classifier.py --model biobert --epochs 5
```

Enable active learning:

```bash
python src/models/transformer_classifier.py --model BiomedBERT --active_learning --al_iterations 10
```

Optimize for high precision:

```bash
python src/models/transformer_classifier.py --model distilbert --target_precision 0.8
```

### Training SVM

```bash
python src/models/svm_classifier.py
```

### Making Predictions

```bash
python src/utils/prediction.py --model biomedbert --input data/evaluation/eval_100.tsv
```

### Running API Server

```bash
python src/api/biomedbert_api.py
```

## Model Performance

Results on external test set (eval_100.tsv):

| Model        | Precision | Recall | F1 Score |
|--------------|-----------|--------|----------|
| BiomedBERT   | TBD       | TBD    | TBD      |
| BioBERT      | TBD       | TBD    | TBD      |
| RoBERTa      | TBD       | TBD    | TBD      |
| DistilBERT   | TBD       | TBD    | TBD      |
| SVM          | TBD       | TBD    | TBD      |

*See [results/cv_results/](results/cv_results/) for detailed metrics.*

## Data

### Training Data

- **training_data_cleaned.csv**: Cleaned training set with labeled passages
- Format: `passage,label` where label is 0 (no interaction) or 1 (interaction)

### Evaluation Sets

- **eval_100.tsv**: Hand-curated test set (100 samples)
- **BiotXBench annotations**: Additional quality-checked annotations
- **GloBI datasets**: Global Biotic Interactions passages

## Active Learning

The transformer classifier supports uncertainty-based active learning:

1. Start with small labeled dataset (default: 500 samples)
2. Train initial model
3. Select most uncertain samples from unlabeled pool
4. Iteratively expand training set
5. Track performance improvement over iterations

Example:
```bash
python src/models/transformer_classifier.py \
  --model biobert \
  --active_learning \
  --al_iterations 10
```

## Threshold Optimization

For high-precision applications (e.g., automated curation), the system can optimize decision thresholds:

```bash
python src/models/transformer_classifier.py \
  --model BiomedBERT \
  --target_precision 0.9
```

This finds the optimal probability threshold to achieve target precision while maximizing recall.

## API Documentation

### Prediction Endpoint

```bash
POST /predict
{
  "text": "The fox hunts mice in the meadow."
}

Response:
{
  "prediction": 1,
  "confidence": 0.94,
  "label": "interaction"
}
```

### Batch Prediction

```bash
POST /predict_batch
{
  "texts": ["sentence 1", "sentence 2", ...]
}
```

## Contact

[contact details removed for anonymous review]
