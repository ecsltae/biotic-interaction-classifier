#!/usr/bin/env python3
"""
Train BiomedBERT and RoBERTa on the new v3 dataset, then create ensemble.
Compare with baseline on eval_100.
"""

import os
import sys
sys.path.insert(0, '/home/egaillac/MetaP/classifier/src/models')

import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, classification_report
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = '/home/egaillac/MetaP/classifier'
DATA_DIR = os.path.join(BASE_DIR, 'data/training')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ensemble_model_v3')
EVAL_FILE = os.path.join(BASE_DIR, 'data/evaluation/eval_100.tsv')

# Training dataset
TRAIN_FILE = os.path.join(DATA_DIR, 'training_data_v3.csv')

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Model configs
MODELS_TO_TRAIN = {
    'biomedbert': {
        'pretrained': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'save_path': os.path.join(MODEL_DIR, 'transformer_BiomedBERT_v3'),
        'weight': 0.65,
    },
    'roberta': {
        'pretrained': 'roberta-base',
        'save_path': os.path.join(MODEL_DIR, 'transformer_roberta_v3'),
        'weight': 0.35,
    },
}

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 2e-5


# =============================================================================
# DATASET
# =============================================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_metrics(pred):
    """Compute metrics for Trainer"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }


# =============================================================================
# TRAINING
# =============================================================================
def train_model(model_name, config, train_texts, train_labels, val_texts, val_labels):
    """Train a single transformer model"""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*70}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['pretrained'],
        num_labels=2
    )

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['save_path'],
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(config['save_path'], 'logs'),
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=DEVICE == 'cuda',
        report_to='none',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print(f"\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time/60:.1f} minutes")

    # Save model and tokenizer
    model.save_pretrained(config['save_path'])
    tokenizer.save_pretrained(config['save_path'])
    print(f"Model saved to: {config['save_path']}")

    # Evaluate on validation
    eval_results = trainer.evaluate()
    print(f"\nValidation Results:")
    for k, v in eval_results.items():
        if 'eval_' in k:
            print(f"  {k.replace('eval_', '')}: {v:.4f}")

    return model, tokenizer, eval_results


# =============================================================================
# ENSEMBLE EVALUATION
# =============================================================================
@torch.no_grad()
def ensemble_predict(texts, models, tokenizers, weights, device=DEVICE, threshold=0.5):
    """Make ensemble predictions"""
    all_probs = []

    for model_name, model in models.items():
        model.eval()
        tokenizer = tokenizers[model_name]

        # Tokenize
        encodings = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Predict in batches
        model_probs = []
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        for batch in loader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            model_probs.append(probs.cpu().numpy())

        model_probs = np.vstack(model_probs)
        all_probs.append(model_probs)

    # Weighted average
    all_probs = np.array(all_probs)
    weight_array = np.array([weights[k] for k in models.keys()])
    weight_array = weight_array / weight_array.sum()

    ensemble_probs = np.average(all_probs, axis=0, weights=weight_array)
    predictions = (ensemble_probs[:, 1] >= threshold).astype(int)

    return predictions, ensemble_probs[:, 1]


def evaluate_on_eval100(models, tokenizers, weights, threshold=0.5):
    """Evaluate ensemble on eval_100.tsv"""
    print(f"\n{'='*70}")
    print("EVALUATION ON EVAL_100")
    print(f"{'='*70}")

    # Load eval data
    eval_df = pd.read_csv(EVAL_FILE, sep='\t')
    texts = eval_df['sentence'].tolist()
    labels = eval_df['evaluation_interaction_identified'].tolist()

    print(f"Loaded {len(texts)} evaluation samples")

    # Predict
    predictions, probs = ensemble_predict(texts, models, tokenizers, weights, threshold=threshold)

    # Compute metrics
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    print(f"\nResults (threshold={threshold}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['No Interaction', 'Biotic Interaction']))

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'threshold': threshold,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("ENSEMBLE V3 TRAINING")
    print("="*70)

    # Load data
    print("\nLoading training data...")
    df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded {len(df)} samples")
    print(f"  Positives: {sum(df['label'] == 1)}")
    print(f"  Negatives: {sum(df['label'] == 0)}")

    # Split into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['passage'].tolist(),
        df['label'].tolist(),
        test_size=0.1,
        random_state=42,
        stratify=df['label'].tolist()
    )

    print(f"\nSplit:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Val: {len(val_texts)}")

    # Train models
    trained_models = {}
    trained_tokenizers = {}
    weights = {}

    for model_name, config in MODELS_TO_TRAIN.items():
        model, tokenizer, _ = train_model(
            model_name, config,
            train_texts, train_labels,
            val_texts, val_labels
        )
        trained_models[model_name] = model.to(DEVICE)
        trained_tokenizers[model_name] = tokenizer
        weights[model_name] = config['weight']

    # Find optimal threshold
    print("\n" + "="*70)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*70)

    _, probs = ensemble_predict(val_texts, trained_models, trained_tokenizers, weights, threshold=0.5)

    precisions, recalls, thresholds = precision_recall_curve(val_labels, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # F1-optimized threshold
    f1_idx = np.argmax(f1_scores)
    f1_threshold = thresholds[f1_idx] if f1_idx < len(thresholds) else 0.5
    print(f"\nF1-optimized threshold: {f1_threshold:.4f}")
    print(f"  At this threshold: P={precisions[f1_idx]:.3f}, R={recalls[f1_idx]:.3f}, F1={f1_scores[f1_idx]:.3f}")

    # Precision-optimized threshold (P > 0.9)
    high_prec_idx = np.argmax(precisions > 0.9)
    prec_threshold = thresholds[high_prec_idx] if high_prec_idx < len(thresholds) else 0.9
    print(f"\nPrecision-optimized threshold: {prec_threshold:.4f}")
    print(f"  At this threshold: P={precisions[high_prec_idx]:.3f}, R={recalls[high_prec_idx]:.3f}")

    # Evaluate on eval_100
    print("\n--- F1-Optimized Threshold ---")
    results_f1 = evaluate_on_eval100(trained_models, trained_tokenizers, weights, threshold=f1_threshold)

    print("\n--- Default Threshold (0.5) ---")
    results_default = evaluate_on_eval100(trained_models, trained_tokenizers, weights, threshold=0.5)

    # Compare with baseline
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)

    # Load baseline results if available
    baseline_file = os.path.join(BASE_DIR, 'ensemble_model/eval_100_results.csv')
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        print("\nBaseline (original 20k dataset):")
        print(baseline_df.to_string(index=False))

        print("\nV3 Results (improved dataset):")
        print(f"  F1-Opt Threshold ({f1_threshold:.3f}): P={results_f1['precision']:.3f}, R={results_f1['recall']:.3f}, F1={results_f1['f1']:.3f}")
        print(f"  Default Threshold (0.5): P={results_default['precision']:.3f}, R={results_default['recall']:.3f}, F1={results_default['f1']:.3f}")
    else:
        print("Baseline results not found. Run evaluate_ensemble_on_real_test.py first.")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_df = pd.DataFrame({
        'Threshold_Type': ['F1-Optimized', 'Default'],
        'Threshold': [f1_threshold, 0.5],
        'Precision': [results_f1['precision'], results_default['precision']],
        'Recall': [results_f1['recall'], results_default['recall']],
        'F1': [results_f1['f1'], results_default['f1']],
        'Accuracy': [results_f1['accuracy'], results_default['accuracy']],
    })

    results_df.to_csv(os.path.join(OUTPUT_DIR, 'eval_100_results_v3.csv'), index=False)
    print(f"\nResults saved to: {OUTPUT_DIR}/eval_100_results_v3.csv")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
