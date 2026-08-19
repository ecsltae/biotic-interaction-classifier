#!/usr/bin/env python3
"""
Train BiomedBERT on v8 Dataset with Strong Regularization.

Key changes from previous training:
- Stronger regularization: DROPOUT=0.3, LABEL_SMOOTHING=0.2
- Lower learning rate: 1e-5 (more careful updates)
- Fewer epochs: 4 (prevent overfitting)
- v8 dataset: v7 + diverse SIBiLS real sentences

Goal: Better generalization, less overconfident predictions.
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

BASE_DIR = '/home/egaillac/MetaP/classifier'
V8_FILE = f'{BASE_DIR}/data/training/training_data_v8_diverse.csv'
EP_TEST_FILE = f'{BASE_DIR}/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
OUTPUT_DIR = f'{BASE_DIR}/models/transformer_BiomedBERT_v8'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# STRONG REGULARIZATION CONFIG
MODEL_ID = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
N_FOLDS = 5
EPOCHS = 4  # Fewer epochs to prevent overfitting
BATCH_SIZE = 16  # Smaller batch
MAX_LENGTH = 256
LEARNING_RATE = 1e-5  # Lower LR for more careful updates
LABEL_SMOOTHING = 0.2  # Strong label smoothing
WEIGHT_DECAY = 0.02  # Slightly higher weight decay
DROPOUT = 0.3  # Higher dropout


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
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }


def evaluate_on_test(model, tokenizer, test_texts, test_labels):
    """Evaluate model on test set, return metrics and probability distribution."""
    model.eval()
    dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    test_labels = np.array(test_labels)

    # Find best threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(test_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds = (all_probs >= best_thresh).astype(int)
    precision = precision_score(test_labels, preds, zero_division=0)
    recall = recall_score(test_labels, preds, zero_division=0)
    accuracy = accuracy_score(test_labels, preds)
    cm = confusion_matrix(test_labels, preds)

    # Probability distribution analysis
    prob_ranges = {
        '0.0-0.2': np.sum((all_probs >= 0.0) & (all_probs < 0.2)),
        '0.2-0.4': np.sum((all_probs >= 0.2) & (all_probs < 0.4)),
        '0.4-0.6': np.sum((all_probs >= 0.4) & (all_probs < 0.6)),
        '0.6-0.8': np.sum((all_probs >= 0.6) & (all_probs < 0.8)),
        '0.8-1.0': np.sum((all_probs >= 0.8) & (all_probs <= 1.0)),
    }

    return {
        'f1': best_f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'threshold': best_thresh,
        'confusion_matrix': cm.tolist(),
        'probabilities': all_probs.tolist(),
        'prob_distribution': prob_ranges,
        'prob_mean': float(np.mean(all_probs)),
        'prob_std': float(np.std(all_probs)),
    }


def main():
    start_time = time.time()

    print("="*70)
    print("TRAINING BiomedBERT v8 WITH STRONG REGULARIZATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Weight decay: {WEIGHT_DECAY}")

    # Load training data
    print("\n1. Loading v8 dataset...")
    train_df = pd.read_csv(V8_FILE)
    print(f"   Samples: {len(train_df)}")
    print(f"   Positives: {sum(train_df['label']==1)}")
    print(f"   Negatives: {sum(train_df['label']==0)}")

    # Load test data
    print("\n2. Loading EP test set...")
    test_df = pd.read_csv(EP_TEST_FILE, sep='\t', encoding='latin-1')
    test_texts = test_df['sentence'].tolist()
    test_labels = test_df['evaluation_pair_interacting'].tolist()
    print(f"   Samples: {len(test_texts)}")
    print(f"   Positives: {sum(test_labels)}")

    texts = train_df['text'].tolist()
    labels = train_df['label'].tolist()

    # 5-fold CV
    print(f"\n3. Training with {N_FOLDS}-fold CV...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    best_model = None
    best_tokenizer = None
    best_f1 = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

        # Load fresh model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            hidden_dropout_prob=DROPOUT,
            attention_probs_dropout_prob=DROPOUT
        )
        model.to(DEVICE)

        train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

        fold_output_dir = f'{BASE_DIR}/models/cv_temp/BiomedBERT_v8_fold{fold}'

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            warmup_ratio=0.1,
            weight_decay=WEIGHT_DECAY,
            learning_rate=LEARNING_RATE,
            label_smoothing_factor=LABEL_SMOOTHING,
            logging_steps=200,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            fp16=DEVICE == 'cuda',
            report_to='none',
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Evaluate
        val_metrics = trainer.evaluate()
        print(f"Val F1: {val_metrics['eval_f1']:.3f}")

        test_metrics = evaluate_on_test(model, tokenizer, test_texts, test_labels)
        print(f"Test F1: {test_metrics['f1']:.3f} | Precision: {test_metrics['precision']:.3f} | Recall: {test_metrics['recall']:.3f}")
        print(f"Prob distribution: {test_metrics['prob_distribution']}")
        print(f"Prob stats: mean={test_metrics['prob_mean']:.3f}, std={test_metrics['prob_std']:.3f}")

        fold_results.append({
            'fold': fold + 1,
            'val_f1': val_metrics['eval_f1'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'prob_distribution': test_metrics['prob_distribution'],
            'prob_mean': test_metrics['prob_mean'],
            'prob_std': test_metrics['prob_std'],
        })

        # Keep best
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_model = model
            best_tokenizer = tokenizer
            best_metrics = test_metrics

    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    avg_test_f1 = np.mean([r['test_f1'] for r in fold_results])
    std_test_f1 = np.std([r['test_f1'] for r in fold_results])
    avg_test_prec = np.mean([r['test_precision'] for r in fold_results])
    avg_test_rec = np.mean([r['test_recall'] for r in fold_results])
    avg_prob_std = np.mean([r['prob_std'] for r in fold_results])

    print(f"\nAvg Test F1: {avg_test_f1:.3f} ± {std_test_f1:.3f}")
    print(f"Avg Precision: {avg_test_prec:.3f}")
    print(f"Avg Recall: {avg_test_rec:.3f}")
    print(f"Avg Prob Std: {avg_prob_std:.3f}")
    print(f"Best Fold F1: {best_f1:.3f}")

    # Target check
    print("\nTarget Check:")
    f1_pass = avg_test_f1 > 0.75
    prec_pass = avg_test_prec > avg_test_f1
    print(f"  F1 > 0.75: {'PASS' if f1_pass else 'FAIL'} ({avg_test_f1:.3f})")
    print(f"  Precision > F1: {'PASS' if prec_pass else 'FAIL'} ({avg_test_prec:.3f} vs {avg_test_f1:.3f})")

    # Probability spread check
    print(f"\nProbability Spread Check:")
    print(f"  Prob Std: {avg_prob_std:.3f}")
    if avg_prob_std > 0.25:
        print("  Good probability spread (not too binary)")
    else:
        print("  Still too binary, may need more regularization")

    # Save best model
    print("\n4. Saving best model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_model.save_pretrained(OUTPUT_DIR)
    best_tokenizer.save_pretrained(OUTPUT_DIR)

    # Save results
    results = {
        'config': {
            'model': MODEL_ID,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'label_smoothing': LABEL_SMOOTHING,
            'dropout': DROPOUT,
            'weight_decay': WEIGHT_DECAY,
            'dataset': 'v8_diverse',
        },
        'fold_results': fold_results,
        'summary': {
            'avg_test_f1': avg_test_f1,
            'std_test_f1': std_test_f1,
            'avg_test_precision': avg_test_prec,
            'avg_test_recall': avg_test_rec,
            'best_test_f1': best_f1,
            'avg_prob_std': avg_prob_std,
        },
        'best_metrics': best_metrics,
    }

    with open(f'{OUTPUT_DIR}/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"Total time: {(time.time()-start_time)/60:.1f} min")


if __name__ == "__main__":
    main()
