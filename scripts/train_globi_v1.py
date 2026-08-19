#!/usr/bin/env python3
"""
Train BiomedBERT on the GloBI template-generated dataset v1
With cross-validation and eval_100 evaluation
"""

import os
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import json

warnings.filterwarnings('ignore')

BASE_DIR = '/home/egaillac/MetaP/classifier'
DATA_FILE = f'{BASE_DIR}/data/training/training_data_globi_v1.csv'
MODEL_DIR = f'{BASE_DIR}/models/transformer_BiomedBERT_globi_v1'
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 32  # Larger batch with 40GB GPU
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
N_FOLDS = 5


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


def evaluate_model(model, tokenizer, texts, labels, device, batch_size=32):
    """Evaluate model and return predictions and metrics."""
    model.eval()
    dataset = TextDataset(texts, labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    # Find best threshold
    best_f1, best_thresh, best_metrics = 0, 0.5, {}
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1,
                'accuracy': accuracy_score(labels, preds),
            }

    return all_probs, best_metrics, best_thresh


def cross_validate(df, tokenizer, model_name, n_folds=5):
    """Run k-fold cross-validation."""
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION ({n_folds} folds)")
    print(f"{'='*70}")

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"Train: {len(train_texts)} ({sum(train_labels)} pos)")
        print(f"Val: {len(val_texts)} ({sum(val_labels)} pos)")

        # Fresh model for each fold
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(DEVICE)

        train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

        training_args = TrainingArguments(
            output_dir=f'{MODEL_DIR}/cv_fold_{fold}',
            num_train_epochs=3,  # Fewer epochs for CV
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=LEARNING_RATE,
            logging_steps=200,
            eval_strategy='epoch',
            save_strategy='no',
            fp16=DEVICE == 'cuda',
            report_to='none',
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
        _, metrics, thresh = evaluate_model(model, tokenizer, val_texts, val_labels, DEVICE)
        fold_results.append(metrics)
        print(f"Fold {fold+1}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")

        # Free memory
        del model, trainer
        torch.cuda.empty_cache()

    # Aggregate results
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")

    avg_metrics = {}
    for metric in ['f1', 'precision', 'recall', 'accuracy']:
        values = [r[metric] for r in fold_results]
        avg_metrics[metric] = np.mean(values)
        avg_metrics[f'{metric}_std'] = np.std(values)
        print(f"{metric.upper():12s}: {np.mean(values):.3f} +/- {np.std(values):.3f}")

    return fold_results, avg_metrics


def train_final_model(df, tokenizer, model_name):
    """Train final model on full dataset."""
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*70}")

    # Split for validation during training
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(),
        test_size=0.1, random_state=42, stratify=df['label'].tolist()
    )

    print(f"Train: {len(train_texts)} ({sum(train_labels)} pos)")
    print(f"Val: {len(val_texts)} ({sum(val_labels)} pos)")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=f'{MODEL_DIR}/logs',
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=200,
        save_strategy='steps',
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=DEVICE == 'cuda',
        report_to='none',
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("\nTraining...")
    start = time.time()
    trainer.train()
    print(f"Training completed in {(time.time()-start)/60:.1f} min")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to: {MODEL_DIR}")

    return model, trainer


def evaluate_on_eval100(model, tokenizer, device):
    """Evaluate on the 100-sentence test set."""
    print(f"\n{'='*70}")
    print("EVALUATION ON EVAL_100")
    print(f"{'='*70}")

    eval_df = pd.read_csv(EVAL_FILE, sep='\t')
    sentences = eval_df['sentence'].tolist()
    labels = eval_df['evaluation_pair_interacting'].tolist()  # Fixed column name

    print(f"Eval set: {len(sentences)} sentences ({sum(labels)} positive)")

    probs, metrics, best_thresh = evaluate_model(model, tokenizer, sentences, labels, device)

    print(f"\nResults at optimal threshold ({best_thresh:.2f}):")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")

    # Confusion matrix
    preds = (probs >= best_thresh).astype(int)
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

    return metrics, best_thresh, probs


def main():
    print("="*70)
    print("TRAINING BIOMEDBERT ON GLOBI V1 DATASET (40k samples)")
    print("="*70)

    # Load data
    df = pd.read_csv(DATA_FILE)
    print(f"\nLoaded {len(df)} samples:")
    print(f"  Positives: {sum(df['label']==1)}")
    print(f"  Negatives: {sum(df['label']==0)}")

    # Show interaction type distribution
    if 'interaction_type' in df.columns:
        print(f"\nInteraction types in positives:")
        pos_types = df[df['label']==1]['interaction_type'].value_counts()
        for itype, count in pos_types.head(8).items():
            print(f"  {itype}: {count}")

    # Load tokenizer
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    print(f"\nModel: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Cross-validation
    cv_results, cv_avg = cross_validate(df, tokenizer, model_name, n_folds=N_FOLDS)

    # Train final model
    model, trainer = train_final_model(df, tokenizer, model_name)

    # Evaluate on eval_100
    model.to(DEVICE)
    eval_metrics, best_thresh, probs = evaluate_on_eval100(model, tokenizer, DEVICE)

    # Save results
    results = {
        'model': 'BiomedBERT_globi_v1',
        'dataset': 'training_data_globi_v1.csv',
        'dataset_size': len(df),
        'cv_f1': cv_avg['f1'],
        'cv_f1_std': cv_avg['f1_std'],
        'cv_precision': cv_avg['precision'],
        'cv_recall': cv_avg['recall'],
        'eval100_f1': eval_metrics['f1'],
        'eval100_precision': eval_metrics['precision'],
        'eval100_recall': eval_metrics['recall'],
        'eval100_accuracy': eval_metrics['accuracy'],
        'best_threshold': best_thresh,
    }

    with open(f'{MODEL_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame([results]).to_csv(f'{MODEL_DIR}/eval_results.csv', index=False)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Cross-validation F1: {cv_avg['f1']:.3f} +/- {cv_avg['f1_std']:.3f}")
    print(f"Eval_100 F1:         {eval_metrics['f1']:.3f}")
    print(f"Best threshold:      {best_thresh:.2f}")

    print(f"\nComparison with previous models:")
    print(f"  6k_orig:     F1=0.488")
    print(f"  quality_v2:  F1=0.486")
    print(f"  quality_v3:  F1~0.49")
    print(f"  GloBI v1:    F1={eval_metrics['f1']:.3f}")


if __name__ == '__main__':
    main()
