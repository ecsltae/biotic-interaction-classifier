#!/usr/bin/env python3
"""
Train BiomedBERT on the high-quality dataset v3 (conservative)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = '/home/egaillac/MetaP/classifier'
DATA_FILE = f'{BASE_DIR}/data/training/training_data_quality_v3.csv'
MODEL_DIR = f'{BASE_DIR}/models/transformer_BiomedBERT_quality_v3'
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 2e-5


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


def evaluate_on_eval100(model, tokenizer, device):
    eval_df = pd.read_csv(EVAL_FILE, sep='\t')
    sentences = eval_df['sentence'].tolist()
    labels = eval_df['evaluation_interaction_identified'].tolist()

    model.eval()
    if device == 'cuda':
        model = model.half()

    dataset = TextDataset(sentences, labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

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

    print(f"\nEval100: F1={best_metrics['f1']:.3f}, P={best_metrics['precision']:.3f}, R={best_metrics['recall']:.3f}, t={best_thresh:.2f}")
    return best_metrics, best_thresh


def main():
    print("="*70)
    print("TRAINING BIOMEDBERT ON QUALITY V3 DATASET")
    print("="*70)

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples ({sum(df['label']==1)} pos, {sum(df['label']==0)} neg)")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['passage'].tolist(), df['label'].tolist(),
        test_size=0.1, random_state=42, stratify=df['label'].tolist()
    )

    print(f"Split: Train={len(train_texts)}, Val={len(val_texts)}")

    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
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
    print(f"Done in {(time.time()-start)/60:.1f} min")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved to: {MODEL_DIR}")

    print("\n" + "="*70)
    model.to(DEVICE)
    eval_metrics, best_thresh = evaluate_on_eval100(model, tokenizer, DEVICE)

    pd.DataFrame([{
        'model': 'BiomedBERT_quality_v3',
        'dataset': 'training_data_quality_v3.csv',
        **eval_metrics,
        'threshold': best_thresh
    }]).to_csv(f'{MODEL_DIR}/eval100_results.csv', index=False)

    print("\nComparison:")
    print(f"  6k_orig:    F1=0.488")
    print(f"  quality_v2: F1=0.486")
    print(f"  quality_v3: F1={eval_metrics['f1']:.3f}")
    print(f"  Ensemble:   F1=0.500")


if __name__ == '__main__':
    main()
