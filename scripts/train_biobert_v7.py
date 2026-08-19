#!/usr/bin/env python3
"""
Train BioBERT on GloBI v7 dataset for ensemble.
Second model to complement BiomedBERT.
"""

import os
import sys
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
DATA_FILE = f'{BASE_DIR}/data/training/training_data_globi_v7_llm_cleaned.csv'
MODEL_DIR = f'{BASE_DIR}/models/transformer_BioBERT_globi_v7'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Hyperparameters
EPOCHS = 8  # More epochs, let early stopping decide
BATCH_SIZE = 64
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1


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


def main():
    print("="*70)
    print("TRAINING BIOBERT ON GLOBI V7 (FOR ENSEMBLE)")
    print("="*70)

    # Load dataset
    print(f"\nLoading dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset: {len(df)} samples ({sum(df['label']==1)} pos, {sum(df['label']==0)} neg)")

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(),
        test_size=0.1, random_state=42, stratify=df['label'].tolist()
    )

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

    # Load BioBERT
    model_name = 'dmis-lab/biobert-v1.1'
    print(f"\nModel: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    eval_steps = steps_per_epoch // 2

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=f'{MODEL_DIR}/logs',
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=DEVICE == 'cuda',
        report_to='none',
        save_total_limit=3,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print(f"\nTraining config: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("Training...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training completed in {elapsed/60:.1f} min")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
