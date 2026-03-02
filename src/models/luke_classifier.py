#!/usr/bin/env python3
"""
LUKE-based Binary Classifier for Species Interaction Detection.

Uses LUKE (Language Understanding with Knowledge-based Embeddings) with
entity-aware self-attention for classifying whether two species in a
sentence have a biotic interaction.

Based on: https://arxiv.org/abs/2010.01057 (Yamada et al., 2020)
Inspired by ArTraDB paper's relationship extraction approach.

Input format:
    Text with entity markers: "<e1>Canis lupus</e1> preys on <e2>Ovis aries</e2>"

Output:
    Binary classification: 0 (no interaction) or 1 (interaction)
"""

import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LukeTokenizer,
    LukeForEntityPairClassification,
    LukeConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Entity marker patterns
E1_START = "<e1>"
E1_END = "</e1>"
E2_START = "<e2>"
E2_END = "</e2>"

ENTITY_PATTERN = re.compile(
    r'<e([12])>(.*?)</e\1>',
    re.DOTALL
)


@dataclass
class LukeExample:
    """A single training/inference example for LUKE."""
    text: str
    text_clean: str
    entity1_span: Tuple[int, int]
    entity2_span: Tuple[int, int]
    entity1_text: str
    entity2_text: str
    label: Optional[int] = None


def parse_entity_markers(text: str) -> Optional[LukeExample]:
    """
    Parse text with entity markers and extract spans.

    Args:
        text: Text with <e1>...</e1> and <e2>...</e2> markers

    Returns:
        LukeExample with clean text and entity spans, or None if parsing fails
    """
    # Find all entity markers
    matches = list(ENTITY_PATTERN.finditer(text))

    if len(matches) < 2:
        return None

    # Get entity 1 and entity 2
    e1_match = None
    e2_match = None

    for m in matches:
        entity_num = m.group(1)
        if entity_num == "1" and e1_match is None:
            e1_match = m
        elif entity_num == "2" and e2_match is None:
            e2_match = m

    if e1_match is None or e2_match is None:
        return None

    # Extract entity texts
    e1_text = e1_match.group(2)
    e2_text = e2_match.group(2)

    # Remove all markers to get clean text
    text_clean = text
    text_clean = text_clean.replace(E1_START, "").replace(E1_END, "")
    text_clean = text_clean.replace(E2_START, "").replace(E2_END, "")

    # Calculate new positions in clean text
    # This is tricky because removing markers shifts positions

    # Find positions in clean text by searching for entity text
    e1_start = text_clean.find(e1_text)
    if e1_start == -1:
        return None
    e1_end = e1_start + len(e1_text)

    # For e2, we need to find it but avoid matching e1 if they're the same
    search_start = 0
    if e1_text == e2_text:
        search_start = e1_end

    e2_start = text_clean.find(e2_text, search_start)
    if e2_start == -1:
        # Try from beginning if not found
        e2_start = text_clean.find(e2_text)
        if e2_start == -1 or e2_start == e1_start:
            return None
    e2_end = e2_start + len(e2_text)

    return LukeExample(
        text=text,
        text_clean=text_clean,
        entity1_span=(e1_start, e1_end),
        entity2_span=(e2_start, e2_end),
        entity1_text=e1_text,
        entity2_text=e2_text
    )


class LukeInteractionDataset(Dataset):
    """
    PyTorch Dataset for LUKE entity pair classification.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: LukeTokenizer = None,
        max_length: int = 256
    ):
        """
        Initialize dataset.

        Args:
            texts: List of texts with entity markers
            labels: List of labels (0 or 1)
            tokenizer: LukeTokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Parse all texts and filter invalid ones
        self.examples = []
        self.valid_indices = []

        for i, text in enumerate(texts):
            example = parse_entity_markers(text)
            if example is not None:
                if labels is not None:
                    example.label = labels[i]
                self.examples.append(example)
                self.valid_indices.append(i)
            else:
                logger.warning(f"Failed to parse example {i}: {text[:100]}...")

        logger.info(f"Parsed {len(self.examples)}/{len(texts)} examples successfully")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize with entity spans
        try:
            encoding = self.tokenizer(
                example.text_clean,
                entity_spans=[example.entity1_span, example.entity2_span],
                add_prefix_space=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Tokenization failed for example {idx}: {e}")
            # Return dummy encoding
            encoding = self.tokenizer(
                "dummy text",
                entity_spans=[(0, 5), (6, 10)],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

        # Squeeze batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if example.label is not None:
            item["labels"] = torch.tensor(example.label, dtype=torch.long)

        return item


class LukeInteractionClassifier:
    """
    LUKE-based binary classifier for species interaction detection.
    """

    def __init__(
        self,
        model_name: str = "studio-ousia/luke-base",
        num_labels: int = 2,
        max_length: int = 256,
        device: Optional[str] = None
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Pretrained LUKE model name
            num_labels: Number of classification labels (2 for binary)
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu/None for auto)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        logger.info(f"Loading LUKE model: {model_name}")
        self.tokenizer = LukeTokenizer.from_pretrained(model_name)
        self.model = LukeForEntityPairClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)

        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def create_dataset(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None
    ) -> LukeInteractionDataset:
        """Create a dataset from texts and labels."""
        return LukeInteractionDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = "./luke_results",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        eval_steps: int = 100,
        save_steps: int = 500,
        early_stopping_patience: int = 3,
        class_weights: Optional[List[float]] = None
    ) -> Dict:
        """
        Train the LUKE classifier.

        Args:
            train_texts: Training texts with entity markers
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_ratio: Warmup ratio
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            early_stopping_patience: Early stopping patience
            class_weights: Optional class weights for imbalanced data

        Returns:
            Training results dictionary
        """
        # Create datasets
        train_dataset = self.create_dataset(train_texts, train_labels)

        eval_dataset = None
        if val_texts is not None and val_labels is not None:
            eval_dataset = self.create_dataset(val_texts, val_labels)

        # Set up class weights if provided
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)

            # Custom trainer with weighted loss
            class WeightedLukeTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
                    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                    return (loss, outputs) if return_outputs else loss

            trainer_class = WeightedLukeTrainer
        else:
            trainer_class = Trainer

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            greater_is_better=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            report_to="none",  # Disable wandb etc.
            fp16=torch.cuda.is_available(),
        )

        # Callbacks
        callbacks = []
        if eval_dataset and early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

        # Trainer
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save final model
        trainer.save_model(f"{output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model")

        logger.info(f"Training complete. Model saved to {output_dir}/final_model")

        return {
            "train_loss": train_result.training_loss,
            "train_steps": train_result.global_step,
            "model_path": f"{output_dir}/final_model"
        }

    def _compute_metrics(self, eval_pred) -> Dict:
        """Compute evaluation metrics."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 16
    ) -> Dict:
        """
        Evaluate the model on a dataset.

        Args:
            texts: Texts with entity markers
            labels: True labels
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        predictions, probabilities = self.predict(texts, batch_size=batch_size)

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)

        report = classification_report(labels, predictions, target_names=["No Interaction", "Interaction"])

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "predictions": predictions,
            "probabilities": probabilities
        }

    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        return_probs: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on new texts.

        Args:
            texts: Single text or list of texts with entity markers
            batch_size: Batch size for inference
            return_probs: Whether to return probabilities

        Returns:
            Tuple of (predictions, probabilities)
        """
        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()

        # Create dataset (no labels)
        dataset = self.create_dataset(texts, labels=None)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits

                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities) if return_probs else None

        return predictions, probabilities

    def save(self, path: str):
        """Save model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and tokenizer from path."""
        self.model = LukeForEntityPairClassification.from_pretrained(path)
        self.tokenizer = LukeTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")


def cross_validate(
    classifier: LukeInteractionClassifier,
    texts: List[str],
    labels: List[int],
    n_folds: int = 5,
    output_dir: str = "./luke_cv_results",
    **train_kwargs
) -> Dict:
    """
    Perform k-fold cross-validation.

    Args:
        classifier: LukeInteractionClassifier instance
        texts: All texts
        labels: All labels
        n_folds: Number of folds
        output_dir: Output directory
        **train_kwargs: Additional arguments for training

    Returns:
        Cross-validation results
    """
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts, labels)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold + 1}/{n_folds}")
        logger.info(f"{'='*60}")

        # Split data
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Reinitialize model for each fold
        classifier.model = LukeForEntityPairClassification.from_pretrained(
            classifier.model_name,
            num_labels=classifier.num_labels
        )
        classifier.model.to(classifier.device)

        # Train
        fold_dir = f"{output_dir}/fold_{fold}"
        classifier.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            output_dir=fold_dir,
            **train_kwargs
        )

        # Evaluate
        results = classifier.evaluate(val_texts, val_labels)
        fold_results.append(results)

        logger.info(f"Fold {fold + 1} Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")
        logger.info(f"  F1: {results['f1']:.4f}")

    # Aggregate results
    avg_results = {
        "accuracy": np.mean([r["accuracy"] for r in fold_results]),
        "precision": np.mean([r["precision"] for r in fold_results]),
        "recall": np.mean([r["recall"] for r in fold_results]),
        "f1": np.mean([r["f1"] for r in fold_results]),
        "std_accuracy": np.std([r["accuracy"] for r in fold_results]),
        "std_precision": np.std([r["precision"] for r in fold_results]),
        "std_recall": np.std([r["recall"] for r in fold_results]),
        "std_f1": np.std([r["f1"] for r in fold_results]),
        "fold_results": fold_results
    }

    logger.info(f"\n{'='*60}")
    logger.info("Cross-Validation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {avg_results['accuracy']:.4f} (+/- {avg_results['std_accuracy']:.4f})")
    logger.info(f"Precision: {avg_results['precision']:.4f} (+/- {avg_results['std_precision']:.4f})")
    logger.info(f"Recall: {avg_results['recall']:.4f} (+/- {avg_results['std_recall']:.4f})")
    logger.info(f"F1: {avg_results['f1']:.4f} (+/- {avg_results['std_f1']:.4f})")

    return avg_results


if __name__ == "__main__":
    # Quick test
    print("Testing LUKE Interaction Classifier...")

    # Test entity marker parsing
    test_text = "The <e1>Canis lupus</e1> is known to prey on <e2>Ovis aries</e2> in alpine regions."
    example = parse_entity_markers(test_text)

    if example:
        print(f"Original: {test_text}")
        print(f"Clean: {example.text_clean}")
        print(f"Entity 1: '{example.entity1_text}' at {example.entity1_span}")
        print(f"Entity 2: '{example.entity2_text}' at {example.entity2_span}")
    else:
        print("Failed to parse example")

    # Test classifier initialization (requires LUKE model download)
    try:
        classifier = LukeInteractionClassifier()
        print(f"\nClassifier initialized on {classifier.device}")

        # Test prediction
        predictions, probs = classifier.predict([test_text])
        print(f"Prediction: {predictions[0]} (prob: {probs[0]})")
    except Exception as e:
        print(f"Classifier test failed (may need to download model): {e}")
