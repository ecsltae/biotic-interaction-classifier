#!/usr/bin/env python3
"""
Train LUKE Model for Species Interaction Classification.

This script trains a LUKE-based binary classifier on the entity-tagged
dataset to predict biotic interactions between species pairs.

Usage:
    # Standard training with validation split
    python train_luke_interaction.py --input data/training/training_data_luke.csv

    # 5-fold cross-validation
    python train_luke_interaction.py --input data/training/training_data_luke.csv --cv

    # Test mode (small subset)
    python train_luke_interaction.py --input data/training/training_data_luke.csv --test
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.luke_classifier import (
    LukeInteractionClassifier,
    cross_validate
)


def load_luke_dataset(
    path: str,
    text_column: str = "text_tagged",
    label_column: str = "label",
    max_samples: int = None
) -> tuple:
    """
    Load LUKE-formatted dataset.

    Args:
        path: Path to CSV file
        text_column: Column containing tagged text
        label_column: Column containing labels
        max_samples: Limit samples (for testing)

    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")

    # Filter to rows with valid tagged text
    df = df.dropna(subset=[text_column, label_column])

    texts = df[text_column].tolist()
    labels = df[label_column].astype(int).tolist()

    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    return texts, labels


def calculate_class_weights(labels: list) -> list:
    """Calculate class weights for imbalanced data."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)

    # Inverse frequency weighting
    weights = [total / (n_classes * counts[i]) for i in range(n_classes)]

    # Normalize so weights sum to n_classes
    weight_sum = sum(weights)
    weights = [w * n_classes / weight_sum for w in weights]

    return weights


def train_luke_model(
    input_path: str,
    output_dir: str,
    text_column: str = "text_tagged",
    label_column: str = "label",
    model_name: str = "studio-ousia/luke-base",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    val_split: float = 0.2,
    use_class_weights: bool = True,
    cv_folds: int = 0,
    max_samples: int = None,
    seed: int = 42
):
    """
    Train LUKE model on species interaction dataset.

    Args:
        input_path: Path to LUKE-formatted CSV
        output_dir: Directory to save model and results
        text_column: Column with tagged text
        label_column: Column with labels
        model_name: LUKE model to use
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Max sequence length
        val_split: Validation split ratio
        use_class_weights: Whether to use class weights
        cv_folds: Number of CV folds (0 = no CV)
        max_samples: Limit samples for testing
        seed: Random seed
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"luke_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("LUKE Species Interaction Classifier Training")
    print("=" * 60)
    print(f"Output directory: {run_dir}")

    # Load data
    texts, labels = load_luke_dataset(
        input_path,
        text_column=text_column,
        label_column=label_column,
        max_samples=max_samples
    )

    # Calculate class weights
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(labels)
        print(f"Class weights: {class_weights}")

    # Initialize classifier
    print(f"\nInitializing LUKE classifier: {model_name}")
    classifier = LukeInteractionClassifier(
        model_name=model_name,
        max_length=max_length
    )

    # Training configuration
    train_config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "eval_steps": 100,
        "save_steps": 500,
        "early_stopping_patience": 3,
        "class_weights": class_weights
    }

    # Run training
    if cv_folds > 0:
        # Cross-validation
        print(f"\n{'='*60}")
        print(f"Running {cv_folds}-fold Cross-Validation")
        print(f"{'='*60}")

        results = cross_validate(
            classifier=classifier,
            texts=texts,
            labels=labels,
            n_folds=cv_folds,
            output_dir=run_dir,
            **train_config
        )

        # Save CV results
        cv_results = {
            "cv_folds": cv_folds,
            "accuracy": float(results["accuracy"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "f1": float(results["f1"]),
            "std_accuracy": float(results["std_accuracy"]),
            "std_precision": float(results["std_precision"]),
            "std_recall": float(results["std_recall"]),
            "std_f1": float(results["std_f1"])
        }

        with open(os.path.join(run_dir, "cv_results.json"), "w") as f:
            json.dump(cv_results, f, indent=2)

    else:
        # Single train/val split
        print(f"\n{'='*60}")
        print(f"Training with {1-val_split:.0%}/{val_split:.0%} train/val split")
        print(f"{'='*60}")

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=val_split,
            stratify=labels,
            random_state=seed
        )

        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")

        # Train
        train_results = classifier.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            output_dir=run_dir,
            **train_config
        )

        # Final evaluation
        print(f"\n{'='*60}")
        print("Final Evaluation on Validation Set")
        print(f"{'='*60}")

        eval_results = classifier.evaluate(val_texts, val_labels)

        print(f"\nResults:")
        print(f"  Accuracy:  {eval_results['accuracy']:.4f}")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall:    {eval_results['recall']:.4f}")
        print(f"  F1:        {eval_results['f1']:.4f}")

        print(f"\nClassification Report:")
        print(eval_results['classification_report'])

        print(f"\nConfusion Matrix:")
        print(eval_results['confusion_matrix'])

        # Save results
        final_results = {
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "accuracy": float(eval_results['accuracy']),
            "precision": float(eval_results['precision']),
            "recall": float(eval_results['recall']),
            "f1": float(eval_results['f1']),
            "confusion_matrix": eval_results['confusion_matrix'].tolist()
        }

        with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)

        # Save the model
        model_path = os.path.join(run_dir, "final_model")
        classifier.save(model_path)

    # Save training config
    config = {
        "input_path": input_path,
        "model_name": model_name,
        "max_length": max_length,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "use_class_weights": use_class_weights,
        "cv_folds": cv_folds,
        "total_samples": len(texts),
        "timestamp": timestamp
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {run_dir}")

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train LUKE model for species interaction classification"
    )

    # Data arguments
    parser.add_argument(
        "--input", "-i",
        default="./data/training/training_data_luke.csv",
        help="Input LUKE-formatted CSV file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./models/luke",
        help="Output directory for model and results"
    )
    parser.add_argument(
        "--text-column",
        default="text_tagged",
        help="Column containing tagged text"
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing labels"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        default="studio-ousia/luke-base",
        help="LUKE model to use"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting"
    )

    # Cross-validation
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run 5-fold cross-validation"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds (with --cv)"
    )

    # Testing
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: use small subset (1000 samples)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Determine max samples
    max_samples = 1000 if args.test else args.max_samples

    # Determine CV folds
    cv_folds = args.cv_folds if args.cv else 0

    # Run training
    train_luke_model(
        input_path=args.input,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        val_split=args.val_split,
        use_class_weights=not args.no_class_weights,
        cv_folds=cv_folds,
        max_samples=max_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
