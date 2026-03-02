#!/usr/bin/env python3
"""
Evaluate Improved LUKE Model.

Tests on:
1. Held-out test set (30% of eval_100 used during training data creation)
2. Full eval_100 for comparison

Uses TaxoNERD for species extraction.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.span_extractor import SpanExtractor, extract_and_tag, TAXONERD_AVAILABLE
from models.luke_classifier import LukeInteractionClassifier


def evaluate_model(model_path: str, test_path: str, use_taxonerd: bool = True):
    """
    Evaluate LUKE model on test set.

    Args:
        model_path: Path to trained LUKE model
        test_path: Path to test CSV file
        use_taxonerd: Whether to use TaxoNERD for extraction
    """
    print("=" * 60)
    print(f"Evaluating: {Path(test_path).name}")
    print("=" * 60)

    # Load test data
    if test_path.endswith('.tsv'):
        df = pd.read_csv(test_path, sep='\t')
        sentence_col = 'sentence'
        label_col = 'evaluation_pair_interacting'
    else:
        df = pd.read_csv(test_path)
        sentence_col = 'text_original' if 'text_original' in df.columns else 'sentence'
        label_col = 'label'

    print(f"Loaded {len(df)} samples")

    sentences = df[sentence_col].tolist()
    ground_truth = df[label_col].astype(int).tolist()
    print(f"Label distribution: {pd.Series(ground_truth).value_counts().to_dict()}")

    # Check if we have pre-tagged text
    if 'text_tagged' in df.columns:
        print("Using pre-tagged text")
        tagged_texts = df['text_tagged'].tolist()
    else:
        # Initialize extractor and tag
        print(f"Tagging with TaxoNERD: {use_taxonerd and TAXONERD_AVAILABLE}")
        extractor = SpanExtractor(
            use_taxonerd=use_taxonerd and TAXONERD_AVAILABLE,
            taxonerd_model='en_ner_eco_md'
        )

        tagged_texts = []
        for sentence in sentences:
            tagged, sp1, sp2 = extract_and_tag(str(sentence), extractor, return_species=True)
            tagged_texts.append(tagged)

    # Load model
    print(f"\nLoading model from: {model_path}")
    classifier = LukeInteractionClassifier()
    classifier.load(model_path)

    # Predict
    print("Running predictions...")
    predictions = []
    probabilities = []

    for text in tagged_texts:
        try:
            pred, probs = classifier.predict([text])
            predictions.append(pred[0])
            prob = probs[0][1] if len(probs[0]) > 1 else probs[0][0]
            probabilities.append(prob)
        except Exception as e:
            predictions.append(0)
            probabilities.append(0.5)

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    cm = confusion_matrix(ground_truth, predictions)

    print(f"\n{'=' * 40}")
    print("RESULTS")
    print(f"{'=' * 40}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")

    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    BASE_DIR = "/home/egaillac/MetaP/classifier"

    # Find the most recent diverse model
    model_dir = os.path.join(BASE_DIR, "models/luke_diverse")
    runs = sorted([d for d in os.listdir(model_dir) if d.startswith("luke_run_")])
    model_path = os.path.join(model_dir, runs[-1], "final_model")
    print(f"Using model: {model_path}")

    print("\n" + "=" * 70)
    print("EVALUATION 1: Held-out Test Set (30% of eval_100)")
    print("=" * 70)

    test_held_out = os.path.join(BASE_DIR, "data/training/test_data_luke_diverse.csv")
    results_held_out = evaluate_model(model_path, test_held_out)

    print("\n" + "=" * 70)
    print("EVALUATION 2: Full eval_100 (includes samples seen during training)")
    print("=" * 70)

    eval_100_path = os.path.join(BASE_DIR, "data/evaluation/biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv")
    results_full = evaluate_model(model_path, eval_100_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nHeld-out Test Set (unseen during training):")
    print(f"  F1: {results_held_out['f1']:.4f}")
    print(f"  Precision: {results_held_out['precision']:.4f}")
    print(f"  Recall: {results_held_out['recall']:.4f}")

    print(f"\nFull eval_100:")
    print(f"  F1: {results_full['f1']:.4f}")
    print(f"  Precision: {results_full['precision']:.4f}")
    print(f"  Recall: {results_full['recall']:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON: Previous LUKE vs Improved LUKE")
    print("=" * 70)
    print("\nPrevious LUKE (regex, synthetic data only):")
    print("  F1: 0.3478 | Precision: 0.5333 | Recall: 0.2581")
    print(f"\nImproved LUKE (TaxoNERD, diverse data):")
    print(f"  F1: {results_full['f1']:.4f} | Precision: {results_full['precision']:.4f} | Recall: {results_full['recall']:.4f}")
