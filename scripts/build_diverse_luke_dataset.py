#!/usr/bin/env python3
"""
Build Diverse LUKE Training Dataset from Real Annotated Data.

Uses multiple evaluation datasets to create a diverse training set:
1. eval_100 - 70% for training, 30% for final testing
2. Other annotated datasets (globi-*, biotx-50*) for additional training

Uses TaxoNERD for robust species extraction.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.span_extractor import (
    SpanExtractor,
    extract_and_tag,
    TAXONERD_AVAILABLE
)


def load_annotated_data(file_path: str, sentence_col: str = "sentence",
                        label_col: str = "evaluation_pair_interacting") -> pd.DataFrame:
    """Load an annotated TSV file."""
    print(f"Loading: {file_path}")

    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

    # Check columns
    if sentence_col not in df.columns:
        print(f"  Warning: '{sentence_col}' not found, skipping")
        return None
    if label_col not in df.columns:
        print(f"  Warning: '{label_col}' not found, skipping")
        return None

    # Extract relevant columns
    df = df[[sentence_col, label_col]].copy()
    df.columns = ['sentence', 'label']
    df['label'] = df['label'].astype(int)
    df = df.dropna()

    print(f"  Loaded {len(df)} samples")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    return df


def process_with_taxonerd(df: pd.DataFrame, extractor: SpanExtractor) -> pd.DataFrame:
    """Process sentences with TaxoNERD for entity tagging."""
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sentence = str(row['sentence'])
        label = row['label']

        # Extract and tag
        tagged, sp1, sp2 = extract_and_tag(sentence, extractor, return_species=True)

        if sp1 is not None and sp2 is not None:
            results.append({
                'text_tagged': tagged,
                'text_original': sentence,
                'label': label,
                'species1': sp1,
                'species2': sp2,
                'has_both_species': True
            })

    return pd.DataFrame(results)


def build_diverse_dataset(
    eval_100_path: str,
    additional_files: list,
    output_dir: str,
    train_split: float = 0.7,
    seed: int = 42
):
    """
    Build diverse training dataset from multiple sources.

    Strategy:
    - Split eval_100 into 70% train / 30% test
    - Add all other annotated files to training
    - Use TaxoNERD for robust species extraction
    """
    print("=" * 70)
    print("Building Diverse LUKE Training Dataset")
    print("=" * 70)
    print(f"TaxoNERD available: {TAXONERD_AVAILABLE}")

    # Initialize extractor (use eco_md which is faster)
    print("\nInitializing TaxoNERD...")
    extractor = SpanExtractor(use_taxonerd=TAXONERD_AVAILABLE, taxonerd_model='en_ner_eco_md')

    # 1. Load eval_100 and split
    print("\n" + "=" * 50)
    print("1. Processing eval_100 (main evaluation set)")
    print("=" * 50)

    eval_100_df = load_annotated_data(eval_100_path)
    if eval_100_df is None:
        raise ValueError("Could not load eval_100")

    # Split eval_100 into train/test
    eval_train, eval_test = train_test_split(
        eval_100_df,
        test_size=1 - train_split,
        stratify=eval_100_df['label'],
        random_state=seed
    )

    print(f"\nEval_100 split: {len(eval_train)} train, {len(eval_test)} test")

    # Process with TaxoNERD
    eval_train_processed = process_with_taxonerd(eval_train, extractor)
    eval_test_processed = process_with_taxonerd(eval_test, extractor)

    print(f"After TaxoNERD: {len(eval_train_processed)} train, {len(eval_test_processed)} test")

    # 2. Load additional annotated files
    print("\n" + "=" * 50)
    print("2. Processing additional annotated files")
    print("=" * 50)

    additional_dfs = []
    for file_path in additional_files:
        if os.path.exists(file_path):
            df = load_annotated_data(file_path)
            if df is not None and len(df) > 0:
                processed = process_with_taxonerd(df, extractor)
                additional_dfs.append(processed)
                print(f"  After TaxoNERD: {len(processed)} valid samples")

    # 3. Combine all training data
    print("\n" + "=" * 50)
    print("3. Combining datasets")
    print("=" * 50)

    all_train = [eval_train_processed] + additional_dfs
    train_df = pd.concat(all_train, ignore_index=True)

    # Remove duplicates based on text_original
    train_df = train_df.drop_duplicates(subset=['text_original'])

    print(f"Total training samples: {len(train_df)}")
    print(f"Training label distribution: {train_df['label'].value_counts().to_dict()}")

    test_df = eval_test_processed
    print(f"Test samples (held out from eval_100): {len(test_df)}")
    print(f"Test label distribution: {test_df['label'].value_counts().to_dict()}")

    # 4. Save datasets
    print("\n" + "=" * 50)
    print("4. Saving datasets")
    print("=" * 50)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "training_data_luke_diverse.csv")
    test_path = os.path.join(output_dir, "test_data_luke_diverse.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")

    # Also save a combined version with the original training data
    print("\n" + "=" * 50)
    print("5. Combining with original training data")
    print("=" * 50)

    # Load original LUKE training data if it exists
    original_luke_path = os.path.join(output_dir, "training_data_luke.csv")
    if os.path.exists(original_luke_path):
        original_df = pd.read_csv(original_luke_path)
        print(f"Original training data: {len(original_df)} samples")

        # Combine with new diverse data
        combined_df = pd.concat([
            original_df[['text_tagged', 'text_original', 'label', 'species1', 'species2']],
            train_df[['text_tagged', 'text_original', 'label', 'species1', 'species2']]
        ], ignore_index=True)

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text_original'])

        combined_path = os.path.join(output_dir, "training_data_luke_combined.csv")
        combined_df.to_csv(combined_path, index=False)

        print(f"Combined training data: {len(combined_df)} samples")
        print(f"Combined data saved to: {combined_path}")
    else:
        print("No original training data found, using diverse data only")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Using TaxoNERD: {TAXONERD_AVAILABLE}")

    return train_df, test_df


if __name__ == "__main__":
    # Paths
    BASE_DIR = "/home/egaillac/MetaP/classifier"
    EVAL_DIR = os.path.join(BASE_DIR, "data/evaluation")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data/training")

    # Main evaluation file
    eval_100_path = os.path.join(EVAL_DIR, "biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv")

    # Additional annotated files
    additional_files = [
        os.path.join(EVAL_DIR, "globi-passage_passages-triplets_2024-02-28_curation_EP.tsv"),
        os.path.join(EVAL_DIR, "globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"),
        os.path.join(EVAL_DIR, "biotx-random_passages-triplets_2024-04-22b_curation_EP_50best-multiples.tsv"),
        os.path.join(EVAL_DIR, "biotx-random_passages-triplets_2024-05-15_curation_EP_50nomultiple.tsv"),
    ]

    # Build dataset
    build_diverse_dataset(
        eval_100_path=eval_100_path,
        additional_files=additional_files,
        output_dir=OUTPUT_DIR,
        train_split=0.7,
        seed=42
    )

    print("\n" + "=" * 70)
    print("NEXT STEP: Train LUKE model on diverse data")
    print("python scripts/train_luke_interaction.py --input data/training/training_data_luke_diverse.csv")
    print("=" * 70)
