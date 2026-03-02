#!/usr/bin/env python3
"""
Build LUKE-formatted Dataset for Species Interaction Classification.

Takes the existing training data (CSV with text, label) and adds:
1. Entity-tagged text with <e1>...</e1> and <e2>...</e2> markers
2. Extracted species names
3. Character spans for LUKE model

Usage:
    python build_luke_dataset.py --input training_data_globi_v3.csv --output training_data_luke.csv
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.span_extractor import (
    SpanExtractor,
    extract_and_tag,
    batch_extract_and_tag,
    TAXONERD_AVAILABLE
)


def build_luke_dataset(
    input_path: str,
    output_path: str,
    text_column: str = "text",
    label_column: str = "label",
    use_taxonerd: bool = True,
    drop_invalid: bool = True,
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Convert standard training data to LUKE format with entity tags.

    Args:
        input_path: Path to input CSV
        output_path: Path to save output CSV
        text_column: Name of text column in input
        label_column: Name of label column in input
        use_taxonerd: Whether to use TaxoNERD for species extraction
        drop_invalid: Whether to drop rows where <2 species found
        max_samples: Limit number of samples (for testing)

    Returns:
        DataFrame with LUKE-formatted data
    """
    print("=" * 60)
    print("Building LUKE-formatted Dataset")
    print("=" * 60)

    # Load input data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)

    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples (testing mode)")

    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Validate columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in data")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in data")

    # Initialize extractor
    print(f"\nInitializing species extractor...")
    print(f"TaxoNERD available: {TAXONERD_AVAILABLE}")
    print(f"Using TaxoNERD: {use_taxonerd and TAXONERD_AVAILABLE}")

    extractor = SpanExtractor(use_taxonerd=use_taxonerd and TAXONERD_AVAILABLE)

    # Process each row
    print(f"\nExtracting species and tagging entities...")

    tagged_texts = []
    species1_list = []
    species2_list = []
    valid_mask = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = str(row[text_column])

        # Extract and tag
        tagged, sp1, sp2 = extract_and_tag(text, extractor, return_species=True)

        tagged_texts.append(tagged)
        species1_list.append(sp1)
        species2_list.append(sp2)

        # Check if valid (has both species)
        is_valid = sp1 is not None and sp2 is not None
        valid_mask.append(is_valid)

    # Add columns to dataframe
    df['text_tagged'] = tagged_texts
    df['text_original'] = df[text_column]
    df['species1'] = species1_list
    df['species2'] = species2_list
    df['has_both_species'] = valid_mask

    # Statistics
    valid_count = sum(valid_mask)
    invalid_count = len(valid_mask) - valid_count
    print(f"\nExtraction results:")
    print(f"  Valid (2 species found): {valid_count} ({100*valid_count/len(df):.1f}%)")
    print(f"  Invalid (<2 species): {invalid_count} ({100*invalid_count/len(df):.1f}%)")

    # Optionally drop invalid rows
    if drop_invalid:
        df_valid = df[df['has_both_species']].copy()
        print(f"\nDropped {invalid_count} invalid rows")
        df = df_valid
    else:
        print(f"\nKept all rows (including {invalid_count} without 2 species)")

    # Select and reorder columns for output
    output_columns = [
        'text_tagged',
        'text_original',
        label_column,
        'species1',
        'species2',
        'has_both_species'
    ]

    # Add any other columns from original data
    for col in df.columns:
        if col not in output_columns and col != text_column:
            output_columns.append(col)

    df_output = df[output_columns].copy()

    # Final statistics by label
    print(f"\nFinal dataset statistics:")
    print(f"  Total samples: {len(df_output)}")
    print(f"  Label distribution:")
    for label, count in df_output[label_column].value_counts().items():
        print(f"    Label {label}: {count} ({100*count/len(df_output):.1f}%)")

    # Save output
    df_output.to_csv(output_path, index=False)
    print(f"\nSaved LUKE dataset to: {output_path}")

    # Save metadata
    metadata = {
        'source_file': input_path,
        'total_samples': len(df_output),
        'valid_samples': valid_count,
        'dropped_invalid': drop_invalid,
        'used_taxonerd': use_taxonerd and TAXONERD_AVAILABLE,
        'label_distribution': df_output[label_column].value_counts().to_dict()
    }

    import json
    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    return df_output


def main():
    parser = argparse.ArgumentParser(
        description="Build LUKE-formatted dataset with entity tags"
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/training/training_data_globi_v3.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/training/training_data_luke.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of text column in input CSV"
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of label column in input CSV"
    )
    parser.add_argument(
        "--no-taxonerd",
        action="store_true",
        help="Disable TaxoNERD (use regex only)"
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep rows where <2 species were found"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for testing"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 1000 samples"
    )

    args = parser.parse_args()

    # Test mode
    max_samples = 1000 if args.test else args.max_samples

    # Build dataset
    build_luke_dataset(
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        label_column=args.label_column,
        use_taxonerd=not args.no_taxonerd,
        drop_invalid=not args.keep_invalid,
        max_samples=max_samples
    )

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Train LUKE model with train_luke_interaction.py")


if __name__ == "__main__":
    main()
