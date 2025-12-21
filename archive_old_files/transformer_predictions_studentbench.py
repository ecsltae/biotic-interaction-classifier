#!/usr/bin/env python3
"""
Script to generate a CSV with predictions from transformer_BiomedBERT_model
using BiotXBench_20250111_QualityCheck_Annotations.xlsx
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the test set from Excel
def load_test_set(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# Load the model and tokenizer
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

# Make predictions
def predict_sentences(model, tokenizer, sentences, device):
    predictions = []
    for sentence in sentences:
        inputs = tokenizer(
            str(sentence),
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()

        predictions.append(pred)

    return predictions

# Main function
def main():
    # Load test set
    test_set_path = 'BiotXBench_20250111_QualityCheck_Annotations.xlsx'
    df = load_test_set(test_set_path, sheet_name="Sentence - All")

    # Extract sentences and labels
    sentences = df.iloc[:, 1].astype(str).tolist()  # Second column
    df['label'] = df.iloc[:, 5].apply(lambda x: 1 if pd.notna(x) else 0)  # 6th column for positive

    # Load model
    model_path = 'transformer_BiomedBERT_model'
    model, tokenizer, device = load_model(model_path)

    # Make predictions
    predictions = predict_sentences(model, tokenizer, sentences, device)

    # Map numerical labels to 'positive' and 'negative'
    df['true_sentiment'] = df['label'].map({1: 'positive', 0: 'negative'})
    df['BiomedBERT_prediction'] = predictions
    df['BiomedBERT_sentiment'] = df['BiomedBERT_prediction'].map({1: 'positive', 0: 'negative'})

    # Rename the second column to 'sentence' if it's not already named so
    df = df.rename(columns={df.columns[1]: 'sentence'})

    # Save the results to an Excel file
    output_file = 'predictions_with_BiomedBERT.xlsx'
    df.to_excel(output_file, columns=['sentence', 'label', 'BiomedBERT_prediction', 'true_sentiment', 'BiomedBERT_sentiment'],
                index=False, sheet_name='Predictions')
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
