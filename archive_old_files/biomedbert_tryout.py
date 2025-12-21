#!/usr/bin/env python3
"""
API for Biotic Interaction Classification using a Fine-Tuned BiomedBERT Model
===============================================================================
This script loads a pre-trained BiomedBERT model and provides a simple API
to classify sentences as positive (contains biotic interaction) or negative.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BioticInteractionClassifier:
    def __init__(self, model_path="transformer_BiomedBERT_model", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the classifier with the model and tokenizer.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()  # Set to evaluation mode

    def classify_sentence(self, sentence, threshold=0.5):
        """
        Classify a sentence as positive or negative for biotic interaction.
        Returns:
            - prediction: "Positive" or "Negative"
            - confidence: confidence score for the prediction
        """
        # Tokenize the input sentence
        inputs = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(self.device)

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0][1].item()  # Confidence for class 1 (Positive)
            prediction = "Positive" if confidence >= threshold else "Negative"

        return prediction, confidence

def main():
    # Initialize the classifier
    classifier = BioticInteractionClassifier()

    # Example usage
    while True:
        sentence = input("\nEnter a sentence to classify (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break

        prediction, confidence = classifier.classify_sentence(sentence)
        print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
