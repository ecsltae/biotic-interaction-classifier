# Script to calculate ROUGE metric score for manual input

import evaluate

# Load the ROUGE metric
rouge = evaluate.load("rouge")

pred = input("Prediction: ").strip()
ref = input("Reference: ").strip()

scores = rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)
for k, v in scores.items():
    print(f"{k.upper()}: {round(v * 100, 4)}")