#!/usr/bin/env python3
"""
Evaluate ensemble on the real 100-sentence test set
"""
import sys
sys.path.insert(0, '/home/egaillac/MetaP/classifier/src/models')

import pandas as pd
from ensemble_classifier import OptimizedEnsembleClassifier

# Load the real test set
print("Loading eval_100.tsv...")
test_df = pd.read_csv('/home/egaillac/MetaP/classifier/data/evaluation/eval_100.tsv', sep='\t')
print(f"Loaded {len(test_df)} samples")

# Prepare data
test_texts = test_df['sentence'].tolist()
test_labels = test_df['evaluation_interaction_identified'].tolist()

# Load ensemble
print("\nLoading ensemble classifier...")
ensemble = OptimizedEnsembleClassifier(optimize=True)

# Evaluate with F1-optimized threshold (from training)
print("\n" + "="*70)
print("EVALUATION ON REAL 100-SENTENCE TEST SET")
print("="*70)

print("\n1. F1-Optimized Threshold (0.390):")
metrics_f1 = ensemble.evaluate(test_texts, test_labels, threshold=0.389886474609375)

print("\n2. Precision-Optimized Threshold (0.944):")
metrics_prec = ensemble.evaluate(test_texts, test_labels, threshold=0.9442626953125)

print("\n3. Default Threshold (0.5):")
metrics_default = ensemble.evaluate(test_texts, test_labels, threshold=0.5)

# Save results
results_df = pd.DataFrame({
    'Threshold_Type': ['F1-Optimized', 'Precision-Optimized', 'Default'],
    'Threshold': [0.389886474609375, 0.9442626953125, 0.5],
    'Precision': [metrics_f1['precision'], metrics_prec['precision'], metrics_default['precision']],
    'Recall': [metrics_f1['recall'], metrics_prec['recall'], metrics_default['recall']],
    'F1': [metrics_f1['f1'], metrics_prec['f1'], metrics_default['f1']],
    'Accuracy': [metrics_f1['accuracy'], metrics_prec['accuracy'], metrics_default['accuracy']],
})

output_path = '/home/egaillac/MetaP/classifier/ensemble_model/eval_100_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\n✓ Results saved to {output_path}")