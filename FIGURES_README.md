# Classifier Figures and Visualizations

This document describes all figures created for the biotic interaction classifier project.

## Manuscript Figures

Located in: `/home/egaillac/MetaP/classifier/manuscript/figures/`

### Figure 1: Dataset Composition
**File:** `figure1_dataset_composition.png` (also PDF)

Two pie charts showing:
- Overall dataset balance (10k positive vs 10k negative)
- Negative examples breakdown (co-occurrence, scientific descriptions, multi-species lists, random)

### Figure 2: Performance Comparison
**File:** `figure2_performance_comparison.png` (also PDF)

Bar chart comparing Precision, Recall, and F1-Score across different models:
- DistilBERT
- BioBERT
- BiomedBERT
- RoBERTa

### Figure 3: False Positive Reduction
**File:** `figure3_false_positive_reduction.png` (also PDF)

Comparison showing impact of strategic negative sampling on false positive rates:
- Baseline (random negatives) vs Improved (diverse negatives)
- Broken down by category: co-occurrence, taxonomic, multi-species, overall

### Figure 4: Precision-Recall Curves
**File:** `figure4_precision_recall.png` (also PDF)

PR curves for all models with Area Under Curve (AUC) values

## Additional Visualizations

Located in: `/home/egaillac/MetaP/classifier/figures/`

### Confusion Matrix
**File:** `confusion_matrix.png`

Heatmap showing:
- True Negatives, False Positives
- False Negatives, True Positives
- Precision, Recall, F1, Accuracy metrics

### Prediction Comparison Table
**File:** `prediction_comparison_table.png`

Visual table showing sentence-by-sentence predictions:
- Original sentence
- True label vs Predicted label
- Color-coded (green=correct, red=incorrect)
- Mimics the Excel screenshot format

### Training Curves
**File:** `training_curves.png`

Four subplots showing:
1. Training loss across all folds
2. Validation F1-score across folds
3. Mean Precision & Recall with confidence intervals
4. Final performance by fold (bar chart)

### Error Analysis
**File:** `error_analysis.png`

Two plots:
1. Total false positives vs false negatives
2. Breakdown of false positive types

### Performance Heatmap
**File:** `performance_heatmap.png`

Heatmap showing F1-scores for each model across different data types:
- Scientific papers
- Abstracts
- Full text
- Mixed

## Actual Results Visualizations

After training completes, run:

```bash
cd /home/egaillac/MetaP/classifier
source /home/egaillac/MetaP/MPvenv/bin/activate
python scripts/visualize_actual_predictions.py
```

This will create:
- `actual_confusion_matrix.png` - Real confusion matrix from your data
- `actual_prediction_comparison.png` - Real prediction comparisons
- `actual_error_distribution.png` - Real error breakdown

## Regenerating Figures

### Manuscript Figures (with placeholder data)
```bash
python scripts/create_manuscript_figures.py
```

### Classifier Visualizations (with placeholder data)
```bash
python scripts/create_prediction_visualizations.py
```

### Real Data Visualizations (after training)
```bash
python scripts/visualize_actual_predictions.py path/to/predictions.csv
```

## Updating with Real Data

Once training completes:

1. Update `create_manuscript_figures.py` with actual metrics from:
   - `transformer_cv_results.csv`
   - `transformer_eval_results.csv`

2. Run the visualization scripts to regenerate with real data

3. Update the LaTeX manuscript with the new figures

## Figure Descriptions for Manuscript

### For Methods Section:
- Figure 1: Shows balanced dataset design and strategic negative sampling

### For Results Section:
- Figure 2: Model performance comparison
- Figure 3: Impact of improved negative sampling
- Figure 4: Precision-recall trade-offs
- Training curves (supplementary)
- Confusion matrix (supplementary)

### For Supplementary Materials:
- Error analysis
- Performance heatmap
- Prediction comparison examples
