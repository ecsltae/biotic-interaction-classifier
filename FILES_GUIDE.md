# Complete Files Guide - Biotic Interaction Classifier

**Generated:** December 22, 2025

---

## 📊 Prediction CSV Files (Like Your Screenshot)

**Location:** `results/predictions/`

### Main Prediction Files:

1. **predictions_Ensemble_F1optimized.csv** ⭐ **RECOMMENDED**
   - Format: Same as your screenshot
   - Columns: sentence, True_label, Ensemble_prediction, True_sentiment, Ensemble_sentiment, Ensemble_probability
   - Shows ensemble predictions with probabilities
   - 100 rows (one per test sentence)

2. **predictions_BiomedBERT.csv**
   - BiomedBERT model predictions only
   - Same format as above

3. **predictions_RoBERTa.csv**
   - RoBERTa model predictions only
   - Same format as above

4. **predictions_ALL_MODELS_comparison.csv** ⭐ **COMPLETE COMPARISON**
   - All 3 models side-by-side
   - Includes: BiomedBERT, RoBERTa, Ensemble predictions
   - Shows which predictions agree/disagree
   - Columns include correct/incorrect flags

### Error Analysis Files:

5. **errors_FalsePositives_Ensemble_F1opt.csv**
   - 10 sentences where ensemble predicted positive but were actually negative
   - Shows overconfident predictions

6. **errors_FalseNegatives_Ensemble_F1opt.csv**
   - 13 sentences where ensemble missed actual interactions
   - Shows underconfident predictions

7. **errors_FalsePositives_BiomedBERT.csv**
   - 7 false positives from BiomedBERT alone

8. **errors_FalseNegatives_BiomedBERT.csv**
   - 17 false negatives from BiomedBERT alone

9. **errors_FalsePositives_RoBERTa.csv**
   - 47 false positives from RoBERTa (very sensitive)

10. **errors_FalseNegatives_RoBERTa.csv**
    - 7 false negatives from RoBERTa (catches most interactions)

---

## 📈 PowerPoint Presentation

**File:** `Biotic_Interaction_Classifier_Presentation.pptx`

**17 Slides:**
1. Title Slide
2. Project Overview
3. Data Development Path (6k → 20k samples)
4. Models Evaluated (4 transformers)
5. Cross-Validation Results (table)
6. Key Finding: BiomedBERT Wins!
7. Real Test Set Results (100 sentences)
8. Ensemble Learning Strategy
9. Inference Speed Optimizations
10. Example: Correctly Identified Interaction ✓
11. Example: Correctly Rejected Non-Interaction ✓
12. Example: False Positive ✗
13. Example: False Negative ✗
14. Error Analysis Summary
15. Recommendations & Next Steps
16. Available Resources
17. Summary

---

## 🤖 Models & Code

### Trained Models (in `models/`):

- `transformer_BiomedBERT_model_enhanced_20k/` - Best individual model (256 MB)
- `transformer_roberta_model/` - High recall model
- `transformer_biobert_model/` - Alternative biomedical model
- `transformer_distilbert_model/` - Faster, lighter model

### Ensemble Model:

- `ensemble_model/` - Configuration and results
  - `ensemble_config.pkl` - Model weights and config
  - `ensemble_eval_results.csv` - Performance metrics
  - `eval_100_results.csv` - Results on real test set

### Source Code:

- `src/models/ensemble_classifier.py` - **NEW** Optimized ensemble
- `src/models/transformer_classifier.py` - Original training script
- `scripts/generate_all_predictions.py` - Generates prediction CSVs
- `scripts/generate_presentation.py` - Creates PowerPoint
- `scripts/evaluate_ensemble_on_real_test.py` - Evaluation script

---

## 📋 Results & Documentation

### Performance Summaries:

- `results/cv_results/transformer_cv_results.csv` - Cross-validation on 20k
- `results/cv_results/transformer_eval_results.csv` - CV evaluation metrics
- `ENSEMBLE_RESULTS_SUMMARY.md` - Complete results document

### Main Documentation:

- `FILES_GUIDE.md` - This file
- `README.md` - Project README
- `ENSEMBLE_RESULTS_SUMMARY.md` - Detailed results

---

## 📊 How to Use the Prediction CSVs

### Open in Excel/LibreOffice:

1. Navigate to `results/predictions/`
2. Open `predictions_Ensemble_F1optimized.csv`
3. You'll see columns like your screenshot:
   - **sentence**: The text being classified
   - **True_label**: Actual label (0=negative, 1=positive)
   - **Ensemble_prediction**: Model's prediction (0 or 1)
   - **True_sentiment**: "positive" or "negative"
   - **Ensemble_sentiment**: Model's sentiment prediction
   - **Ensemble_probability**: Confidence score (0.0 to 1.0)

### Color Highlighting (Optional):

To match your screenshot, you can:
1. Highlight rows where `True_label != Ensemble_prediction` (errors) in orange
2. Use conditional formatting on probability column
3. Bold the header row

---

## 🎯 Quick Stats

### Ensemble Model (F1-Optimized):
- **Precision**: 50.0%
- **Recall**: 43.5%
- **F1**: 46.5%
- **Accuracy**: 77.0%

### BiomedBERT Alone:
- **Precision**: 46.2%
- **Recall**: 26.1%
- **F1**: 33.3%
- **Accuracy**: 76.0%

### RoBERTa Alone:
- **Precision**: 25.4%
- **Recall**: 69.6%
- **F1**: 37.2%
- **Accuracy**: 46.0%

---

## 🔍 Finding Specific Examples

### To find good classifications:
```python
import pandas as pd
df = pd.read_csv('results/predictions/predictions_Ensemble_F1optimized.csv')

# Correctly identified interactions
correct_positives = df[(df['True_label'] == 1) & (df['Ensemble_prediction'] == 1)]

# Correctly rejected non-interactions
correct_negatives = df[(df['True_label'] == 0) & (df['Ensemble_prediction'] == 0)]
```

### To find errors:
```python
# False positives (predicted interaction, but wrong)
false_positives = df[(df['True_label'] == 0) & (df['Ensemble_prediction'] == 1)]

# False negatives (missed interaction)
false_negatives = df[(df['True_label'] == 1) & (df['Ensemble_prediction'] == 0)]
```

---

## 📧 File Locations Summary

All files are in: `/home/egaillac/MetaP/classifier/`

**Most Important Files:**
1. `results/predictions/predictions_Ensemble_F1optimized.csv` - Main predictions
2. `Biotic_Interaction_Classifier_Presentation.pptx` - PowerPoint
3. `results/predictions/predictions_ALL_MODELS_comparison.csv` - Full comparison
4. `ENSEMBLE_RESULTS_SUMMARY.md` - Complete results doc

**For Presentation:**
- Use PowerPoint (already has examples)
- Reference prediction CSVs for specific sentences
- Show error analysis files for model limitations

---

**Questions?** All scripts are in `scripts/` directory and can be re-run to regenerate files.