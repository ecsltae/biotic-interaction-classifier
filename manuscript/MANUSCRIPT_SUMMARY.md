# Manuscript and Figures Summary

## What's Been Created

### 1. LaTeX Manuscript ✓
**File:** `biotic_interaction_classifier.tex`

Complete academic paper with:
- Title, abstract, keywords
- Introduction (background, previous work, contribution)
- Methods (dataset construction, model architecture, training, evaluation)
- Results (placeholder tables for your actual results)
- Discussion (findings, limitations, future work)
- Conclusion
- Bibliography

**Status:** Ready to compile, needs actual results filled in

### 2. Bibliography ✓
**File:** `references.bib`

Contains key references for:
- BERT, BioBERT, BiomedBERT
- DistilBERT, RoBERTa
- Transformers library
- Placeholder references (need to be replaced with domain-specific papers)

### 3. Manuscript Figures (4 figures) ✓

All in both PNG (high-res) and PDF format:

1. **Figure 1: Dataset Composition** - Shows your 20k balanced dataset and negative sampling strategy
2. **Figure 2: Performance Comparison** - Bar charts comparing models across metrics
3. **Figure 3: False Positive Reduction** - Impact of strategic negative sampling
4. **Figure 4: Precision-Recall Curves** - PR curves for all models

### 4. Additional Visualizations (5 figures) ✓

For presentations, supplementary materials, or detailed analysis:

1. **Confusion Matrix** - Detailed performance breakdown
2. **Prediction Comparison Table** - Like your Excel screenshot showing sentence-by-sentence results
3. **Training Curves** - 4-panel figure showing training dynamics across folds
4. **Error Analysis** - FP vs FN distribution and breakdown
5. **Performance Heatmap** - Model performance across data types

### 5. Scripts for Future Updates ✓

- `create_manuscript_figures.py` - Regenerate manuscript figures
- `create_prediction_visualizations.py` - Create detailed analysis figures
- `visualize_actual_predictions.py` - Load real CSV data and create visualizations

## What to Do Next

### Immediate (When Training Completes)

1. **Get your results:**
   ```bash
   cat transformer_cv_results.csv
   cat transformer_eval_results.csv
   ```

2. **Update manuscript tables:**
   - Table 1: Cross-validation results (lines ~100-110 in .tex)
   - Table 2: Evaluation set results (lines ~115-125 in .tex)

3. **Visualize actual predictions:**
   ```bash
   python scripts/visualize_actual_predictions.py path/to/predictions.csv
   ```

### Before Submission

1. **Update bibliography:**
   - Add relevant ecological interaction papers
   - Add previous NLP/text mining work in bioinformatics
   - Add BioTXplorer/SIBILS references
   - Replace placeholder references

2. **Update figure data:**
   - Edit `create_manuscript_figures.py` with actual metrics
   - Regenerate figures with real data
   - Replace placeholder percentages in Figure 3

3. **Add BioTXplorer details:**
   - User statistics if available
   - Screenshot of interface (save as figure5_biotxplorer.png)
   - Integration architecture diagram

4. **Proofread and expand:**
   - Add specific examples of correct/incorrect classifications
   - Expand comparison with previous work
   - Add author contributions, funding, etc.

## Compiling the Manuscript

```bash
cd /home/egaillac/MetaP/classifier/manuscript
pdflatex biotic_interaction_classifier.tex
bibtex biotic_interaction_classifier
pdflatex biotic_interaction_classifier.tex
pdflatex biotic_interaction_classifier.tex
```

Or upload to Overleaf for easier editing.

## File Locations

```
classifier/
├── manuscript/
│   ├── biotic_interaction_classifier.tex  # Main paper
│   ├── references.bib                      # Bibliography
│   ├── figures/                            # 4 manuscript figures (PNG + PDF)
│   ├── README.md                           # Compilation instructions
│   ├── INSERT_FIGURES.txt                  # LaTeX code to insert figures
│   └── MANUSCRIPT_SUMMARY.md               # This file
│
├── figures/                                # 5 additional visualizations
│
├── scripts/
│   ├── create_manuscript_figures.py
│   ├── create_prediction_visualizations.py
│   └── visualize_actual_predictions.py
│
└── FIGURES_README.md                       # Complete figure documentation
```

## Target Journals

Consider these for submission:
- **BMC Bioinformatics** (open access, bioinformatics focus)
- **Bioinformatics** (Oxford, high impact)
- **PLoS Computational Biology** (open access, computational focus)
- **Ecological Informatics** (ecology + informatics)
- **Database** (database/tool paper)

## Current Status

✅ Manuscript structure complete
✅ All figures created (with placeholder data)
✅ Bibliography framework ready
✅ Visualization scripts ready

⏳ Training running (DistilBERT on improved 20k dataset)
⏳ Waiting for final results to update tables

📋 TODO: Add BioTXplorer details, update references, fill in real metrics
