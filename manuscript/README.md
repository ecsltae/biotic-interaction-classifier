# Manuscript: Biotic Interaction Classifier

## Files

- `biotic_interaction_classifier.tex` - Main LaTeX manuscript
- `references.bib` - Bibliography file
- `figures/` - Directory for figures (create as needed)

## Compiling

To compile the manuscript:

```bash
cd /home/egaillac/MetaP/classifier/manuscript
pdflatex biotic_interaction_classifier.tex
bibtex biotic_interaction_classifier
pdflatex biotic_interaction_classifier.tex
pdflatex biotic_interaction_classifier.tex
```

Or use a LaTeX editor like Overleaf, TeXstudio, or VS Code with LaTeX Workshop extension.

## TODO

### Content to Add

1. **Results Tables** - Fill in TBD values with actual results from:
   - `../transformer_cv_results.csv` (cross-validation results)
   - `../transformer_eval_results.csv` (evaluation set results)

2. **Figures to Create**:
   - Figure 1: Dataset composition pie chart
   - Figure 2: Comparison of false positive rates (baseline vs. improved negatives)
   - Figure 3: Precision-recall curves for different models
   - Figure 4: BioTXplorer interface screenshot showing classifier integration

3. **References to Update**:
   - Replace example placeholders in `references.bib`
   - Add relevant ecological interaction papers
   - Add previous biotic interaction extraction work
   - Add BioTXplorer/SIBILS references

4. **Sections to Expand**:
   - Add actual user statistics from BioTXplorer integration
   - Include comparison with previous methods (if applicable)
   - Add qualitative examples of correctly classified sentences

### When Results Are Available

Run this to get your final metrics:
```bash
cat ../transformer_cv_results.csv
cat ../transformer_eval_results.csv
```

Then update Tables 1 and 2 in the manuscript.

## Journal Target

Consider submitting to:
- BMC Bioinformatics
- Bioinformatics
- PLoS Computational Biology
- Ecological Informatics
- Database
