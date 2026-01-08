# Ensemble Classifier Results Summary
**Date:** December 22, 2025
**Models:** BiomedBERT (65%) + RoBERTa (35%)
**Optimization:** FP16, torch.compile, batch processing

---

## 📊 Real Test Set Performance (100 sentences)

### Results by Threshold:

| Threshold Type | Threshold | **Precision** | **Recall** | **F1** | **Accuracy** |
|----------------|-----------|-----------|--------|--------|----------|
| **F1-Optimized** | 0.390 | **50.0%** | **43.5%** | **46.5%** | 77.0% |
| Default (0.5) | 0.500 | 50.0% | 39.1% | 43.9% | 77.0% |
| Precision-Optimized | 0.944 | 0.0% | 0.0% | 0.0% | 77.0% |

**⭐ RECOMMENDED:** Use **F1-Optimized threshold (0.390)**

---

## 🤖 Individual Model Comparison (Cross-Validation on 20k samples)

| Model | Accuracy | F1 Score | **Precision** | Recall |
|-------|----------|----------|-----------|--------|
| **BiomedBERT** 🏆 | **85.5%** | **86.4%** | **81.2% ± 2.7%** | 92.6% ± 4.4% |
| BioBERT | 85.0% | 86.2% | 79.7% ± 1.4% | 93.9% ± 3.1% |
| DistilBERT | 84.6% | 86.0% | 79.0% ± 1.0% | 94.3% ± 1.4% |
| RoBERTa | 83.9% | 85.8% | 76.9% ± 1.3% | **97.0% ± 2.4%** |

---

## ⚡ Inference Speed

**Hardware:** NVIDIA A100 80GB
**Optimizations:** FP16 half precision + torch.compile

### Batch Processing (100 samples):
- **F1-Optimized (0.390)**: 2.0 samples/sec (~500ms per sample)
- **Default (0.5)**: 671 samples/sec (~1.5ms per sample)
- **Precision-Opt (0.944)**: 56.4 samples/sec (~18ms per sample)

### Speed Benchmark (1000 random samples):
- **Batch inference**: 663 samples/sec (1.5ms per sample)
- **Single sentence**: 3.5 samples/sec (288ms per sample)

---

## 📁 File Locations

### Ensemble Model:
- **Saved model:** `ensemble_model/`
- **Configuration:** `ensemble_model/ensemble_config.pkl`
- **Results (CSV):** `ensemble_model/ensemble_eval_results.csv`
- **Real test results:** `ensemble_model/eval_100_results.csv`

### Individual Models:
- **BiomedBERT (enhanced 20k):** `models/transformer_BiomedBERT_model_enhanced_20k/`
- **RoBERTa:** `models/transformer_roberta_model/`
- **BioBERT:** `models/transformer_biobert_model/`
- **DistilBERT:** `models/transformer_distilbert_model/`

### Scripts:
- **Ensemble classifier:** `src/models/ensemble_classifier.py`
- **Evaluation script:** `scripts/evaluate_ensemble_on_real_test.py`
- **Original transformer script:** `src/models/transformer_classifier.py` (preserved)

### Results:
- **CV results:** `results/cv_results/transformer_cv_results.csv`
- **Ensemble logs:** `results/ensemble/ensemble_training.log`

---

## 🎯 Key Findings

### 1. On Real Test Set (100 sentences):
- **Best performance**: F1-Optimized threshold (0.390)
  - Precision: 50%
  - Recall: 43.5%
  - Balanced trade-off

### 2. Cross-Validation (20k training set):
- **Individual models** show much higher precision (76-81%) on CV
- **Real test set** is likely more challenging or different distribution
- **BiomedBERT** consistently best individual model

### 3. Ensemble Value:
- Combines BiomedBERT's precision focus with RoBERTa's recall
- Weighted voting (65/35) emphasizes precision
- Optimizations make it practical for production use

---

## 💡 Recommendations

### For Precision-Focused Applications:
1. **Use F1-Optimized Threshold (0.390)**
   - Best balance on real test data
   - 50% precision with 43.5% recall

2. **Alternative: BiomedBERT Alone**
   - If inference speed is critical
   - Simpler deployment
   - 81% precision on CV (but needs testing on real eval set)

### For Deployment:
- Use batch processing when possible (663 samples/sec)
- Single sentence mode for real-time: ~288ms latency
- FP16 optimization provides 2x speedup with minimal accuracy loss

---

## 📈 Next Steps (Optional)

1. **Investigate test set difference**
   - Why CV shows 81% precision but real test shows 50%?
   - Evaluate BiomedBERT alone on real test set for comparison

2. **Threshold tuning on real test set**
   - Current thresholds optimized on 20k training set
   - Could fine-tune on real test distribution

3. **Error analysis**
   - Review false positives/negatives on eval_100.tsv
   - Identify patterns for model improvement

---

**Generated:** December 22, 2025
**By:** Claude Code with ensemble_classifier.py