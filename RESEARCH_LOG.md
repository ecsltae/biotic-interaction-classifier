# MetaP Classifier — Research Log

Living document. Updated as work progresses.
Last updated: 2026-04-20.

---

## Quick Reference — Current Best Results

| Model | Dataset | Script | EP-relax F1 | Prec | Rec |
|-------|---------|--------|-------------|------|-----|
| BiomedBERT | v7 | train_cv_regularized | **0.788** | — | — |
| FLAN-T5-large (simple) | v12 | flan_t5_classifier | avg=0.780 / best=**0.800** | 0.737 | 0.875 |
| FLAN-T5-base | v11_1 | flan_t5_classifier | avg=0.781 / best=**0.818** | — | — |
| BiomedBERT cv_reg × FLAN-T5-base v12 | — | ensemble_biomedbert_flant5 | **0.857** (geometric) / **0.849** (arith) | — | — |
| BiomedBERT cv_reg × FLAN-T5-base v11.1 | — | ensemble_biomedbert_flant5 | **0.846** (arith) / 0.843 (geo) | — | — |
| BiomedBERT v11_reg × FLAN-T5-base v11.1 | — | ensemble | 0.827 (geo) | — | — |
| BiomedBERT | v15 (no v7) | train_cv_regularized (fixed) | avg=0.653 ±0.025 / best=0.696 | 0.599 | 0.721 | ← regression, see below |
| BiomedBERT | v15b (+ v7 Qwen, 61% templates) | train_cv_regularized (fixed) | avg=0.632 / best=0.660 | 0.627 | 0.637 |
| FLAN-T5-base | v15b (+ v7 Qwen, 61% templates) | flan_t5_classifier (fixed) | avg=0.628 / best=0.667 | — | — |
| BiomedBERT | v16 real-only (14,553 rows) | train_cv_regularized | avg=0.653 / best=0.696 | 0.599 | 0.721 |
| FLAN-T5-base | v16 real-only | flan_t5_classifier | — | — | — |
| BiomedBERT | v17a (balanced+4k hard_neg) | train_cv_regularized | avg=0.688 / best=0.748 | — | — |
| FLAN-T5-base | v17a | flan_t5_classifier | avg=0.583 / best=0.644 | 0.690 | 0.604 |
| BiomedBERT | v17b (Qwen recalibrated) | train_cv_regularized | avg=0.663 / best=0.692 | 0.668 | 0.662 |
| FLAN-T5-base | v17b (Qwen recalibrated) | flan_t5_classifier | avg=0.610 / best=0.695 | 0.702 | 0.688 |
| BiomedBERT | v18 hybrid (real+v7 gap-fill) | train_cv_regularized | avg=0.648 / best=0.701 | 0.699 | 0.608 |
| FLAN-T5-base | v18 hybrid | flan_t5_classifier | 0.619 (fine-tuned) | 0.722 | 0.542 |
| **Distilled BiomedBERT v2** | distilled from ensemble | distill_ensemble.py (T=2,α=0.5) | **0.808** | 0.750 | 0.875 |
| **Distilled BiomedBERT v3** | distilled from ensemble | distill_ensemble.py (T=4,α=0.9) | **0.808** | 0.784 | 0.833 |

**Current target:** beat F1=0.788 (BiomedBERT v7) / F1=0.818 (FLAN-T5-base v11.1).
**✅ Achieved:** Distilled BiomedBERT v2 = EP F1=0.808 — single model, same size as BiomedBERT, 3.3× faster than ensemble.
**BiomedBERT pattern (definitive, 2026-04-13):** Every addition of real sentences to v7 templates hurts BiomedBERT on EP-relax. v7 (templates only, F1=0.788) > v12 (0.729) > v17a (0.688) > v18 hybrid (0.648). The templates were exactly matched to EP-relax linguistic patterns. Real PMC sentences introduce distribution mismatch.
**FLAN-T5 pattern:** Diverse real sentences work best (v11.1 F1=0.818). Templates alone insufficient for T5.
**Teacher-student conclusion:** Qwen-labeled real sentences have NOT beaten v7 BiomedBERT or v11.1 FLAN-T5 in 6 training runs. The historical ensemble (v7 BiomedBERT × v11.1 FLAN-T5 = F1=0.865) remains the best result.
**⚠ EP-relax used as test above is now contaminated once eval_qwen_disagreements enter training — new test set being curated (200 sentences, v15_test_batch1/2 in curation queue).**

---

## Dataset Version History

### v1–v6 — Pure Templates (2023–early 2024)
- Fully synthetic sentences generated from GloBI species pairs
- v1: ~40K templates, no validation
- v2: added hard negatives
- v3–v4: minor refinements, 19 real sentences introduced
- v5–v6: balance tuning
- **Never used for final evaluation — baseline only**

### v7 — LLM-Validated Gold Standard ⭐
- 25,081 samples (29% pos, 71% neg)
- Validated by Claude API (Anthropic) — each positive confirmed by LLM
- **Best discriminative result: BiomedBERT F1=0.788 (EP-relax)**
- File: `data/training/training_data_globi_v7_llm_cleaned.csv`
- **Key lesson:** v7 positives are templates, not real PMC sentences. Qwen3.5-122B later assessed the 175 pathogenOf positives and rejected 63% as too formulaic (e.g. "Laboratory experiments demonstrated that disease caused by X in Y was characterized in wild populations").

### v8–v9 — Regression (2024)
- v8: SIBiLS infection-biased sentences → F1=0.695
- v9: Regex-labeled noise → F1=0.644
- **Lesson:** Noisy auto-labeling hurts more than it helps

### v10 — First Real PMC Sentences (2024)
- Added real sentences from Europe PMC via `fetch_epmc_direct.py`
- Problem: 92% of positives were pathogen/infection → model became biased
- BiomedBERT F1=0.722 (worse than v7)
- **Lesson:** Diversity of interaction types matters as much as volume

### v10.1 / v11_1 — Diverse Real Sentences
- Targeted harvesting by ecological category (predation, herbivory, pollination, symbiosis)
- v10.1: ecologically balanced, F1 recovered
- v11_1: FLAN-T5-base best result: avg F1=0.781, best fold F1=0.818
- Files: `data/training/training_data_v10_1.csv`, `data/training/training_data_v11_1.csv`

### v12 — Expanded Real Sentences
- 27,652 samples: v7_llm_cleaned + epmc_direct + globi_pmc_v2 + external_db
- Score>0 filter applied as LLM validation proxy → MISTAKE (removed valid implicit interactions)
- BiomedBERT F1=0.729, FLAN-T5-large F1=0.800
- **Lesson:** score>0 filter is NOT a good proxy for LLM validation

### v14 — Latest Before Teacher Approach
- Built 2026-03-15, SIBiLS over-pruned again
- FLAN-T5-base F1=0.706 → regression
- **Root cause confirmed:** same score>0 filter issue

### v15 — Teacher-Labeled (ASSEMBLED 2026-04-03) ← current work
See section below. First training run done (F1=0.653 — regression). v15b in progress overnight.

---

## Teacher-Student Approach (2026-03-27 to present)

### Concept
Use a large free LLM (Qwen3.5-122B, running locally via ollama) as **teacher** to label all collected sentences. Train lightweight student models (BiomedBERT, FLAN-T5-base) on those labels.

**Why not use BiomedBERT+T5 as teacher:** circular — they are the students. The teacher must be independent and larger.

**Why Qwen3.5-122B:** 125B parameter MoE model, free (local), strict semantic understanding. Rejects co-occurrence sentences that merely mention organisms without describing an interaction.

### Full Corpus Labeling (completed 2026-03-28)
- Script: `classifier/scripts/teacher_label_full.py`
- Input: 6 source files, 44,178 unique sentences
- Output: `classifier/results/research_agent/all_sources_qwen122b_labeled.csv`
- Runtime: 37.6 hours at ~3s/row on 80GB + 20GB A100
- **Results: 4,065 YES (9.2%) / 40,113 NO (90.8%)**

| Source | Rows | YES | Rate |
|--------|------|-----|------|
| epmc_direct_sentences.csv | 19,518 | 1,980 | 10.1% |
| epmc_direct_sentences_v2.csv | 13,631 | 1,305 | 9.6% |
| globi_pmc_real_sentences.csv | 392 | 69 | 17.6% |
| globi_pmc_sentences_v2.csv | 275 | 48 | 17.5% |
| external_db_sentences.csv | 268 | 23 | 8.6% |
| globi_sibils_real.csv | 10,094 | 640 | 6.3% |

The 9.2% positive rate reflects Qwen's strictness — it correctly rejects:
- Sentences mentioning organisms without describing an interaction
- Methodology sentences ("we tested X against Y")
- Background/review sentences ("X is known to be a pathogen of Y" without active interaction)

**Key insight:** GloBI-PMC sources (17-18% rate) are most reliable — articles were retrieved because GloBI cited them as interaction evidence.

### Negation Handling
Checked teacher positives for negated interactions ("cats don't eat dogs" = label 0).
- 288/4,065 (7.1%) contain negation words
- Only ~8 are direct negations of the interaction itself
- Qwen's prompt explicitly requires "an actual interaction occurring" — handles negations correctly
- Error rate < 0.2% — negligible

---

## V15 Dataset Build

### Architecture
```
Step 1: validate_eval_sets.py     → Qwen-validates all 7 eval files (DONE)
Step 2: build_negative_pool.py    → scores NO-labels with lexicon, rechecks strong-signal (DONE)
Step 3: human curation            → eval disagreements + pathogen borderline (IN PROGRESS)
Step 4: validate_v7_with_qwen.py  → Qwen-validates ~7k v7 non-pathogenOf templates (RUNNING overnight)
Step 5: assemble_v15_dataset.py   → single dataset.csv for CV, 200 test candidates held out (DONE, reruns after step 4)
Step 6: train_cv_regularized.py   → BiomedBERT v15b (QUEUED after step 4)
```

### Step 1 — Eval Set Validation (DONE)
- Script: `classifier/scripts/validate_eval_sets.py`
- 599 unique sentences across 7 eval files validated by Qwen
- Output: `classifier/data/evaluation/eval_sets_qwen_validated.csv`
- **Agreement: 404/599 (67.4%)**
  - Gold=POS / Qwen=NEG: 179 (Qwen too strict on indirect/background statements)
  - Gold=NEG / Qwen=POS: 16 (possible gold label errors)
- Gold labels remain authoritative — disagreements imported to curation queue as `eval_qwen_disagreements`

### Step 2 — Negative Pool (DONE)
- Script: `classifier/scripts/build_negative_pool.py`
- Uses binary re-check (same YES/NO prompt) for strong-signal negatives
- **Key fix:** confidence prompt returned empty strings from Qwen — replaced with binary re-check
- **Key fix:** `keep_alive: -1` in all ollama requests prevents model eviction between calls
- Results:
  - Definitely clean (lexicon=0): **12,000** (sampled)
  - Confirmed clean (2× Qwen=NO, strong lexicon): **3,300** → all confirmed, 0 flipped
  - Weak signal (spot-check CSV): **12,177**
  - Needs curation: **0** — Qwen was 100% consistent on second pass
- Output: `classifier/data/training/negatives_clean.csv` (15,300 rows)

### pathogenOf — Special Handling
pathogenOf was underrepresented in teacher positives (only 87). Multiple actions taken:

1. **v7 pathogenOf audit:** Ran 175 v7 pathogenOf sentences through Qwen → only **64 accepted (37%)**. The rest are formulaic templates Qwen correctly rejects. Only the 64 Qwen-validated ones are used. File: `data/training/v7_pathogenOf_qwen_validated.csv`

2. **Targeted EPMC harvest:** Script `classifier/scripts/fetch_pathogen_sentences.py` — explicit pathogen-host pair queries ("X infects Y", "X is a pathogen of Y") → **56 new Qwen-confirmed positives**. File: `data/training/pathogen_harvested.csv`

3. **Human curation of borderline rejections:** 100 Qwen-rejected GloBI-PMC pathogenOf sentences with interaction verbs imported to curation queue. User reviewed 20 → **6 confirmed positives**. File: `data/training/curated_pathogen_borderline.csv`

**Final pathogenOf count: 64 + 87 + 56 + 6 = 213 positives** (all Qwen or human validated)

### Positive Sources Summary (v15 / v15b)

| Source | Count | Type | Validated by | In v15 | In v15b |
|--------|-------|------|-------------|--------|---------|
| Qwen3.5-122B teacher (all categories) | 4,065 | Real PMC | Qwen | ✓ | ✓ |
| v7 non-pathogenOf (Qwen-validated) | ~6,600 est. | Templates | Qwen | ✗ | ✓ (running) |
| v7 pathogenOf (Qwen-accepted only) | 64 | Templates | Qwen | ✓ | ✓ |
| EPMC targeted pathogen harvest | 56 | Real PMC | Qwen | ✓ | ✓ |
| Human-curated pathogen borderline | 6 | Real PMC | Human | ✓ | ✓ |
| eval_100 gold=POS (BiTeM/SIB) | 25 | Real PMC | Human | ✓ | ✓ |
| eval_qwen_disagreements (curated) | 0 → TBD | Real PMC | Human | ✗ | pending |
| **v15 total** | **~4,216** | | | | |
| **v15b total (est.)** | **~10,800** | | | | |

**Key decision (2026-04-03):** v7 non-pathogenOf was excluded from v15 (not Qwen-validated). Qwen is now validating all 7,076 sentences overnight (~93% YES rate so far → expect ~6,600 accepted). v15b will include these.

### Negative Sources Summary (v15)

| Source | Count | Validated by |
|--------|-------|-------------|
| Clean negatives (lexicon=0 + Qwen=NO) | 10,540 (sampled to 2.5× pos) | Qwen |
| eval_qwen_disagreements confirmed neg | 0 → TBD (post-curation) | Human |

Target neg:pos ratio: 2.5

### Test Set Strategy (changed 2026-04-03)
- **Old plan:** fixed test set = all 7 eval files (599 sentences)
- **New plan:** eval_qwen_disagreements go INTO training after curation → those eval files are no longer a valid test set
- **New test set:** 200 sentences randomly sampled from v15 pool (real sentences only, stratified by label), imported to curation queue as `v15_test_batch1` (100) + `v15_test_batch2` (100)
- User curates batch 1 first → gold test set for interim evaluation
- EP-relax (99 sentences) still used in training script for historical comparison but is being contaminated as eval sentences enter training

### Curation Status (2026-04-03)
| Queue | Pending | Approved | Skip | Notes |
|-------|---------|----------|------|-------|
| `eval_qwen_disagreements` | 162 | 27 (25 auto-approved gold=POS) | 6 | **Needs full curation** |
| `globi_pmc_pathogenOf_borderline` | 80 | 8 | 12 | Optional more pathogenOf |
| `v15_test_batch1` | 98 | 0 | 0 | **Curate first for test set** |
| `v15_test_batch2` | 95 | 0 | 0 | Curate later |
| `sibils_mongodb` | 0 | 380 | 43 | Old backlog, done |

60 uncertain items saved to `data/training/for_colleague_review.csv` for external review.

---

## V15 Training Results (2026-04-03)

### BiomedBERT v15 — REGRESSION (F1=0.653)
- Dataset: 14,555 rows (4,158 pos, 28.6%) — **no v7 non-pathogenOf**
- Script: `train_cv_regularized.py` (fixed — eval_100 injection removed)
- EP-relax test: **avg F1=0.653 ±0.025**, best fold=0.696

| Fold | Val F1 | EP F1 | Prec | Rec |
|------|--------|-------|------|-----|
| 1 | 0.857 | 0.642 | 0.586 | 0.708 |
| 2 | 0.857 | 0.654 | 0.607 | 0.708 |
| 3 | 0.840 | 0.696 | 0.597 | 0.833 |
| 4 | 0.845 | 0.620 | 0.596 | 0.646 |
| 5 | 0.839 | 0.654 | 0.607 | 0.708 |
| **avg** | **0.848** | **0.653** | 0.599 | 0.721 |

**Why regression (vs v7 F1=0.788):**
1. Only 4,216 positives (vs ~8k in v12) — removing v7 templates halved training signal
2. kleptoparasiteOf dominated 33% of positives (1,396/4,158) — model biased toward one category
3. Val F1=0.848 but EP F1=0.653 → big distribution mismatch: model fits EPMC style, EP test has different linguistic patterns
4. `train_cv_regularized.py` was injecting `eval_100.tsv` 5× into training (now fixed)

**Model saved:** `models/transformer_BiomedBERT_v15/`

### BiomedBERT v15b — DONE (2026-04-04, avg EP F1=0.632)
- Dataset: 25,945 rows (10,765 pos 41.5%) — **includes 6,635 v7 Qwen-validated templates (61.6% of pos)**
- Script: `train_cv_regularized.py` (eval_100 injection removed)
- EP-relax test: **avg F1=0.632 ±0.029**, best fold=0.660

| Fold | Val F1 | EP F1 | Prec | Rec |
|------|--------|-------|------|-----|
| 1 | 0.927 | 0.646 | 0.627 | 0.667 |
| 2 | 0.929 | 0.653 | 0.660 | 0.646 |
| 3 | 0.921 | 0.660 | 0.635 | 0.688 |
| 4 | 0.924 | 0.619 | 0.612 | 0.625 |
| 5 | — | 0.581 | 0.600 | 0.562 |
| **avg** | **0.925** | **0.632** | 0.627 | 0.637 |

**Model saved:** `models/transformer_BiomedBERT_v15b/`

### FLAN-T5-base v15b — DONE (2026-04-05, avg EP F1=0.628)
- Same dataset as BiomedBERT v15b (25,945 rows)
- Script: `src/models/flan_t5_classifier.py` (eval_100 injection removed), 6 epochs
- EP-relax test: **avg F1=0.628 ±0.034**, best fold=0.667
- Runtime: 750 min (~12.5h)
- **Model saved:** `models/flan_t5_base_v15b/`

### Root Cause Analysis — v15b Regression (F1=0.63 vs v7=0.788)

**Primary cause: 61.6% of positives are v7 templates**
- v15b has 6,635 Qwen-validated v7 templates out of 10,765 positives
- Templates ("X eats Y in foraging behavior") → model learns template linguistic patterns
- EP test = entirely real curated sentences → distribution mismatch
- Val F1=0.925 (fits training distribution) vs EP F1=0.632 (fails on real language) confirms this

**Secondary causes:**
- 41.5% positive rate (vs 29% in v7) → may hurt precision on balanced test sets
- Negatives are too easy (lexicon=0 only) → model doesn't learn hard negative patterns
- Qwen is less strict than Claude API was for v7: 93% YES rate on templates suggests it approves borderline cases Claude would reject

**Hypothesis to test:** train on **real sentences only** (no v7 templates) → should match or exceed v15 (F1=0.653) since v7 templates add noise, not signal

---

## Key Technical Issues & Fixes

### ollama + Qwen3.5-122B

| Problem | Fix |
|---------|-----|
| Empty response from `/api/generate` | Use `/api/chat` endpoint with message format |
| Model evicted between requests | Add `"keep_alive": -1` to every request |
| Two concurrent scripts → 60s/call slowdown | Run scripts **sequentially**, not in parallel |
| Confidence prompt returns empty string | Use binary YES/NO re-check instead of 0-100 score |
| First batch of requests timeout | Model was unloaded; warm it first with a test call |
| 122B needs 82GB+ VRAM | Two GPUs: 80GB A100 + 20GB A100 = 100GB total |
| torch 2.5 incompatible with transformers 5.x | Upgraded to torch 2.7.1+cu128 (CUDA 12.8 driver) |
| train_cv_regularized.py injected eval_100 5× into training | Removed GOLD_WEIGHT oversampling — dataset.csv already contains any needed gold data |

### GPU Setup
- GPU0 (20GB A100): 16GB used for model layers
- GPU1 (80GB A100): 69GB used for model layers
- Total model weight: ~83GB in Q4_K_M quantization
- ollama automatically splits across both GPUs

### Dataset Quality Lessons

| Mistake | Lesson |
|---------|--------|
| score>0 filter as LLM proxy (v12/v14) | Removes valid implicit interactions → F1 regression |
| SIBiLS pathogen-only harvest (v10) | 92% same category → model bias |
| v7 templates as "gold standard" | 63% of pathogenOf templates rejected by stricter Qwen |
| Concurrent Qwen calls | Serialized queuing causes 20× slowdown on 122B model |

---

## Scripts Created (v15 Build)

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/teacher_label_full.py` | Label all 44K sentences with Qwen3.5-122B | Done |
| `scripts/validate_eval_sets.py` | Qwen-validate all 7 eval sets | Done |
| `scripts/build_negative_pool.py` | Build clean + recheck strong-signal negatives | Done |
| `scripts/fetch_pathogen_sentences.py` | Targeted EPMC harvest for pathogenOf | Done |
| `scripts/assemble_v15_dataset.py` | Final assembly: single dataset.csv + 200 test candidates | Done — reruns after v7 |
| `scripts/validate_v7_with_qwen.py` | Qwen-validate v7 non-pathogenOf templates (7,076) | Running overnight |
| `scripts/pipeline_v15.sh` | Autonomous pipeline: v7 → assemble → train → email | Running (PID 2797) |
| `tools/curate_web.py` | Gradio curation UI on port 7860 | Fixed: dynamic sources + source_file badge |

---

## Curation Queue Status

| Source | Pending | Approved | Uncertain | Notes |
|--------|---------|----------|-----------|-------|
| `eval_qwen_disagreements` | 195 | 0 | 0 | Gold/Qwen disagreements on test set |
| `globi_pmc_pathogenOf_borderline` | 80 | 8 | 12 | User reviewed 20 sentences |
| `sibils_mongodb` | 0 | 380 | 42 | Old backlog, partially done |
| `globi_sibils` | 1,250 | 0 | 0 | Old backlog, skip for v15 |
| `sibils_diverse` | 5,162 | 0 | 0 | Old backlog, skip for v15 |

**Priority:** `eval_qwen_disagreements` → especially the 16 gold=NEG/Qwen=POS cases.

### Launching Curation UI
Run in VS Code terminal (so VS Code auto-detects port and offers to forward):
```bash
source /home/egaillac/MetaP/MPvenv/bin/activate && python classifier/tools/curate_web.py --port 7860
```
Then go to **🔬 Curate — Pending** tab → Source filter → select source → Load batch.

---

## Evaluation Sets

| File | Rows | Pos | Notes |
|------|------|-----|-------|
| `eval_100.tsv` | 100 | 31 | = biotx-random_100original; primary test set |
| `globi-relax_..._EP.tsv` | 100 | 48 | **Primary EP test set** |
| `globi-passage_..._EP.tsv` | 100 | 85 | Secondary |
| `biotx-random_..._50best-multiples.tsv` | 50 | 20 | Hard cases |
| `biotx-random_..._50nomultiple.tsv` | 50 | 12 | Hard cases |
| `gen_set_100.csv` | 100 | 60 | With category + difficulty labels |
| (all combined in `eval_sets_qwen_validated.csv`) | 599 | 286 | Fixed test set for v15 |

---

## V16–V18 Results (2026-04-13)

### Synthetic Gold Test Set (100 sentences, 50 pos / 50 neg)
Claude-generated sentences with 100% certain labels. Used to verify models understand clear interactions.

| Model | Synth F1 | Prec | Rec | Notes |
|-------|----------|------|-----|-------|
| BiomedBERT v11 regularized | **0.931** | 0.922 | 0.940 | Best overall — balanced P/R |
| BiomedBERT v12 regularized | 0.918 | 0.938 | 0.900 | Best precision |
| FLAN-T5-base v11.1 | **0.904** | 0.870 | 0.940 | Best T5 — balanced P/R |
| BiomedBERT v11.1 | 0.902 | 0.885 | 0.920 | |
| FLAN-T5-base v16 real-only | 0.891 | 0.817 | 0.980 | |
| FLAN-T5-base v17a | 0.885 | 0.794 | 1.000 | Recall collapse |
| BiomedBERT v15 | 0.885 | 0.794 | 1.000 | Recall collapse |
| FLAN-T5-base v17b | 0.877 | 0.781 | 1.000 | Recall collapse |
| BiomedBERT v17b | 0.877 | 0.781 | 1.000 | Recall collapse |
| BiomedBERT v16 real-only | 0.862 | 0.758 | 1.000 | Recall collapse |
| FLAN-T5-base v15b | 0.860 | 0.766 | 0.980 | |
| BiomedBERT v17a | 0.855 | 0.746 | 1.000 | Recall collapse |
| FLAN-T5-base v7 | 0.845 | 0.742 | 0.980 | |
| FLAN-T5-base v14 | 0.842 | 0.889 | 0.800 | Best T5 precision |
| BiomedBERT v15b | 0.840 | 0.725 | 1.000 | Recall collapse |

**Key diagnostic:** v15–v17 models all have Rec=1.000 on synthetic (classify everything as positive) but collapse on EP-relax. Models trained on real PMC sentences learn to say YES to everything — they cannot discriminate borderline cases. The v11/v12 models (trained on v7 templates) maintain balanced P/R because templates taught them to say NO to non-interaction patterns.

### EP-relax Results — v16/v17 Progression

| Model | Avg EP F1 | Best Fold | Prec | Rec | eval_100 F1 |
|-------|-----------|-----------|------|-----|-------------|
| BiomedBERT v17a (bal.+4k hard_neg) | 0.688 | 0.748 | — | — | — |
| BiomedBERT v17b (Qwen recal.) | 0.663 | 0.692 | 0.668 | 0.662 | — |
| FLAN-T5-base v17b | 0.610 | 0.695 | 0.702 | 0.688 | 0.732 |
| FLAN-T5-base v17a | 0.583 | 0.644 | 0.690 | 0.604 | 0.716 |
| BiomedBERT v18 hybrid | **pending** | | | | |

### Root Cause Confirmed (2026-04-12)

The v18 dataset build bug revealed everything: `v7_non_pathogen_qwen_validated.csv` has `qwen_label=1` (int), not `"YES"` (str). The first v18 build silently got 0 gap-fill templates → trained on same data as v17. After fix: 5,009 gap-fill templates added (endoparasiteOf: 3,731, preysOn: 682, hasHost: 596). Dataset grows from 16,148 → 27,342 rows, positive rate 18.8% → 29.4%.

### v18 Dataset Composition
- Real sentences (Qwen teacher): 2,944 pos (pollinates, eats, parasiteOf, kleptoparasiteOf≤300, pathogenOf, symbioticWith)
- v7 gap-fill templates: 5,009 pos (endoparasiteOf, preysOn, hasHost — types absent from EPMC corpus)
- Other curated: 93 pos (pathogen_harvested, curated_borderline, eval_100 gold)
- Easy negatives: 9,111 | Hard negatives: 4,000
- **Total: 27,342 rows, 8,046 pos (29.4%)**

---

## Knowledge Distillation (2026-04-15 to 2026-04-18)

### Concept
Compress the best ensemble (BiomedBERT cv_reg × FLAN-T5-base v12, geo, EP F1=0.857) into a single student model. Student has same architecture as BiomedBERT-base (109M params). 3.3× smaller than ensemble, one forward pass at inference.

**Loss:** `α × T² × KL(student_soft ∥ teacher_soft) + (1−α) × CE(student, hard_labels)`

Soft labels generated once from the 44K EPMC corpus. Each sentence gets p_ensemble = sqrt(p_bert × p_t5). These "soft targets" carry the teacher's uncertainty — more information than hard 0/1 labels.

**Soft labels file:** `data/training/distillation_soft_labels.csv` (44,178 rows)

### BiomedBERT Student Variants

| Variant | T | α | EP F1 | eval_100 F1 | Synth F1 | Val F1 | Notes |
|---------|---|---|-------|------------|---------|--------|-------|
| v1 | 4 | 0.7 | 0.785 | 0.659 | 0.948 | 0.748 | Hinton defaults |
| **v2** | **2** | **0.5** | **0.808** | **0.650** | **0.959** | **0.715** | **Best EP** ⭐ |
| v3 | 4 | 0.9 | 0.808 | 0.706 | 0.916 | 0.515 | Best eval_100; low val_F1 but good test |
| v6 | 1.5 | 0.5 | 0.779 | 0.640 | 0.959 | 0.724 | Too sharp — hurts |

**Key finding:** T=2 is the sweet spot. T=4 with high α (v3) gives lower val_F1 (model focuses on soft-label alignment rather than binary boundary) but surprisingly good test F1. T=1.5 (v6) near-hard-label territory — loses the calibration benefit of soft targets.

### Alternative Student Architectures (T=2, α=0.5)

| Student | Params | EP F1 | eval_100 F1 | Synth F1 | Notes |
|---------|--------|-------|------------|---------|-------|
| BiomedBERT-base | 109M | **0.808** | 0.650 | 0.959 | Best — domain-matched pretraining |
| SciBERT | 109M | 0.793 | 0.592 | **0.980** | Best synth gold; science vocab helps generalization |
| DistilBERT | 66M | 0.792 | 0.593 | 0.942 | 40% smaller, only −2 EP F1 — good compression ratio |

**Architecture conclusion:** BiomedBERT wins on EP-relax (domain-matched). SciBERT interesting for general science text (best synth gold). DistilBERT viable if deployment size matters.

### Post-Distillation Fine-Tuning on v18 Data — FAILED

Fine-tuned distilled_v2 for 3 more epochs on v18_hybrid (27K rows, LR=5e-6):
- val_F1 during fine-tuning: 0.977 (fits v18 distribution perfectly)
- EP-relax after: **0.617** (was 0.808 — catastrophic forgetting)
- eval_100 after: 0.577 (was 0.650)

**Conclusion:** v18 template data causes catastrophic forgetting of EP-relax patterns. Templates teach the model to classify on lexical cues ("preyed upon", "endoparasite of") rather than the contextual understanding gained from distillation. The 37-point EP regression confirms that distillation knowledge is overwritten by fine-tuning on templates.

### Ensemble Combinations (summary)

| Combination | EP F1 | eval_100 F1 | Synth F1 | Models needed |
|-------------|-------|------------|---------|--------------|
| orig BERT × T5 geo | 0.857 | (0.942*) | 0.950 | 2 models |
| v2 × v3 × orig BERT × T5 geo | 0.832 | (0.979*) | 0.970 | 4 models |
| v2 × T5 geo | 0.812 | (0.979*) | 0.970 | 2 models |
| v3 × T5 geo | 0.816 | (0.979*) | 0.933 | 2 models |
| v2 × v3 arith | 0.810 | 0.677 | 0.970 | 2 models |
| distilled_v2 alone | 0.808 | 0.650 | 0.959 | 1 model ⭐ |
| distilled_v3 alone | 0.808 | 0.706 | 0.916 | 1 model |

*eval_100 results for combos including orig BERT or T5 are inflated — eval_100 positives were in training data.

**Practical conclusion:** distilled_v2 at EP=0.808 is only 4.9 points below the full ensemble (0.857), for 3.3× less inference cost. For any combo involving the original teacher models, eval_100 F1 is inflated due to training contamination.

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/distill_ensemble.py` | Full pipeline: soft label generation + student training + eval. Args: `--skip-labels`, `--epochs`, `--temperature`, `--alpha`, `--lr`, `--output-dir`, `--results-dir`, `--student-model` |
| `scripts/finetune_distilled.py` | Fine-tune a distilled checkpoint on new data |
| `scripts/eval_distilled_ensembles.py` | Evaluate all distilled model combination strategies |
| `scripts/pipeline_30h_autonomous.sh` | Ran all distillation experiments autonomously |

---

## Next Steps

### Currently running (2026-04-13)
- `pipeline_v17.sh` (PID 127628): BiomedBERT v18 training → FLAN-T5-base v18 training
- Email notifications on each step completion

### After distillation (2026-04-20 status)
- **Best single model:** `models/distilled_BiomedBERT_v2` — EP F1=0.808, Synth F1=0.959. Deploy for inference.
- **Best ensemble:** original `transformer_BiomedBERT_cv_regularized` × `flan-t5-base_v12` (geo) — EP F1=0.857.
- **v18 results:** confirmed gap-fill templates hurt performance (EP F1=0.648 vs v7=0.788). Template+real data mixes don't generalize to EP-relax.
- **Fine-tuning after distillation is destructive** — don't fine-tune distilled models on template data.
- **Ceiling analysis:** The 5-point gap (0.808 → 0.857) between distilled student and full ensemble is the endoparasiteOf coverage gap. EP-relax has 58% endoparasiteOf/hasHost/preysOn positives; none in EPMC training corpus. Closing this gap requires real sentences of these types, not templates.

### Open items
1. **Curate `v15_test_batch1`** (98 sentences pending) → need clean test set not contaminated by training
2. **Harvest real endoparasiteOf/hasHost sentences** from literature (not templates) → retrain distillation soft labels
3. **Relation extraction** — downstream task for knowledge graph; build on top of distilled_v2

---

## Infrastructure

- **Port 8001:** Original ensemble API (`bash classifier/start_api.sh`)
- **Port 8002:** Enriched pipeline with FLAN-T5 as Layer 3 (`bash classifier/start_pipeline_generative.sh`)
- **Port 7860:** Curation web UI (Gradio, manual launch)
- **ollama:** Qwen3.5-122B at `http://localhost:11434`
- **Email notifications:** `bash classifier/scripts/notify.sh "Subject" "Body"`
- **Venv:** `source /home/egaillac/MetaP/MPvenv/bin/activate`
