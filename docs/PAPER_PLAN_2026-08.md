# Resubmission Plan — Biotic Interaction Classifier

**Manuscript audited:** `/home/egaillac/MetaP/classifier/manuscript/biotic_interaction_classifier_ARR.tex` (988 lines). All line numbers below re-derived from that file today (2026-08-25). Numbers marked ✱ were recomputed by me today from the repo, not taken from the upstream analyses.

---

## 1. Diagnosis

The paper claims to present a system and then, in its own abstract, reports that the system does not beat the baseline it is compared against: L99–101 says the champion "is statistically indistinguishable from the **stronger** single-task template-trained baseline (F1 = 0.875, p = 0.760)." Everything offered to fill that hole is either unreproducible or below the resolution of the experiment. The +0.202 F1 decomposition (L87–96, L733–740) rests on four checkpoints that no longer exist — `models/multitask/` contains exactly one directory — and `scripts/eval_corrected_testset.py` prints `SKIP … not found` and continues rather than raising, so the source JSON cannot be regenerated. The abstract's most specific mechanistic claim, "NER pre-training epochs are counterproductive" (L82–84), is five single unseeded runs; `experiments/multitask/train.py` contains no seeding call, two byte-identical commands produced 0.8743 and 0.8346, and a 48-configuration factorial puts the `ner_pretrain` main effect at 0.002 AUPRC. The one comparison the paper omits — champion vs. ensemble, p = 0.091, present in the same stored JSON as the four p-values it does report — is the one that would falsify "surpasses the ensemble teacher" (L97–99). No trivial baseline appears anywhere (`grep`: "majority", "trivial", "all-positive" = 0 hits), while one of the five test sources, EP-passage, is 85% positive so that predicting "yes" unconditionally scores F1 0.919 against the paper's best 0.818. And the released repo — linked at L771 — contains `scripts/build_v14_dataset.py:59-61`, which filters positives through a lexicon and leaves negatives untouched, plus `results/deleted_models_retrain_guide.md` ("Models deleted on 2026-03-02 to free disk space") as the first file in the listing. The paper is not rejectable because a baseline is missing; the baseline is Table 1 row 6 and it wins. It is rejectable because it advertises three contributions of which one is a tie, one is noise, and one (interaction-type mismatch as the ceiling, L169–171, L675–682) is contradicted by the project's own `category_breakdown.json`, where per-category recall is flat and the supposedly starved Pathogen/Infection category is the best category for both models.

---

## 2. Recommended framing

### Title

> **Cleaning Only the Positives: How Class-Asymmetric Quality Filters Install Lexical Shortcuts in Relation-Extraction Corpora**

### Abstract (215 words)

> Relation-extraction corpora are routinely cleaned before training: implausible positives are removed by a rule, a lexicon, or an LLM judge. We show that when cleaning is applied to the positive class only, it installs a lexical shortcut that no standard integrity check detects. In the biotic-interaction pipeline we audit, three separate stages filter positives while leaving negatives untouched, and a test suite enforces the asymmetry, failing any build in which more than 20% of positives lack a lexicon cue or more than 2% of negatives carry one. The result is a corpus in which a 106-pattern keyword matcher separates the classes at F1 0.857 (P(cue∣pos) = 0.949, P(cue∣neg) = 0.129) but reaches only 0.638 on the human-curated benchmark (0.644, 0.449). Models inherit the asymmetry rather than the task: false-negative rates on cue-free positives are 2.3–2.6× those on cue-bearing positives, concentrating roughly a third of the total error budget in one cell of a 2×2 table, and a distilled multi-task model is statistically indistinguishable from a plain fine-tuned baseline (F1 0.874 vs 0.875, p = 0.760). We reproduce the effect causally by injecting the same one-sided filter into ChemProt, give a three-number diagnostic computable without retraining, and release a marginal-matched repair that reduces cue-only separability to F1 0.537.

Every number above is verified: ✱ the cue statistics and 0.5365 I recomputed today (§7 of this plan); 0.874/0.875/p = 0.760 is exact in `results/new_testset/corrected_testset_results.json`; ✱ 106 = `len(STRONG_TERMS)=65 + len(WEAK_TERMS)=41` in `src/data/interaction_lexicon.py`; the 20%/2% thresholds are `assert` statements in `tests/test_training_data.py` GATE 6a/6b. **The ChemProt sentence is contingent on E6 below and must not be written until E6 lands.**

### Contribution bullets

1. **A named construction hazard with a mechanism.** Class-asymmetric cleaning: the cue is not what annotators wrote, it is what the pipeline *kept*. Distinct from annotation artifacts (annotator behaviour) and from adversarial filtering (deliberate, symmetric, disclosed). We document three independent instances inside one project, one of which is described in the rejected paper's own Methods as a quality measure (L197: "Every positive sentence was individually validated by a large language model to remove formulaic or linguistically implausible candidates").
2. **The QA suite is the enforcement mechanism.** GATE 6a fails a build if >20% of positives lack a cue; GATE 6b fails it if >2% of negatives carry a strong un-negated interaction term, on the documented grounds that such negatives are "likely mislabeled." The evaluation benchmark's negatives carry interaction language ~45% of the time. The gate that guarantees label quality is the gate that guarantees the benchmark cannot be matched.
3. **A retraining-free diagnostic and a reporting protocol.** Three numbers — P(cue∣pos), P(cue∣neg), cue-only F1 — computed on the training corpus *and* on the benchmark, plus cue-stratified recall and precision at deployment prevalence on every results table. We show the diagnostic predicts the trained model's per-stratum error profile.
4. **Causal demonstration on a public corpus.** Injecting the identical one-sided filter into ChemProt, with cues induced from data we did not build and a test set we never touch, reproduces the error signature.
5. **A repair, with its own cost measured.** Marginal matching reduces cue-only F1 from 0.857 to 0.537; we report what that costs on aggregate F1 rather than assuming it is free.

### Why this framing and not the alternatives

**Against the system paper (current framing).** Dead. Three headline claims: one tie, one null, one contradicted. Four checkpoints deleted. Not repairable by rewriting.

**Against "Which Pair? — the existential/pair-specific formulation" (pair conditioning as the lead).** Three independent problems. (i) Prior art: entity markers are standard practice since Baldini Soares et al. 2019 and typed markers since Zhou & Chen 2022; "the field's default beats a formulation we chose ourselves" is not a contribution. (ii) The effect is regime-confined: `sweep_v2.py:55` admits the pair flag was varied only in regime P (v14 hard labels); soft-label training with no pair conditioning already reaches test299 AUPRC 0.8351 against pair-conditioned regime P's 0.8379. Pair conditioning has never been shown to improve the best available configuration. (iii) The sweep it comes from has `seed == 1` on all 48 rows, so the quoted "within-cell std" is the spread across twelve different configs, not run variance — the "5–8× the noise" claim has no denominator. Pair conditioning becomes a **conditional Section 6**, gated on E10/E11 below, phrased as a structural instance of the same mismatch, never as the paper's fix.

**Against the combined "Trained on Cues, Tested on Pairs".** Attractive but currently unshippable: it needs "48 configurations with three seeds" (false — n = 1), "22nd of 30 by AUC" (wrong model and wrong denominator: `all_models_eval_499_test_set.md` ranks 33 rows, and row 22 is the *cold-start ablation*, not the warm-start champion), and a test299 loss attributed to a champion checkpoint that no longer exists. Fold its Section 3 (the factorial as a statement about ablation practice) into the recommended paper's Section 5, restated honestly as *unresolvable at the available resolution*, not as *null*.

**The decisive argument for Framing A:** it is the only framing whose entire evidence base sits in files that still exist, needs no deleted checkpoint, is independent of the single-seed sweep, and is *deterministic* — corpus statistics carry no seed variance, so they are the only large quantitative results in the project that sit outside the 0.040 F1 noise floor.

---

## 3. Venue

### Recommended: ARR October 2026 cycle → commit to NAACL 2027

**Verified by me today (2026-08-25):**

| Fact | Source |
|---|---|
| ARR October 2026 cycle: **submission October 12, 2026**; reviewer registration / reviews due / author response / meta-review release all listed **TBA**; **cycle end December 20, 2026** | [aclrollingreview.org/dates](https://aclrollingreview.org/dates) |
| That cycle commits to **NAACL 2027 and COLING 2027**, commitment deadline given as **December 20, 2026** | [aclrollingreview.org/dates](https://aclrollingreview.org/dates) |
| NAACL 2027's own CFP gives **meta-reviews December 18, 2026; commitment December 23, 2026; notification February 10, 2027; conference June 1–5, 2027, San Francisco**; all deadlines 11:59pm UTC−12 | [2027.naacl.org](https://2027.naacl.org/calls/main_conference_papers/) |
| ⚠ **The two official pages disagree on the commitment date (Dec 20 vs Dec 23).** Plan to December 20. | both, as above |
| ARR area to select at submission: **"Resources and Evaluation"** (present in ARR's 30-area list); secondary **"Interpretability and Analysis of Models for NLP"**. NAACL 2027's own topic list as I fetched it names "Interpretability and Model Analysis" and "Information Extraction and Parsing" but did **not** show a resources/evaluation area — the fetch may be partial; verify before committing. | [aclrollingreview.org/cfp](https://aclrollingreview.org/cfp), [2027.naacl.org](https://2027.naacl.org/calls/main_conference_papers/) |
| ACL 2027: final ARR submission listed only as **"January, 2027"**, no commitment date published | [aclrollingreview.org/dates](https://aclrollingreview.org/dates) |

**Compliance items I verified in the ARR CFP today, all of which the current manuscript fails:**

- *"The limitations section is mandatory… titled 'Limitations'… does not count toward the page limit… appearing after the conclusion but before references… Papers lacking this section face desk rejection."* The manuscript has `\subsection{Limitations}` at **L684**, numbered, inside `\section{Discussion}` (L616), **before** `\section{Conclusion}` (L722). Moving it recovers roughly three-quarters of a page for free — which the reframe needs.
- *Responsible NLP Research checklist is mandatory; "incorrect, incomplete, or misleading responses can result in desk rejection."* Checklist item C3 asks whether you report descriptive statistics and whether it is transparent that you are reporting a single run. F1 = 0.874 is one run from a seedless trainer with a measured 0.040 run-to-run floor, and CI [0.843, 0.902] is a *test-set bootstrap*, not training variance. This is a signed declaration, not a reviewer's guess.
- *"ARR will not consider any paper that is under review in a journal or another conference at the time of submission… covering all journals and refereed conferences without exception."* **This kills the "submit to ARR and a journal in parallel" idea.** Sequential only.

**Working backwards from Oct 12:** experiments frozen ~Sept 20, full draft Oct 5, one buffer week. Author registration within 48h of the deadline for all five authors on L60.

### Second choice: COLING 2027

Same ARR submission, different commitment. **The choice is made in December, from the meta-review, not now.** Score ≥ 4 → NAACL; 3–3.5 → COLING. Note COLING 2027 is the first COLING fully on ARR, drawing the same reviewed pool, so the historical looseness (COLING 2025 direct: 31.5% long) is not a safe planning basis; the residual advantage is post-meta-review self-selection, which is inference, not measurement. COLING 2027 also accepts commitments from earlier ARR cycles.

### Fallback

**If Oct 12 slips: TACL.** Rolling monthly deadlines (the 1st), no cliff, ACL Anthology archival, counts as a main ACL venue in ARR's own reviewer-qualification list, and accepted papers may be presented at NAACL/ACL/EMNLP. This removes the "seven weeks forces the factorial" pressure entirely and is a better fallback than waiting for the January 2027 ACL cycle. Same no-concurrent-submission rule.

**If the NLP framing does not land: Database (Oxford).** Scope verified verbatim: *"articles relevant to the annotation process such as standards for curation, annotation best practices, annotation methodologies, the use of automated and semi-automated methods for annotation and the measures for annotation consistency."* That is exactly what class-asymmetric cleaning is. Rolling, open access, no working KG required, and the readership is the GloBI/BioCreative/SIBiLS community. GigaScience is the alternative if the D1 repair produces a clean effect — it is the only candidate journal whose stated criteria are *"reproducibility, usability and utility, rather than subjective assessment of 'impact'."*

**Do not plan around the Insights workshop.** It exists (7th edition, EMNLP 2026 Budapest, Thursday Oct 29) and its CFP names three of this paper's findings almost verbatim, but its submission page 404s, an October event implies a deadline already passed, the ARR-commitment route is unusable without ARR reviews, and it would spend the strongest result on a short-paper venue while foreclosing ARR under the dual-submission rule.

---

## 4. Claim disposition table

Dispositions: **KEEP** (survives verbatim into the reframed paper) · **RESTATE** (survives with the exact replacement wording given) · **WITHDRAW** (does not appear in the reframed paper).

### Abstract (L68–108)

| # | Claim (line) | Disposition | Replacement / reason |
|---|---|---|---|
| A1 | Biodiversity PMC "over 85 million files" (L71–72) | **RESTATE** | The cited source (Pasche et al., BISS 2023) describes MEDLINE 30M+ abstracts and PMC 5M+ full texts. → *"Biodiversity PMC indexes tens of millions of biomedical and biodiversity documents"* — or cite a source that supports 85M. |
| A2 | "a 122-billion-parameter local model (Qwen3.5-122B; qwen2025technical)" (L74–75) | **RESTATE** | `qwen2025technical` is the *Qwen3* Technical Report (arXiv:2505.09388, May 2025); Qwen3.5-122B-A10B is a Feb-2026 MoE release with ~10B active parameters. → *"a locally hosted Qwen3.5-122B-A10B mixture-of-experts model (122B total, ~10B active parameters)"* with the correct citation. |
| A3 | F1 = 0.874 (P 0.925, R 0.829, CI [0.843, 0.902]) (L86–87) | **RESTATE** | Arithmetic exact but the checkpoint is deleted and it is a single unseeded run. → *"On a 500-sentence multi-source test set, a distilled multi-task model reaches F1 0.874 and a plain template-trained BiomedBERT reaches 0.875 (McNemar p = 0.760); the trainer that produced them has no seed and two byte-identical invocations differ by 0.040 F1."* |
| A4 | "+0.202 F1 (p<0.001) vs identical arch. with hard CE" (L87–89) | **WITHDRAW** | Both checkpoints deleted; not regenerable. |
| A5 | "+0.162 F1 from soft-label distillation alone" (L89–90) | **WITHDRAW** | Same. Direction may be re-established by E7; do not report until it is. |
| A6 | "moves AUC by only +0.024" (L91–93) | **WITHDRAW** | Same deleted checkpoints. |
| A7 | "remaining +0.040 F1 (p=0.002)… neither is sufficient alone" (L94–96) | **WITHDRAW** | 0.040 *is* the measured noise floor — the same two numbers, 0.8743 and 0.8346. McNemar tests two prediction vectors, not a training recipe. |
| A8 | "NER pre-training epochs are counterproductive when task-tuned weights are reused" (L82–84) | **WITHDRAW** | Factorial main effect 0.002 AUPRC; five single unseeded runs. |
| A9 | "surpasses the ensemble teacher (F1 = 0.850)" (L97–98) | **RESTATE** | → *"matches the ensemble teacher (0.874 vs 0.850; McNemar p = 0.091, n.s.), and is ranked below it on AUPRC."* Report p = 0.091 alongside the other four. |
| A10 | "at 3.3× lower inference cost" (L99) | **RESTATE** | No latency or FLOP measurement exists. → *"with roughly one third the parameters of the two-model ensemble (analytic parameter count; end-to-end latency not measured)"* — or measure it. |
| A11 | "statistically indistinguishable from the stronger single-task template-trained baseline (F1 = 0.875, p = 0.760)" (L99–101) | **KEEP** | Exact against `corrected_testset_results.json`. The one abstract sentence that survives verbatim, and the new paper's opening observation. |
| A12 | "On real-literature sources alone… leads the baseline (F1 0.847 vs 0.821)" (L102–104) | **WITHDRAW** | The body reports this same comparison as *directional, p = 0.067* (L646–648); the gap (+0.026) is below the 0.040 floor; the 0.847 half has no artifact. A post-hoc subgroup rescue of a null in a paper whose thesis is that post-hoc reasoning about nulls is how this project got here. |
| A13 | "deployed upstream of BiotXplorer" (L105–107) | **RESTATE** | `api/fastapi_multitask.py:33` points at a deleted checkpoint. → *"An earlier ensemble configuration is deployed upstream of BiotXplorer; the model reported here is not in production."* Verify the service state before writing either version. |

### Introduction and contributions (L111–172)

| # | Claim (line) | Disposition | Replacement / reason |
|---|---|---|---|
| I1 | "23 of 171 ROBI relation types (ROBIext v2025)" (L121–123) | **RESTATE** | Count verified (171 concepts in `robiext_v2025.json`), but it is cited to `poelen2014global` — a 2014 paper for a 2025 ontology. Fix the citation. |
| I2 | GloBI "over 13 million records from more than 700 tabular data sources" (L119–120) | **RESTATE** | The cited 2014 paper describes ~700,000 interactions from 19 sources. Cite a current GloBI release, or restate to what the 2014 paper supports. |
| I3 | Contribution 1: "44K-sentence training corpus with **calibrated** labels" (L162–164) | **RESTATE** | The paper itself reports teacher ECE = 0.148 (L308–309). → *"a 44K-sentence LLM- and ensemble-labelled corpus; the ensemble's probabilities are miscalibrated on human-curated data (ECE 0.148)."* Also fix the released file: on disk it is 50,041 rows / 4,243 positives, not 44,178 / 4,065. |
| I4 | Contribution 2: warm-start + omitting NER pre-training "surpasses cold-start (+0.040 F1)" (L165–168) | **WITHDRAW** | See A7. |
| I5 | Contribution 3: "distribution mismatch analysis identifying the interaction types responsible for the performance ceiling" (L169–171) | **RESTATE — this is the pivot of the whole reframe** | `category_breakdown.json` shows flat per-category recall (champion 0.75–0.90) and Pathogen/Infection — 3% of training positives — is the *best* category for both models. → *"A quantitative analysis identifying the mechanism responsible for the ceiling: the training corpus and the benchmark define the positive class on different signals, because positives (and only positives) were filtered through an interaction lexicon."* |

### Methods (L175–490)

| # | Claim (line) | Disposition | Replacement / reason |
|---|---|---|---|
| M1 | Template corpus "25,081 clean sentences (7,251 positive)" (L202) | **KEEP** | Exact against `v7_data.csv`. |
| M1b | "Every **positive** sentence was individually validated by an LLM to remove formulaic or linguistically implausible candidates" (L197) | **RESTATE — promote to a headline** | This is instance #1 of the hazard, stated in the paper as a virtue. → keep the sentence, add: *"Negatives were not validated. This asymmetry is the subject of Section 4."* |
| M2 | "Of 44,178 harvested sentences, 4,065 (9.2%) were accepted as positives" / "The accepted 44,178 sentences were re-scored" (L214, L216–217) | **RESTATE** | Internally contradictory two sentences apart, and the released file has 50,041 rows / 4,243 positives. Ship the corpus version actually used and restate to match it exactly. |
| M3 | "SVM and Random Forest on TF-IDF achieved F1 ≈ 0.62" (L229–231) | **RESTATE or WITHDRAW** | No result file located; not in Table 1. Either produce the artifact or drop. In the new paper, the cue-only regex (F1 0.857 train / 0.638 benchmark) replaces it as the trivial-model row. |
| M4 | Template-trained BiomedBERT "achieves F1 = 0.875 on the 500-sentence test set" (L247) | **KEEP** | Checkpoint survives and reproduces. |
| M5 | Ensemble "achieves F1 = 0.850" (L272) | **KEEP** | Both components survive. |
| M6 | "T = 2, α = 0.5 identified by grid search over six student configurations" (L304–305) | **KEEP** | As stated; App A already flags 4 of 6 rows stale with ‡. |
| M7 | "ECE = 0.148 on EP-A; mean confidence 0.967 vs accuracy 0.804" (L308–310) | **RESTATE** | No supporting value found in `results/`. Recompute and re-report, or drop. |
| M8 | Best student "achieves F1 = 0.858" (L310–311) | **KEEP** | `distilled_v2` = 0.8582; checkpoint survives. |
| M9 | "≈4.2M binomial taxon names plus 21K common names" (L329–330) | **KEEP** | Matches `data.py:19-20`. |
| M10 | Interaction lexicon "591 terms… supplemented by 50 biomedical terms" (L338–342) | **RESTATE — critical** | 591 is the SIBiLS/ROBI **NER-supervision** lexicon. The lexicon that `has_signal` calls is `src/data/interaction_lexicon.py`, ✱**106 regex patterns (65 STRONG + 41 WEAK)**. Printing "641-term" anywhere is a checkable error. → distinguish the two objects explicitly and state which one the diagnostic uses. |
| M11 | Table 2 NER schemes: basic 0.861 / typed 0.794 / full 0.858 / full_typed 0.840 (L366–369) | **WITHDRAW** | Factorial `ner_scheme` main effect 0.004. Three values reproduce; `full_typed = 0.840` has no provenance (`ablation.json` gives 0.7919). And the caption (L357–359) openly selects the second-worst scheme "for its HOST/PATHOGEN typing" — a capability that emits zero spans in production. |
| M12 | Table 1, all six rows (L392–403) | **RESTATE** | Every value matches `corrected_testset_results.json` to 3 dp. Rows 1–3 are unreproducible; strike them, keep rows 4–6, and add: all-positive baseline row, cue-only baseline row, AUPRC column, precision-at-deployment-prevalence column. |
| M13 | Cold-start "achieves F1 = 0.835" (L430–431) | **WITHDRAW** | Checkpoint deleted. |
| M14 | Warm-start "achieves F1 = 0.874 at τ = 0.360" (L437) | **WITHDRAW** | Checkpoint deleted. (The τ float signature does confirm the threshold came from a validation sweep, not the test set — that mechanism claim survives.) |
| M15 | "0 NER epochs 0.874, 1 → 0.860, 2 → 0.822… clear negative correlation" (L438–441) | **WITHDRAW** | Spread 0.052 against a 0.040 floor, n = 1 per cell. |
| M16 | Test set composition: 500 rows, 281 positive (56.2%), five sources 99/100/104/100/97 (L452–458) | **KEEP** | Reproduces exactly. |
| M17 | **Omission:** EP-A is not disclosed to be the EP-relax benchmark on which the champion was selected | **RESTATE (add)** | 99/99 exact normalised match, independently confirmed twice. → *"EP-A is the EP-relax benchmark used throughout this project for model selection; approximately 30 multi-task configurations were trained and one selected by EP-relax F1. Results are reported with and without this source."* This also voids L698–699's "our clean anchor". |
| M18 | **Omission:** 197 of 500 rows (39.4%) are LLM-generated | **RESTATE (add)** | The paper says only "synthetic" (L455) and "authored fresh for this evaluation" (L932). State the generator, the prompt, and release the generation script. |
| M19 | "τ is optimised on the validation split, not the test set" (L460–461) | **RESTATE** | True of τ, misdirecting overall. → keep, then add the configuration-selection disclosure from M17. Also disclose that `evaluate.py:104-110` sweeps 90 thresholds on the same 100 EP-relax rows it reports, and that `tier2_triple_extractor.py` shipped with a hard-coded 0.13 equal to the EP-relax F1-argmax (now `None`, resolved from the checkpoint — cite the commit SHA). |
| M20 | "standard 15% held-out split" (L462, L878, L892) | **RESTATE** | `train.py:162` passes `val_frac=0.1`. → **10%**. |
| M21 | "Bootstrap CIs with 10,000 resamples" (L467–468) | **KEEP** | Exact. |
| M22 | "McNemar's test with continuity correction" (L469–470) | **KEEP** | Matches `eval_corrected_testset.py:117`. |
| M23 | §5.3: "These checks confirm the test set is independent of the training data used to produce every reported result, with one exception" (L476–481) | **WITHDRAW** | Directly contradicted by `results/contamination_check.json`. Rewrite §5.3 from scratch. |
| M24 | "eval-100 and BioTx-random share 96 of 100 sentences" (L481–484) | **KEEP** | Arithmetic consistent (100 + 100 − 96 = 104). |

### Results (L493–613)

| # | Claim (line) | Disposition | Replacement / reason |
|---|---|---|---|
| R1 | "AUC 0.912–0.921 tight cluster; hard-CE 0.857" (L503–505) | **WITHDRAW** | Sourced from deleted checkpoints. |
| R2 | "hard-CE AUC gap 0.055–0.064 vs F1 gap 0.177–0.202" (L506–511) | **WITHDRAW** | Same. |
| R3 | Finding 1: soft labels are the largest lever, +0.162 F1 (L515–522) | **RESTATE, pending E7** | Direction plausibly survives the noise floor but both checkpoints are gone. → do not report until re-run with ≥3 seeds. If it survives: *"Soft-label distillation is the only training-recipe factor whose effect exceeds the measured run-to-run variance (Δ = x ± y over 3 seeds)."* |
| R4 | Finding 2: NER pre-training epochs matter only under warm-start (L527–533) | **WITHDRAW** | See A8. |
| R5 | Finding 3: NER scheme richness does not predict cold-start performance (L535–542) | **WITHDRAW** | Main effect 0.004. Also strike L536–539's justification for retaining `full_typed` "because its HOST/PATHOGEN typing is required for the downstream knowledge-graph use case." |
| R6 | Finding 4: "+0.040 over cold-start; gap with template-trained is 0.0003, not significant" (L544–550) | **Split: first half WITHDRAW, second half KEEP** | 0.0003 / p = 0.7604 is exact and becomes the new paper's opening observation. |
| R7 | The three reported McNemar counts, 113/35, 32/11, 23/20 (L574–582) | **KEEP** | All exact. |
| R8 | "cold-start significantly weaker than template-trained (p = 0.045)" (L584–586) | **RESTATE or WITHDRAW** | This pairwise test is not in the stored JSON. Supply provenance or delete — including from the Bonferroni count at L653. |
| R9 | CIs [0.843, 0.902] and [0.844, 0.903] overlap (L588–591) | **KEEP** | Exact. |
| R10 | **Omission:** champion vs ensemble p = 0.091 | **RESTATE (add)** | Present in the same JSON as the four reported p-values. Report it. This is the single most damaging omission in the paper. |
| R11 | **Omission:** no trivial baseline anywhere | **RESTATE (add)** | ✱ all-positive F1: test500 0.720, test299 0.706, EP-passage 0.919. Add as a row on every table and state that no model beats the trivial baseline on EP-passage. |
| R12 | **Omission:** no seeds, no variance, no configuration count | **RESTATE (add)** | State the noise floor (0.040 F1), the absence of seeding in `train.py`, and that ~30 configurations were trained and 11 reported. |
| R13 | **Omission:** no AUPRC | **RESTATE (add)** | The ranking inverts on AUPRC. Report it as a primary metric. |
| R14 | Training mix: predation 42%, herbivory 22%, pollination 22%, parasitism 10%, pathogen 3% (L598–602) | **RESTATE** | No stored breakdown located. Recompute and cite the artifact. |
| R15 | "68% of test positives (Parasitism/Host 41.6%, Pathogen/Infection 26.3%)" (L602–604) | **RESTATE** | `category_breakdown.json` records 111 and 81 → 39.5% / 28.8%. Fix or explain the discrepancy. |

### Discussion (L616–719)

| # | Claim (line) | Disposition | Replacement / reason |
|---|---|---|---|
| D1 | Per-source figures: EP-A 0.840 vs 0.781; EP-passage 0.818 vs 0.805; gen-set-100 0.909 vs 0.838; "within 0.001" on the other two (L637–644) | **WITHDRAW** | `export_figures.py:106-107` hardcodes both rows; the champion row is bit-identical (not "within 0.001") to the baseline on two independent ~100-row sources and cannot be recomputed. Regenerate Figure 1 from stored per-sentence probabilities or remove it. |
| D2 | "champion leads on the 303 real-literature sentences (directional, p = 0.067)" (L646–648) | **WITHDRAW** | See A12. |
| D3 | "baseline leads on the 197 synthetic sentences (0.946 vs 0.913)" (L648–649) | **WITHDRAW** | No artifact. |
| D4 | Bonferroni caveat, "five comparisons reported here" (L652–655) | **RESTATE** | Six with p = 0.091. But do not lean on multiplicity: p = 0.002 survives Bonferroni either way and dies on the noise floor instead. Conflating the two hands a reviewer an easy rebuttal. |
| D5 | §6.3 "The NER auxiliary task improves classification"; "the NER head produces entity labels at inference time at no extra cost"; "The HOST/PATHOGEN distinction it outputs directly encodes knowledge-graph edge direction" (L662–674) | **WITHDRAW** | No α = 1.0 classification-only control appears anywhere, and App B *already contains* one: "distilled student (no NER)" F1 = 0.858 (L861) beats every cold-start multi-task row in the same table (0.840, 0.835). On the paper's own numbers the NER head *costs* ~0.02 F1. The head emits zero HOST and zero PATHOGEN in production. |
| D6 | §6.4 "further improvement more likely from expanding parasitism/pathogen coverage than from architectural changes" (L675–682) | **WITHDRAW** | Flat per-category recall; the project's own record that PMC harvest augmentation hurt in every configuration. This recommendation would send the project to do the one thing already measured not to work. |
| D7 | Limitation 1: "only 5.1% of distillation sentences appear elsewhere… mean p_ensemble = 0.20" (L688–694) | **RESTATE** | No artifact. Recompute or drop. |
| D8 | Limitation 1: "EP-A is human-curated and independent of both teachers, our clean anchor against this risk" (L698–699) | **WITHDRAW** | EP-A is the model-selection set. This sentence inverts the actual risk. |
| D9 | Limitation 2: "95% CIs span ±0.030 F1" (L704) | **KEEP** | Correct. |
| D10 | Limitation 4: "NER head not directly evaluated" (L710–715) | **RESTATE** | → *"The NER head is not merely unevaluated: two of its four entity types, HOST and PATHOGEN, are never predicted at inference, and downstream triple extraction over 34,880 sentences produced an empty output file."* |

### Conclusion (L722–758) and appendices

| # | Claim (line) | Disposition |
|---|---|---|
| C1 | "F1 = 0.874 (P 0.925, R 0.829)" (L725–726) | **RESTATE** — inherits A3 |
| C2 | "+0.162 F1… AUC change of +0.024" (L733–737) | **WITHDRAW** — inherits A5/A6 |
| C3 | "warm-start + skipping NER pre-training adds +0.040 F1… total +0.202" (L737–740) | **WITHDRAW** — inherits A7 |
| C4 | "statistically indistinguishable from the template-trained baseline (0.875, p = 0.760)" (L740–742) | **KEEP** |
| C5 | "while surpassing the ensemble teacher at 3.3× lower inference cost" (L742–744) | **RESTATE** — inherits A9/A10 |
| C6 | "constitutes the first component of a knowledge graph construction pipeline" (L747–750) | **RESTATE** → *"is intended as the first component of a knowledge graph pipeline; no knowledge graph has yet been produced from it."* |
| C7 | Future work item (3), amplicon reranking (L752–758) | **WITHDRAW** — the reranker is model-free and a no-op on 5 of 7 datasets |
| P1 | App A distillation grid, v1 0.851 / v2 0.858 / four ‡-flagged rows (L802–808) | **KEEP** — including the ‡ disclosure, which is honest and should be imitated elsewhere |
| P2 | App B, 14 rows (L841–863) | **RESTATE** — strike the three warm-start rows (deleted checkpoints) and `full_typed_a05 = 0.840` (no provenance); keep the five verifiable rows; **promote the "distilled student (no NER) = 0.858" row into the main text**, since it is the missing multi-task control |
| P3 | App C threshold figure: "F1 nearly flat across τ ≈ 0.09–0.92" (L896–900) | **WITHDRAW** — the 300/700 calibration split does not exist as a file; the only trace is a hardcoded array in `export_figures.py`, and the figure is explicitly from a different model on a different split |
| P4 | App D exact-match 0/0 (L918–921) | **RESTATE** — under normalised matching there are 2, not 0/1; state the matching rule and the count |
| P5 | App D "exactly one test sentence" in the 34,880 corpus (L922–927) | **RESTATE** — same |
| P6 | App D "no additional near-duplicates are found beyond the single exact match" (L935–937) | **WITHDRAW** — `contamination_check.json` records 4 matched records (3 distinct sentences, similarities 0.855–0.931) against `distillation_44k`, the champion's own training corpus |
| P7 | App D "three checks between the test set and **every** training corpus used to produce a reported model" (L916); taxon-pair overlap 1%/1%/0% (L939–944) | **RESTATE** — the pair check has fields only for v7 and v14, never for the champion's 44K corpus. Either run it (E13) or narrow the scope sentence to the corpora actually checked |
| P8 | App E.1 "the HOST/PATHOGEN distinction… directly encodes the direction of the knowledge graph edge… enabling the downstream relation extractor to assign typed edges" (L971–986) | **WITHDRAW** — every sentence contradicted by measurement |

---

## 5. Experiment list, ordered

GPU costs assume ~11 min/run on the 80GB A100 (upper bound; observed sweep cadence is faster).

| # | Experiment | GPU | What it buys | Status |
|---|---|---|---|---|
| **E0** | **Pin the cue definition.** ✱ Already resolved today: `build_v14_dataset.py:35-37` defines `has_signal` as `strength > 0.0`, which is *looser* than the lexicon's own `has_signal` flag (`raw_strength ≥ 0.15 or ≥2 strong matches`). That is the entire source of the 0.857-vs-0.897 discrepancy between the two prior analyses. **Both were right, for different cue definitions.** Report the builder's definition as primary and the other as a robustness row. | 0 | Removes the single "not independently reproducible" flag from the paper's headline number. | **DONE** (see §7) |
| **E1** | **Instrument the evaluator.** Add to `src/eval/core.py`: all-positive baseline, cue-only baseline, cue-stratified recall/FNR, AUPRC, precision at deployment prevalence. Emit on every run. | 0 | Every table in the paper becomes reviewer-proof, and every future run carries the reporting protocol the paper argues for. Retroactively rescoreable from stored probabilities. | ~0.5 day |
| **E2** | **Learned-cue replication.** Re-run the signature table under three cue sets that had no role in corpus construction: (a) train-only log-odds top-k, k ∈ {25, 50, 100}; (b) ROBI relation labels alone; (c) filter built from a random half of the lexicon, diagnostic run on the other half. | 0 | **Kills the circularity objection.** Non-negotiable. One arm already survives: `interaction_taxonomy.scan_globi_terms`, a different code path, gives benchmark-negatives 44.2% vs training-negatives 4.5% vs template-negatives 0.0%. | ~1 day |
| **E3** | **Literature audit.** 25–30 recent distant-supervision or LLM-cleaned biomedical/biodiversity RE corpus papers plus their released build scripts; tabulate how many apply a cleaning step to one class only. | 0 | **Decides whether this is a main-track paper or a case study.** Non-trivial rate → general claim. Near-zero rate → demote to Database (Oxford). Do this before writing Section 1. | ~2 days |
| **E4** | **Public-corpus signature table.** ChemProt, BC5CDR, BioRED, train-induced cues, train and test splits separately. All three verified downloadable without credentials (`bigbio/chemprot` ships parquet; `bigbio/bc5cdr` ships `CDR_Data.zip`; `BIORED.zip` HTTP 200). Skip DDI-2013 — `bigbio/ddi_corpus` ships only loader scripts. | 0 | Converts a self-audit into a field instrument, and tests the directional prediction that candidate-pair construction is a *structural defence* against cue asymmetry. Publishable on its own. | ~2–3 days |
| **E5** | **THE GATE — injection into ChemProt.** Induce cues from ChemProt train, filter positives only, subsample to hold size and class balance constant, train BiomedBERT × 3 seeds, evaluate on the **untouched** ChemProt test set. Report induced train gap, test F1 before/after, cue-stratified FNR before/after. | ~2.5 h | The causal demonstration. Cue set induced from data you did not build, test set never touched — the circularity objection cannot reach it. **Without this the paper is a case study; with it, it is a construction hazard.** Run before writing Section 1. | not started |
| **E6** | **D0 vs D1 × 3 seeds on the V2 trainer,** reported with cue-stratified recall, not just aggregate F1. | ~2.5 h | The repair's downstream effect. Three outcomes, all publishable: D1 helps (thesis confirmed) / D1 flat but FNR asymmetry drops (robustness at no cost — pre-register this as the primary outcome) / D1 hurts (the delta *is* the shortcut's contribution to the headline F1 — say so). | **IN FLIGHT** — `results/v2_stage3/` writing now, D0s and D2 arms present, seeds s1/s2/s3, both GPUs at 96–99% |
| **E7** | **Neutral-target D1 variant:** rebuild D1 to P(cue∣pos) = P(cue∣neg) — a target that uses no benchmark statistic — and train 3 seeds. | ~0.6 h | **Closes the sharpest self-inflicted wound.** `v2_d1_v14_signalmatched.manifest.json` and the D3 manifest both record `target_marginals.source: "test299"` — the repair was built from two aggregate statistics of the evaluation set, in a paper about test-set-fitted decisions. The manifests will be released. | not started |
| **E8** | **Seeded noise floor for the V2 trainer:** 5 seeds × 3 configs. | ~2.8 h | An actual variance denominator, and an honest answer to Responsible-NLP checklist item C3. Also permits an honest re-test of the one recipe claim worth rescuing (soft labels, R3). | not started |
| **E9** | **Contamination re-run + disclosure.** Near-duplicate and taxon-pair checks against the champion's 44K corpus; publish the 3 distinct near-duplicates (0.855–0.931) and the exact-match counts under a stated normalisation rule. | 0 | Closes the App-D contradiction, which is the paper's only credibility (rather than correctness) problem. | ~0.5 day |
| **E10** | **Pair-claim prerequisites (blocking, if §6 is to exist):** recover taxon pairs for the 201 test500 rows that lack them — `score_model.py:135` currently prints *"no pair columns; scoring without pair conditioning"* and feeds a pair-trained model a bare sentence, which is the exact train/inference mismatch that produced the withdrawn v16 negative — plus query-shuffle and no-query controls. | 0 + ~0.6 h | Without this, the apparent pair regression on test500 (AUPRC 0.798 → 0.712) is an artifact you are publishing, and the format-vs-content objection is unanswered. | not started |
| **E11** | **Regime-S pair arm:** soft labels × pair on/off × 3 seeds. | ~1.2 h | **Gates whether Section 6 exists at all.** Currently soft-label training without pair conditioning (test299 AUPRC 0.8351) already matches pair-conditioned hard-label training (0.8379). If pair conditioning does not improve the best configuration, it is a scoped observation and future work, not a fix. | not started |
| **E12** | **LLM baseline + LLM error asymmetry:** Qwen3.5-122B zero-shot on test299, scored with the same cue stratification. Local Ollama only — no API key, per project rules. | 0 | Supplies the baseline whose absence is a standard 2026 ARR objection (`grep`: "zero-shot", "GPT", "in-context", "LLM baseline" = 0 hits), **and** tests whether the LLM shows the same asymmetry. If it does, the paper stops being about your model and becomes about how the subfield measures this task. | not started |

**Total new GPU: ~10 hours.** Time is not the constraint; the zero-GPU work (E2, E3, E4, E10) and the writing are.

**Freeze date: ~Sept 20** for Oct 12 submission. If E5 or E3 will not land by then, go to TACL rather than burning the cycle on a half-analysed submission.

---

## 6. What to cut, what to move to Limitations

### Cut entirely

1. **The +0.202/+0.162/+0.024 decomposition** — abstract L87–96, Finding 1 L515, Table 1 rows 1–3, Conclusion L727–740. Four checkpoints deleted; `eval_corrected_testset.py` SKIPs rather than raises.
2. **The NER pre-training claim and its "representation disruption" mechanism** — L82–84, L429–448, L527–533, App B warm-start rows.
3. **The NER-scheme arm entirely** — §Multi-task L351–369, Finding 3 L535–542, including L536–539's justification for retaining `full_typed` for a KG use case that does not exist.
4. **§6.3 "NER as Zero-Cost Auxiliary Supervision"** (L662–674) — refuted by the paper's own App B row at L861.
5. **App E.1 "Utility for BiotXplorer and Knowledge Graph Construction"** (L971–986) — every sentence contradicted.
6. **App C threshold figure** (L868–908) — the underlying split does not exist as a file.
7. **Figure 1 and the per-source narrative in §6.2** (L637–649) unless regenerated from stored per-sentence probabilities.
8. **Amplicon reranking from future work** (L752–758) — model-free, no-op on 5 of 7 datasets.
9. **The abstract's real-literature subgroup lead** (L102–104) — a p = 0.067 post-hoc split promoted to the abstract, in a paper whose new thesis is that this is exactly how the project went wrong.
10. **"This work raises no ethical concerns"** (L761) — wrong sentence for a paper whose core is a self-audit finding a dataset shortcut, a reporting-set-fitted production threshold, and deleted ablation checkpoints. Retitle to "Ethical considerations" per ARR guidance and write it properly.

### Move to Limitations (which must first move to an unnumbered `\section*{Limitations}` after the Conclusion — it is currently a numbered `\subsection` at L684 inside Discussion, which is both a desk-reject risk and a waste of free page space)

1. **The diagnostic's cue set is the project's own lexicon.** Mitigated by E2, not eliminated. State the residual dependence.
2. **The primary benchmark is 299 rows, a third of which (EP-A, 99 rows) is the project's model-selection set**, another third (EP-passage) is below the trivial baseline for every model, and one third (eval-100) has 31 sentences exact-matched into non-reported training corpora. Disclose all three yourself, in Section 2, before a reviewer finds them in an appendix.
3. **Test-set-informed corpus design.** D1 and D3 were built to marginals read off test299. Disclosed explicitly, plus the neutral-target variant from E7 as the version that uses no benchmark statistic. Never report a repair and its target statistic on the same rows.
4. **The V1 trainer had no seed; the noise floor is 0.040 F1; the V2 trainer's variance is measured only for the configurations in E8.**
5. **Pair conditioning is established only within the hard-label v14 corpus** and has not been shown to improve the best available configuration; test500 pair columns are missing for 201 rows.
6. **The NER head never predicts HOST or PATHOGEN; downstream triple extraction over 34,880 sentences produced an empty file.** This is a finding, not an apology — but the honest statement belongs here as well as in the body.
7. **Deployment precision.** The repo computes `precision_at_deploy_prevalence` at prevalence 0.0848 and records values in the 0.29–0.42 range across models and benchmarks. A paper reporting P = 0.925 on a 56%-positive benchmark while its own result files record ~0.35 at the deployment prior must report both. (The exact values differ between `results/v2/` files and need one pinned computation.)
8. **English only** (keep, L716–718).

---

## 7. The five papers that must be cited

All five verified today against ACL Anthology primary pages.

| # | Citation | Why it is mandatory |
|---|---|---|
| 1 | **Gururangan, Swayamdipta, Levy, Schwartz, Bowman & Smith.** *Annotation Artifacts in Natural Language Inference Data.* NAACL-HLT 2018 (Vol. 2, Short Papers), **pp. 107–112**. [N18-2017](https://aclanthology.org/N18-2017/) | The canon. A submission reporting a lexical-cue artifact without this citation reads as unaware of the field. Concede it in the first paragraph of Related Work. |
| 2 | **Poliak, Naradowsky, Haldar, Rudinger & Van Durme.** *Hypothesis Only Baselines in Natural Language Inference.* \*SEM 2018, **pp. 180–191**. [S18-2023](https://aclanthology.org/S18-2023/) | The partial-input diagnostic's direct parent. Your cue-only baseline is its lexical analogue; position explicitly against it. |
| 3 | **Gururangan, Card, Dreier, Gade, Wang, Wang, Zettlemoyer & Smith.** *Whose Language Counts as High Quality? Measuring Language Ideologies in Text Data Selection.* EMNLP 2022, **pp. 2562–2580**. [2022.emnlp-main.165](https://aclanthology.org/2022.emnlp-main.165/) | The closest conceptual parent and the one the prior analyses missed: a *quality filter* encoding a bias it was never meant to encode. This is the citation that makes GATE 6a/6b legible as a general phenomenon rather than a local bug. |
| 4 | **Huang, Hao, Ye, Zhu, Feng & Zhao.** *Does Recommend-Revise Produce Reliable Annotations? An Analysis on Missing Instances in DocRED.* ACL 2022 (Vol. 1, Long Papers), **pp. 6241–6252**. [2022.acl-long.432](https://aclanthology.org/2022.acl-long.432/) | The nearest structural precedent in RE: a construction procedure produces one-sided false negatives, and models trained on the corpus *inherit the same bias*. Cite it and say precisely how your mechanism differs (theirs is annotator incentive, yours is an automated filter). |
| 5 | **El Khettari, Quiniou & Chaffron.** *Building a Corpus for Biomedical Relation Extraction of Species Mentions.* BioNLP @ ACL 2023, **pp. 248–254**. [2023.bionlp-1.21](https://aclanthology.org/2023.bionlp-1.21/) | The SSI corpus: manually annotated, sentence-level, binary species–species relation extraction. This is your task's nearest published neighbour and it is cited nowhere in the current 23-entry bibliography. It is also a better external calibration corpus than ChemProt for the diagnostic (same input format, same unit, public, small) — and its gut-microbiota scope leaves your ecological breadth as a real differentiator, if you say so yourself. |

**Next tier, if space allows:** Niven & Kao, ACL 2019, pp. 4658–4664 (cue productivity/coverage — the closest existing analogue to your P(cue∣·) pair); Rosenman, Jacovi & Goldberg, *Exposing Shallow Heuristics of Relation Extraction Models with Challenge Data*, EMNLP 2020, pp. 3702–3710 (the parent for a zero-signal-positive challenge subset — note the title, which one upstream analysis got wrong); Boisson, Espinosa-Anke & Camacho-Collados, EMNLP 2023 (an applied-domain construction-artifact audit accepted at a main track — the template for this paper's shape); Ben-Hur & Noble, BMC Bioinformatics 7(Suppl 1):S2, 2006 (negative-example construction biases accuracy estimates in computational biology — the mirror image of your positive-side filter, and the citation that makes the work legible to bioinformatics reviewers); Baldini Soares et al., ACL 2019, pp. 2895–2905 and Zhou & Chen, AACL-IJCNLP 2022 (mandatory the moment Section 6 exists). **Verify pages for Gardner et al. EMNLP 2021 and Geirhos et al. Nat. Mach. Intell. 2020 before typesetting — neither was independently confirmed.**

---

## 8. Pre-emptive rebuttals

### Objection 1 — "You filtered the positives with lexicon L and then measured the shortcut with lexicon L. This is a tautology."

This is the first thing a competent reviewer will write, and it is correct about the naive version of the experiment. Three-part answer, in this order.

**(a) The tautology is the mechanism, and the mechanism is not the finding.** Yes, the filter selected for the cue — that is what a class-asymmetric filter does, and stating it plainly costs nothing. What is *not* tautological is everything downstream: the filter was applied at construction time by someone trying to improve label quality; it is enforced by a test suite that fails the build otherwise; it passed every leakage check the field runs (exact match, TF-IDF near-duplicate at 0.85, taxon-pair overlap — App D, L918–944); and it deterministically relocates the trained model's error budget onto a stratum that aggregate F1 cannot see. Nothing about the cue's origin predicts a 2.3–2.6× FNR asymmetry or a third of the error budget landing in one cell. That has to be measured, and it is the paper.

**(b) The signature survives cue sets we did not design (E2).** Train-only log-odds top-k, ROBI relation labels alone, and a held-out half of the lexicon. One arm of this already exists and survives: `interaction_taxonomy.scan_globi_terms` — a different code path from the builder's — gives benchmark negatives 44.2% vs training negatives 4.5% vs template negatives 0.0%.

**(c) The injection experiment cannot be reached by this objection at all (E5).** Cues induced from ChemProt's own training split, filter applied to ChemProt, evaluation on ChemProt's untouched test set. We built none of it. If the error signature reproduces there, the finding is a construction hazard.

Finally, do not overstate in the other direction: cite Feng, Wallace & Boyd-Graber (ACL 2019) defensively — partial-input *failure* proves nothing, so the benchmark's lower gap (+0.196) must not be presented as proof the benchmark is clean.

### Objection 2 — "Shortcut learning has been known since 2018. Where is the new insight?"

Concede the canon in the first paragraph of Related Work, then draw three lines it does not cross.

**The cue is not what annotators wrote; it is what the pipeline kept.** Gururangan 2018 and Poliak 2018 describe artifacts of *human* elicitation. Adversarial filtering (AFLite, SWAG) describes a filter applied *deliberately and symmetrically* by dataset authors. Neither covers an automated quality filter applied to one class during corpus cleaning — which, in 2026, is the modal way biomedical and biodiversity RE corpora are built, now that the filter is usually an LLM judge. E3 quantifies how common this is; if the rate is non-trivial, the claim is general, and if it is near-zero, be honest and take the paper to Database.

**The bias is installed by the quality-assurance system itself.** `tests/test_training_data.py` GATE 6a fails the build if >20% of positives lack a lexicon cue; GATE 6b fails it if >2% of negatives carry a strong un-negated interaction term, on the stated grounds that such negatives are "likely mislabeled." The benchmark's negatives carry interaction language roughly 45% of the time. The gate that certifies label quality is the gate that certifies the benchmark cannot be matched. That framing — *the quality filter installs the shortcut it is testing for* — has a published parent (Gururangan et al., EMNLP 2022) in pretraining data selection and, as far as our searches go, no instance in relation extraction. (Absence is not provable by web search; run an ACL Anthology + Semantic Scholar full-text query before printing any "to our knowledge" sentence.)

**Nobody has audited this domain.** Searches for shortcut or partial-input analyses of GloBI-derived or biodiversity interaction corpora return only ecological coverage-bias work. TaxoNERD, the ecological text-mining literature, and the 2025–26 LLM-extraction preprints all report F1 without a partial-input control. Pair this with the trivial-baseline result — no model in the paper beats "always yes" on EP-passage — and the claim is field-level, not self-reported.

### Objection 3 — "This is a project post-mortem, not a paper. The authors describe fixing their own bugs."

The real risk, and it is a writing risk as much as an evidence risk.

**Lead with a controlled experiment, not a confession.** Section 3 is the factorial presented as a methodological result about ablation practice: recipe-level factors in this literature are routinely reported at effect sizes below unmeasured run-to-run variance, and here is a worked case where four such findings evaporate. State it as *unresolvable at the available resolution*, never as *proven null* — the sweep is n = 1 per cell (all 48 rows carry `seed == 1`), which cannot establish a null, and E8 supplies the variance denominator that makes the statement legitimate.

**Strip the first person from the failure narrative.** "A corpus built this way," not "our corpus." "A system deployed with a benchmark-fitted threshold," not "we deployed." The facts are identical; the register moves from confession to analysis, and reviewers respond to register far more than they admit.

**Give the community a checklist with costs and yields attached.** Cue-only baseline: 0 GPU, caught a 0.857-vs-0.638 gap. Trivial baseline: 0 GPU, caught a source no model beats. Seed replication: 0.18 GPU-h per run, caught a floor that voids four published findings. Pre-declared thresholds: 0 GPU, caught a production threshold equal to the benchmark's F1-argmax. Non-empty-output assertions: 0 GPU, would have caught zero triples over 34,880 sentences on day one.

**And the generalisation is the answer, not the framing.** E4 and E5 are what convert this from a lab notebook into a method. Run them before writing Section 1; if E5 does not land, the honest venue is Database (Oxford), where a biocuration-methodology finding is the unit of contribution and a case study is acceptable.

### Objection 4 — "Your benchmark and your repair are both contaminated by the thing you are criticising."

The sharpest available objection, because the evidence is in the artifacts you are going to release. Answer it in Section 2, before anyone raises it.

**On the benchmark.** test299 is 99 rows of EP-A (the EP-relax set on which ~30 configurations were selected, 99/99 exact match), 100 rows of EP-passage (85% positive, below the trivial baseline for every model), and 104 rows of eval-100/BioTx (31 of which are exact-matched into non-reported training corpora, with 25 gold positives living in `data/training/`). Disclose the composition, report every headline with and without EP-A, and demote test500 to a secondary table with its own disclosure that 99 of its rows are EP-relax and 197 are LLM-generated.

**On the repair.** `v2_d1_v14_signalmatched.manifest.json` and the D3 manifest both record `target_marginals.source: "test299"`. This is not row-level leakage — no sentences move — but it is test-set-informed corpus design in a paper about test-set-fitted decisions, and it is in a file you are releasing as your reproducibility contribution. Three moves, all cheap: state it explicitly in Section 2; add the neutral-target variant (P(cue∣pos) = P(cue∣neg), which uses no benchmark statistic, E7, 0.6 GPU-h); and never report a repair and its target statistic on the same rows.

**On the general principle.** The paper's argument is that measurement decisions made by reading the evaluation set are invisible until someone looks. Pre-registering the E6 read-out — writing down, before the runs land, which of the three outcomes maps to which section text — costs a paragraph and is itself a demonstration of the discipline being argued for. Do it, and say in the paper that you did.

---

## 9. Numbers I recomputed today (authoritative)

Run 2026-08-25 with `src/data/interaction_lexicon.score_sentence`. **The two prior analyses disagreed because they used two different cue definitions; both were correct.** `build_v14_dataset.py:35-37` defines the filter as `strength > 0.0`, so that row is primary.

| Corpus | Cue definition | P(cue∣pos) | P(cue∣neg) | gap | cue-only F1 |
|---|---|---|---|---|---|
| v14 train (n=34,880, 32.7% pos) | `strength > 0` ← **the filter actually applied** | 0.949 | 0.129 | 0.820 | **0.857** |
| v14 train | `has_signal` (≥0.15 or ≥2 strong) | 0.924 | 0.067 | 0.857 | 0.897 |
| test299 (n=299, 54.5% pos) | `strength > 0` | 0.644 | 0.449 | 0.196 | **0.638** |
| test299 | `has_signal` | 0.626 | 0.419 | 0.207 | 0.634 |
| test500 (n=500, 56.2% pos) | `strength > 0` | 0.662 | 0.356 | 0.306 | 0.683 |
| D1 repair (n=26,588) | `strength > 0` | 0.644 | 0.449 | 0.195 | **0.5365** — matches the manifest exactly |

Lexicon size: `len(STRONG_TERMS) = 65`, `len(WEAK_TERMS) = 41` → **106 regex patterns**. Do not write "641-term" anywhere; 591 + 50 is the SIBiLS/ROBI NER-supervision lexicon (manuscript L338), a different object.

In-flight check at time of writing: both GPUs at 96–99%; `results/v2_stage3/` contains D0s and D2 arms at seeds s1/s2/s3, most recent `train_summary.json` written 13:20 today.

**Sources:** [ACL Rolling Review — Dates and Venues](https://aclrollingreview.org/dates) · [ARR Call for Papers](https://aclrollingreview.org/cfp) · [NAACL 2027 Call for Main Conference Papers](https://2027.naacl.org/calls/main_conference_papers/) · [Gururangan et al. NAACL 2018](https://aclanthology.org/N18-2017/) · [Poliak et al. \*SEM 2018](https://aclanthology.org/S18-2023/) · [Gururangan et al. EMNLP 2022](https://aclanthology.org/2022.emnlp-main.165/) · [Huang et al. ACL 2022](https://aclanthology.org/2022.acl-long.432/) · [El Khettari et al. BioNLP 2023](https://aclanthology.org/2023.bionlp-1.21/)