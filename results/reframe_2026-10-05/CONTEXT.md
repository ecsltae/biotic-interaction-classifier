# Paper A, sentence-level reframe: context for the unattended stages (2026-10-05)

All paths are relative to `~/MetaP/classifier` unless absolute.

## 1. Where the paper stands

`paperA/paperA.tex` (→ `paperA.pdf`) is submission-ready for ACL Rolling Review and is the user's
submission version. **No stage modifies it** (nor `paperA.pdf`, `paperA/references.bib`,
`paperA/SUBMISSION.md`). Title: "Which Pair Is It About? Pair-Conditioned Verification of
Literature-Mined Biotic Interactions". State: 8-page body (Conclusion ends on page 8), 17 pages in
all, abstract 196 words (limit 200), pubcheck clean, 785 numbers audited
(`results/paperA_v2/NUMBER_AUDIT_2026-10-04.md`, `scripts/audit_paper_numbers.py`), 58 references
verified (`scripts/check_references.py`).

What it shows today (pair level):
- Setting: literature-mining pipelines propose (taxon, relation, taxon) candidates by co-occurrence,
  then filter them. The paper argues the filter should ask "does the passage assert an interaction
  between *these two*?" (pair-conditioned verification), not "does the passage describe an
  interaction?" (sentence-level question).
- Controlled comparison: one encoder (BiomedBERT-base, 110M), one 48,338-row corpus labelled by a
  32B teacher (Qwen3-32B, local Ollama, zero-shot, temperature 0, asked per candidate whether the
  passage supports *this* triple), three seeds, only the input differs (sentence / pair / triple).
- 437-row biodiversity benchmark (expert-graded candidates, PAIR-level gold, 246 positive, 7 source
  blocks): AUPRC 0.851 (sentence) → 0.918 (pair); F1 0.775 → 0.871 (McNemar p = 1.3e-7); the
  relation term adds nothing.
- Zero-shot LLMs (Qwen3 0.6B–32B, Qwen3.5-122B) show the same gap; the pair question beats the
  sentence question at every size from 1.7B.
- BioRED, BC8 protocol, sentence-scope candidates: AUPRC 0.738 → 0.843, entity-pair F1 +9.6 points;
  no gain on sentences naming two entities, +0.120 on sentences naming more.
- Appendices: teacher prompts, zero-shot scaling, soft-label distillation, encoders, escalation
  cascade, error examples, candidate rules, artifacts and compute, negative results.

## 2. The user's decision (2026-10-05): the target is SENTENCE level

The user, verbatim:
- "i want to say the sentence level because that is my intention since the begining and is the
  spirit of the curations done and the test set. could the paper go that way or do we loose too much ?"
- "some other sentences have sentence 1 and pair level 0. we want sentence level in the end"
- "i think a paper strength would also be to go from pair level knowledge to sentence level. no ? ...
  in the end what [the application owner] wants is the sentence level. the whole goal is to assess whether or not a
  sentence has a biotic interaction or not.. and this part is clear from the start."
- On BiotXplorer: "we do have something strong with the biotxplorer for the paper, is that not
  novelty ? we don't want to overlap with the upcoming biotxplorer paper from [the application owner] but that is
  something to sell no ? or is it off topic ?" (the application owner runs BiotXplorer, the application,
  and has a paper about it coming.)
- On uncertain items: UNSURE answers are removed: "it is too uncertain. several annotators blindly
  have different responses".

Decisions already taken (do not re-open):
- Dual labels. SENTENCE: "does the passage describe any biotic interaction?" PAIR: "do THESE two
  organisms interact with each other, as written?" Each YES / NO / UNSURE; UNSURE rows are dropped.
- BiotXplorer is the **application** (where verified output goes, why the sentence decision
  matters), **not the novelty**. Nothing that overlaps its own upcoming paper (no description of
  its internals, no claims about it beyond what is public). Double-blind: no names, affiliations, or
  identifying URLs anywhere in drafts.
- The paper's comparison models stay as they are: Qwen3 (3.0 family) and Qwen3.5-122B as baselines
  and the Qwen3-32B teacher for the corpus. Do not swap in newer Qwen versions for comparisons.
- The submission version stays submittable. The reframe is a separate draft the user will compare.

## 3. The reframe to build: "ask about the pair, even if you want the sentence"

Claim to test (adversarially): for passages retrieved by co-occurrence, a sentence-level decision
read off pair-level verification (max over the passage's candidate pairs) is at least as good as a
model trained to answer the sentence question directly; and only pair-level answers can populate a
pair database (a perfect sentence filter has limited pair precision).

Sandbox evidence so far (`sandbox/sentence_level/{biored_sentence,biodiv_sentence}.py`, results in
`sandbox/sentence_level/results/`):

BioRED, BC8 test split, 2,582 sentences with at least one candidate pair, 0.838 positive. A sentence
is positive iff it co-mentions at least one entity pair that BioRED's gold relates. **Careful:**
derived exactly from gold, but BioRED relations are document-level, so this means "co-mentions a
related pair", not necessarily "this sentence expresses the relation". Thresholds from the
development split.

| arm | AUPRC (3 seeds) | P | R | F1 |
|---|---|---|---|---|
| pair-max: the paper's pair arm, max over the sentence's candidates | 0.966 ± 0.003 | 0.930 | 0.980 | 0.954 |
| sentlab: sentence-input model trained on sentence labels (one row per sentence, same recipe) | 0.952 ± 0.003 | 0.878 | 0.989 | 0.930 |
| sentence: the paper's sentence arm (trained on candidate rows) | 0.929 ± 0.002 | 0.872 | 0.976 | 0.922 |

Biodiversity, 437 rows, **SILVER** sentence labels (candidate pair-positive → 1; otherwise the
majority of three LLMs' answers to the sentence question: Qwen3-32B, Qwen3.5-122B, Qwen3.8-27B;
provisional until the user's human SENTENCE labels exist): 0.778 sentence-positive vs 0.563
pair-positive. Block-held-out thresholds.
- pair-max over every pair of TaxoNERD taxon mentions plus the candidate (mean 14.3 pairs per
  passage; mentions cached in `sandbox/sentence_level/results/taxonerd_mentions.json`):
  AUPRC 0.966, P 0.864, R 0.932, F1 0.897
- the paper's sentence arm: AUPRC 0.957, P 0.915, R 0.791, F1 0.849
- pair-own (the candidate only): AUPRC 0.962
- A perfect sentence filter accepting every sentence-positive candidate has pair-level precision
  0.724 (246/340).

Weaknesses a reviewer will raise; the analysis must address them, not hide them:
1. The biodiversity sentence arm was trained on candidate labels, not sentence labels, so it is not
   a fair sentence-level competitor. BioRED has a fair one (sentlab); biodiversity does not yet.
2. Silver labels partly come from LLM sentence verdicts, one of them the teacher's: circularity.
3. BioRED's base rate is 0.838: AUPRC differences of about 0.014 need paired significance tests.
4. Max over pairs is the multi-instance "at least one" assumption. The novelty is not the max; it is
   that, with the same teacher, data and encoder, verifying pairs gives a better sentence filter
   than asking the sentence question, and also yields the pair.
5. TaxoNERD-based pair enumeration adds a mention detector the sentence model does not need: report
   what it costs (pairs per passage, time) and what happens with the candidate pair alone.

## 4. File map

- Benchmark: `src/eval/core.py: clean_benchmark()` (437 rows, SHA-asserted; columns include
  sentence, species1, species2, label (pair-level gold), source (block)).
- Scoring helpers: `scripts/paperA_tables.py` (`order_free`, `block_held_out`, `prf`, `llm_rows`;
  arm paths `ARMS_BIODIV` = sentence `models/sentence_baseline/xenc_s{k}`, pair
  `models/pair_baseline/xenc_s{k}`, triple `models/student_v3/xenc_s{k}`; `ARMS_BIORED` =
  `models/biored_bc8/{sentence,pair}_s{k}`). Cached biodiversity scores:
  `results/paperA_v2/S_biodiv_{sentence,pair,triple}.npy`.
- Training: `experiments/multitask/train_student.py --data --dev-data --input-format
  {sentence,pair,triple,...} --seed --epochs 3 --out` (the paper's recipe; writes
  `student_config.json` with `threshold_dev`). The BioRED sentlab models used
  `--data data/benchmarks/biored_bc8_sentence/train.csv --dev-data .../dev.csv --input-format sentence`.
- Teacher corpus: `data/training/distill/v3_combined_train.csv` (48,338 rows; columns text,
  source_species, target_species, interaction_type, label, kind; 34,242 unique passages, 1.41 rows
  per passage).
- BioRED: candidates `data/benchmarks/biored_bc8/{train,dev,test}.csv`; sentence tables
  `data/benchmarks/biored_bc8_sentence/{train,dev,test}.csv`; models
  `models/biored_bc8/{sentence,pair,pair_mark,mark_canon}_s{1-3}`,
  `models/sandbox_sentence/biored_sentlab_s{1-3}`.
- Zero-shot LLMs: `scripts/llm_baseline.py` (`--bench biodiv|biored|bioreddev`, `--model qwen3:32b`,
  `--tag=-v035` for the newer server; `OLLAMA_URL` env picks the server; prompt dicts `BIODIV` and
  `BIORED` with keys sentence/pair; `ask()` returns P(YES) from first-token log-probabilities, think
  off). Outputs `results/paperA_v2/llm/{bench}_{model}{tag}_{question}.csv` (column `verdict`,
  `p_yes`). The BioRED LLM runs used a 3,000-candidate sample (`--n 3000`).
- Ollama: system Ollama 0.18.3 at http://localhost:11434 (qwen3:32b = the paper's teacher,
  qwen3.5:122b, smaller qwen3); user-level Ollama 0.35.1 at http://localhost:11435 (unit
  `ollama-latest`; qwen3.8:27b, also a qwen3:32b copy). GPU: one A100 80 GB, free at launch.
- Annotation sheets (the user fills them; columns F = SENTENCE, G = PAIR, each YES/NO/UNSURE,
  H = note):
  - `data/evaluation/gold_review_2026-10-02_v2_BLIND.xlsx`, sheet `review`, 22 items. Rows map to
    benchmark rows through the sealed key
    `results/paperA_rebuild_2026-09-28/gold_review_v2_KEY_do_not_open_before_review.csv` (columns
    item, bench_row, source, _kind, label, p_triple, p_shipped). Read only `item` and `bench_row`
    for mapping, unless all 22 items have both answers.
  - `data/evaluation/second_annotation_2026-10-04_BLIND.xlsx`, sheet `annotate`, 437 rows in a
    randomised, block-balanced order. The 20 rows marked `SKIP (gold-review item N)` are the
    gold-review items, so ignore them. Map rows through `data/evaluation/second_annotation_2026-10-04_KEY_rows_only.csv`,
    which has columns item, bench_row, source and review_item and contains no labels.
  - At launch (01:35), the server copy of the 22-item sheet had no answers and the 437-row sheet had only the SKIP rows. The user may upload
    filled copies later, so check again at the start of each session.
  - **UPDATE 02:45.** The user uploaded their completed 22-item review:
    `data/evaluation/gold_review_2026-10-02_v2_eg_curated.xlsx` (sheet `review`, same layout, plus
    column I with a model's prefill reasoning).
    - Use it as the human-label source for the 22 items, NOT the `_BLIND.xlsx` copy, which stays
      empty.
    - Never use `gold_review_2026-10-02_v2_PREFILLED_by_claude.xlsx` as labels: it is a model's
      prefill, which the user then corrected.
    - The comparison with the sealed key (A0) is done:
      `results/reframe_2026-10-05/gold_review_22.md` (with `_comparison.csv` and `_whatif.json`).
      `gold_review_report.py` may regenerate that file; the parent session's version, with the
      provenance notes, is kept in `gold_review_22.parent.md` (read both).
    - At 02:50 `scripts/sentence_level/common.py` GOLD_SHEET was switched to the eg_curated file
      (messages to the running sessions were held undelivered).
      PAIR agrees with gold on 15/22 (κ 0.36), and on 11/11 controls. All 7 disagreements (items
      2, 3, 4, 10, 12, 16, 18) side with both models. These are proposed flips; the user has not
      decided, so the paper's gold is UNCHANGED.
    - The answers are NOT blind: the user adjudicated a pre-filled sheet. Never call it blind.
    - SENTENCE answers: YES 15, NO 4, UNSURE 3 (items 3, 4, 5; dropped). Item 4 has PAIR YES with
      SENTENCE UNSURE, probably a slip; it stays dropped.
    - The 22 items are NOT a random sample (11 were selected by model-vs-gold disagreement). Use
      their SENTENCE labels only as a spot check of the silver labels, never as a benchmark
      result.
- Paper structure (paperA.tex): Introduction; Setting; Two formulations, one corpus; Results
  (Conditioning on the pair; A public benchmark: BioRED; The query, not the encoder; An operating
  curve, not a verdict); What the remaining errors are made of; Candidate rules; The deployed
  configuration; Related work; Conclusion; Limitations; appendices.
- Email: `bash ~/MetaP/classifier/scripts/notify.sh "Subject" "Body"`.
- Memory (update at the end, reviewer stage only):
  `~/.claude-hesso/projects/-home-USER-MetaP/memory/project_paper_a.md`.

## 5. Stage outputs (where each stage writes)

- analysis → `scripts/sentence_level/`, `results/paperA_v2/sentence_level/*.json`,
  `results/reframe_2026-10-05/ANALYSIS.md`, GPU job marker `state/gpu_sentlab.{done,failed}`.
- literature → `results/reframe_2026-10-05/LITERATURE.md`, `paperA/reframe_extra.bib`.
- writer → `paperA/paperA_reframe.tex` / `.pdf`, `paperA/REFRAME_NOTES.md`; local commit.
- reviewer → `results/reframe_2026-10-05/REVIEW.md`, fixes to the draft, memory update, the email.
