# Stage: analysis (runs in parallel with the literature stage)

Goal: turn the sentence-level sandbox into reproducible evidence of paper quality, and run the one
experiment the biodiversity claim lacks: a fair sentence-level competitor. Be adversarial. If the
evidence turns out weaker than the sandbox suggested, write that plainly. The user prefers a true
small claim to a large fragile one.

Work in this order. A1 comes first because it is the long GPU job; do A0 and A2–A6 while it runs.

## A0. The annotation sheets (5 minutes)
UPDATE 02:45: the 22-item comparison is already done (CONTEXT.md §4). A relaunched session skips the
comparison. It uses `gold_review_2026-10-02_v2_eg_curated.xlsx` as the 22-item label source in A5.

Count the filled SENTENCE and PAIR answers in both sheets, ignoring the SKIP rows. If, and only if,
all 22 items of the gold-review sheet have both answers, compare the PAIR answers with the key's
`label` and write `results/reframe_2026-10-05/gold_review_22.md` containing:
- agreement (n, %, Cohen's kappa);
- each disagreement (item, passage excerpt, taxa, the user's answer, current gold, `p_triple` and
  `p_shipped`);
- the UNSURE items;
- the SENTENCE versus PAIR answers (how many sentence-YES and pair-NO).

These are proposals for the user, never applied. If the sheet is incomplete, record only the counts
and do not read the key's labels.

## A1. The fair biodiversity competitor (start it first; detached GPU job)
The paper's biodiversity sentence arm learned from candidate labels. The fair competitor learns
from the teacher's answer to the sentence question.
- Teacher: the paper's teacher, `qwen3:32b` on the system Ollama (:11434), the same server as the
  corpus labels. Prompt: `BIODIV["sentence"]` from `scripts/llm_baseline.py`, scored through its
  `ask()` (P(YES); label = greedy answer YES). Import these; do not copy the prompt by hand.
- Throughput: measure on 200 random unique passages of `data/training/distill/v3_combined_train.csv`
  (`text` column), trying 1 and 4 concurrent requests. If all 34,242 unique passages fit in 2.5
  hours, label them all. Otherwise, label a random sample (seed 0) sized to fit 2.5 hours, of at
  least 8,000 passages. If 8,000 passages would take more than 3.5 hours, do not run A1: write
  `state/gpu_sentlab.failed` with the reason and continue with the rest of the stage.
- Output: a NEW file, `data/training/distill/v3_sentence_teacher_qwen3-32b.csv` (text, p_yes, answer,
  label). Append as you go and resume after an interruption; never overwrite.
- Students: one row per labelled passage, trained with `experiments/multitask/train_student.py
  --input-format sentence`, the paper's recipe (3 epochs, seeds 1–3). The development set is 10%
  of the passages, grouped by passage and passed as `--dev-data` (build the split once, seed 0, and
  save it as new files). Output to `models/sentence_level/biodiv_sentlab_s{1,2,3}`.
- If only a sample was labelled, also train the matched pair arm: `--input-format pair` on the
  corpus rows whose passage is in the labelled sample, with the same development passages held out,
  into `models/sentence_level/biodiv_pair_matched_s{1,2,3}`. Without it, the comparison confounds
  question with data size. The full-corpus pair arm (`models/pair_baseline/xenc_s{k}`) stays as the
  reference.
- Score all arms on the 437 passages at the sentence level:
  - sentlab: one score per passage;
  - pair arms: max over the TaxoNERD pairs plus the candidate, using the cached mentions, as in the
    sandbox;
  - thresholds: block-held-out, as in the paper.
  Write the results to `results/paperA_v2/sentence_level/biodiv_sentlab.json`.
- Run all of A1 as ONE detached script: labelling, then training, then scoring, under `timeout`.
  The script itself writes `results/reframe_2026-10-05/state/gpu_sentlab.done` when it succeeds or
  `gpu_sentlab.failed` (with the reason and the log path) when it does not; check the outputs, not
  only the exit code. The orchestrator waits for one of the two markers before starting the writer.
  Launch it as unit `reframe-gpu-sentlab` (add `-2`, `-3`, and so on if you relaunch it). Re-running the
  script must resume: labelling appends, and training skips finished models. Before launching, check
  that no copy is already running (`systemctl --user list-units 'reframe-*'`).

## A2. Reproducible scripts
Move the sandbox analysis into `scripts/sentence_level/` and keep the sandbox files. Write one
module per benchmark plus `make_sentence_tables.py`, which writes every number the draft may cite to
`results/paperA_v2/sentence_level/*.json`. Follow the conventions of `scripts/paperA_tables.py`:
order-free scoring, block-held-out thresholds on biodiversity, development thresholds on BioRED,
type hints, argparse, docstrings. The scripts must reproduce the sandbox numbers in CONTEXT.md §3
exactly. If they do not, find out why and record it.

## A3. BioRED at the sentence level
- AUPRC per seed and as mean ± sd; P/R/F1 at development thresholds. Arms: sentlab, sentence,
  pair-max.
- Paired bootstrap over sentences (10,000 replicates, seed 0): 95% CIs for the AUPRC differences
  pair-max − sentlab and pair-max − sentence. Exact McNemar at the development thresholds.
- Strata: the number of distinct entities in the sentence (2 vs ≥3) and the number of candidate
  pairs (1 vs ≥2). Is the advantage concentrated where several pairs compete?
- The precision ceiling on BioRED: the pair-level precision of a perfect sentence filter (positive
  candidates divided by all candidates in positive sentences).
- Label semantics: quantify how far "co-mentions a related pair" is from "expresses a relation", if
  BioRED's annotations allow it. If they do not, state the limitation precisely.
- Zero-shot LLM rows at the sentence level, from the existing
  `results/paperA_v2/llm/biored_*_{sentence,pair}.csv`. Use only sentences whose every candidate
  was scored: the sentence question gives P(YES) per sentence, and pair-max is the max over the
  sentence's candidates.
  - Check coverage first.
  - If fewer than 300 test sentences are complete for qwen3-32b, score the missing candidates of a
    random sample of 500 test sentences (seed 0) with qwen3:32b on :11434, following the
    `llm_baseline.py` conventions, into NEW files. Do this only if the job takes 45 minutes or
    less, and run it after or alongside A1 without starving it.
  - Also report the other Qwen sizes where coverage allows.

## A4. Biodiversity at the sentence level (silver labels, provisional)
- Reproduce the current numbers and add bootstrap CIs (resample passages).
- Sensitivity of the silver labels. Use three definitions:
  - (i) majority of the three LLMs (the current one);
  - (ii) the two non-teacher LLMs: Qwen3.5-122B and Qwen3.8-27B both YES gives 1, both NO gives 0,
    and rows where they disagree are dropped;
  - (iii) Qwen3.5-122B alone.
  
  Report whether the order of the arms changes under any of them. Under each definition,
  pair-positive candidates stay positive.
- The 0.724 precision ceiling with its CI.
- What TaxoNERD pair enumeration costs (pairs per passage, scoring time on the GPU and on a CPU for
  a sample) and what the candidate pair alone gives (pair-own).
- Do not score LLMs at the sentence level against silver labels: that is circular. Score them only
  against human labels (A5).

## A5. Human sentence labels (`--human`)
Read the SENTENCE answers from both sheets. Map rows through the item→bench_row columns only, and
drop UNSURE and SKIP rows. Evaluate every arm, and the LLM sentence/pair-max rows, on the
human-labelled subset. Report human-vs-silver agreement (Cohen's kappa), and the PAIR answers
against the benchmark's pair gold, as agreement only.

It must run cleanly with no labels (write "no human labels yet"). Running it again later must
regenerate everything with one command; put that command in ANALYSIS.md.

## A6. ANALYSIS.md
Write `results/reframe_2026-10-05/ANALYSIS.md`. It must contain:
- every result with its JSON path;
- the commands that regenerate the results;
- the caveats;
- a final section, "What the evidence supports and what it does not", on the claim "ask about the
  pair, even if you want the sentence".

That final section must say:
- whether the claim holds on BioRED with significance;
- whether it holds on biodiversity under each silver definition, and against the fair competitor
  from A1 if that has finished (if not, say it is pending and that the reviewer stage integrates
  it);
- what would change the conclusion.

Do not commit; the writer stage commits. Write `state/analysis.done` once A0 and A2–A6 are complete
and A1 is launched, or `state/gpu_sentlab.failed` exists. The A1 results may arrive after you
finish.
