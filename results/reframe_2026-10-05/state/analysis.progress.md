# analysis stage, progress (session 1, started 2026-10-05 01:47)

## A0 annotation sheets, DONE (01:50)
- gold_review_2026-10-02_v2_BLIND.xlsx (sheet review): 22 rows, 0 SENTENCE answers, 0 PAIR answers.
- second_annotation_2026-10-04_BLIND.xlsx (sheet annotate): 437 rows, 20 SKIP rows; of the 417 others,
  0 SENTENCE answers, 0 PAIR answers.
- Sheet incomplete -> key labels NOT read (only item, bench_row columns). No gold_review_22.md written.
- Next: A1 throughput measurement, then launch the detached GPU job.

## A1 launched (01:53), unit reframe-gpu-sentlab
- Throughput (results/paperA_v2/sentence_level/sentlab_throughput.json): 200 passages, 0.344 s/passage at
  1 worker, 0.203 s/passage at 4 workers -> all 34,242 passages ~1.93 h < 2.5 h: label ALL (4 workers).
  Matched pair arm therefore not needed (trained only if the 3-h safety stop leaves a sample).
- Scripts: scripts/sentence_level/{sentlab_label.py, sentlab_split.py, gpu_sentlab.sh}. Job log
  results/reframe_2026-10-05/logs/gpu_sentlab*.log. Labelling order = seed-0 permutation (any prefix is a
  random sample); safety stop 04:53. Step 4 calls `scripts/sentence_level/biodiv_sentence.py --sentlab`
  which must exist (A2) before labelling ends (~03:50) and must write biodiv_sentlab.json with keys
  "arms" and "by_silver".
- BioRED LLM coverage (A3): only 149 test sentences complete for every Qwen size (49 for qwen3.5-122b),
  < 300 -> fill qwen3-32b on a 500-sentence sample (next).

## A3 BioRED LLM fill launched (01:57), unit reframe-biored-llm (2 workers, timeout 45 min)
- scripts/sentence_level/biored_llm_fill.{py,sh}; outputs results/paperA_v2/sentence_level/llm/biored_qwen3-32b_sent500_{sentence,pair}.csv
  (500 random test sentences, seed 0: 250 sentence questions + 2,598 missing pair candidates). Log logs/biored_llm_fill.log.

## A2 (partial, 02:12)
- scripts/sentence_level/common.py (AP vectorised = sklearn incl. ties, bootstrap, exact McNemar, Wilson, kappa,
  human-sheet reader; gold-review key read via item/bench_row only; key bench_row indexes the 449-row file;
  gold-review items 12, 14 = clean rows 66, 87 also appear unskipped in the 437-row sheet (items 45, 159)).
- scripts/sentence_level/biodiv_sentence.py -> results/paperA_v2/sentence_level/biodiv_sentence.json DONE (A4 core).
  Reproduces the sandbox: P/R/F1/accepts/thresholds/pair count/ceiling identical; pair-max AUPRC differs <=1.8e-5:
  cause = TF32 (paper convention, used here) vs FP32 (sandbox); FP32 reproduces all seeds + ensemble bit-identically
  (scripts/sentence_level/order_check.py -> order_check.json). Row order has no effect.
- Subagent: BioRED label semantics DONE -> scripts/sentence_level/biored_label_semantics.py,
  results/paperA_v2/sentence_level/biored_label_semantics.json.
- Subagent running: TaxoNERD cost -> scripts/sentence_level/taxonerd_cost.py, biodiv_enum_cost.json.
- Next: biored_sentence.py (A3), then --sentlab/--human checks, make_sentence_tables.py, ANALYSIS.md.

## A2/A3/A4/A5 code DONE (02:26)
- scripts/sentence_level/: common.py, biodiv_sentence.py (silver A4; --sentlab A1 scoring; --human A5),
  biored_sentence.py (A3; --fp32 reproduction), biored_llm_fill.py/.sh, biodiv_llm_enum.py/.sh, order_check.py,
  gold_review_report.py (A0 when the sheet is complete), make_sentence_tables.py (one command, ~2 min, cached scores),
  subagents' biored_label_semantics.py and taxonerd_cost.py.
- Score cache keyed by model path + student_config hash (old arm-named cache moved to scores/superseded_names/).
- `--sentlab` path tested with stand-in models (sentence baseline): JSON has "arms" and "by_silver" as gpu_sentlab.sh expects.
- `--human` tested with synthetic labels (in memory only); without labels writes "no human labels yet".
  Gold-review rows (incl. clean rows 66, 87) are excluded from human analyses until the 22-item sheet is complete.
- gold_review_report.py tested on a synthetic sheet+key in /tmp (deleted); real sheet empty -> no key labels read.
- Results so far: biored_sentence.json, biodiv_sentence.json, biodiv_human.json, biored_label_semantics.json,
  biodiv_enum_cost.json, order_check.json, sentence_tables.json.
- Queued unit reframe-biodiv-llm-enum: waits for sentlab_split.json, then qwen3:32b over the 6,253 enumerated
  biodiversity pairs (for the A5 LLM pair-max row) -> results/paperA_v2/sentence_level/llm/biodiv_qwen3-32b_enum_pair.csv.
- Next: ANALYSIS.md (A6) with current numbers; re-run make_sentence_tables.py when the BioRED fill finishes;
  optional `biored_sentence.py --fp32` once labelling is done (GPU).

## 02:35, BioRED fill finished (exit 0, 02:32); tables regenerated
- make_sentence_tables.py run: biored_sentence.json now has qwen3-32b random500 (500/500 complete).
- biodiv strata by pairs scored (1 vs >=2) added to biodiv_sentence.json.
- Adversarial code-review subagent running (read-only).
- Next: write results/reframe_2026-10-05/ANALYSIS.md (A6). Later: biored_sentence.py --fp32 after labelling;
  integrate A1 (biodiv_sentlab.json) when gpu_sentlab.done appears; then analysis.done.
- 02:36 ANALYSIS.md drafted (A1 pending); waiting for code review + A1

## 02:50, CONTEXT.md update (02:40-02:45) integrated + code review fixes
- The user's completed 22-item review: data/evaluation/gold_review_2026-10-02_v2_eg_curated.xlsx (parent switched
  common.GOLD_SHEET at ~02:40; kept). Parent wrote gold_review_22.md (+.parent.md, _comparison.csv, _whatif.json):
  A0 DONE by the parent. gold_review_report.py now never replaces it; writes gold_review_22.script.md (same numbers:
  15/22, kappa 0.364, controls 11/11).
- --human: benchmark = 437-row sheet only; 22-item review = spot check of silver only (biodiv_human.json
  gold_review_spot_check: kappa majority3 0.46, nonteacher2 0.81, q122b 0.68 on 19 items).
- Code-review subagent: no statistical/alignment bugs. Fixed: human() split by sheet; gpu_sentlab.sh resume
  (atomic mv; running job keeps the old copy; skip labelling once split exists); sentlab_split sample_only from the
  split; random500 row skipped if incomplete; order_check cache path; assertion message; degenerate bootstrap
  replicates dropped+counted; strata same_partition flag.
- make_sentence_tables.py re-run; ANALYSIS.md updated (A0, A5, caveats); number audit: all decimals traceable.
- Next: when labelling ends (~04:15): order_check (--fp32-check) GPU run; when gpu_sentlab.done: re-run tables
  (incl. --sentlab), update ANALYSIS §6/§9, write analysis.done. If A1 is late (> ~05:40), write analysis.done
  with A1 pending.

## 04:20, A1 labelling complete
- 34,242/34,242 passages labelled by 04:08 (YES rate 0.6545, all answers well formed); split 30,818 / 3,424
  (sample_only false -> no matched pair arm). sentlab_s1 done 04:15 (dev AUPRC 0.9854); s2, s3 training.
- sentlab_label_stats.py added (teacher sentence answers vs candidate labels) -> SL/sentlab_label_stats.json.
- reframe-biodiv-llm-enum started 04:08 (6,253 pairs; slow under contention).
- biored_sentence.py --fp32 timed out at 590 s under contention; 17/18 fp32 score caches written; re-run after training.
- Next: wait for gpu_sentlab.done; make_sentence_tables.py; finish --fp32; integrate A1 in ANALYSIS §6/§9; analysis.done.

## 04:35, stage complete
- A1 done 04:26 (gpu_sentlab.done): sentlab beats pair-max on biodiversity under all 3 silver definitions; advantage
  lies entirely in pair-negative passages (LLM-decided labels; circular); on gold-decided positives sentlab ~ pair arms.
  Decomposition added to biodiv_sentence.py (`decomposition`). BioRED FP32 check: bit-identical with float32 ensemble.
- ANALYSIS.md rewritten (§6 A1, §8 caveats, §9 conclusion); number audit clean; sentence_tables.json regenerated.
- analysis.done written. Still running (detached, independent): reframe-biodiv-llm-enum (qwen3:32b over 6,253 pairs,
  only for a future human-label LLM pair-max row).
