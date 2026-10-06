#!/usr/bin/env bash
# A1 of the 2026-10-05 sentence-level reframe: the fair biodiversity sentence-level competitor.
#   1. the teacher (qwen3:32b, system Ollama :11434) answers the SENTENCE question for every unique
#      corpus passage, in a fixed random order (scripts/sentence_level/sentlab_label.py), appending;
#      a safety stop (STOP_AT, default launch + 3 h) leaves a random sample if it runs long
#   2. one 10% development split by passage (seed 0), built once (sentlab_split.py); if only a sample
#      was labelled, also the matched pair-arm files
#   3. students, the paper's recipe (BiomedBERT-base, 3 epochs, seeds 1-3): sentlab (input format
#      sentence) and, after a partial labelling only, the matched pair arm; finished models are skipped
#   4. sentence-level scoring of every arm on the 437 passages (biodiv_sentence.py --sentlab)
# Writes results/reframe_2026-10-05/state/gpu_sentlab.done on success, gpu_sentlab.failed (reason, log)
# otherwise. Re-running resumes. Everything ends by 11:15 (timeout on each step).
set -u
cd $HOME/MetaP/classifier
source $HOME/MetaP/MPvenv/bin/activate
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
S=results/reframe_2026-10-05/state; LD=results/reframe_2026-10-05/logs; LOG=$LD/gpu_sentlab.log
D=data/training/distill; LAB=$D/v3_sentence_teacher_qwen3-32b.csv
OUTJ=results/paperA_v2/sentence_level/biodiv_sentlab.json
END=$(date -d "2026-10-05 11:15" +%s)
STOP_AT=${STOP_AT:-$(date -d '+3 hours' '+%F %H:%M')}
T0=$(date +%s)
log() { echo "[$(date '+%F %T')] $*" >> $LOG; }
fail() {
  log "FAILED: $*"
  printf 'A1 (gpu_sentlab) failed at %s\nreason: %s\nlog: %s (step logs: %s/gpu_sentlab_*.log)\n' \
    "$(date '+%F %T')" "$*" "$PWD/$LOG" "$PWD/$LD" > $S/gpu_sentlab.failed
  exit 1
}
run() { local t=$(( END - $(date +%s) )); [ $t -gt 120 ] || fail "no time left before 11:15 for: $*"; timeout $t "$@"; }
if [ -f $S/gpu_sentlab.failed ]; then mv $S/gpu_sentlab.failed $S/gpu_sentlab.failed.prev.$(date +%H%M); fi
log "start (pid $$), labelling stops at $STOP_AT at the latest"

# 1. teacher sentence labels (skipped once the split exists: the students train on that split, so
#    labelling more passages after it was built would only make the bookkeeping disagree with it)
if [ -f $D/v3_sentence_teacher_qwen3-32b_train.csv ] && [ -f $D/v3_sentence_teacher_qwen3-32b_dev.csv ]; then
  log "split exists: labelling skipped"
else
  run python3 scripts/sentence_level/sentlab_label.py --workers 4 --stop-at "$STOP_AT" >> $LD/gpu_sentlab_label.log 2>&1 \
    || fail "labelling exited with code $?"
fi
read n ok yes <<< $(python3 -c "
import pandas as pd; d = pd.read_csv('$LAB'); a = d.answer.astype(str).str.strip().str.upper()
print(d.text.nunique(), round(float(a.str.startswith(('YES', 'NO')).mean()), 4), round(float(d.label.mean()), 4))")
log "labelled passages: $n (well-formed answers $ok, YES rate $yes)"
[ "${n:-0}" -ge 8000 ] || fail "only ${n:-0} passages labelled (< 8000)"
python3 -c "import sys; sys.exit(0 if float('$ok') >= 0.99 else 1)" || fail "only $ok of the answers start with YES/NO"

# 2. split
run python3 scripts/sentence_level/sentlab_split.py >> $LOG 2>&1 || fail "split exited with code $?"
sample=$(python3 -c "import json; print(int(json.load(open('results/paperA_v2/sentence_level/sentlab_split.json'))['sample_only']))")
n=$(python3 -c "import json; print(json.load(open('results/paperA_v2/sentence_level/sentlab_split.json'))['labelled_passages'])")   # passages in the split

# 3. students
train() {   # name data dev format seed
  local out=models/sentence_level/biodiv_$1_s$5
  if [ -f $out/student_config.json ]; then log "skip $out (finished)"; return 0; fi
  log "train $out"
  run python3 experiments/multitask/train_student.py --data $2 --dev-data $3 --input-format $4 --seed $5 \
      --epochs 3 --out $out > $LD/gpu_sentlab_train_$1_s$5.log 2>&1 || fail "training $out exited with code $?"
  [ -f $out/student_config.json ] || fail "training $out wrote no student_config.json"
  log "done $out: $(python3 -c "import json; c = json.load(open('$out/student_config.json')); print('dev AUPRC', round(c['best_dev_auprc'], 4), 'thr', c['threshold_dev'])")"
}
for k in 1 2 3; do
  train sentlab $D/v3_sentence_teacher_qwen3-32b_train.csv $D/v3_sentence_teacher_qwen3-32b_dev.csv sentence $k
done
if [ "$sample" = 1 ]; then
  for k in 1 2 3; do train pair_matched $D/v3_pair_matched_train.csv $D/v3_pair_matched_dev.csv pair $k; done
fi

# 4. scoring on the 437 passages
[ -f $OUTJ ] && cp $OUTJ $OUTJ.bak.$(date +%H%M)
run python3 scripts/sentence_level/biodiv_sentence.py --sentlab >> $LD/gpu_sentlab_score.log 2>&1 \
  || fail "scoring exited with code $? (see $LD/gpu_sentlab_score.log)"
python3 - <<PY || fail "scoring output $OUTJ missing, stale or incomplete"
import json, os, sys
j = json.load(open("$OUTJ"))
sys.exit(0 if os.path.getmtime("$OUTJ") > $T0 and "sentlab" in j["arms"] else 1)
PY
log "finished; results in $OUTJ"
{ echo "A1 (gpu_sentlab) finished at $(date '+%F %T')"
  echo "labelled passages: $n of 34242 (YES rate $yes), sample only: $sample"
  echo "labels: $LAB; split: $D/v3_sentence_teacher_qwen3-32b_{train,dev}.csv"
  echo "models: models/sentence_level/biodiv_sentlab_s{1,2,3}$([ "$sample" = 1 ] && echo ', models/sentence_level/biodiv_pair_matched_s{1,2,3}')"
  echo "results: $OUTJ"
  python3 -c "
import json; j = json.load(open('$OUTJ'))
for lab, r in j['by_silver'].items():
    print(lab + ': ' + '; '.join(f\"{a} AUPRC {v['auprc']:.3f} F1 {v['F1']:.3f}\" for a, v in r['arms'].items()))"
} > $S/gpu_sentlab.done
