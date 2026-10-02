#!/usr/bin/env bash
# Overnight GPU queue, 2026-10-03. Local models only (no API key). Two queues run in parallel and
# coordinate through marker files in $R/state:
#
#   T (training)  T1 recipe sweep, seed 1, BioRED pair at sentence scope + biodiversity pair
#                 T2 whole-abstract BioRED, 8 epochs, untyped and typed query, seed 1
#                 T3 [after relabel.done] soft-label students, sentence/pair/triple x 3 seeds
#                 -> train.done
#   L (Ollama)    L1 pull Qwen3 0.6b..30b
#                 L2 zero-shot scaling: biodiversity 437 (3 questions) + BioRED 3,000 (2)
#                 L3 teacher P(yes) on the 48,338 training rows (qwen3:32b) -> relabel.done
#                 L4 [after train.done: the 122B needs the whole GPU] qwen3.5:122b, biodiversity
#                    437 + BioRED 1,000 (a prefix of the 3,000 sample)  -> llm.done
#
# Every step is skipped if its output exists, so the script can be re-run after a failure.
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"
R=results/overnight_2026-10-03; mkdir -p $R/state $R/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/queue.log; }
TS=experiments/multitask/train_student.py
BASE=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
LARGE=microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract
B=data/benchmarks/biored_bc8; BD=data/benchmarks/biored_bc8_doc; BDT=data/benchmarks/biored_bc8_doc_typed
CORPUS=data/training/distill/v3_combined_train.csv

train() {  # train <out> <args...>
  local out=$1; shift
  [ -f $out/student_config.json ] && { log "skip $out"; return 0; }
  log "train $out $*"
  python3 $TS "$@" --out $out > $R/logs/$(echo $out | tr / _).log 2>&1 \
    && log "  $(tail -1 $R/logs/$(echo $out | tr / _).log)" || log "  FAILED $out (see logs)"
}

queue_T() {
  # T1: recipe sweep, seed 1 (selection happens on development data only)
  for cfg in "ep5_lr2e5 $BASE 2e-5 5 0" "ep5_lr3e5 $BASE 3e-5 5 0" "ep5_lr1e5 $BASE 1e-5 5 0" \
             "large_lr1e5 $LARGE 1e-5 3 16" "large_lr2e5 $LARGE 2e-5 3 16"; do
    set -- $cfg; name=$1; enc=$2; lr=$3; ep=$4; mb=$5
    train models/recipe_sweep/biored_pair_${name}_s1 --data $B/train.csv --dev-data $B/dev.csv \
      --input-format pair --encoder $enc --lr $lr --epochs $ep --micro-batch $mb --seed 1
    train models/recipe_sweep/biodiv_pair_${name}_s1 --data $CORPUS \
      --input-format pair --encoder $enc --lr $lr --epochs $ep --micro-batch $mb --seed 1
  done
  # T2: whole-abstract BioRED (does not bear on the sentence case; lowest priority)
  [ -f $BDT/train.csv ] || python3 scripts/convert_biored_doc.py --typed --src $HOME/biored_baseline/bioredirect \
      --out $BDT > $R/logs/convert_doc_typed.log 2>&1
  train models/biored_bc8_doc/pair_ep8_s1 --data $BD/train.csv --dev-data $BD/dev.csv \
    --input-format pair --max-len 512 --epochs 8 --seed 1
  train models/biored_bc8_doc/typed_ep8_s1 --data $BDT/train.csv --dev-data $BDT/dev.csv \
    --input-format pair --max-len 512 --epochs 8 --seed 1
  # T3: soft-label students, once the teacher's P(yes) exists
  log "T waiting for relabel.done"
  until [ -f $R/state/relabel.done ] || [ -f $R/state/relabel.failed ]; do sleep 60; done
  [ -f $R/state/relabel.done ] && for fmt in sentence pair triple; do for s in 1 2 3; do
    train models/soft_students/${fmt}_s${s} --data data/training/distill/v3_combined_train_soft.csv \
      --soft-col soft_label --input-format $fmt --seed $s
  done; done
  touch $R/state/train.done; log "T done"
}

queue_L() {
  for t in 0.6b 1.7b 4b 8b 14b 30b; do
    ollama list | grep -q "^qwen3:$t " || { log "pull qwen3:$t"; ollama pull qwen3:$t > $R/logs/pull_$t.log 2>&1 || log "  pull FAILED $t"; }
  done
  for m in qwen3:0.6b qwen3:1.7b qwen3:4b qwen3:8b qwen3:14b qwen3:30b; do
    log "zero-shot $m"
    python3 scripts/llm_baseline.py --bench biodiv --model $m >> $R/logs/llm_scaling.log 2>&1 || log "  FAILED biodiv $m"
    python3 scripts/llm_baseline.py --bench biored --model $m --n 3000 >> $R/logs/llm_scaling.log 2>&1 || log "  FAILED biored $m"
  done
  log "teacher P(yes) relabel (qwen3:32b, 48,338 rows)"
  python3 scripts/soft_relabel.py > $R/logs/soft_relabel.log 2>&1 \
    && { touch $R/state/relabel.done; log "relabel done"; } \
    || { touch $R/state/relabel.failed; log "  relabel FAILED"; }
  log "L waiting for train.done before the 122B"
  until [ -f $R/state/train.done ]; do sleep 60; done
  log "zero-shot qwen3.5:122b"
  python3 scripts/llm_baseline.py --bench biodiv --model qwen3.5:122b --workers 1 >> $R/logs/llm_122b.log 2>&1 || log "  FAILED biodiv 122b"
  python3 scripts/llm_baseline.py --bench biored --model qwen3.5:122b --n 1000 --workers 1 >> $R/logs/llm_122b.log 2>&1 || log "  FAILED biored 122b"
  touch $R/state/llm.done; log "L done"
}

log "queue start"
queue_T & PT=$!
queue_L & PL=$!
wait $PT $PL
log "queue finished"
bash scripts/notify.sh "Overnight GPU queue finished" "$(tail -40 $R/queue.log)" || true
