#!/usr/bin/env bash
# BiomedBERT-large (lr 2e-5, 3 epochs, micro-batch 16) arms, selected on development data by the
# overnight recipe sweep (BioRED pair dev AUPRC 0.907 vs 0.888 base; biodiversity pair dev 0.866 vs
# 0.838). Seed 1 of the pair arms comes from models/recipe_sweep/*_large_lr2e5_s1 (symlinked).
# Runs one training at a time; every step is skipped if its output exists.
set -u
cd "$(dirname "$0")/.."
source ../MPvenv/bin/activate
R=results/overnight_2026-10-03; mkdir -p $R/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/large_chain.log; }
LARGE=microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract
B=data/benchmarks/biored_bc8; CORPUS=data/training/distill/v3_combined_train.csv
COMMON="--encoder $LARGE --lr 2e-5 --epochs 3 --micro-batch 16"
train() {  # train <out> <args...>
  local out=$1; shift
  [ -f $out/student_config.json ] && { log "skip $out"; return 0; }
  log "train $out $*"
  python3 experiments/multitask/train_student.py "$@" --out $out > $R/logs/$(echo $out | tr / _).log 2>&1 \
    && log "  $(tail -1 $R/logs/$(echo $out | tr / _).log)" || log "  FAILED $out"
}
log "chain start"
for s in 2 3; do train models/large/biored_pair_s$s --data $B/train.csv --dev-data $B/dev.csv --input-format pair $COMMON --seed $s; done
for s in 1 2 3; do train models/large/biored_sentence_s$s --data $B/train.csv --dev-data $B/dev.csv --input-format sentence $COMMON --seed $s; done
touch $R/state/large_biored.done
[ -f $R/state/large_stop ] && { log "stop marker: skipping biodiversity arms"; exit 0; }
for s in 1 2 3; do
  train models/large/biodiv_sentence_s$s --data $CORPUS --input-format sentence $COMMON --seed $s
  [ $s -gt 1 ] && train models/large/biodiv_pair_s$s --data $CORPUS --input-format pair $COMMON --seed $s
  [ -f $R/state/large_stop ] && { log "stop marker"; exit 0; }
done
for s in 1 2 3; do
  train models/large/biodiv_triple_s$s --data $CORPUS --input-format triple $COMMON --seed $s
  [ -f $R/state/large_stop ] && { log "stop marker"; exit 0; }
done
touch $R/state/large_biodiv.done
log "chain done"
