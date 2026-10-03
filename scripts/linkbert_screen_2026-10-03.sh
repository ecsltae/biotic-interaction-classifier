#!/usr/bin/env bash
# Session-2 encoder screen, 2026-10-03: BioLinkBERT-base under the base recipe (lr 2e-5, 3 epochs),
# pair arm, seed 1, development data only (BioRED: the BioRED test split; biodiversity: the
# internal pair-grouped split), exactly like the session-1 recipe sweep. Waits for the 122B
# zero-shot run to free the GPU.
# Pre-set bar for scoring any test set: BioRED pair dev AUPRC >= 0.905 (the BiomedBERT-large
# three-seed mean, which did not transfer to test) AND biodiversity pair dev >= 0.848 (base
# seed 1 0.838 + 0.01). Below the bar it is a seventh recipe in the negative-results count.
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"
R=results/overnight_2026-10-03; mkdir -p $R/state $R/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/queue.log; }
TS=experiments/multitask/train_student.py
ENC=michiyasunaga/BioLinkBERT-base
B=data/benchmarks/biored_bc8; CORPUS=data/training/distill/v3_combined_train.csv
train() {  # train <out> <args...>
  local out=$1; shift
  [ -f $out/student_config.json ] && { log "skip $out"; return 0; }
  log "train $out $*"
  python3 $TS "$@" --out $out > $R/logs/$(echo $out | tr / _).log 2>&1 \
    && log "  $(tail -1 $R/logs/$(echo $out | tr / _).log)" || log "  FAILED $out (see logs)"
}
while systemctl --user is-active --quiet paperA-122b-s2b; do sleep 30; done
log "linkbert screen start"
train models/recipe_sweep/biored_pair_linkbert_s1 --data $B/train.csv --dev-data $B/dev.csv --input-format pair --encoder $ENC --lr 2e-5 --epochs 3 --micro-batch 0 --seed 1
train models/recipe_sweep/biodiv_pair_linkbert_s1 --data $CORPUS --input-format pair --encoder $ENC --lr 2e-5 --epochs 3 --micro-batch 0 --seed 1
touch $R/state/linkbert_screen.done
log "linkbert screen done"
