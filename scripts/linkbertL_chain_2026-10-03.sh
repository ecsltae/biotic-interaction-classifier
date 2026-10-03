#!/usr/bin/env bash
# BioLinkBERT-large arms (lr 2e-5, 3 epochs, micro-batch 16), trained only if the screen
# (scripts/linkbertL_screen_2026-10-03.sh) is ahead of the base recipe by >= 0.01 dev AUPRC on both
# splits (BioRED pair >= 0.898, biodiversity pair >= 0.848). The pair arms come first on both
# benchmarks (the adoption decision rests on them), then the sentence and triple arms.
# Seed 1 of the pair arms comes from the screen (symlinked). Stops at $R/state/linkbertL_stop.
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"
R=results/overnight_2026-10-03; mkdir -p $R/state $R/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/queue.log; }
ENC=michiyasunaga/BioLinkBERT-large
B=data/benchmarks/biored_bc8; CORPUS=data/training/distill/v3_combined_train.csv
COMMON="--encoder $ENC --lr 2e-5 --epochs 3 --micro-batch 16"
train() {  # train <out> <args...>
  local out=$1; shift
  [ -f $R/state/linkbertL_stop ] && { log "stop marker: $out not trained"; exit 0; }
  [ -f $out/student_config.json ] && { log "skip $out"; return 0; }
  log "train $out $*"
  python3 experiments/multitask/train_student.py "$@" --out $out > $R/logs/$(echo $out | tr / _).log 2>&1 \
    && log "  $(tail -1 $R/logs/$(echo $out | tr / _).log)" || log "  FAILED $out"
}
until [ -f $R/state/linkbertL_screen.done ]; do sleep 20; done
dev() { python3 -c "import json; print(json.load(open('$1/student_config.json'))['best_dev_auprc'])"; }
a=$(dev models/recipe_sweep/biored_pair_linkbertL_s1); b=$(dev models/recipe_sweep/biodiv_pair_linkbertL_s1)
if ! python3 -c "import sys; sys.exit(not ($a >= 0.8977 and $b >= 0.8483))"; then
  log "linkbert-large chain: bar not cleared (BioRED pair dev $a, biodiversity pair dev $b); nothing trained"; exit 0
fi
log "linkbert-large chain start (BioRED pair dev $a, biodiversity pair dev $b)"
mkdir -p models/linkbertL
ln -sfn ../recipe_sweep/biored_pair_linkbertL_s1 models/linkbertL/biored_pair_s1
ln -sfn ../recipe_sweep/biodiv_pair_linkbertL_s1 models/linkbertL/biodiv_pair_s1
for s in 2 3; do train models/linkbertL/biored_pair_s$s --data $B/train.csv --dev-data $B/dev.csv --input-format pair $COMMON --seed $s; done
for s in 2 3; do train models/linkbertL/biodiv_pair_s$s --data $CORPUS --input-format pair $COMMON --seed $s; done
touch $R/state/linkbertL_pair.done
for s in 1 2 3; do train models/linkbertL/biored_sentence_s$s --data $B/train.csv --dev-data $B/dev.csv --input-format sentence $COMMON --seed $s; done
for s in 1 2 3; do train models/linkbertL/biodiv_sentence_s$s --data $CORPUS --input-format sentence $COMMON --seed $s; done
for s in 1 2 3; do train models/linkbertL/biodiv_triple_s$s --data $CORPUS --input-format triple $COMMON --seed $s; done
touch $R/state/linkbertL_all.done
log "linkbert-large chain done"
