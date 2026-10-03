#!/usr/bin/env bash
# BioLinkBERT-base arms (base recipe: lr 2e-5, 3 epochs), trained only if the session-2 screen
# (scripts/linkbert_screen_2026-10-03.sh) clears both pre-set development bars: BioRED pair dev
# AUPRC >= 0.905 and biodiversity pair dev >= 0.848. Seed 1 of the pair arms comes from the screen
# (models/recipe_sweep/*_pair_linkbert_s1, symlinked). BioRED arms first (-> linkbert_biored.done),
# then the biodiversity arms unless $R/state/linkbert_stop exists. One training at a time.
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"
R=results/overnight_2026-10-03; mkdir -p $R/state $R/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/queue.log; }
ENC=michiyasunaga/BioLinkBERT-base
B=data/benchmarks/biored_bc8; CORPUS=data/training/distill/v3_combined_train.csv
COMMON="--encoder $ENC --lr 2e-5 --epochs 3 --micro-batch 0"
train() {  # train <out> <args...>
  local out=$1; shift
  [ -f $out/student_config.json ] && { log "skip $out"; return 0; }
  log "train $out $*"
  python3 experiments/multitask/train_student.py "$@" --out $out > $R/logs/$(echo $out | tr / _).log 2>&1 \
    && log "  $(tail -1 $R/logs/$(echo $out | tr / _).log)" || log "  FAILED $out"
}
until [ -f $R/state/linkbert_screen.done ]; do sleep 20; done
dev() { python3 -c "import json; print(json.load(open('$1/student_config.json'))['best_dev_auprc'])"; }
a=$(dev models/recipe_sweep/biored_pair_linkbert_s1); b=$(dev models/recipe_sweep/biodiv_pair_linkbert_s1)
if ! python3 -c "import sys; sys.exit(not ($a >= 0.905 and $b >= 0.848))"; then
  log "linkbert chain: bar not cleared (BioRED pair dev $a, biodiversity pair dev $b); nothing trained"; exit 0
fi
log "linkbert chain start (BioRED pair dev $a, biodiversity pair dev $b)"
mkdir -p models/linkbert
ln -sfn ../recipe_sweep/biored_pair_linkbert_s1 models/linkbert/biored_pair_s1
ln -sfn ../recipe_sweep/biodiv_pair_linkbert_s1 models/linkbert/biodiv_pair_s1
for s in 2 3; do train models/linkbert/biored_pair_s$s --data $B/train.csv --dev-data $B/dev.csv --input-format pair $COMMON --seed $s; done
for s in 1 2 3; do train models/linkbert/biored_sentence_s$s --data $B/train.csv --dev-data $B/dev.csv --input-format sentence $COMMON --seed $s; done
touch $R/state/linkbert_biored.done
for s in 1 2 3; do
  [ -f $R/state/linkbert_stop ] && { log "stop marker: biodiversity arms not trained further"; exit 0; }
  train models/linkbert/biodiv_sentence_s$s --data $CORPUS --input-format sentence $COMMON --seed $s
  [ $s -gt 1 ] && train models/linkbert/biodiv_pair_s$s --data $CORPUS --input-format pair $COMMON --seed $s
  train models/linkbert/biodiv_triple_s$s --data $CORPUS --input-format triple $COMMON --seed $s
done
touch $R/state/linkbert_biodiv.done
log "linkbert chain done"
