#!/usr/bin/env bash
# Run all 6 multi-task training experiments sequentially on the A100.
# Launch from /home/egaillac/MetaP/ with the MPvenv activated.
#
# Usage:
#   source MPvenv/bin/activate && bash classifier/scripts/run_experiments.sh 2>&1 | tee /tmp/experiments.log

set -e
cd /home/egaillac/MetaP

TRAIN="python classifier/experiments/multitask/train.py"
COMMON="--ner-scheme full_typed --alpha 0.5 --batch-size 16 --epochs 3 --temperature 2.0"
PRETRAIN2="--pretrain-ner-epochs 2"
PRETRAIN1="--pretrain-ner-epochs 1"
PRETRAIN0="--pretrain-ner-epochs 0"
COLD="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
WARM="classifier/models/transformer_BiomedBERT_cv_regularized"

DATA_V7="classifier/data/training/v7_softlabels.csv"
DATA_44K="classifier/data/training/distillation_44k.csv"
DATA_MIX="classifier/data/training/v7_distill_mix.csv"

echo "========================================"
echo "Exp 1: mt_v7_softlabels (v7 soft, cold, NER=2)"
echo "========================================"
$TRAIN \
  --data $DATA_V7 --encoder "$COLD" \
  $COMMON $PRETRAIN2 \
  --output-dir classifier/models/multitask/mt_v7_softlabels \
  --results-dir classifier/results/multitask/mt_v7_softlabels

echo "========================================"
echo "Exp 2: mt_v7_softlabels_warm (v7 soft, warm, NER=2)"
echo "========================================"
$TRAIN \
  --data $DATA_V7 --encoder "$WARM" \
  $COMMON $PRETRAIN2 \
  --output-dir classifier/models/multitask/mt_v7_softlabels_warm \
  --results-dir classifier/results/multitask/mt_v7_softlabels_warm

echo "========================================"
echo "Exp 3: mt_distill_warm_ner1 (44k soft, warm, NER=1)"
echo "========================================"
$TRAIN \
  --data $DATA_44K --encoder "$WARM" \
  $COMMON $PRETRAIN1 \
  --output-dir classifier/models/multitask/mt_distill_warm_ner1 \
  --results-dir classifier/results/multitask/mt_distill_warm_ner1

echo "========================================"
echo "Exp 4: mt_distill_warm_ner0 (44k soft, warm, NER=0)"
echo "========================================"
$TRAIN \
  --data $DATA_44K --encoder "$WARM" \
  $COMMON $PRETRAIN0 \
  --output-dir classifier/models/multitask/mt_distill_warm_ner0 \
  --results-dir classifier/results/multitask/mt_distill_warm_ner0

echo "========================================"
echo "Exp 5: mt_v7_distill_mix (mixed, cold, NER=2)"
echo "========================================"
$TRAIN \
  --data $DATA_MIX --encoder "$COLD" \
  $COMMON $PRETRAIN2 \
  --output-dir classifier/models/multitask/mt_v7_distill_mix \
  --results-dir classifier/results/multitask/mt_v7_distill_mix

echo "========================================"
echo "Exp 6: mt_v7_distill_mix_warm (mixed, warm, NER=1)"
echo "========================================"
$TRAIN \
  --data $DATA_MIX --encoder "$WARM" \
  $COMMON $PRETRAIN1 \
  --output-dir classifier/models/multitask/mt_v7_distill_mix_warm \
  --results-dir classifier/results/multitask/mt_v7_distill_mix_warm

echo ""
echo "All 6 experiments complete!"
echo "Results in classifier/results/multitask/mt_*/"
