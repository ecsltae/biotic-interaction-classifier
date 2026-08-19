#!/usr/bin/env bash
# Option D: Retrain BiomedBERT teacher on v14 → rescore 44k → train multitask.
# Run AFTER Phase 1 experiments are complete (GPU must be free).
#
# Usage:
#   source MPvenv/bin/activate && bash classifier/scripts/run_option_d.sh 2>&1 | tee /tmp/option_d.log

set -e
cd /home/egaillac/MetaP

echo "========================================"
echo "Option D1: Retrain BiomedBERT on v14"
echo "========================================"
python classifier/scripts/train_cv_regularized.py \
  --train-data classifier/data/training/training_data_v14.csv \
  --models BiomedBERT \
  --suffix v14

echo "========================================"
echo "Option D2: Score distillation_44k with v14 teacher"
echo "========================================"
python classifier/scripts/score_all.py \
  --input  classifier/data/training/distillation_44k.csv \
  --output classifier/data/training/distillation_v14teacher.csv \
  --bert-model classifier/models/transformer_BiomedBERT_v14 \
  --gpu

echo "========================================"
echo "Option D3: Train multitask on v14-teacher soft labels"
echo "========================================"
python classifier/experiments/multitask/train.py \
  --data  classifier/data/training/distillation_v14teacher.csv \
  --encoder microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --ner-scheme full_typed --alpha 0.5 --pretrain-ner-epochs 2 --epochs 3 \
  --batch-size 16 --temperature 2.0 \
  --output-dir  classifier/models/multitask/mt_v14teacher \
  --results-dir classifier/results/multitask/mt_v14teacher

echo ""
echo "Option D complete! Evaluate with:"
echo "  python classifier/scripts/eval_new_experiments.py --gpu"
