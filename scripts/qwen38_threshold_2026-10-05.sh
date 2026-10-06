#!/usr/bin/env bash
# A default threshold for the qwen3.8-taught verifier, chosen on held-out data, never on the 437 rows.
# Three retrains of the deployed recipe with a 10% pair-grouped development split (the protocol that
# validated joint_a05_s1's 0.5): each records its F1-maximising development threshold. The 437 rows
# are then scored once per model at that threshold, and the full-data model joint_qwen38_s1 at the
# median of the three.
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"
R=results/qwen38_threshold_2026-10-05; mkdir -p $R
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/run.log; }
SPECIES=data/training/distill/v4_species_train_qwen38.csv
for s in 1 2 3; do
  out=models/dirhead/joint_qwen38_dev_s$s
  [ -f $out/student_config.json ] || python3 experiments/multitask/train_direction.py --bin-data $SPECIES --out $out \
      --alpha 0.5 --epochs 2 --seed $s --val-frac 0.1 > $R/train_s$s.log 2>&1 || { log "train FAILED s$s"; continue; }
  thr=$(python3 -c "import json; print(json.load(open('$out/student_config.json'))['threshold_dev'])")
  log "seed $s: development threshold $thr"
  python3 scripts/eval_shipping.py --model $out --name qwen38_dev_s$s --threshold $thr > $R/eval_s$s.log 2>&1 || log "eval FAILED s$s"
done
med=$(python3 -c "
import json, statistics
print(statistics.median(json.load(open(f'models/dirhead/joint_qwen38_dev_s{s}/student_config.json'))['threshold_dev'] for s in (1, 2, 3)))")
log "median development threshold $med -> full-data model joint_qwen38_s1"
python3 scripts/eval_shipping.py --model models/dirhead/joint_qwen38_s1 --name qwen38_at_dev --threshold $med > $R/eval_full.log 2>&1 || log "eval FAILED full"
python3 - >> $R/run.log 2>&1 <<'PY'
import json
for n in ("a05", "qwen38_dev_s1", "qwen38_dev_s2", "qwen38_dev_s3", "qwen38_at_dev"):
    o = json.load(open(f"results/shipping_2026-10-02/eval_{n}.json")); d = o["direction"]["pred"]
    print(f"{n:15} thr {o['threshold']:.2f} AUPRC {o['auprc']:.4f} | rules P {o['with_rules']['P']:.3f} "
          f"R {o['with_rules']['R']:.3f} F1 {o['with_rules']['F1']:.3f} FP {o['with_rules']['fp']} | "
          f"direction coverage {d['coverage']:.2f} accuracy {d['accuracy_on_answered']:.3f}")
PY
log "done"
