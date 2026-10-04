#!/usr/bin/env bash
# Queued after the qwen3.8 triple relabel: species-level labels from qwen3.8:27b, then the deployed
# verifier retrained on them with the shipped recipe, then scored exactly as the handoff scores it.
#   1. wait for data/training/distill/v3_combined_train_qwen38.csv (written by teacher_qwen38_2026-10-04.sh)
#   2. species relabel of its NO rows -> data/training/distill/v4_species_train_qwen38.csv (new version)
#   3. train_direction.py with the joint_a05_s1 recipe (mark_canon, alpha 0.5, 2 epochs, seed 1,
#      no dev split, same direction data) -> models/dirhead/joint_qwen38_s1
#   4. eval_shipping.py -> results/shipping_2026-10-02/eval_qwen38.json, next to eval_a05.json
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"
export OLLAMA_URL=http://127.0.0.1:11435
R=results/species_qwen38_2026-10-04; mkdir -p $R
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/run.log; }
TRIPLE=data/training/distill/v3_combined_train_qwen38.csv
SPECIES=data/training/distill/v4_species_train_qwen38.csv
log "waiting for $TRIPLE"
until [ -f $TRIPLE ]; do sleep 120; done
sleep 30
log "species relabel with qwen3.8:27b"
python3 scripts/species_relabel_llm.py --model qwen3.8:27b --triple $TRIPLE --out $SPECIES > $R/relabel.log 2>&1 \
  || { log "relabel FAILED"; bash scripts/notify.sh "qwen3.8 species relabel FAILED" "$(tail -20 $R/relabel.log)"; exit 1; }
log "$(tail -1 $R/relabel.log)"
log "train the joint verifier on the qwen3.8 species labels"
python3 experiments/multitask/train_direction.py --bin-data $SPECIES --out models/dirhead/joint_qwen38_s1 \
  --alpha 0.5 --epochs 2 --seed 1 --val-frac 0 > $R/train.log 2>&1 || { log "train FAILED"; exit 1; }
log "score it like the handoff does"
python3 scripts/eval_shipping.py --model models/dirhead/joint_qwen38_s1 --name qwen38 > $R/eval.log 2>&1 || log "eval FAILED"
python3 - >> $R/run.log 2>&1 <<'PY'
import json
for n in ("a05", "qwen38"):
    o = json.load(open(f"results/shipping_2026-10-02/eval_{n}.json"))
    d = o["direction"]["pred"]
    print(f"{n:7} AUPRC {o['auprc']:.4f} | no rules P {o['no_rules']['P']:.3f} R {o['no_rules']['R']:.3f} F1 {o['no_rules']['F1']:.3f}"
          f" | rules P {o['with_rules']['P']:.3f} R {o['with_rules']['R']:.3f} F1 {o['with_rules']['F1']:.3f}"
          f" | direction coverage {d['coverage']:.2f} accuracy {d['accuracy_on_answered']:.3f}")
PY
log "done"
bash scripts/notify.sh "qwen3.8 species labels and retrained verifier" "$(tail -12 $R/run.log)" || true
