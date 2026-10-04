#!/usr/bin/env bash
# The teacher becomes the latest Qwen (user's decision, 2026-10-04): qwen3.8:27b, served by the
# user-level Ollama v0.35.1 (127.0.0.1:11435, unit `ollama-latest`, 4,096-token context). The
# paper's zero-shot comparison stays on Qwen3 (0.6B-32B) and Qwen3.5-122B and is not touched.
#   1. qwen3.8:27b and qwen3:32b scored on the same server (BioRED dev 1,000 pair; biodiversity
#      437, three questions), files suffixed -v035: a record of how the two teachers compare
#   2. the 48,338 training rows relabelled with qwen3.8:27b, same prompt and inputs
#      -> data/training/distill/v3_combined_train_qwen38_soft.csv (P(yes)) and
#         data/training/distill/v3_combined_train_qwen38.csv (its verdicts as labels)
#   3. sentence / pair / triple x 3 seeds on its verdicts, the paper's recipe otherwise
#      -> models/teacher_qwen38/
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"
export OLLAMA_URL=http://127.0.0.1:11435
R=results/teacher_qwen38_2026-10-04; mkdir -p $R/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/run.log; }
log "start"
for m in qwen3.8:27b qwen3:32b; do
  python3 scripts/llm_baseline.py --bench bioreddev --model $m --forms pair --n 1000 --tag=-v035 --workers 4 >> $R/logs/llm.log 2>&1 || log "FAILED bioreddev $m"
  python3 scripts/llm_baseline.py --bench biodiv --model $m --tag=-v035 --workers 4 >> $R/logs/llm.log 2>&1 || log "FAILED biodiv $m"
done
log "scores: $(grep -E 'AUPRC' $R/logs/llm.log | tr '\n' ';')"
C=data/training/distill/v3_combined_train_qwen38
log "relabel with qwen3.8:27b"
python3 scripts/soft_relabel.py --model qwen3.8:27b --workers 4 --out ${C}_softlabels.csv > $R/logs/relabel.log 2>&1 \
  || { log "relabel FAILED"; bash scripts/notify.sh "qwen3.8 relabel FAILED" "$(tail -20 $R/logs/relabel.log)"; exit 1; }
log "$(tail -2 $R/logs/relabel.log | head -1)"
python3 - "$C" <<'PY'
import sys, pandas as pd
c = sys.argv[1]; d = pd.read_csv(c + "_soft.csv")
d.assign(label=(d.soft_label >= 0.5).astype(int)).to_csv(c + ".csv", index=False)
print(f"{c}.csv: {len(d)} rows, positive rate {(d.soft_label >= 0.5).mean():.3f}")
PY
for fmt in sentence pair triple; do for s in 1 2 3; do
  out=models/teacher_qwen38/${fmt}_s${s}
  [ -f $out/student_config.json ] && continue
  log "train $out"
  python3 experiments/multitask/train_student.py --data ${C}.csv --input-format $fmt --seed $s --out $out \
    > $R/logs/train_${fmt}_s${s}.log 2>&1 && log "  $(tail -1 $R/logs/train_${fmt}_s${s}.log)" || log "  FAILED $out"
done; done
log "done"
bash scripts/notify.sh "qwen3.8 teacher: relabel and students done" "$(tail -30 $R/run.log)" || true
