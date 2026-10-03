#!/usr/bin/env bash
# Search for a better local teacher, then relabel the corpus with it and retrain the three arms.
#
# v2: every candidate and the reference (qwen3:32b) run on one Ollama version, v0.35.1, served at
# 127.0.0.1:11435 from the user-level unit `ollama-latest` (the system Ollama, 0.18.3, cannot pull
# qwen3.6 / qwen3.8 / gemma4). Their output files carry the suffix -v035. The 122B stays on the
# system server, as a reference only.
#
# Candidates: open models in the Ollama library (2026-10-04) that run locally and fit entirely on
# the 80 GB GPU. Each is scored zero-shot, with the paper's prompts, on
#   - BioRED's development split (1,000-candidate sample, pair question): the SELECTION set. It is
#     public, human-annotated and disjoint from both reporting sets;
#   - the 437-row biodiversity benchmark (three questions): reported, never used to choose.
# Rule, fixed before any result: the teacher is the candidate with the highest BioRED-dev AUPRC; it
# replaces qwen3:32b only if it beats qwen3:32b's dev AUPRC by >= 0.01. Then the 48,338 training
# rows are relabelled with it (same prompt) and the sentence / pair / triple arms are retrained
# (3 seeds, the paper's recipe) on its verdicts -- a new training-data version, nothing overwritten.
# Candidates other than the best so far are removed from disk after scoring (re-pullable); models
# that were on the machine before this run are never removed.
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"
R=results/teacher_search_2026-10-04; mkdir -p $R/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/search.log; }
CANDIDATES="qwen3.8:27b qwen3.6:35b qwen3.6:27b gemma4:31b medgemma:27b qwen3.5:35b qwen3.5:27b glm-4.7-flash muse-glimmer:30b clef:27b llama3.3:70b gpt-oss:120b"
NEW=/home/egaillac/opt/ollama-0.35.1/bin/ollama
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_URL=http://127.0.0.1:11435     # the v0.35.1 server
OLL() { $NEW "$@"; }
PRESENT="$(OLL list | awk 'NR>1{print $1}')"
devauprc() {  # AUPRC of a model's BioRED-dev pair answers, or -1
  python3 - "$1" <<'PY'
import sys, pandas as pd
from sklearn.metrics import average_precision_score as ap
f = f"results/paperA_v2/llm/bioreddev_{sys.argv[1].replace(':', '-')}-v035_pair.csv"
try:
    o = pd.read_csv(f); print(f"{ap(o.label, o.p_yes):.4f}" if len(o) == 1000 else "-1")
except Exception:
    print("-1")
PY
}
score() {  # score <model> [extra llm_baseline args]: on the v0.35.1 server, files suffixed -v035
  local m=$1; shift
  python3 scripts/llm_baseline.py --bench bioreddev --model $m --forms pair --n 1000 --tag -v035 "$@" >> $R/logs/llm.log 2>&1 \
    || log "  FAILED bioreddev $m"
  python3 scripts/llm_baseline.py --bench biodiv --model $m --tag -v035 "$@" >> $R/logs/llm.log 2>&1 || log "  FAILED biodiv $m"
}

log "search start (v2, Ollama 0.35.1)"
# reference: the current teacher, on the same server as the candidates
echo "$PRESENT" | grep -qx qwen3:32b || OLL pull qwen3:32b > $R/logs/pull_qwen3_32b.log 2>&1
PRESENT="$(OLL list | awk 'NR>1{print $1}')"
score qwen3:32b
REF=$(devauprc qwen3:32b); log "reference qwen3:32b dev AUPRC $REF"

BEST=qwen3:32b; BESTV=$REF
for m in $CANDIDATES; do
  echo "$PRESENT" | grep -qx "$m" && pre=1 || pre=0
  [ $pre = 1 ] || { log "pull $m"; OLL pull $m > $R/logs/pull_${m//[:\/]/_}.log 2>&1 || { log "  pull FAILED $m"; continue; }; }
  score $m
  v=$(devauprc $m); log "$m dev AUPRC $v (best so far $BEST $BESTV)"
  if python3 -c "import sys; sys.exit(0 if float('$v') > float('$BESTV') else 1)"; then
    old=$BEST; BEST=$m; BESTV=$v
    [ "$old" != qwen3:32b ] && ! echo "$PRESENT" | grep -qx "$old" && OLL rm $old > /dev/null 2>&1 && log "  removed $old"
  else
    [ $pre = 0 ] && OLL rm $m > /dev/null 2>&1 && log "  removed $m"
  fi
done
# the 122B needs nearly the whole GPU, so it is scored last, alone, on the system server (no suffix)
OLL stop $BEST > /dev/null 2>&1; sleep 5
OLLAMA_URL=http://localhost:11434 python3 scripts/llm_baseline.py --bench bioreddev --model qwen3.5:122b --forms pair \
  --n 1000 --num-gpu 44 --num-ctx 4096 --workers 1 >> $R/logs/llm.log 2>&1 || log "  FAILED bioreddev 122b"
log "reference qwen3.5:122b (system server) dev AUPRC $(python3 -c "
import pandas as pd; from sklearn.metrics import average_precision_score as ap
o = pd.read_csv('results/paperA_v2/llm/bioreddev_qwen3.5-122b_pair.csv'); print(round(ap(o.label, o.p_yes), 4))" 2>/dev/null) (not eligible: 1.4 s/call)"
log "selection: best $BEST dev AUPRC $BESTV vs qwen3:32b $REF"
if ! python3 -c "import sys; sys.exit(0 if float('$BESTV') >= float('$REF') + 0.01 else 1)"; then
  log "no candidate beats the current teacher by 0.01 on BioRED dev: no relabel"
  bash scripts/notify.sh "Teacher search finished: no better teacher" "$(tail -30 $R/search.log)" || true
  exit 0
fi
TAG=${BEST//[:.]/-}
CORPUS=data/training/distill/v3_combined_train_${TAG}
log "relabel the corpus with $BEST -> ${CORPUS}_soft.csv"
python3 scripts/soft_relabel.py --model $BEST --workers 4 --out ${CORPUS}_softlabels.csv > $R/logs/relabel.log 2>&1 \
  || { log "  relabel FAILED"; exit 1; }
python3 - "$CORPUS" <<'PY'
import sys, pandas as pd
c = sys.argv[1]; d = pd.read_csv(c + "_soft.csv")
d.assign(label=(d.soft_label >= 0.5).astype(int)).to_csv(c + ".csv", index=False)
print(f"{c}.csv: {len(d)} rows, positive rate {(d.soft_label >= 0.5).mean():.3f}")
PY
for fmt in sentence pair triple; do for s in 1 2 3; do
  out=models/teacher_${TAG}/${fmt}_s${s}
  [ -f $out/student_config.json ] && continue
  log "train $out"
  python3 experiments/multitask/train_student.py --data ${CORPUS}.csv --input-format $fmt --seed $s \
    --out $out > $R/logs/train_${fmt}_s${s}.log 2>&1 && log "  $(tail -1 $R/logs/train_${fmt}_s${s}.log)" \
    || log "  FAILED $out"
done; done
log "done: teacher $BEST, students in models/teacher_${TAG}/"
bash scripts/notify.sh "Teacher search finished: $BEST" "$(tail -40 $R/search.log)" || true
