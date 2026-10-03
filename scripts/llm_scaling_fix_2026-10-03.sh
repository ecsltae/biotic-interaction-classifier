#!/usr/bin/env bash
# Zero-shot scaling, the two sizes whose default Ollama tags (qwen3:4b, qwen3:30b) are the 2507
# re-releases: they ignore think=false, so the first token is prose and P(YES) is 0 on every row.
# The original hybrid-thinking Qwen3 builds (40,960 context, like qwen3:32b) are the tags below.
# Waits for an idle Ollama: after the teacher relabel and before the T3 students finish (the 122B
# starts then), or after the 122B. Never runs alongside another Ollama job of the overnight queue.
set -u
cd "$(dirname "$0")/.."
source ../MPvenv/bin/activate
R=results/overnight_2026-10-03; S=$R/state
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/queue.log; }
# NOWAIT=1: run now (with no training on the GPU, qwen3:32b, a 4B and the 30B-A3B fit side by side)
[ -n "${NOWAIT:-}" ] || until { [ -f $S/relabel.done ] || [ -f $S/relabel.failed ]; } && [ ! -f $S/train.done ] || [ -f $S/llm.done ]; do sleep 60; done
for m in qwen3:4b-q4_K_M qwen3:30b-a3b-q4_K_M; do
  # do not start a model once the 122B is due (train.done without llm.done)
  until [ ! -f $S/train.done ] || [ -f $S/llm.done ]; do sleep 60; done
  log "zero-shot $m (scaling fix)"
  python3 scripts/llm_baseline.py --bench biodiv --model $m >> $R/logs/llm_scaling_fix.log 2>&1 || log "  FAILED biodiv $m"
  python3 scripts/llm_baseline.py --bench biored --model $m --n 3000 >> $R/logs/llm_scaling_fix.log 2>&1 || log "  FAILED biored $m"
done
touch $S/scaling_fix.done; log "scaling fix done"
