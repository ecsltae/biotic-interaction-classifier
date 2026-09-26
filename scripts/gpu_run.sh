#!/usr/bin/env bash
# Wait for a GPU with enough free memory, then run the command pinned to it.
# Three sessions share these GPUs, so a run that grabs whatever is free at launch
# time gets OOM-killed minutes later. This polls until a slot is genuinely free.
NEED=${NEED:-9000}
TRIES=${TRIES:-240}
for t in $(seq 1 $TRIES); do
  G=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F', ' -v n=$NEED '{f=$2-$3; if (f>n && f>best) {best=f; g=$1}} END {if (g!="") print g}')
  if [ -n "$G" ]; then
    echo "[gpu_run] GPU $G free enough (need ${NEED}MiB), launching: $*" >&2
    CUDA_VISIBLE_DEVICES=$G "$@"
    exit $?
  fi
  sleep 15
done
echo "[gpu_run] no GPU with ${NEED}MiB free after $((TRIES*15))s" >&2; exit 1
