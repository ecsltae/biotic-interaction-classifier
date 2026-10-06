#!/usr/bin/env bash
# Waits until the A1 labelling has finished (its split summary exists, or A1 failed), so that this
# job never competes with it for the teacher's server, then scores the enumerated biodiversity pairs
# with qwen3:32b (biodiv_llm_enum.py). Ends by 11:15 at the latest.
cd $HOME/MetaP/classifier
source $HOME/MetaP/MPvenv/bin/activate
unset OLLAMA_URL
L=results/reframe_2026-10-05/logs/biodiv_llm_enum.log
S=results/reframe_2026-10-05/state
END=$(date -d "2026-10-05 11:15" +%s)
echo "[$(date '+%F %T')] waiting for the A1 labelling to finish" >> $L
while [ ! -f results/paperA_v2/sentence_level/sentlab_split.json ] && [ ! -f $S/gpu_sentlab.failed ]; do
  [ $(date +%s) -gt $(( END - 1800 )) ] && { echo "[$(date '+%F %T')] gave up waiting" >> $L; exit 1; }
  sleep 60
done
echo "[$(date '+%F %T')] start" >> $L
timeout $(( END - $(date +%s) )) python3 scripts/sentence_level/biodiv_llm_enum.py --model qwen3:32b --workers 4 >> $L 2>&1
echo "[$(date '+%F %T')] exit $?" >> $L
