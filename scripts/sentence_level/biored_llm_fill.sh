#!/usr/bin/env bash
cd $HOME/MetaP/classifier
source $HOME/MetaP/MPvenv/bin/activate
unset OLLAMA_URL
L=results/reframe_2026-10-05/logs/biored_llm_fill.log
timeout 2700 python3 scripts/sentence_level/biored_llm_fill.py --workers 2 >> $L 2>&1
echo "exit $? at $(date '+%F %T')" >> $L
