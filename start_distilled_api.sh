#!/usr/bin/env bash
# Start the Multi-task BiomedBERT API on port 8003
# Champion model: mt_distill_warm_ner0 (warm-start, 0 NER pretrain epochs, F1=0.874)
# Survives logout (nohup), accessible to colleagues (0.0.0.0)

set -e
cd /home/egaillac/MetaP
source MPvenv/bin/activate

PORT=8003
LOG="classifier/logs/api_distilled.log"
mkdir -p classifier/logs

# Kill any existing instance (uvicorn.run() is in-process, so match on the script name, not "uvicorn")
pkill -f "fastapi_multitask.py" 2>/dev/null || true
sleep 1

echo "Starting Multi-task BiomedBERT API (mt_distill_warm_ner0)..."
echo "  Model: multitask/mt_distill_warm_ner0 (NER scheme=full_typed, α=0.5, warm-start, 0ep NER pretrain)"
echo "  500-sentence test set: F1=0.874 | AUC=0.918 | beats ensemble (F1=0.850)"
echo "  Port: $PORT"
echo "  Log:  $LOG"
echo ""

nohup python -u classifier/api/fastapi_multitask.py \
    >> "$LOG" 2>&1 &

PID=$!
echo "Started PID=$PID"
echo ""

# Wait for it to come up
for i in $(seq 1 15); do
    sleep 2
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✓ API is up."
        echo ""
        echo "  Local:      http://localhost:$PORT"
        echo "  Colleagues: http://$(hostname -I | awk '{print $1}'):$PORT"
        echo "  Docs:       http://$(hostname -I | awk '{print $1}'):$PORT/docs"
        echo ""
        echo "Example:"
        echo "  curl -s -X POST http://localhost:$PORT/predict \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"text\": \"Wolbachia infects Drosophila melanogaster.\"}' | python3 -m json.tool"
        echo ""
        echo "To stop:    pkill -f 'fastapi_multitask.py'"
        echo "To monitor: tail -f $LOG"
        exit 0
    fi
done

echo "WARNING: API did not respond after 30s. Check $LOG"
tail -20 "$LOG"
exit 1
