#!/bin/bash
# Start the Literature Trust Service on port 8003.
set -e
source /home/egaillac/MetaP/MPvenv/bin/activate
cd /home/egaillac/MetaP
echo "Starting trust service on port 8003 ..."
uvicorn classifier.api.trust_service:app --host 0.0.0.0 --port 8003 --workers 2
