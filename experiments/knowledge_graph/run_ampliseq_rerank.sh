#!/usr/bin/env bash
# Ampliseq re-ranking via BiotXplorer + GloBI ecological coherence.
# Runs overnight: OTT resolution → BiotXplorer pair queries → re-ranking → report.
# Usage: nohup bash classifier/experiments/knowledge_graph/run_ampliseq_rerank.sh &

set -euo pipefail
cd /home/egaillac/MetaP
source MPvenv/bin/activate

KG_DIR="classifier/experiments/knowledge_graph"
OUT="$KG_DIR/results/ampliseq_rerank"
LOG="$OUT/run.log"

mkdir -p "$OUT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Ampliseq re-ranking pipeline started ==="
log "APIs: SiBILS (OTT resolution) + BiotXplorer (text-mined interactions) + GloBI (fallback)"

python -u "$KG_DIR/ampliseq_rerank.py" \
    --asv-table   "final_results/COI_ASVs_20260108/dada2/ASV_table.tsv" \
    --taxonomy    "final_results/COI_ASVs_20260108/dada2/species_top50.csv" \
    --globi-cache "classifier/data/globi/globi_lookup_cache.pkl" \
    --kg-path     "$KG_DIR/results/tier1/kg_v14.pkl" \
    --output-dir  "$OUT" \
    --beta        1.0 \
    --top-n       20 \
    2>&1 | tee -a "$LOG"

log "=== Pipeline complete ==="

SUMMARY=$(python3 -c "
import pandas as pd
df = pd.read_csv('$OUT/rank_changes_summary.csv')
boosted = df[df['avg_delta_rank'] > 0]
n_bx = (df['avg_biotxplorer_score'] > 0).sum()
if len(boosted):
    top = boosted.iloc[0]
    msg = f\"{len(boosted)}/{len(df)} species boosted. Top: {top['species']} (Δ={top['avg_delta_rank']:+.2f}). BiotXplorer coverage: {n_bx}/{len(df)} species.\"
else:
    msg = f\"0/{len(df)} species boosted. BiotXplorer coverage: {n_bx}/{len(df)} species.\"
print(msg)
" 2>/dev/null || echo "see log")

log "Result: $SUMMARY"
bash classifier/scripts/notify.sh "Ampliseq re-ranking done" \
    "$SUMMARY — report at $OUT/report.md"
