#!/usr/bin/env bash
# Knowledge Graph sandbox — autonomous pipeline with checkpointing and 5h self-limit.
# Runs four tiers; skips any tier whose sentinel file (tier*_done) already exists.
# At 95% of 5h (≈17100s), emails the user, sleeps 5h, then relaunches itself.
#
# Usage:
#   nohup bash classifier/experiments/knowledge_graph/pipeline_kg.sh >> .../pipeline_kg.log 2>&1 &

set -euo pipefail
cd /home/egaillac/MetaP
source MPvenv/bin/activate

SELF="$(realpath "${BASH_SOURCE[0]}")"
KG_DIR="classifier/experiments/knowledge_graph"
RESULTS="$KG_DIR/results"
LOG="$RESULTS/pipeline_kg.log"
NOTIFY="classifier/scripts/notify.sh"
SESSION_START=$(date +%s)
LIMIT_SECONDS=17100   # 95% of 5h = 4h45m

mkdir -p "$RESULTS/tier1" "$RESULTS/tier2" "$RESULTS/tier3" "$RESULTS/tier4"

log()    { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
notify() { bash "$NOTIFY" "$1" "$2" 2>/dev/null || true; }

elapsed() { echo $(( $(date +%s) - SESSION_START )); }

check_limit() {
    local secs
    secs=$(elapsed)
    if [ "$secs" -ge "$LIMIT_SECONDS" ]; then
        log "⏱  5h limit reached (${secs}s elapsed). Sleeping 5h then relaunching."
        notify "KG pipeline: 5h pause" \
               "Completed tiers so far — pipeline will resume automatically in 5h. See $LOG"
        sleep 18000   # 5 hours
        log "Relaunching pipeline after 5h sleep..."
        nohup bash "$SELF" >> "$LOG" 2>&1 &
        exit 0
    fi
    log "  Elapsed: $(( secs / 60 ))m $(( secs % 60 ))s"
}

log "=== KG pipeline started ==="

# ── Paths ─────────────────────────────────────────────────────────────────────
V14="classifier/data/training/training_data_v14.csv"
TAX_CACHE="classifier/data/taxonomies/globi_taxonomy_cache.json"
MODEL="classifier/models/multitask/full_typed_a05_ner2"
EP_RELAX="classifier/data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
GLOBI_TSV="classifier/data/globi/interactions.tsv.gz"

TIER1_DONE="$RESULTS/tier1/tier1_done"
TIER2_DONE="$RESULTS/tier2/tier2_done"
TIER3_DONE="$RESULTS/tier3/tier3_done"
TIER4_DONE="$RESULTS/tier4/tier4_done"

# ── Tier 1: KG from v14 data (~30 min) ───────────────────────────────────────
if [ ! -f "$TIER1_DONE" ]; then
    log "=== TIER 1: Building KG from v14 data ==="
    python -u "$KG_DIR/tier1_build_kg.py" \
        --v14-path    "$V14" \
        --taxonomy-cache "$TAX_CACHE" \
        --output-dir  "$RESULTS/tier1" \
        2>&1 | tee -a "$LOG"
    T1=$(python3 -c "
import json; s=json.load(open('$RESULTS/tier1/kg_stats.json'))
print(f\"Nodes={s['n_nodes']}, Edges={s['n_edges']}, Categories={list(s['category_counts'].keys())[:4]}\")
" 2>/dev/null || echo "see log")
    log "TIER 1 done: $T1"
    notify "KG Tier 1 done" "$T1"
else
    log "TIER 1: SKIPPED (tier1_done exists)"
fi
check_limit

# ── Tier 2: NER-based triple extraction (~1h) ─────────────────────────────────
if [ ! -f "$TIER2_DONE" ]; then
    log "=== TIER 2: NER-based triple extraction ==="
    python -u "$KG_DIR/tier2_triple_extractor.py" \
        --v14-path   "$V14" \
        --model-dir  "$MODEL" \
        --tier1-kg   "$RESULTS/tier1/kg_v14.pkl" \
        --output-dir "$RESULTS/tier2" \
        2>&1 | tee -a "$LOG"
    T2=$(python3 -c "
import json; s=json.load(open('$RESULTS/tier2/kg_stats.json'))
print(f\"Nodes={s['n_nodes']}, Edges={s['n_edges']}, HighConfTriples={s.get('n_high_confidence_triples',0)}\")
" 2>/dev/null || echo "see log")
    log "TIER 2 done: $T2"
    notify "KG Tier 2 done" "$T2"
else
    log "TIER 2: SKIPPED (tier2_done exists)"
fi
check_limit

# ── Tier 3: SpERT training + evaluation (~1.5h) ───────────────────────────────
if [ ! -f "$TIER3_DONE" ]; then
    log "=== TIER 3a: Building weak supervision data ==="
    python -u "$KG_DIR/tier3_train_spert.py" \
        --phase build-ws \
        --v14-path    "$V14" \
        --output-dir  "$RESULTS/tier3" \
        2>&1 | tee -a "$LOG"
    check_limit

    log "=== TIER 3b: Training SpERT ==="
    python -u "$KG_DIR/tier3_train_spert.py" \
        --phase train \
        --ws-data          "$RESULTS/tier3/weak_supervision_train.jsonl" \
        --pretrained-encoder "$MODEL" \
        --output-dir       "$RESULTS/tier3" \
        --epochs 3 --lr 2e-5 \
        2>&1 | tee -a "$LOG"
    check_limit

    log "=== TIER 3c: EP-relax evaluation ==="
    python -u "$KG_DIR/tier3_train_spert.py" \
        --phase evaluate \
        --model-dir  "$RESULTS/tier3/spert_model" \
        --ep-relax   "$EP_RELAX" \
        --output-dir "$RESULTS/tier3" \
        2>&1 | tee -a "$LOG"

    T3=$(python3 -c "
import json; r=json.load(open('$RESULTS/tier3/ep_relax_eval.json'))
f1=r['spert']['best_f1']; d=r['delta_f1']
print(f\"SpERT EP-F1={f1:.3f} (delta={d:+.3f}). {r['recommendation'][:80]}\")
" 2>/dev/null || echo "see log")
    log "TIER 3 done: $T3"
    notify "KG Tier 3 done" "$T3"
    touch "$TIER3_DONE"
else
    log "TIER 3: SKIPPED (tier3_done exists)"
fi
check_limit

# ── Tier 4: Publishable explorations (~1.5h) ──────────────────────────────────
if [ ! -f "$TIER4_DONE" ]; then
    log "=== TIER 4: Publishable explorations ==="
    HARVEST_FLAG=""
    [ ! -f "$GLOBI_TSV" ] && log "  GloBI TSV not found — harvest will be skipped"

    python -u "$KG_DIR/tier4_explorations.py" \
        --tier2-kg   "$RESULTS/tier2/kg_tier2.pkl" \
        --model-dir  "$MODEL" \
        --ep-relax   "$EP_RELAX" \
        --globi-tsv  "$GLOBI_TSV" \
        --output-dir "$RESULTS/tier4" \
        2>&1 | tee -a "$LOG"

    T4=$(python3 -c "
import json; r=json.load(open('$RESULTS/tier4/exploration_report.json'))
a=r.get('experiment_a',{}).get('delta_f1','?')
b=r.get('experiment_b',{}).get('overall_recall','?')
c=r.get('experiment_c',{}).get('new_triples_added','?')
print(f\"ExpA delta_f1={a}, ExpB recall={b:.1%}, ExpC new_triples={c}\")
" 2>/dev/null || echo "see log")
    log "TIER 4 done: $T4"
    notify "KG Tier 4 done (pipeline complete)" "$T4. Full report at $RESULTS/tier4/exploration_report.json"
else
    log "TIER 4: SKIPPED (tier4_done exists)"
fi

log "=== KG pipeline complete ==="
notify "KG pipeline COMPLETE" "All 4 tiers done. Results in $RESULTS. Check exploration_report.json for findings."
