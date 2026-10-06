#!/usr/bin/env bash
# One stage of the Paper A reframe: a headless Claude Code session (claude.ai subscription, no API
# key) with the shared rules, the stage prompt and a deadline. Skips a finished stage and never runs
# the same stage twice at once. Exit 0 iff the stage is done.
#   bash run_stage.sh <analysis|literature|writer|reviewer> <deadline, epoch seconds>
set -u
R=$HOME/MetaP/classifier/results/reframe_2026-10-05
STAGE=$1; DEADLINE=$2
L=$R/logs/orchestrator.log
[ -f $R/state/$STAGE.done ] && exit 0
exec 9>$R/state/$STAGE.lock
flock -n 9 || { echo "[$(date '+%F %T')] $STAGE is already running elsewhere" >> $L; exit 1; }
export CLAUDE_CONFIG_DIR=$HOME/.claude-hesso
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
cd $HOME/MetaP/classifier
left=$(( DEADLINE - $(date +%s) - 120 ))
[ $left -lt 1500 ] && { echo "[$(date '+%F %T')] $STAGE: only ${left}s left, not starting" >> $L; exit 1; }
n=$(( $(ls $R/logs/$STAGE.*.out 2>/dev/null | wc -l) + 1 ))
prompt="$(sed -e "s/{STAGE}/$STAGE/g" -e "s/{START}/$(date '+%F %H:%M')/g" \
              -e "s/{DEADLINE}/$(date -d @$(( $(date +%s) + left )) '+%F %H:%M')/g" $R/prompts/common.md
          echo; cat $R/prompts/$STAGE.md)"
echo "[$(date '+%F %T')] $STAGE attempt $n start (timeout ${left}s)" >> $L
t0=$(date +%s)
timeout $left claude -p "$prompt" --permission-mode bypassPermissions --disallowedTools Workflow \
    --output-format text > $R/logs/$STAGE.$n.out 2> $R/logs/$STAGE.$n.err 8>&- 9>&-
rc=$?
el=$(( $(date +%s) - t0 ))
done_=$([ -f $R/state/$STAGE.done ] && echo yes || echo no)
echo "[$(date '+%F %T')] $STAGE attempt $n end rc=$rc after ${el}s done=$done_" >> $L
if [ $done_ = no ] && { [ $el -lt 300 ] || grep -qiE "usage limit|limit reached|hit your limit|rate.limit" $R/logs/$STAGE.$n.out $R/logs/$STAGE.$n.err; }; then
  echo "$STAGE attempt $n: $(tail -c 300 $R/logs/$STAGE.$n.out $R/logs/$STAGE.$n.err | tr '\n' ' ')" >> $R/state/limit_hits
fi
[ $done_ = yes ]
