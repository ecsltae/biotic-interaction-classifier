#!/usr/bin/env bash
# Paper A sentence-level reframe, unattended (2026-10-05). Stages: analysis || literature, then the
# GPU job the analysis stage launches, then writer, then reviewer. One launch works at most 4 h 45
# min (the 5-hour rule) and nothing runs after 12:00. A later launch resumes unfinished stages from
# their progress files. A fixed relaunch fires at 05:45, after the usage-limit reset; a launch
# stopped by the usage limit after that time schedules a retry 45 min later (at most two retries).
set -u
R=$HOME/MetaP/classifier/results/reframe_2026-10-05
S=$R/state; L=$R/logs/orchestrator.log
HARD_STOP=$(date -d "2026-10-05 12:00:00" +%s); RELAUNCH_AT=$(date -d "2026-10-05 05:45:00" +%s)
exec 8>$S/orchestrator.lock
flock -w 3600 8 || { echo "[$(date '+%F %T')] previous launch still running after 1 h; exiting" >> $L; exit 0; }
[ -f $S/reviewer.done ] && { echo "[$(date '+%F %T')] all stages done; nothing to do" >> $L; exit 0; }
NOW=$(date +%s); DEADLINE=$(( NOW + 17100 )); [ $DEADLINE -gt $HARD_STOP ] && DEADLINE=$HARD_STOP
LAUNCH=$(( $(ls $S/launch.* 2>/dev/null | wc -l) + 1 )); date > $S/launch.$LAUNCH
log() { echo "[$(date '+%F %T')] launch $LAUNCH: $*" >> $L; }
left() { echo $(( DEADLINE - $(date +%s) )); }
[ $(left) -lt 1800 ] && { log "less than 30 min before the hard stop; not starting"; exit 0; }
log "start, deadline $(date -d @$DEADLINE '+%F %T')"
hits0=$(cat $S/limit_hits 2>/dev/null | wc -l)

bash $R/run_stage.sh analysis $DEADLINE & pa=$!
sleep 20
bash $R/run_stage.sh literature $DEADLINE & pl=$!
wait $pa $pl
# the detached GPU job of the analysis stage (teacher sentence labels -> sentence-label students)
while [ -f $S/analysis.done ] && [ ! -f $S/gpu_sentlab.done ] && [ ! -f $S/gpu_sentlab.failed ] \
      && [ $(left) -gt 3600 ]; do sleep 120; done
if [ -f $S/analysis.done ] && [ -f $S/literature.done ]; then
  bash $R/run_stage.sh writer $DEADLINE && bash $R/run_stage.sh reviewer $DEADLINE
fi

status=$(for s in analysis literature gpu_sentlab writer reviewer; do
  if [ -f $S/$s.done ]; then st=done; elif [ -f $S/$s.failed ]; then st=FAILED; else st="not done"; fi
  printf '  %-12s %s\n' $s "$st"; done)
log "end:$(echo "$status" | tr -s ' \n' ' ')"
[ -f $S/reviewer.done ] && exit 0
hits1=$(cat $S/limit_hits 2>/dev/null | wc -l)
retries=$(ls $S/retry.* 2>/dev/null | wc -l)
if [ $hits1 -gt $hits0 ] && [ $(date +%s) -lt $RELAUNCH_AT ]; then
  next="Stopped by the usage limit; the relaunch at 05:45 resumes from the progress files."
elif [ $hits1 -gt $hits0 ] && [ $retries -lt 2 ] && [ $(( $(date +%s) + 2700 + 1800 )) -lt $HARD_STOP ]; then
  date > $S/retry.$((retries + 1))
  systemd-run --user --unit=reframe-retry$((retries + 1)) --on-active=45min /bin/bash $R/orchestrate.sh >> $L 2>&1
  next="Stopped by the usage limit; retry scheduled at $(date -d '+45 min' '+%H:%M')."
elif [ $(date +%s) -lt $RELAUNCH_AT ]; then
  next="The relaunch at 05:45 resumes from the progress files."
else
  next="No further launch is scheduled; the work waits for your instructions."
fi
log "$next"
bash $HOME/MetaP/classifier/scripts/notify.sh "Paper A reframe: launch $LAUNCH ended before the final report" \
"Stage status:
$status

$next
Progress files: $S/*.progress.md
Log: $L
Last lines of the stage logs:
$(for f in $(ls -t $R/logs/*.out 2>/dev/null | head -2); do echo "--- $f"; tail -c 1200 $f; done)"
