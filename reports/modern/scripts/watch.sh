#!/bin/bash
# Read-only monitor for the supervised pipeline.
#
# Heartbeat every 2 hours.  The underlying check still runs every 2 minutes,
# because a dead supervisor found two hours late is two hours of nothing
# running: faults are reported the moment they appear, and only the routine
# "still fine" line is held back to the two-hour cadence.  Each fault reports
# once per occurrence rather than on every check, so a persistent condition
# does not flood the channel.
cd /home/abc/temp/modern
HEARTBEAT=7200
CHECK=120
last_beat=0
last_total=-1
stalled_since=$(date +%s)
alerted=""

while true; do
  status=$(python3 - <<'PY'
import json
try:
    p = json.load(open('/home/abc/temp/modern/snapshot/pipeline_status.json'))
except Exception as exc:
    print(f"BAD_STATUS {type(exc).__name__}"); raise SystemExit
q = p['queues']
done = sum(v['completed'] for v in q.values())
err  = sum(v['errors'] for v in q.values())
parts = " ".join(f"{k}:{v['completed']}/{v['planned']}" for k, v in q.items()
                 if v['completed'] or k == p['stage'].split('_')[0])
lo, hi = p.get('remaining_hours_range', [0, 0])
print(f"{p['status']} {p['stage']} pid={p['pid']} child={p.get('child_pid')} "
      f"done={done} err={err} | {parts} | 剩余 {lo:.1f}-{hi:.1f}h")
PY
)
  total=$(echo "$status" | grep -o 'done=[0-9]*' | cut -d= -f2)
  errs=$(echo "$status"  | grep -o 'err=[0-9]*'  | cut -d= -f2)
  pid=$(echo "$status"   | grep -o 'pid=[0-9]*'  | cut -d= -f2)
  free_gb=$(df --output=avail -BG /home/abc | tail -1 | tr -dc '0-9')
  now=$(date +%s)

  fire() {   # fire <key> <message>: emit once per distinct condition
    case "$alerted" in *"|$1|"*) return;; esac
    alerted="$alerted|$1|"; echo "告警 $2"
  }

  if [ -n "$pid" ] && ! ps -p "$pid" > /dev/null 2>&1; then
    fire dead "守护进程 $pid 已不存在 | $status"
  fi
  if [ -n "$errs" ] && [ "$errs" -gt 0 ] 2>/dev/null; then
    fire "err$errs" "出现错误回合 err=$errs | $status"
  fi
  if [ -n "$free_gb" ] && [ "$free_gb" -lt 15 ] 2>/dev/null; then
    fire disk "磁盘仅剩 ${free_gb}G | $status"
  fi
  if [ "$total" != "$last_total" ]; then
    last_total=$total; stalled_since=$now
    alerted=$(echo "$alerted" | sed 's/|stall|//')   # progress clears a stall
  elif [ $((now - stalled_since)) -gt 1800 ]; then
    fire stall "30 分钟无新回合 | $status"
  fi

  if [ $((now - last_beat)) -ge $HEARTBEAT ]; then
    echo "$(date +%H:%M) $status 磁盘${free_gb}G"; last_beat=$now
  fi
  sleep $CHECK
done
