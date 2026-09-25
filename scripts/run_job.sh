#!/usr/bin/env bash
# Long-running commands owned by this project, detached from the caller's shell and tool timeout.
#   run_job.sh start <name> -- <cmd> [args...]  new session; writes <name>.pid.json, output to <name>.log
#   run_job.sh wait <name> [timeout_s]          polls (default 540 s); prints the log tail and "job <name> finished: exit=<code>"
#                                               returns 0 (exit 0), 1 (other exit code), 124 (still running), 3 (stopped or died)
#   run_job.sh stop <name>                      SIGTERM, then SIGKILL after 30 s, to the recorded process group, only if it is ours
# Files live in $AUTOFLY_JOBS_DIR (default ROOT/runs/jobs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS=${AUTOFLY_JOBS_DIR:-$ROOT/runs/jobs}
USAGE="usage: run_job.sh start <name> -- <cmd...> | wait <name> [timeout_s] | stop <name>"
MODE="${1:?$USAGE}"
NAME="${2:?$USAGE}"
shift 2
mkdir -p "$JOBS"
PIDFILE=$JOBS/$NAME.pid.json
EXITFILE=$JOBS/$NAME.exit
STOPPED=$JOBS/$NAME.stopped
LOG=$JOBS/$NAME.log
MARKER="autofly_job:$NAME"

# python3 rather than jq: jq is not installed on a stock host, and every host that runs this project has python3.
recorded() { env -u PYTHONPATH python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$PIDFILE" "$1"; }

is_ours() {  # recorded PID alive, argv[3] is our marker, process group unchanged
  local pid pgid
  pid=$(recorded pid)
  pgid=$(recorded pgid)
  kill -0 "$pid" 2>/dev/null || return 1
  [ "$(tr '\0' '\n' < "/proc/$pid/cmdline" | sed -n 4p)" = "$MARKER" ] || return 1
  [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pgid" ]
}

case "$MODE" in
  start)
    { [ "${1:-}" = "--" ] && [ $# -ge 2 ]; } || { echo "$USAGE"; exit 2; }
    shift
    if [ -f "$PIDFILE" ] && is_ours; then
      echo "job $NAME is still running with pid $(recorded pid)"
      exit 1
    fi
    rm -f "$PIDFILE" "$EXITFILE" "$STOPPED"
    setsid bash -c 'exit_file=$1; shift; code=0; "$@" || code=$?; echo "$code" > "$exit_file"' \
      "$MARKER" "$EXITFILE" "$@" < /dev/null > "$LOG" 2>&1 &
    PID=$!
    PGID=$PID
    for _ in $(seq 50); do  # wait until setsid has made the job its own process group
      SEEN=$(ps -o pgid= -p "$PID" | tr -d ' ' || true)
      [ -z "$SEEN" ] && break
      PGID=$SEEN
      [ "$PGID" = "$PID" ] && break
      sleep 0.1
    done
    env -u PYTHONPATH python3 -c 'import json, sys, time; pid, pgid, log = sys.argv[1:4]; print(json.dumps({"pid": int(pid), "pgid": int(pgid), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": log, "cmd": sys.argv[4:]}))' \
      "$PID" "$PGID" "$LOG" "$@" > "$PIDFILE"
    echo "job $NAME started: pid $PID, log $LOG"
    ;;
  wait)
    TIMEOUT=${1:-540}
    [ -f "$PIDFILE" ] || { echo "no job $NAME"; exit 2; }
    START=$(date +%s)
    while [ ! -f "$EXITFILE" ]; do
      if [ -f "$STOPPED" ]; then echo "job $NAME was stopped"; exit 3; fi
      if ! kill -0 "$(recorded pid)" 2>/dev/null; then
        sleep 1
        if [ ! -f "$EXITFILE" ]; then tail -n 20 "$LOG"; echo "job $NAME died without an exit code"; exit 3; fi
        break
      fi
      if [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
        tail -n 5 "$LOG"
        echo "job $NAME still running after ${TIMEOUT} s (pid $(recorded pid)); call wait again"
        exit 124
      fi
      sleep 2
    done
    CODE=$(cat "$EXITFILE")
    tail -n 20 "$LOG"
    echo "job $NAME finished: exit=$CODE"
    [ "$CODE" = "0" ] || exit 1
    ;;
  stop)
    [ -f "$PIDFILE" ] || { echo "no_job"; exit 0; }
    PID=$(recorded pid)
    PGID=$(recorded pgid)
    if ! kill -0 "$PID" 2>/dev/null; then echo "not_running"; exit 0; fi
    is_ours || { echo "pid $PID does not match job $NAME (marker or process group); not stopping it"; exit 1; }
    touch "$STOPPED"
    kill -TERM -- "-$PGID"
    RESULT=terminated
    for _ in $(seq 150); do kill -0 -- "-$PGID" 2>/dev/null || break; sleep 0.2; done
    if kill -0 -- "-$PGID" 2>/dev/null; then kill -KILL -- "-$PGID"; RESULT=killed; sleep 1; fi
    echo "$RESULT"
    ;;
  *)
    echo "$USAGE"
    exit 2
    ;;
esac
