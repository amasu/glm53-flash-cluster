#!/usr/bin/env bash
# GLM endpoint watchdog: probe :SERVING_PORT, restart the stack if the engine
# is down. Runs from the orchestrator machine (e.g. cron: `*/5 * * * *
# ~/glm53-flash-cluster/watchdog.sh`). No LLM; script only.
#
# Generic by design: it does not care WHICH stack is up — it probes the
# shared endpoint and, on failure, restarts via cluster.sh takeover, which
# brings up whatever STACK is set in .env. Single-stack policy: only one
# profile ever runs, so the shared container names (glm53-head/worker) are
# the whole surface.
#
# Host-agnostic: HEAD_HOST / WORKER_IP / SERVING_PORT from .env or environment.
# 24/7 watchdog: this root-level script is what the watchdog cron/restart
# path runs (replaces the deleted lab/lab-watchdog.sh — see NOTES-512k.md).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/glm53-watchdog.log"
ts() { date '+%H:%M:%S'; }

if [ -f "$REPO_DIR/.env" ]; then set -a; . "$REPO_DIR/.env"; set +a; fi
: "${HEAD_HOST:?HEAD_HOST must be set in .env or environment}"
: "${WORKER_IP:?WORKER_IP must be set in .env or environment}"
PORT="${SERVING_PORT:-8000}"

if curl -sfS -m 8 "http://$HEAD_HOST:$PORT/v1/models" >/dev/null 2>&1; then
  exit 0
fi

# endpoint down — is the head container freshly started (<35 min => boot in progress)?
age_min=$(ssh -T -o ConnectTimeout=8 -o BatchMode=yes "$HEAD_HOST" \
  "docker ps --format '{{.RunningFor}}' --filter name=glm53-head --filter name=glm53-lab-head --filter name=glm53-vision-head --filter name=vllm_glm53_head 2>/dev/null" | head -1)
# "RunningFor" examples: "1 minute ago", "23 minutes ago", "2 hours ago", "3 days ago"
to_min() {
  local n
  case "$1" in
    *second*) echo 0 ;;
    *minute*) n=$(echo "$1" | grep -oE '[0-9]+' | head -1); echo "${n:-1}" ;;
    *hour*)   n=$(echo "$1" | grep -oE '[0-9]+' | head -1); echo $(( ${n:-0} * 60 )) ;;
    *)        echo 9999 ;;  # days / unknown => treat as old
  esac
}
a=$(to_min "$age_min"); a=${a:-9999}
if [[ "$a" -lt 35 ]]; then
  echo "[$(ts)] endpoint down but head container ${age_min} (<35 min) — boot in progress, skipping" >> "$LOG"
  exit 0
fi

echo "[$(ts)] ENDPOINT DOWN (head container age: ${age_min:-unknown}) — restarting via cluster.sh takeover" >> "$LOG"
bash "$REPO_DIR/cluster.sh" takeover >> "$LOG" 2>&1
echo "[$(ts)] restart issued — engine needs ~17 min to come back" >> "$LOG"
