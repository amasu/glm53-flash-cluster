#!/usr/bin/env bash
# GLM lab-quant watchdog: probe :8000, restart the stack if the engine is down.
# Runs from the orchestrator machine. No LLM; script only.
# Host-agnostic: HEAD_HOST / WORKER_IP / SERVING_PORT from .env or environment.
set -uo pipefail
LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/glm53-lab-watchdog.log"
ts() { date '+%H:%M:%S'; }

if [ -f "$LAB_DIR/../.env" ]; then set -a; . "$LAB_DIR/../.env"; set +a; fi
: "${HEAD_HOST:?HEAD_HOST must be set in .env or environment}"
: "${WORKER_IP:?WORKER_IP must be set in .env or environment}"
PORT="${SERVING_PORT:-8000}"

if curl -sfS -m 8 "http://$HEAD_HOST:$PORT/v1/models" >/dev/null 2>&1; then
  exit 0
fi

# endpoint down — is a lab-stack container freshly started (<35 min => boot in progress)?
age_min=$(ssh -T -o ConnectTimeout=8 -o BatchMode=yes "$HEAD_HOST" \
  "docker ps --format '{{.RunningFor}}' --filter name=glm53-lab-head --filter name=glm53-vision-head 2>/dev/null" | head -1)
# "RunningFor" examples: "1 minute ago", "23 minutes ago", "2 hours ago", "3 days ago"
to_min() {
  case "$1" in
    *second*) echo 0 ;;
    *minute*) echo "$1" | grep -oE '^[0-9]+' ;;
    *hour*)   echo "$1" | grep -oE '^[0-9]+' | awk '{print $1*60}' ;;
    *)        echo 9999 ;;  # days / unknown => treat as old
  esac
}
a=$(to_min "$age_min"); a=${a:-9999}
if [[ "$a" -lt 35 ]]; then
  echo "[$(ts)] endpoint down but head container ${age_min} (<35 min) — boot in progress, skipping" >> "$LOG"
  exit 0
fi

echo "[$(ts)] ENDPOINT DOWN (head container age: ${age_min:-unknown}) — restarting lab stack" >> "$LOG"
cd "$LAB_DIR"
# clean both ranks, then up
ssh -T -o ConnectTimeout=8 -o BatchMode=yes "$HEAD_HOST" "docker rm -f glm53-lab-head >/dev/null 2>&1" || true
ssh -T -o ConnectTimeout=8 -o BatchMode=yes "$HEAD_HOST" \
  "ssh -o ConnectTimeout=8 $WORKER_IP \"docker rm -f glm53-lab-worker >/dev/null 2>&1\"" || true
sleep 10
bash ./lab-launch.sh up >> "$LOG" 2>&1
echo "[$(ts)] restart issued — engine needs ~17 min to come back" >> "$LOG"
