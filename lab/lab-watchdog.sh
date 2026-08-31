#!/usr/bin/env bash
# GLM lab-quant watchdog: probe :8000, restart the stack if the engine is down.
# Runs from the Mac (needs ssh to aitopatom-6253.local). No LLM; script only.
set -uo pipefail
LAB_DIR="/Users/oussama/glm53-flash-cluster/lab"
LOG="/tmp/glm53-lab-watchdog.log"
ts() { date '+%H:%M:%S'; }

if curl -sfS -m 8 http://aitopatom-6253.local:8000/v1/models >/dev/null 2>&1; then
  exit 0
fi

# endpoint down — is the head container freshly started (<35 min => boot in progress)?
age_min=$(ssh -T -o ConnectTimeout=8 -o BatchMode=yes aitopatom-6253.local \
  "docker ps --format '{{.RunningFor}}' --filter name=glm53-lab-head 2>/dev/null" | head -1)
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
ssh -T -o ConnectTimeout=8 -o BatchMode=yes aitopatom-6253.local "docker rm -f glm53-lab-head >/dev/null 2>&1" || true
ssh -T -o ConnectTimeout=8 -o BatchMode=yes aitopatom-6253.local \
  'ssh -o ConnectTimeout=8 10.100.90.4 "docker rm -f glm53-lab-worker >/dev/null 2>&1"' || true
sleep 10
bash ./lab-launch.sh up >> "$LOG" 2>&1
echo "[$(ts)] restart issued — engine needs ~17 min to come back" >> "$LOG"
