#!/usr/bin/env bash
# Orchestrate the GLM-5.3-Flash vLLM TP=2 cluster from the Mac.
# The head and worker run the same docker-compose.yml; launch order is enforced
# here (worker rank 1 first, head rank 0 second) per the mp executor's requirement.
#
# usage: cluster.sh <preflight|up|down|status|logs>
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

HEAD_SSH="ssh $HEAD_HOST"
WORKER_SSH="ssh $HEAD_HOST ssh $WORKER_IP"

# Correct two-hop runner: base64-encode the command so it survives both ssh
# hops verbatim (no shell splitting at the first hop; sshd joins argv with
# spaces, so any unquoted multi-token command would be mangled on the head).
# base64 alphabet [A-Za-z0-9+/=] is shell-inert, so the inner single quotes hold.
run_worker() {
  local b64
  b64=$(printf '%s' "$1" | base64 | tr -d '\n')
  # Mac expands $b64 (defined here) into the string; inner single-quotes keep it
  # one token across the head hop; worker runs: echo <b64> | base64 -d | bash
  ssh -T "$HEAD_HOST" "ssh -T $WORKER_IP 'echo \"$b64\" | base64 -d | bash'"
}

preflight() {
  $HEAD_SSH 'ss -ltn | grep -E ":8000\b" && { echo "HEAD: port 8000 BUSY"; exit 1; } ; echo "HEAD: port 8000 free"'
  run_worker 'ss -ltn | grep -E ":8000\b" && { echo "WORKER: port 8000 BUSY"; exit 1; } ; echo "WORKER: port 8000 free"'
  run_worker "test -f /var/tmp/glm-5.3-flash-nvfp4/config.json && echo 'WORKER: weights staged' || { echo 'WORKER: weights MISSING'; exit 1; }"
  $HEAD_SSH "test -f /var/tmp/glm-5.3-flash-nvfp4/config.json && echo 'HEAD: weights staged' || { echo 'HEAD: weights MISSING'; exit 1; }"
  run_worker "docker image inspect glm53:v8 >/dev/null && echo 'WORKER: image present' || { echo 'WORKER: image MISSING'; exit 1; }"
  $HEAD_SSH "docker image inspect glm53:v8 >/dev/null && echo 'HEAD: image present' || { echo 'HEAD: image MISSING'; exit 1; }"
  echo "preflight OK"
}

up() {
  # Pre-launch ritual: drop page caches on both nodes (GB10 unified memory)
  $HEAD_SSH 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
  run_worker 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true

  echo "==> Stopping any stale rank containers"
  $HEAD_SSH "cd $REMOTE_DIR && docker compose --env-file .env down --remove-orphans 2>/dev/null || true"
  run_worker "cd $REMOTE_DIR && docker compose --env-file .env down --remove-orphans 2>/dev/null || true"

  echo "==> Launching WORKER (rank 1) first"
  run_worker "cd $REMOTE_DIR && docker compose --env-file .env up -d glm53-worker"
  sleep 20
  run_worker "docker ps --filter name=vllm_glm53_worker --format '{{.Names}} {{.Status}}'"

  echo "==> Launching HEAD (rank 0)"
  $HEAD_SSH "cd $REMOTE_DIR && docker compose --env-file .env up -d glm53-head"

  echo "==> Both ranks up. Engine takes 14-21 min (warm caches faster)."
  echo "    Watch:  cluster.sh logs"
}

down() {
  # Hard-won rule: tear down BOTH ranks (worker first is irrelevant; both must go)
  $HEAD_SSH "cd $REMOTE_DIR && docker compose --env-file .env down --remove-orphans 2>/dev/null || true"
  run_worker "cd $REMOTE_DIR && docker compose --env-file .env down --remove-orphans 2>/dev/null || true"
  echo "both ranks stopped"
}

status() {
  echo "--- containers ---"
  $HEAD_SSH "docker ps -a --filter name=vllm_glm53 --format 'HEAD   {{.Names}} | {{.Status}}'"
  run_worker "docker ps -a --filter name=vllm_glm53 --format 'WORKER {{.Names}} | {{.Status}}'"
  echo "--- endpoint ---"
  curl -fsS "http://$HEAD_HOST:8000/v1/models" 2>/dev/null && echo || echo "endpoint not answering yet"
}

logs() {
  $HEAD_SSH "docker logs -f --tail 40 vllm_glm53_head"
}

case "${1:-}" in
  preflight|up|down|status|logs) "$1" ;;
  *) echo "usage: cluster.sh <preflight|up|down|status|logs>" >&2; exit 1 ;;
esac
