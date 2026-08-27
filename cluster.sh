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

preflight() {
  $HEAD_SSH 'ss -ltn | grep -E ":8000\b" && { echo "HEAD: port 8000 BUSY"; exit 1; } ; echo "HEAD: port 8000 free"'
  $WORKER_SSH 'ss -ltn | grep -E ":8000\b" && { echo "WORKER: port 8000 BUSY"; exit 1; } ; echo "WORKER: port 8000 free"'
  $WORKER_SSH "test -f /var/tmp/glm-5.3-flash-nvfp4/config.json && echo 'WORKER: weights staged' || { echo 'WORKER: weights MISSING'; exit 1; }"
  $HEAD_SSH "test -f /var/tmp/glm-5.3-flash-nvfp4/config.json && echo 'HEAD: weights staged' || { echo 'HEAD: weights MISSING'; exit 1; }"
  $WORKER_SSH "docker image inspect glm53:v8 >/dev/null && echo 'WORKER: image present' || { echo 'WORKER: image MISSING'; exit 1; }"
  $HEAD_SSH "docker image inspect glm53:v8 >/dev/null && echo 'HEAD: image present' || { echo 'HEAD: image MISSING'; exit 1; }"
  echo "preflight OK"
}

up() {
  # Pre-launch ritual: drop page caches on both nodes (GB10 unified memory)
  $HEAD_SSH 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
  $WORKER_SSH 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true

  echo "==> Stopping any stale rank containers"
  $HEAD_SSH "docker compose -f $REMOTE_DIR/docker-compose.yml down --remove-orphans 2>/dev/null || true"
  $WORKER_SSH "docker compose -f $REMOTE_DIR/docker-compose.yml down --remove-orphans 2>/dev/null || true"

  echo "==> Launching WORKER (rank 1) first"
  $WORKER_SSH "cd $REMOTE_DIR && docker compose --env-file .env up -d glm53-worker"
  sleep 20
  $WORKER_SSH "docker ps --filter name=vllm_glm53_worker --format '{{.Names}} {{.Status}}'"

  echo "==> Launching HEAD (rank 0)"
  $HEAD_SSH "cd $REMOTE_DIR && docker compose --env-file .env up -d glm53-head"

  echo "==> Both ranks up. Engine takes 14-21 min (warm caches faster)."
  echo "    Watch:  cluster.sh logs"
}

down() {
  # Hard-won rule: tear down BOTH ranks (worker first is irrelevant; both must go)
  $HEAD_SSH "cd $REMOTE_DIR && docker compose --env-file .env down --remove-orphans 2>/dev/null || true"
  $WORKER_SSH "cd $REMOTE_DIR && docker compose --env-file .env down --remove-orphans 2>/dev/null || true"
  echo "both ranks stopped"
}

status() {
  echo "--- containers ---"
  $HEAD_SSH "docker ps -a --filter name=vllm_glm53 --format 'HEAD   {{.Names}} | {{.Status}}'"
  $WORKER_SSH "docker ps -a --filter name=vllm_glm53 --format 'WORKER {{.Names}} | {{.Status}}'"
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
