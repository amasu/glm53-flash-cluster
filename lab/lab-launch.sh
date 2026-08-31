#!/usr/bin/env bash
# Take over the GLM endpoint with the lab-quant stack: stop the standing
# LibertAIDAI stack (cluster.sh down), then launch glm53:lab on :8000.
# Single-stack policy: only ONE GLM stack runs on this pair at a time.
#
# Host-agnostic: HEAD_HOST / WORKER_IP / CLUSTER_DIR come from .env (repo
# root, shared with cluster.sh) or real environment variables. Explicit
# environment variables WIN over .env (so one-off overrides like
# LAB_STACK=lab-vision ./lab-launch.sh up work without editing .env).
# LAB_STACK selects the compose file + container/service prefix:
#   LAB_STACK=lab        -> docker-compose-lab.yaml,        glm53-lab-*
#   LAB_STACK=lab-vision -> docker-compose-lab-vision.yaml, glm53-vision-*
#
# Usage: lab-launch.sh <takeover|up|down|status|logs>
#   takeover = old stack down, wait for :8000 free, up
#   up       = launch lab worker->head (assumes :8000 free, i.e. old stack down)
set -euo pipefail
cd "$(dirname "$0")"

# Explicit env wins over .env: remember preset values, re-apply after sourcing
_LAB_STACK_PRESET="${LAB_STACK-}"

# .env resolution: explicit LAB_ENV_FILE, then ./ .env, then ../.env
ENV_FILE="${LAB_ENV_FILE:-}"
if [ -z "$ENV_FILE" ]; then
  for c in ".env" "../.env"; do
    [ -f "$c" ] && ENV_FILE="$c" && break
  done
fi
if [ -n "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi
[ -n "$_LAB_STACK_PRESET" ] && LAB_STACK="$_LAB_STACK_PRESET"
: "${HEAD_HOST:?HEAD_HOST must be set in .env or environment}"
: "${WORKER_IP:?WORKER_IP must be set in .env or environment}"
# CLUSTER_DIR: only used by `takeover` (which stops the old stack via
# cluster.sh on the orchestrator). Default to the repo's usual location so
# the other subcommands don't require it to be set.
CLUSTER_DIR="${CLUSTER_DIR:-$HOME/glm53-flash-cluster}"

case "${LAB_STACK:-lab}" in
  lab)        COMPOSE_FILE="docker-compose-lab.yaml"; SVC_PREFIX="glm53-lab" ;;
  lab-vision) COMPOSE_FILE="docker-compose-lab-vision.yaml"; SVC_PREFIX="glm53-vision" ;;
  *) echo "unknown LAB_STACK: ${LAB_STACK} (want lab or lab-vision)" >&2; exit 2 ;;
esac
# Node-side compose resolves .env relative to its cwd: ~/glm53-lab/.env on both
# nodes (a copy of the cluster .env — see deploy notes). The orchestrator only
# sources .env for variable resolution; it never runs compose itself.
COMPOSE="docker compose --env-file .env -f $COMPOSE_FILE"

run_worker() {
  local b64
  b64=$(printf '%s' "$1" | base64 | tr -d '\n')
  ssh -T "$HEAD_HOST" "ssh -T -o StrictHostKeyChecking=no $WORKER_IP 'echo \"$b64\" | base64 -d | bash'"
}

wait_port_free() {
  local i
  for i in $(seq 1 60); do
    if ssh -T "$HEAD_HOST" "ss -ltn | grep -q ':${SERVING_PORT:-8000}\b'" ; then
      echo "port ${SERVING_PORT:-8000} still busy (rank $i/60, 10s)"
      sleep 10
    else
      echo "port ${SERVING_PORT:-8000} free"
      return 0
    fi
  done
  echo "FATAL: port ${SERVING_PORT:-8000} still busy after 10 min" >&2
  return 1
}

case "${1:-}" in
  takeover)
    echo "==> Stopping standing GLM stack (cluster.sh down)"
    bash "$CLUSTER_DIR/cluster.sh" down
    echo "==> Waiting for :${SERVING_PORT:-8000} to free"
    wait_port_free
    echo "==> Dropping page caches"
    ssh -T "$HEAD_HOST" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
    run_worker 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
    "$0" up
    ;;
  up)
    echo "==> Preflight"
    ssh -T "$HEAD_HOST" "ss -ltn | grep -E ':${SERVING_PORT:-8000}\b' && { echo 'HEAD: port ${SERVING_PORT:-8000} BUSY — old stack still up?'; exit 1; }; echo 'head: ${SERVING_PORT:-8000} free'"
    run_worker "ss -ltn | grep -E ':${SERVING_PORT:-8000}\b' && { echo 'WORKER: port ${SERVING_PORT:-8000} BUSY'; exit 1; }; echo 'worker: ${SERVING_PORT:-8000} free'"
    ssh -T "$HEAD_HOST" "test -f ${WEIGHTS_DIR_LAB:?WEIGHTS_DIR_LAB must be set in .env}/config.json && echo 'head: weights staged'"
    run_worker "test -f ${WEIGHTS_DIR_LAB:?WEIGHTS_DIR_LAB must be set in .env}/config.json && echo 'worker: weights staged'"
    ssh -T "$HEAD_HOST" "docker image inspect ${LAB_IMAGE:-glm53:lab} >/dev/null && echo 'head: image ok'"
    run_worker "docker image inspect ${LAB_IMAGE:-glm53:lab} >/dev/null && echo 'worker: image ok'"

    echo "==> Launching WORKER (rank 1) first"
    run_worker "cd ~/glm53-lab && $COMPOSE up -d ${SVC_PREFIX}-worker"
    sleep 20
    run_worker "docker ps --filter name=${SVC_PREFIX}-worker --format '{{.Names}} {{.Status}}'"

    echo "==> Launching HEAD (rank 0)"
    ssh -T "$HEAD_HOST" "cd ~/glm53-lab && $COMPOSE up -d ${SVC_PREFIX}-head"
    echo "==> Both ranks up. Engine takes 14-21 min. Watch: $0 logs"
    ;;
  down)
    # Single-stack policy: clear ANY lab stack container on both ranks
    ssh -T "$HEAD_HOST" "docker rm -f glm53-lab-head glm53-vision-head 2>/dev/null || true"
    run_worker "docker rm -f glm53-lab-worker glm53-vision-worker 2>/dev/null || true"
    echo "lab stack(s) stopped"
    ;;
  status)
    ssh -T "$HEAD_HOST" "docker ps -a --filter name=glm53-lab --filter name=glm53-vision --format 'HEAD   {{.Names}} | {{.Status}}'"
    run_worker "docker ps -a --filter name=glm53-lab --filter name=glm53-vision --format 'WORKER {{.Names}} | {{.Status}}'"
    curl -fsS "http://$HEAD_HOST:${SERVING_PORT:-8000}/v1/models" 2>/dev/null | head -c 400 && echo || echo "endpoint :${SERVING_PORT:-8000} not answering yet"
    ;;
  logs)
    ssh -T "$HEAD_HOST" "docker logs -f --tail 40 ${SVC_PREFIX}-head"
    ;;
  *)
    echo "usage: lab-launch.sh <takeover|up|down|status|logs>"; exit 2
    ;;
esac
