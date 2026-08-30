#!/usr/bin/env bash
# Take over the GLM endpoint with the lab-quant stack: stop the standing
# LibertAIDAI stack (cluster.sh down), then launch glm53:lab on :8000.
# Single-stack policy: only ONE GLM stack runs on this pair at a time.
#
# Usage: lab-launch.sh <takeover|up|down|status|logs>
#   takeover = old stack down, wait for :8000 free, up
#   up       = launch lab worker->head (assumes :8000 free, i.e. old stack down)
set -euo pipefail
cd "$(dirname "$0")"

HEAD_HOST=aitopatom-6253.local
WORKER_IP=10.100.90.4
CLUSTER_DIR="$HOME/glm53-flash-cluster"
COMPOSE="docker compose -f docker-compose-lab.yaml"

run_worker() {
  local b64
  b64=$(printf '%s' "$1" | base64 | tr -d '\n')
  ssh -T "$HEAD_HOST" "ssh -T -o StrictHostKeyChecking=no $WORKER_IP 'echo \"$b64\" | base64 -d | bash'"
}

wait_port_free() {
  local i
  for i in $(seq 1 60); do
    if ssh -T "$HEAD_HOST" "ss -ltn | grep -q ':8000\\b'" ; then
      echo "port 8000 still busy (rank $i/60, 10s)"
      sleep 10
    else
      echo "port 8000 free"
      return 0
    fi
  done
  echo "FATAL: port 8000 still busy after 10 min" >&2
  return 1
}

case "${1:-}" in
  takeover)
    echo "==> Stopping standing GLM stack (cluster.sh down)"
    bash "$CLUSTER_DIR/cluster.sh" down
    echo "==> Waiting for :8000 to free"
    wait_port_free
    echo "==> Dropping page caches"
    ssh -T "$HEAD_HOST" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
    run_worker 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
    "$0" up
    ;;
  up)
    echo "==> Preflight"
    ssh -T "$HEAD_HOST" "ss -ltn | grep -E ':8000\\b' && { echo 'HEAD: port 8000 BUSY — old stack still up?'; exit 1; }; echo 'head: 8000 free'"
    run_worker "ss -ltn | grep -E ':8000\\b' && { echo 'WORKER: port 8000 BUSY'; exit 1; }; echo 'worker: 8000 free'"
    ssh -T "$HEAD_HOST" "test -f /var/tmp/glm-5.3-flash-lab-nvfp4/config.json && echo 'head: weights staged'"
    run_worker "test -f /var/tmp/glm-5.3-flash-lab-nvfp4/config.json && echo 'worker: weights staged'"
    ssh -T "$HEAD_HOST" "docker image inspect glm53:lab >/dev/null && echo 'head: image ok'"
    run_worker "docker image inspect glm53:lab >/dev/null && echo 'worker: image ok'"

    echo "==> Launching WORKER (rank 1) first"
    run_worker "cd ~/glm53-lab && $COMPOSE up -d glm53-lab-worker"
    sleep 20
    run_worker "docker ps --filter name=glm53-lab-worker --format '{{.Names}} {{.Status}}'"

    echo "==> Launching HEAD (rank 0)"
    ssh -T "$HEAD_HOST" "cd ~/glm53-lab && $COMPOSE up -d glm53-lab-head"
    echo "==> Both ranks up. Engine takes 14-21 min. Watch: $0 logs"
    ;;
  down)
    ssh -T "$HEAD_HOST" "cd ~/glm53-lab && $COMPOSE down --remove-orphans 2>/dev/null || true"
    run_worker "cd ~/glm53-lab && $COMPOSE down --remove-orphans 2>/dev/null || true"
    echo "lab stack stopped"
    ;;
  status)
    ssh -T "$HEAD_HOST" "docker ps -a --filter name=glm53-lab --format 'HEAD   {{.Names}} | {{.Status}}'"
    run_worker "docker ps -a --filter name=glm53-lab --format 'WORKER {{.Names}} | {{.Status}}'"
    curl -fsS http://$HEAD_HOST:8000/v1/models 2>/dev/null && echo || echo "endpoint :8000 not answering yet"
    ;;
  logs)
    ssh -T "$HEAD_HOST" "docker logs -f --tail 40 glm53-lab-head"
    ;;
  *)
    echo "usage: lab-launch.sh <takeover|up|down|status|logs>"; exit 2
    ;;
esac
