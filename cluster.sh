#!/usr/bin/env bash
# Orchestrate the GLM-5.3-Flash vLLM TP=2 cluster from the orchestrator (e.g.
# a Mac). ONE compose file (docker-compose.yml) + ONE entrypoint
# (exec-vllm.sh) serve every stack: the stack is pure configuration, sourced
# from stacks/<STACK>.env — on the orchestrator AND on each node, so every
# stack runs the exact same code with only different variables.
#
# Host-agnostic: site values come from .env (see example.env).
#
# usage: cluster.sh [STACK] <preflight|mirror|up|down|status|logs|takeover>
# STACK   optional; default from STACK in .env (currently: lab-vision)
#           available: lab lab-vision v9-512k v9-262k-fp8 v8-262k
#   mirror   = rsync repo (incl. .env + stacks/) to both nodes at REMOTE_DIR
#   takeover = down (any stale stack) + drop caches -> up <STACK>
#   NOTE: SINGLE-STACK POLICY — one profile runs at a time; `down` removes
#         any glm53 rank containers regardless of which stack they belong to.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "FATAL: no .env here — copy example.env to .env and fill it in" >&2; exit 1; }
set -a; source .env; set +a

: "${HEAD_HOST:?set HEAD_HOST in .env (see example.env)}"
: "${WORKER_IP:?set WORKER_IP in .env (see example.env)}"
: "${REMOTE_DIR:?set REMOTE_DIR in .env (see example.env)}"
SERVING_PORT="${SERVING_PORT:-8000}"

# -------- stack resolution --------------------------------------------------
# STACK arg (e.g. `cluster.sh v9-512k up`) wins over .env; default = lab-vision
# (the actual production profile; see the running glm53-vision-* containers)
STACK="${STACK:-lab-vision}"
if [[ "${1:-}" =~ ^(lab|lab-vision|v9-512k|v9-262k-fp8|v8-262k)$ ]]; then
  STACK="$1"; shift
fi
STACK_FILE="stacks/${STACK}.env"
[[ -f "$STACK_FILE" ]] || { echo "FATAL: unknown/missing stack '$STACK' (want one of: lab lab-vision v9-512k v9-262k-fp8 v8-262k; file $STACK_FILE)" >&2; exit 1; }

# Layering: site .env (already sourced) <- stack file (sourced later, wins:
# pins its knobs + maps its image/weights/ports). The stack file may
# reference site vars (e.g. WEIGHTS_DIR_LAB from .env) via ${VAR:?}
# expansions, which fail fast here if the site value is missing.
set -a; source "$STACK_FILE"; set +a

echo "==> stack: $STACK (file: $STACK_FILE)"
echo "    image=$IMAGE weights=$WEIGHTS_DIR master_port=$MASTER_PORT"

HEAD_SSH="ssh $HEAD_HOST"

# Correct two-hop runner: base64-encode the command so it survives both ssh
# hops verbatim (no shell splitting at the first hop; sshd joins argv with
# spaces, so any unquoted multi-token command would be mangled on the head).
# base64 alphabet [A-Za-z0-9+/=] is shell-inert, so the inner single quotes hold.
run_worker() {
  local b64
  b64=$(printf '%s' "$1" | base64 | tr -d '\n')
  ssh -T "$HEAD_HOST" "ssh -T $WORKER_IP 'echo \"$b64\" | base64 -d | bash'"
}

preflight() {
  $HEAD_SSH "ss -ltn | grep -E \":$SERVING_PORT\\b\" && { echo \"HEAD: port $SERVING_PORT BUSY\"; exit 1; } ; echo \"HEAD: port $SERVING_PORT free\""
  run_worker "ss -ltn | grep -E \":$SERVING_PORT\\b\" && { echo \"WORKER: port $SERVING_PORT BUSY\"; exit 1; } ; echo \"WORKER: port $SERVING_PORT free\""
  $HEAD_SSH "cd '$REMOTE_DIR' && test -f '$STACK_FILE' || { echo 'HEAD: $STACK_FILE MISSING — run: cluster.sh mirror'; exit 1; }"
  run_worker "cd '$REMOTE_DIR' && test -f '$STACK_FILE' || { echo 'WORKER: $STACK_FILE MISSING — run: cluster.sh mirror'; exit 1; }"
  run_worker "test -f \"$WEIGHTS_DIR/config.json\" && echo 'WORKER: weights staged' || { echo 'WORKER: weights MISSING'; exit 1; }"
  $HEAD_SSH "test -f \"$WEIGHTS_DIR/config.json\" && echo 'HEAD: weights staged' || { echo 'HEAD: weights MISSING'; exit 1; }"
  run_worker "docker image inspect \"$IMAGE\" >/dev/null && echo 'WORKER: image present' || { echo 'WORKER: image MISSING'; exit 1; }"
  $HEAD_SSH "docker image inspect \"$IMAGE\" >/dev/null && echo 'HEAD: image present' || { echo 'HEAD: image MISSING'; exit 1; }"
  echo "preflight OK [$STACK]"
}

# Local junk that must NOT reach the nodes: the .git dir, build-image.sh's
# multi-GB vendored author repo (it clones to $REPO_DIR *inside* this repo's
# dir), gitignored artifact dirs, logs. .env IS mirrored on purpose (README).
RSYNC_EXCLUDES="--exclude .git/ --exclude tonyd2wild-repo/ --exclude data/ --exclude runs/ --exclude __pycache__/ --exclude '*.log'"

mirror() {
  echo "==> Mirroring repo (incl. .env + stacks/) to head $HEAD_HOST:$REMOTE_DIR"
  $HEAD_SSH "mkdir -p '$REMOTE_DIR'"
  rsync -a $RSYNC_EXCLUDES ./ "$HEAD_HOST:$REMOTE_DIR/"
  echo "==> Mirroring repo (incl. .env + stacks/) to worker $WORKER_IP:$REMOTE_DIR (via head)"
  run_worker "mkdir -p '$REMOTE_DIR'"
  # Ship head -> worker: source .env on the head for WORKER_IP (not in login env)
  $HEAD_SSH "cd '$REMOTE_DIR' && set -a; . ./.env; set +a; rsync -a $RSYNC_EXCLUDES . \"\$WORKER_IP:$REMOTE_DIR/\""
  echo "==> mirrored to both nodes"
}

# Node-side sourcing, used inside an explicitly fail-fast `bash` (see up()).
# The stack file is sourced AFTER the site .env so the stack pins its own
# knobs; `set -a` exports them so compose's bare-name passthrough picks them up.
NODE_SOURCING="set -a; . ./.env; . ./$STACK_FILE; set +a"

up() {
  # Launch order is load-bearing (mp executor rendezvous): worker rank 1
  # first, then head rank 0. Each rank's command is self-contained and
  # fail-fast (set -euo pipefail) so a missing dir/file aborts instead of
  # launching compose from the wrong place.
  echo "==> Launching WORKER (rank 1) first"
  run_worker "set -euo pipefail; cd '$REMOTE_DIR' && $NODE_SOURCING && docker compose up -d glm53-worker"
  sleep 20
  run_worker "docker ps --filter name=glm53-worker --format '{{.Names}} {{.Status}}'"

  echo "==> Launching HEAD (rank 0)"
  $HEAD_SSH "set -euo pipefail; cd '$REMOTE_DIR' && $NODE_SOURCING && docker compose up -d glm53-head"

  echo "==> Both ranks up [$STACK]. Engine takes 14-21 min (warm caches faster)."
  echo "    Watch:  cluster.sh $STACK logs"
}

down() {
  # Hard-won rule: tear down BOTH ranks (worker first is irrelevant; both must
  # go). Single-stack policy: target the known container names so any
  # previously-running stack (new or legacy) is cleared without knowing which
  # one it was.
  $HEAD_SSH "docker rm -f glm53-head glm53-lab-head glm53-vision-head vllm_glm53_head vllm_glm53_lab_head vllm_glm53_vision_head 2>/dev/null | true"
  run_worker "docker rm -f glm53-worker glm53-lab-worker glm53-vision-worker vllm_glm53_worker vllm_glm53_lab_worker vllm_glm53_vision_worker 2>/dev/null | true"
  echo "both ranks stopped"
}

takeover() {
  echo "==> Takeover: stopping any stale stack, then bringing up [$STACK]"
  down
  echo "==> Waiting for :$SERVING_PORT to free"
  local i
  for i in $(seq 1 60); do
    if $HEAD_SSH "ss -ltn | grep -q ':${SERVING_PORT}\\b'"; then
      echo "port $SERVING_PORT still busy (round $i/60, 10s)"; sleep 10
    else
      echo "port $SERVING_PORT free"; break
    fi
  done
  echo "==> Dropping page caches"
  $HEAD_SSH 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
  run_worker 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
  up
}

status() {
  echo "--- containers [$STACK] ---"
  $HEAD_SSH "docker ps -a --filter name=glm53 --format 'HEAD   {{.Names}} | {{.Status}}'"
  run_worker "docker ps -a --filter name=glm53 --format 'WORKER {{.Names}} | {{.Status}}'"
  echo "--- endpoint ---"
  curl -fsS "http://$HEAD_HOST:$SERVING_PORT/v1/models" 2>/dev/null | head -c 400 && echo || echo "endpoint not answering yet"
}

logs() {
  $HEAD_SSH "docker logs -f --tail 40 glm53-head"
}

case "${1:-}" in
  preflight) preflight ;;
  mirror)    mirror ;;
  up)        up ;;
  down)      down ;;
  status)    status ;;
  logs)      logs ;;
  takeover)  takeover ;;
  *) echo "usage: cluster.sh [STACK] <preflight|mirror|up|down|status|logs|takeover>" >&2
     echo "  STACK: lab | lab-vision | v9-512k | v9-262k-fp8 | v8-262k (default: \$STACK from .env = lab-vision)" >&2
     exit 1 ;;
esac
