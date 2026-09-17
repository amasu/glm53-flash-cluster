#!/usr/bin/env bash
# Run on the HEAD node (or orchestrated from the orchestrator via cluster.sh).
# Two entry points:
#
#   fetch-weights.sh                       LibertAIDAI/GLM-5.3-Flash-NVFP4
#                                          (182 GiB) -> WEIGHTS_DIR + worker
#   fetch-weights.sh nvidia-nvfp4          NVIDIA official NVFP4 weights
#                                          (nvidia/GLM-5.3-Flash-NVFP4 @
#                                          $NVFP4_REVISION, 33 shards) ->
#                                          WEIGHTS_DIR_NVFP4, PLUS the
#                                          mandatory DFlash2 draft
#                                          (incoai/GLM-5.3-Flash-DFlash2) ->
#                                          DRAFT_DIR/$DRAFT_NAME, PLUS the
#                                          pilcothink runtime image shipped
#                                          head -> worker over the fabric
#                                          (all on the fabric, verified).
#
# Both are resumable: present files/shards are skipped, rsync is delta-aware.
# Host-agnostic: source .env (see example.env) or export the variables first.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] && { set -a; source .env; set +a; }
: "${WORKER_IP:?set WORKER_IP in .env (see example.env)}"

VENV="/tmp/hfvenv"
setup_hf() {
  if [[ ! -x "$VENV/bin/hf" ]]; then
    echo "==> setting up hf CLI in $VENV"
    python3 -m venv "$VENV" 2>/dev/null \
      || { apt-get install -y -q python3-venv || sudo apt-get install -y -q python3-venv; python3 -m venv "$VENV"; }
    "$VENV/bin/pip" install -q -U pip "huggingface_hub[cli]"
  fi
}

ship_image_to_worker() {
  local image="$1"
  if docker image inspect "$image" >/dev/null 2>&1; then
    local head_id worker_id
    head_id=$(docker image inspect "$image" --format '{{.Id}}')
    worker_id=$(ssh -o BatchMode=yes "$WORKER_IP" "docker image inspect $image --format '{{.Id}}' 2>/dev/null" || true)
    if [[ "$head_id" = "$worker_id" && -n "$head_id" ]]; then
      echo "==> worker already has $image ($head_id)"
      return
    fi
  fi
  echo "==> transferring $image over the fabric (docker save | ssh docker load)"
  docker pull "$image" 2>/dev/null || true
  docker save "$image" | ssh -o BatchMode=yes "$WORKER_IP" "docker load" | tail -2
  ssh -o BatchMode=yes "$WORKER_IP" "docker image inspect $image >/dev/null && echo 'worker: $image present'"
}

case "${1:-libertai}" in
libertai)
  MODEL_ID="${MODEL_ID:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
  DIR="${WEIGHTS_DIR:-/var/tmp/glm-5.3-flash-nvfp4}"
  EXPECTED_SHARDS=120
  FIRST=00001; LAST=00120
  ;;
nvidia-nvfp4)
  MODEL_ID="${MODEL_ID_NVFP4:-nvidia/GLM-5.3-Flash-NVFP4}"
  REVISION="${NVFP4_REVISION:-09b04e5e74bca08ca8549fc736d4cdd8624bfde3}"
  DRAFT_REPO="${DRAFT_REPO:-incoai/GLM-5.3-Flash-DFlash2}"
  IMAGE="${NVFP4_IMAGE:-pilcothink/vllm_spark_glm53:0.28}"
  DIR="${WEIGHTS_DIR_NVFP4:?WEIGHTS_DIR_NVFP4 must be set in .env}"
  DRAFT_ROOT="${DRAFT_DIR:?DRAFT_DIR must be set in .env}"
  # recipe knobs (incl. DRAFT_NAME) come from the stack file — the same
  # single source of truth cluster.sh uses at launch time
  set -a; . ./stacks/nvfp4-dflash2.env; set +a
  EXPECTED_SHARDS=33
  FIRST=00001; LAST=00033
  ;;
*)
  echo "usage: fetch-weights.sh [libertai|nvidia-nvfp4]" >&2
  exit 1
  ;;
esac

setup_hf
mkdir -p "$DIR"
echo "==> downloading $MODEL_ID -> $DIR (resume-safe; ~${EXPECTED_SHARDS} shards)"
if [[ -n "${REVISION:-}" ]]; then
  HF_HUB_DOWNLOAD_TIMEOUT=60 "$VENV/bin/hf" download "$MODEL_ID" --revision "$REVISION" --local-dir "$DIR"
else
  HF_HUB_DOWNLOAD_TIMEOUT=60 "$VENV/bin/hf" download "$MODEL_ID" --local-dir "$DIR"
fi

echo "==> checksum anchors (head)"
cd "$DIR"
sha256sum config.json generation_config.json "model-$FIRST-of-$(printf '%05d' "$EXPECTED_SHARDS").safetensors" "model-$LAST-of-$(printf '%05d' "$EXPECTED_SHARDS").safetensors" > /tmp/glm53-checksums.txt
cat /tmp/glm53-checksums.txt
ls "$DIR"/*.safetensors | wc -l | grep -q "^${EXPECTED_SHARDS}$" || { echo "FATAL: expected $EXPECTED_SHARDS shards, got $(ls "$DIR"/*.safetensors | wc -l)" >&2; exit 1; }

echo "==> rsync to worker over fabric"
ssh "$WORKER_IP" "mkdir -p $DIR"
rsync -a --info=progress2 "$DIR/" "$WORKER_IP:$DIR/"

echo "==> rsync checksum anchors to worker (real verification, not just a count)"
rsync -a /tmp/glm53-checksums.txt "$WORKER_IP:/tmp/glm53-checksums.txt"

echo "==> verify worker copy (sha256sum -c must PASS; set -e aborts on mismatch)"
ssh "$WORKER_IP" "cd $DIR && sha256sum -c /tmp/glm53-checksums.txt && ls *.safetensors | wc -l && du -sh $DIR"

# --- nvidia-nvfp4 extras: the DFlash2 draft + the runtime image -------------
if [[ "${1:-libertai}" == "nvidia-nvfp4" ]]; then
  DRAFT="$DRAFT_ROOT/$DRAFT_NAME"
  echo "==> fetching $DRAFT_REPO -> $DRAFT (resume-safe; ~2.2 GiB)"
  mkdir -p "$DRAFT"
  HF_HUB_DOWNLOAD_TIMEOUT=60 "$VENV/bin/hf" download "$DRAFT_REPO" --local-dir "$DRAFT"
  test -f "$DRAFT/config.json" || { echo "FATAL: draft incomplete at $DRAFT" >&2; exit 1; }

  echo "==> rsync draft to worker at the SAME path"
  ssh "$WORKER_IP" "mkdir -p $DRAFT_ROOT"
  rsync -a --info=progress2 "$DRAFT/" "$WORKER_IP:$DRAFT/"
  ssh "$WORKER_IP" "test -f \"$DRAFT/config.json\" && echo \"worker: draft staged at $DRAFT\" || { echo 'FATAL: worker draft missing'; exit 1; }"

  echo "==> runtime image $IMAGE"
  ship_image_to_worker "$IMAGE"

  echo "==> nvidia-nvfp4 staging complete"
  echo "    weights: $DIR (both nodes)"
  echo "    draft  : $DRAFT (both nodes)"
  echo "    image  : $IMAGE (both nodes)"
  echo "    next   : cluster.sh nvfp4-dflash2 up   (port \$SERVING_PORT=8000, master \$MASTER_PORT_NVFP4)"
fi
