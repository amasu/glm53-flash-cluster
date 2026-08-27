#!/usr/bin/env bash
# Build the GLM-5.3-Flash sm_121 patch chain (tonyd2world v1..v8) on the HEAD node,
# then ship the final image to the worker over the QSFP fabric.
#
#   v1 = NoPE-MLA backend for SM121 (FA2 path) on the day-0 image
#   (v2 = env-gated NaN-debug hooks — author's debug side-branch, NOT in the
#    production chain: v3's Dockerfile FROMs sm121-nope-mla, not v2's tag)
#   v3 = FlashInfer 0.6.18 nightly (0.6.17 FA2-MLA NaN at 64-256-row batches)
#   v4 = NCCL re-pin 2.30.7 (nightly sabotages to 2.29.7 -> fabric "internal error")
#   v5 = nvidia-cutlass-dsl re-pin 4.6.2
#   v6 = PDL off on SM12x
#   v7 = indexer top-k hardening (torch.empty -> full(-1) + clamp)
#   v8 = fp8-KV shared-memory tile fix
#   (v9/InstantTensor deliberately skipped: unstable in multi-node TP2)
#
# Author's intermediate tags (each Dockerfile's FROM names the previous stage):
#   v1 -> sm121-nope-mla, v3 -> sm121-fi618, v4 -> sm121-fi618-nccl,
#   v5 -> sm121-final, v6 -> sm121-v6, v7 -> sm121-v7, v8 -> sm121-v8
#
# Prereq: run on the head node, .env next to this script (see example.env).
#         ~1-2 h total (base image pull + nightly pip).
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] && { set -a; source .env; set +a; }
: "${WORKER_IP:?set WORKER_IP in .env (see example.env)}"
IMAGE="${IMAGE:-glm53:v8}"

REPO_DIR="${REPO_DIR:-$HOME/glm53-flash-cluster/tonyd2world-repo}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}"
# production chain: Dockerfile source file -> output tag
STAGES=(
  "Dockerfile.glm53-sm121|sm121-nope-mla"
  "Dockerfile.glm53-sm121-v3|sm121-fi618"
  "Dockerfile.glm53-sm121-v4|sm121-fi618-nccl"
  "Dockerfile.glm53-sm121-v5|sm121-final"
  "Dockerfile.glm53-sm121-v6|sm121-v6"
  "Dockerfile.glm53-sm121-v7|sm121-v7"
  "Dockerfile.glm53-sm121-v8|sm121-v8"
)
FINAL_IMAGE="$IMAGE"
WORKER_SSH="${WORKER_SSH:-ssh $WORKER_IP}"

if [[ ! -f "$REPO_DIR/docker/Dockerfile.glm53-sm121" ]]; then
  echo "==> Cloning tonyd2world/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark"
  git clone https://github.com/tonyd2world/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark "$REPO_DIR"
fi
cd "$REPO_DIR/docker"

echo "==> Ensuring day-0 base image ($BASE_IMAGE, arm64, CUDA 13)"
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"

# Each stage's Dockerfile hardcodes the author's intermediate tag as FROM.
# Rewrite FROM to the previously built stage and build in chain order.
prev="$BASE_IMAGE"
n=${#STAGES[@]}
i=0
for stage in "${STAGES[@]}"; do
  i=$((i+1))
  df="${stage%%|*}"; tag="${stage##*|}"
  echo "==> Stage $i/$n: $tag (FROM $prev)  [$df]"
  sed "s|^FROM .*|FROM $prev|" "$df" > Dockerfile.build
  docker build -f Dockerfile.build -t "glm53:$tag" "$REPO_DIR/docker"
  prev="glm53:$tag"
done

docker tag "$prev" "$FINAL_IMAGE"
echo "==> Shipping $FINAL_IMAGE to worker via $WORKER_SSH (docker save | docker load)"
docker save "$FINAL_IMAGE" | $WORKER_SSH docker load

echo "==> Image identity on both nodes (must match):"
docker images --format '{{.Repository}}:{{.Tag}} | {{.ID}}' | grep "$IMAGE" || true
$WORKER_SSH "docker images --format '{{.Repository}}:{{.Tag}} | {{.ID}}' | grep $IMAGE" || true
