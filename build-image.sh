#!/usr/bin/env bash
# Build the GLM-5.3-Flash sm_121 patch chain (tonyd2world v1..v8) on the HEAD node,
# then ship glm53:v8 to the worker over the QSFP fabric.
#
#   v1 = NoPE-MLA backend for SM121 (FA2 path) on the day-0 image
#   v2 = env-gated NaN debug hooks
#   v3 = FlashInfer 0.6.18 nightly (0.6.17 FA2-MLA NaN at 64-256-row batches)
#   v4 = NCCL re-pin 2.30.7 (nightly sabotages to 2.29.7 -> fabric "internal error")
#   v5 = nvidia-cutlass-dsl re-pin 4.6.2
#   v6 = PDL off on SM12x
#   v7 = indexer top-k hardening (torch.empty -> full(-1) + clamp)
#   v8 = fp8-KV shared-memory tile fix
#   (v9/InstantTensor deliberately skipped: unstable in multi-node TP2)
#
# Prereq: run on the head node. ~1-2 h total (base image pull + nightly pip).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/glm53-flash-cluster/tonyd2world-repo}"
BASE_IMAGE="vllm/vllm-openai:glm53-flash-arm64-cu130"
STAGE_TAGS=(sm121-nope-mla sm121-fi618 sm121-fi618-nccl sm121-final sm121-v7 sm121-v8)
FINAL_IMAGE="glm53:v8"
WORKER_SSH="${WORKER_SSH:-ssh 10.100.90.4}"

if [[ ! -f "$REPO_DIR/docker/Dockerfile.glm53-sm121" ]]; then
  echo "==> Cloning tonyd2world/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark"
  git clone https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark "$REPO_DIR"
fi
cd "$REPO_DIR/docker"

echo "==> Pulling day-0 base image ($BASE_IMAGE, arm64, CUDA 13)"
docker pull "$BASE_IMAGE"

# Each stage's Dockerfile hardcodes the author's intermediate tag as FROM.
# Rewrite FROM to the previously built stage and build v1 -> v8 in order.
prev="$BASE_IMAGE"
for i in "${!STAGE_TAGS[@]}"; do
  tag="${STAGE_TAGS[$i]}"
  if [[ $i -eq 0 ]]; then df="$REPO_DIR/docker/Dockerfile.glm53-sm121"; else df="$REPO_DIR/docker/Dockerfile.glm53-sm121-v$((i+1))"; fi
  echo "==> Stage $((i+1))/6: $tag (FROM $prev)  [$(basename "$df")]"
  sed "s|^FROM .*|FROM $prev|" "$df" > Dockerfile.build
  docker build -f Dockerfile.build -t "glm53:$tag" "$REPO_DIR/docker"
  prev="glm53:$tag"
done

docker tag "$prev" "$FINAL_IMAGE"
echo "==> Shipping $FINAL_IMAGE to worker via $WORKER_SSH (docker save | docker load)"
docker save "$FINAL_IMAGE" | $WORKER_SSH docker load

echo "==> Image identity on both nodes (must match):"
docker images --format '{{.Repository}}:{{.Tag}} | {{.ID}}' | grep '^glm53:v8' || true
$WORKER_SSH "docker images --format '{{.Repository}}:{{.Tag}} | {{.ID}}' | grep '^glm53:v8'" || true
