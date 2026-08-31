#!/usr/bin/env bash
# Build the lab-quant image (glm53:lab) from the vendored build context in
# lab/build/, on the HEAD node, then ship it to the WORKER node.
#
# Host-agnostic: HEAD_HOST / WORKER_IP / REMOTE_DIR from .env; image tag from
# LAB_IMAGE (default glm53:lab). Run from the orchestrator (Mac or any machine
# with ssh to the head). The build is fast (~30 s): the day-0 base is pulled
# by digest and only the patch layer is rebuilt. All patches apply with zero
# fuzz against the pinned base — provenance in lab/build/UPSTREAM.md.
#
# To build while sitting ON the head node instead:
#   cd lab/build && docker build -t glm53:lab . && ssh $WORKER_IP 'docker load' < <(docker save glm53:lab)
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

if [ -f .env ]; then set -a; . .env; set +a; fi
: "${HEAD_HOST:?HEAD_HOST must be set in .env or environment}"
: "${WORKER_IP:?WORKER_IP must be set in .env or environment}"
: "${REMOTE_DIR:?REMOTE_DIR must be set in .env or environment}"
LAB_IMAGE="${LAB_IMAGE:-glm53:lab}"
BUILD_DIR_NAME="lab/build"
REMOTE_BUILD_DIR="$REMOTE_DIR/$BUILD_DIR_NAME"

# Inner script executed ON THE HEAD: build, self-verify markers, ship to worker.
read -r -d '' INNER <<EOF || true
set -euo pipefail
cd '${REMOTE_BUILD_DIR}'
echo '==> Building ${LAB_IMAGE} from lab/build/ (digest-pinned base)'
docker build -t '${LAB_IMAGE}' .
echo '==> Verifying patch markers in the built image'
docker run --rm '${LAB_IMAGE}' bash -lc \
  'grep -q quantprobe /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py && grep -q hc_attn_base /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py && echo markers ok' \
  > /dev/null && echo 'markers ok'
echo '==> Shipping ${LAB_IMAGE} to ${WORKER_IP}'
docker save '${LAB_IMAGE}' | ssh -T -o StrictHostKeyChecking=no '${WORKER_IP}' 'docker load'
echo '==> Done: ${LAB_IMAGE} on both nodes'
EOF
INNER_B64=$(printf '%s' "$INNER" | base64 | tr -d '\n')

echo "==> Mirroring lab/build/ to the head at ${REMOTE_BUILD_DIR}"
ssh -T "$HEAD_HOST" "mkdir -p '${REMOTE_BUILD_DIR}'"
rsync -a ./lab/build/ "$HEAD_HOST:$REMOTE_BUILD_DIR/"

echo "==> Building on the head (this may pull the day-0 base first time)"
ssh -T "$HEAD_HOST" "echo '$INNER_B64' | base64 -d | bash"

echo "==> Verifying the image is present on the worker"
ssh -T "$HEAD_HOST" "ssh -T -o StrictHostKeyChecking=no '$WORKER_IP' 'docker image inspect ${LAB_IMAGE} > /dev/null && echo worker: image ok'"
