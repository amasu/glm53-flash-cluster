#!/usr/bin/env bash
# Download LibertAIDAI/GLM-5.3-Flash-NVFP4 (182 GiB) on the HEAD node,
# then rsync it to the worker at the identical path over the QSFP fabric.
# Local weights on BOTH nodes is the validated choice (NFS-serving eats unified
# memory: +33% KV headroom measured with local weights in the reference deploy).
set -euo pipefail

MODEL_ID="LibertAIDAI/GLM-5.3-Flash-NVFP4"
DIR="/var/tmp/glm-5.3-flash-nvfp4"
WORKER_SSH="${WORKER_SSH:-ssh 10.100.90.4}"
WORKER_DIR="/var/tmp/glm-5.3-flash-nvfp4"

mkdir -p "$DIR"

echo "==> Downloading $MODEL_ID -> $DIR (182 GiB, 120 shards; resume-safe)"
# huggingface-cli resumes partial downloads; the image's python has it too, but host-side is fine:
pip install -q "huggingface_hub[cli]" 2>/dev/null || true
huggingface-cli download "$MODEL_ID" --local-dir "$DIR"

echo "==> Checksum anchors"
sha256sum "$DIR/config.json" "$DIR/model-00001-of-00120.safetensors" | tee /tmp/glm53-checksums.txt

echo "==> rsync to worker ($WORKER_SSH:$WORKER_DIR) over fabric"
$WORKER_SSH "mkdir -p $WORKER_DIR"
rsync -a --info=progress2 "$DIR" "$WORKER_SSH:$WORKER_DIR"

echo "==> Verify worker copy"
$WORKER_SSH "sha256sum $WORKER_DIR/config.json $WORKER_DIR/model-00001-of-00120.safetensors; du -sh $WORKER_DIR"
diff <(cut -d' ' -f1 /tmp/glm53-checksums.txt) \
     <($WORKER_SSH "cd $WORKER_DIR && sha256sum config.json model-00001-of-00120.safetensors | cut -d' ' -f1") \
  && echo "CHECKSUMS MATCH"
