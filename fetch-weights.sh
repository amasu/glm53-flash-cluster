#!/usr/bin/env bash
# Run on the HEAD node. Downloads LibertAIDAI/GLM-5.3-Flash-NVFP4 (182 GiB),
# stages it at /var/tmp/glm-5.3-flash-nvfp4, rsyncs to the worker at the same
# path over the fabric, and verifies checksum anchors on both sides.
# Local weights on BOTH nodes: validated +33% KV headroom vs NFS.
set -euo pipefail

MODEL_ID="LibertAIDAI/GLM-5.3-Flash-NVFP4"
DIR="/var/tmp/glm-5.3-flash-nvfp4"
WORKER_SSH="${WORKER_SSH:-ssh 10.100.90.4}"

VENV="/tmp/hfvenv"
if [[ ! -x "$VENV/bin/hf" ]]; then
  echo "==> setting up hf CLI in $VENV"
  python3 -m venv "$VENV" 2>/dev/null \
    || { apt-get install -y -q python3-venv || sudo apt-get install -y -q python3-venv; python3 -m venv "$VENV"; }
  "$VENV/bin/pip" install -q -U pip "huggingface_hub[cli]"
fi

mkdir -p "$DIR"
echo "==> downloading $MODEL_ID -> $DIR (resume-safe; ~182 GiB, 120 shards)"
HF_HUB_DOWNLOAD_TIMEOUT=60 \
  "$VENV/bin/hf" download "$MODEL_ID" --local-dir "$DIR"

echo "==> checksum anchors (head)"
cd "$DIR"
sha256sum config.json generation_config.json model-00001-of-00120.safetensors model-00120-of-00120.safetensors > /tmp/glm53-checksums.txt
cat /tmp/glm53-checksums.txt
ls "$DIR"/*.safetensors | wc -l | grep -q '^120$' || { echo "FATAL: expected 120 shards, got $(ls "$DIR"/*.safetensors | wc -l)" >&2; exit 1; }

echo "==> rsync to worker over fabric"
$WORKER_SSH "mkdir -p $DIR"
rsync -a --info=progress2 "$DIR" "$WORKER_SSH:$DIR"

echo "==> verify worker copy"
$WORKER_SSH "cd $DIR && sha256sum config.json generation_config.json model-00001-of-00120.safetensors model-00120-of-00120.safetensors; ls *.safetensors | wc -l; du -sh $DIR"
