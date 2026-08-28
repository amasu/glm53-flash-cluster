#!/usr/bin/env bash
# Container entrypoint for glm53:v9 — 512K-context experiment (round 2).
# Round-1 failure mode (documented): the full 512K reference memory profile
# OOM'd the GB10 driver (NVRM NV_ERR_NO_MEMORY during the engine warmup
# forward) because THIS stack keeps the multimodal processor alive
# (~15.7 GiB anon in the API front-end). The proven 512K reference recipe
# (forum #381350 post 55) used --language-model-only; that is the delta this
# variant adds, plus its remaining memory-shaping flags:
#   10 GiB pinned KV pool, 4096 batched-tokens, chunked prefill,
#   prefix caching, custom-all-reduce off, autotune/cutedsl-warmup off,
#   GMU 0.90, MTP k=3.
# The v9 image carries the sparse-MLA indexer CC-12.x guard required for
# 512K-context shapes (see docker/patch_v9_512k.py).
# If this boots and passes quality, promote it; otherwise revert to
# exec-vllm.sh (262K + FP8 KV minimal profile).
set -euo pipefail

NODE_RANK="${NODE_RANK:?NODE_RANK must be 0 or 1}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "FATAL: bad NODE_RANK=$NODE_RANK" >&2; exit 2; }
MODEL_PATH="${MODEL_PATH:-/models/glm-5.3-flash-nvfp4}"
HEAD_ADDR="${HEAD_ADDR:?}"
MASTER_PORT="${MASTER_PORT:-29521}"
SERVING_PORT="${SERVING_PORT:-8000}"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: $MODEL_PATH/config.json missing" >&2; exit 1; }

echo "=============================================================="
echo " GLM-5.3-Flash-NVFP4 vLLM rank=$NODE_RANK image=$(vllm --version 2>/dev/null || echo unknown)"
echo " model=$MODEL_PATH master=$HEAD_ADDR:$MASTER_PORT port=$SERVING_PORT"
echo " profile: 512K + FP8 KV 10GiB pin + language-model-only"
echo "=============================================================="

vllm serve "$MODEL_PATH" \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port "$SERVING_PORT" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 10737418240 \
  --max-model-len 524288 \
  --max-num-seqs 6 --block-size 2304 --moe-backend marlin \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --kernel-config '{"enable_cutedsl_warmup":false,"enable_flashinfer_autotune":false}' \
  --enforce-eager \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --language-model-only \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$HEAD_ADDR" --master-port "$MASTER_PORT" \
  ${HEADLESS:-} &
VLLM_PID=$!

if [[ "$NODE_RANK" == "0" ]]; then
  echo "[rank0] waiting up to 40 min for /v1/models on 127.0.0.1:$SERVING_PORT ..."
  for i in $(seq 1 480); do
    if curl -fsS "http://127.0.0.1:$SERVING_PORT/v1/models" >/dev/null 2>&1; then
      echo "[rank0] ENGINE READY — /v1/models answered:"
      curl -fsS "http://127.0.0.1:$SERVING_PORT/v1/models" | head -c 600; echo
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[rank0] FATAL: vllm serve died during warmup (poll $i)" >&2
      wait "$VLLM_PID" || exit 1
    fi
    sleep 5
  done
fi

# Foreground the vllm process so the container stays alive; propagate its exit.
wait "$VLLM_PID"
