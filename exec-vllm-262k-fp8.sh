#!/usr/bin/env bash
# Container entrypoint for glm53:v8 (both ranks).
# Runs vllm serve for the GLM-5.3-Flash NVFP4 TP=2 cluster, then (rank 0 only)
# polls /v1/models until the OpenAI endpoint answers.
# All cluster parameters arrive as env vars from docker-compose.yml.
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
echo "=============================================================="

# Launch flag history (step-1 change 2026-08-28, 262K->512K + FP8 KV):
#   * --kv-cache-dtype fp8            SM90 NoPE path dequantizes in-kernel
#                                     (has_flashinfer_sm90_nope_mla()=True on
#                                     this build; fp8_ds_mla is the SM120-only
#                                     packed layout — NOT used here). Pools the
#                                     KV at ~2x tokens/GiB vs the old bf16.
#   * --kv-cache-memory-bytes 10GiB   deterministic pinned pool (~1.16M tokens
#                                     at block-size 2304); GMU-based sizing has
#                                     no headroom at 91.23 GiB weights/rank.
#   * --max-model-len 524288          512K context. Requires the v9 image's
#                                     sparse-MLA indexer CC-12.x guard (62
#                                     CTAs/128KB smem requested at 512K shapes
#                                     vs 48 SMs/99KB on GB10).
#   * --max-num-batched-tokens 4096   prefill-chunk scratch ceiling; 8192 OOMs
#                                     the GB10 driver at 512K shapes.
#   * --kernel-config autotune off    FlashInfer autotune scratch at 512K
#                                     shapes OOMs (NVRM NV_ERR_NO_MEMORY); the
#                                     non-autotuned sparse-MLA fallback is fine.
#   * chunked prefill / prefix caching / disable-custom-all-reduce: proven
#                                     settings from the 512K reference recipe
#                                     (forum #381350 post 55).
# UNCHANGED on purpose (step-1 A/B is context+KV only): moe-backend marlin
# (silent FP4 corruption fix), enforce-eager (MTP+CUDA-graph quality bug),
# MTP k=4, block-size 2304, gm-0.90.
# Bisection note (2026-08-28): the full 512K reference profile (10 GiB pinned
# KV + 4096 batched-tokens + prefix/chunked-prefill) OOMs the GB10 unified
# memory on THIS stack, which keeps multimodal enabled (the 512K reference
# recipe used --language-model-only, saving ~15.7 GiB). Step 1 therefore
# takes the minimal, low-risk change: --kv-cache-dtype fp8 only (2x token
# density), GMU/block/MTP/multimodal left at the proven v8 values. 512K is a
# separate follow-up that adds --language-model-only + chunked prefill.
vllm serve "$MODEL_PATH" \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port "$SERVING_PORT" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-seqs 6 --block-size 2304 --moe-backend marlin \
  --enforce-eager \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
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
