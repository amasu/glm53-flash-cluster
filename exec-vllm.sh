#!/usr/bin/env bash
# Container entrypoint for glm53 (both ranks). Generic by design: this file
# contains NO stack policy — every stack-specific serving value arrives via
# the environment, exported from stacks/<STACK>.env (plus site .env) before
# docker compose runs. Any stack differs from any other ONLY in these
# configuration variables; the code below is common to all.
#
# Knobs (all optional; a stack file pins its proven values — see benchmarks.md):
#   SERVED_MODEL_NAME        (default glm-5.3-flash)
#   GPU_MEMORY_UTILIZATION   (default 0.90 — with a pinned KV pool the engine
#                             skips memory profiling; 0.90 leaves activation
#                             headroom for the warmup forward on GB10 UMA)
#   KV_CACHE_DTYPE           e.g. fp8 | fp8_ds_mla (empty = engine default)
#   KV_CACHE_MEMORY_BYTES    hermetic KV pin in bytes (empty = GMU-sized pool)
#   MAX_MODEL_LEN            (default 524288)
#   MAX_NUM_SEQS             (default 16)
#   BLOCK_SIZE               (default 256)
#   MAX_NUM_BATCHED_TOKENS   empty = flag omitted (vLLM default)
#   MOE_BACKEND              e.g. marlin (empty = flag omitted)
#   SPECULATIVE_CONFIG       raw JSON, e.g. {"method":"mtp","num_speculative_tokens":3}
#                            (empty = flag omitted)
#   SWITCHES                 extra boolean flags, space separated, e.g.
#                            "--enable-chunked-prefill --enable-prefix-caching
#                             --disable-custom-all-reduce --enforce-eager
#                             --language-model-only --skip-mm-profiling"
#   KERNEL_CONFIG            raw JSON for --kernel-config (empty = omitted)
#   LIMIT_MM_PER_PROMPT      raw JSON for --limit-mm-per-prompt (empty = omitted)
#   TENSOR_PARALLEL_SIZE     (default 2)
# Node plumbing comes from docker-compose.yml: NODE_RANK, MODEL_PATH,
# HEAD_ADDR, MASTER_PORT, SERVING_PORT, HEADLESS.
set -euo pipefail

NODE_RANK="${NODE_RANK:?NODE_RANK must be 0 or 1}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "FATAL: bad NODE_RANK=$NODE_RANK" >&2; exit 2; }
MODEL_PATH="${MODEL_PATH:-/models/model}"
HEAD_ADDR="${HEAD_ADDR:?HEAD_ADDR must be set}"
MASTER_PORT="${MASTER_PORT:-29521}"
SERVING_PORT="${SERVING_PORT:-8000}"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: $MODEL_PATH/config.json missing" >&2; exit 1; }

# ---- build the argv as an array (JSON flags must stay single argvs) --------
args=(
  serve "$MODEL_PATH"
  --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash}"
  --host 0.0.0.0 --port "$SERVING_PORT"
  --trust-remote-code
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --max-model-len "${MAX_MODEL_LEN:-524288}"
  --max-num-seqs "${MAX_NUM_SEQS:-16}"
  --block-size "${BLOCK_SIZE:-256}"
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  --distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
  --nnodes 2 --node-rank "$NODE_RANK"
  --master-addr "$HEAD_ADDR" --master-port "$MASTER_PORT"
)

# Conditional flags: appended only when the stack configures them
[ -n "${KV_CACHE_DTYPE:-}" ]        && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[ -n "${MOE_BACKEND:-}" ]           && args+=(--moe-backend "$MOE_BACKEND")
[ -n "${SPECULATIVE_CONFIG:-}" ]    && args+=(--speculative-config "$SPECULATIVE_CONFIG")
[ -n "${KERNEL_CONFIG:-}" ]         && args+=(--kernel-config "$KERNEL_CONFIG")
[ -n "${LIMIT_MM_PER_PROMPT:-}" ]   && args+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")

# SWITCHES: space-separated boolean flags (word split intentional)
if [ -n "${SWITCHES:-}" ]; then
  # shellcheck disable=SC2206
  args+=($SWITCHES)
fi

[ -n "${HEADLESS:-}" ] && args+=(--headless)

echo "=============================================================="
echo " GLM-5.3-Flash-NVFP4 vLLM rank=$NODE_RANK image=$(vllm --version 2>/dev/null || echo unknown)"
echo " model=$MODEL_PATH master=$HEAD_ADDR:$MASTER_PORT port=$SERVING_PORT"
echo " stack=${STACK_NAME:-unset} ctx=${MAX_MODEL_LEN:-default} KV=${KV_CACHE_DTYPE:-bf16}${KV_CACHE_MEMORY_BYTES:+ pin=${KV_CACHE_MEMORY_BYTES}B}"
echo "=============================================================="
printf ' argv:'; printf ' %s' "${args[@]}"; echo

vllm "${args[@]}" &
VLLM_PID=$!

if [[ "$NODE_RANK" == "0" ]]; then
  echo "[rank0] waiting up to 40 min for /v1/models on 127.0.0.1:$SERVING_PORT ..."
  ready=""
  for i in $(seq 1 480); do
    if curl -fsS "http://127.0.0.1:$SERVING_PORT/v1/models" >/dev/null 2>&1; then
      echo "[rank0] ENGINE READY — /v1/models answered:"
      curl -fsS "http://127.0.0.1:$SERVING_PORT/v1/models" | head -c 600; echo
      ready=1
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[rank0] FATAL: vllm serve died during warmup (poll $i)" >&2
      wait "$VLLM_PID" || exit 1
    fi
    sleep 5
  done
  [[ -n "$ready" ]] || { echo "[rank0] FATAL: engine not ready after 40 min, aborting" >&2; exit 1; }
fi

# Foreground the vllm process so the container stays alive; propagate its exit.
wait "$VLLM_PID"
