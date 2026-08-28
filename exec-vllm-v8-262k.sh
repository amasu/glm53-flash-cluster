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

vllm serve "$MODEL_PATH" \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port "$SERVING_PORT" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
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
