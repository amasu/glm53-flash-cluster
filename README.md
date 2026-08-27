# GLM-5.3-Flash-NVFP4 on 2× DGX Spark — vLLM TP=2 (compose)

Head `aitopatom-6253` (10.100.90.1, rank 0) + worker `spark-cfb3` (10.100.90.4, rank 1),
serving `LibertAIDAI/GLM-5.3-Flash-NVFP4` (320B total / 18B active, 181 GiB, `glm5_next`)
at `http://aitopatom-6253.local:8000/v1` as `glm-5.3-flash`.

## Why a patched image (research, 2026-08-27)

- Model card: on GB10/sm_121 **vLLM ❌** (echoes prompt — sm_121 MLA kernel asserts
  `pe_dim==64`, model is NoPE `qk_rope_head_dim=0`); **SGLang ✅** is the card-blessed
  engine. Upstream vLLM lacks `glm5_next` (PR #53906 open); day-0 image
  `vllm/vllm-openai:glm53-flash-arm64-cu130` works on B200 only.
- **tonyd2world/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark** — world-first vLLM TP2 on
  2× Spark, same checkpoint, day-0. Root-caused 7 day-0 bugs → patch chain v1→v8:
  NoPE-MLA-for-SM121 (FA2), FlashInfer 0.6.18 nightly (0.6.17 FA2 NaN at 64–256-row
  batches), NCCL 2.30.7 re-pin (nightly → 2.29.7 fabric death), cutlass-dsl 4.6.2,
  PDL off on SM12x, indexer top-k hardening, fp8-KV smem-tile fix.
  Validated: 262K ctx, ~14.3 tok/s (bf16 KV) / ~21.8 tok/s (fp8 KV + MTP-4).
- The eugr/spark-vllm-docker b12x stack can't fix these (image-level, not
  orchestration-level); it CAN have carried the PR, but its custom MLA backends are
  tuned for the pe_dim=64 layout class this model can't use. So: tonyd image +
  plain compose orchestration, tuned to the fabric.

## Layout

- `docker-compose.yml` — both ranks; run via `cluster.sh` (worker first, then head)
- `exec-vllm.sh` — container entrypoint (baked into `glm53:v8` at /workspace)
- `build-image.sh` — head: clone tonyd2world repo, build v1→v8, ship to worker
- `fetch-weights.sh` — head: HF download + rsync to worker + checksum verify
- `cluster.sh` + `.env` — Mac-side orchestration (preflight/up/down/status/logs)

## Deploy (done 2026-08-27)

1. `./build-image.sh` on head → `glm53:v8` on both nodes (IDs verified identical)
2. `./fetch-weights.sh` on head → `/var/tmp/glm-5.3-flash-nvfp4` on both, sha256-verified
3. `cluster.sh preflight` → `cluster.sh up` → boot 14–21 min → `cluster.sh logs`
4. Smoke: `curl http://aitopatom-6253.local:8000/v1/models`

## Key serve flags (load-bearing — do not "clean up")

| flag | why |
|---|---|
| `--block-size 2304` | aligner default 2176 → kpool pages tile by 32 not 64; DeepGEMM arch-12 fp8 paged-MQA needs 64-entry pages |
| `--gpu-memory-utilization 0.85` | 0.78–0.80 starve KV cache at long context |
| `--moe-backend marlin` | card's known-good sm_121 fallback |
| `--enforce-eager` | CUDA graphs unvalidated on this arch at day-0; MTP still engages |
| `--tool-call-parser glm47` | `glm` fails SILENTLY (empty content, tool_calls null) |
| `--reasoning-parser glm45` | without it trace lands in `content` with bare `</think>` |
| `--speculative-config mtp/4` | MTP head is BF16 in this checkpoint; acceptance 2.5–2.9 |
| `NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0` | unvalidated on consumer Blackwell / unified memory |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | 320B MoE warmup; else killed mid-init |

Thinking ON by default (effort max). Per-request off:
`"chat_template_kwargs": {"enable_thinking": false}` (max_tokens includes reasoning tokens).
Reasoning effort: `chat_template_kwargs.reasoning_effort` = low|high|max.

## Ops rules (hard-won, from reference deploy + NVIDIA forums)

- Launch order: worker rank 1 → wait ~25 s → head rank 0 (mp executor rendezvous)
- Tear down BOTH ranks before any relaunch; capture `docker logs` before removing
- Two consecutive unexplained rank deaths = stop and diagnose, never crash-loop
- Forum-documented failure classes: TP2 node-drop after first prompt (#358755);
  total host freeze during heavy multi-node prefill (#376882) → keep MTU 9000,
  cap context at 262K on day 1, drop_caches before launches
- KV headroom: local weights both nodes = +33% vs NFS; stage 2 = fp8 KV
  (`--kv-cache-dtype fp8_e4m3 --kv-cache-memory 5905580032`, 672K-token pool)

## Rollback

DeepSeek-V4-Flash-0731 cluster (port 8000) restarts from
`~/deepseek-v4-flash-optimization/tier2/exec-script-*.sh` when this stack is down.
