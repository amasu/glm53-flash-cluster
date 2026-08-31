# GLM-5.3-Flash on 2× DGX Spark — vLLM TP=2

Two-node DGX Spark cluster (head = rank 0, worker = rank 1) serving
`GLM-5.3-Flash-NVFP4` (320B total / 18B active) as `glm-5.3-flash` over
vLLM TP=2 on the QSFP/RoCE fabric.

## One source of code, many stacks

The whole system is **one entrypoint + one compose file + one orchestrator**,
shared by every serving stack. A stack is **pure configuration**: a
`stacks/<name>.env` file that sets the image, weights path, and every
serving flag. No code is copied, patched, or duplicated per stack.

```
docker-compose.yml      one compose file, all stacks
exec-vllm.sh            one container entrypoint, all stacks (no stack policy)
cluster.sh              one orchestrator: pick a stack, launch both ranks
watchdog.sh             auto-restart whichever stack is configured in .env
stacks/*.env            the ONLY thing that differs between stacks
```

The standing config is the **lab-quant stack** (`STACK=lab`):
`local-inference-lab/GLM-5.3-Flash-NVFP4` (mixed-precision NVFP4 experts +
MXFP8 MTP drafter) on `glm53:lab` — 512K context, `fp8_ds_mla` KV with a
10 GiB pinned pool (1,164,369 tokens, 2.22× concurrency @512K), MTP k=3,
`1024/16` batched-tokens/num-seqs, `--language-model-only`, `--enforce-eager`
(load-bearing). 24–30 tok/s decode; 90/100 tool-call quality.

## The stacks

| `STACK` | image | weights | profile |
|---|---|---|---|
| **`lab`** | `glm53:lab` | lab-quant (`WEIGHTS_DIR_LAB`) | 512K, `fp8_ds_mla` KV 10 GiB pin, MTP k=3, `1024/16`, LMO, eager — **standing config, 90/100** |
| `lab-vision` | `glm53:lab` | lab-quant | same as `lab` + multimodal ON (`--skip-mm-profiling` + modern `--limit-mm-per-prompt`), 9 GiB KV pin |
| `v9-512k` | `glm53:v9` | LibertAIDAI | 512K, `fp8` KV 9 GiB pin, MTP k=4, LMO — primary rollback path, 89/100 |
| `v9-262k-fp8` | `glm53:v9` | LibertAIDAI | 262K, `fp8` KV unpinned, MTP k=4, mm ON — historical A/B point |
| `v8-262k` | `glm53:v9` | LibertAIDAI | 262K, bf16 KV, MTP k=4, mm ON — day-0 bring-up, 89/100 |

Switching stacks is a config change, not a code change:

```bash
cluster.sh <stack> takeover   # stop any running stack, then bring up <stack>
```

**Single-stack policy:** only one profile runs at a time. `down`/`takeover`
remove any `glm53-*` rank containers regardless of which stack launched
them.

## Host-agnostic

Every site-specific value (hostnames, IPs, ports, paths, interface names)
comes from `.env` — copy `example.env` → `.env`, fill it in, and deploy it in
three places:

1. next to `cluster.sh` on the orchestrator (your workstation, e.g. a Mac)
2. at `REMOTE_DIR` on the **head** node
3. at `REMOTE_DIR` on the **worker** node

The stack file (`stacks/<STACK>.env`) and the entrypoint are part of the repo
and get mirrored to both nodes by `cluster.sh mirror`. Scripts and compose
fail fast with a message naming the missing variable.

## Configuration variables

**Site values (`.env`, shared by all stacks):**

| Variable | Purpose |
|---|---|
| `STACK` | default stack (used when `cluster.sh` gets no stack argument) |
| `HEAD_HOST` | ssh address of the head node (orchestrator → head) |
| `HEAD_IP` | head IP on the fabric (rank 0 rendezvous / NCCL / VLLM_HOST_IP) |
| `WORKER_IP` | worker address as reached *from the head* (rank 1) |
| `REMOTE_DIR` | where this repo lives on both cluster nodes |
| `WEIGHTS_DIR` | LibertAIDAI weights on both nodes (v9-*/v8-* stacks) |
| `WEIGHTS_DIR_LAB` | lab-quant weights on both nodes (lab / lab-vision stacks) |
| `VLLM_CACHE_DIR` | vLLM/HF cache dir on both nodes |
| `SERVING_PORT` | OpenAI endpoint port (default 8000) |
| `MASTER_PORT` | rendezvous port for the LibertAIDAI stacks (default 29521) |
| `MASTER_PORT_LAB` | rendezvous port for the lab stacks (default 29500) |
| `IMAGE` / `LAB_IMAGE` | image tags (default `glm53:v9` / `glm53:lab`) |
| `FABRIC_RANGE` | interconnect subnet, e.g. `10.0.0.0/24` |
| `IF_NAME` | fabric interface name on both nodes (`ip link`) |
| `NCCL_IB_HCA` | RoCE/IB HCA list (`ibdev2netdev`) |
| `NCCL_IB_GID_INDEX` | RoCE GID index (default 3) |

**Per-stack knobs (`stacks/<STACK>.env`)** — the only thing that differs
between stacks: `IMAGE`, `WEIGHTS_DIR`, `MASTER_PORT`, and the serving knobs
consumed by `exec-vllm.sh` (`GPU_MEMORY_UTILIZATION`, `KV_CACHE_DTYPE`,
`KV_CACHE_MEMORY_BYTES`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `BLOCK_SIZE`,
`MAX_NUM_BATCHED_TOKENS`, `MOE_BACKEND`, `SPECULATIVE_CONFIG`, `SWITCHES`,
`KERNEL_CONFIG`, `LIMIT_MM_PER_PROMPT`, …). See the header block of
`exec-vllm.sh` for the full knob list.

## Layout

- `docker-compose.yml` — one compose file for both ranks, all stacks
- `exec-vllm.sh` — one container entrypoint; builds the `vllm serve` argv
  entirely from environment; **contains no stack policy**
- `stacks/{lab,lab-vision,v9-512k,v9-262k-fp8,v8-262k}.env` — the stacks
- `cluster.sh` — orchestrator: `cluster.sh [STACK] <preflight|mirror|up|down|status|logs|takeover>`
- `watchdog.sh` — probe `:SERVING_PORT`; restart via `cluster.sh takeover` (cron on the orchestrator)
- `build-image.sh` — head: build the `glm53:v9` patch chain, ship to worker
- `docker/lab-build.sh` — build `glm53:lab` from the vendored `docker/labbuild/` context, ship to worker
- `fetch-weights.sh` — head: download LibertAIDAI weights + rsync to worker + verify
- `docker/labbuild/` — vendored build context for `glm53:lab` (digest-pinned base + patches + provenance)
- `benchmarks.md` — investigation + benchmark log (quality, speed, pools, forensics)
- `NOTES-512k.md` — 512K upgrade notes + crash forensics + rollback
- `example.env` / `.env` — configuration templates

## Why a patched image (research, 2026-08-27)

- Model card: on GB10/sm_121 **vLLM ❌** (echoes prompt — sm_121 MLA kernel
  asserts `pe_dim==64`, model is NoPE `qk_rope_head_dim=0`); **SGLang ✅** is
  the card-blessed engine. Upstream vLLM lacks `glm5_next` (PR #53906 open).
- **tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark** — world-first vLLM
  TP2 on 2× Spark, same checkpoint, day-0. Root-caused 7 day-0 bugs → patch
  chain v1→v8; validated 262K ctx, ~14.3 tok/s (bf16 KV) / ~21.8 tok/s
  (fp8 KV + MTP-4).
- `build-image.sh` assembles that chain into `glm53:v9` (adds the 512K-context
  sparse-MLA indexer CC-12.x guard from `docker/patch_v9_512k.py`).
- `glm53:lab` adds the lab-checkpoint patches (FujitsuPolycom) for the
  mixed-precision quant.

## Quick Start (from the orchestrator)

```bash
# 1. Clone and configure
git clone https://github.com/amasu/glm53-flash-cluster && cd glm53-flash-cluster
cp example.env .env
$EDITOR .env            # HEAD_HOST, HEAD_IP, WORKER_IP, REMOTE_DIR,
                        # FABRIC_RANGE, IF_NAME, NCCL_IB_HCA, STACK=lab
set -a; source .env; set +a

# 2. Mirror the repo (incl. .env + stacks/) to both nodes
./cluster.sh mirror

# 3. Build the image(s) on the head, ship to the worker
./docker/lab-build.sh            # glm53:lab (lab stacks)
./build-image.sh                 # glm53:v9  (LibertAIDAI stacks)

# 4. Download weights (~186 GB lab-quant, or ~182 GB LibertAIDAI) to both nodes
./fetch-weights.sh               # LibertAIDAI -> WEIGHTS_DIR
#    lab-quant -> WEIGHTS_DIR_LAB (huggingface-cli download
#    local-inference-lab/GLM-5.3-Flash-NVFP4 --revision 378ca545…)

# 5. Preflight, then launch the configured stack (default: lab)
./cluster.sh preflight
./cluster.sh lab up             # or: ./cluster.sh lab-vision up, ./cluster.sh v9-512k up, …

# 6. Watch warmup (14-21 min), then smoke-test
./cluster.sh lab logs
curl "http://$HEAD_HOST:${SERVING_PORT:-8000}/v1/models"
```

Switching stacks (single-stack policy — stops the old one first):

```bash
./cluster.sh v9-512k takeover    # rollback to the LibertAIDAI profile
./cluster.sh lab-vision takeover # switch to the vision-enabled lab profile
```

Optionally run `watchdog.sh` from cron (`*/5 * * * *`) to auto-restart the
configured stack if the endpoint dies (it skips a live boot via container age).

## Key serve flags (load-bearing — do not “clean up”)

The standing lab stack's flags live in `stacks/lab.env`; the LibertAIDAI
stacks' in `stacks/v9-*.env` / `stacks/v8-262k.env`. Why each one matters:

| flag | why |
|---|---|
| `--block-size 2304` (v9 stacks) / `256` (lab) | v9: aligner default 2176 → kpool pages tile by 32 not 64; DeepGEMM arch-12 fp8 paged-MQA needs 64-entry pages. lab: 256 is the recipe value. |
| `--gpu-memory-utilization 0.90` | with a pinned KV pool the engine skips memory profiling; 0.90 leaves activation headroom for the warmup forward (lower values just waste UMA) |
| `--kv-cache-dtype fp8` / `fp8_ds_mla` + `--kv-cache-memory-bytes …` | SM90 NoPE path dequantizes in-kernel; the pin makes the pool deterministic instead of GMU-dependent |
| `--max-model-len 524288` | 512K context; requires the image's sparse-MLA indexer CC-12.x guard |
| `--max-num-batched-tokens 4096` (v9) / `1024` (lab) | 8192 OOMs the GB10 driver at 512K shapes; 1024 is the lab-quant proven value (the gist's 4096 OOM-kills the head worker on this quant) |
| `--kernel-config '{…autotune…:false}'` | autotune/cutedsl-warmup scratch at 512K shapes OOMs (`NV_ERR_NO_MEMORY`) |
| `--speculative-config mtp/k` | MTP is lossless vs k; k is a speed knob. k=4 on the LibertAIDAI stacks, k=3 on the lab stack (benchmarks.md §6) |
| `--language-model-only` | drops the ~15.7 GiB multimodal front-end; without it the pinned-KV + gm-0.90 profile OOMs at warmup on the 121.69 GiB UMA line. lab-vision turns this back on (see `stacks/lab-vision.env`) |
| `--moe-backend marlin` | card's known-good sm_121 fallback (LibertAIDAI stacks) |
| `--enforce-eager` | CUDA graphs unvalidated on this arch; also a quality finding for the lab quant. MTP still engages |
| `--tool-call-parser glm47` | `glm` fails SILENTLY (empty content, tool_calls null) |
| `--reasoning-parser glm45` | without it trace lands in `content` with bare `</think>` |
| `--skip-mm-profiling` + modern `--limit-mm-per-prompt` (lab-vision) | vision tower ~60K KV tokens instead of ~300K; legacy `{"video": 1}` format is rejected by this build |
| `NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0` | unvalidated on consumer Blackwell / unified memory |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | 320B MoE warmup; else killed mid-init |

Thinking ON by default (effort max). Per-request off:
`"chat_template_kwargs": {"enable_thinking": false}` (max_tokens includes
reasoning tokens). Reasoning effort: `chat_template_kwargs.reasoning_effort`
= low|high|max.

## Ops rules (hard-won, from reference deploy + NVIDIA forums)

- Launch order: worker rank 1 → wait ~20 s → head rank 0 (mp executor rendezvous)
- Tear down BOTH ranks before any relaunch; capture `docker logs` before removing
- Two consecutive unexplained rank deaths = stop and diagnose, never crash-loop
- Forum-documented failure classes: TP2 node-drop after first prompt (#358755);
  total host freeze during heavy multi-node prefill (#376882) → keep MTU 9000,
  cap context at 262K on day 1, drop_caches before launches (done by `takeover`)
- KV headroom: local weights both nodes = +33% vs NFS; fp8 KV + pinned pool
  is the standing default (see `benchmarks.md` for the pin-size vs draft-depth
  trade)

## Credits

This repo is an orchestration layer on top of the day-0 work of others — the
sm_121 patch chain, the debugging knowledge, and the failure folklore all come
from these sources:

**Patches & reference deploy**

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
  — world-first vLLM TP=2 on 2× DGX Spark. Built the v1→v8 sm_121 patch chain
  that `build-image.sh` assembles. Their repo is also the source of the ops
  rules in this README.
- Day-0 base image [`vllm/vllm-openai:glm53-flash-arm64-cu130`](https://hub.docker.com/r/vllm/vllm-openai)
  and upstream [vLLM PR #53906](https://github.com/vllm-project/vllm/pull/53906)
  (`glm5_next` architecture support).
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — the b12x
  Spark serving stack; evaluated as an alternative carrier.

**Checkpoints & quantizations**

- Upstream model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
  (320B MoE, released 2026-08-26).
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
  — uniform NVFP4 quant staged by `fetch-weights.sh` (the v9-*/v8-* stacks).
- [local-inference-lab/GLM-5.3-Flash-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
  — mixed-precision quant behind the lab stacks.
- [0rand/glm-5.3-flash-nvfp4-2x-dgx-sparks](https://github.com/0rand/glm-5.3-flash-nvfp4-2x-dgx-sparks)
  — packaging of the lab quant for 2× Spark.

**512K-context & lab-quant recipes**

- NVIDIA forum thread
  [GLM-5.3-Flash: 320B total parameters / 18B active](https://forums.developer.nvidia.com/t/glm-5-3-flash-320b-total-parameters-18b-active/381350):
  post 55 (0rand) — 512K-context recipe; post #124 — the `1024/16` sizing.
- [FujitsuPolycom/glm53-flash-tp2-spark](https://github.com/FujitsuPolycom/glm53-flash-tp2-spark)
  (Apache-2.0) — the `sparse_attn_indexer*.patch` pair (CC-12.x guard) and the
  `model.patch` + `modelopt.patch` lab-checkpoint patches, vendored in
  `docker/labbuild/` and mirrored by `docker/patch_v9_512k.py`.
- **kilork** — the `local-inference-lab` mixed-precision recipe
  ([gist](https://gist.github.com/kilork/a887667f4f423b7cc324859cd5e32ebd),
  92/100 hardmode, incl. the `--enforce-eager` quality finding).
- [kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark)
  — the NoPE-MLA rope-pad + sm120 topk mod (`patch_mla.py`, vendored in
  `docker/labbuild/`; engaged via `VLLM_MLA_NOPE_PAD_ROPE=1`).

**Benchmark harness**

- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)
  — the harness behind every quality score in this repo (seed-42 hardmode,
  88 scenarios).

**NVIDIA DGX Spark / GB10 forum threads** (failure classes that shaped the ops
rules — launch order, both-rank teardown, no crash-looping):

- [TP=2 node-drop after the first prompt](https://forums.developer.nvidia.com/t/two-spark-cluster-with-vllm-using-tensor-parallel-size-2-causes-one-node-to-drop-while-the-others-gpu-goes-100-forever/358755)
- [Total host freeze during multi-node TP=2 prefill](https://forums.developer.nvidia.com/t/total-host-freeze-not-process-hang-during-multi-node-tp-2-vllm-prefill-on-2x-dgx-spark-gb10-zero-forensic-trace-across-kdump-watchdogs-netconsole/376882)

More GB10 collective wisdom: the
[DGX Spark / GB10 forum category](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719).
