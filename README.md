# GLM-5.3-Flash-NVFP4 on 2× DGX Spark — vLLM TP=2 (compose)

Two-node DGX Spark cluster (head = rank 0, worker = rank 1) serving
`LibertAIDAI/GLM-5.3-Flash-NVFP4` (320B total / 18B active, 181 GiB, `glm5_next`)
as `glm-5.3-flash` over vLLM TP=2 on the QSFP/RoCE fabric.

**Current config (2026-08-31):** the standing config is the **lab-quant stack**
(`lab/`): `local-inference-lab/GLM-5.3-Flash-NVFP4` (mixed-precision NVFP4
experts + MXFP8 MTP drafter) served as `glm-5.3-flash` — 512K context,
`fp8_ds_mla` KV with 10 GiB pinned pool (1,164,369 tokens, 2.22× concurrency
@512K), MTP k=3, batched-tokens 1024 / num-seqs 16, `--language-model-only`
(text-only), `--enforce-eager` (load-bearing). 24–30 tok/s decode; 90/100
tool-call quality (seed-42 hardmode; 0rand's own 91/100 reproduced within
1 pt). The former LibertAIDAI v8/v9 profiles (89/100) remain as a two-stack
rollback path. See `benchmarks.md` for the full investigation log,
`NOTES-512k.md` for the 512K upgrade + crash forensics, and `lab/build/README.md`
for the lab image recipe.

**Host-agnostic:** every site-specific value (hostnames, IPs, ports, paths,
interface names) comes from `.env`. Copy `example.env` → `.env`, fill it in,
and deploy it in three places:

1. next to `cluster.sh` on the orchestrator (your workstation, e.g. a Mac)
2. at `REMOTE_DIR` on the **head** node
3. at `REMOTE_DIR` on the **worker** node

Scripts and compose fail fast with a message naming the missing variable.

## Configuration variables (`example.env`)

| Variable | Purpose |
|---|---|
| `HEAD_HOST` | ssh-reachable address of the head node (orchestrator → head) |
| `HEAD_IP` | head node IP on the fabric (rank 0 rendezvous / NCCL / VLLM_HOST_IP) |
| `WORKER_IP` | worker address as reached *from the head* (two-hop ssh, rank 1) |
| `REMOTE_DIR` | where this repo lives on both cluster nodes |
| `WEIGHTS_DIR` | where model weights are staged on both nodes |
| `VLLM_CACHE_DIR` | vLLM/HF cache dir on both nodes |
| `SERVING_PORT` | OpenAI endpoint port (default 8000) |
| `MASTER_PORT` | torch distributed rendezvous port (default 29521) |
| `IMAGE` | built image tag (default `glm53:v9`) |
| `FABRIC_RANGE` | interconnect subnet, e.g. `10.0.0.0/24` |
| `IF_NAME` | fabric interface name on both nodes (`ip link`) |
| `NCCL_IB_HCA` | RoCE/IB HCA list (`ibdev2netdev`) |
| `NCCL_IB_GID_INDEX` | RoCE GID index (default 3) |

## Why a patched image (research, 2026-08-27)

- Model card: on GB10/sm_121 **vLLM ❌** (echoes prompt — sm_121 MLA kernel asserts
  `pe_dim==64`, model is NoPE `qk_rope_head_dim=0`); **SGLang ✅** is the card-blessed
  engine. Upstream vLLM lacks `glm5_next` (PR #53906 open); day-0 image
  `vllm/vllm-openai:glm53-flash-arm64-cu130` works on B200 only.
- **tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark** — world-first vLLM TP2 on
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
- `exec-vllm.sh` — container entrypoint, **default = current 512K config**
  (512K ctx, fp8 KV 9 GiB pin, MTP k=4, `--language-model-only`, gm 0.90).
  Alternative profiles are kept side-by-side and swapped by
  `cp exec-vllm-<profile>.sh exec-vllm.sh` on both nodes, then down/up:
  `exec-vllm-512k.sh` (512K + 10 GiB pin + MTP-3, 1.44M pool, 87/100),
  `exec-vllm-512k-k4.sh` (= the current default, 89/100),
  `exec-vllm-262k-fp8.sh` (262K + fp8 KV, 610K pool, multimodal ON, 90/100),
  `exec-vllm-v8-262k.sh` (day-0 262K + bf16 KV, 311K pool, 89/100).
- `build-image.sh` — head: clone tonyd2wild repo, build v1→v8, add v9 guard, ship to worker
- `fetch-weights.sh` — head: HF download + rsync to worker + checksum verify
- `cluster.sh` + `.env` — orchestrator-side control (preflight/up/down/status/logs)
- `example.env` — configuration template (copy to `.env`, fill in)
- `docker/lab-build.sh` — build `glm53:lab` from the vendored `lab/build/`
  context (digest-pinned base + 5 patches), on the head, then ship to the worker
- `lab/docker-compose-lab.yaml` + `lab/lab-launch.sh` + `lab/lab-watchdog.sh`
  — the **lab-quant stack** (standing config): takeover/up/down/status/logs
  from the orchestrator; watchdog auto-restarts the provider endpoint
- `lab/build/` — vendored build context for `glm53:lab`: digest-pinned
  Dockerfile + 5 patches + provenance (`UPSTREAM.md`) + recipe README (92/100
  hardmode on the upstream rig)
- `benchmarks.md` — investigation + benchmark log (quality, speed, pools, forensics)
- `NOTES-512k.md` — 512K upgrade notes + crash forensics + rollback

## Quick Start (run on the head node)

```bash
# 1. Clone and configure
git clone https://github.com/amasu/glm53-flash-cluster && cd glm53-flash-cluster
cp example.env .env
$EDITOR .env            # set WORKER_IP, HEAD_IP, FABRIC_RANGE, IF_NAME, NCCL_IB_HCA, REMOTE_DIR
set -a; source .env; set +a

# 2. Mirror the repo (incl. .env) to the worker at $REMOTE_DIR
ssh "$WORKER_IP" "mkdir -p $REMOTE_DIR"
rsync -a ./ "$WORKER_IP:$REMOTE_DIR/"

# 3. Build the image (~1-2 h) and ship it to the worker
./build-image.sh

# 4. Download weights (~182 GiB) and rsync them to the worker
./fetch-weights.sh
#    ...or, if weights are already staged on the head at $WEIGHTS_DIR:
#    ssh "$WORKER_IP" "mkdir -p $WEIGHTS_DIR"
#    rsync -a --info=progress2 "$WEIGHTS_DIR/" "$WORKER_IP:$WEIGHTS_DIR/"

# 5. Launch — worker (rank 1) first, then head (rank 0)
docker compose --env-file .env up -d glm53-worker
sleep 25
docker compose --env-file .env up -d glm53-head

# 6. Watch warmup (14-21 min), then smoke-test
docker logs -f vllm_glm53_head
curl "http://localhost:${SERVING_PORT:-8000}/v1/models"
```

`cluster.sh` stays for orchestration from a separate workstation (it ssh-hops
head → worker); from the head node, plain compose is enough.

## Deploy (from a workstation, via cluster.sh)

1. Configure: copy `example.env` → `.env` (orchestrator + both nodes), fill in
2. `./build-image.sh` on head → final image on both nodes (IDs verified identical)
3. `./fetch-weights.sh` on head → weights at `WEIGHTS_DIR` on both, sha256-verified
4. `cluster.sh preflight` → `cluster.sh up` → boot 14–21 min → `cluster.sh logs`
5. Smoke: `curl http://"$HEAD_HOST":"$SERVING_PORT"/v1/models`

## Standing config: the lab-quant stack (`lab/`)

The 90/100 lab quant replaces the LibertAIDAI stack as the standing config.
Same `.env` (plus the `LAB_*` / `WEIGHTS_DIR_LAB` / `MASTER_PORT_LAB` block in
`example.env`), same host-agnostic contract.

```bash
# 0. Clone, configure, mirror to both nodes (same .env as the main stack)
git clone https://github.com/amasu/glm53-flash-cluster && cd glm53-flash-cluster
cp example.env .env   # fill in HEAD_HOST, HEAD_IP, WORKER_IP, REMOTE_DIR,
                      # WEIGHTS_DIR_LAB, fabric vars (FABRIC_RANGE, IF_NAME, NCCL_IB_HCA)
set -a; source .env; set +a

# 1. Weights (~186 GB, local-inference-lab quant @ rev 378ca545…) on BOTH nodes
#    at $WEIGHTS_DIR_LAB (huggingface-cli download --revision 378ca54585c46542bad1f3cb3ed0d73ae51cdb62)

# 2. Build glm53:lab from the vendored context and ship it to the worker
./docker/lab-build.sh       # ~30 s + base pull; greps/verifies patch markers

# 3. Mirror the repo to both nodes at $REMOTE_DIR (compose + .env live there)
rsync -a ./ "$HEAD_HOST:$REMOTE_DIR/" && rsync -a ./ "$WORKER_IP:$REMOTE_DIR/"
#    on the head also stage the lab launch dir: ~/glm53-lab = $REMOTE_DIR/lab

# 4. Launch (worker rank 1 first, then head) — single-stack policy:
lab/lab-launch.sh takeover  # stops the old stack first, then brings lab up
#    or, if :8000 is already free: lab/lab-launch.sh up

# 5. Boot 14–21 min; check the boot markers in lab/build/README.md
#    (weight-load GiB, [quantprobe] algo=MXFP8, KV pool 1,164,369 tokens)
lab/lab-launch.sh status && lab/lab-launch.sh logs
```

Rollback to the LibertAIDAI stack: `lab/lab-launch.sh down` then
`cluster.sh up` (the `glm53:v9` image + old weights stay on both nodes).
Optionally run `lab/lab-watchdog.sh` from cron to auto-restart the lab
stack if the endpoint dies (it skips a live boot via container age).

**lab-vision variant (vision enabled, 2026-08-31):** `lab/docker-compose-lab-vision.yaml`
is the same profile with the multimodal front-end ON, shaped by 0rand's
proven config (forum #381350 #127/#130): `--skip-mm-profiling` (vision
tower costs ~60K KV tokens instead of ~300K) + modern-format
`--limit-mm-per-prompt` (legacy `{"video": 1}` is rejected by this vLLM
build). Select with `LAB_STACK=lab-vision lab/lab-launch.sh up` (or set
`LAB_STACK=lab-vision` in `.env` — explicit env vars win over `.env`).
Containers: `glm53-vision-head/worker`. `down` clears both stacks
(single-stack policy). Same image, same weights, same standing flags
(512K, 1024/16, MTP-3, gm 0.90) — the mm flags differ, and the KV pin is
**9 GiB** (`LAB_VISION_KV_CACHE_BYTES`): 10 GiB OOMs the warmup forward
with the mm front-end on the GB10 UMA line (benchmarks.md §9). Quality on
the clean same-harness A/B: 92.5 vs LMO 89.5 — no text-quality cost.

## Key serve flags (load-bearing — do not "clean up")

LibertAIDAI profile (`exec-vllm.sh`, 512K + MTP-4 + 9 GiB pin) — the rollback
path; the standing lab stack's equivalent flags are in
`lab/docker-compose-lab.yaml` (+ `--skip-mm-profiling`/`--limit-mm-per-prompt`
for `lab-vision`):

| flag | why |
|---|---|
| `--block-size 2304` | aligner default 2176 → kpool pages tile by 32 not 64; DeepGEMM arch-12 fp8 paged-MQA needs 64-entry pages |
| `--gpu-memory-utilization 0.90` | with a pinned KV pool the engine skips memory profiling; 0.90 leaves activation headroom for the warmup forward (lower values just waste UMA) |
| `--kv-cache-dtype fp8` + `--kv-cache-memory-bytes 9663676416` | SM90 NoPE path dequantizes in-kernel (`has_flashinfer_sm90_nope_mla()`); the pin makes the pool deterministic (1,261,444 tok) instead of GMU-dependent |
| `--max-model-len 524288` | 512K context; requires the v9 image's CC-12.x sparse-MLA indexer guard |
| `--max-num-batched-tokens 4096` | 8192 OOMs the GB10 driver at 512K shapes |
| `--kernel-config '{"enable_cutedsl_warmup":false,"enable_flashinfer_autotune":false}'` | autotune/cutedsl-warmup scratch at 512K shapes OOMs (`NV_ERR_NO_MEMORY`) |
| `--speculative-config mtp/4` | MTP head is BF16 in this checkpoint; per-position acceptance 0.81/0.67/0.51/0.42, acceptance length ~3.1–3.4. k is a speed knob only (MTP is lossless vs k); k=3 costs quality on this model (see benchmarks.md §6) |
| `--language-model-only` | drops the ~15.7 GiB multimodal front-end; without it the pinned-KV + gm-0.90 profile OOMs at warmup on the 121.69 GiB UMA line |
| `--moe-backend marlin` | card's known-good sm_121 fallback |
| `--enforce-eager` | CUDA graphs unvalidated on this arch; MTP still engages |
| `--tool-call-parser glm47` | `glm` fails SILENTLY (empty content, tool_calls null) |
| `--reasoning-parser glm45` | without it trace lands in `content` with bare `</think>` |
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
- KV headroom: local weights both nodes = +33% vs NFS; fp8 KV + pinned pool
  is the current default (1.26M-token pool; see `benchmarks.md` for the
  10 GiB / 1.44M-token variant and the pin-size vs draft-depth trade)

## Credits

This repo is an orchestration layer on top of the day-0 work of others — the
sm_121 patch chain, the debugging knowledge, and the failure folklore all come
from these sources:

**Patches & reference deploy**

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
  — world-first vLLM TP=2 on 2× DGX Spark. Built the v1→v8 sm_121 patch chain
  (NoPE-MLA for SM121, FlashInfer 0.6.18 nightly, NCCL 2.30.7 re-pin,
  cutlass-dsl 4.6.2, PDL off, indexer top-k hardening, fp8-KV smem-tile fix)
  that `build-image.sh` assembles and ships. Their repo is also the source of
  the ops rules in this README.
- Day-0 base image [`vllm/vllm-openai:glm53-flash-arm64-cu130`](https://hub.docker.com/r/vllm/vllm-openai)
  (vLLM team) and upstream [vLLM PR #53906](https://github.com/vllm-project/vllm/pull/53906)
  (`glm5_next` architecture support, open at the time of writing).
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — the b12x
  Spark serving stack; evaluated as an alternative carrier before choosing the
  tonyd2wild chain (its custom MLA backends target the `pe_dim==64` layout
  class this model can't use).

**Checkpoints & quantizations**

- Upstream model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
  (320B MoE, released 2026-08-26) — the checkpoints below are NVFP4
  quantizations of it.
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
  — the uniform-NVFP4 quant staged by `fetch-weights.sh` (the v8/v9 profiles,
  89–90/100).
- [local-inference-lab/GLM-5.3-Flash-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
  — mixed-precision quant (NVFP4 experts + MXFP8 MTP drafter) behind the
  current standing lab-quant stack `(glm53:lab)`, 90/100.
- [0rand/glm-5.3-flash-nvfp4-2x-dgx-sparks](https://github.com/0rand/glm-5.3-flash-nvfp4-2x-dgx-sparks)
  — packaging of the lab quant for 2× Spark; the `lab/` stack is built on it.

**512K-context & lab-quant recipes**

- NVIDIA forum thread [GLM-5.3-Flash: 320B total parameters / 18B active](https://forums.developer.nvidia.com/t/glm-5-3-flash-320b-total-parameters-18b-active/381350)
  — the community thread behind most of the current config:
  - **post 55 (0rand)** — the 512K-context recipe whose reference memory
    shaping `exec-vllm-512k.sh` and `lab/docker-compose-lab.yaml` follow.
  - **post #124 (0rand)** — the proven sampling config (temp 0.1, parallel 4,
    2 trials) and the `1024/16` batched-tokens/num-seqs server flags that our
    0rand-replica benchmark (benchmarks.md §8) reproduces within 1 pt.
- [FujitsuPolycom/glm53-flash-tp2-spark](https://github.com/FujitsuPolycom/glm53-flash-tp2-spark)
  (Apache-2.0) — two distinct contributions, both vendored in this repo's
  `lab/build/`:
  - the `sparse_attn_indexer*.patch` pair mirrored by
    `docker/patch_v9_512k.py` (CC-12.x sparse-MLA indexer guard; without it
    GB10 hard-aborts at 512K shapes — see `NOTES-512k.md`);
  - the `model.patch` + `modelopt.patch` lab-checkpoint patches (naming shim,
    MTP MIXED_PRECISION quantization fix) applied unmodified by
    `lab/build/Dockerfile`.
- **kilork** — the `local-inference-lab` mixed-precision recipe
  ([gist `a887667f4f423b7cc324859cd5e32ebd`](https://gist.github.com/kilork/a887667f4f423b7cc324859cd5e32ebd),
  92/100 hardmode, incl. the `--enforce-eager` quality finding) that the
  lab-quant swap follows; served boot markers (KV pool 1,164,369 tokens)
  match it exactly. See `lab/build/README.md`.
- [kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark)
  — the NoPE-MLA rope-pad + sm120 topk mod (`patch_mla.py`, vendored in
  `lab/build/`; load-bearing for `fp8_ds_mla` KV on sm_121, engaged via
  `VLLM_MLA_NOPE_PAD_ROPE=1`).

**Benchmark harness**

- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)
  — the harness that produces every quality score in this repo (seed-42
  hardmode, 88 scenarios); see benchmarks.md §Methodology for the protocols.

**NVIDIA DGX Spark / GB10 forum threads** (failure classes that shaped the ops
rules — launch order, both-rank teardown, no crash-looping):

- [Two-Spark cluster with vLLM using tensor-parallel-size 2 causes one node to
  drop while the other's GPU goes 100% forever](https://forums.developer.nvidia.com/t/two-spark-cluster-with-vllm-using-tensor-parallel-size-2-causes-one-node-to-drop-while-the-others-gpu-goes-100-forever/358755)
  — TP=2 node-drop after the first prompt; motivates the both-rank teardown and
  diagnose-don't-crash-loop rules.
- [Total host freeze (not process hang) during multi-node TP=2 vLLM prefill on
  2× DGX Spark GB10, zero forensic trace across kdump/watchdogs/netconsole](https://forums.developer.nvidia.com/t/total-host-freeze-not-process-hang-during-multi-node-tp-2-vllm-prefill-on-2x-dgx-spark-gb10-zero-forensic-trace-across-kdump-watchdogs-netconsole/376882)
  — heavy-prefill host freeze; motivates MTU 9000, the 262K day-1 context cap,
  and pre-launch drop_caches.

More GB10 collective wisdom: the
[DGX Spark / GB10 forum category](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719).
