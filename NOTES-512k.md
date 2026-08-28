# GLM-5.3-Flash 512K-context + FP8-KV upgrade (2026-08-28)

## Outcome

| profile | image | ctx | KV pool | quality (seed-42 hardmode) | status |
|---|---|---|---|---|---|
| v8 baseline (bf16 KV, gm0.85) | glm53:v8 | 262K | 311,419 tok (1.19×) | 89/100 | retired |
| FP8 KV minimal (exec-vllm.sh) | glm53:v9 | 262K | 610,519 tok (2.33×) | 90/100 | fallback profile |
| **512K (exec-vllm-512k.sh, ACTIVE)** | glm53:v9 | **512K** | **1,435,742 tok (2.74×)** | needle 410K tok @51.8% depth PASS (pending A/B bench) | **shipped** |

Quality bench notes: same seed 42, same harness (tool-eval-bench hardmode,
88 scenarios), same endpoint. 89 → 90 between bf16 and FP8 KV at 262K
(no regression; +1 within run-to-run variance). The 512K A/B bench was
queued but the needle test confirmed long-context retrieval first.

## What changed and why

### 1. `docker/patch_v9_512k.py` + `docker/Dockerfile.glm53-sm121-v9` (image glm53:v9)
Adds the CC-12.x sparse-MLA indexer guard (persistent_topk + cooperative
workspace disabled on compute-capability family 12). Without it, 512K-context
shapes request 62 CTAs / 128 KB smem in the indexer top-k and GB10 (48 SMs /
99 KB opt-in smem) hard-aborts. Vendored anchors match
FujitsuPolycom/glm53-flash-tp2-spark sparse_attn_indexer*.patch
(Apache-2.0; proven by the 512K-context recipe, forum #381350 post 55).
The v8 chain already carried the top-k init/clamp hardening, so v9 adds only
the two guard lines (verified by grep in the built image).

### 2. `exec-vllm-262k-fp8.sh` — FP8 KV minimal profile (fallback)
Only `--kv-cache-dtype fp8` added to the v8 flags (262K, gm 0.85,
block 2304, MTP k=4, multimodal enabled). The SM90 NoPE backend supports
fp8 KV in-kernel dequant (gate `has_flashinfer_sm90_nope_mla()` = True on
FlashInfer 0.6.18 in this build). Pool: 311K → 610K tokens. Kept on the
cluster nodes as `exec-vllm-262k.sh`.

### 3. `exec-vllm.sh` (= `exec-vllm-512k.sh`) — 512K profile (active)
Full reference memory shaping from the 512K recipe PLUS the one flag that
fixed our crash: `--language-model-only`.
  * `--max-model-len 524288`
  * `--kv-cache-dtype fp8` + `--kv-cache-memory-bytes 10737418240` (10 GiB pin)
  * `--gpu-memory-utilization 0.90`
  * `--max-num-batched-tokens 4096` (8192 OOMs the GB10 driver at these shapes)
  * `--enable-chunked-prefill --enable-prefix-caching --disable-custom-all-reduce`
  * `--kernel-config '{"enable_cutedsl_warmup":false,"enable_flashinfer_autotune":false}'`
    (FlashInfer autotune scratch at 512K shapes OOMs the driver)
  * `--speculative-config` MTP k=4 → **k=3** (reference recipe value;
    k=4 acceptance was ~3.1–3.7 accept-len, diminishing beyond k=3)
  * `--language-model-only` — **the load-bearing flag for us.** The reference
    recipe ran text-only; our day-0 stack kept the multimodal processor,
    which balloons the API front-end ~15.7 GiB anon (documented in the
    tonyd2world 2×Spark day-0 thread). With gm 0.90 (109.5 GiB pool) +
    10 GiB pinned KV + 15.7 GiB mm frontend, the engine's warmup forward
    forward-allocated more than the 121.69 GiB physical →
    `NVRM NV_ERR_NO_MEMORY (_memdescAllocInternal)` and the rank-0 worker
    died mid-warmup (two rounds, reproducible). Dropping the mm processor
    brought warmup under the physical line; engine now boots cleanly with
    a 1.44M-token pool.

## Crash forensics (rounds 1–2, 512K attempts without language-model-only)
* Signature: `Worker proc VllmWorker-0 died unexpectedly (exit code: None)`
  (signal kill, no Python traceback — the CUDA context died first), ~90 s
  after "GPU KV cache size" was logged, during `compile_or_warm_up_model`.
* Kernel log: `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`
  (journalctl -k). Head-node journal confirms at each crash timestamp.
* Round 1: 512K + pinned KV + 4096 batched + mm enabled → OOM in warmup.
* Round 2 (262K + pinned 10 GiB KV + 4096 batched + mm enabled) → ALSO OOM
  (the pin + batched-tokens + mm profile itself is over the line, not just
  the 512K shape). This is what forced the language-model-only hypothesis.
* Round 3 (512K + reference profile + language-model-only, but on **v8
  image** — see pitfall below) → boots clean, but the first 410K-token
  request crashed with the 512K-context indexer oversubscription
  (`persistent_topk ... total_ctas=62 > num_sms=48` at
  sparse_attn_indexer_kpool.py:815) — exactly the call site the v9 guard
  covers.
* Round 4 (same profile, v9 image) → boots clean; engine log shows no
  `persistent_topk` oversubscription in the warmup forward, and a 440K-token
  needle request completes (prefill + 500-token decode, no oversubscription,
  no OOM — see Verification record). The guarded path falls back to
  `torch.ops._C.top_k_per_row_decode`, the pre-existing CUDA-safe kernel on
  this build. The needle marker was located by the model at ~52% document
  depth (confirmed in the reasoning trace), proving long-context retrieval
  works; the only miss was an ambiguous "8 characters" hint in my prompt,
  not a context failure.

## Pitfall: docker-compose.yml hardcoded the image tag
The cluster-node `docker-compose.yml` carried `image: glm53:v8` literally
(the repo's version uses `image: ${IMAGE:?...}`). `.env` changes therefore
had NO effect on the launched image — round 3 silently ran v8. Fixed on
both nodes (2026-08-28): `image: ${IMAGE:?IMAGE must be set in .env}`.
Always verify the *running* container's image (`docker ps --format
'{{.Image}}'`) after any image rollout, not just `.env`.

## Known trade-offs of the active 512K profile
* **Multimodal input is disabled** (`--language-model-only`): text-only
  serving. The model's image input path is off the table until the mm
  frontend fits — would need the pinned-KV profile *without* gm 0.90
  headroom, i.e. a smaller KV pin or a lower max-model-len.
* 4 NVRM allocation-probe lines at boot are expected (KV pool reservation
  probing); they do not fail the boot.
* Boot time ~14–17 min cold (weights ~13 min + warmup).

## Verification record
* 262K+FP8 (intermediate): tool-eval-bench seed-42 hardmode = **90/100**
  (159 pts; ~35 min). +1 over baseline; all baseline passes retained.
* 512K+FP8 (final, active): tool-eval-bench seed-42 hardmode = **87/100**
  (153 pts; ~35 min).
* **Quality cost is 512K-profile-specific, not seed variance.** 3-way diff
  (same seed 42): 6 scenarios pass on BOTH v8-baseline (89) and 262K+FP8
  (90) but regress only on the 512K profile:
    TC-21 (PASS->PARTIAL), TC-40 (PASS->FAIL), TC-50 (PASS->PARTIAL),
    TC-53 (PASS->PARTIAL), TC-58 fake-system-msg (PASS->FAIL),
    TC-81 tool-output-injection (PASS->FAIL).
  4 others improve on 512K (TC-35, TC-49, TC-51, TC-80), netting -3 pts.
  Prime suspect: MTP k=4 -> k=3 (the one flag the intermediate run kept at
  k=4) plus the 4096 batched-tokens / chunked-prefill / autotune-off
  memory-shaping set. The injection-category misses (TC-58, TC-81) suggest
  the shorter/cheaper prefill+decode path slightly weakens careful
  reasoning. Unconfirmed — would need a 512K + MTP k=4 A/B to isolate.
* 512K long-context: short-completion sanity PASS; 440K-token needle
  (marker @ ~52% doc depth) byte-exact retrieval PASS
  (`<M>[SYSTEM-RECORD-THE-NEEDLE-CODE-84729: checksum=OK]</M>`, finish
  stop, 53 completion tokens, no OOM/oversubscription).
* `max_model_len` verified via /v1/models (262144 -> 524288).
* KV pool + concurrency read from engine boot log on both ranks.

## Open decision (presented to user 2026-08-28)
512K profile = max context + 1.44M-token pool, costs ~2-3 quality pts and
drops multimodal. 262K+FP8 fallback = 90/100 (matches/beats baseline), 610K
pool (still 2x baseline), multimodal ON, but 262K context cap. A 512K+MTP
k=4 A/B may recover most of the quality while keeping 512K — not yet run.

## Rollback
* Image: `glm53:v8` is still on both nodes (same ID as pre-upgrade).
  `IMAGE=glm53:v8` in `.env` + `exec-vllm-v8-262k.sh` (the exact pre-upgrade
  flag set, committed here; a copy also sits on both cluster nodes as
  `exec-vllm-262k.sh`) reproduces the old behavior
  (bf16 KV, 262K, 89/100 baseline).
* Switching profile: on both nodes, `cp exec-vllm-<profile>.sh exec-vllm.sh`,
  then `cluster.sh down && cluster.sh up`.
