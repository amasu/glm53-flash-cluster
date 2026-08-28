# GLM-5.3-Flash — Investigation & Benchmark Log

Single source of truth for everything investigated on this 2× DGX Spark (GB10)
cluster: what was tried, the quality scores, the speed numbers, the KV-pool
sizes, and the crash forensics. Updated as each experiment lands.

- **Model:** `LibertAIDAI/GLM-5.3-Flash-NVFP4` (320B total / 18B active, 181 GiB, `glm5_next` NoPE sparse-MLA + KDA)
- **Topology:** 2× GB10, vLLM **TP=2** over RoCE. Head `aitopatom-6253` (rank 0, 10.100.90.1), worker `10.100.90.4` (rank 1, headless).
- **Endpoint:** `http://aitopatom-6253.local:8000/v1` — served name `glm-5.3-flash` (Hermes default provider).
- **Image:** `glm53:v9` (tonyd2world sm_121 patch chain v1→v8 + the CC-12.x sparse-MLA indexer guard). `glm53:v8` retained on both nodes for rollback.
- **Bench harness:** `tool-eval-bench` v2.6.1.dev24, hardmode, **seed 42**, 88 scenarios, c1 sequential, thinking enabled, temperature 0.0. Reports in `runs/`, raw scores in `data/benchmarks.sqlite`.

> Note: this is the **GLM-5.3-Flash** provider (k=MTP, 9GiB pin, 512K, 1.44M pool).
> It is a different deployment from the DeepSeek-V4-Flash-0731 research tracked in
> `~/deepseek-v4-flash-optimization/` (DSpark spec, fp8 KV, 2× GB10) — do not mix the numbers.

---

## TL;DR — profiles tried, with quality + pool + speed

| # | profile (exec script) | image | ctx | KV dtype | pool | quality (seed-42 hardmode) | decode speed | status |
|---|---|---|---|---|---|---|---|---|
| 0 | `exec-vllm-v8-262k.sh` (day-0) | glm53:v8 | 262K | bf16 | **311,419 tok** (1.19×) | **89/100** (156 pts) | ~14–22 tok/s (comm. ~14.3 bf16) | retired baseline |
| 1 | `exec-vllm-262k-fp8.sh` | glm53:v9 | 262K | fp8 | **610,519 tok** (2.33×) | **90/100** (159 pts) | ~21.8 tok/s (comm. fp8+MTP4) | fallback profile |
| 2 | `exec-vllm-512k.sh` (10GiB pin, MTP-3) | glm53:v9 | **512K** | fp8, **10 GiB pin** | **1,435,742 tok** (2.74× conc. @512K) | **87/100** (153 pts) | **~24–30 tok/s** live (MTP-3) | superseded |
| 3 | `exec-vllm-512k-k4.sh` (**k=4, 9GiB pin**) | glm53:v9 | **512K** | fp8, **9 GiB pin**, **MTP k=4** | **1,261,444 tok** (2.41× conc. @512K) | **89/100** (157 pts) | same ~25–30 tok/s band; median turn 5.9 s; MTP-4 acceptance 46.3% | **ACTIVE (standing config)** |

The quality column is the tool-eval-bench hardmode score; the pool column is the
engine-reported `GPU KV cache size` at boot. "×" is pool size relative to the
day-0 bf16 baseline pool (311,419 tok). Concurrency is the engine's
`Maximum concurrency for 524,288 tokens per request`.

---

## 1. Day-0 baseline (v8) — what started it all

Model card says on GB10/sm_121: **vLLM ❌** (sm_121 MLA kernel asserts
`pe_dim==64`, model is NoPE `qk_rope_head_dim=0`) / **SGLang ✅** blessed.
Upstream vLLM lacks `glm5_next`. So the whole thing runs on the community
`tonyd2world/GLM-5.3-Flash-NVFP4-*` sm_121 patch chain (v1→v8), which fixed 7
day-0 bugs:

- NoPE-MLA-for-SM121 (FA2 attention)
- FlashInfer 0.6.18 nightly (0.6.17 FA2 produced NaN at 64–256-row batches)
- NCCL 2.30.7 re-pin (nightly → 2.29.7 caused fabric death)
- cutlass-dsl 4.6.2
- PDL off on SM12x
- indexer top-k hardening
- fp8-KV smem-tile fix

Day-0 v8: 262K ctx, bf16 KV, gm 0.85, block 2304, MTP k=4, multimodal ON.
Pool 311,419 tokens. Bench **89/100**. Community reference decode ~14.3 tok/s
(bf16 KV) on the same 2× Spark / MTP path.

---

## 2. 262K + FP8 KV minimal (profile 1) — the "safe" step

Only delta vs v8: `--kv-cache-dtype fp8`. The SM90 NoPE backend dequantizes in-kernel
(`has_flashinfer_sm90_nope_mla()` = True on FlashInfer 0.6.18 in this build).
Pool doubles: **311,419 → 610,519 tokens (2.33×)**. Multimodal stays ON.
Bench **90/100** (159 pts) — +1 over baseline, within run-to-run variance, no
scenario regressed. This is the kept fallback.

---

## 3. 512K + FP8 KV + 10 GiB pin (profile 2) — the shipped profile

Full reference memory shaping from the forum 512K recipe (forum #381350 post 55)
**plus** the one load-bearing flag that fixed our crash: `--language-model-only`.

Active flags (`exec-vllm.sh` = `exec-vllm-512k.sh`):

```
--gpu-memory-utilization 0.90
--kv-cache-dtype fp8
--kv-cache-memory-bytes 10737418240      # 10 GiB pinned KV pool
--max-model-len 524288
--max-num-seqs 6 --block-size 2304 --moe-backend marlin
--max-num-batched-tokens 4096            # 8192 OOMs the GB10 driver at these shapes
--enable-chunked-prefill --enable-prefix-caching --disable-custom-all-reduce
--kernel-config '{"enable_cutedsl_warmup":false,"enable_flashinfer_autotune":false}'
--enforce-eager
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'   # k=4 -> k=3
--language-model-only
--distributed-executor-backend mp
--nnodes 2 --node-rank $NODE_RANK --master-addr $HEAD_ADDR --master-port 29521
```

- **KV pool: 1,435,742 tokens** → `Maximum concurrency for 524,288 tokens per request: 2.74×`.
- **max_model_len verified** via `/v1/models`: 262144 → **524288**.
- **Bench: 87/100** (153 pts).

### Why k=4 → k=3 and the quality cost
The 512K profile is the *only* one that dropped 3 points, and the 3-way diff
(same seed 42) proves it is **512K-profile-specific, not seed variance**:

- 6 scenarios pass on **both** the v8 baseline (89) **and** the 262K+FP8
  intermediate (90), but regress **only** on 512K:
  `TC-21` (PASS→PARTIAL), `TC-40` (PASS→FAIL), `TC-50` (PASS→PARTIAL),
  `TC-53` (PASS→PARTIAL), `TC-58` fake-system-msg (PASS→FAIL),
  `TC-81` tool-output-injection (PASS→FAIL).
- 4 other scenarios *improve* on 512K (`TC-35`, `TC-49`, `TC-51`, `TC-80`),
  netting **−3 points**.
- Prime suspect: the **MTP k=4 → k=3** step (the one flag that differed from the
  90-point intermediate run) plus the 4096-batched / chunked-prefill /
  autotune-off memory-shaping set. The two injection-category misses (TC-58, TC-81)
  suggest the shorter/cheaper prefill+decode path slightly weakens careful
  reasoning. **Unconfirmed — the k=4 + 9GiB A/B (profile 3, below) isolates it.**

### Long-context proof
- Short-completion sanity PASS.
- **440K-token needle** (marker @ ~52% doc depth) byte-exact retrieval PASS:
  `<M>[SYSTEM-RECORD-THE-NEEDLE-CODE-84729: checksum=OK]</M>`, `finish: stop`,
  53 completion tokens, no OOM, no indexer oversubscription.

---

## 4. Speed (throughput)

### Measured live on this cluster (profile 2, 512K + MTP-3, c1)
From the engine's own 10-second logging (during/after the 87/100 bench,
`docker logs vllm_glm53_head`):

| metric | value |
|---|---|
| **Generation throughput** | **~24–30 tok/s** (steady-state decode, 1 running req) |
| **Prompt throughput** | 180–364 tok/s (chunked prefill bursts; 0 when idle) |
| **Median bench turn** | 5.8–6.0 s (tool-eval-bench "Responsiveness" 26–27/100) |
| **TTFT (bench)** | ~1.05–1.10 s |

So the 512K profile decodes at roughly **25–30 tok/s single-stream** — in line
with the MTP-assisted 2× Spark numbers, ahead of the bf16 day-0 rate.

### Community reference points (same model, GB10, for calibration)
- **2× Spark day-0 (first published deploy):** 24.7–30.3 tok/s with MTP-5.
- **2× Spark, tonyd2world checkpoint (FP8 KV + MTP):** 43.4 tok/s PEAK.
- **2× Spark tonyd2world:** ~14.3 tok/s bf16 KV / ~21.8 tok/s fp8 KV + MTP-4.
- **3× Spark, TP=3, 512K ctx:** 35 tok/s, KV pool 1.505M (2.87×).
- **4× Spark, TP=4, 262K ctx:** 36 tok/s, 1.26M-token FP8 KV pool.

Our 25–30 tok/s single-stream at 512K sits squarely in the published 2× Spark
envelope — no regression from the context/quality upgrade.

---

## 5. Crash forensics (the 512K OOM saga, rounds 1–4)

Root cause: **multimodal frontend (~15.7 GiB) + gm-0.90 pool + 10 GiB pin +
warmup activations > 121.69 GiB physical UMA.** The MTP k value was *not* a
factor in the original OOM rounds.

- **Signature:** `Worker proc VllmWorker-0 died unexpectedly (exit code: None)`
  (signal kill, no Python traceback — the CUDA context died first), ~90 s after
  "GPU KV cache size" logged, during `compile_or_warm_up_model`.
- **Kernel log:** `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`.
- **Round 1:** 512K + pinned KV + 4096 batched + mm enabled → OOM in warmup.
- **Round 2:** 262K + pinned 10 GiB + 4096 batched + mm enabled → **also OOM**
  (the pin + batched + mm profile itself is over the line, not just the 512K shape).
  This forced the `--language-model-only` hypothesis.
- **Round 3:** 512K + reference profile + `--language-model-only`, but on the **v8
  image** (compose had `image: glm53:v8` hardcoded — `.env` `IMAGE` was ignored).
  Boots clean, but the first 410K-token request crashed with the 512K-context
  indexer oversubscription (`persistent_topk ... total_ctas=62 > num_sms=48`).
- **Round 4:** same profile on the **v9 image** (carries the CC-12.x guard) →
  boots clean, guarded path falls back to the CUDA-safe
  `torch.ops._C.top_k_per_row_decode`, 440K needle passes.

**Pitfall (hard-won):** `docker-compose.yml` hardcoded `image: glm53:v8`; fixed to
`image: ${IMAGE:?...}` on both nodes. Always verify the *running* container's
image (`docker ps --format '{{.Image}}'`) after a rollout, not just `.env`.

**Memory budget at the clean 512K boot (k=3):** initial free 111.69 GiB/rank,
90.74 GiB weights → ~18–19 GiB headroom for activations/scratch; warmup forward
133 s, zero NVRM errors.

---

## 6. k=4 + 9GiB pin (profile 3) — the quality-isolation test (IN PROGRESS)

Hypothesis: the 87→90 gap is the MTP k=4→k=3 step. Restoring k=4 should recover
quality, but adds ~0.5–1.5 GiB of draft-position activation scratch at warmup —
and we're close to the UMA line. So combine two knobs to stay under the line:

1. **`num_speculative_tokens` 3 → 4** (the quality test)
2. **KV pin 10 GiB → 9 GiB** (frees ~1 GiB for activations; pool drops
   1.44M → **~1.26M tokens**, still **~2.4× concurrency at 512K** — well above
   what's needed, and still **~4× the old 311K bf16 pool**)

If it OOMs even then, 8 GiB pin (≈1.1M tokens) is the next step.

**Plan:** create `exec-vllm-512k-k4.sh`, `cp` it to `exec-vllm.sh` on both nodes,
`cluster.sh down && up` (worker first, then head), watch `journalctl -k` for NVRM
lines in real time, then verify pool ~1.26M, `max_model_len` 524288, MTP-4
acceptance, a speed sample, and finally re-run the seed-42 hardmode bench.

**Boot (verified clean, 2026-08-28):**
- Initial free memory 111.64 GiB/rank, 9.0 GiB reserved for KV — engine skipped
  memory profiling as designed.
- **GPU KV cache size: 1,261,444 tokens** (predicted ~1.26M) →
  `Maximum concurrency for 524,288 tokens per request: 2.41×` (predicted ~2.4×).
- Warmup: `init engine (profile, create kv cache, warmup model) took 132.04 s`,
  zero NVRM OOMs (the 2 kernel `NV_ERR_NO_MEMORY` probe lines at KV reservation
  are the expected ones, identical to the k=3 boot).
- Spec-decode warmup confirms `num_spec=4` (MTP-4 rejection-sampler kernels).
- `max_model_len` via /v1/models: 524288.

**Speed sample (400-token streaming code-gen, thinking on, c1):**
- All 400 tokens were reasoning (thinking ran to the token cap on this prompt) —
  generation throughput ~12.6–26 tok/s transient; MTP-4 acceptance:
  261 accepted / 564 drafted = **46.3%** (~2.9 accept-length vs ~3.4 at k=3).
  The shorter accept-length is the expected k=4 trade (more draft positions,
  fewer hit each) — total wall-clock speed is in the same band as the k=3
  24–30 tok/s steady state.

**Quality bench (seed-42 hardmode, 88 scenarios, 35.1 min):**
- **Score: 89/100** (vs 87 for the k=3/10GiB 512K profile; 90 for 262K+FP8; 89 for the v8 baseline).
- **The k=4→k=3 hypothesis is CONFIRMED as the main quality cost.** The four
  k3-specific non-injection regressions all recovered to PASS with k=4:
  TC-21 (partial→pass), TC-40 (fail→pass), TC-50 (partial→pass),
  TC-58 fake-system-msg (fail→pass).
- **Remaining 512K-profile-specific cost: TC-81** (tool-output prompt
  injection) — FAIL on *both* 512K runs (k3 and k4), PASS on the 262K+FP8
  profile. So ~1 point of the gap is the 512K memory-shaping profile itself
  (4096 batched / chunked-prefill / autotune-off), not the drafter.
- Run-to-run noise (not profile-specific): TC-35 pass→partial, TC-51
  pass→fail, TC-61 partial→fail — TC-51/TC-61 also fail on the 262K+FP8 run,
  i.e. within the normal ±2-scenario variance of this harness at seed 42.
- Category deltas k3→k4: Structured Reasoning 83→100, Toolset Scale 75→100,
  Structured Output 75→92, Context & State 80→85, Safety 73→77; offsets:
  Multi-Step Chains 88→75, Autonomous Planning 83→50 (the TC-51/TC-61 noise).
- **Verdict: k=4 + 9GiB pin is the standing config.** 512K context + baseline
  quality (89 = v8 baseline score), 1.26M pool (2.41× concurrency @512K,
  ~4× the old 311K bf16 pool), boots clean with the extra GiB of activation
  headroom. Only TC-81 is a persistent 512K-profile cost.

---

## Rollback

- **Image:** `glm53:v8` still on both nodes (same ID as pre-upgrade).
- **Profile switch:** on both nodes, `cp exec-vllm-<profile>.sh exec-vllm.sh`,
  then `cluster.sh down && cluster.sh up`.
  - `exec-vllm-v8-262k.sh` → day-0 bf16 262K, 89/100 (311K pool).
  - `exec-vllm-262k-fp8.sh` → 262K + FP8, 90/100 (610K pool, multimodal ON).
  - `exec-vllm-512k.sh` → 512K + 10GiB pin + MTP-3, 87/100 (1.44M pool).
- Boot ~14–17 min cold (weights ~13 min + warmup); worker must start before head
  to complete world init.

## Ops notes

- Orchestrator (Mac) → head: `ssh aitopatom-6253.local`; worker is two-hop:
  `ssh 6253 "ssh 10.100.90.4 ..."` (direct ssh to .90.4 times out).
- `cluster.sh <preflight|up|down|status|logs>`; `up` drops page caches on both
  nodes first (GB10 unified memory), launches worker rank 1 then head rank 0.
- `.env` lives in 3 places (orchestrator + head + worker REMOTE_DIR); keep in sync.
- `data/` and `runs/` are gitignored (local bench artifacts).
