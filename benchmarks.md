# GLM-5.3-Flash — Investigation & Benchmark Log

Single source of truth for everything investigated on this 2× DGX Spark (GB10)
cluster: what was tried, the quality scores, the speed numbers, the KV-pool
sizes, and the crash forensics. Updated as each experiment lands.

- **Model (quant):** **ACTIVE = `local-inference-lab/GLM-5.3-Flash-NVFP4`** @ rev
  `378ca545…` (MIXED_PRECISION: NVFP4 experts + MXFP8 MTP drafter, 186 GB).
  Retired: `LibertAIDAI/GLM-5.3-Flash-NVFP4` (uniform NVFP4, 182 GiB) — kept for rollback.
  Both: 320B total / 18B active, `glm5_next` NoPE sparse-MLA + KDA.
- **Topology:** 2× GB10, vLLM **TP=2** over RoCE. Head `aitopatom-6253` (rank 0, 10.100.90.1), worker `10.100.90.4` (rank 1, headless).
- **Endpoint:** `http://aitopatom-6253.local:8000/v1` — served name `glm-5.3-flash` (Hermes default provider).
- **Image (ACTIVE):** `glm53:lab` (day-0 `vllm-openai:glm53-flash-arm64-cu130` @ digest `905c0293…` + 5 lab patches: modelopt MTP-namespace fix + quantprobe, naming shim, CC-12.x sparse-MLA indexer guards, NoPE-MLA rope-pad). `glm53:v9` (tonyd2wild chain) retained on both nodes for rollback.
- **Bench harness:** `tool-eval-bench` (dev24 → dev32 over the log's lifetime; see §Methodology), hardmode, **seed 42**, 88 scenarios. Two protocols were used: **c1/greedy** (standing quality protocol) and the **0rand-param replica** (parallel 4, trials 2, temp 0.1 — see §8).
- **Watchdog:** `lab/lab-watchdog.sh` runs every 15 min (Mac cron job `glm-lab-watchdog`); probes :8000, auto-restarts the stack if down.

> Note: this is the **GLM-5.3-Flash** provider (lab-quant, 10 GiB pin, 512K, 1.16M pool).
> It is a different deployment from the DeepSeek-V4-Flash-0731 research tracked in
> `~/deepseek-v4-flash-optimization/` (DSpark spec, fp8 KV, 2× GB10) — do not mix the numbers.

---

## TL;DR — profiles tried, with quality + pool + speed

| # | profile | image | quant | ctx | KV dtype | pool | quality (seed-42 hardmode, c1 greedy) | decode speed | status |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `exec-vllm-v8-262k.sh` (day-0) | glm53:v8 | LibertAIDAI | 262K | bf16 | **311,419 tok** (1.19×) | **89/100** (156 pts) | ~14–22 tok/s (comm. ~14.3 bf16) | retired baseline |
| 1 | `exec-vllm-262k-fp8.sh` | glm53:v9 | LibertAIDAI | 262K | fp8 | **610,519 tok** (2.33×) | **90/100** (159 pts) | ~21.8 tok/s (comm. fp8+MTP4) | fallback profile |
| 2 | `exec-vllm-512k.sh` (10GiB pin, MTP-3) | glm53:v9 | LibertAIDAI | **512K** | fp8, **10 GiB pin** | **1,435,742 tok** (2.74× conc. @512K) | **87/100** (153 pts) | **~24–30 tok/s** live (MTP-3) | superseded |
| 3 | `exec-vllm-512k-k4.sh` (**k=4, 9GiB pin**) | glm53:v9 | LibertAIDAI | **512K** | fp8, **9 GiB pin**, **MTP k=4** | **1,261,444 tok** (2.41× conc. @512K) | **89/100** (157 pts) | ~25–30 tok/s; median turn 5.9 s; MTP-4 acceptance 46.3% | retired (superseded by 4) |
| **4** | **`lab/docker-compose-lab.yaml` (lab-quant, MTP-3, 1024 batch)** | **glm53:lab** | **lab MIXED** | **512K** | **fp8_ds_mla, 10 GiB pin**, block 256 | **1,164,369 tok** (2.22× conc. @512K) | **90/100** (156/174) c1 greedy; **90/100** (158/176) 0rand-param replica | 24–30 tok/s band; MTP-3 accept ~2.8–3.0, 61–67% draft acceptance @ c1 | **ACTIVE (standing config)** |

The quality column is the tool-eval-bench hardmode score; the pool column is the
engine-reported `GPU KV cache size` at boot. "×" is pool size relative to the
day-0 bf16 baseline pool (311,419 tok). Concurrency is the engine's
`Maximum concurrency for 524,288 tokens per request`.

---

## 1. Day-0 baseline (v8) — what started it all

Model card says on GB10/sm_121: **vLLM ❌** (sm_121 MLA kernel asserts
`pe_dim==64`, model is NoPE `qk_rope_head_dim=0`) / **SGLang ✅** blessed.
Upstream vLLM lacks `glm5_next`. So the whole thing runs on the community
`tonyd2wild/GLM-5.3-Flash-NVFP4-*` sm_121 patch chain (v1→v8), which fixed 7
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
- **2× Spark, tonyd2wild checkpoint (FP8 KV + MTP):** 43.4 tok/s PEAK.
- **2× Spark tonyd2wild:** ~14.3 tok/s bf16 KV / ~21.8 tok/s fp8 KV + MTP-4.
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

## 7. Sampler A/B — greedy vs vendor-recommended vs moderate (2026-08-29)

Question: can temperature/top_p improve output quality over the standing greedy
serving config? Method: `tool-eval-bench` 69-scenario suite, seed 42, same box,
same protocol, three payload-only sampler variants (no restarts):

| Sampler | Score | Runtime |
|---|---|---|
| **Greedy (temp 0)** — standing config | **89/100** | 1616 s |
| Vendor-recommended (temp 1.0, top_p 0.95) | 88/100 | 1835 s |
| Moderate (temp 0.6, top_p 0.95) | 86/100 | 1708 s |

- **13/69 scenarios flip between samplers — both directions** (TC-33: greedy
  ❌ → t1.0 ✅; TC-38: greedy ✅ → t1.0 ❌). Sampling noise, not signal; the
  1–3 point spread is inside this harness's ±2-scenario seed-42 variance.
- **Safety: TC-58 (fake system message in file)** — greedy and t0.6 fully PASS;
  t1.0 drops to PARTIAL. Consistent with the Qwen3.8-27B finding (the
  thinking-mode sampler regressed TC-58 there too): hotter sampling weakens
  injection resistance.
- Runtime cost of hot sampling: +10–13% wall clock.
- **Verdict: greedy stays.** Vendor temp-1.0 guidance targets chat-style use;
  for tool-calling/agent work greedy is empirically superior on this stack.
  The knob that actually moves quality here is `reasoning_effort`, not
  temperature.

Reports: `~/tool-eval-runs/runs/2026/08/` labels `glm53-sampler-{greedy,
t10-p095,t06-p095}`; raw logs beside them. Run by Hermes (tool-eval-bench v2.6).

### 7.1 `reasoning_effort=max` addendum (same day)

Probe confirmed the knob works: default ≈ deep thinking (1,038 completion tok
on a test prompt), `low`/`high` shallow (~400 tok), `max` deepest (1,150 tok).
Full 69-scenario run with `--backend-kwargs '{"reasoning_effort": "max"}'`:

| Config | Score | Runtime |
|---|---|---|
| Default (greedy, deep thinking) | 89/100 | 1616 s |
| `reasoning_effort=max` | 88/100 | 1610 s |

**Verdict: no gain.** The default profile already reasons at near-max depth on
tool-call scenarios, so forcing max changes nothing measurable — 88 vs 89 is
harness variance (±2 at seed 42). Downgrading effort to `low`/`high` would cut
thinking tokens ~60% for latency gains, at unknown quality cost (untested).
Default greedy remains the standing config.

---

## Methodology — how the scores are produced, and how to read them

**Harness.** `tool-eval-bench` (SeraphimSerapis), hardmode suite (88 scenarios,
Categories A–P, pass=2 / partial=1 / fail=0 pts; final score = points/max × 100).
Runs are stored in `~/tool-eval-runs/data/benchmarks.sqlite` (`scenario_runs`,
per-scenario verdicts + notes) with markdown reports under
`~/tool-eval-runs/runs/2026/08/`. This cluster's older runs are also mirrored in
`data/benchmarks.sqlite` here.

**Standing quality protocol (used for profiles 0–3, and profile 4's first bench):**
`--seed 42 --hardmode --backend vllm --base-url http://aitopatom-6253.local:8000/v1`,
**c1 sequential**, thinking ON, **temperature 0.0 (greedy)**, `max_turns 8`,
request timeout 120 s, ~35 min per run. All profile comparisons in §3–§7 use this
protocol, so they are apples-to-apples.

**0rand-parameter protocol (used for the §8 replica):** matches
0rand's forum 381350 #124 command — `--parallel 4 --trials 2 --timeout 360
--max-turns 32`, `chat_template_kwargs={"thinking":true,"reasoning_effort":"max"}`,
`temperature 0.1`, `top_p 1` (top_k left at default/unrestricted on both rigs).
The harness itself warns that `--parallel >1` can cause server-saturation timeouts
recorded as FAIL even when the model reasoned correctly — so **parallel-protocol
scores are comparable to 0rand's parallel scores, but not to c1 scores**.

**What makes "same benchmark" give different scores (documented empirically, 2026-08-30/31):**
1. **Decoding is not deterministic** even at temp 0: FP reduction order varies with
   batch composition; speculative decoding (MTP) adds accept/reject stochasticity;
   cluster state (clocks, page cache, concurrency) perturbs logits.
2. **Sampling params** — temp 0.1 vs 0.0, top_p, `reasoning_effort`: our A/Bs (§7)
   show greedy ≈ best for this stack; 0rand's temp 0.1/effort-max flips individual
   scenarios without changing the headline much.
3. **Concurrency** — c1 vs `--parallel 4` changes batch composition (see 1) and
   latency; the 0rand-param run's two trials scored **89 and 90** individually.
4. **Trials/averaging** — `--trials 2` reports mean ± CI plus Pass@2 (ceiling) /
   Pass^2 (reliability floor); a single run has no variance estimate.
5. **Harness version & denominator** — dev24 vs dev25 vs dev32 change scenario
   point budgets (max 176 → 174 → 176 observed across runs); never compare totals
   across different denominators.
6. **Transient infra** — client connection drops / saturation timeouts count as
   scenario FAILs (observed: TC-50 "All connection attempts failed" in one c1 run;
   the 0/100 run of 2026-08-30 was an engine-down artifact, not a model result).

**Noise floor.** At seed 42, run-to-run scenario flips of ±2–3 (≈ ±1–2 headline
points) are normal. Per-scenario diffs and category deltas are the reliable signal;
a single-point headline delta is not a verdict. For a stable number: N≥3 repeats,
mean ± variance, exclude transient errors, hold harness version + protocol fixed.

**Server-side reproducibility rule (lab-quant):** serve with
`--max-num-batched-tokens 1024 --max-num-seqs 16` (0rand's proven config). The
gist's 4096-batch variant OOM-kills the head worker on batched forwards on GB10 —
see §8 "Config fix".

---

## 8. Lab-quant swap (2026-08-30) — `local-inference-lab` mixed-precision, the standing config

Replaced the LibertAIDAI uniform-NVFP4 quant with `local-inference-lab/GLM-5.3-Flash-NVFP4`
@ rev `378ca545…` (MIXED_PRECISION: NVFP4 experts layers 3–44 g16 + **MXFP8 MTP drafter**
layer 45 g32, ~186 GB, KLD ~0.04). Recipe: kilork gist `a887667f` (92/100 hardmode claim)
via the `0rand/glm-5.3-flash-nvfp4-2x-dgx-sparks` packaging. New image `glm53:lab` = day-0
base (digest-pinned `905c0293…`) + the 5 lab patches (modelopt MTP-namespace fix + quantprobe,
naming shim, CC-12.x indexer guards, NoPE-MLA rope-pad). Served on :8000 under the same
`glm-5.3-flash` name; single-stack policy (the old `vllm_glm53_head/worker` was stopped first).

### Boot markers (all matched the gist)
- Weight load: `Model loading took 92.19 GiB and 803 s`
- quantprobe: `quantization=modelopt_mixed`, layers 3–44 `algo=NVFP4` (MTP namespace fix live —
  without modelopt.patch this dies with `KeyError model.layers.45.mtp_block…w2_weight_scale`)
- `GPU KV cache size: 1,164,369 tokens` = 2.22× @512K — **exact match** to the gist's boot marker
- `Application startup complete` (~17 min cold)

### Result (seed-42 hardmode, 88 scenarios, c1, ~39 min)
| profile | image | quant | quality | pool |
|---|---|---|---|---|
| 3 (k=4, 9GiB pin) — previous standing | glm53:v9 | LibertAIDAI NVFP4 | 89/100 (157/176) | 1.26M |
| **8 (lab-quant, MTP-3, 10GiB pin) — ACTIVE** | glm53:lab | **lab MIXED** | **90/100 (156/174)** | **1,164,369 (2.22×)** |

Per-scenario diff vs the k4 baseline (same seed 42):
- **Recovered:** TC-51 fail→partial, TC-53 partial→pass, TC-61 fail→partial, TC-67 partial→pass,
  **TC-81 tool-output-injection fail→pass** (the one persistent 512K-profile miss is gone).
- Regressed: TC-50 pass→**fail (all connection attempts failed — transient harness/client drop,
  not model; endpoint healthy pre+post)**, TC-68 pass→fail, TC-75 pass→partial, TC-80 pass→fail.
- Net: +1 on the headline score; category deltas favor **Autonomous Planning 50→83** and
  **Multi-Step Chains 75→88**, offset by Structured Output 92→83 / Hard Mode 92→89.
- Verdict: **lab-quant is the new standing config** — real gain (injection robustness +
  planning/chains), within ±2 of the k4 baseline overall. MTP-3 is the gist's proven depth for
  THIS quant (accept ~2.8–3.0, 61–67% draft acceptance @ c1). Speed in the 24–30 tok/s band.

### Rollback (lab → old LibertAIDAI k4 profile)
```
lab-launch.sh down                                   # stop the lab stack on :8000
# then restore the old profile: on BOTH nodes
cp exec-vllm-512k-k4.sh exec-vllm.sh
cluster.sh up
```
- Old weights still on both nodes: `/var/tmp/glm-5.3-flash-nvfp4` (LibertAIDAI, 182 GB).
- Old image `glm53:v9` still on both nodes. `cluster.sh` / `exec-vllm-*.sh` unchanged.
- Lab weights kept at `/var/tmp/glm-5.3-flash-lab-nvfp4` on both nodes (re-boot in minutes via
  `lab-launch.sh up`); lab image `glm53:lab` retained on both nodes.
- Lab launch files live in `~/glm53-lab/` on both nodes (`docker-compose-lab.yaml`,
  `lab-launch.sh`); local source of truth: `glm53-flash-cluster/lab/`.
- **Gotcha hit this round:** the lab compose needed `cap_add: [IPC_LOCK]` + `ulimits.memlock=-1`
  (matching the old docker-compose.yml) or NCCL `ibv_reg_mr_iova2` → `Cannot allocate memory`
  on world-init. Also: `lab-launch.sh` is a Mac-orchestrator tool (two-hop ssh); run it from the
  Mac, not the head.

### 0rand exact-parameter replica (2026-08-31) — matches his post #124 within 1 pt
Re-ran with 0rand's *actual* sampling params (his forum 381350 #124 command):
`--seed 42 --hardmode --parallel 4 --trials 2 --timeout 360 --max-turns 32`,
`--backend-kwargs '{"chat_template_kwargs":{"thinking":true,"reasoning_effort":"max"},
"temperature":0.1,"top_p":1}'` (top_k left at default/unrestricted on both sides).
Our rig: **90/100 (158/176), 74 pass / 10 partial / 4 fail** vs his forum **91/100
(161/176), 74/13/1**. Trial stats in our run: mean 89.5, 95% CI [89.0, 90.0],
Pass@2 86.4%, Pass^2 79.5% — i.e. the two trials scored 89 and 90.

This is the live demonstration of "same benchmark, different scores":
| run | method | score | pass/partial/fail |
|---|---|---|---|
| A | old quant k4, c1, dev24, t0 | 89 | (157/176) |
| B | lab quant, c1, dev24, t0, trials1 (standing) | 90 | 73/10/5 |
| C | lab quant, **0rand params p4, dev32, t0.1, trials2** | 90 | 74/10/4 |
| D | 0rand forum #124 (his rig, p4, dev25, t0.1, trials2) | 91 | 74/13/1 |

All three "90/90/91" runs are **not** the same scenario outcomes: B→C flips 6 scenarios
(TC-33 fail→pass, TC-50 fail→pass, TC-51 partial→pass, TC-61 partial→fail, TC-67 pass→partial,
TC-82 pass→partial) purely from temp 0.1 + parallel-4 batching + 2-trial averaging — same quant,
same rig. And C's two trials themselves scored 89 vs 90. So the "true" quality is ~89–90 ± a
scenarios' worth of flip, not a fixed integer.

### Config fix that came out of matching 0rand (RESOLVED)
0rand's **actual** running config differs from the kilork gist on two knobs:
`MAX_NUM_BATCHED_TOKENS=*** "1024 = quality-first") and `MAX_NUM_SEQS=16` (gist said 4096/8).
The gist's 4096 **OOM-killed our head worker on the first parallel-4 forward** under
MTP + enforce-eager + fp8_ds_mla on GB10 unified memory (repro: worker signal-killed mid
scheduler step, `total_num_scheduled_tokens=2747`, right after TileLang JIT-compiled
`mhc_pre_big_fuse_with_norm_tilelang`; also an idle death ~2 h in). Switched the lab compose to
exactly 0rand's 1024/16 → clean boot (KV pool back to 1,164,369) + a 4-concurrent probe survived
the workload that killed 4096. **Standing rule: never serve the lab quant with
batched-tokens >1024 on this stack.** Watchdog: `lab-watchdog.sh` (Mac cron every 15 min,
job `glm-lab-watchdog`) probes :8000 and auto-restarts the stack if down.

Remaining minor deltas vs 0rand's rig (not sampling): we're on harness dev32 (his dev25), his
`MAX_MODEL_LEN=500000` + multimodal ON + gm 0.89 (KV pool 724,358) vs our 512K +
language-model-only + gm 0.90 (KV pool 1,164,369). KV pool size doesn't affect c1 quality, only
concurrency headroom.

## Rollback

**Current ACTIVE stack** = lab-quant (`glm53:lab`, `lab/docker-compose-lab.yaml`,
`lab-launch.sh`), serving :8000 under the name `glm-5.3-flash`. Rollback in
either direction is a few minutes:

- **lab → old LibertAIDAI k4 profile** (profiles 0–3):
  ```
  cd glm53-flash-cluster/lab && bash lab-launch.sh down      # stop the lab stack on :8000
  # on BOTH nodes:
  cp exec-vllm-512k-k4.sh exec-vllm.sh                       # the 89/100 standing LibertAIDAI profile
  cd glm53-flash-cluster && bash cluster.sh up               # worker first, then head
  ```
  Old LibertAIDAI weights (182 GB) + `glm53:v9` image are still on both nodes.
  `cluster.sh` / `exec-vllm-*.sh` are unchanged by the lab work.
- **old LibertAIDAI → lab** (re-promote):
  `cluster.sh down` then `cd lab && bash lab-launch.sh up`.
  Lab weights (`/var/tmp/glm-5.3-flash-lab-nvfp4`) + `glm53:lab` image retained on
  both nodes, so this re-boots in ~17 min.
- **Other LibertAIDAI profiles** (if the k4 line is ever reverted): on both nodes,
  `cp exec-vllm-<profile>.sh exec-vllm.sh`, then `cluster.sh down && cluster.sh up`.
  - `exec-vllm-v8-262k.sh` → day-0 bf16 262K, 89/100 (311K pool).
  - `exec-vllm-262k-fp8.sh` → 262K + FP8, 90/100 (610K pool, multimodal ON).
  - `exec-vllm-512k.sh` → 512K + 10GiB pin + MTP-3, 87/100 (1.44M pool).
- **Image fallback:** `glm53:v8` (day-0) still on both nodes; `glm53:v9` is the
  LibertAIDAI 512K image; `glm53:lab` is the current lab image.
- Boot ~14–17 min cold (weights ~13 min + warmup); worker must start before head
  to complete world init.

## Ops notes

- Orchestrator (Mac) → head: `ssh aitopatom-6253.local`; worker is two-hop:
  `ssh 6253 "ssh 10.100.90.4 ..."` (direct ssh to .90.4 times out).
- `cluster.sh <preflight|up|down|status|logs>` (LibertAIDAI stack); `lab/lab-launch.sh
  <takeover|up|down|status|logs>` (lab stack). `up` drops page caches on both nodes
  first (GB10 unified memory), launches worker rank 1 then head rank 0. **Single-stack
  policy:** only one GLM stack serves :8000 at a time; `lab-launch.sh takeover` stops
  the other before bringing the lab up.
- `lab/lab-watchdog.sh` (Mac cron `glm-lab-watchdog`, every 15 min): probes :8000,
  auto-restarts the lab stack if down (skips if the container is <35 min old, i.e.
  mid-boot). Log: `/tmp/glm53-lab-watchdog.log`.
- `.env` lives in 3 places (orchestrator + head + worker REMOTE_DIR); keep in
  sync. The lab compose files (`lab/docker-compose-lab*.yaml`) read it via
  `--env-file .env` from `~/glm53-lab` on each node; `lab-launch.sh` honors
  `LAB_STACK` (lab | lab-vision), and explicit env vars win over `.env`.
- `data/` and `runs/` are gitignored (local bench artifacts).

## 9. lab-vision profile (2026-08-31) — vision ON, 0rand #130 shape

**Goal:** match 0rand's "vision enabled" setup (forum #381350 #123/#127/#130)
on our lab stack, on :8000 as the serving endpoint, without breaking the
standing LMO profile.

**Recipe (0rand's #127/#130 delta, applied on top of the lab profile):**

```
--language-model-only REMOVED          # mm processor + vision tower loaded
--skip-mm-profiling                    # vision tower ~60K KV tokens instead
                                       # of ~300K max-size MM profile (sergio_l #127)
--limit-mm-per-prompt '{"image": 4, "video": {"count": 1, "num_frames": 32,
                                 "width": 512, "height": 512}}'
                                       # MODERN dict format; legacy {"video": 1}
                                       # is rejected by this vLLM build (0rand #130)
```

Same image/weights/standing flags otherwise: 512K, block 256, fp8_ds_mla,
1024/16, MTP-3, gm 0.90, enforce-eager. Containers `glm53-vision-head/worker`,
selected via `LAB_STACK=lab-vision lab/lab-launch.sh up`.

**Boot results (this stack, GB10 121.69 GiB UMA):**

| pin | warmup forward | result |
|---|---|---|
| 10 GiB (`LAB_KV_CACHE_BYTES`) | OOM | `Worker proc VllmWorker-0 died unexpectedly (exit code: None)` ~90 s after "GPU KV cache size: 1,164,369 tokens" — the NOTES-512k rounds 1–2 signature (mm front-end ~15.7 GiB + pinned KV + warmup activations > UMA line). Pool *was* 1,164,369 (skip-mm-profiling confirmed — no ~300K mm reservation), so the crash is host-anon + activations, not KV reservation |
| **9 GiB (`LAB_VISION_KV_CACHE_BYTES`, standing default)** | clean | booted ~16 min; **KV pool 1,022,844 tokens (1.95× @512K)**; vision smoke PASS ("blue background with a red square in the center" on a synthetic image, 396 prompt tokens incl. image) |

**Delta vs the LMO standing profile:** pool 1,164,369 (2.22×) → 1,022,844
(1.95×); vision input enabled. The 9 GiB pin is the same activation-headroom
fix that rescued the k4 LibertAIDAI profile (§6) — the mm front-end eats
headroom on this UMA line, and `--skip-mm-profiling` does not change host-anon
usage.

**Quality — two protocols, one critical confound:**

The naive c1-greedy comparison (vision **88/100** vs LMO **90/100**) is
**invalid as a vision-only delta**: the LMO c1 baseline ran on
`tool-eval-bench v2.6.1.dev24`, the vision c1 run on `dev32`. Different harness
= different scoring. Per-scenario deltas on that pair (net −1 pt) are
therefore harness-confounded, not a clean vision signal.

The **clean A/B is the 0rand-#124 protocol** (parallel 4, 2 trials, temp 0.1,
effort max, seed 42) with the **same dev32 harness** on both stacks:

| stack | trials | mean | CI | Pass@2 / Pass^2 |
|---|---|---|---|---|
| LMO lab (dev32) | 89, 90 | **89.5 ± 0.7** | [89.0, 90.0] | 86.4% / 79.5% |
| **lab-vision (dev32)** | 91, 94 | **92.5 ± 2.1** | [91.0, 94.0] | 89.8% / 85.2% |

So on the only same-harness A/B, **vision is *higher* than LMO (+3.0 mean,
+3.4pp Pass@2, +5.7pp Pass^2)** — the opposite of a regression. Per-scenario
deltas on trial 1 (LMO 90 → vision 91): TC-80 fail→pass (+2),
TC-35/61/67/75/82 partial→pass (+1 each), TC-51 pass→fail (−2),
TC-40/69 pass→partial (−1 each). Net +3 — i.e. vision shows **no text-quality
cost**; the trial spread (±2.1 vs LMO's ±0.7) is the flaky-scenario
sensitivity documented in §Methodology, not a vision effect.

**Why the "lower score" impression (88 < 90) — root cause, verified:**
1. **Harness version**, not vision. c1-greedy baseline (90) = dev24; vision
   c1 (88) = dev32. dev24→dev32 changed the 88-scenario scoring; the −1 pt
   is within that harness shift's noise, not attributable to the mm flags.
2. **Protocol choice.** c1-greedy (our standing protocol) and the 0rand
   parallel-4 protocol rank scenarios differently (batch composition +
   temp 0.1 + effort-max all flip individual scenarios). Comparing across
   protocols — as "his 91 vs our 88" would — is not a like-for-like test.
3. **Trial variance.** Even LMO's identical-protocol two trials span
   89–90; vision spans 91–94. A single-trial headline is not a verdict
   (documented rule: N≥3 repeats for a stable number).

**Verdict (revised):** on the clean same-harness 0rand protocol, lab-vision
(92.5) **outperforms** LMO (89.5); on the confounded c1-greedy, it is −1 pt
(88 vs 90, different harness). Either way, **enabling vision + the 9 GiB pin
shows no systematic text-quality regression**, and the "88 < 90" is a
harness/protocol artifact, not a vision penalty. The 9 GiB pin remains
load-bearing (10 GiB OOMs the warmup forward with the mm front-end).

**Stability finding (costs ~1 boot cycle):** the first real forward after a
cold boot can trigger a TileLang/Triton JIT-compile crash of the head TP0
worker (`mhc_pre_big_fuse_with_norm_tilelang` / `_kpool_tail_seed_kernel`),
same MTP+eager+fp8_ds_mla-on-GB10 class as the §8 config fix. **Mitigation that
worked: JIT-prime the stack right after `Application startup complete`** (a
few thinking-ON text turns to ~768 tokens + one image request, c1) before
running the bench; the engine then stayed up for both full protocols.

**Vision smoke (synthetic image, c1, data-URL):** PASS — "blue background with
a red square in the center" (exact match; 396 prompt tokens incl. image embed).
Note: engine warns `video.num_frames override (32) exceeds model's maximum
number of frames (9), will be ignored` — the 32-frame limit is 0rand's string,
but the model caps at 9 frames; harmless for image use.

**Operational wrap-up:** lab-vision is a drop-in for the standing lab profile
(same image/weights/flags, only the mm flags + pin differ) that adds image
input. The 9 GiB pin is load-bearing (10 GiB OOMs the warmup forward with the
mm front-end on the GB10 UMA line).

**Rollback:** `lab/lab-launch.sh down` + `LAB_STACK=lab lab/lab-launch.sh up`
(LMO standing profile, 10 GiB pin).
