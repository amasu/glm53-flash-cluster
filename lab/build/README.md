# GLM-5.3-Flash NVFP4 (lab quant) — 92/100 tool-eval on 2× NVIDIA DGX Spark

Reproducible build + launch for **`local-inference-lab/GLM-5.3-Flash-NVFP4`**
(MIXED_PRECISION: NVFP4 experts + MXFP8 MTP drafter) on a DGX Spark pair
(GB10 / sm_121), vLLM day-0 image + 5 patches, TP=2 over RoCE, 512K ctx,
MTP k=3.

Measured (`tool-eval-bench run --seed 42 --hardmode --perf --concurrency 1,2`):

| arm | quality | decode tg t/s c1 / c2 |
|---|---|---|
| **this config (eager)** | **92/100** | 28.4 / 40.5 |
| same with CUDA graphs | 89/100 | 28.4 / 43.4 |

**`--enforce-eager` is load-bearing**: MTP + CUDA graphs silently costs ~3
quality points on the day-0 dev build (acceptance metrics stay healthy — it
corrupts sampled output, not the draft).

## Files

| file | what |
|---|---|
| `Dockerfile` | build: day-0 base + the 5 patches (grep/py_compile self-verify) |
| `patch_mla.py`, `*.patch` | the 5 patches (attribution in `UPSTREAM.md`) |
| `run-rank0.sh` | docker run for rank 0 (+ rank-1 deltas in comments) |
| `recipe.toml` | equivalent [sm121-launcher](https://git.kilork.org/kilork/sm121-launcher) recipe |

## Pinned coordinates (the tested, reproducible set)

| thing | mutable ref | pinned to |
|---|---|---|
| base image | `vllm/vllm-openai:glm53-flash-arm64-cu130` | `@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce` (Dockerfile ARG default) |
| model | `local-inference-lab/GLM-5.3-Flash-NVFP4` @ main | revision `378ca54585c46542bad1f3cb3ed0d73ae51cdb62` (`--revision`, download + serve it) |

Tags and `main` branches move (the lab repo was updated the day of this
write-up); digests/revisions don't. Re-derive after upstream updates:
`docker images --digests` post-pull, and the `sha` field of
`https://huggingface.co/api/models/<repo>`.

## Build

```bash
docker build -t glm53-flash-lab:local .   # no --build-arg: the Dockerfile ARG default is already digest-pinned
# (override ONLY with tag@digest, never a bare tag)
```

~30 s (patch layer only). All patches apply with zero fuzz against that base.

## Run

`huggingface-cli download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 378ca54585c46542bad1f3cb3ed0d73ae51cdb62` on **both** nodes (~186 GB; NAS-shared cache is fine — weights then load in ~700 s), edit
`MASTER_ADDR` + fabric env in `run-rank0.sh`, run on both nodes.

## Boot markers (check in order — each has caught a real failure mode)

1. Weight load: `Model loading took ~92.7 GiB and ~700 s`
2. `[quantprobe] ... prefix=model.layers.45.mlp.experts algo=MXFP8`
   — if `algo=None`, the modelopt MTP fix is not live; the serve will die.
3. `GPU KV cache size: ~1,164,369 tokens` (= 2.22× concurrency at 524288)
4. `Application startup complete` (~17 min cold)

## Why each non-obvious knob exists (each proven by a boot that died without it)

- `--enforce-eager` — quality bug above.
- `--kernel-config` autotune/warmup **off** — FlashInfer autotune scratch at
  512K shapes OOMs the GB10 driver (`NVRM NV_ERR_NO_MEMORY
  _memdescAllocInternal`). The non-autotuned sparse-MLA fallback is fine.
- `--kv-cache-memory-bytes 10737418240` — deterministic 10 GiB KV pool;
  GMU-sized KV at high ctx interacts badly with unified memory
  (vLLM #48140 reads `MemFree`).
- `--max-num-batched-tokens 4096` — scan/graph scratch scales with the
  prefill chunk; 8192 OOMs at 512K ctx.
- `sparse_attn_indexer*` patches — CC-12.0 guards: indexer topk fits 48 SMs
  at ≤262K ctx but requests 62 CTAs at 512K and the fallback needs 128 KB
  smem (GB10 has 99 KB) → hard abort.
- `patch_mla.py` + `VLLM_MLA_NOPE_PAD_ROPE=1` — NoPE-MLA rope-pad to DeepSeek
  512+64 `fp8_ds_mla` layout + sm120 topk width fix.
- `modelopt.patch` — MTP quantization namespace fix (else:
  `KeyError model.layers.45.mtp_block...w2_weight_scale` at load).
- `model.patch` — checkpoint naming shim (attn_hc submodules, forget_gate,
  fused conv1d).

Not needed from the upstream FujitsuPolycom stack: flashkda prefill +
mm-renderer patches (base/fabric specific), bundled NCCL 2.30.7 (direct-cable
ring), instanttensor loader (optional ~700 s → ~30 s weight-load speed-up).

Upstream credits: [FujitsuPolycom/glm53-flash-tp2-spark](https://github.com/FujitsuPolycom/glm53-flash-tp2-spark)
(Apache-2.0), [kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark),
and @0rand for the eager-mode finding.
