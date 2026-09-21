# Release Notes

## lvllm-v2.5.2 (2026-09-21)

**Base:** lvllm-v2.5.1 · **Release Type:** patch (Ada support + FP8_PB_WO weights)

### Fixes

- **DeepSeek-V4.x failed engine startup on SM89 (Ada)** — the CuTeDSL
  dequant-gather / indexer-q kernel picks keyed on "cutlass installed + native
  fp8" (SM89+), but those kernels emit inline PTX (`cvt.rn.bf16.f16`,
  `mul.bf16x2`) with an sm_90 ISA floor, so ptxas aborted every Ada host with
  "requires .target sm_90 or higher". Adds `cutedsl_kernels_supported()`
  (fp8 gate AND capability major ≥ 9, per device, fail closed) and applies it
  at all six pick sites; SM89 now falls back to the Triton path already
  validated on SM80/86. Kernel selection is unchanged on every previously
  validated config (SM86, SM90+, mixed-arch TP4); verified by gate-value tests
  on SM86/SM120 (no SM89 hardware on the build host, fallback path static-
  verified only).
- **Qwen3.8-Flash-Next FP8 builds** — `FP8_PB_WO` layers now serve with the
  2-D `weight_scale_inv` convention (upstream #54126 contract), making the
  W4A16-4o6-FP8 checkpoint work end to end (SM86 TP2 MTP decode 61.8–65.9
  t/s, accept 3.2–3.5, HTML smoke green).

### Scripts / config

- qwen38 serve scripts point at the W4A16-4o6-FP8 checkpoint.

### Regression

- CuTeDSL gate values re-checked on this tree's 2×3090+2×5060 Ti hosts:
  per-device selection identical to lvllm-v2.5.1 (3090→Triton, 5060 Ti→CuTeDSL,
  simulated SM89→Triton). FlashInfer asset unchanged from lvllm-v2.5.1.

---

## lvllm-v2.5.1 (2026-09-20)

**Base:** lvllm-v2.5.0 · **Release Type:** patch (crash fixes + prefill speed + scheduling)

### Behavior changes (read before upgrading)

- **`VLLM_DSV4_UNPACK_INDEXER` is now ON by default.** The indexer-pool carve-out below is
  active unless you set `VLLM_DSV4_UNPACK_INDEXER=0` (escape hatch keeps the old packed
  layout). Cost: ~5 % more bytes per pool block (≈ −11 % block count); this is the fix for
  the mixed-arch Xid31 crash, so opting out is only meaningful on single-architecture hosts.
- **YAML config `key: false` now actually takes effect.** `--config` files expanded booleans
  to `--no-*` flags only when the flag was visible on the top-level parser; serve options live
  on the `serve` subparser, so every `*: false` line (e.g. `async_scheduling: false`) was
  silently dropped until now. Anything that relied on a dropped `false` line keeps its
  default only by accident — check your YAML.

### Fixes

- **DeepSeek-V4.x mixed-arch TP prefix-reuse crash (Xid31 `FAULT_PDE`)** — under block-outermost
  interleaved KV packing, indexer pages packed with a ≈1 MB `stride0` faulted on Blackwell page
  tables on SM120 ranks once prefix reuse reached the same block depth. The unpacked indexer
  region (now default; `VLLM_DSV4_UNPACK_INDEXER=0` opts out) carves indexer layers out of the
  interleaved pool into a dense tail region (natural per-page `stride0`); MLA/compressor groups
  keep interleaving. Validated on the exact crash seed under both guarded (blocking) and
  production (FULL-graph) timings, plus GSM8K 100/100 on SM120 TP2. (PR #111)

- **Hybrid-model scheduling livelocks on tight KV pools** — three scheduler-side fixes:
  the full-sequence admission gate now splits recycling-aware groups from full-attention
  groups (`skip_capped_groups`), so a fine-block group whose window+in-flight peak exceeds
  the pool falls through to the chunked path instead of retrying forever (32 k/320 k prompts
  stuck at `wait=1`); a running prefill chunk whose only preempt victim is itself defers one
  step (waits for its own in-flight step to settle) instead of self-preempting into a
  recompute loop (16 k long-prompt prefill ran ≈3× every chunk under async scheduling);
  and waiting-request shrink retries are bounded. Startup in-flight accounting now counts
  one scheduler run (`×1 mnb`) instead of the batch-deque product.

- **Hybrid KV capacity collapse from pattern-split MLA buckets** — MLA buckets are no
  longer split into multiple groups (each split replicated its own block table over the
  same token columns, so every column paid one block id per split); one group per MLA
  bucket restores the vllm-ds4 single-group layout and shrinks per-request reservations.

- **Startup KV readout** — the GPU-KV-cache line now reports pool bytes, block count and
  the longest single request that actually fits (`max servable length`), instead of the
  misleading `concurrency × max_model_len` aggregate; `VLLM_KV_SIZE_DEBUG=1` additionally
  dumps per-spec/per-group geometry and step-level scheduler traces for chunk diagnosis.

- **Config files: `--no-*` resolution fixed** (see behavior changes above); a warning is
  logged when no negated flag exists for a `false` key instead of dropping it silently.

### Performance

- **SM8x sparse-MLA prefill attention is tiled** (DeepSeek-V4-0731 and V4.1) — the
  per-(query, head) kernel re-read every gathered KV row once per query and head, which on
  Ampere capped prefill at roughly half of the achievable bandwidth. The port tiles the work
  (256-query × 512-topk chunks, 8 heads share one KV-row load, fp32 online-softmax state
  chunked to stay in L2); contract matches the old kernel bit-for-bit per test
  (flat row ids, valid-first/`-1` padding, per-query lengths, sink merged with zero value).
  Measured long-chunk prefill: **967–986 tok/s** on V4-0731 mixed TP4 (was ~440), **710–750**
  on V4.1. Escape hatch `VLLM_DSV4_SM8X_TILED_PREFILL=0`; SM120 path untouched.

- **DeepSeek-V4 prefill indexer slow torch path on SM80/86** — the non-paged MQA-logits entry
  point always ran the chunked torch reference (`.float()` dequant of the whole K buffer,
  fp32 matmul chunks), capping prefill and adding multi-GB transient allocations. It now
  dispatches to a tiled Triton kernel (uint8 e4m3 decode once per K tile, exact bf16 MMA,
  `-inf`-outside-window contract preserved; BLOCK_N auto-drops 256→128 on small-smem devices).
  Reference parity ≤6e-7 relative in-window.

### Model checkpoints

- **Qwen3.8-Flash-Next (NVFP4)**: the validated checkpoint is
  [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://www.modelscope.cn/models/RadixArk/Qwen3.8-Flash-Next-NVFP4)
  (also on Hugging Face; modelopt 0.46.0 release build — MoE/attention NVFP4, MTP draft
  left BF16, FP8 n-gram PLE table handled by the byte-gather path). Same-named third-party
  *mixed* dev builds that quantize the MTP draft to FP8 are not yet supported on SM80/86
  (no FP8 MoE backend for the draft there; scheduled for 2.5.2 — until then disable MTP or
  use the RadixArk checkpoint).

### Diagnostics (all default-off)

- `VLLM_SM8X_GUARD` host-side bound assertions on SM8x block/slot tables (downgrades a hard PDE
  fault to a catchable `RuntimeError` with the offending slot), plus `GEOM`/`TOUCH`/`BTW`
  geometry probes, `VLLM_TILELANG_SKIP_ROCM_TARGET_DETECT` (tilelang hip-detector fork-exec
  hang under sanitizers) and `VLLM_DSV41_FORCE_SM8X_FLOOR`.

### Scripts / config

- v4.1 parsers in `config.yaml`, TP4 serve tweaks, and a DeepSeek-V4 SM120 A/B serve script.

### Regression (release matrix, mixed-arch TP4 unless noted)

- `tests/v1/core` 715 passed (the 4 non-passing = 3 GPU-visibility/env classes verified by
  re-run + 1 HF-download e2e class, all unrelated to these changes) ·
  `test_contiguous_kv_packing` 26 · `test_cutedsl_arch_pin`/config 34 · deferred-shrink 3 ·
  tiled-prefill kernel tests 6 (bit-for-bit vs old kernel) · same-seed soak crash points
  (r2 ≈22.1k tokens, r12 compression point) green · GSM8K first 100, greedy: **100/100**
  (SM120 TP2).
- Decode on this tree: **V4.1** plain 25.2–25.7 t/s / dspark 23–31 (accept 3.5–4.6,
  net gain small — dspark prefill tax +8.7 s/chunk logged, missing sglang
  `speculative-attention-mode decode` equivalent logged as future work) ·
  **V4-0731** dspark 38–46 (past record) · **Qwen3.8-Flash-Next + MTP** 41–62 t/s
  (accept 2.5–3.6) · **GLM-5.3-Flash** ~22 t/s · HTML smoke green on all.

---

# lvllm-v2.5.0

**Base Version:** vllm commit 71888f507a (`upstream/main`, 2026-09-14) + lk_moe
**Release Type:** Feature release (Lvllm)

## Models and supported architectures

| Model | SM80 | SM86 | SM89 | SM90 | SM100 | SM120 | spec-decode |
|-------|------|------|------|------|-------|-------|-------------|
| DeepSeek-V4.1-Flash | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ fixed here | ✅ dspark |
| DeepSeek-V4-Flash (0731) | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ native | ✅ dspark |
| Qwen3.8-Flash-Next | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ new | ✅ MTP |
| GLM-5.3-Flash | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ fixed here | — |

- **native** — done by upstream vLLM.
- **new** — support added by this Lvllm release.
- **fixed here** — upstream path corrected by this Lvllm release.

## Installation

```bash
pip install lvllm-2.5.0-*.whl
```

All dependencies resolve automatically, including the FlashInfer wheel
attached to this release (`flashinfer_python-0.7.0-py3-none-any.whl`,
contains the SM120 sparse-MLA fix #5075; not on PyPI).
First serve JIT-compiles the sparse-MLA kernels (nvcc required, a few
minutes), then results are cached under `~/.cache/flashinfer/`.
On machines with mixed architectures add `FLASHINFER_CUDA_ARCH_LIST="12.0f"`.

## Benchmark reference (2× RTX 3090, TP2)

### DeepSeek-V4.1-Flash

| Item | Reference |
|------|-----------|
| GPUs | 2× RTX 3090 (SM86) |
| CPU | 2× AMD EPYC 7642 48-Core (96c/192t total, NPS4 ⇒ 8 NUMA nodes), `LK_THREADS=48` |
| Host RAM | machine 1 TiB DDR4-3200 (16-channel, 8 per socket); requirement ≥ 640 GB (peak observed ≈ 590 GB resident, MoE/engram host-resident) |

Launch: [`commands/dsv41_serve_tp2_3090_dspark.sh`](./commands/dsv41_serve_tp2_3090_dspark.sh)

Measured (single request, greedy):

| Setup | Plain decode | dspark |
|-------|--------------|--------|
| 2× RTX 5060 Ti (SM120, TP2) | 25 t/s | 32–37 t/s |
| 2× RTX 3090 (SM86, TP2) | 27 t/s | 26–40 t/s |
| 2× RTX 3090 + 2× RTX 5060 Ti (mixed, TP4) | 22 t/s | 24–34 t/s |

### DeepSeek-V4-Flash (0731)

Same hardware as V4.1. Launch: [`commands/dsv4_0731_serve_tp2_5060ti.sh`](./commands/dsv4_0731_serve_tp2_5060ti.sh)
(dspark variant `…_dspark.sh`; SM86/TP4 variants also under `commands/`).

Measured (single request, greedy):

| Setup | Plain decode | dspark |
|-------|--------------|--------|
| 2× RTX 5060 Ti (SM120, TP2) | 28 t/s | up to 44.8 t/s |
| 2× RTX 3090 (SM86, TP2) | 30.6–31.1 t/s | 36–43 t/s |
| 2× RTX 3090 + 2× RTX 5060 Ti (mixed, TP4) | 30.5–33.6 t/s | 38–43.5 t/s |

### Qwen3.8-Flash-Next

| Item | Reference |
|------|-----------|
| GPUs | 2× RTX 3090 + 2× RTX 5060 Ti (mixed SM86/SM120), or 2× same-arch |
| CPU | 2× AMD EPYC 7642 48-Core (96c/192t total, NPS4 ⇒ 8 NUMA nodes), `LK_THREADS=48` |
| Host RAM | machine 1 TiB DDR4-3200 (16-channel, 8 per socket); PLE/FP8 table host-resident (peak observed ≈ 335 GB resident) |

Launch: [`commands/qwen38_serve_tp4.sh`](./commands/qwen38_serve_tp4.sh) (plain) /
[`commands/qwen38_serve_tp4_mtp.sh`](./commands/qwen38_serve_tp4_mtp.sh) (MTP).

Measured (single request, greedy):

| Setup | Plain decode | MTP |
|-------|--------------|-----|
| 2× RTX 3090 (SM86, TP2) | 45 t/s | — |
| 2× RTX 3090 + 2× RTX 5060 Ti (mixed, TP4) | 40.3–40.6 t/s | up to 78 t/s (accept ≈ 3.3) |

### GLM-5.3-Flash

| Item | Reference |
|------|-----------|
| GPUs | 2× RTX 3090 + 2× RTX 5060 Ti (mixed SM86/SM120), or 2× same-arch |
| CPU | 2× AMD EPYC 7642 48-Core (96c/192t total, NPS4 ⇒ 8 NUMA nodes), `LK_THREADS=48` |
| Host RAM | machine 1 TiB DDR4-3200 (16-channel, 8 per socket); model ~190 GiB / 33 shards (nv-community NVFP4), MoE host-resident |

Launch: [`commands/glm53_serve_tp2_5060ti_plain.sh`](./commands/glm53_serve_tp2_5060ti_plain.sh) (SM120 plain) /
[`commands/glm53_serve_tp4.sh`](./commands/glm53_serve_tp4.sh) (mixed TP4; `--kv-cache-dtype
bfloat16` selects the SM8x Triton sparse-MLA path). SM86/SM89 reach a bf16 KV cache via the
new SM8x sparse-MLA backend. MTP draft is present but shows no net gain over plain, so the
default scripts stay plain.

Measured (single request, greedy):

| Setup | Plain decode | MTP |
|-------|--------------|-----|
| 2× RTX 5060 Ti (SM120, TP2) | 20.8 t/s | — (no net gain) |
| 2× RTX 3090 (SM86, TP2) | 22–23 t/s | — |
| 2× RTX 3090 + 2× RTX 5060 Ti (mixed, TP4) | 22.4 t/s | — |

## See also

- [Lsglang](https://github.com/guqiong96/Lsglang) — the same lk_moe integration for sglang.
- [README.md](./README.md) — full integration guide, benchmark and configuration reference.
