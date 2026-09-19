# Release Notes

## lvllm-v2.5.1 (2026-09-19)

**Base:** lvllm-v2.5.0 · **Release Type:** patch (crash fix + diagnostics + scripts)

### Fixes

- **DeepSeek-V4.x mixed-arch TP prefix-reuse crash (Xid31 `FAULT_PDE`)** — under block-outermost
  interleaved KV packing, indexer pages packed with a ≈1 MB `stride0` faulted on Blackwell page
  tables on SM120 ranks once prefix reuse reached the same block depth. The new
  `VLLM_DSV4_UNPACK_INDEXER=1` gate carves indexer layers out of the interleaved pool into a
  dense tail region (natural per-page `stride0`); MLA/compressor groups keep interleaving.
  Zero effect unless the env is set. Validated on the exact crash seed under both guarded
  (blocking) and production (FULL-graph) timings, plus GSM8K 100/100 on SM120 TP2. (PR #111)

### Diagnostics (all default-off)

- `VLLM_SM8X_GUARD` host-side bound assertions on SM8x block/slot tables (downgrades a hard PDE
  fault to a catchable `RuntimeError` with the offending slot), plus `GEOM`/`TOUCH`/`BTW`
  geometry probes, `VLLM_TILELANG_SKIP_ROCM_TARGET_DETECT` (tilelang hip-detector fork-exec
  hang under sanitizers) and `VLLM_DSV41_FORCE_SM8X_FLOOR`.

### Scripts / config

- v4.1 parsers in `config.yaml`, TP4 serve tweaks, and a DeepSeek-V4 SM120 A/B serve script.

### Regression

- `test_contiguous_kv_packing` 26 passed · same-seed soak crash points (r2 ≈22.1k tokens,
  r12 compression point) green · GSM8K first 100, greedy: **100/100**.

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
