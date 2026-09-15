# Release Notes — lvllm-v2.5.0

**Base Version:** vllm commit 71888f507a (`upstream/main`, 2026-09-14) + lk_moe
**Release Type:** Feature release (Lvllm)

## Models and supported architectures

| Model | SM80 | SM86 | SM89 | SM90 | SM100 | SM120 | spec-decode |
|-------|------|------|------|------|-------|-------|-------------|
| DeepSeek-V4.1-Flash | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ fixed here | ✅ dspark |
| DeepSeek-V4-Flash (0731) | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ native | ✅ dspark |
| Qwen3.8-Flash-Next | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ new | ✅ MTP |
| GLM-5.3-Flash | ready | ready | ready | native | native | ready | — |

- **native** — done by upstream vLLM.
- **new** — done by this Lvllm release.
- **fixed here** — done by this Lvllm release.
- **ready** — done by this Lvllm release, pending weights on the reference machine.

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
| Host RAM | machine 1 TiB DDR4-3200 (16-channel, 8 per socket); model requirement TBD (no weights on the reference machine yet) |

## See also

- [Lsglang](https://github.com/guqiong96/Lsglang) — the same lk_moe integration for sglang.
- [README.md](./README.md) — full integration guide, benchmark and configuration reference.
