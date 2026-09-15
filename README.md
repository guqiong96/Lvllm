# LvLLM

**LvLLM = [vLLM](https://github.com/vllm-project/vllm) + [lk_moe](https://pypi.org/project/lk-moe/)**,
plus **SM80/86/89 adaptation** and **SM120 tuning/fixes** for popular new models
(DeepSeek-V4/V4.1, Qwen3.8, GLM-5.3, …).

- **lk_moe** is the CPU+GPU hybrid, NUMA-aware MoE engine (pip-installable); vLLM provides the GPU
  path. LvLLM is lk_moe's integration into vLLM **plus** the low-arch (SM80/86/89) bring-up and
  SM120 fixes for these models.
- **Fully optional**: with `LVLLM_MOE_NUMA_ENABLED=0` it behaves exactly like stock vLLM.

---

## Model support

| Model | SM80 | SM86 | SM89 | SM90 | SM100 | SM120 | spec-decode |
|-------|------|------|------|------|-------|-------|-------------|
| DeepSeek-V4.1-Flash | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ fixed | ✅ dspark |
| DeepSeek-V4-Flash (0731) | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ native | ✅ dspark |
| Qwen3.8-Flash-Next | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ new | ✅ MTP |
| GLM-5.3-Flash | ready | ready | ready | native | native | ready | — |

`native` = upstream vLLM · `new` / `fixed` = done by this Lvllm release · `ready` = code done,
pending weights to validate. Full per-model hardware tables, benchmarks and CLI: **[`RELEASE_NOTES.md`](./RELEASE_NOTES.md)**.

### Previously verified (lk_moe hybrid) models

Original MOE models verified on vLLM's Qwen3 / GLM / MiniMax lines:

| Family | Models |
|---|---|
| Qwen3 | 3.6 / 3.5-35B-A3B, 3.5-122B-A10B, 3.5-397B-A17B, Coder-Next, Next-80B-A3B-Instruct, Coder-30B-A3B-Instruct, VL-30B-A3B-Instruct |
| GLM | GLM-5.2-NVFP4 [sm120], GLM-4.7(-Flash) / 4.6V |
| MiniMax | M3 / M2.7 / M2.5 / M2.1 |
| Others | gemma-4-26B-A4B-it, NVIDIA-Nemotron-3-Super-120B-A12B-BF16, Ornith-1.0-35B-FP8, Kimi k2.6 / k2.5 |

Unlisted original MOE models from the Qwen3 / GLM / MiniMax lines are theoretically supported,
pending testing.

---

## Benchmarks

Single request, greedy decode (t/s). Each cell = **plain / spec** t/s; spec = **dspark** for
DeepSeek-V4.x, **MTP** for Qwen3.8. `—` = not applicable. Full hardware tables: [`RELEASE_NOTES.md`](./RELEASE_NOTES.md).

| Model | SM120 TP2 (2× 5060 Ti) | SM86 TP2 (2× 3090) | Mixed TP4 (3090×2 + 5060 Ti×2) |
|-------|-------------------------|--------------------|--------------------------------|
| DeepSeek-V4.1-Flash | 25 / 32–37 | 27 / 26–40 | 22 / 24–34 |
| DeepSeek-V4-Flash (0731) | 28 / up to 44.8 | 30.6–31.1 / 36–43 | 30.5–33.6 / 38–43.5 |
| Qwen3.8-Flash-Next | — | 45 / — | 40.3–40.6 / up to 78 |
| GLM-5.3-Flash | ready | ready | ready |

### Reference hardware

All numbers above were measured on one host:

| Item | Spec |
|---|---|
| CPU | 2× AMD EPYC 7642 48-Core (96c/192t total, NPS4 ⇒ 8 NUMA nodes), `LK_THREADS=48` |
| Host RAM | 1 TiB DDR4-3200 (16-channel, 8 per socket) |
| RAM headroom | DeepSeek-V4.x peak ≈ 590 GB resident (≥ 640 GB advised); Qwen3.8 peak ≈ 335 GB |

---

## Why lk_moe

lk_moe spans a MoE model across **VRAM + system memory** and schedules experts across **CPU + GPU**
with NUMA awareness — reaching ~100% VRAM utilization and overlapping GPU prefill with hybrid decode.

| Role (per MoE layer) | Meaning | Env |
|---|---|---|
| **master switch** | `0` = stock vLLM pure-GPU, `1` = hybrid | `LVLLM_MOE_NUMA_ENABLED` |
| GPU-prefill layer | big batches on GPU, small on CPU | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` + `LVLLM_GPU_PREFETCH_WINDOW` |
| GPU-resident layer | weights stay in VRAM | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |

Requires x86 AVX2+ and an NVIDIA GPU (SM75+).

---

## Launch

Ready-made serve scripts (per model × topology) live in **[`commands/`](./commands/)**:

```
commands/
  dsv4_0731_serve_tp2_{3090,5060ti}[_dspark].sh   dsv4_0731_serve_tp4[_dspark].sh   # DeepSeek-V4-Flash 0731
  dsv41_serve_tp2_3090_dspark.sh                                                     # DeepSeek-V4.1-Flash
  qwen38_serve_tp2_{3090,5060ti}[_mtp].sh          qwen38_serve_tp4[_mtp].sh         # Qwen3.8-Flash-Next
```

Each script is a complete, self-contained `vllm serve …` (env + args). Pick by model and GPU
topology, adjust the model path, and run:

```bash
bash commands/dsv4_0731_serve_tp4_dspark.sh
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `LVLLM_MOE_NUMA_ENABLED` | `0` | hybrid on/off (`0` = stock vLLM) |
| `LK_THREADS` | — | threads = physical cores ÷ #GPUs |
| `LK_THREAD_BINDING` | `CPU_CORE` | `CPU_CORE` (best) / `NUMA_NODE` |
| `LVLLM_GPU_PREFETCH_WINDOW` | — | prefetch window, typically `1` |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | — | GPU prefill starts at input ≥ value; `0` = off |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | none | expert layers in VRAM, e.g. `0`, `0-1,9` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | `1` | avoid NUMA node OOM |
| `LK_POWER_SAVING` | `0` | `1` = CPU power saving |

On a **mixed-arch host** add `FLASHINFER_CUDA_ARCH_LIST="<all ranks>"` (e.g. `"8.6 12.0f"`).

### Optimization tips

- Output performance: `--compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY"}'`.
- Enable GPU prefill: `LVLLM_GPU_PREFETCH_WINDOW=1`, `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096`,
  `--max-num-batched-tokens 32000`; disable with `…MIN_BATCH_SIZE=0` + `--max-num-batched-tokens 4096`.
- Heterogeneous TP (>2 PCIe GPUs): add `--disable-custom-all-reduce`.
- BIOS NUMA: AMD EPYC NPS4 / Intel SNC4, node count a multiple of GPU count.

---

## Install

```bash
pip install https://github.com/guqiong96/Lvllm/releases/download/lvllm-v2.5.0/lvllm-2.5.0-cp312-cp312-manylinux_2_34_x86_64.whl
# deps (incl. the bundled FlashInfer wheel with the SM120 sparse-MLA fix) resolve automatically
```

From source (needs GCC ≥ 13 for `<format>`):

```bash
git clone https://github.com/guqiong96/Lvllm.git && cd Lvllm
for d in /opt/rh/gcc-toolset-{14,13}/enable; do [ -f "$d" ] && source "$d" && break; done   # RHEL/Rocky
VLLM_VERSION_OVERRIDE="2.5.0" CMAKE_BUILD_TYPE=Release pip install -e . --no-build-isolation
```

---

## Patches

- **In this repo**: only the lk_moe integration — [`patches/01_lk_moe__71888f507a.patch`](./patches/01_lk_moe__71888f507a.patch),
  a single diff of the current tree over upstream base `71888f507a`.
  ```bash
  git checkout 71888f507a && git apply patches/01_lk_moe__71888f507a.patch
  ```
- The per-model / PR patches are **not** shipped here; they live in `~/Downloads/opencode/patches/`.

---

## Release / history

See **<https://github.com/guqiong96/Lvllm/releases>**.
