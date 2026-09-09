# LvLLM — GPU + NUMA Dual Parallel [[中文]](./README_cn.md)

LvLLM is a special extension of [vLLM](https://github.com/vllm-project/vllm) that fully utilizes CPU
and GPU computing resources, featuring an efficient **GPU parallel + NUMA parallel** architecture
for MOE model hybrid inference.

The actual hybrid inference engine is **[lk_moe](https://pypi.org/project/lk-moe/)**, vllm only
provides the "GPU path", lk_moe provides the "hybrid path". LvLLM is the concrete integration case
of lk_moe into vllm.

> **Release policy:** LvLLM version updates are released **in sync with vllm releases** — on top of
> a fresh vllm tag we keep the code "as-is + lk_moe". We do **not** pile on extra features; unless a
> necessary bug-fix patch is required, the diff against upstream stays minimal (just the lk_moe layer).

---

## 一、Why lk_moe?

lk_moe lets the MOE model footprint span **VRAM + system memory**, and schedules expert
computation across **CPU + GPU** with NUMA awareness:

- **Memory + VRAM load balancing**: total footprint = VRAM + memory, so a model can be
  "1+1=2" and reach 100% VRAM utilization.
- **CPU-GPU hybrid decode / prefill + GPU prefill**: three computing modes, with GPU prefill
  running in parallel with hybrid decoding for near-100% GPU utilization.
- **NUMA thread optimization**: cross-node communication as low as 3%, L3 cache hit rate over 50%.

| Hybrid modes | Env control |
|---|---|
| **master switch** — `0` = stock vllm pure-GPU inference (all modes below off), `1` = enable hybrid | `LVLLM_MOE_NUMA_ENABLED` |
| CPU prefill / GPU prefill | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` + `LVLLM_GPU_PREFETCH_WINDOW` |
| GPU prefill & decode | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |

Note 1: x86 CPUs with AVX2+ instruction sets and Nvidia GPUs with sm75+ architectures.

---

## 二、How to integrate lk_moe

lk_moe is a pip-installable package (`pip install lk_moe`). It exposes a small set of C++ kernel
classes (`MOE_WNA16`, `MOE_FP8`, `MOE_MXFP4`, `LKEmbedding`, ...) driven by a `MOEConfigV2` config.
The engine handles expert weight placement (VRAM / pinned NUMA host memory), NUMA-aware scheduling,
and quantized kernel execution internally.

The integration work in vllm is therefore **only about routing each MOE layer to lk_moe**
(which layers stay on GPU, which go hybrid, which quant kernel to use) and **keeping the feature
optional** so the branch stays 100% compatible with stock behavior when disabled.

### Core integration principle

> **Every MOE layer can be one of three roles.** The role is decided by a few env vars, and the
> rest of the engine is unchanged.

| Role | Meaning | Decision |
|---|---|---|
| GPU-resident layer | all weights in VRAM, original GPU path | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |
| CPU layer (hybrid) | MoE weights in memory, attn in VRAM; GPU computes attn + CPU computes MoE | default when enabled |
| GPU-prefill layer | large batches on GPU, small batches on CPU | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` |

### Minimal integration checklist

1. **Add the dependency** — `lk_moe` in `requirements`.
2. **Add a feature gate** — `is_lk_moe_feature_enabled()` (reads `LVLLM_MOE_NUMA_ENABLED`) so all
   hybrid behavior is off by default and the branch behaves exactly like stock vllm.
3. **Wire the MOE layer** — in the fused-MoE layer, resolve each layer's role, build a
   `lk_moe.MOEConfigV2`, instantiate the quant-appropriate `MOE_*` class, and call it in `forward`.
4. **Register per-quantization kernels** — each quant method exposes its own LK MoE kernel class.
5. **Handle weight loading / placement** — keep CPU-resident weights off the GPU device.
6. **(Optional) extras** — CPU-resident embedding (`LKEmbedding`) and NUMA thread binding.

### Case study — LvLLM (vllm) file-by-file

The whole lk_moe integration is captured as a single portable patch at
[`patches/01_lk_moe__v0.29.0.patch`](./patches/01_lk_moe__v0.29.0.patch) — the full diff between
upstream `v0.29.0` and the merge commit `"Merge v0.29.0 into lk_moe branch"`. Apply it to
a clean `v0.29.0` checkout with `git apply patches/01_lk_moe__v0.29.0.patch`.

| File | What it does |
|---|---|
| `vllm/envs.py` | the feature-gate helpers: `is_lk_moe_feature_enabled`, `is_lk_moe_cpu_layer`, `is_lk_moe_gpu_resident_layer`, `is_lk_moe_gpu_prefill_layer`, `get_gpu_prefetch_window`, ... |
| `vllm/model_executor/layers/fused_moe/routed_experts.py` | **the core**: resolve layer role, build `MOEConfigV2`, instantiate `MOE_WNA16` / `MOE_FP8` / `MOE_MXFP4` per quant, and dispatch in `forward` (GPU resident → `quant_method.apply`; hybrid → `_cpu_decode` / `_cpu_prefill` / `_gpu_prefill`) |
| `vllm/model_executor/layers/quantization/{fp8,mxfp4,auto_awq,...}.py` | each quant method registers its LK MoE kernel (e.g. `MOE_FP8`, `MOE_MXFP4`) |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/*` | compressed-tensors W8A8-FP8 / W4A4-NVFP4 / WNA16 MoE each register their LK kernel |
| `vllm/model_executor/model_loader/utils.py` | keep CPU-resident layers / lk-embedding off the GPU device; run `process_weights_after_loading` / `clean_weights_after_loading` for lk_moe layers |
| `vllm/utils/numa_utils.py` | when `LVLLM_ENABLE_NUMA_INTERLEAVE=1`, launch workers under `numactl --interleave=all` |

The same method is applied to sglang in the [Lsglang](https://github.com/guqiong96/Lsglang)
repository, plus dedicated DeepSeek-V4 branches: [Lvllmds4](https://github.com/guqiong96/Lvllmds4)
(SM120+) and [Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x) (SM80+).

---

## 三、Example — LvLLM (with benchmarks)

### Performance benchmark

Open GPU Prefill, `max_num_batched_tokens=8192` (row 1) / `32768` (row 2):

| Model | Version | CPU | Memory | GPU | Prefill | Decode | Spec. Decoding |
|-------|---------|-----|--------|-----|---------|--------|---------|
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllm-v2.4.0 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | 850 t/s [in 32768] | 28 t/s [in 32768] | 30~46 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllmds4-x-v2.3.9 | EPYC 7642 *2 | 16ch ddr4 3200 | 3090 * 2 | 1060 t/s [in 32768] | 26 t/s [in 32768] | 35~47 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllmds4-v2.3.9 | EPYC 9684x *2 | 24ch ddr5 4800 | pro 6000 * 1 | 3100 t/s [in 131072] | 75 t/s [in 131072] | 100~115 t/s |

### Version history

```bash
2026-09-09: Lvllm-v2.4.0 - synced upstream vllm to v0.29.0, lk_moe integration (README/RELEASE_NOTES/patch)
2026-07-17: lvllm-v2.3.6 - add dtype float16 support for SM75 GPU Prefill
2026-07-08: lvllm-v2.3.2 - add ModelOpt W4A16 NVFP4 quantization types support
2026-07-05: lvllm-v2.3.0 - Optimize GPU prefill speed, CPU AVX512 optimization, removed LVLLM_GPU_RESIDENT_MOE_EXPERTS
2026-06-05: lvllm-v2.2.0 - Upgraded lk_moe module, added support for nvfp4, mxfp4 quantization types, added LVLLM_GPU_RESIDENT_MOE_EXPERTS, removed LVLLM_MOE_USE_WEIGHT, LVLLM_MOE_QUANT_ON_GPU
2026-04-06: lvllm-v2.1.0 - Enhanced power saving effect with LK_POWER_SAVING=1, supports FP8+BF16+AWQ4bit hybrid MOE layer inference
2026-03-22: lvllm-v2.0.0 - FP8 MoE models with INT4 expert quantization support layer-wise loading to reduce peak memory usage, LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1
2026-03-19: lvllm-v1.9.10 - Fixed known issues, added support for new moe model types without gate_proj, e.g., NVIDIA-Nemotron-3-Super-120B-A12B-BF16
2026-03-11: lvllm-v1.9.2 - FP8, AWQ4bit models no longer occupy additional memory when GPU Prefill is enabled, FP8 models removed TO_DTYPE runtime type conversion, KEEP does not support GPU Prefill for now
2026-03-05: lvllm-v1.9.0 - Optimized GPU prefill and regular prefill to ensure output quality
2026-03-01: lvllm-v1.8.10 - Fixed known issues, added new model support
2026-02-02: lvllm-v1.7.0 - Added EP parallel support, running minimax-m2.1 model on 8 GPUs requires --enable_expert_parallel
2026-01-26: lvllm-v1.6.1 - fp8 models support FP8 + INT4 inference, supports GPU Prefill acceleration (high memory usage!)
2026-01-25: lvllm-v1.6.0 - fp8 models support GPU Prefill acceleration (high memory usage!)
2026-01-24: lvllm-v1.5.8 - AWQ 4-bit symmetric quantization models support GPU Prefill acceleration
2026-01-21: lvllm-v1.5.7 - Fixed numerical stability issues with MiniMax-M2.1 model
2026-01-08: lvllm-v1.5.1 - For long context scenarios, supports separation of prefill and decoding, GPU prefill runs in parallel with CPU-GPU hybrid decoding
```

### Supported models

Most original MOE models verified by vLLM (Qwen3/GLM/MiniMax series etc.):
gemma-4-26B-A4B-it, NVIDIA-Nemotron-3-Super-120B-A12B-BF16, Ornith-1.0-35B-FP8, Qwen3.6/3.5-35B-A3B,
Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, Qwen3-Coder-Next, Qwen3-Next-80B-A3B-Instruct,
Qwen3-Coder-30B-A3B-Instruct, Qwen3-VL-30B-A3B-Instruct, MiniMax-M3/M2.7/M2.5/M2.1, GLM-5.2-NVFP4
[sm120], GLM-4.7(-Flash)/4.6V, Kimi k2.6/k2.5, **deepseek-ai/DeepSeek-V4-Flash-0731 [sm120]**.

Unlisted original MOE models from Qwen3, GLM, and MiniMax series are theoretically supported and
pending actual testing.

### Supported quantization formats

| Model File | Runtime Format |
|------------|----------------|
| bfloat16 | bfloat16/float16 |
| float16 | bfloat16/float16 |
| fp8 model | fp8 |
| nvfp4 model | NVFP4 and ModelOpt W4A16 NVFP4 |
| mxfp4 model <sup>Note 1</sup> | mxfp4 |
| awq 4bit symmetric quantization model <sup>Note 1</sup> | w4a16 |

```bash
Note 1: AWQ 4bit symmetric quantization models are available at https://hf-mirror.com/cyankiwi
Note 2: DeepSeek V4 - SM80/SM86/SM89 requires a dedicated version:
https://github.com/guqiong96/Lvllmds4-x/releases SM80+  (another SM120 version:
https://github.com/guqiong96/Lvllmds4/releases SM120)
```

### Quick start

```bash
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=CPU_CORE \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024 \
LK_POWER_SAVING=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve /home/guqiong/Downloads/DeepSeek-V4-Flash-0731 \
  --host 0.0.0.0 \
  --port 8070 \
  --tensor-parallel-size 2 \
  --max-model-len 32000 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "mode": "VLLM_COMPILE"}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --dtype bfloat16 \
  --max-num-seqs 2 \
  --enable-auto-tool-choice \
  --kv-cache-dtype fp8_ds_mla \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --disable-custom-all-reduce
```

### Configuration parameters

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LVLLM_MOE_NUMA_ENABLED` | core | `0` | enable hybrid inference: `1`-on, `0`-off (off = same as stock vllm) |
| `LK_THREAD_BINDING` | perf | `CPU_CORE` | `CPU_CORE` bind by core, `NUMA_NODE` bind by node |
| `LK_THREADS` | perf | - | thread count = (physical cores) / (#GPUs) |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU | none | expert layers resident in VRAM, e.g. `0`, `0-1`, `0,9` |
| `LVLLM_GPU_PREFETCH_WINDOW` | prefill | none | prefetch window size, typically `1` |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | prefill | none | GPU prefill starts when input len >= value; `0` disables |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | perf | 1 | `1`: avoid NUMA node OOM |
| `LK_POWER_SAVING` | power | 0 | `1`: enable CPU power saving |

### Installation

```bash
# Uninstall old CUDA and NVIDIA driver, then install CUDA 13.2.1
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall
wget https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_595.58.03_linux.run
sudo sh cuda_13.2.1_595.58.03_linux.run

conda create -n Lvllm python==3.12.11 && conda activate Lvllm
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
sudo apt-get install libnuma-dev      # Ubuntu  /  sudo dnf install numactl-devel  # Rocky

pip install https://github.com/guqiong96/Lvllm/releases/download/lvllm-v2.4.0/lvllm-2.4.0-cp312-cp312-manylinux_2_34_x86_64.whl
# check the latest version at: https://github.com/guqiong96/Lvllm/releases
pip install https://github.com/guqiong96/Lvllm/releases/download/lvllm-v2.4.0/flashinfer_cubin-0.6.16.post3-py3-none-any.whl
```

From source:

```bash
git clone https://github.com/guqiong96/Lvllm.git
cd Lvllm
pip install setuptools_scm setuptools_rust
pip install torchaudio triton torchvision torch==2.13.0
VLLM_VERSION_OVERRIDE="2.4.0" CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip install -e . --no-build-isolation -vvv
```

### Optimization

- **MoE resident in VRAM**: `LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5` (format `0,1,8-9`; some models start at non-zero layer, e.g. Step-3.5-Flash at layer 3).
- **Enable GPU prefill**: `LVLLM_GPU_PREFETCH_WINDOW=1`, `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096`, `--max-num-batched-tokens 32000`.
- **Disable GPU prefill**: `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0`, `--max-num-batched-tokens 4096`.
- **Thread binding**: `LK_THREAD_BINDING=CPU_CORE` (best), `NUMA_NODE` (fixes extreme issues on virtualization / multi-instance).
- **BIOS NUMA**: AMD EPYC NPS4 / Intel XEON SNC4; use 2,4,8 nodes (multiple of GPU count is best), up to 32.
- **Thread count**: HT on → physical cores ÷ GPUs; HT off → (physical cores-2) ÷ GPUs.
- **Output performance**: `--compilation_config.mode VLLM_COMPILE` (RTX 2080ti+), `--compilation_config.cudagraph_mode FULL_DECODE_ONLY`.
- **VRAM**: `--max-num-batched-tokens 32000` drives max-batch VRAM usage.
- **CPU power saving**: `LK_POWER_SAVING=1`.

---

## 四、How to generate the lk_moe patch

The portable patch `patches/01_lk_moe__v0.29.0.patch` is generated by diffing the current tree against
the upstream tag (the local `v0.29.0` tag, no network needed). **Note: the patch only contains the
lk_moe integration code changes — it does NOT include the README.md / README_cn.md / RELEASE_NOTES.md
docs.**

```bash
git diff v0.29.0 -- . ':!README.md' ':!README_cn.md' ':!RELEASE_NOTES.md' > patches/01_lk_moe__v0.29.0.patch
```

Apply it on a clean upstream `v0.29.0` checkout:

```bash
git clone --branch v0.29.0 https://github.com/vllm-project/vllm.git
cd vllm
git apply ../Lvllm/patches/01_lk_moe__v0.29.0.patch
```

---

## 五、Release / packaging

The LvLLM release workflow is a clean editable-install + wheel build + auditwheel repair + upload:

```bash
# clean any previous build artifacts
rm -rf build/ CMakeCache.txt CMakeFiles/ *.egg-info/

# arch list covering the supported GPUs (Ampere sm75/sm80/sm86/sm89,
# Hopper sm90, Blackwell sm100/sm120)
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0"

# editable install to verify
VLLM_VERSION_OVERRIDE="2.4.0" CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip install -e . --no-build-isolation -vvv

# build the wheel
VLLM_VERSION_OVERRIDE="2.4.0" CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip wheel . --no-build-isolation -v --wheel-dir=dist

# repair the wheel (exclude the system/CUDA libs)
auditwheel repair dist/lvllm*-2.4.0-cp312-cp312-linux_x86_64.whl -w dist/ \
  --exclude libtorch_cuda.so \
  --exclude libtorch_cpu.so \
  --exclude libtorch_python.so \
  --exclude libc10.so \
  --exclude libc10_cuda.so \
  --exclude libcublas.so.13 \
  --exclude libcublasLt.so.13 \
  --exclude libcufft.so.12 \
  --exclude libcusparse.so.12 \
  --exclude libcusparseLt.so.0 \
  --exclude libcurand.so.10 \
  --exclude libcudnn.so \
  --exclude libnccl.so.2 \
  --exclude libnvshmem_host.so.3 \
  --exclude libnvrtc.so.13.2.78 \
  --exclude libnvJitLink.so.13 \
  --exclude libcuda.so.595.58.03 \
  --exclude libnvrtc-builtins.so

```
