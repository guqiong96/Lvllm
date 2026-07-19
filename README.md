# LvLLM GPU + NUMA Dual Parallel [[中文]](./README_cn.md)

LvLLM is a special extension of vLLM that fully utilizes CPU and GPU computing resources. It features an efficient GPU parallel + NUMA parallel architecture, suitable for MOE model hybrid inference.

> **Core Engine:** The actual hybrid inference functionality—including CPU-GPU collaborative computation, NUMA-aware scheduling, expert weight management, and quantization kernel execution—is powered entirely by **[lk_moe](https://pypi.org/project/lk-moe/)**, a highly optimized MOE hybrid inference engine. Within LvLLM (for vLLM) and [Lsglang](https://github.com/guqiong96/Lsglang) (for sglang), each MOE layer can flexibly choose between the original GPU computation path or invoke lk_moe for hybrid inference. For DeepSeek V4 (SM120 architecture), a specialized version [Lvllmds4](https://github.com/guqiong96/Lvllmds4) is also available.

## System Features

- **GPU + NUMA Dual Parallel**: Supports three computing modes: CPU-GPU hybrid decoding, CPU-GPU hybrid prefill, and GPU prefill
- **Memory + VRAM Load Balancing**: Total model footprint = VRAM + memory, accommodating model 1+1=2, 100% VRAM utilization <sup>Note 1</sup>
- **GPU Prefill Optimization**: GPU prefill runs in parallel with CPU-GPU hybrid decoding, achieving nearly 100% GPU utilization
- **NUMA Thread Optimization**: Cross-node communication as low as 3%, L3 cache hit rate above 50%, GPU load can reach 33% to 50% during decoding

## Relationship with vLLM

LvLLM uses the latest vLLM source code and has redesigned the MOE model hybrid inference module, maintaining 100% full compatibility with vLLM<sup>Note 1</sup>.

Note 1: x86 CPUs with AVX2 or higher instruction set and Nvidia GPU sm75 or higher architecture

## Usage Guide [[中文]](./README_cn.md)
- [Version Changes](#version-changes)
- [Supported Models](#supported-models)
- [Supported Quantization Formats](#supported-quantization-formats)
- [Run Command Reference](#run-command-reference)
- [Configuration Parameters](#configuration-parameters)
- [Installation Steps](#installation-steps)
- [Optimization](#optimization)

## Version Changes

```bash
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
2026-01-04: v1.4.0 Optimized decode for speed improvement
2025-12-28: Optimized inference speed: bfloat16, awq4bit; Optimized NUMA data access for multi-GPU; Enabled NUMA nodes for multi-GPU for optimal performance; Removed GGUF model support
2025-12-16 v1.2.0 Synced upstream vllm code to latest, lk_moe optimization to reduce memory usage
2025-12-14 v1.1.2 Added AWQ-4bit symmetric quantization model inference support
2025-12-9: Added LVLLM_MOE_USE_WEIGHT environment variable, supports two modes for MOE module to inference fp8 models:
2025-11-1: Added tensor parallel, pipeline multi-GPU inference support https://b23.tv/xzHieMs
2025-10-30: Added Qwen3 series GGUF hybrid inference support (excluding Qwen3-Coder-30B-A3B-Instruct GGUF) [Check new parameters in config.yaml]
2025-10-19: FP8 supports GPU+NUMA hybrid inference for MOE models!! [VRAM FP8 precision, memory FP16 precision] Verified with GLM-4.5-Air-FP8
2025-10-14: Enabled cuda graph, decode speed doubled!! Output quality improved!!
2025-09-30 Verified: Qwen3-Next-80B-A3B-Instruct, Qwen3-Coder-30B-A3B-Instruct
```

## Supported Models

Most original MOE models verified by vLLM

| Model Name | Status |
|------------|--------|
| gemma-4-26B-A4B-it | ✅ Tested |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | ✅ Tested |
| Ornith-1.0-35B-FP8 | ✅ Tested |
| Qwen3.6-35B-A3B | ✅ Tested |
| Qwen3.5-35B-A3B | ✅ Tested |
| Qwen3.5-122B-A10B | ✅ Tested |
| Qwen3.5-397B-A17B | ✅ Tested |
| Qwen3-Coder-Next | ✅ Tested |
| Qwen3-Next-80B-A3B-Instruct | ✅ Tested |
| Qwen3-Coder-30B-A3B-Instruct | ✅ Tested |
| Qwen3-VL-30B-A3B-Instruct | ✅ Tested |
| MiniMax-M3 | ✅ Tested |
| MiniMax-M2.7 | ✅ Tested |
| MiniMax-M2.5 | ✅ Tested |
| MiniMax-M2.1 | ✅ Tested |
| GLM-4.7 | ✅ Tested |
| GLM-4.7-Flash | ✅ Tested |
| GLM-4.6V | ✅ Tested |
| Kimi k2.6 | ✅ Tested |
| Kimi k2.5 | ✅ Tested |

Unlisted original MOE models from Qwen3, GLM, and MiniMax series are theoretically supported and pending actual testing.

## Supported Quantization Formats

| Model File | Runtime Format |
|------------|----------------|
| bfloat16 | bfloat16/float16 |
| float16 | bfloat16/float16 |
| fp8 model | fp8 |
| nvfp4 model | NVFP4 and ModelOpt W4A16 NVFP4 |
| mxfp4 model <sup>Note 1</sup> | mxfp4 |
| awq 4bit symmetric quantization model <sup>Note 1</sup> | w4a16 |

Note 1: AWQ 4bit symmetric quantization models are available at https://hf-mirror.com/cyankiwi
Note 2: deepseek v4 requires a dedicated version:
https://github.com/guqiong96/Lvllmds4/releases

## Run Command Reference

```bash
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-1,33-34 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1 \
vllm serve \
    --model /home/guqiong/Models/Qwen3.6-35B-A3B \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Qwen3.6-35B-A3B \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32000 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

## Configuration Parameters

| Environment Variable | Type | Default | Description | Notes |
|---------------------|------|---------|-------------|-------|
| `LVLLM_MOE_NUMA_ENABLED` | Core Parameter | `0` | Enable hybrid inference: `1`-enable, `0`-disable | Set to `0` to disable hybrid inference, behavior matches vLLM |
| `LK_THREAD_BINDING` | Performance Parameter | `CPU_CORE` | Thread binding strategy: `CPU_CORE`-bind to CPU cores, `NUMA_NODE`-bind to NUMA nodes | Default binds to CPU cores, try NUMA_NODE if performance issues occur |
| `LK_THREADS` | Performance Parameter | Auto-calculated | Thread count: physical cores - 4 | For multi-GPU multi-process: (physical cores - 4) / number of processes |
| `OMP_NUM_THREADS` | Performance Parameter | System logical cores | OpenMP threads: set same as `LK_THREADS` | |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU Prefill Parameter | None | MOE expert layers resident on GPU: `0`-layer 0, `0-1`-layers 0 to 1, `0,9`-layers 0 and 9 | After reserving KV Cache VRAM, allocating multiple layers improves performance and reduces corresponding memory usage |
| `LVLLM_GPU_PREFETCH_WINDOW` | GPU Prefill Parameter | None | Prefetch window size `1`: prefetch 1 layer of MOE experts | Generally 1-2 layers is sufficient |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | GPU Prefill Parameter | None | Minimum input length for GPU prefill `4096`: GPU prefill starts when input length reaches this value | Should not be set too small, set to 0 to disable GPU prefill |
| `LK_POWER_SAVING` | CPU Power Saving | 0 | `1`: enable CPU power saving mode, `0`: disable | Recommended: `0` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | Performance Parameter | 0 | `0`: fast model loading, `1`: slow loading to avoid OOM | Recommended: use `0` if memory is sufficient, `1` if memory is tight |

| Parameter | Example Value | Description |
|-----------|--------------|-------------|
| `tensor-parallel-size` | `2` | Tensor parallel size, <= number of GPUs |
| `compilation_config.cudagraph_mode` | `FULL_DECODE_ONLY` | Enable CUDA graph mode, recommended |
| `enable_prefix_caching` | `true` | Enable prefix caching, recommended |
| `enable-chunked-prefill` | `true` | Enable chunked prefill, recommended |
| `max_num_batched_tokens` | `18000` | Maximum batched tokens, recommended: 1024 without GPU prefill, 32000 with GPU prefill |
| `compilation_config.mode` | `VLLM_COMPILE` | Optimize model, recommended |

## Installation Steps

### 1. Install CUDA 13.2.1

```bash
# Uninstall old CUDA and NVIDIA driver
sudo /usr/local/cuda/bin/cuda-uninstaller   
sudo nvidia-uninstall

# Download and install CUDA 13.2.1
wget https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_595.58.03_linux.run
sudo sh cuda_13.2.1_595.58.03_linux.run
```

### 2. Create Python Environment

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm

# Upgrade libstdcxx-ng (to avoid glibcxx version issues)
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install NUMA library
sudo apt-get install libnuma-dev      # Ubuntu
sudo dnf install numactl-devel        # Rocky Linux
```
 

### 3. Install LvLLM

```bash
pip install https://github.com/guqiong96/Lvllm/releases/download/lvllm-v2.3.7/lvllm-2.3.7-cp312-cp312-manylinux_2_34_x86_64.whl
# check the latest version at: https://github.com/guqiong96/Lvllm/releases
```

## Compile and Install Lvllm

```bash 
git clone https://github.com/guqiong96/Lvllm.git
cd Lvllm
pip install setuptools_scm setuptools_rust
pip install torchaudio triton torchvision torch==2.11.0
VLLM_VERSION_OVERRIDE="2.3.4" CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```

## Optimization

### Keep MoE Layers Resident in VRAM - Linear Speedup for Decode and Prefill
```bash
# Keep MoE layers 0-5 resident in VRAM
# Format 0,1,8-9 means layers 0,1,8-9 are resident in VRAM
# Some models start at non-zero layer index, e.g., Step-3.5-Flash starts at layer 3
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5
```

### Enable GPU Prefill
```bash
LVLLM_GPU_PREFETCH_WINDOW=1
# Start GPU prefill when input length reaches 4096
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096
# Set maximum batched tokens accordingly
--max-num-batched-tokens 32000
```

### Disable GPU Prefill
```bash
# Disable GPU prefill
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0
# Set maximum batched tokens accordingly
--max-num-batched-tokens 4096
```

### Bind Threads to CPU Cores
```bash
# Bind to CPU cores (including hyper-threading logical cores), best performance
LK_THREAD_BINDING=CPU_CORE
# Bind to NUMA nodes, secondary option, resolves extreme performance issues on virtualized platforms and multi-instance deployment
LK_THREAD_BINDING=NUMA_NODE
```

### BIOS NUMA Settings
```bash
AMD EPYC: Set NPS4 for best performance
Intel XEON: Set SNC4 for best performance
# Some virtualized platforms or Intel platforms should not use 5 or 10 nodes, use 2 nodes to avoid performance issues
Typically: 2, 4, or 8 nodes, up to 32 nodes supported. More nodes = better performance. Best performance when node count is multiple of GPU count.
```

### Thread Count Settings
```bash
# Thread count <= (cores - x) / tensor parallel size (TP size), where x is threads reserved for other tasks, minimum 4 threads
# 96 cores, 2 GPUs: 44 threads per GPU, 88 total threads, 8 threads reserved for other tasks
LK_THREADS=44
# Total threads exceeding physical core count may cause performance issues
# Although the system auto-adjusts thread count, manual setting is recommended for testing
```

### Output Performance
```bash
# Supports RTX 2080ti and above
--compilation_config.mode VLLM_COMPILE
# Enable CUDAGraph
--compilation_config.cudagraph_mode FULL_DECODE_ONLY
```

### VRAM Settings
```bash
# Maximum batched tokens consumes significant VRAM, adjust accordingly
--max-num-batched-tokens 32000
```

### CPU Power Saving
```bash
# Enable to reduce power consumption during idle inference
LK_POWER_SAVING=1
```