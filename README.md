# LvLLM GPU and NUMA Dual Parallelism [[中文说明]](./README_cn.md)

LvLLM is a special extension of vLLM that fully utilizes CPU and GPU computing resources with an efficient GPU parallel + NUMA parallel architecture, suitable for MOE model hybrid inference.

## System Features

- **GPU + NUMA Dual Parallelism**: Supports three computing modes: CPU-GPU hybrid decoding, CPU-GPU hybrid prefill, and GPU prefill
- **VRAM + Memory Load Balancing**: Total model footprint = VRAM + memory, accommodating 1+1=2 models, with 100% VRAM utilization <sup>Note 1</sup>
- **GPU Prefill Optimization**: GPU prefill runs in parallel with CPU-GPU hybrid decoding, achieving nearly 100% GPU utilization
- **NUMA Thread Optimization**: Cross-node communication reduced to as low as 3%, L3 cache hit rate over 50%, GPU load can reach 33% to 50% during decoding

Note 1: Enabling GPU prefill for models other than BF16 and F16 original models will additionally occupy memory [!not VRAM]

## Relationship with vLLM

Lvllm uses the latest vLLM source code and has redesigned and implemented MOE model hybrid inference modules, maintaining 100% full compatibility with vLLM.

## Usage Instructions [[中文说明]](./README_cn.md)
- [Version Changes](#version-changes)
- [Supported Models](#supported-models)
- [Performance Reference](#performance-reference)
- [Running Commands](#running-commands)
- [Configuration File](#configuration-file)
- [Installation Steps](#installation-steps)
- [Update](#update)

## Version Changes

```bash
2026-02-02：lvllm-v1.7.0 - support for EP parallelism, 8-card running minimax-m2.1 model requires setting --enable_expert_parallel
2026-01-26: lvllm-v1.6.1 - fp8 model support for FP8 + INT4 inference, support for GPU Prefill acceleration (high memory usage!)
2026-01-25: lvllm-v1.6.0 - fp8 model support for GPU Prefill acceleration (high memory usage!)
2026-01-24: lvllm-v1.5.8 - AWQ 4-bit symmetric quantized model support for GPU Prefill acceleration
2026-01-21: lvllm-v1.5.7 - Fixed numerical calculation stability issues in MiniMax-M2.1 model
2026-01-08: lvllm-v1.5.1 - For long context scenarios, supports separation of prefill and decoding, GPU prefill runs in parallel with CPU-GPU hybrid decoding
2026-01-04: v1.4.0 Optimized decode speed
2025-12-28: Optimized inference speed: bfloat16, awq4bit; optimized NUMA data access for multi-GPU; enabled NUMA nodes for multi-GPU for best performance; removed GGUF model support
2025-12-16 v1.2.0 Synchronized upstream vllm code to latest, optimized lk_moe to reduce memory usage
2025-12-14 v1.1.2 Added AWQ-4bit quantized model (symmetric quantization avx2 version) inference support - verified with cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit and cpatonn/
2025-12-9: Added LVLLM_MOE_USE_WEIGHT environment variable, supporting MOE modules to use two modes for fp8 model inference:
2025-11-1: Supports tensor parallelism, pipeline multi-card inference https://b23.tv/xzHieMs
2025-10-30: Supports Qwen3 series model GGUF hybrid inference (excluding Qwen3-Coder-30B-A3B-Instruct GGUF) [Check new parameters in config.yaml]
2025-10-19: FP8 supports GPU+NUMA hybrid inference for MOE models!! [VRAM FP8 precision, memory FP16 precision] Verified with GLM-4.5-Air-FP8
2025-10-14: Enabled cuda graph, decode speed doubled!! Output quality improved!!
2025-09-30 Verified: Qwen3-Next-80B-A3B-Instruct, Qwen3-Coder-30B-A3B-Instruct

```

## Supported Models

Most of the original MOE models verified by vLLM

| Model Name | Status |
|---------|------|
| Qwen3-Coder-Next | ✅ Tested |
| Qwen3-Next-80B-A3B-Instruct | ✅ Tested |
| Qwen3-Coder-30B-A3B-Instruct | ✅ Tested |
| Qwen3-VL-30B-A3B-Instruct | ✅ Tested |
| MiniMax-M2.1 | ✅ Tested |
| GLM-4.7 | ✅ Tested |
| GLM-4.7-Flash | ✅ Tested |
| GLM-4.6V | ✅ Tested |
| Kimi k2.5 | ✅ Tested |

Unlisted original MOE models from Qwen3 series, GLM series, and MiniMax series are theoretically supported and pending actual testing.

## Unsupported Models

| Model Name | Status |
|---------|------|
| DeepSeek-V3.2| Pending |

## Supported Model Weight Formats and Runtime Formats

| Model File | Runtime Format |
|---------|------------|
| bfloat16 | bfloat16/float16| 
| float16 | bfloat16/float16| 
| fp8 model | fp8, fp8+bfloat16, fp8+int4 | 
| awq 4bit symmetric quantized model <sup>Note 1</sup> | int4 |

Note 1: https://hf-mirror.com/cyankiwi provides AWQ 4bit symmetric quantized models


## Performance Reference

| Model | Runtime Format | Prefill Speed (tokens/s) | Decode Speed (tokens/s) | CPU | GPU | Memory |
|------|----------|---------------------|-------------------|----------|---------|---------|
| Qwen3-Next-80B-A3B-Instruct Original | bfloat16 |15000 <sup>Note 1</sup> | 90 | Dual EPYC 9555ES | Single Nvidia RTX Pro 6000 | 6400MT/s |
| MiniMax-M2.1 Original | fp8+bfloat16 | 5000 <sup>Note 1</sup> | 29 | Dual EPYC 9684x | Single Nvidia RTX 5090 | 4800MT/s |

Note 1: Enabling GPU Prefill, Input Length 32K-64K

## Running Commands

```bash
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=88 OMP_NUM_THREADS=88 vllm serve --config config.yaml # GPU prefill not enabled
```

```bash
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=88 OMP_NUM_THREADS=88 LVLLM_MOE_USE_WEIGHT=INT4 LVLLM_GPU_RESIDENT_MOE_LAYERS=0-1 LVLLM_GPU_PREFETCH_WINDOW=1 LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 vllm serve --config config.yaml # GPU prefill enabled
```
```bash
# When encountering performance issues, you can try binding threads by NUMA node and reduce the number of threads
```

```bash 
--enable_expert_parallel # EP parallelism enabled，run MiniMax-M2.1 model with 8 GPUs
```

| Environment Variable | Type | Default Value | Description | Notes |
|--------|------|--------|------|------|
| `LVLLM_MOE_NUMA_ENABLED` | Core Parameter | `0` | Whether to enable hybrid inference: `1`-enabled, `0`-disabled | Set to `0` to disable hybrid inference, behavior is the same as vLLM |
| `LK_THREAD_BINDING` | Performance Parameter | `CPU_CORE` | Thread binding policy: `CPU_CORE` - bind by CPU core, `NUMA_NODE` - bind by NUMA node | Default is binding by CPU core, when encountering performance issues, you can try binding by NUMA node |
| `LK_THREADS` | Performance Parameter | Auto-calculated | Number of threads: physical cores - 4 | For multi-GPU multi-process, (physical cores - 4) divided by number of processes |
| `OMP_NUM_THREADS` | Performance Parameter | System logical core count | OpenMP thread count: set to same as `LK_THREADS` | |
| `LVLLM_MOE_USE_WEIGHT` | Performance Parameter | `TO_DTYPE` | Runtime expert weight format `TO_DTYPE`: same as dtype in config.yaml, bfloat16/float16, `KEEP`: same as model, `INT4`: int4 |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU Prefill Parameter | None | MOE expert layers resident on GPU `0`: layer 0, `0-1`: layers 0 to 1, `0,9`: layers 0 and 9 | After reserving sufficient KV Cache VRAM, allocating multiple layers can increase performance and reduce corresponding memory usage, including layer 0 to achieve acceleration effect |
| `LVLLM_GPU_PREFETCH_WINDOW` | GPU Prefill Parameter | None | Prefetch window size `1`: prefetch 1 layer of MOE experts | Generally, prefetching 1 to 2 layers is sufficient |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | GPU Prefill Parameter | None | Minimum input length for using GPU prefill `4096`: when input length reaches this value, start GPU prefill | The value should not be too small, set to 0 to disable GPU prefill function |

## Configuration File

Example config.yaml, `Recommended values` do not need to be modified when running different models

```bash
model: "/home/guqiong/Models/GLM-4.7-Flash-AWQ-4bit"  # Model directory
host: "0.0.0.0"                                       # Service binding IP address
port: 8070                                            # Service binding port number
tensor-parallel-size: 2                               # Tensor parallelism size, less than or equal to number of GPUs,
#pipeline-parallel-size: 2                            # Pipeline parallelism size, less than or equal to number of GPUs
max-model-len: 18000                                  # Maximum context length, less than or equal to model maximum length
gpu-memory-utilization: 0.92                          # GPU VRAM allocation percentage for lvllm, less than or equal to 1
trust-remote-code: true                               # Whether to trust remote code, recommended value
tokenizer-mode: "auto"                                # Tokenizer mode, recommended value
swap-space: 0                                         # Swap space size, in GB, recommended value
served-model-name: "GLM-4.7-Flash-AWQ-4bit"           # Service model name
compilation_config.cudagraph_mode: "FULL_DECODE_ONLY" # Enable CUDA graph mode, recommended value
enable_prefix_caching: true                           # Enable prefix caching, recommended value
enable-chunked-prefill: true                          # Enable chunked prefill, recommended value
max_num_batched_tokens: 18000                         # Maximum number of batched tokens, recommended value when GPU prefill is disabled: 1024, recommended value when GPU prefill is enabled: same as max-model-len
dtype: "bfloat16"                                     # Model intermediate calculation data type, recommended value bfloat16 or float16
max_num_seqs: 4                                       # Maximum concurrent request sequences, recommended value 1 to 4
compilation_config.mode: "VLLM_COMPILE"               # Optimize model, recommended value
# kv_cache_dtype: "fp8"                               # KV Cache data type, can be enabled for 40-series, 50-series GPUs
# enable-auto-tool-choice: true                       # Enable auto tool choice
# tool-call-parser: "minimax_m2"                      # MiniMax M2.1 model configuration parameter
# reasoning-parser: "minimax_m2_append_think"         # MiniMax M2.1 model configuration parameter
# tool-call-parser: glm47                             # GLM4.7 model configuration parameter
# reasoning-parser: glm45                             # GLM4.7 model configuration parameter
# tool-call-parser: "kimi_k2"                        # Kimi k2.5 model configuration parameter
# reasoning-parser: "kimi_k2"                        # Kimi k2.5 model configuration parameter
# reasoning-parser: "step3p5"                         # Kimi k2.5 model configuration parameter                                         
# tool-call-parser: "step3p5"                         # Kimi k2.5 model configuration parameter                                         
# hf-overrides.num_nextn_predict_layers: 1            # Kimi k2.5 model configuration parameter
# speculative_config.method: "step3p5_mtp"            # Kimi k2.5 model configuration parameter
# speculative_config.num_speculative_tokens: 1        # Kimi k2.5 model configuration parameter
# disable-cascade-attn: true                          # Step-3.5-Flash model configuration parameter
# tool-call-parser: "qwen3_coder"                     # Qwen3-Coder-Next model configuration parameter
```

## Installation Steps

### 1. Install CUDA 12.9

```bash
# Uninstall old version CUDA and NVIDIA drivers
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall

# Download and install CUDA 12.9
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run
```

### 2. Create Python Environment

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm

# Upgrade libstdcxx-ng (avoid glibcxx version issues)
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install NUMA library
sudo apt-get install libnuma-dev      # Ubuntu
sudo dnf install numactl-devel        # Rocky Linux
```

### 3. Install Dependencies

```bash
# Clone repository
git clone https://github.com/guqiong96/Lvllm.git
cd Lvllm

# Install PyTorch 2.9.1
pip install torchaudio triton torchvision torch==2.9.1

# Use existing PyTorch
python use_existing_torch.py

# Install build dependencies
pip install -r requirements/build.txt
```

### 4. Install Lvllm

```bash
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```

**Parameter Explanation:**
- `MAX_JOBS=32 NVCC_THREADS=1`: Reduce compilation memory usage
- `CMAKE_BUILD_TYPE=Release`: Performance optimization option
- `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"`: Performance optimization option


## Update

If Lvllm is already installed and you need to update to the latest version, execute the following commands:

```bash
# Normal situation
git pull --force
# If conflicts occur
git fetch origin
git reset --hard origin/main

# Install PyTorch 2.9.1
pip uninstall torchaudio triton torchvision torch
pip install torchaudio triton torchvision torch==2.9.1

# Qwen3-VL GLM4.6V requires xformers to be installed

# Compile and install
python use_existing_torch.py
pip install -r requirements/build.txt
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv

rm -rf ~/.cache/vllm
```

Simple update of Lvllm, when Lvllm's update does not involve upstream vllm updates, only need to execute the following commands:
```bash
git pull --force
pip uninstall lk_moe
pip install lk_moe
rm -rf ~/.cache/vllm
```