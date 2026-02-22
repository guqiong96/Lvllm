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
- [How to Run Qwen3.5](#how-to-run-qwen35)
- [How to Run MiniMax-M2.5](#how-to-run-minimax-m25)
- [Configuration Parameters](#configuration-parameters)
- [Installation Steps](#installation-steps)
- [Update](#update)
- [Optimization Tips](#optimization-tips)

## Version Changes

```bash
2026-02-18: lvllm-v1.8.1 - fix known issues, support new models
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
| Qwen3.5-397B-A17B | ✅ Tested |
| Qwen3-Coder-Next | ✅ Tested |
| Qwen3-Next-80B-A3B-Instruct | ✅ Tested |
| Qwen3-Coder-30B-A3B-Instruct | ✅ Tested |
| Qwen3-VL-30B-A3B-Instruct | ✅ Tested |
| MiniMax-M2.5 | ✅ Tested |
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

## How to Run Qwen3.5
```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h

PYTORCH_ALLOC_CONF=expandable_segments:True \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=0 \
vllm serve \
    --model /home/guqiong/Models/Qwen3.5-397B-A17B-FP8 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len 40000 \
    --gpu-memory-utilization 0.80 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --swap-space 0 \
    --served-model-name Qwen3.5-397B-A17B-FP8 \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 4 \
    --compilation_config.mode VLLM_COMPILE \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --language-model-only
```

# How to Run MiniMax-M2.5

 
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-1 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
vllm serve \
    --model /home/guqiong/Downloads/MiniMax-M2.5 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len 50000 \
    --gpu-memory-utilization 0.80 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --swap-space 0 \
    --served-model-name MiniMax-M2.5 \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --dtype bfloat16 \
    --max-num-seqs 4 \
    --compilation_config.mode VLLM_COMPILE \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think 
```
```bash 
# reduce performance issue when encountering performance problem, you can try binding by NUMA node and reduce thread count
```

```bash 
--enable_expert_parallel # enable expert parallelism, 8-card inference of minimax-m2.1 model requires setting
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
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | Performance Parameter | 0 | `0`：load model quickly, `1`：load model slowly to avoid OOM | Suggested value: use `0` when memory is abundant, use `1` when memory is tight |
| `LVLLM_MOE_QUANT_ON_GPU` | Performance Parameter | 0 | `0`：enable CPU expert quantization, `1`：enable GPU expert quantization | enable if GPU memory is abundant (only effective at loading time, inference will not occupy extra GPU memory)，accelerate model loading speed |

## Configuration Parameters

| Parameter | Example Value | Description |
|-----------|-------|-------------|
| `model` | `/home/guqiong/Models/Models/MiniMax-M2.5` | Model directory path |
| `host` | `0.0.0.0` | Service binding IP address |
| `port` | `8070` | Service binding port number |
| `tensor-parallel-size` | `2` | Tensor parallel size, less than or equal to the number of GPUs |
| `pipeline-parallel-size` | `2` (commented) | Pipeline parallel size, less than or equal to the number of GPUs |
| `max-model-len` | `18000` | Maximum context length, less than or equal to the model's maximum length |
| `gpu-memory-utilization` | `0.92` | GPU memory utilization percentage allocated to vLLM, less than or equal to 1 |
| `trust-remote-code` | `true` | Whether to trust remote code, recommended value: `true` |
| `tokenizer-mode` | `auto` | Tokenizer mode, recommended value: `auto` |
| `swap-space` | `0` | Swap space size in GB, recommended value: `0` |
| `served-model-name` | `MiniMax-M2.5` | Served model name |
| `compilation_config.cudagraph_mode` | `FULL_DECODE_ONLY` | Enable CUDA graph mode, recommended value |
| `enable_prefix_caching` | `true` | Enable prefix caching, recommended value |
| `enable-chunked-prefill` | `true` | Enable chunked prefill, recommended value |
| `max_num_batched_tokens` | `18000` | Maximum number of batched tokens, recommended value: `1024` when GPU prefill is disabled, `max-model-len` when GPU prefill is enabled |
| `dtype` | `bfloat16` | Model intermediate calculation data type, recommended value: `bfloat16` or `float16` |
| `max_num_seqs` | `4` | Maximum concurrent request sequences, recommended value: `1` to `4` |
| `compilation_config.mode` | `VLLM_COMPILE` | Optimize model, recommended value |
 

| Parameter | Description |
|-----------|-------------|
| `enable-auto-tool-choice` | Allow tool calls (commented) |
| `kv_cache_dtype: "fp8"` | KV Cache data type, enabled for 40-series and 50-series GPUs (commented) |
| `speculative-config` | Speculative decoding configuration, not recommended (commented) |
| `mm-encoder-tp-mode: "data"` | encoder TP模式 (commented) |

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
pip install torch==2.9.1

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
git fetch && git reset --hard origin/main && git clean -fd # This command is suitable for regular users; those who want to keep their local modifications should know to handle them in advance

# Install PyTorch 2.9.1
pip uninstall torchaudio triton torchvision torch vllm
pip install torch==2.9.1

# Qwen3-VL GLM4.6V requires xformers to be installed

# Compile and install
python use_existing_torch.py
pip install -r requirements/build.txt
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv

rm -rf ~/.cache/vllm
```  
 
## Optimization Tips
 
### MoE Resident in VRAM, Linear Increase in Decode and Prefill Speed
```bash
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 # 0-5 MoE layers resident in VRAM
#LVLLM_GPU_RESIDENT_MOE_LAYERS=0,1,8-9 # 0,1,8-9 MoE layers resident in VRAM
#LVLLM_GPU_RESIDENT_MOE_LAYERS="" # Disable MoE resident in VRAM
``` 
 
### Enable GPU Prefill
```bash
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-2 # 0-2 MoE layers resident in VRAM, to achieve optimal performance with GPU prefill enabled, include layer 0
#LVLLM_GPU_RESIDENT_MOE_LAYERS=3-4 # Some models have non-zero starting layer numbers, such as Step-3.5-Flash model starting at 3
LVLLM_GPU_PREFETCH_WINDOW=1 # Prefetch 1 layer, recommended value is 1-2, more is meaningless
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 # Start GPU prefill when input length reaches 4096, can be decreased or increased based on CPU prefill performance, starting prefill earlier or later
max_num_batched_tokens: 65536 # Same as context size for optimal performance, can be appropriately reduced based on VRAM availability, exceeding context size is meaningless
``` 
 
### Disable GPU Prefill
```bash
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 # Disable GPU prefill
#LVLLM_GPU_PREFILL_MIN_BATCH_SIZE="" # Disable GPU prefill 
max_num_batched_tokens: 1024 # 1024 to 8192, too large is meaningless (occupies too much VRAM and long startup time)
``` 
 
### Thread Binding to CPU Cores
```bash
LK_THREAD_BINDING=CPU_CORE # Bind to CPU cores (including hyper-threading logical cores), optimal performance
#LK_THREAD_BINDING=NUMA_NODE # Bind to NUMA nodes, second choice to solve extreme performance issues on virtualization platforms
``` 
### BIOS NUMA Settings
```bash
AMD EPYC: Set NPS4 for optimal performance
Intel XEON: Set SNC4 for optimal performance
General: 2,4,8 nodes, maximum support for 32 nodes, more nodes are better, node count being a multiple of GPUs for optimal performance # Some virtualization platforms or Intel platforms should not set 5 or 10 nodes, set to 2 nodes to avoid performance issues
```
 
### Thread Count Settings
```bash
# Thread count <= (core count - x) / tensor parallelism size (TP size)  # x threads reserved for other tasks, at least 4 threads
LK_THREADS=44                    # 96 cores, 2 GPUs, 44 threads per GPU, 88 threads total, 8 threads reserved for other tasks
# Too many threads may cause performance issues        # Although the system automatically adjusts the number of threads, manual setting is recommended for testing
```
### Decode Performance
```bash
compilation_config.mode: "VLLM_COMPILE"                 # Supports 2080ti and above GPUs
compilation_config.cudagraph_mode: "FULL_DECODE_ONLY"   # Enable CUDAGraph
```
 
### VRAM Settings
```bash
gpu-memory-utilization: 0.80 # 24G VRAM with GPU prefill enabled, leave sufficient temporary VRAM for calculations, otherwise long context prefill performance will drop significantly, startup time will be too long
# gpu-memory-utilization: 0.92 # 24G VRAM with GPU prefill disabled, no need to leave too much VRAM
# max_num_seqs: 1 # Maximum 1 concurrent, maximum VRAM savings
max_num_seqs: 4 # Maximum 4 concurrent, regular VRAM savings
max_num_batched_tokens: 1024  # Save VRAM when GPU prefill is disabled, performance remains unchanged, but if enable GPU prefill will cause performance drop
max_num_batched_tokens: 65536 # or larger and less than context size, enable GPU prefill, obtain best performance, but if disable GPU prefill will cause performance drop
```
### CPU Power Saving
```bash
LK_POWER_SAVING=1 # enable low power mode while inference, reduce CPU temperature, slightly reduce performance 
```

### FP8 Model Weight Runtime Format
```bash
LVLLM_MOE_USE_WEIGHT=INT4 # Model MoE expert weights use INT4 inference, other parts remain FP8, enabling almost no impact on accuracy, speed order: INT4 > TO_DTYPE > KEEP
```

### Model Loading with NUMA Interleaving
```bash
LVLLM_ENABLE_NUMA_INTERLEAVE=1 # Slow model loading can prevent OOM. Recommended values: use `0` when memory is sufficient during model file loading, use `1` when memory is limited.
```

### Model Loading with GPU Expert Quantization
```bash
LVLLM_MOE_QUANT_ON_GPU=1 # Enable GPU expert quantization, recommended values: enable when sufficient GPU memory is available (only effective during model loading, not during inference), speed up model loading
```
