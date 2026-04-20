# LvLLM GPU and NUMA Dual Parallelism [[中文说明]](./README_cn.md)

LvLLM is a special extension of vLLM that fully utilizes CPU and GPU computing resources with an efficient GPU parallel + NUMA parallel architecture, suitable for MOE model hybrid inference.

## System Features

- **GPU + NUMA Dual Parallelism**: Supports three computing modes: CPU-GPU hybrid decoding, CPU-GPU hybrid prefill, and GPU prefill
- **VRAM + Memory Load Balancing**: Total model footprint = VRAM + memory, accommodating 1+1=2 models, with 100% VRAM utilization <sup>Note 1</sup>
- **GPU Prefill Optimization**: GPU prefill runs in parallel with CPU-GPU hybrid decoding, achieving nearly 100% GPU utilization
- **NUMA Thread Optimization**: Cross-node communication reduced to as low as 3%, L3 cache hit rate over 50%, GPU load can reach 33% to 50% during decoding
 
## Relationship with vLLM

Lvllm uses the latest vLLM source code and has redesigned and implemented MOE model hybrid inference modules, maintaining 100% full compatibility with vLLM<sup>Note 1</sup>.

Note 1: x86 CPUs with AVX2 or above instruction sets and Nvidia GPUs are supported.

## Usage Instructions [[中文说明]](./README_cn.md)
- [Version Changes](#version-changes)
- [Supported Models](#supported-models)
- [Performance Reference](#performance-reference)
- [How to Run Qwen3.6-35B-A3B](#how-to-run-qwen36-35b-a3b)
- [How to Run gemma-4-26B-A4B-it](#how-to-run-gemma-4-26b-a-4b-it)
- [How to Run NVIDIA-Nemotron-3-Super-120B-A12B-BF16](#how-to-run-nvidia-nemotron-3-super-120b-a12b-bf16)
- [How to Run Qwen3.5-122B-A10B](#how-to-run-qwen35-122b-a10b)
- [How to Run Qwen3.5-397B-A17B](#how-to-run-qwen35-397b-a17b)
- [How to Run MiniMax-M2.7](#how-to-run-minimax-m27)
- [How to Run Kimi-K2.6](#how-to-run-kimi-k26)
- [How to Run GLM-4.7-FP8](#how-to-run-glm-47-fp8)
- [Configuration Parameters](#configuration-parameters)
- [Installation Steps](#installation-steps)
- [Update](#update)
- [Optimization Tips](#optimization-tips)

## Version Changes

```bash
2026-04-06: lvllm-v2.1.0 - improve LK_POWER_SAVING=1 for power saving, support FP8+BF16+AWQ4bit hybrid MoE layer inference
2026-03-22: lvllm-v2.0.0 - FP8 MoE models support layer-wise loading when quantizing INT4 experts, reducing peak memory usage, LVLLM_ENABLE_MOE_LAYERWISEISE_LOAD=1
2026-03-19: lvllm-v1.9.10 - fix known issues，Supports the new moe model type, which does not have gate_proj, for example: NVIDIA-Nemotron-3-Super-120B-A12B-BF16
2026-03-11: lvllm-v1.9.2 - FP8、AWQ4bit MoE Models enable GPU Prefill acceleration without additional memory occupation, FP8 MoE Model cancel TO_DTYPE runtime type conversion, KEEP model temporarily not support GPU Prefill
2026-03-05: lvllm-v1.9.0 - Optimize GPU prefill and regular prefill to ensure output quality
2026-03-01: lvllm-v1.8.10 - fix known issues, support new models
2026-02-02：lvllm-v1.7.0 - support for EP parallelism, 8-card running minimax-m2.1 model requires setting --enable_expert_parallel
2026-01-26: lvllm-v1.6.1 - fp8 model support for FP8 + INT4 inference, support for GPU Prefill acceleration (high memory usage!)
2026-01-25: lvllm-v1.6.0 - fp8 model support for GPU Prefill acceleration (high memory usage!)
2026-01-24: lvllm-v1.5.8 - AWQ 4-bit symmetric quantized model support for GPU Prefill acceleration
2026-01-21: lvllm-v1.5.7 - Fixed numerical calculation stability issues in MiniMax-M2.1 model
2026-01-08: lvllm-v1.5.1 - For long context scenarios, supports separation of prefill and decoding, GPU prefill runs in parallel with CPU-GPU hybrid decoding
2026-01-04: v1.4.0 Optimized decode speed
2025-12-28: Optimized inference speed: bfloat16, awq4bit; optimized NUMA data access for multi-GPU; enabled NUMA nodes for multi-GPU for best performance; removed GGUF model support
2025-12-16 v1.2.0 Synchronized upstream vllm code to latest, optimized lk_moe to reduce memory usage
2025-12-14 v1.1.2 Added AWQ-4bit symmetric quantized model inference support
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
| gemma-4-26B-A4B-it | ✅ Tested |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | ✅ Tested |
| Qwen3.6-35B-A3B | ✅ Tested |
| Qwen3.5-35B-A3B | ✅ Tested |
| Qwen3.5-122B-A10B | ✅ Tested |
| Qwen3.5-397B-A17B | ✅ Tested |
| Qwen3-Coder-Next | ✅ Tested |
| Qwen3-Next-80B-A3B-Instruct | ✅ Tested |
| Qwen3-Coder-30B-A3B-Instruct | ✅ Tested |
| Qwen3-VL-30B-A3B-Instruct | ✅ Tested |
| MiniMax-M2.7 | ✅ Tested |
| MiniMax-M2.5 | ✅ Tested |
| MiniMax-M2.1 | ✅ Tested |
| GLM-4.7 | ✅ Tested |
| GLM-4.7-Flash | ✅ Tested |
| GLM-4.6V | ✅ Tested |
| Kimi k2.6 | ✅ Tested |
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
| fp8 model | fp8, fp8+int4 | 
| awq 4bit symmetric quantized model <sup>Note 1</sup> | int4 |

Note 1: https://hf-mirror.com/cyankiwi provides AWQ 4bit symmetric quantized models


## Performance Reference

| Model | Runtime Format | Prefill Speed (tokens/s) | Decode Speed (tokens/s) | CPU | GPU | Memory |
|------|----------|---------------------|-------------------|----------|---------|---------|
| Qwen3-Next-80B-A3B-Instruct Original | bfloat16 |15000 <sup>Note 1</sup> | 90 | Dual EPYC 9555ES | Single Nvidia RTX Pro 6000 | 6400MT/s |
| MiniMax-M2.1 Original | fp8+bfloat16 | 5000 <sup>Note 1</sup> | 29 | Dual EPYC 9684x | Single Nvidia RTX 5090 | 4800MT/s |

Note 1: Enabling GPU Prefill, Input Length 32K-64K

## How to Run Qwen3.6-35B-A3B
```bash
#pip uninstall transformers -y
#pip install transformers==4.57.6

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
VLLM_SKIP_P2P_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
LVLLM_ENABLE_MOE_LAYERWISEISE_LOAD=1 \
vllm serve \
    --model /home/guqiong/Models/Qwen3.6-35B-A3B \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Qwen3.6-35B-A3B \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32000 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

## How to Run gemma-4-26b-a-4b-it
```bash
pip uninstall transformers -y
pip install transformers==5.5.0

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
VLLM_SKIP_P2P_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
vllm serve \
    --model /home/guqiong/Models/gemma-4-26B-A4B-it \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name gemma-4-26B-A4B-it \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32000 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4
```

## How to Run NVIDIA-Nemotron-3-Super-120B-A12B-BF16
```bash
pip uninstall transformers -y
pip install transformers==4.57.6

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
VLLM_SKIP_P2P_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
vllm serve \
    --model /home/guqiong/Models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32000 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

## How to Run Qwen3.5-122B-A10B
```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h

pip uninstall transformers
pip install transformers==4.57.6

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
VLLM_SKIP_P2P_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
LVLLM_ENABLE_MOE_LAYERWISEISE_LOAD=1 \
vllm serve \
    --model /home/guqiong/Models/Qwen3.5-122B-A10B \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Qwen3.5-122B-A10B \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```


## How to Run Qwen3.5-397B-A17B
```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h

pip uninstall transformers -y
pip install transformers==4.57.6

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
VLLM_SKIP_P2P_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
LVLLM_ENABLE_MOE_LAYERWISEISE_LOAD=1 \
vllm serve \
    --model /home/guqiong/Models/Qwen3.5-397B-A17B-FP8 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Qwen3.5-397B-A17B-FP8 \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

# How to Run MiniMax-M2.7

 
```bash

pip uninstall transformers -y
pip install transformers==4.57.6

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
VLLM_SKIP_P2P_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
vllm serve \
    --model /home/guqiong/Models/MiniMax-M2.7 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name MiniMax-M2.7 \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32000 \
    --dtype bfloat16 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think 
```
```bash 
# reduce performance issue when encountering performance problem, you can try binding by NUMA node and reduce thread count
```

```bash 
--enable_expert_parallel # enable expert parallelism, 8-card inference of minimax-m2.1、minimax-m2.5 model requires setting
```

## How to Run Kimi-K2.6

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h

pip uninstall transformers -y
pip install transformers==4.57.6

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
VLLM_SKIP_P2P_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
vllm serve \
    --model /home/guqiong/Models/Kimi-K2.6 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Kimi-K2.6 \
    --compilation_config.cudagraph_mode FULL_AND_PIECEWISE \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32000 \
    --dtype bfloat16 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 

```


## How to Run GLM-4.7-FP8

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h

pip uninstall transformers -y
pip install transformers==5.3.0

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
VLLM_SKIP_P2P_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
vllm serve \
    --model /home/guqiong/Models/GLM-4.7-FP8 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name GLM-4.7-FP8 \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32000 \
    --dtype bfloat16 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 

```

## Configuration Parameters

| Environment Variable | Type | Default Value | Description | Notes |
|--------|------|--------|------|------|
| `LVLLM_MOE_NUMA_ENABLED` | Core Parameter | `0` | Whether to enable hybrid inference: `1`-enabled, `0`-disabled | Set to `0` to disable hybrid inference, behavior is the same as vLLM |
| `LK_THREAD_BINDING` | Performance Parameter | `CPU_CORE` | Thread binding policy: `CPU_CORE` - bind by CPU core, `NUMA_NODE` - bind by NUMA node | Default is binding by CPU core, when encountering performance issues, you can try binding by NUMA node |
| `LK_THREADS` | Performance Parameter | Auto-calculated | Number of threads: physical cores - 4 | For multi-GPU multi-process, (physical cores - 4) divided by number of processes |
| `OMP_NUM_THREADS` | Performance Parameter | System logical core count | OpenMP thread count: set to same as `LK_THREADS` | |
| `LVLLM_MOE_USE_WEIGHT` | Performance Parameter | `INT4` |  FP8 model runtime expert weight format `KEEP`: same as model, `INT4`: int4 |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU Prefill Parameter | None | MOE expert layers resident on GPU `0`: layer 0, `0-1`: layers 0 to 1, `0,9`: layers 0 and 9 | After reserving sufficient KV Cache VRAM, allocating multiple layers can increase performance and reduce corresponding memory usage, including layer 0 to achieve acceleration effect |
| `LVLLM_GPU_PREFETCH_WINDOW` | GPU Prefill Parameter | None | Prefetch window size `1`: prefetch 1 layer of MOE experts | Generally, prefetching 1 to 2 layers is sufficient |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | GPU Prefill Parameter | None | Minimum input length for using GPU prefill `4096`: when input length reaches this value, start GPU prefill | The value should not be too small, set to 0 to disable GPU prefill function |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | Performance Parameter | 0 | `0`：load model quickly, `1`：load model slowly to avoid OOM | Suggested value: use `0` when memory is abundant, use `1` when memory is tight |
| `LVLLM_MOE_QUANT_ON_GPU` | Performance Parameter | 0 | `0`：enable CPU expert quantization, `1`：enable GPU expert quantization | enable if GPU memory is abundant (only effective at loading time, inference will not occupy extra GPU memory)，accelerate model loading speed |



| Parameter | Example Value | Description |
|-----------|-------|-------------|
| `model` | `/home/guqiong/Models/Models/MiniMax-M2.7` | Model directory path |
| `host` | `0.0.0.0` | Service binding IP address |
| `port` | `8070` | Service binding port number |
| `tensor-parallel-size` | `2` | Tensor parallel size, less than or equal to the number of GPUs |
| `pipeline-parallel-size` | `2` (commented) | Pipeline parallel size, less than or equal to the number of GPUs |
| `max-model-len` | `18000` | Maximum context length, less than or equal to the model's maximum length |
| `gpu-memory-utilization` | `0.92` | GPU memory utilization percentage allocated to vLLM, less than or equal to 1 |
| `trust-remote-code` | `true` | Whether to trust remote code, recommended value: `true` |
| `tokenizer-mode` | `auto` | Tokenizer mode, recommended value: `auto` |
| `served-model-name` | `MiniMax-M2.7` | Served model name |
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

### 1. Install CUDA 13.2

```bash
# Uninstall old version CUDA and NVIDIA drivers
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall

# Download and install CUDA 13.2
wget https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_595.45.04_linux.run
sudo sh cuda_13.2.0_595.45.04_linux.run
```

### 2. Create Python Environment

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm
pip install setuptools_scm

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

# Install PyTorch 2.11.0
pip install torchaudio triton torchvision torch==2.11.0

```

### 4. Install Lvllm

```bash
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```

**Parameter Explanation:**
- `MAX_JOBS=32 NVCC_THREADS=1`: Reduce compilation memory usage
- `CMAKE_BUILD_TYPE=Release`: Performance optimization option
- `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"`: Performance optimization option


## Update

If Lvllm is already installed and you need to update to the latest version, execute the following commands:

```bash 
# This command is suitable for regular users; those who want to keep their local modifications should know to handle them in advance
git fetch && git reset --hard origin/main && git clean -fd 

# Install PyTorch 2.11.0
pip uninstall torchaudio triton torchvision torch vllm
pip install torchaudio triton torchvision torch==2.11.0

# Qwen3-VL GLM4.6V requires xformers to be installed

# Compile and install
rm -rf .deps/*build* 
rm -rf build 
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv

rm -rf ~/.cache/vllm
rm -rf ~/.cache/flashinfer
rm -rf ~/.triton/cache
```  
 
## Optimization Tips
 
### MoE Resident in VRAM, Linear Increase in Decode and Prefill Speed
```bash
# 0-5 MoE layers resident in VRAM
# 0,1,8-9 means 0,1,8-9 MoE layers resident in VRAM
# Some models have non-zero starting layer numbers, such as Step-3.5-Flash model starting at 3
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 
``` 
 
### Enable GPU Prefill
```bash
# Prefetch 1 layer, recommended value is 1, more is meaningless
LVLLM_GPU_PREFETCH_WINDOW=1 
# Start GPU prefill when input length reaches 4096, can be decreased or increased based on CPU prefill performance, starting prefill earlier or later
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 
# exceeding context size is meaningless
--max-num-batched-tokens 32000 
``` 
 
### Disable GPU Prefill
```bash
# Disable GPU prefill
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 
# 1024 to 8192, too large is meaningless (occupies too much VRAM and long startup time)
--max-num-batched-tokens 4096
``` 
 
### Thread Binding to CPU Cores
```bash
# Bind to CPU cores (including hyper-threading logical cores), optimal performance
LK_THREAD_BINDING=CPU_CORE 
# Bind to NUMA nodes, second choice to solve extreme performance issues on virtualization platforms
LK_THREAD_BINDING=NUMA_NODE 
``` 
### BIOS NUMA Settings
```bash
AMD EPYC: Set NPS4 for optimal performance
Intel XEON: Set SNC4 for optimal performance
# Some virtualization platforms or Intel platforms should not set 5 or 10 nodes, set to 2 nodes to avoid performance issues
General: 2,4,8 nodes, maximum support for 32 nodes, more nodes are better, node count being a multiple of GPUs for optimal performance 
```
 
### Thread Count Settings
```bash
# Thread count <= (core count - x) / tensor parallelism size (TP size)  # x threads reserved for other tasks, at least 4 threads
# 96 cores, 2 GPUs, 44 threads per GPU, 88 threads total, 8 threads reserved for other tasks 
LK_THREADS=44                   
# if the total number of threads exceeds the physical core count, it may cause performance issues   
# although the system will automatically adjust the number of threads, it is recommended to manually set it for testing     

```
### Decode Performance
```bash
 # Supports 2080ti and above GPUs
--compilation_config.mode VLLM_COMPILE 
 # Enable CUDAGraph               
--compilation_config.cudagraph_mode FULL_DECODE_ONLY  
```
 
### VRAM Settings
```bash
# 24G VRAM with GPU prefill enabled, leave sufficient temporary VRAM for calculations, otherwise long context prefill performance will drop significantly, startup time will be too long
--gpu-memory-utilization 0.9046 
# Maximum 4 concurrent, regular VRAM savings
--max-num-seqs 2 
# Save VRAM when GPU prefill is disabled, performance remains unchanged, but if enable GPU prefill will cause performance drop
--max-num-batched-tokens 4096
# or larger and less than context size, enable GPU prefill, obtain best performance, but if disable GPU prefill will cause performance drop
--max-num-batched-tokens 32000 
```
### CPU Power Saving
```bash
# enable low power mode while inference, reduce CPU temperature, slightly reduce performance 
LK_POWER_SAVING=1 
```

### FP8 Model Weight Runtime Format
```bash
 # Model MoE expert weights use INT4 inference, other parts remain FP8, enabling almost no impact on accuracy, speed order: INT4 > TO_DTYPE > KEEP
LVLLM_MOE_USE_WEIGHT=INT4
```

### Model Loading with NUMA Interleaving
```bash
# Slow model loading can prevent OOM. Recommended values: use `0` when memory is sufficient during model file loading, use `1` when memory is limited.
LVLLM_ENABLE_NUMA_INTERLEAVE=1 
# FP8 MoE model layerwise loading, reduce peak memory usage
LVLLM_ENABLE_MOE_LAYERWISEISE_LOAD=1
```

### Model Loading with GPU Expert Quantization
```bash
# Enable GPU expert quantization, recommended values: enable when sufficient GPU memory is available (only effective during model loading, not during inference), speed up model loading
LVLLM_MOE_QUANT_ON_GPU=1 
```

### CPU Prefill Optimization
```bash
# It is allowed to increase the CPU prefill speed by increasing the max_num_batched_tokens parameter, for example --max-num-batched-tokens 4096. If GPU prefill is enabled, the smaller value between LVLLM_GPU_PREFILL_MIN_BATCH_SIZE and max_num_batched_tokens will be used.
--max-num-batched-tokens 4096
```
