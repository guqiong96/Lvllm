# LvLLM GPU、NUMA 双并行 [[English]](./README.md)

LvLLM是vllm的特别扩展，充分利用CPU和GPU计算资源，高效的GPU并行+NUMA并行架构，适用于MOE模型混合推理。

## 系统特性

- **GPU + NUMA 双并行**: 支持CPU-GPU混合解码、CPU-GPU混合预填充、GPU预填充三种计算方式
- **显存 + 内存负载均衡**: 模型总体占用=显存+内存，容纳模型1+1=2, 100%显存利用率 <sup>注1</sup>
- **GPU 预填充优化**: GPU预填充与CPU-GPU混合解码并行，接近100%显卡利用率
- **NUMA 线程优化**: 跨节点通信占比低至3%，三级缓存命中50%以上，解码阶段可推动GPU负载达到33%至50% 
 
## 与vLLM的关系

Lvllm使用最新的vLLM源码，重新设计实现了MOE模型混合推理模块，保持了对vLLM的100%完全兼容<sup>注1</sup>。

注1：x86带有AVX2以上指令集的CPU和Nvidia GPU

## 使用说明 [[English]](./README.md)
- [版本变更](#版本变更)
- [支持的模型](#支持的模型)
- [性能参考](#性能参考)
- [如何运行gemma-4-26B-A4B-it](#如何运行gemma-4-26b-a-4b-it)
- [如何运行NVIDIA-Nemotron-3-Super-120B-A12B-BF16](#如何运行nvidia-nemotron-3-super-120b-a12b-bf16)
- [如何运行Qwen3.5-122B-A10B](#如何运行qwen35-122b-a10b)
- [如何运行Qwen3.5-397B-A17B](#如何运行qwen35-397b-a17b)
- [如何运行MiniMax-M2.7](#如何运行minimax-m27)
- [如何运行Kimi-K2.5](#如何运行kimi-k25)
- [如何运行GLM-4.7-FP8](#如何运行glm-47-fp8)
- [配置参数](#配置参数)
- [安装步骤](#安装步骤) 
- [更新](#更新)
- [优化](#优化)

## 版本变更
 
```bash
2026-04-06: lvllm-v2.1.0 - 增强使用LK_POWER_SAVING=1节能效果，支持FP8+BF16+AWQ4bit的混合MOE层推理
2026-03-22: lvllm-v2.0.0 - FP8 MoE模型使用INT4专家量化时支持逐层加载，减少峰值内存占用，LVLLM_ENABLE_MOE_LAYERWISEISE_LOAD=1
2026-03-19: lvllm-v1.9.10 - 修复已知问题，支持新的moe模型类型[没有gate_proj], 例如：NVIDIA-Nemotron-3-Super-120B-A12B-BF16
2026-03-11: lvllm-v1.9.2 - FP8、AWQ4bit模型开启GPU Prefill加速不再占用额外内存, FP8模型取消TO_DTYPE运行时类型转换、KEEP暂不支持开启GPU Prefill
2026-03-05: lvllm-v1.9.0 - 优化GPU预填充和常规预填充，确保输出质量
2026-03-01: lvllm-v1.8.10 - 修复已知问题，增加新模型支持
2026-02-02：lvllm-v1.7.0 - 支持EP并行，8卡运行minimax-m2.1模型需要设置--enable_expert_parallel
2026-01-26: lvllm-v1.6.1 - fp8 模型支持 FP8 + INT4 推理，支持GPU Prefill加速(内存占用很高!) 
2026-01-25: lvllm-v1.6.0 - fp8 模型支持 GPU Prefill加速(内存占用很高!)
2026-01-24: lvllm-v1.5.8 - AWQ 4-bit 对称量化模型支持 GPU Prefill加速
2026-01-21: lvllm-v1.5.7 - 修复MiniMax-M2.1模型数值计算稳定问题
2026-01-08: lvllm-v1.5.1 - 针对长上下文场景，支持预填充与解码分离，GPU预填充与CPU-GPU混合解码并行
2026-01-04: v1.4.0 优化decode提升速度
2025-12-28：优化推理速度：bfloat16、awq4bit；优化多GPU的NUMA数据访问；为多GPU启用NUMA节点以实现最佳性能; 取消GGUF模型支持 
2025-12-16 v1.2.0 同步上游vllm代码至最新，lk_moe优化降低内存占用
2025-12-14 v1.1.2 增加AWQ-4bit对称量化模型推理支持
2025-12-9: 增加LVLLM_MOE_USE_WEIGHT环境变量，支持MOE模块使用两种模式推理fp8模型：
2025-11-1： 支持张量并行、流水线多卡推理 https://b23.tv/xzHieMs
2025-10-30: 支持Qwen3系列模型GGUF混合推理（不包含Qwen3-Coder-30B-A3B-Instruct GGUF） [查看config.yaml里面的新参数]
2025-10-19: FP8支持GPU+NUMA 混合推理MOE模型！！ [显存FP8精度，内存FP16精度] 已验证GLM-4.5-Air-FP8
2025-10-14: 开启cuda graph , decode 速度翻倍！！ 输出质量提高！！
2025-09-30 已验证：Qwen3-Next-80B-A3B-Instruct、Qwen3-Coder-30B-A3B-Instruct 
 
```

## 支持的模型

vLLM已验证的大部分原版MOE模型
 
| 模型名称 | 状态 |
|---------|------|
| gemma-4-26B-A4B-it | ✅ 已测试通过 |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | ✅ 已测试通过 |
| Qwen3.5-35B-A3B | ✅ 已测试通过 |
| Qwen3.5-122B-A10B | ✅ 已测试通过 |
| Qwen3.5-397B-A17B | ✅ 已测试通过 |
| Qwen3-Coder-Next | ✅ 已测试通过 |
| Qwen3-Next-80B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-Coder-30B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-VL-30B-A3B-Instruct | ✅ 已测试通过 | 
| MiniMax-M2.7 | ✅ 已测试通过 |
| MiniMax-M2.5 | ✅ 已测试通过 |
| MiniMax-M2.1 | ✅ 已测试通过 |
| GLM-4.7 | ✅ 已测试通过 |
| GLM-4.7-Flash  | ✅ 已测试通过 |
| GLM-4.6V | ✅ 已测试通过 |
| Kimi k2.5 | ✅ 已测试通过 |

未列出的Qwen3系列、GLM系列、MiniMax系列的原版MOE模型理论上支持，待实际测试。


## 尚未支持的模型

| 模型名称 | 状态 |
|---------|------|
| DeepSeek-V3.2| 待定 |
 

## 支持的模型权重格式及运行时格式

| 模型文件 | 运行时格式 | 
|---------|------------|
| bfloat16 | bfloat16/float16| 
| float16 | bfloat16/float16| 
| fp8模型 | fp8、fp8+int4 | 
| awq 4bit对称量化模型 <sup>注1</sup>| int4 | 

注1：https://hf-mirror.com/cyankiwi 提供AWQ 4bit对称量化模型

## 性能参考

| 模型 | 运行时格式 | 预填充速度(tokens/s) | 解码速度(tokens/s) | CPU | GPU |内存 |
|------|----------|---------------------|-------------------|----------|---------|---------|
| Qwen3-Next-80B-A3B-Instruct原版 | bfloat16 |15000 <sup>注1</sup> | 90 | 双路 EPYC 9555ES  | 单卡 Nvidia RTX Pro 6000 | 6400MT/s  |
| MiniMax-M2.1原版 | fp8+bfloat16 | 5000 <sup>注1</sup> | 29 | 双路 EPYC 9684x  | 单卡 Nvidia RTX 5090 | 4800MT/s  |

注1：开启GPU预填充，输入长度32K-64K

## 如何运行gemma-4-26b-a-4b-it
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
    --max-model-len 160000 \
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



## 如何运行NVIDIA-Nemotron-3-Super-120B-A12B-BF16
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
    --max-model-len 52000 \
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


## 如何运行Qwen3.5-122B-A10B
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
    --model /home/guqiong/Models/Qwen3.5-122B-A10B \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len 40000 \
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


## 如何运行Qwen3.5-397B-A17B
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
    --max-model-len 70000 \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Qwen3.5-397B-A17B-FP8 \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 18000 \
    --max-num-seqs 2 \
    --compilation_config.mode VLLM_COMPILE \
    --kv-sharing-fast-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```


## 如何运行MiniMax-M2.7

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
    --max-model-len 50000 \
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
# 遇到性能问题时可尝试按NUMA节点绑定线程, 并减少线程数量
```

```bash 
# 启用EP并行, 8卡运行minimax-m2.1、minimax-m2.5模型需设置
--enable_expert_parallel 
```

## 如何运行Kimi-K2.5

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
    --model /home/guqiong/Models/Kimi-K2.5 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len 33000 \
    --gpu-memory-utilization 0.9046 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Kimi-K2.5 \
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


## 如何运行GLM-4.7-FP8

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
    --max-model-len 40000 \
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


## 配置参数

| 环境变量 | 类型 | 默认值 | 说明 | 备注 |
|--------|------|--------|------|------|
| `LVLLM_MOE_NUMA_ENABLED` | 核心参数 | `0` | 是否启用混合推理: `1`-启用，`0`-禁用 | 设置为`0`禁用混合推理，行为与vLLM相同 |
| `LK_THREAD_BINDING` | 性能参数 | `CPU_CORE` | 线程绑定策略: `CPU_CORE`-按CPU核心绑定，`NUMA_NODE`-按NUMA节点绑定 | 默认按CPU核心绑定, 遇到性能问题时可尝试按NUMA节点绑定 |
| `LK_THREADS` | 性能参数 | 自动计算 | 线程数量: 物理核心数-4 | 多GPU多进程时，物理核心数-4除以进程数量 |
| `OMP_NUM_THREADS` | 性能参数 | 系统逻辑核心数量 | OpenMP线程数: 设置为`LK_THREADS`相同 |   | 
| `LVLLM_MOE_USE_WEIGHT` | 性能参数 | `INT4` | FP8模型运行时专家权重格式 `KEEP`: 与模型一致，`INT4`: int4  |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU预填充参数 | 无 | 常驻GPU的MOE专家层`0`: 第0层，`0-1`: 第0层到第1层，`0,9`: 第0层和第9层 | 留足KV Cache显存后，分配多层可增加性能，并减少对应的内存占用，包含0层才有加速效果 |
| `LVLLM_GPU_PREFETCH_WINDOW` | GPU预填充参数 | 无 | 预取窗口大小`1`: 预取1层MOE专家 |  一般预取1到2层即可 |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | GPU预填充参数 | 无 | 使用GPU预填充的最小输入长度`4096`：输入长度达到该值后，启动GPU预填充 | 设置值不宜过小，设置为0则关闭GPU预填充功能 |
| `LK_POWER_SAVING` | cpu节能 | 0 | `1`：启用cpu节能模式，`0`：禁用cpu节能模式 | 建议值：`0` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | 性能参数 | 0 | `0`：快速加载模型，`1`：慢速加载模型可避免OOM | 建议值：加载模型文件时，内存充裕使用`0`，内存紧张使用`1` |
| `LVLLM_MOE_QUANT_ON_GPU` | 性能参数 | 0 | `0`：不启用GPU专家量化，`1`：启用GPU专家量化 | 显存充足可启用（仅加载时有效，推理时不会额外占用显存），加快模型加载速度 |

 
| 参数 | 示例值 | 说明 |
|-----------|-------|-------------|
| `model` | `/home/guqiong/Models/Models/MiniMax-M2.5` | 模型目录路径 |
| `host` | `0.0.0.0` | 服务绑定IP地址 |
| `port` | `8070` | 服务绑定端口号 |
| `tensor-parallel-size` | `2` | 张量并行大小，小于等于GPU数量 |
| `pipeline-parallel-size` | `2` (commented) | 流水线并行大小，小于等于GPU数量 |
| `max-model-len` | `18000` | 最大上下文长度，小于等于模型最大长度 |
| `gpu-memory-utilization` | `0.92` | 分配给vLLM的GPU显存分配百分比，小于等于1 |
| `trust-remote-code` | `true` | 是否信任远程代码，建议值 |
| `tokenizer-mode` | `auto` | 分词器模式，建议值 |
| `served-model-name` | `Models/MiniMax-M2.5` | 服务模型名称 |
| `compilation_config.cudagraph_mode` | `FULL_DECODE_ONLY` | 启用CUDA图模式，建议值 |
| `enable_prefix_caching` | `true` | 启用前缀缓存，建议值 |
| `enable-chunked-prefill` | `true` | 启用分块预填充，建议值 |
| `max_num_batched_tokens` | `18000` | 最大批量填充令牌数，关闭GPU预填充时建议值：1024，开启GPU预填充时建议值：同max-model-len |
| `dtype` | `bfloat16` | 模型中间计算数据类型，建议值bfloat16或float16 |
| `max_num_seqs` | `4` | 最大并发请求序列，建议值1到4 |
| `compilation_config.mode` | `VLLM_COMPILE` | 优化模型，建议值 |
 
 
| 参数 | 说明 |
|-----------|-------------|
| `kv_cache_dtype: "fp8"` | KV Cache数据类型，40系、50系GPU可开启 (commented) |
| `speculative-config` | 推测解码配置，建议值关闭 (commented) |
| `mm-encoder-tp-mode: "data"` | encoder TP模式 (commented) |
 

## 安装步骤

### 1. 安装CUDA 13.2

```bash
# 卸载旧版本CUDA和NVIDIA驱动
sudo /usr/local/cuda/bin/cuda-uninstaller   
sudo nvidia-uninstall

# 下载并安装CUDA 13.2
wget https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_595.45.04_linux.run
sudo sh cuda_13.2.0_595.45.04_linux.run
```

### 2. 创建Python环境

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm

# 升级libstdcxx-ng（避免glibcxx版本问题）
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 安装NUMA库
sudo apt-get install libnuma-dev      # Ubuntu
sudo dnf install numactl-devel        # Rocky Linux
```

### 3. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/guqiong96/Lvllm.git
cd Lvllm

# 安装PyTorch 2.10.0
pip install torchaudio triton torchvision torch==2.10.0

# 安装构建依赖
pip install -r requirements/build.txt
```
 
### 4. 安装Lvllm

```bash 
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```

**参数说明：**
- `MAX_JOBS=32 NVCC_THREADS=1`: 减少编译内存占用
- `CMAKE_BUILD_TYPE=Release`: 性能优化选项
- `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release`: 性能优化选项
 
 
 
## 更新

如果已安装Lvllm，需要更新到最新版本，请执行以下命令：

```bash 
# 此命令适合普通用户，如果保留本地修改内容的用户应知道提前做处理
git fetch && git reset --hard origin/main && git clean -fd 

# 安装PyTorch 2.10.0
pip uninstall torchaudio triton torchvision torch vllm
pip install torchaudio triton torchvision  torch==2.10.0

# Qwen3-VL GLM4.6V 需要安装 xformers
  
# 编译安装
pip install -r requirements/build.txt
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv

rm -rf ~/.cache/vllm
rm -rf ~/.cache/flashinfer
rm -rf ~/.triton/cache
```
 
## 优化

### MoE常驻显存, 线性增加decode和prefill速度
```bash
# 0-5层MoE层常驻显存
# 格式 0,1,8-9 表示 0,1,8-9层MoE层常驻显存
# 少数模型起始层号不为0，例如Step-3.5-Flash模型起始为3 
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 
``` 

### 开启GPU预填充
```bash
# 预取1层, 建议值为1, 多了无意义
LVLLM_GPU_PREFETCH_WINDOW=1
# 输入长度达到4096启动GPU prefill，根据cpu prefill性能可减小或加大， 提前或推后启动prefill
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 
# 超过上下文大小无意义
--max-num-batched-tokens 32000 
``` 

### 关闭GPU预填充
```bash
#  关闭GPU预填充
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0
# 1024至8192，太大无意义（占用显存及启动时间过长）
--max-num-batched-tokens 4096 
``` 

### 线程绑定到CPU核心
```bash
# 绑定到CPU核心（包括超线程逻辑核心）, 最佳性能
LK_THREAD_BINDING=CPU_CORE 
# 绑定到NUMA节点, 次优选择，解决部署在虚拟化平台的极端性能问题，以及多实例运行
LK_THREAD_BINDING=NUMA_NODE 
``` 
### BIOS NUMA 设置
```bash
AMD EPYC：设置NPS4获得最佳性能
Intel XEON：设置SNC4获得最佳性能
# 部分虚拟化平台或Intel平台不要设置5、10节点，设置2节点避免性能问题
通常：2,4,8个节点，最多支持32节点，节点越多越好，节点数为GPU倍数获得最佳性能 
```

### 线程数设置
```bash
# 线程数 <= （核心数 - x）/ 张量并行数（TP size） x 留给其它任务的线程，至少4线程
# 96核心，2个GPU， 每个GPU 44线程， 88线程, 剩余8线程留给其它任务
LK_THREADS=44                    
# 总的线程数超过物理核心数量可能会引发性能问题   
# 虽然系统会自动条件线程数，但建议手动设置进行测试     
```
### 输出性能
```bash
# 支持2080ti及以上GPU
--compilation_config.mode VLLM_COMPILE  
 # 开启CUDAGraph               
--compilation_config.cudagraph_mode FULL_DECODE_ONLY  
```

### 显存设置
```bash
# 24G显存开启GPU预填充时，留出足够临时显存用于计算，否则会导致长上下文预填充性能大幅下降，启动时间过长
--gpu-memory-utilization 0.9046
# 最多4并发，常规节省显存
--max-num-seqs 2
# 关闭GPU预填充时,节省显存，性能不变，但如果开启GPU预填充会导致性能下降
--max-num-batched-tokens 4096  
# 开启GPU预填充时，4096~32000，GPU预填充加速情况、显存大小调节，超过上下文大小无意义
--max-num-batched-tokens 32000
```
### CPU节能
```bash
# 开启后推理时降低CPU温度，性能轻微降低
LK_POWER_SAVING=1 
```

### FP8模型权重运行时格式
```bash
# FP8模型MoE专家权重使用INT4推理，其余部分依旧为FP8，开启几乎不影响精度， 速度排序：INT4 > KEEP
LVLLM_MOE_USE_WEIGHT=INT4 
```

### 模型加载
```bash
# 慢速加载模型可避免OOM，建议值：加载模型文件时，内存充裕使用`0`，内存紧张使用`1`
LVLLM_ENABLE_NUMA_INTERLEAVE=1 
# FP8 MoE模型逐层加载，减少峰值内存占用
LVLLM_ENABLE_MOE_LAYERWISEISE_LOAD=1
```

### 模型加载专家量化
```bash
# 启用GPU专家量化，建议值：显存充足可启用（仅加载时有效，推理时不会额外占用显存），加快模型加载速度
LVLLM_MOE_QUANT_ON_GPU=1 
```

### CPU预填充优化
```bash
# 允许通过加大max_num_batched_tokens参数来提高cpu prefill速度，例如--max-num-batched-tokens 4096，如果开启了gpu prefill则取LVLLM_GPU_PREFILL_MIN_BATCH_SIZE、max_num_batched_tokens两者最小值
--max-num-batched-tokens 4096
```





