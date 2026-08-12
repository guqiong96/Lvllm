# LvLLM GPU、NUMA 双并行 [[English]](./README.md)

LvLLM是vllm的特别扩展，充分利用CPU和GPU计算资源，高效的GPU并行+NUMA并行架构，适用于MOE模型混合推理。

> **核心引擎：** 实际的混合推理功能——包括 CPU-GPU 协同计算、NUMA 感知调度、专家权重管理以及量化内核执行——完全由高度优化的 MOE 混合推理引擎 **[lk_moe](https://pypi.org/project/lk-moe/)** 提供。在 LvLLM（用于 vLLM）和 [Lsglang](https://github.com/guqiong96/Lsglang)（用于 sglang）中，每个 MOE 层可灵活选择原始 GPU 计算路径或调用 lk_moe 进行混合推理。针对 DeepSeek V4，还提供了特化版本 [Lvllmds4](https://github.com/guqiong96/Lvllmds4)（SM120+）和 [Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x)（SM80+）。

## 系统特性

- **GPU + NUMA 双并行**: 支持CPU-GPU混合解码、CPU-GPU混合预填充、GPU预填充三种计算方式
- **显存 + 内存负载均衡**: 模型总体占用=显存+内存，容纳模型1+1=2, 100%显存利用率 <sup>注1</sup>
- **GPU 预填充优化**: GPU预填充与CPU-GPU混合解码并行，接近100%显卡利用率
- **NUMA 线程优化**: 跨节点通信占比低至3%，三级缓存命中50%以上，解码阶段可推动GPU负载达到33%至50% 
 
## 与vLLM的关系

Lvllm使用最新的vLLM源码，重新设计实现了MOE模型混合推理模块，保持了对vLLM的100%完全兼容<sup>注1</sup>。

注1：x86带有AVX2以上指令集的CPU和Nvidia GPU sm75以上架构

## 使用说明 [[English]](./README.md)
- [性能基准](#性能基准)
- [版本变更](#版本变更)
- [支持的模型](#支持的模型)
- [支持的量化格式](#支持的量化格式)
- [运行命令参考](#运行命令参考)
- [配置参数](#配置参数)
- [安装步骤](#安装步骤) 
- [优化](#优化)

## 性能基准
Open GPU Prefill, max_num_batched_tokens=8192 (Row 1), max_num_batched_tokens=32768 (Row 2)
| Model | Version | CPU | Memory | GPU | Prefill | Decode | Speculative Decoding |
|-------|---------|-----|--------|-----|---------|--------|---------|
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllm-v2.3.10 | EPYC 7642 *2 | 16 channels ddr4 3200 | 5060Ti * 2 | 850 t/s [input 32768]| 28 t/s [input 32768]| 30~46 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllmds4-x-v2.3.9 | EPYC 7642 *2 | 16 channels ddr4 3200 | 3090 * 2 | 1060 t/s [input 32768]| 26 t/s [input 32768]| 35~47 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllmds4-v2.3.9 | EPYC 9684x *2 | 24 channels ddr5 4800 | pro 6000 * 1 | 3100 t/s [input 131072]| 75 t/s [input 131072]| 100~115 t/s |

## 版本变更
 
```bash
2026-07-17: lvllm-v2.3.6 - 新增 dtype float16 支持 for SM75 GPU Prefill
2026-07-08: lvllm-v2.3.2 - 新增 ModelOpt W4A16 NVFP4 量化类型支持
2026-07-05: lvllm-v2.3.0 - 优化GPU预填充速度，CPU AVX512优化，取消LVLLM_GPU_RESIDENT_MOE_EXPERTS
2026-06-05: lvllm-v2.2.0 - 升级lk_moe模块, 新增支持nvfp4, mxfp4量化类型，增加LVLLM_GPU_RESIDENT_MOE_EXPERTS, 取消LVLLM_MOE_USE_WEIGHT、LVLLM_MOE_QUANT_ON_GPU
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
| Ornith-1.0-35B-FP8 | ✅ 已测试通过 |
| Qwen3.6-35B-A3B | ✅ 已测试通过 |
| Qwen3.5-35B-A3B | ✅ 已测试通过 |
| Qwen3.5-122B-A10B | ✅ 已测试通过 |
| Qwen3.5-397B-A17B | ✅ 已测试通过 |
| Qwen3-Coder-Next | ✅ 已测试通过 |
| Qwen3-Next-80B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-Coder-30B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-VL-30B-A3B-Instruct | ✅ 已测试通过 | 
| MiniMax-M3 | ✅ 已测试通过 |
| MiniMax-M2.7 | ✅ 已测试通过 |
| MiniMax-M2.5 | ✅ 已测试通过 |
| MiniMax-M2.1 | ✅ 已测试通过 |
| GLM-5.2 nvfp4| ✅ 已测试通过[sm120] |
| GLM-4.7 | ✅ 已测试通过 |
| GLM-4.7-Flash  | ✅ 已测试通过 |
| GLM-4.6V | ✅ 已测试通过 |
| Kimi k2.6 | ✅ 已测试通过 |
| Kimi k2.5 | ✅ 已测试通过 |
| deepseek-ai/DeepSeek-V4-Flash-0731 | ✅ 已测试通过 [sm120]|

未列出的Qwen3系列、GLM系列、MiniMax系列的原版MOE模型理论上支持，待实际测试。



## 支持的量化格式

| 模型文件 | 运行时格式 | 
|---------|------------|
| bfloat16 | bfloat16/float16| 
| float16 | bfloat16/float16| 
| fp8模型 | fp8 | 
| nvfp4模型 | NVFP4 and ModelOpt W4A16 NVFP4 | 
| mxfp4模型 <sup>注1</sup>| mxfp4 | 
| awq 4bit对称量化模型 <sup>注1</sup>| w4a16 | 
```bash
注1：https://hf-mirror.com/cyankiwi 提供AWQ 4bit对称量化模型
注2：DeepSeek V4 SM80、SM86、SM89需使用专用版本：
https://github.com/guqiong96/Lvllmds4-x/releases SM80+

另外有一个SM120版本：
https://github.com/guqiong96/Lvllmds4/releases SM8120

```

 

## 运行命令参考


```bash
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=48 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-1,33-34 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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

 

## 配置参数

| 环境变量 | 类型 | 默认值 | 说明 | 备注 |
|--------|------|--------|------|------|
| `LVLLM_MOE_NUMA_ENABLED` | 核心参数 | `0` | 是否启用混合推理: `1`-启用，`0`-禁用 | 设置为`0`禁用混合推理，行为与vLLM相同 |
| `LK_THREAD_BINDING` | 性能参数 | `CPU_CORE` | 线程绑定策略: `CPU_CORE`-按CPU核心绑定，`NUMA_NODE`-按NUMA节点绑定 | 默认按CPU核心绑定, 遇到性能问题时可尝试按NUMA节点绑定 |
| `LK_THREADS` | 性能参数 | - | 线程数量:（总物理核心数） ÷ 显卡数量 | 未开启超线程：（总物理核心数-2） ÷ 显卡数量 |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU预填充参数 | 无 | 常驻GPU的MOE专家层`0`: 第0层，`0-1`: 第0层到第1层，`0,9`: 第0层和第9层 | 留足KV Cache显存后，分配多层可增加性能，并减少对应的内存占用 |
| `LVLLM_GPU_PREFETCH_WINDOW` | GPU预填充参数 | 无 | 预取窗口大小`1`: 预取1层MOE专家 |  一般预取1到2层即可 |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | GPU预填充参数 | 无 | 使用GPU预填充的最小输入长度`4096`：输入长度达到该值后，启动GPU预填充 | 设置值不宜过小，设置为0则关闭GPU预填充功能 |
| `LK_POWER_SAVING` | cpu节能 | 0 | `1`：启用cpu节能模式，`0`：禁用cpu节能模式 | 建议值：`0` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | 性能参数 | 0 | `0`：快速加载模型，`1`：慢速加载模型可避免OOM | 建议值：加载模型文件时，内存充裕使用`0`，内存紧张使用`1` |

 
| 参数 | 示例值 | 说明 |
|-----------|-------|-------------|  
| `tensor-parallel-size` | `2` | 张量并行大小，小于等于GPU数量 | 
| `compilation_config.cudagraph_mode` | `FULL_DECODE_ONLY` | 启用CUDA图模式，建议值 |
| `enable_prefix_caching` | `true` | 启用前缀缓存，建议值 |
| `enable-chunked-prefill` | `true` | 启用分块预填充，建议值 |
| `max_num_batched_tokens` | `18000` | 最大批量填充令牌数，关闭GPU预填充时建议值：1024，开启GPU预填充时建议值：32000 |
| `compilation_config.mode` | `VLLM_COMPILE` | 优化模型，建议值 |
 


## 安装步骤

### 1. 安装CUDA 13.2.1

```bash
# 卸载旧版本CUDA和NVIDIA驱动
sudo /usr/local/cuda/bin/cuda-uninstaller   
sudo nvidia-uninstall

# 下载并安装CUDA 13.2.1
wget https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_595.58.03_linux.run
sudo sh cuda_13.2.1_595.58.03_linux.run
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

### 3. 安装Lvllm

```bash 
pip install https://github.com/guqiong96/Lvllm/releases/download/lvllm-v2.3.10/lvllm-2.3.10-cp312-cp312-manylinux_2_34_x86_64.whl

# 最新版本查看https://github.com/guqiong96/Lvllm/releases

pip install https://github.com/guqiong96/Lvllm/releases/download/lvllm-v2.3.10/flashinfer_cubin-0.6.16.post3-py3-none-any.whl
```

## 从源码编译安装Lvllm

```bash 
git clone https://github.com/guqiong96/Lvllm.git
cd Lvllm
pip install setuptools_scm setuptools_rust
pip install torchaudio triton torchvision torch==2.13.0
VLLM_VERSION_OVERRIDE="2.3.10" CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
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
LVLLM_GPU_PREFETCH_WINDOW=1
# 输入长度达到4096启动GPU prefill
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 
# 配合设置最大批处理大小
--max-num-batched-tokens 32000 
``` 

### 关闭GPU预填充
```bash
#  关闭GPU预填充
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0
# 配合设置最大批处理大小
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
# 有超线程：总物理核心数 ÷ 显卡数量， 关闭超线程: （总物理核心数-2） ÷ 显卡数量
# 96核心，2个GPU， 每个GPU 48线程
LK_THREADS=48                    
# 总的线程数超过物理核心数量可能会引发性能问题
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

# 最大批处理大小占用显存量很大，根据情况调整
--max-num-batched-tokens 32000
```
### CPU节能
```bash
# 开启后推理时降低空闲时功耗
LK_POWER_SAVING=1 
``` 