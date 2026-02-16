# LvLLM GPU、NUMA 双并行 [[English]](./README.md)

LvLLM是vllm的特别扩展，充分利用CPU和GPU计算资源，高效的GPU并行+NUMA并行架构，适用于MOE模型混合推理。

## 系统特性

- **GPU + NUMA 双并行**: 支持CPU-GPU混合解码、CPU-GPU混合预填充、GPU预填充三种计算方式
- **显存 + 内存负载均衡**: 模型总体占用=显存+内存，容纳模型1+1=2, 100%显存利用率 <sup>注1</sup>
- **GPU 预填充优化**: GPU预填充与CPU-GPU混合解码并行，接近100%显卡利用率
- **NUMA 线程优化**: 跨节点通信占比低至3%，三级缓存命中50%以上，解码阶段可推动GPU负载达到33%至50% 
 
注1：除BF16、F16原版模型以外开启GPU预填充将额外占用内存[!非显存]

## 与vLLM的关系

Lvllm使用最新的vLLM源码，重新设计实现了MOE模型混合推理模块，保持了对vLLM的100%完全兼容。

## 使用说明 [[English]](./README.md)
- [版本变更](#版本变更)
- [支持的模型](#支持的模型)
- [性能参考](#性能参考)
- [运行命令](#运行命令)
- [配置文件](#配置文件)
- [安装步骤](#安装步骤) 
- [更新](#更新)
- [优化](#优化)

## 版本变更
 
```bash  
2026-02-02：lvllm-v1.7.0 -  支持EP并行，8卡运行minimax-m2.1模型需要设置--enable_expert_parallel
2026-01-26: lvllm-v1.6.1 - fp8 模型支持 FP8 + INT4 推理，支持GPU Prefill加速(内存占用很高!) 
2026-01-25: lvllm-v1.6.0 - fp8 模型支持 GPU Prefill加速(内存占用很高!)
2026-01-24: lvllm-v1.5.8 - AWQ 4-bit 对称量化模型支持 GPU Prefill加速
2026-01-21: lvllm-v1.5.7 - 修复MiniMax-M2.1模型数值计算稳定问题
2026-01-08: lvllm-v1.5.1 - 针对长上下文场景，支持预填充与解码分离，GPU预填充与CPU-GPU混合解码并行
2026-01-04: v1.4.0 优化decode提升速度
2025-12-28：优化推理速度：bfloat16、awq4bit；优化多GPU的NUMA数据访问；为多GPU启用NUMA节点以实现最佳性能; 取消GGUF模型支持 
2025-12-16 v1.2.0 同步上游vllm代码至最新，lk_moe优化降低内存占用
2025-12-14 v1.1.2 增加AWQ-4bit量化模型（对称量化 avx2版本）推理支持 -，验证通过 cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit and cpatonn/
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
| Qwen3-Coder-Next | ✅ 已测试通过 |
| Qwen3-Next-80B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-Coder-30B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-VL-30B-A3B-Instruct | ✅ 已测试通过 | 
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
| fp8模型 | fp8、fp8+bfloat16、fp8+int4 | 
| awq 4bit对称量化模型 <sup>注1</sup>| int4 | 

注1：https://hf-mirror.com/cyankiwi 提供AWQ 4bit对称量化模型

## 性能参考

| 模型 | 运行时格式 | 预填充速度(tokens/s) | 解码速度(tokens/s) | CPU | GPU |内存 |
|------|----------|---------------------|-------------------|----------|---------|---------|
| Qwen3-Next-80B-A3B-Instruct原版 | bfloat16 |15000 <sup>注1</sup> | 90 | 双路 EPYC 9555ES  | 单卡 Nvidia RTX Pro 6000 | 6400MT/s  |
| MiniMax-M2.1原版 | fp8+bfloat16 | 5000 <sup>注1</sup> | 29 | 双路 EPYC 9684x  | 单卡 Nvidia RTX 5090 | 4800MT/s  |

注1：开启GPU预填充，输入长度32K-64K

## 运行命令
 
```bash 
# 未启用GPU预填充
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=88 OMP_NUM_THREADS=88 LVLLM_MOE_USE_WEIGHT=INT4 vllm serve --config config.yaml
```

```bash 
# 启用GPU预填充
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=88 OMP_NUM_THREADS=88 LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-1 LVLLM_GPU_PREFETCH_WINDOW=1 LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 vllm serve --config config.yaml
```
```bash 
# 遇到性能问题时可尝试按NUMA节点绑定线程, 并减少线程数量
```

```bash 
--enable_expert_parallel # 启用EP并行, 8卡运行minimax-m2.1模型需设置
```

| 环境变量 | 类型 | 默认值 | 说明 | 备注 |
|--------|------|--------|------|------|
| `LVLLM_MOE_NUMA_ENABLED` | 核心参数 | `0` | 是否启用混合推理: `1`-启用，`0`-禁用 | 设置为`0`禁用混合推理，行为与vLLM相同 |
| `LK_THREAD_BINDING` | 性能参数 | `CPU_CORE` | 线程绑定策略: `CPU_CORE`-按CPU核心绑定，`NUMA_NODE`-按NUMA节点绑定 | 默认按CPU核心绑定, 遇到性能问题时可尝试按NUMA节点绑定 |
| `LK_THREADS` | 性能参数 | 自动计算 | 线程数量: 物理核心数-4 | 多GPU多进程时，物理核心数-4除以进程数量 |
| `OMP_NUM_THREADS` | 性能参数 | 系统逻辑核心数量 | OpenMP线程数: 设置为`LK_THREADS`相同 |   | 
| `LVLLM_MOE_USE_WEIGHT` | 性能参数 | `TO_DTYPE` | 运行时专家权重格式`TO_DTYPE`: 与config.yaml中dtype一致,bfloat16/float16, `KEEP`: 与模型一致，`INT4`: int4  |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU预填充参数 | 无 | 常驻GPU的MOE专家层`0`: 第0层，`0-1`: 第0层到第1层，`0,9`: 第0层和第9层 | 留足KV Cache显存后，分配多层可增加性能，并减少对应的内存占用，包含0层才有加速效果 |
| `LVLLM_GPU_PREFETCH_WINDOW` | GPU预填充参数 | 无 | 预取窗口大小`1`: 预取1层MOE专家 |  一般预取1到2层即可 |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | GPU预填充参数 | 无 | 使用GPU预填充的最小输入长度`4096`：输入长度达到该值后，启动GPU预填充 | 设置值不宜过小，设置为0则关闭GPU预填充功能 |
| `LK_POWER_SAVING` | cpu节能 | 0 | `1`：启用cpu节能模式，`0`：禁用cpu节能模式 | 建议值：`0` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | 性能参数 | 0 | `0`：快速加载模型，`1`：慢速加载模型可避免OOM | 建议值：加载模型文件时，内存充裕使用`0`，内存紧张使用`1` |
| `LVLLM_MOE_QUANT_ON_GPU` | 性能参数 | 0 | `0`：不启用GPU专家量化，`1`：启用GPU专家量化 | 显存充足可启用（仅加载时有效，推理时不会额外占用显存），加快模型加载速度 |


## 配置文件

config.yaml示例, `建议值`在运行不同模型时无需修改

```bash  
model: "/home/guqiong/Models/Models/MiniMax-M2.5"  #模型目录
host: "0.0.0.0"                                       # 服务绑定IP地址
port: 8070                                            # 服务绑定端口号
tensor-parallel-size: 2                               # 张量并行大小， 小于等于GPU数量，   
#pipeline-parallel-size: 2                            # 流水线并行大小， 小于等于GPU数量     
max-model-len: 18000                                  # 最大上下文长度， 小于等于模型最大长度
gpu-memory-utilization: 0.92                          # 分配给lvllm的GPU显存分配百分比， 小于等于1
trust-remote-code: true                               # 是否信任远程代码， 建议值
tokenizer-mode: "auto"                                # 分词器模式， 建议值
swap-space: 0                                         # 交换空间大小， 单位GB， 建议值
served-model-name: "Models/MiniMax-M2.5"              # 服务模型名称
compilation_config.cudagraph_mode: "FULL_DECODE_ONLY" # 启用CUDA图模式， 建议值
enable_prefix_caching: true                           # 启用前缀缓存， 建议值
enable-chunked-prefill: true                          # 启用分块预填充， 建议值  
max_num_batched_tokens: 18000                         # 最大批量填充令牌数， 关闭GPU预填充时建议值：1024，开启GPU预填充时建议值：同max-model-len
dtype: "bfloat16"                                     # 模型中间计算数据类型， 建议值bfloat16或float16
max_num_seqs: 4                                       # 最大并发请求序列， 建议值1到4
compilation_config.mode: "VLLM_COMPILE"               # 优化模型， 建议值
# enable-auto-tool-choice: true                       # 允许工具调用
# kv_cache_dtype: "fp8"                               # KV Cache数据类型， 40系、50系GPU可开启 
# speculative-config: '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'  # 推测解码， 建议值关闭
# tool-call-parser: "minimax_m2"                      # MiniMax M2.1 模型配置参数
# reasoning-parser: "minimax_m2_append_think"         # MiniMax M2.1 模型配置参数
# tool-call-parser: glm47                             # GLM4.7 模型配置参数
# reasoning-parser: glm45                             # GLM4.7 模型配置参数
# tool-call-parser: "kimi_k2"                         # Kimi k2.5 模型配置参数
# reasoning-parser: "kimi_k2"                         # Kimi k2.5 模型配置参数
# mm-encoder-tp-mode: "data"                         #  encoder TP mode
# reasoning-parser: "step3p5"                         # Step-3.5-Flash 模型配置参数                                              
# tool-call-parser: "step3p5"                         # Step-3.5-Flash 模型配置参数                                              
# hf-overrides.num_nextn_predict_layers: 1            # Step-3.5-Flash 模型配置参数, 建议不使用
# speculative_config.method: "step3p5_mtp"            # Step-3.5-Flash 模型配置参数, 建议不使用
# speculative_config.num_speculative_tokens: 1        # Step-3.5-Flash 模型配置参数, 建议不使用
# disable-cascade-attn: true                          # Step-3.5-Flash 模型配置参数
# tool-call-parser: "qwen3_coder"                     # Qwen3-Coder-Next 模型配置参数

```


## 安装步骤

### 1. 安装CUDA 12.9

```bash
# 卸载旧版本CUDA和NVIDIA驱动
sudo /usr/local/cuda/bin/cuda-uninstaller   
sudo nvidia-uninstall

# 下载并安装CUDA 12.9 
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run
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

# 安装PyTorch 2.9.1
pip install torch==2.9.1

# 使用现有PyTorch
python use_existing_torch.py

# 安装构建依赖
pip install -r requirements/build.txt
```
 
### 4. 安装Lvllm

```bash 
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```

**参数说明：**
- `MAX_JOBS=32 NVCC_THREADS=1`: 减少编译内存占用
- `CMAKE_BUILD_TYPE=Release`: 性能优化选项
- `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release`: 性能优化选项
 
 
 
## 更新

如果已安装Lvllm，需要更新到最新版本，请执行以下命令：

```bash 
git fetch && git reset --hard origin/main && git clean -fd # 此命令适合普通用户，如果保留本地修改内容的用户应知道提前做处理

# 安装PyTorch 2.9.1 
pip uninstall torchaudio triton torchvision torch vllm
pip install torch==2.9.1

# Qwen3-VL GLM4.6V 需要安装 xformers
  
# 编译安装
python use_existing_torch.py 
pip install -r requirements/build.txt
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv

rm -rf ~/.cache/vllm
```
 
## 优化

### MoE常驻显存, 线性增加decode和prefill速度
```bash
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 # 0-5层MoE层常驻显存
#LVLLM_GPU_RESIDENT_MOE_LAYERS=0,1,8-9 # 0,1,8-9层MoE层常驻显存
#LVLLM_GPU_RESIDENT_MOE_LAYERS="" # 关闭MoE常驻显存
``` 

### 开启GPU预填充
```bash
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-2 # 0-2层MoE常驻显存, 开启GPU预填充包含0层方可发挥最佳性能
#LVLLM_GPU_RESIDENT_MOE_LAYERS=3-4 # 少数模型起始层号不为0，例如Step-3.5-Flash模型起始为3 
LVLLM_GPU_PREFETCH_WINDOW=1 # 预取1层, 建议值为1-2, 多了无意义
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 #  输入长度达到4096启动GPU prefill，根据cpu prefill性能可减小或加大， 提前或推后启动prefill
max_num_batched_tokens: 65536 # 与上下文大小相同获得最佳性能，可根据显存情况适当调小，超过上下文大小无意义
``` 

### 关闭GPU预填充
```bash
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 #  关闭GPU预填充
#LVLLM_GPU_PREFILL_MIN_BATCH_SIZE="" # 关闭GPU预填充 
max_num_batched_tokens: 1024 # 1024至8192，太大无意义（占用显存及启动时间过长）
``` 

### 线程绑定到CPU核心
```bash
LK_THREAD_BINDING=CPU_CORE # 绑定到CPU核心（包括超线程逻辑核心）, 最佳性能
#LK_THREAD_BINDING=NUMA_NODE # 绑定到NUMA节点, 次优选择，解决部署在虚拟化平台的极端性能问题
``` 
### BIOS NUMA 设置
```bash
AMD EPYC：设置NPS4获得最佳性能
Intel XEON：设置SNC4获得最佳性能
通常：2,4,8个节点，最多支持32节点，节点越多越好，节点数为GPU倍数获得最佳性能 # 部分虚拟化平台或Intel平台不要设置5、10节点，设置2节点避免性能问题
```

### 线程数设置
```bash
线程数 <= （核心数 - x）/ 张量并行数（TP size）  # x 留给其它任务的线程，至少4线程
LK_THREADS=44                    # 96核心，2个GPU， 每个GPU 44线程， 88线程, 剩余8线程留给其它任务
线程数太大可能会引发性能问题        # 虽然系统会自动条件线程数，但建议手动设置进行测试
```
### decode性能
```bash
compilation_config.mode: "VLLM_COMPILE"                 # 支持2080ti及以上GPU
compilation_config.cudagraph_mode: "FULL_DECODE_ONLY"   # 开启CUDAGraph
```

### 显存设置
```bash
gpu-memory-utilization: 0.80 # 24G显存开启GPU预填充时，留出足够临时显存用于计算，否则会导致长上下文预填充性能大幅下降，启动时间过长
# gpu-memory-utilization: 0.92 # 24G显存关闭GPU预填充时，无需留出过多显存 
# max_num_seqs: 1 # 最多1并发，最大节省显存
max_num_seqs: 4 # 最多4并发，常规节省显存
max_num_batched_tokens: 1024  # 关闭GPU预填充时,节省显存，性能不变，但如果开启GPU预填充会导致性能下降
max_num_batched_tokens: 65536 或更大 # 开启GPU预填充时，获得最佳性能，但如果关闭GPU预填充会导致性能下降 
```
### CPU节能
```bash
LK_POWER_SAVING=1 # 开启后推理时降低CPU温度，性能轻微降低
```

### FP8模型权重运行时格式
```bash
LVLLM_MOE_USE_WEIGHT=INT4 # 模型MoE专家权重使用INT4推理，其余部分依旧为FP8，开启几乎不影响精度， 速度排序：INT4 > TO_DTYPE > KEEP
```





