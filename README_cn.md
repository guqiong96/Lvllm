# LvLLM — 面向 vllm 的 lk_moe 混合推理 [[English]](./README.md)

LvLLM 是 [vLLM](https://github.com/vllm-project/vllm) 的一个特殊扩展，充分利用 CPU 和 GPU 计算资源，
采用高效的 **GPU 并行 + NUMA 并行**架构，适用于 MOE 模型混合推理。

真正的混合推理引擎是 **[lk_moe](https://pypi.org/project/lk-moe/)**，vllm 只提供"GPU 路径"，
lk_moe 提供"混合路径"，LvLLM 是 lk_moe 集成到 vllm 的具体案例。

> **发布策略：** LvLLM 的版本更新将**随 vllm release 同步发布** —— 在最新 vllm tag 之上保持
> "原样 + lk_moe"。除非有必要的错误纠正补丁，否则不叠加额外功能，与上游的差异始终保持最小（仅
> lk_moe 这一层）。

---

## 为什么要用 lk_moe？

lk_moe 让 MOE 模型的占用横跨**显存 + 内存**，并在 NUMA 感知下把专家计算调度到 **CPU + GPU**：

- **显存 + 内存负载均衡**：模型总体占用 = 显存 + 内存，可实现 "1+1=2"、100% 显存利用率。
- **CPU-GPU 混合解码 / 预填充 + GPU 预填充**：三种计算方式，GPU 预填充与混合解码并行，接近 100%
  显卡利用率。
- **NUMA 线程优化**：跨节点通信占比低至 3%，三级缓存命中率 50% 以上。

| 混合模式 | 环境变量控制 |
|---|---|
| **总开关** —— `0` 即 vllm 纯GPU推理（下面所有模式均关闭），`1` 即启用混合推理 | `LVLLM_MOE_NUMA_ENABLED` |
| CPU预填充 / GPU 预填充 | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` + `LVLLM_GPU_PREFETCH_WINDOW` |
| GPU预填充和解码 | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |

注1：x86 带 AVX2 以上指令集的 CPU 和 Nvidia GPU sm75 以上架构。

---

## 如何集成 lk_moe

lk_moe 通过 `pip install lk_moe` 安装，它对外暴露少量 C++ 内核类（`MOE_WNA16`、
`MOE_FP8`、`MOE_MXFP4`、`LKEmbedding` 等），由 `MOEConfigV2` 配置驱动。引擎内部处理专家权重放置
（显存 / 钉住的 NUMA 主机内存）、NUMA 感知调度和量化内核执行。

vllm 侧的集成工作**只是把每个 MOE 层路由到 lk_moe**（哪些层留在 GPU、哪些走混合、
用哪个量化内核），并**保持该功能可选**，关闭时与原生 vllm 完全一致。

### 核心集成原则

> **每个 MOE 层可以扮演三种角色之一：** 角色由几个环境变量决定，引擎其它部分不变。

| 角色 | 含义 | 判定 |
|---|---|---|
| GPU 常驻层 | 所有权重在显存，走原始 GPU 路径 | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |
| CPU 层（混合） | MoE权重在内存，attn在显存，GPU 计算attn + cpu计算MoE | 开启时的默认 |
| GPU 预填充层 | 大批量在 GPU 计算，小批量走CPU | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` |

### 最小集成清单

1. **添加依赖** —— 在 `requirements` 加入 `lk_moe`。
2. **加功能开关** —— `is_lk_moe_feature_enabled()`（读取 `LVLLM_MOE_NUMA_ENABLED`），默认关闭所有
   混合行为，分支行为与原生 vllm 完全一致。
3. **接通 MOE 层** —— 在 fused-MoE 层中解析每层的角色、构造 `lk_moe.MOEConfigV2`、实例化对应量化
   的 `MOE_*` 类并在 `forward` 中调用。
4. **注册各量化内核** —— 每种量化方法暴露对应的 LK MoE 内核类。
5. **处理权重加载/放置** —— 常驻 CPU 的权重不要放到 GPU 设备上。
6. **（可选）扩展** —— CPU 常驻 embedding（`LKEmbedding`）与 NUMA 线程绑定。

### 集成案例 — LvLLM（vllm）逐文件

整个 lk_moe 集成被整理为**一个可移植补丁**：[`patches/01_lk_moe__3116c5d.patch`](./patches/01_lk_moe__3116c5d.patch)
—— 即上游 base commit `3116c5d` 与当前 lk_moe 分支的全部代码差异（不包含 README/RELEASE_NOTES 文档）。
在干净的 `3116c5d` checkout 上执行 `git apply patches/01_lk_moe__3116c5d.patch` 即可。

| 文件 | 作用 |
|---|---|
| `vllm/envs.py` | 功能开关辅助函数：`is_lk_moe_feature_enabled`、`is_lk_moe_cpu_layer`、`is_lk_moe_gpu_resident_layer`、`is_lk_moe_gpu_prefill_layer`、`get_gpu_prefetch_window` 等 |
| `vllm/model_executor/layers/fused_moe/routed_experts.py` | **核心**：解析层角色、构造 `MOEConfigV2`、按量化实例化 `MOE_WNA16` / `MOE_FP8` / `MOE_MXFP4`，并在 `forward` 分发（GPU 常驻 → `quant_method.apply`；混合 → `_cpu_decode` / `_cpu_prefill` / `_gpu_prefill`） |
| `vllm/model_executor/layers/quantization/{fp8,mxfp4,auto_awq,...}.py` | 每种量化方法注册其 LK MoE 内核（如 `MOE_FP8`、`MOE_MXFP4`） |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/*` | compressed-tensors 的 W8A8-FP8 / W4A4-NVFP4 / WNA16 MoE 各自注册 LK 内核 |
| `vllm/model_executor/model_loader/utils.py` | 让 CPU 常驻层 / lk-embedding 不上 GPU 设备；为 lk_moe 层执行 `process_weights_after_loading` / `clean_weights_after_loading` |
| `vllm/utils/numa_utils.py` | 当 `LVLLM_ENABLE_NUMA_INTERLEAVE=1` 时，用 `numactl --interleave=all` 启动 worker |

相同的方法应用于 sglang，见 [Lsglang](https://github.com/guqiong96/Lsglang) 仓库，另有
DeepSeek-V4 专用分支：[Lvllmds4](https://github.com/guqiong96/Lvllmds4)（SM120+）和
[Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x)（SM80+）。

---

## 三、效果实例 — LvLLM（含基准）

### 性能基准

开启 GPU 预填充，`max_num_batched_tokens=8192`（第 1 行）/ `32768`（第 2 行）：

| 模型 | 版本 | CPU | 内存 | GPU | Prefill | Decode | 推测解码 |
|-------|---------|-----|--------|-----|---------|--------|---------|
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllm-v2.4.0 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | 850 t/s [输入 32768] | 28 t/s [输入 32768] | 30~46 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllmds4-x-v2.3.9 | EPYC 7642 *2 | 16ch ddr4 3200 | 3090 * 2 | 1060 t/s [输入 32768] | 26 t/s [输入 32768] | 35~47 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lvllmds4-v2.3.9 | EPYC 9684x *2 | 24ch ddr5 4800 | pro 6000 * 1 | 3100 t/s [输入 131072] | 75 t/s [输入 131072] | 100~115 t/s |

### 版本变更

```bash
2026-09-10: Lvllm - 基于上游 base commit 3116c5d 重建（UVA PLE-offload + Engram TP），lk_moe 集成（README/RELEASE_NOTES/patch）
2026-07-17: lvllm-v2.3.6 - 新增 dtype float16 支持 for SM75 GPU Prefill
2026-07-08: lvllm-v2.3.2 - 新增 ModelOpt W4A16 NVFP4 量化类型支持
2026-07-05: lvllm-v2.3.0 - 优化GPU预填充速度，CPU AVX512优化，取消LVLLM_GPU_RESIDENT_MOE_EXPERTS
2026-06-05: lvllm-v2.2.0 - 升级lk_moe模块, 新增支持nvfp4, mxfp4量化类型，增加LVLLM_GPU_RESIDENT_MOE_EXPERTS, 取消LVLLM_MOE_USE_WEIGHT、LVLLM_MOE_QUANT_ON_GPU
2026-04-06: lvllm-v2.1.0 - 增强使用LK_POWER_SAVING=1节能效果，支持FP8+BF16+AWQ4bit的混合MOE层推理
2026-03-22: lvllm-v2.0.0 - FP8 MoE模型使用INT4专家量化时支持逐层加载，减少峰值内存占用，LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1
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
```

### 支持的模型与量化格式

vllm 已验证的大部分原版 MOE 模型（Qwen3/GLM/MiniMax 等系列）：
gemma-4-26B-A4B-it、NVIDIA-Nemotron-3-Super-120B-A12B-BF16、Ornith-1.0-35B-FP8、
Qwen3.6/3.5-35B-A3B、Qwen3.5-122B-A10B、Qwen3.5-397B-A17B、Qwen3-Coder-Next / 30B-A3B、
Qwen3-Next-80B-A3B-Instruct、Qwen3-VL-30B-A3B-Instruct、MiniMax-M3/M2.7/M2.5/M2.1、
GLM-5.2-NVFP4 [sm120]、GLM-4.7(-Flash)/4.6V、Kimi k2.6/k2.5、**deepseek-ai/DeepSeek-V4-Flash-0731 [sm120]**。

未列出的 Qwen3 系列、GLM 系列、MiniMax 系列的原版 MOE 模型理论上支持，待实际测试。

运行时支持的量化格式：

| 模型文件 | 运行时格式 |
|---------|------------|
| bfloat16 | bfloat16/float16 |
| float16 | bfloat16/float16 |
| fp8模型 | fp8 |
| nvfp4模型 | NVFP4 and ModelOpt W4A16 NVFP4 |
| mxfp4模型 <sup>注1</sup> | mxfp4 |
| awq 4bit对称量化模型 <sup>注1</sup> | w4a16 |

```bash
注1：https://hf-mirror.com/cyankiwi 提供AWQ 4bit对称量化模型
注2：DeepSeek V4 SM80、SM86、SM89需使用专用版本：
https://github.com/guqiong96/Lvllmds4-x/releases SM80+  （另有 SM120 版本：
https://github.com/guqiong96/Lvllmds4/releases SM120）
```

### 快速开始（DeepSeek V4 Flash [RTX 3090 *2 OR 5060Ti *2]）

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

### 配置参数

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `LVLLM_MOE_NUMA_ENABLED` | 核心参数 | `0` | 是否启用混合推理: `1`-启用，`0`-禁用（`0` 时与原生 vllm 相同） |
| `LK_THREAD_BINDING` | 性能参数 | `CPU_CORE` | `CPU_CORE` 按 CPU 核心绑定，`NUMA_NODE` 按 NUMA 节点绑定 |
| `LK_THREADS` | 性能参数 | - | 线程数 =（物理核心数）÷ 显卡数量 |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU参数 | 无 | 常驻 GPU 显存的专家层：`0`、`0-1`、`0,9` |
| `LVLLM_GPU_PREFETCH_WINDOW` | 预填充参数 | 无 | 预取窗口大小，一般 `1` 即可 |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | 预填充参数 | 无 | 输入长度达到该值后启动 GPU 预填充；`0` 关闭 |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | 性能参数 | 1 | `1`：避免 NUMA 节点 OOM |
| `LK_POWER_SAVING` | CPU节能 | 0 | `1`：启用 CPU 节能模式 |

### 安装

```bash
# 卸载旧版本CUDA和NVIDIA驱动，然后安装CUDA 13.2.1
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall
wget https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_595.58.03_linux.run
sudo sh cuda_13.2.1_595.58.03_linux.run

conda create -n Lvllm python==3.12.11 && conda activate Lvllm
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
sudo apt-get install libnuma-dev      # Ubuntu  /  sudo dnf install numactl-devel  # Rocky

pip install https://github.com/guqiong96/Lvllm/releases/download/lvllm-v2.4.0/lvllm-2.4.0-cp312-cp312-manylinux_2_34_x86_64.whl
# 最新版本查看 https://github.com/guqiong96/Lvllm/releases
pip install https://github.com/guqiong96/Lvllm/releases/download/lvllm-v2.4.0/flashinfer_cubin-0.6.16.post3-py3-none-any.whl
```

从源码编译：

```bash
git clone https://github.com/guqiong96/Lvllm.git
cd Lvllm
pip install setuptools_scm setuptools_rust
pip install torchaudio triton torchvision torch==2.13.0
VLLM_VERSION_OVERRIDE="2.4.0" CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip install -e . --no-build-isolation -vvv
```

### 打包发布示例

LvLLM 的发布流程是：可编辑安装 + 构建 wheel + auditwheel 修复 + 上传 PyPI：

```bash
# 清理旧的构建产物
rm -rf build/ CMakeCache.txt CMakeFiles/ *.egg-info/

# 架构列表，覆盖受支持的 GPU（Ampere sm75/sm80/sm86/sm89、
# Hopper sm90、Blackwell sm100/sm120）
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0"

# 先可编辑安装验证
VLLM_VERSION_OVERRIDE="2.4.0" CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip install -e . --no-build-isolation -vvv

# 构建 wheel
VLLM_VERSION_OVERRIDE="2.4.0" CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip wheel . --no-build-isolation -v --wheel-dir=dist

# 修复 wheel（排除系统/CUDA 动态库）
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

### 优化

- **MoE 常驻显存**：`LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5`（格式 `0,1,8-9`；少数模型起始层号不为 0，
  例如 Step-3.5-Flash 起始为 3）。
- **开启 GPU 预填充**：`LVLLM_GPU_PREFETCH_WINDOW=1`、`LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096`、
  `--max-num-batched-tokens 32000`。
- **关闭 GPU 预填充**：`LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0`、`--max-num-batched-tokens 4096`。
- **线程绑定**：`LK_THREAD_BINDING=CPU_CORE`（最佳）、`NUMA_NODE`（解决虚拟化平台/多实例的极端性能问题）。
- **BIOS NUMA**：AMD EPYC NPS4 / Intel XEON SNC4；通常 2,4,8 节点（GPU 倍数最佳），最多 32 节点。
- **线程数**：有超线程 → 物理核心数 ÷ 显卡数；关闭超线程 →（物理核心数-2）÷ 显卡数。
- **输出性能**：`--compilation_config.mode VLLM_COMPILE`（2080ti 及以上）、`--compilation_config.cudagraph_mode FULL_DECODE_ONLY`。
- **显存**：`--max-num-batched-tokens 32000` 决定最大批处理占用的显存量。
- **CPU 节能**：`LK_POWER_SAVING=1`。

---

## 四、如何生成 lk_moe 补丁

可移植补丁 `patches/01_lk_moe__3116c5d.patch` 通过对比当前代码树与上游 base commit 生成（使用本地
`3116c5d` commit，无需联网）。**注意：补丁只包含 lk_moe 集成的代码改动，不包含
README.md / README_cn.md / RELEASE_NOTES.md 等文档。**

```bash
git diff 3116c5d -- . ':!README.md' ':!README_cn.md' ':!RELEASE_NOTES.md' > patches/01_lk_moe__3116c5d.patch
```

在干净的上游 `3116c5d` checkout 上应用：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 3116c5d
git apply ../Lvllm/patches/01_lk_moe__3116c5d.patch
```
