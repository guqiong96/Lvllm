## LvLLM GPU+NUMA 混合推理MOE大模型！！！ 一张3090运行qwen3-next-80b, 预处理590ts，解码40ts ！！

# 2025-10-14 开启cuda graph , decode 速度翻倍！！ 输出质量提高！！
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/b9760c71-d07b-423a-9e8d-f70c3a007a1b" />




# 2025-09-30 已验证：Qwen3-Next-80B-A3B-Instruct、Qwen3-Coder-30B-A3B-Instruct 
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/c37da729-a692-4b20-b7f5-b7798acd22c4" />
 

# 当前限制：
1、仅支持dtype: "bfloat16"

2、仅支持compilation_config.cudagraph_mode: "NONE" [2025.10.14已没有限制]

3、仅支持moe模型

4、仅支持max_num_batched_tokens: 1024

## 安装步骤

### 1. 安装CUDA 12.8

```bash
# 卸载旧版本CUDA和NVIDIA驱动
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall

# 下载并安装CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run

# 升级libstdcxx-ng  （避免glibcxx_3.4.32 not found， 新增的vllm._lk_C模块无法加载退回到原始vllm模式，最后显存溢出）
conda install -c conda-forge libstdcxx-ng
```

### 2. 创建并激活Python环境

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm
```

### 3. 克隆仓库并安装依赖

```bash
# 克隆Lvllm仓库
git clone https://github.com/guqiong96/Lvllm.git

# 使用现有PyTorch
python use_existing_torch.py

# 安装构建依赖
MAX_JOBS=32 NVCC_THREADS=1 pip install -r requirements/build.txt
```

### 4. 克隆第三方依赖库(可选,github网络好可以直接第5步)

```bash
cd ~/Downloads/Lvllm/.deps

git clone https://github.com/nvidia/cutlass.git cutlass-src
cd cutlass-src
git checkout v4.0.0
cd ..

git clone https://github.com/oneapi-src/oneDNN.git oneDNN-src
cd oneDNN-src
git checkout v3.9
cd ..

git clone https://github.com/vllm-project/FlashMLA flashmla-src
cd flashmla-src
git checkout a757314c04eedd166e329e846c820eb1bdd702de
cd ..

git clone https://github.com/vllm-project/flash-attention.git vllm-flash-attn-src
cd vllm-flash-attn-src
git checkout ee4d25bd84e0cbc7e0b9b9685085fd5db2dcb62a
cd ..

# 安装指定版本的llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git llama_cpp-src
cd llama_cpp-src
git checkout a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
cd ..

# 安装flashinfer-python(可选)
pip install flashinfer-python==0.3.1 
```

### 5. 安装Lvllm

```bash
cd ~/Downloads/Lvllm
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```
MAX_JOBS=32 NVCC_THREADS=1 减少编译时内存占用，避免卡死
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" 性能选项

## 启动命令 使用flashinfer-python

使用以下命令启动Lvllm服务:
 
```bash 
LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS="88" OMP_NUM_THREADS="88" VLLM_ATTENTION_BACKEND="FLASHINFER" vllm serve --config ~/Downloads/Lvllm/config.yaml
```
修改config.yaml里面配置参数
LK_THREADS: 总计使用的CPU线程数，一般比总的线程数少10%，例如48核心96线程，LK_THREADS="88"
OMP_NUM_THREADS：torch并发线程数，保持与LK_THREADS一致

### 配置说明

配置文件 `config.yaml` 包含以下主要参数：

- `model`: 模型路径 (`/Downloads/Qwen3-Next-80B-A3B-Instruct`)
- `host`: 主机地址 (`0.0.0.0`，表示监听所有IPv4地址)
- `port`: 服务端口 (`8070`)
- `tensor-parallel-size`: 张量并行大小 (`1`)
- `max-model-len`: 最大模型序列长度 (`66000`)
- `gpu-memory-utilization`: GPU内存利用率 (`0.8`)
- `trust-remote-code`: 信任远程代码 (`true`) 
- `enable_prefix_caching`: 启用前缀缓存 (`true`)
- `enable-chunked-prefill`: 启用分块预填充 (`true`)
- `max_num_batched_tokens`: 最大批处理令牌数 (`1024`)

根据实际环境需求，可以修改配置文件中的参数或调整环境变量值。

![78af18a5395c987b3f716bb11cc2cad7](https://github.com/user-attachments/assets/d3fe8f56-b8bc-4b28-84e6-6ebf9ff0e9bd)

