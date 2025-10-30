# LvLLM GPU+NUMA 混合推理MOE大模型 - 本地部署推理模型

## 2025-10-30: 支持部分GGUF模型混合推理 [查看config.yaml里面的新参数]

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/898ad4e5-a562-43c5-a10f-d130ec8ba0a0" />

```bash
# 需要合并GGUF为单个文件： 
llama-gguf-split --merge ~/Models/XXXXX-00001-of-00005.gguf ~/Models/XXXX-merged.gguf
```

## 2025-10-19: FP8支持GPU+NUMA 混合推理MOE模型！！ [显存FP8精度，内存FP16精度] 已验证GLM-4.5-Air-FP8
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/272b4e89-48e8-4cb5-8b8c-a892725dfe06" />


## 2025-10-14: 开启cuda graph , decode 速度翻倍！！ 输出质量提高！！

config.yaml里面设置dtype: "float16"相比不设置或设置为dtype: "bfloat16" 有1.5倍prefill速度提升，带amx的至强可能不受影响

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/b9760c71-d07b-423a-9e8d-f70c3a007a1b" />




## 2025-09-30 已验证：Qwen3-Next-80B-A3B-Instruct、Qwen3-Coder-30B-A3B-Instruct 
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/c37da729-a692-4b20-b7f5-b7798acd22c4" />
 

# 当前限制：
1、仅支持原版BF16模型、FP8原版或FP8量化模型 [2025-10-19: FP8支持GPU+NUMA 混合推理MOE模型, 2025.10.30 支持GGUF单文件模型]

2、仅支持compilation_config.cudagraph_mode: "NONE" [2025.10.14已没有限制]

3、仅支持moe模型

4、仅支持max_num_batched_tokens: 1024

5、仅支持单卡推理

## 安装步骤

### 1. 安装CUDA 12.9 ( 50系显卡注意https://github.com/guqiong96/Lvllm/issues/5 ）

```bash
# 卸载旧版本CUDA和NVIDIA驱动
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall

# 下载并安装CUDA 12.9 
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run


```

### 2. 创建并激活Python环境及一些系统库

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm

# 升级libstdcxx-ng  （避免glibcxx_3.4.32 not found， 新增的vllm._lk_C模块无法加载退回到原始vllm模式，最后显存溢出）
conda install -c conda-forge libstdcxx-ng

# 安装NUMA库 ubuntu
sudo apt-get install libnuma-dev
# 安装NUMA库 rocky linux
sudo dnf install numactl-devel
```

### 3. 克隆仓库并安装依赖

```bash
# 克隆Lvllm仓库
git clone https://github.com/guqiong96/Lvllm.git


# 安装PyTorch 2.8.0 （可选 Qwen3-VL 需要安装 xformers、torchvision）
pip uninstall torch
pip install triton xformers torchvision torch==2.8.0

# 50 系列 GPU 需要安装 xformers==0.0.33.dev1086 
pip install xformers==0.0.33.dev1086 
 

# 使用现有PyTorch
python use_existing_torch.py

# 安装构建依赖
pip install -r requirements/build.txt
```

### 4. 克隆第三方依赖库(可选,github网络好可以直接第5步)

```bash

mkdir -p .deps
cd .deps

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

# 安装flashinfer-python
pip install flashinfer-python==0.4.1 
```

### 5. 安装Lvllm

```bash
# 一般安装
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```

```bash 
# AMX指令集支持安装
ENABLE_AMX_INT8=1  MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```
ENABLE_AMX_INT8 

MAX_JOBS=32 NVCC_THREADS=1 减少编译时内存占用，避免卡死
CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" 性能选项

## 启动命令

使用以下命令启动Lvllm服务: 
```bash 
LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS="88" OMP_NUM_THREADS="88" vllm serve --config ~/Downloads/Lvllm/config.yaml
```
VLLM_ATTENTION_BACKEND="FLASHINFER": 这个环境变量已不是最优选项[2025-10-21]
修改config.yaml里面配置参数
LK_THREADS: 总计使用的CPU线程数，一般比总的线程数少10%，例如48核心96线程，LK_THREADS="88"
OMP_NUM_THREADS：torch并发线程数，保持与LK_THREADS一致

### 错误排查
运行以下命令，将错误输出提交至Issues或微信群
```bash 
python -c "import  vllm._lk_C"
```

### 更新已有Lvllm

如果已安装Lvllm，需要更新到最新版本，请执行以下命令：

```bash
git pull --force
python use_existing_torch.py 
pip install -r requirements/build.txt
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```

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

 GGUF模型重要参数：
- `hf-config-path`: HF模型配置路径 (`/home/guqiong/Downloads/Qwen3-Coder-30B-A3B-Instruct`)
- `tokenizer`: 分词器路径 (`/home/guqiong/Downloads/Qwen3-Coder-30B-A3B-Instruct`)

根据实际环境需求，可以修改配置文件中的参数或调整环境变量值。

# LvLLM GPU+NUMA Hybrid Inference for MOE Large Models!!! Run qwen3-next-80b on a single RTX 3090, with 590 tokens/s prefill and 40 tokens/s decoding!

## October 30, 2025: Supports some GGUF model hybrid inference [view new params in config.yaml]

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/b95902d5-4ce8-4bdb-9bc8-68f9e74acaaf" />

```bash
# Need to merge GGUF into a single file： 
llama-gguf-split --merge ~/Models/XXXXX-00001-of-00005.gguf ~/Models/XXXX-merged.gguf
```
## October 14, 2025: CUDA Graph Enabled, Decoding Speed Doubled!!! Output Quality Improved!!!

Setting dtype: "float16" in config.yaml provides a 1.5x prefill speed increase compared to not setting it or setting it to dtype: "bfloat16". Xeon processors with AMX may not be affected.

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/b9760c71-d07b-423a-9e8d-f70c3a007a1b" />

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/2ae72c33-cea9-4777-8862-5bd711d3f004" />


## October 19, 2025: FP8 supports GPU NUMA hybrid inference MOE models!! [VRAM FP8 precision, memory FP16 precision]

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/272b4e89-48e8-4cb5-8b8c-a892725dfe06" />

## September 30, 2025: Verified Models: Qwen3-Next-80B-A3B-Instruct, Qwen3-Coder-30B-A3B-Instruct
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/c37da729-a692-4b20-b7f5-b7798acd22c4" />


# Current Limitations:
1. Only supports dtype: "bfloat16" and "float16" "fp8" [October 19, 2025: FP8 supports GPU NUMA hybrid inference MOE models, October 30, 2025: Supports GGUF model hybrid inference single gguf file]

2. Only supports compilation_config.cudagraph_mode: "NONE" [No limitation as of October 14, 2025]

3. Only supports MOE models

4. Only supports max_num_batched_tokens: 1024

## Installation Steps

### 1. Install CUDA 12.9  ( Attention to 50 series GPUs https://github.com/guqiong96/Lvllm/issues/5 ）

```bash
# Uninstall old CUDA and NVIDIA drivers
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall

# Download and install CUDA 12.9
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run


```

### 2. Create and Activate Python Environment and Some System Libraries

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm

# Upgrade libstdcxx-ng (to avoid glibcxx_3.4.32 not found error, which prevents loading vllm._lk_C module and causes memory overflow)
conda install -c conda-forge libstdcxx-ng
# install libnuma-dev on ubuntu
sudo apt-get install libnuma-dev
# install numactl-devel on rocky linux
sudo dnf install numactl-devel
```

### 3. Clone Repository and Install Dependencies

```bash
# Clone Lvllm repository
git clone https://github.com/guqiong96/Lvllm.git 
 
# Install PyTorch 2.8.0 Optional（Qwen3 Qwen3-VL models need install xformers、torchvisionn）
pip uninstall torch
pip install triton xformers torchvision torch==2.8.0

# 50 series GPUs require installing xformers==0.0.33.dev1086 for Qwen3 Qwen3-VL models
pip install xformers==0.0.33.dev1086 

# Use existing PyTorch
python use_existing_torch.py

# Install build dependencies
pip install -r requirements/build.txt

```

### 4. Clone Third-party Dependencies (Optional, skip if GitHub connection is good)

```bash

mkdir -p .deps
cd .deps

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

# Install specific version of llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git llama_cpp-src
cd llama_cpp-src
git checkout a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
cd ..

# Install flashinfer-python
pip install flashinfer-python==0.4.1 
```

### 5. Install Lvllm

```bash
# General Installation
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```
```bash 
# AMX instruction set supports installation
ENABLE_AMX_INT8=1  MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```
MAX_JOBS=32 NVCC_THREADS=1 reduces memory usage during compilation to avoid freezing
CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" for performance optimization

## Startup Command Using FlashInfer

Use the following command to start the Lvllm service:
(Running the Qwen3-Next-80B-A3B-Instruct-FP8 model, setting the environment variable VLLM_MARLIN_USE_ATOMIC_ADD=1 slightly increases decode speed but slightly decreases prefill speed) 
```bash 
LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS="88" OMP_NUM_THREADS="88" VLLM_ATTENTION_BACKEND="FLASHINFER" vllm serve --config ~/Downloads/Lvllm/config.yaml
```

Use the following command to start the Lvllm service (Qwen3-VL does not support VLLM_ATTENTION_BACKEND="FLASHINFER" ):
```bash 
LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS="88" OMP_NUM_THREADS="88" vllm serve --config ~/Downloads/Lvllm/config.yaml
```

Modify the configuration parameters in config.yaml
LK_THREADS: Total CPU threads to use, typically 10% less than total threads (e.g., 88 for a 48-core 96-thread processor)
OMP_NUM_THREADS: Torch concurrency threads, should be the same as LK_THREADS

### Troubleshooting
Run the following command and submit the error output to Issues or WeChat group
```bash 
python -c "import  vllm._lk_C"
```

### Update Existing LvllmIf 

Lvllm is already installed and needs to be updated to the latest version, please execute the following command:

```bash
git pull --force
python use_existing_torch.py 
pip install -r requirements/build.txt
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```

### Configuration Explanation

The configuration file `config.yaml` contains the following main parameters:

- `model`: Model path (`/Downloads/Qwen3-Next-80B-A3B-Instruct`)
- `host`: Host address (`0.0.0.0`, meaning listen on all IPv4 addresses)
- `port`: Service port (`8070`)
- `tensor-parallel-size`: Tensor parallel size (`1`)
- `max-model-len`: Maximum model sequence length (`66000`)
- `gpu-memory-utilization`: GPU memory utilization (`0.8`)
- `trust-remote-code`: Trust remote code (`true`)
- `enable_prefix_caching`: Enable prefix caching (`true`)
- `enable-chunked-prefill`: Enable chunked prefill (`true`)
- `max_num_batched_tokens`: Maximum number of batched tokens (`1024`)
Important parameters of the GGUF model:
- `hf-config-path`: Path to the HF model configuration (`/home/guqiong/Downloads/Qwen3-Coder-30B-A3B-Instruct`)
- `tokenizer`: Path to the tokenizer (`/home/guqiong/Downloads/Qwen3-Coder-30B-A3B-Instruct`)

Depending on the actual environment 
You can modify the parameters in the configuration file or adjust the environment variable values according to your actual environment needs.


![90118eea5ff29b8975840b83b7168368](https://github.com/user-attachments/assets/9879df82-f03f-49c7-980d-4c9b4cb42be2)



