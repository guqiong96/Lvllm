
# LvLLM GPU and NUMA Dual Parallelism

LvLLM is a special extension of vllm that makes full use of CPU and memory resources, reduces GPU memory requirements, and features an efficient GPU parallel and NUMA parallel architecture, supporting hybrid inference for MOE large models.


## 2026-01-04: v1.4.0 Optimize decode to improve speed


## 2025-12-28: Optimize inference speed: bfloat16, awq4bit; optimize NUMA data access for multiple GPUs; enable NUMA nodes for multiple GPUs to achieve the best performance

Known issues with this version: GGUF model are not supported, FP8 models are not supported on GPUs that support E4M3FNUZ


## 2025-12-16: v1.2.0 ynchronized the upstream vllm code to the latest version, optimized lk_moe to reduce memory usage
 
 
## 2025-12-14: v1.1.2 Added inference support for the AWQ-4bit quantized model (symmetric quantization - avx2 version), cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit and cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit has passed verification

torch version updated to 2.9.1(do not use torch 2.9.0)

 
## 2025-12-9: Added the LVLLM_MOE_USE_WEIGHT environment variable to support MOE modules using two modes to infer fp8 models

LVLLM_MOE_USE_WEIGHT="KEEP": lk_moe inference uses the original weight format fp8_e4m3, reducing memory usage during inference.

LVLLM_MOE_USE_WEIGHT="TO_DTYPE": lk_moe inference uses the configured parameter dtype: float16 or bfloat16, which increases fp8 model inference speed, but will increase memory usage and loading time. TO_DTYPE is the default value.

torch version updated to 2.9.1(do not use torch 2.9.0)

## 2025-11-1: Support tensor parallelism and pipeline for multi-card inference   https://b23.tv/xzHieMs
```bash
LK_THREADS and OMP_NUM_THREADS Configuration Rules:
1. Single GPU Inference (N): LK_THREADS and OMP_NUM_THREADS should be set to total number of cores minus 4. If hyper-threading is enabled, set it to total number of threads minus 8.
2. Multi-GPU Inference (N/number of GPUs): For each GPU, set LK_THREADS and OMP_NUM_THREADS to N divided by the number of GPUs.
```
```bash
Tensor parallelism on the old machine, prefill speed 555 tokens/s, decoding speed 47 tokens/s, 80B BF16 model
```

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/d6751a1a-3cf6-451e-9915-4a617af72a15" />

```bash
Pipeline parallelism on the old machine, prefill speed 689 tokens/s, decoding speed 40 tokens/s, 80B BF16 model
```
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/72b0b0d5-f4eb-459b-a777-519bf470b3d6" />

## October 30, 2025:Support Qwen3 series models GGUF hybrid inference (excluding Qwen3-Coder-30B-A3B-Instruct GGUF) [view new params in config.yaml]

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/b95902d5-4ce8-4bdb-9bc8-68f9e74acaaf" />

```bash
# Need to merge GGUF into a single file： 
llama-gguf-split --merge ~/Models/XXXXX-00001-of-00005.gguf ~/Models/XXXX-merged.gguf
```
## October 14, 2025: CUDA Graph Enabled, Decoding Speed Doubled!!! Output Quality Improved!!!


<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/b9760c71-d07b-423a-9e8d-f70c3a007a1b" />

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/2ae72c33-cea9-4777-8862-5bd711d3f004" />


## October 19, 2025: FP8 supports GPU NUMA hybrid inference MOE models!! [VRAM FP8 precision, memory FP16 precision]

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/272b4e89-48e8-4cb5-8b8c-a892725dfe06" />

## September 30, 2025: Verified Models: Qwen3-Next-80B-A3B-Instruct, Qwen3-Coder-30B-A3B-Instruct
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/c37da729-a692-4b20-b7f5-b7798acd22c4" />


# Current Limitations:
1. Only supports dtype: "bfloat16" and "float16" "fp8" [2025-12-14: Added inference support for the AWQ-4bit quantized model (symmetric quantization)，October 19, 2025: FP8 supports GPU NUMA hybrid inference MOE models, October 30, 2025: Only supports GGUF model hybrid inference for Qwen3 series (excluding Qwen3 Next) in single gguf file]

2. Only supports compilation_config.cudagraph_mode: "NONE" [No limitation as of October 14, 2025]

3. Only supports MOE models

4. Only supports max_num_batched_tokens: 1024

5. Only supports single-card inference (support for multi-GPU tensor parallelism (TP) and pipeline parallelism (PP) inference will be available from 2025-11-1)

6. **Known Issue**: Using torch2.9.0 will result in garbled output, which is a compatibility issue between torch2.9.0 and vllm

7. **Known Issue**: There seems to be a slight interference with output quality when multiple requests are made concurrently, which currently appears to be unrelated to LK MOE.

## Installation Steps

### 1. Install CUDA 12.9  ( Attention to 50 series GPUs https://github.com/guqiong96/Lvllm/issues/5 ）

```bash
# Uninstall old CUDA and NVIDIA drivers
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall

# To completely uninstall and clean:
https://www.bilibili.com/opus/1131154984017068033?spm_id_from=333.1387.0.0
[https://github.com/guqiong96/Lvllm/issues/5 ](https://github.com/guqiong96/Lvllm/issues/8)

# Download and install CUDA 12.9
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run


```

### 2. Create and Activate Python Environment and Some System Libraries

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm

# Upgrade libstdcxx-ng (to avoid glibcxx_3.4.32 not found error, which prevents loading lk_moe module and causes memory overflow)
# triton compile version `GLIBCXX_3.4.30' not found  
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# install libnuma-dev on ubuntu
sudo apt-get install libnuma-dev
# install numactl-devel on rocky linux
sudo dnf install numactl-devel
```

### 3. Clone Repository and Install Dependencies

```bash
# Clone Lvllm repository
git clone https://github.com/guqiong96/Lvllm.git 
 
# Install PyTorch 2.9.1 (Optional: Qwen3-VL requires installation of xformers and torchvision)
pip uninstall torchaudio triton xformers torchvision torch
pip install torchaudio triton torchvision torch==2.9.1
pip install xformers

# Previous generations before RTX 50 series GPUs needed to install xformers==0.0.33.dev1090 to run Qwen3-VL properly. Now we need to determine if newer versions of xformers have resolved this issue.

# Use existing PyTorch
python use_existing_torch.py

# Install build dependencies
pip install -r requirements/build.txt

```


### 4. Install Lvllm

```bash
# General Installation
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```
MAX_JOBS=32 NVCC_THREADS=1 reduces memory usage during compilation to avoid freezing 
 
### 5. Run Lvllm
Use the following command to start the Lvllm service:
```bash 
LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS="88" OMP_NUM_THREADS="88" vllm serve --config ~/Downloads/Lvllm/config.yaml
```

Modify the configuration parameters in config.yaml 
LK_THREADS and OMP_NUM_THREADS Configuration Rules:
1. Single GPU Inference (N): LK_THREADS and OMP_NUM_THREADS should be set to total number of cores minus 4. If hyper-threading is enabled, set it to total number of threads minus 8.
2. Multi-GPU Inference (N/number of GPUs): For each GPU, set LK_THREADS and OMP_NUM_THREADS to N divided by the number of GPUs.

LVLLM_MOE_USE_WEIGHT="KEEP": lk_moe inference uses the original weight format fp8_e4m3, reducing memory usage during inference.

LVLLM_MOE_USE_WEIGHT="TO_DTYPE": lk_moe inference uses the configured parameter dtype: float16 or bfloat16, which increases fp8 model inference speed, but will increase memory usage and loading time. TO_DTYPE is the default value.

### Troubleshooting
Run the following command and submit the error output to GitHub Issues 
```bash 
python -c "import  lk_moe"
```

### Update Existing Lvllm

Lvllm is already installed and needs to be updated to the latest version, please execute the following command:

```bash
# Normal situation
git pull --force
# When a conflict occurs
git fetch origin
git reset --hard origin/main

# Install PyTorch 2.9.1 (Optional: Qwen3-VL requires installation of xformers and torchvision)
pip uninstall torchaudio triton xformers torchvision torch
pip install torchaudio triton torchvision torch==2.9.1 xformers 

# Previous generations before RTX 50 series GPUs needed to install xformers==0.0.33.dev1090 to run Qwen3-VL properly. Now we need to determine if newer versions of xformers have resolved this issue.

python use_existing_torch.py 
pip install -r requirements/build.txt
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv

rm -rf ~/.cache/vllm
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

 


# This project is a branch of vLLM and incorporates source code from the following open-source projects:
1. **llama.cpp**- Project URL: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)- Purpose: GGML related definitions
2. **llamafile**- Project URL: [https://github.com/Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile)- Purpose: GGUF weight quantization, dequantization, and matrix multiplication




# LvLLM GPU、NUMA双并行

​LvLLM是vllm的特别扩展，充分利用cpu和内存资源，降低显卡显存要求，高效的GPU并行+NUMA并行架构，支持混合推理MOE大模型 


## 2026-01-04: v1.4.0 优化decode提升速度


## 2025-12-28：优化推理速度：bfloat16、awq4bit；优化多GPU的NUMA数据访问；为多GPU启用NUMA节点以实现最佳性能


版本已知问题：GGUF模型不支持，FP8模型在E4M3FNUZ的GPU上不支持


## 2025-12-16 v1.2.0 同步上游vllm代码至最新，lk_moe优化降低内存占用


## 2025-12-14 v1.1.2 增加AWQ-4bit量化模型（对称量化 avx2版本）推理支持 -，验证通过 cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit and cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit

torch 版本升级至 2.9.1 
 

## 2025-12-9: 增加LVLLM_MOE_USE_WEIGHT环境变量，支持MOE模块使用两种模式推理fp8模型：

LVLLM_MOE_USE_WEIGHT="KEEP": lk_moe推理使用权重原始格式fp8_e4m3, 降低推理内存占用

LVLLM_MOE_USE_WEIGHT="TO_DTYPE": lk_moe推理使用配置参数dtype：float16或bfloat16, 提高fp8模型推理速度， 但会增加内存占用延长加载时间，TO_DTYPE为默认值

torch 版本升级至 2.9.1 

## 2025-11-1： 支持张量并行、流水线多卡推理 https://b23.tv/xzHieMs
```bash
LK_THREADS、OMP_NUM_THREADS设置规则：
1、单GPU推理(N)：LK_THREADS、OMP_NUM_THREADS 设置为总的核心数量-4 , 开启超线程则设置为总的线程数量-8
2、多GPU推理(N/GPU数量)：每个GPU的LK_THREADS、OMP_NUM_THREADS 设置为N/(GPU数量)
```

## 2025-10-31: 更新

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/66c48cf8-ac00-4928-90db-7519ff349fd8" />


## 2025-10-30: 支持Qwen3系列模型GGUF混合推理（不包含Qwen3-Coder-30B-A3B-Instruct GGUF） [查看config.yaml里面的新参数]

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/898ad4e5-a562-43c5-a10f-d130ec8ba0a0" />

```bash
# 需要合并GGUF为单个文件： 
llama-gguf-split --merge ~/Models/XXXXX-00001-of-00005.gguf ~/Models/XXXX-merged.gguf
```

## 2025-10-19: FP8支持GPU+NUMA 混合推理MOE模型！！ [显存FP8精度，内存FP16精度] 已验证GLM-4.5-Air-FP8
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/272b4e89-48e8-4cb5-8b8c-a892725dfe06" />


## 2025-10-14: 开启cuda graph , decode 速度翻倍！！ 输出质量提高！！



<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/b9760c71-d07b-423a-9e8d-f70c3a007a1b" />




## 2025-09-30 已验证：Qwen3-Next-80B-A3B-Instruct、Qwen3-Coder-30B-A3B-Instruct 
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/c37da729-a692-4b20-b7f5-b7798acd22c4" />
 

# 当前限制：
1. 仅支下列类型的模型："bfloat16" 和 "float16" [2025-12-14：AWQ-4bit量化模型（对称量化）推理支持，2025年10月19日：FP8支持GPU NUMA混合推理MOE模型，2025年10月30日：仅支持Qwen3系列的GGUF模型混合推理（不包含Qwen3 Next）单个gguf文件]

2、仅支持compilation_config.cudagraph_mode: "NONE" [2025.10.14已没有限制]

3、仅支持moe模型

4、仅支持max_num_batched_tokens: 1024

5、仅支持单卡推理(2025-11-1支持多GPU张量并行(TP)、流水线并行(PP)推理)

6. **已知问题**：使用torch2.9.0会导致输出乱码，这是torch2.9.0和vllm的兼容性问题

7. **已知问题**：多个请求并发时似乎会轻微干扰输出质量，目前看与LK MOE无关

## 安装步骤

### 1. 安装CUDA 12.9 ( 50系显卡注意https://github.com/guqiong96/Lvllm/issues/5 ）

```bash
# 卸载旧版本CUDA和NVIDIA驱动
sudo /usr/local/cuda/bin/cuda-uninstaller   
sudo nvidia-uninstall

# 如需彻底卸载清理：
https://www.bilibili.com/opus/1131154984017068033?spm_id_from=333.1387.0.0
[https://github.com/guqiong96/Lvllm/issues/5 ](https://github.com/guqiong96/Lvllm/issues/8)
 
# 下载并安装CUDA 12.9 
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run


```

### 2. 创建并激活Python环境及一些系统库

```bash
conda create -n Lvllm python==3.12.11
conda activate Lvllm

# 升级libstdcxx-ng  （避免glibcxx_3.4.32 not found， 新增的lk_moe模块无法加载退回到原始vllm模式，最后显存溢出）
# triton编译时 version `GLIBCXX_3.4.30' not found 
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 安装NUMA库 ubuntu
sudo apt-get install libnuma-dev
# 安装NUMA库 rocky linux
sudo dnf install numactl-devel
```

### 3. 克隆仓库并安装依赖

```bash
# 克隆Lvllm仓库
git clone https://github.com/guqiong96/Lvllm.git


# 安装PyTorch 2.9.1 （可选 Qwen3-VL 需要安装 xformers、torchvision）
pip uninstall torchaudio triton xformers torchvision torch
pip install torchaudio triton torchvision torch==2.9.1 xformers
 
# 50 系列 GPU 之前需要安装 xformers==0.0.33.dev1090 以后版本才能正常运行Qwen3-VL，现在需要确定xformers新版本是否解决了问题
 
 

# 使用现有PyTorch
python use_existing_torch.py

# 安装构建依赖
pip install -r requirements/build.txt
```
 
### 4. 安装Lvllm

```bash 
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv
```
  

MAX_JOBS=32 NVCC_THREADS=1 减少编译时内存占用，避免卡死
CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" 性能选项

## 启动命令

使用以下命令启动Lvllm服务: 
```bash 
LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS="88" OMP_NUM_THREADS="88" LVLLM_MOE_USE_WEIGHT="TO_DTYPE" vllm serve --config ~/Downloads/Lvllm/config.yaml
```
VLLM_ATTENTION_BACKEND="FLASHINFER": 这个环境变量已不是最优选项[2025-10-21]
1、单GPU推理(N)：LK_THREADS、OMP_NUM_THREADS 设置为总的核心数量-4 , 开启超线程则设置为总的线程数量-8
2、多GPU推理(N/GPU数量)：每个GPU的LK_THREADS、OMP_NUM_THREADS 设置为N/(GPU数量)

LVLLM_MOE_USE_WEIGHT="KEEP": lk_moe推理使用权重原始格式，例如fp8_e4m3, 降低推理内存占用

LVLLM_MOE_USE_WEIGHT="TO_DTYPE": lk_moe推理使用配置文件格式float16或bfloat16, 提高fp8模型推理速度， 但会增加内存占用延长加载时间，默认TO_DTYPE
 

### 错误排查
运行以下命令，将错误输出提交至GitHub Issues 
```bash 
python -c "import lk_moe"
```

### 更新已有Lvllm

如果已安装Lvllm，需要更新到最新版本，请执行以下命令：

```bash
# 正常情况
git pull --force
# 如果出现冲突
git fetch origin
git reset --hard origin/main

# 安装PyTorch 2.9.1 （可选 Qwen3-VL 需要安装 xformers、torchvision）
pip uninstall torchaudio triton xformers torchvision torch
pip install torchaudio triton torchvision torch==2.9.1 xformers 

# 50 系列 GPU 之前需要安装 xformers==0.0.33.dev1090 才能正常运行Qwen3-VL，现在需要确定xformers新版本是否解决了问题
 
python use_existing_torch.py 
pip install -r requirements/build.txt
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e . --no-build-isolation -vvv

rm -rf ~/.cache/vllm
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

# 本项目为vLLM的分支，引用了以下开源项目的源代码：

1. **llama.cpp**
   - 项目地址：[https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
   - 用途：GGML相关定义

2. **llamafile**
   - 项目地址：[https://github.com/Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile)
   - 用途：GGUF权重量化、反量化及矩阵乘法


