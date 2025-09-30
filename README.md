## LvLLM 2025-09-30 混合推理MOE大模型！！！ 

# 已验证：Qwen3-Next-80B-A3B-Instruct

--单任务：

<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/c37da729-a692-4b20-b7f5-b7798acd22c4" />

--并发任务：

  任务一：
  
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/feeb26d2-fc3d-40f5-bc70-3b32ee42b2b3" />

  任务二：
  
<img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/a6abf4f8-58df-47a8-8532-373a94c1b4b2" />

# 当前限制：
1、仅支持dtype: "bfloat16"

2、仅支持compilation_config.cudagraph_mode: "NONE"

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
pip install -r requirements/build.txt
```

### 4. 克隆第三方依赖库(可选,github网络好可以直接第5步)

```bash
cd ~/Downloads/Lvllm/.deps

git clone https://github.com/NVIDIA/cutlass.git cutlass-src
git clone https://github.com/oneapi-src/oneDNN.git oneDNN-src
git clone https://github.com/InternLM/flashmla.git flashmla-src
git clone https://github.com/vllm-project/vllm-flash-attn.git vllm-flash-attn-src

# 安装指定版本的llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git llama.cpp-src
cd llama.cpp-src
git checkout a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
cd ..

# 安装flashinfer-python(可选)
pip install flashinfer-python 
```

### 5. 安装Lvllm

```bash
cd ~/Downloads/Lvllm
MAX_JOBS=32 NVCC_THREADS=1 pip install -e . --no-build-isolation -MAX_JOBS=32 NVCC_THREADS=1 vv
```

## 启动命令

使用以下命令启动Lvllm服务(需要去修改config.yaml里面配置参数）:

```bash
LK_THREADS="88" OMP_NUM_THREADS="88"  vllm serve --config ~/Downloads/Lvllm/config.yaml
```
# 使用flashinfer-python(可选)
```bash 
LK_THREADS="88" OMP_NUM_THREADS="88" VLLM_ATTENTION_BACKEND="FLASHINFER" vllm serve --config ~/Downloads/Lvllm/config.yaml
```

### 配置说明

配置文件 `config.yaml` 包含以下主要参数：

- `model`: 模型路径 (`/Downloads/Qwen3-Next-80B-A3B-Instruct`)
- `host`: 主机地址 (`0.0.0.0`，表示监听所有IPv4地址)
- `port`: 服务端口 (`8070`)
- `tensor-parallel-size`: 张量并行大小 (`1`)
- `max-model-len`: 最大模型序列长度 (`10000`)
- `gpu-memory-utilization`: GPU内存利用率 (`0.8`)
- `max-num-seqs`: 最大并发序列数 (`4`)
- `trust-remote-code`: 信任远程代码 (`true`)
- `swap-space`: 交换空间大小 (GB) (`4`)
- `dtype`: 数据类型 (`bfloat16`)
- `enable_prefix_caching`: 启用前缀缓存 (`true`)
- `enable-chunked-prefill`: 启用分块预填充 (`true`)
- `max_num_batched_tokens`: 最大批处理令牌数 (`1024`)

根据实际环境需求，可以修改配置文件中的参数或调整环境变量值。

![07e0fccaf80ecd2e9bc8476bb5b95514](https://github.com/user-attachments/assets/2178f29c-e7eb-4f8c-8649-b16d545c3fcc)
