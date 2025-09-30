## LvLLM 2025-09-30 vllm使用numa加速moe模型！！！ 

当前限制：
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
conda create -n vllm python==3.12.11
conda activate vllm
```

### 3. 克隆仓库并安装依赖

```bash
# 克隆vLLM仓库
git clone https://github.com/vllm-project/vllm.git

# 使用现有PyTorch
python use_existing_torch.py

# 安装构建依赖
pip install -r requirements/build.txt
```

### 4. 克隆第三方依赖库(可选,github网络好可以直接第5步)

```bash
cd ~/Downloads/vllm/.deps

git clone https://github.com/NVIDIA/cutlass.git cutlass-src
git clone https://github.com/oneapi-src/oneDNN.git oneDNN-src
git clone https://github.com/InternLM/flashmla.git flashmla-src
git clone https://github.com/vllm-project/vllm-flash-attn.git vllm-flash-attn-src

# 安装指定版本的llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git llama.cpp-src
cd llama.cpp-src
git checkout a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
cd ..
```

### 5. 安装vLLM

```bash
cd ~/Downloads/vllm
pip install -e . --no-build-isolation -vvv
```

## 启动命令

使用以下命令启动vLLM服务:

```bash
LK_THREADS="88" OMP_NUM_THREADS="88" VLLM_ATTENTION_BACKEND="FLASHINFER" TORCH_COMPILE_DISABLE="1" vllm serve --config ~/Downloads/vllm/config.yaml
```

### 配置说明

配置文件 `config.yaml` 包含以下主要参数：

- `model`: 模型路径 (`/Downloads/Qwen3-Next-80B-A3B-Instruct`)
- `host`: 主机地址 (`::`，表示监听所有IPv4和IPv6地址)
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