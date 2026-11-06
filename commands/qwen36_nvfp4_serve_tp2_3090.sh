#!/usr/bin/env bash
# Qwen3.6-35B-A3B-NVFP4 · 2x RTX 3090 (SM86), TP2, plain decode.
CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=0,3 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=cutlass \
LK_GPU_PREFILL_SUB_M=32768 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS=0-1,38-39 \
LK_POOL_SLOTS=128 \
LK_POOL_LOG=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_PLE_CPU_OFFLOAD=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve /home/guqiong/Models/Qwen3.6-35B-A3B-NVFP4 \
    --host 0.0.0.0 \
    --port 8070\
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Qwen3.6-35B-A3B-NVFP4 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 --allowed-origins '["*"]'

