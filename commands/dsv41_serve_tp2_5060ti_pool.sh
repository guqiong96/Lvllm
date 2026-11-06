#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 · 2× RTX 5060 Ti (SM120), TP2, pool decode (GPU half).
# Test-only script: GPU1/2, port 8072 (the 3090 GPU0/3 :8070 service is untouched).

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
CUDA_VISIBLE_DEVICES=1,2 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=cutlass \
LK_GPU_PREFILL_SUB_M=8192 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_POOL_SLOTS=8 \
LK_POOL_DECODE=${LK_POOL_DECODE:-1} \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_ENGRAM_DROP_PAGE_CACHE=0 \
vllm serve ~/Models/DeepSeek-V4.1-Flash-1017 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_ds_mla \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 --port 8072
