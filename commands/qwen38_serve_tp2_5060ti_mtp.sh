#!/usr/bin/env bash
# Qwen3.8-Flash-Next (NVFP4) · 2× RTX 5060 Ti (SM120), TP2, MTP speculative decode.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=1,2 \
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
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=16 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_PLE_CPU_OFFLOAD=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve ~/Models/Qwen3.8-Flash-Next-NVFP4-W4A16-4o6-FP8 \
  --host 0.0.0.0 \
  --port 8070 \
  --tensor-parallel-size 2 \
  --max-model-len 66000 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --tokenizer-mode auto \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --dtype bfloat16 \
  --max-num-seqs 2 \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
