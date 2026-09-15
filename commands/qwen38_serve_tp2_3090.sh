#!/usr/bin/env bash
# Qwen3.8-Flash-Next (NVFP4) · 2× RTX 3090 (SM86), TP2, plain decode.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,3 \
VLLM_PLE_CPU_OFFLOAD=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=CPU_CORE \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve ~/Models/Qwen3.8-Flash-Next-NVFP4 \
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
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --disable-custom-all-reduce
