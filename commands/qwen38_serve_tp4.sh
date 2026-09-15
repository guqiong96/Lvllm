#!/usr/bin/env bash
# Qwen3.8-Flash-Next (NVFP4) · 4 GPUs (2×3090 + 2×5060 Ti), TP4, plain decode.
# Mixed-arch host: FLASHINFER_CUDA_ARCH_LIST must cover every rank's arch.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=CPU_CORE \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LVLLM_EMBEDDING_NUMA_ENABLED=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
FLASHINFER_CUDA_ARCH_LIST="8.6 12.0f" \
vllm serve ~/Models/Qwen3.8-Flash-Next-NVFP4 \
  --tensor-parallel-size 4 \
  --max-model-len 128000 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-prefix-caching --enable-chunked-prefill \
  --served-model-name Qwen3.8-Flash-Next \
  --trust-remote-code
