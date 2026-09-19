#!/usr/bin/env bash
# Qwen3.8-Flash-Next (NVFP4) · 4 GPUs (2×3090 + 2×5060 Ti), TP4, MTP speculative decode.

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
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0 \
vllm serve ~/Models/Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 \
  --port 8070 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --served-model-name Qwen3.8-Flash-Next \
  --trust-remote-code
