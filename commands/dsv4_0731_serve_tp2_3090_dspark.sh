#!/usr/bin/env bash
# DeepSeek-V4-Flash (0731) · 2× RTX 3090 (SM86), TP2, dspark.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=CPU_CORE \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
vllm serve ~/Downloads/DeepSeek-V4-Flash-0731 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_ds_mla \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --disable-custom-all-reduce \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --host 0.0.0.0 --port 8070
