#!/usr/bin/env bash
# DeepSeek-V4-Flash (0731) · 4 GPUs (2×3090 + 2×5060 Ti), TP4, plain decode.
# FLASHINFER_CUDA_ARCH_LIST must cover every rank's arch (mixed-arch host).
# ⚠ SM8x ranks on the V4 path are NOT yet validated in this tree (mHC fix
# landed, but the fp8 blockwise DeepGEMM gate answers per-rank ⇒ mixed
# groups can still fork the SF layout). Homogeneous runs are the tested ones.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
FLASHINFER_CUDA_ARCH_LIST="8.6 12.0f" \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=CPU_CORE \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
vllm serve ~/Downloads/DeepSeek-V4-Flash-0731 \
  --tensor-parallel-size 4 \
  --max-model-len 320000 \
  --max-num-batched-tokens 8192 \
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
  --disable-custom-all-reduce \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --host 0.0.0.0 --port 8070
