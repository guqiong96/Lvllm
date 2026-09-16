#!/usr/bin/env bash
# GLM-5.3-Flash-NVFP4 - 4 GPUs (2x3090 + 2x5060 Ti), TP4 mixed, plain decode (no MTP).
# Mixed-arch: the group capability floor is 8.x, so every rank runs the sm8x
# Triton sparse-MLA backend (bf16 KV) and the kpool block alignment anchors to
# the floor (cuda.py:440). FLASHINFER_CUDA_ARCH_LIST must cover both arches.
# KV budget = smallest rank (16G 5060Ti); keep mml/mnb small (hetero).
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
FLASHINFER_CUDA_ARCH_LIST="8.6 12.0f" \
VLLM_USE_V2_MODEL_RUNNER=1 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=CPU_CORE \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
vllm serve ~/Models/GLM-5.3-Flash-NVFP4 \
  --served-model-name GLM-5.3-Flash \
  --host 0.0.0.0 --port 8070 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce
