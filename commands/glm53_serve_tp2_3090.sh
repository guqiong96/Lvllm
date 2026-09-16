#!/usr/bin/env bash
# GLM-5.3-Flash-NVFP4 - 2x RTX 3090 (SM86) TP2 - plain decode.
# Same GDN hybrid family as Qwen3.8 -> v2 model runner. MoE host-resident via
# lk_moe (NVFP4 routed experts). On sm86 flashinfer autotune is skipped
# (gated to cap>=90), so no fp4_gemm collective; sparse MLA must fall to the
# sm8x Triton path (sm8x_sparse_mla_enabled, floor==8).
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce
