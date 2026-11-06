#!/usr/bin/env bash
# GLM-5.3-Flash-NVFP4 - 2x RTX 3090 (SM86) TP2 - plain decode.
# Same GDN hybrid family as Qwen3.8 -> v2 model runner. MoE host-resident via
# lk_moe (NVFP4 routed experts). On sm86 flashinfer autotune is skipped
# (gated to cap>=90), so no fp4_gemm collective; sparse MLA must fall to the
# sm8x Triton path (sm8x_sparse_mla_enabled, floor==8).
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
LK_GPU_PREFILL_SUB_M=8192 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_POOL_SLOTS=12 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
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
  --enable-chunked-prefill
