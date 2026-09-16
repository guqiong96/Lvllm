#!/usr/bin/env bash
# VRAM note (2026-09-16, measured on 5060Ti x2 TP2): gpu prefill on
#   (LVLLM_GPU_PREFILL_MIN_BATCH_SIZE>0) plus --max-num-batched-tokens 8192 (or more)
#   => CUDA OOM (weights + activations nearly fill the 16G cards). Lower the batch
#   size yourself, or set LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 to disable gpu prefill.
#   Parameters kept at defaults (not tuned).
# GLM-5.3-Flash-NVFP4 - 2x RTX 5060 Ti (SM120) TP2 - plain (no MTP)
# Status: functional first run. Prereq: DeepGEMM installed, flashinfer with sm120
#   sparse-MLA decode API.
# Note: on SM120 KV is forced to fp8_ds_mla (do NOT pass --kv-cache-dtype bfloat16);
#   do not set block-size manually (kpool needs the 256 alignment, auto-resolved).
# FlashInfer autotune (TP2, sm120): the generic autotune dummy run hangs in a
#   three-way deadlock (seen ab06/ab07). Root cause is NOT a device-index/gating
#   bug (both TP GPUs are the same 5060Ti sm120) -- it is first-time kernel
#   instantiation landing inside the autotuner's cross-rank synchronized window:
#   rank0 pre-warms the leader-only sparse-MLA decode autotune, then the generic
#   dummy's first-touch of the sparse-MLA *prefill* cold kernel (CUDA lazy-load,
#   never exercised: FULL_DECODE_ONLY capture + skip_attn profiling) skews against
#   rank1's fp4_gemm choose_one all_reduce -> ring. Autotune is left at its
#   DEFAULT (enabled) here; decode is CPU-MoE-bound so autotune gives NO gain
#   (measured 20.6 vs 20.7 t/s). If it hangs on that ring at startup, disable
#   it from the command line (NOT baked into this script), either:
#     --kernel-config '{"enable_flashinfer_autotune":false}'   # simplest
#   or, to keep the sparse-MLA decode tuning:
#     VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp4_gemm               # breaks the ring,
#                                                              # verified -> serve, same t/s
#   The kpool prefill write/seed Triton JITs are separately pre-compiled via the
#   JIT-warmup registry (glm5next Indexer), closing that part of the window.
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce
