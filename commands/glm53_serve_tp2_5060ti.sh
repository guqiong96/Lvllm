#!/usr/bin/env bash
# VRAM note (2026-09-16, measured on 5060Ti x2 TP2): gpu prefill on
#   (LVLLM_GPU_PREFILL_MIN_BATCH_SIZE>0) plus --max-num-batched-tokens 8192 (or more)
#   => CUDA OOM (weights + activations nearly fill the 16G cards). Lower the batch
#   size yourself, or set LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 to disable gpu prefill.
#   Parameters kept at defaults (not tuned).
#   Low-VRAM recipe = LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 and
#   --max-num-batched-tokens 1024 (measured smallest footprint).
# GLM-5.3-Flash-NVFP4 - 2x RTX 5060 Ti (SM120) TP2 - MTP draft
# Status: draft (2026-09-16 audit output, not yet measured). Prereq: DeepGEMM
#   installed, flashinfer with sm120 sparse-MLA decode API.
# Note: on SM120 KV is forced to fp8_ds_mla (do NOT pass --kv-cache-dtype bfloat16);
#   do not set block-size manually (kpool needs the 256 alignment, auto-resolved).
# FlashInfer autotune (TP2, sm120): the generic autotune dummy run can hang in a
#   three-way deadlock (see the plain script's detailed writeup). Root cause is NOT
#   a device-index bug (both TP GPUs are the same 5060Ti sm120) -- it is first-time
#   kernel instantiation landing inside the autotuner's cross-rank sync window.
#   Autotune is left at its DEFAULT (enabled) here. If it hangs at startup, disable
#   it from the command line (NOT baked into this script), either:
#     --kernel-config '{"enable_flashinfer_autotune":false}'   # simplest
#   or, to keep the sparse-MLA decode tuning:
#     VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp4_gemm               # breaks the ring
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
