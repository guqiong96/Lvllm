#!/usr/bin/env bash
# DeepSeek-V4.1-Flash · 4 GPUs (2×3090 + 2×5060 Ti), TP4, dspark.
# ⚠ SM8x ranks on the V4 path are NOT yet validated in this tree (mHC fix
# landed, but the fp8 blockwise DeepGEMM gate answers per-rank ⇒ mixed
# groups can still fork the SF layout). Homogeneous runs are the tested ones.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=cutlass \
LK_GPU_PREFILL_SUB_M=4096 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=16 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve ~/Models/DeepSeek-V4.1-Flash-1017 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_ds_mla \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 --port 8070
