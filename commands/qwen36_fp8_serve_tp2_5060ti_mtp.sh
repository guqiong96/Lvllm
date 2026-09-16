#!/usr/bin/env bash
# Qwen3.6-35B-A3B-FP8 · 2x RTX 5060 Ti (SM120), TP2, MTP3 speculative decode.
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=128 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-1 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
vllm serve /home/guqiong/Models/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 \
    --port 8070 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --tokenizer-mode auto \
    --served-model-name Qwen3.6-35B-A3B-FP8 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 12000 \
    --max-num-seqs 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --disable-custom-all-reduce
