#!/usr/bin/env bash
# v4flash 重复输出 A/B：sm120 纯 5060Ti TP2 起服（两版同参数，仅引擎版本不同）
# 用法: VERSION=2.5.0|2.3.10 MAXLEN=100000 bash dsv4_ab_serve_tp2_sm120.sh
set -euo pipefail
VERSION="${VERSION:-2.5.0}"
MAXLEN="${MAXLEN:-100000}"
MODEL=/home/guqiong/Downloads/DeepSeek-V4-Flash-0731

if [ "$VERSION" = "2.3.10" ]; then
  VLLM_BIN=/home/guqiong/venvs/lv2310/bin/vllm
  EXTRA=()
else
  VLLM_BIN=/home/guqiong/.conda/envs/Lvllm/bin/vllm
  EXTRA=()
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS=24 OMP_NUM_THREADS=1 LK_THREAD_BINDING=CPU_CORE \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 LK_POWER_SAVING=1 \
"$VLLM_BIN" serve "$MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_ds_mla \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --enable-prefix-caching --enable-chunked-prefill --enable-auto-tool-choice --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --port 8070 "${EXTRA[@]}"
