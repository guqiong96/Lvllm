#!/usr/bin/env bash
# Qwen3.8-Flash-Next (NVFP4) · 2× RTX 5060 Ti (SM120), TP2, plain decode.
#
# Port 8072 is the ONLY service this script may touch.  Never kill by process
# name ("vllm serve" / "VLLM::Worker" match the 3090 production service on
# port 8070 too) and never use GPU index numbers (always UUIDs).

# 1) Take the pid file / port owner, if any.
if [ -f /tmp/opencode/qwen38_8072.pid ]; then
    PID=$(cat /tmp/opencode/qwen38_8072.pid 2>/dev/null)
fi
[ -z "$PID" ] && PID=$(ss -tlnp 2>/dev/null | grep ':8072' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
if [ -n "$PID" ]; then
    # 2) Verify before killing: cmdline must be our 8072 serve, never GPU0/3.
    CMD=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
    case "$CMD" in
        *qwen38_serve_tp2_5060ti.sh*|*vllm\ serve*8072*) ;;
        *) echo "refuse to kill pid=$PID (not our 8072 service): $CMD"; PID=; ;;
    esac
    if [ -n "$PID" ]; then
        kill -TERM "$PID" 2>/dev/null
        # 3) Wait for graceful exit, escalate only on the same pid.
        for i in $(seq 1 30); do
            sleep 2
            kill -0 "$PID" 2>/dev/null || break
        done
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
    fi
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=1,2 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=cutlass \
LK_GPU_PREFILL_SUB_M=32768 \
LK_GPU_PREFILL_D_SLOTS=16384 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=${LK_POOL_SLOTS:-48} \
LK_POOL_DECODE=${LK_POOL_DECODE:-1} \
LK_POOL_SPLIT_PROBE=${LK_POOL_SPLIT_PROBE:-0} \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_PLE_CPU_OFFLOAD=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve ~/Models/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4 \
  --host 0.0.0.0 \
  --port 8072 \
  --tensor-parallel-size 2 \
  --max-model-len 66000 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --tokenizer-mode auto \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --dtype bfloat16 \
  --max-num-seqs 2 \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}'
