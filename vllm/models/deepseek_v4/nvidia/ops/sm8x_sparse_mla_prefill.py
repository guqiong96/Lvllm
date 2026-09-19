# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiled Triton sparse-MLA prefill for the SM8x route.

The per-(token, head) kernel in ``common/ops/sparse_mla_kernels`` re-reads
every gathered KV row once per query and per head, which on Ampere makes the
prefill attention bandwidth-bound. This port of the vllm-ds4 formulation
tiles the work instead: each program owns one query x ``HEAD_BLOCK`` heads, so
each KV row is loaded once per 8 heads, and the fp32 online-softmax state
(``max_score`` / ``denom`` / ``acc``) is chunked over queries and topk
candidates to keep it in L2.

Contract matches ``sparse_mla_fwd_with_sink`` exactly: flat row ids into the
gathered KV, valid entries first then ``-1`` padding, per-query ``lens``,
attn-sink merged at the end with zero value.
"""

import os

import torch
import triton
import triton.language as tl

# Same defaults as the vllm-ds4 Triton prefill path
# (VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE / TOPK_CHUNK_SIZE).
QUERY_CHUNK_SIZE = 256
TOPK_CHUNK_SIZE = 512
HEAD_BLOCK = 8


def sm8x_tiled_prefill_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_SM8X_TILED_PREFILL", "1") == "1"


@triton.jit
def _accumulate_multihead_kernel(
    q_ptr,
    kv_flat_ptr,
    indices_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_d: tl.constexpr,
    stride_indices_t,
    stride_indices_c: tl.constexpr,
    stride_state_t,
    stride_state_h,
    stride_acc_t,
    stride_acc_h,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    HEAD_BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK_T + tl.arange(0, HEAD_BLOCK_T)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    state_base = token_idx * stride_state_t
    running_max = tl.load(
        max_score_ptr + state_base + head_offsets * stride_state_h,
        mask=head_mask,
        other=float("-inf"),
    )
    running_denom = tl.load(
        denom_ptr + state_base + head_offsets * stride_state_h,
        mask=head_mask,
        other=0.0,
    )
    acc_base = token_idx * stride_acc_t
    running_acc = tl.load(
        acc_ptr
        + acc_base
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    valid_len = tl.load(lens_ptr + token_idx)
    local_eff = tl.minimum(
        num_candidates,
        tl.maximum(valid_len - candidate_offset, 0),
    )

    for candidate_idx in range(0, local_eff):
        kv_index = tl.load(
            indices_ptr
            + token_idx * stride_indices_t
            + candidate_idx * stride_indices_c
        )
        if kv_index >= 0:
            kv = tl.load(
                kv_flat_ptr + kv_index.to(tl.int64) * stride_kv_t + dim_offsets,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, scores)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(scores - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(
        max_score_ptr + state_base + head_offsets * stride_state_h,
        running_max,
        mask=head_mask,
    )
    tl.store(
        denom_ptr + state_base + head_offsets * stride_state_h,
        running_denom,
        mask=head_mask,
    )
    tl.store(
        acc_ptr
        + acc_base
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        running_acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _accumulate_single_kernel(
    q_ptr,
    kv_flat_ptr,
    indices_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_d: tl.constexpr,
    stride_indices_t,
    stride_indices_c: tl.constexpr,
    stride_state_t,
    stride_state_h,
    stride_acc_t,
    stride_acc_h,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr + token_idx * stride_q_t + head_idx * stride_q_h + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t + head_idx * stride_acc_h + dim_offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(
        tl.float32
    )

    valid_len = tl.load(lens_ptr + token_idx)
    local_eff = tl.minimum(
        num_candidates,
        tl.maximum(valid_len - candidate_offset, 0),
    )

    for candidate_idx in range(0, local_eff):
        kv_index = tl.load(
            indices_ptr
            + token_idx * stride_indices_t
            + candidate_idx * stride_indices_c
        )
        if kv_index >= 0:
            kv = tl.load(
                kv_flat_ptr + kv_index.to(tl.int64) * stride_kv_t + dim_offsets,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


@triton.jit
def _finish_with_sink_kernel(
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    sink_ptr,
    output_ptr,
    stride_state_t,
    stride_state_h,
    stride_acc_t,
    stride_acc_h,
    stride_acc_d: tl.constexpr,
    stride_output_t,
    stride_output_h,
    stride_output_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    sink = tl.load(sink_ptr + head_idx).to(tl.float32)
    has_tokens = running_denom > 0.0
    has_sink = sink > -float("inf")
    valid_max = tl.where(has_tokens, running_max, -float("inf"))
    valid_sink = tl.where(has_sink, sink, -float("inf"))
    merge_max = tl.maximum(valid_max, valid_sink)
    has_any = has_tokens | has_sink
    safe_merge_max = tl.where(has_any, merge_max, 0.0)
    safe_running_max = tl.where(has_tokens, running_max, safe_merge_max)
    safe_sink = tl.where(has_sink, sink, safe_merge_max)
    subset_scale = tl.where(has_tokens, tl.exp(safe_running_max - safe_merge_max), 0.0)
    subset_weight = running_denom * subset_scale
    sink_weight = tl.where(has_sink, tl.exp(safe_sink - safe_merge_max), 0.0)
    total_weight = subset_weight + sink_weight
    inv_total = tl.where(total_weight > 0.0, 1.0 / total_weight, 0.0)

    acc_values = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + offsets * stride_acc_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    acc_values = tl.where(has_tokens, acc_values, 0.0)
    output = acc_values * subset_scale * inv_total
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        output,
        mask=dim_mask,
    )


_state_buffers: dict[
    tuple[torch.device, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _get_state_buffers(
    device: torch.device, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, num_heads, head_dim)
    buffers = _state_buffers.get(key)
    if buffers is None:
        buffers = (
            torch.empty(
                QUERY_CHUNK_SIZE, num_heads, dtype=torch.float32, device=device
            ),
            torch.empty(
                QUERY_CHUNK_SIZE, num_heads, dtype=torch.float32, device=device
            ),
            torch.empty(
                QUERY_CHUNK_SIZE,
                num_heads,
                head_dim,
                dtype=torch.float32,
                device=device,
            ),
        )
        _state_buffers[key] = buffers
    return buffers


def sm8x_tiled_sparse_mla_prefill(
    q: torch.Tensor,  # (Tq, H_pad, D)
    kv_flat: torch.Tensor,  # (R, D) gathered rows
    indices: torch.Tensor,  # (Tq, K) int32 flat row ids, valid first, -1 padding
    lens: torch.Tensor,  # (Tq,) int32 valid count per query
    scale: float,
    attn_sink: torch.Tensor,  # (H,)
    output: torch.Tensor,  # (Tq, H_pad, D)
    num_heads: int,
) -> None:
    num_tokens = q.shape[0]
    head_dim = q.shape[-1]
    num_candidates = indices.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    max_score, denom, acc = _get_state_buffers(q.device, num_heads, head_dim)

    for token_start in range(0, num_tokens, QUERY_CHUNK_SIZE):
        token_end = min(token_start + QUERY_CHUNK_SIZE, num_tokens)
        q_chunk = q[token_start:token_end]
        indices_chunk = indices[token_start:token_end]
        lens_chunk = lens[token_start:token_end]
        num_rows = token_end - token_start
        state_max = max_score[:num_rows]
        state_denom = denom[:num_rows]
        state_acc = acc[:num_rows]
        state_max.fill_(float("-inf"))
        state_denom.zero_()
        state_acc.zero_()

        for candidate_start in range(0, num_candidates, TOPK_CHUNK_SIZE):
            candidate_end = min(candidate_start + TOPK_CHUNK_SIZE, num_candidates)
            candidates = indices_chunk[:, candidate_start:candidate_end]
            if num_heads >= HEAD_BLOCK:
                grid = (num_rows, triton.cdiv(num_heads, HEAD_BLOCK))
                _accumulate_multihead_kernel[grid](
                    q_chunk,
                    kv_flat,
                    candidates,
                    lens_chunk,
                    state_max,
                    state_denom,
                    state_acc,
                    q_chunk.stride(0),
                    q_chunk.stride(1),
                    q_chunk.stride(2),
                    kv_flat.stride(0),
                    kv_flat.stride(1),
                    candidates.stride(0),
                    candidates.stride(1),
                    state_max.stride(0),
                    state_max.stride(1),
                    state_acc.stride(0),
                    state_acc.stride(1),
                    state_acc.stride(2),
                    num_heads,
                    head_dim,
                    candidate_end - candidate_start,
                    candidate_start,
                    scale,
                    HEAD_BLOCK_T=HEAD_BLOCK,
                    BLOCK_D=block_d,
                    num_warps=8,
                )
            else:
                grid = (num_rows, num_heads)
                _accumulate_single_kernel[grid](
                    q_chunk,
                    kv_flat,
                    candidates,
                    lens_chunk,
                    state_max,
                    state_denom,
                    state_acc,
                    q_chunk.stride(0),
                    q_chunk.stride(1),
                    q_chunk.stride(2),
                    kv_flat.stride(0),
                    kv_flat.stride(1),
                    candidates.stride(0),
                    candidates.stride(1),
                    state_max.stride(0),
                    state_max.stride(1),
                    state_acc.stride(0),
                    state_acc.stride(1),
                    state_acc.stride(2),
                    num_heads,
                    head_dim,
                    candidate_end - candidate_start,
                    candidate_start,
                    scale,
                    BLOCK_D=block_d,
                    num_warps=8,
                )

        out_chunk = output[token_start:token_end]
        finish_block_d = min(128, triton.next_power_of_2(head_dim))
        grid = (num_rows * num_heads, triton.cdiv(head_dim, finish_block_d))
        _finish_with_sink_kernel[grid](
            state_max,
            state_denom,
            state_acc,
            attn_sink,
            out_chunk,
            state_max.stride(0),
            state_max.stride(1),
            state_acc.stride(0),
            state_acc.stride(1),
            state_acc.stride(2),
            out_chunk.stride(0),
            out_chunk.stride(1),
            out_chunk.stride(2),
            num_heads,
            head_dim,
            BLOCK_D=finish_block_d,
            num_warps=4,
        )

    if q.shape[1] > num_heads:
        output[:, num_heads:].zero_()
