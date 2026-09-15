"""Sink-aware sparse MLA over materialized BF16 KV — SM80/86 path.

Ported from the sglang SM80 work (itself first ported from vllm-ds4), keeping
the sglang patch style: pure Triton, no fp8 types, no CUDA.

Two entry points:
  * ``matmul_sparse_mla_attention_with_sink`` — decode: per-token materialized
    KV ``(T, K, D)``, scores via a batched bmm, sink-aware online reduction in
    Triton. ~3x faster than any shmem-capped tilelang v1 on SM86.
  * ``sparse_mla_fwd_with_sink`` — prefill: shared gathered KV rows indexed
    per query (flash-style online softmax with attn-sink init), matching the
    ``flash_mla_sparse_fwd`` contract (flat row indices, ``topk_length``).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def _npow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


@triton.jit
def _finish_materialized_scores_with_sink_candidate_block_kernel(
    scores_ptr,
    kv_ptr,
    valid_tokens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_scores_t,
    stride_scores_h,
    stride_scores_c,
    stride_kv_t,
    stride_kv_c,
    stride_kv_d,
    stride_valid_t,
    stride_valid_c,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    head_dim: tl.constexpr,
    num_candidates: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    candidate_offsets = tl.arange(0, BLOCK_K)
    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim

    max_score = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    for candidate_start in range(0, num_candidates, BLOCK_K):
        candidates = candidate_start + candidate_offsets
        candidate_mask = candidates < num_candidates
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidates * stride_valid_c,
            mask=candidate_mask,
            other=0,
        ).to(tl.int1)
        scores = tl.load(
            scores_ptr
            + token_idx * stride_scores_t
            + head_idx * stride_scores_h
            + candidates * stride_scores_c,
            mask=candidate_mask & is_valid,
            other=-float("inf"),
        ).to(tl.float32)
        max_score = tl.maximum(max_score, tl.max(scores, axis=0))

    denom = tl.exp(tl.load(attn_sink_ptr + head_idx).to(tl.float32) - max_score)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for candidate_start in range(0, num_candidates, BLOCK_K):
        candidates = candidate_start + candidate_offsets
        candidate_mask = candidates < num_candidates
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidates * stride_valid_c,
            mask=candidate_mask,
            other=0,
        ).to(tl.int1)
        scores = tl.load(
            scores_ptr
            + token_idx * stride_scores_t
            + head_idx * stride_scores_h
            + candidates * stride_scores_c,
            mask=candidate_mask & is_valid,
            other=-float("inf"),
        ).to(tl.float32)
        weights = tl.exp(scores - max_score)
        denom += tl.sum(weights, axis=0)
        kv = tl.load(
            kv_ptr
            + token_idx * stride_kv_t
            + candidates[:, None] * stride_kv_c
            + dim_offsets[None, :] * stride_kv_d,
            mask=(candidate_mask & is_valid)[:, None] & dim_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(kv.to(tl.float32) * weights[:, None], axis=0)

    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_idx * stride_out_h
        + dim_offsets * stride_out_d,
        acc / denom,
        mask=dim_mask,
    )


@triton.jit
def _finish_materialized_scores_with_sink_kernel(
    scores_ptr,
    kv_ptr,
    valid_tokens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_scores_t,
    stride_scores_h,
    stride_scores_c,
    stride_kv_t,
    stride_kv_c,
    stride_kv_d,
    stride_valid_t,
    stride_valid_c,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    running_max = tl.load(attn_sink_ptr + head_offsets, mask=head_mask, other=0.0).to(
        tl.float32
    )
    running_denom = tl.full((HEAD_BLOCK,), 1.0, tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    for candidate_idx in range(0, num_candidates):
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidate_idx * stride_valid_c
        )
        if is_valid:
            score = tl.load(
                scores_ptr
                + token_idx * stride_scores_t
                + head_offsets * stride_scores_h
                + candidate_idx * stride_scores_c,
                mask=head_mask,
                other=-float("inf"),
            ).to(tl.float32)
            kv = tl.load(
                kv_ptr
                + token_idx * stride_kv_t
                + candidate_idx * stride_kv_c
                + dim_offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    result = running_acc / running_denom[:, None]
    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_offsets[:, None] * stride_out_h
        + dim_offsets[None, :] * stride_out_d,
        result,
        mask=matrix_mask,
    )


def finish_materialized_sparse_mla_scores_with_sink(
    scores: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    num_heads: int | None = None,
    head_block_size: int = 1,
    candidate_block_size: int | None = None,
) -> None:
    num_tokens, _, num_candidates = scores.shape
    active_heads = num_heads if num_heads is not None else output.shape[1]
    head_dim = kv.shape[2]
    block_d = 512 if head_dim <= 512 else _npow2(head_dim)

    if candidate_block_size is not None:
        grid = (num_tokens, active_heads, triton.cdiv(head_dim, block_d))
        _finish_materialized_scores_with_sink_candidate_block_kernel[grid](
            scores,
            kv,
            valid_tokens,
            attn_sink,
            output,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
            valid_tokens.stride(0),
            valid_tokens.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            head_dim,
            num_candidates,
            BLOCK_K=candidate_block_size,
            BLOCK_D=block_d,
            num_warps=8,
        )
    else:
        grid = (num_tokens, triton.cdiv(active_heads, head_block_size))
        _finish_materialized_scores_with_sink_kernel[grid](
            scores,
            kv,
            valid_tokens,
            attn_sink,
            output,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
            valid_tokens.stride(0),
            valid_tokens.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            active_heads,
            head_dim,
            num_candidates,
            HEAD_BLOCK=head_block_size,
            BLOCK_D=block_d,
            num_warps=8,
        )
    if output.shape[1] > active_heads:
        output[:, active_heads:].zero_()


def matmul_sparse_mla_attention_with_sink(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    num_heads: int | None = None,
    score_buffer: torch.Tensor | None = None,
    head_block_size: int = 1,
    candidate_block_size: int | None = None,
) -> None:
    """Sink-aware sparse MLA over materialized BF16 KV ``(T, K, D)``."""
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    active_heads = num_heads if num_heads is not None else output.shape[1]
    num_tokens = q.shape[0]
    num_candidates = kv.shape[1]
    if score_buffer is None:
        score_buffer = torch.empty(
            (num_tokens, active_heads, num_candidates),
            dtype=torch.float32,
            device=q.device,
        )
    torch.bmm(
        q[:, :active_heads].float(),
        kv.float().transpose(1, 2),
        out=score_buffer,
    )
    score_buffer.mul_(scale)
    finish_materialized_sparse_mla_scores_with_sink(
        score_buffer,
        kv,
        valid_tokens,
        attn_sink,
        output,
        num_heads=active_heads,
        head_block_size=head_block_size,
        candidate_block_size=candidate_block_size,
    )


@triton.jit
def _sparse_mla_fwd_with_sink_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_r,
    stride_kv_d,
    stride_idx_t,
    stride_idx_k,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    qk_scale,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program per (query, head); online softmax over gathered rows.
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_idx * stride_q_h
        + dim_offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    length = tl.load(lens_ptr + token_idx)
    running_max = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    running_denom = 1.0
    running_acc = tl.zeros((BLOCK_D,), tl.float32)

    for k_start in range(0, topk, BLOCK_K):
        candidates = k_start + tl.arange(0, BLOCK_K)
        k_mask = (candidates < length) & (candidates < topk)
        row = tl.load(
            indices_ptr + token_idx * stride_idx_t + candidates * stride_idx_k,
            mask=k_mask,
            other=0,
        ).to(tl.int64)
        row_safe = tl.where(k_mask, tl.maximum(row, 0), 0)
        kv = tl.load(
            kv_ptr + row_safe[:, None] * stride_kv_r + dim_offsets[None, :] * stride_kv_d,
            mask=k_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(kv * q[None, :], axis=1) * qk_scale
        scores = tl.where(k_mask, scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=0))
        previous_weight = tl.exp(running_max - next_max)
        weights = tl.exp(scores - next_max)
        running_acc = running_acc * previous_weight + tl.sum(
            kv * weights[:, None], axis=0
        )
        running_denom = running_denom * previous_weight + tl.sum(weights, axis=0)
        running_max = next_max

    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_idx * stride_out_h
        + dim_offsets * stride_out_d,
        running_acc / running_denom,
        mask=dim_mask,
    )


def sparse_mla_fwd_with_sink(
    q: torch.Tensor,  # (Tq, H, D)
    kv: torch.Tensor,  # (R, D) gathered rows shared by all queries
    indices: torch.Tensor,  # (Tq, K) int32 row ids into kv (may be <0 beyond lens)
    topk_length: torch.Tensor,  # (Tq,) int32 valid prefix per query
    scale: float,
    attn_sink: torch.Tensor,  # (H,)
    output: torch.Tensor,  # (Tq, H, D)
    num_heads: int | None = None,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if output.dim() == 4:
        assert output.shape[1] == 1
        output = output[:, 0]
    active_heads = num_heads if num_heads is not None else q.shape[1]
    num_tokens = q.shape[0]
    topk = indices.shape[-1]
    head_dim = q.shape[-1]
    block_d = 512 if head_dim <= 512 else _npow2(head_dim)
    _sparse_mla_fwd_with_sink_kernel[(num_tokens, active_heads)](
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        scale,
        active_heads,
        head_dim,
        topk,
        BLOCK_K=16,
        BLOCK_D=block_d,
        num_warps=8,
    )
    if q.shape[1] > active_heads:
        output[:, active_heads:].zero_()
