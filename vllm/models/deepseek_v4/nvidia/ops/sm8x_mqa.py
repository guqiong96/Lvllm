# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80/86 MQA-logits path for the DeepSeek-V4 indexers.

Blueprint: the sglang SM80 final form (patch ``03_dsv4f_sm80_support``),
whose rowwise kernel and block-major cache view are the reference; the
vllm-ds4 module was used only as a layout cross-check. Ampere has no fp8
tensor cores and DeepGEMM ships no sm8x cubins, so the DeepGEMM-only
``fp8_fp4_(paged_)mqa_logits`` entry points route here instead: Q/K are
loaded as uint8 and decoded to f32 in-kernel (``fp8_emulate``), and the dot
runs at ``input_precision="tf32"``. Unlike the pure-torch fallback, the
rowwise kernel only scans each row's live ``context_len``.
"""

import torch

from vllm.model_executor.layers.quantization.utils.fp8_emulate import (
    e4m3fn_u8_to_f32,
)
from vllm.triton_utils import tl, triton


def _view_packed_fp8_paged_mqa_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 values + fp32 scales from indexer cache block storage.

    Block-major (segregated) page contract, matching ``indexer_k_norm_rope_store``:
    page ``b = [ all tokens' fp8 values (block_size*head_dim B) |
    all tokens' fp32 scales (block_size*4 B) | pad ]``.

    The tensor handed in is usually a *nominal* view into a larger shared
    page (the indexer is packed beside the MLA latent pages inside each
    block, so ``stride(0)`` > ``block_size * head_dim_with_scale`` and the
    per-token strides are decorative). Only the storage offset and the
    page stride are real — rebuild both regions with ``as_strided`` from
    the block-major contract instead of reshaping the nominal view
    (sglang fix #5 lesson, extended to packed pages).
    """
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 kv_cache, got {kv_cache.dtype}")
    if kv_cache.dim() == 3:
        num_blocks, block_size = kv_cache.shape[:2]
    elif kv_cache.dim() == 4:
        num_blocks, block_size, num_kv_heads = kv_cache.shape[:3]
        if num_kv_heads != 1:
            raise ValueError(f"Expected one KV head, got {num_kv_heads}")
    else:
        raise ValueError(
            f"Expected 3D or 4D kv_cache, got {kv_cache.dim()} dimensions"
        )
    head_dim_with_scale = kv_cache.shape[-1]
    scale_bytes = head_dim_with_scale - head_dim
    if scale_bytes <= 0 or scale_bytes % torch.float32.itemsize != 0:
        raise ValueError(
            "Expected kv_cache last dimension to contain FP8 values followed "
            f"by fp32 scale bytes; got head_dim={head_dim}, "
            f"last_dim={head_dim_with_scale}"
        )
    page_stride = int(kv_cache.stride(0))
    base = int(kv_cache.storage_offset())
    if page_stride < block_size * (head_dim + scale_bytes):
        raise ValueError(
            "Indexer page stride is smaller than the block-major contract; "
            f"stride(0)={page_stride}, need >="
            f"{block_size * (head_dim + scale_bytes)}"
        )
    if (base + block_size * head_dim) % torch.float32.itemsize != 0:
        raise ValueError(
            "Scale plane is not 4-byte aligned (storage_offset="
            f"{base}, block_size*head_dim={block_size * head_dim})"
        )

    kv_values = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, head_dim),
        stride=(page_stride, head_dim, head_dim, 1),
        storage_offset=base,
    ).view(torch.float8_e4m3fn)
    kv_scale = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, scale_bytes),
        stride=(page_stride, scale_bytes, scale_bytes, 1),
        storage_offset=base + block_size * head_dim,
    ).view(torch.float32)
    return kv_values, kv_scale


@triton.jit
def _fp8_paged_mqa_logits_rowwise_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows: tl.constexpr,
    logits_width: tl.constexpr,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-row paged-MQA logits kernel (SM80/SM86 Ampere).

    From the sglang SM80 final form (which itself traced back to
    vllm-ds4 sm12x_mqa.py). Each program handles one logical row
    (``batch * next_n + q_pos``) across a ``BLOCK_N``-wide window of token
    positions and masks to the row's live ``context_len`` (programs past it
    early-exit with -inf), so decode cost tracks the actual sequence length
    instead of the padded budget. Q/K arrive as uint8 e4m3 bytes and decode
    in-kernel; ``tl.dot`` runs at ``input_precision="tf32"``.
    """
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_rows
    valid_n = offs_local_n < logits_width
    batch = row // next_n
    q_pos = row - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_row,
        other=0,
    )
    if token_start + pid_n * BLOCK_N >= context_len:
        logits = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
        tl.store(
            logits_ptr + row * stride_lm + offs_local_n * stride_ln,
            logits,
            mask=valid_row & valid_n,
        )
        return
    context_mask = valid_n & (offs_n < context_len)

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr + batch * stride_btb + block_rank * stride_btk,
        mask=valid_row & context_mask,
        other=0,
    )

    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h0 in tl.range(0, num_heads, BLOCK_H):
        heads = h0 + tl.arange(0, BLOCK_H)
        valid_h = heads < num_heads
        scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = e4m3fn_u8_to_f32(
                tl.load(
                    q_ptr
                    + batch * stride_qb
                    + q_pos * stride_qn
                    + heads[:, None] * stride_qh
                    + d[None, :] * stride_qd,
                    mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                    other=0,
                )
            )
            k = e4m3fn_u8_to_f32(
                tl.load(
                    kv_ptr
                    + block_idx[None, :] * stride_kvb
                    + block_offset[None, :] * stride_kvs
                    + d[:, None] * stride_kvd,
                    mask=context_mask[None, :] & (d[:, None] < head_dim),
                    other=0,
                )
            )
            scores += tl.dot(q, k, input_precision="tf32")

        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + row * stride_wm + heads * stride_wh,
            mask=valid_row & valid_h,
            other=0.0,
        )
        logits += tl.sum(weighted * weight[:, None], axis=0)

    logits = tl.where(context_mask & valid_row, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_lm + offs_local_n * stride_ln,
        logits,
        mask=valid_row & valid_n,
    )


def fp8_paged_mqa_logits_rowwise_triton(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """SM80/SM86 rowwise paged-MQA logits (Triton, live-context only).

    Pre-condition: ``head_dim % 64 == 0`` and ``num_heads % 4 == 0`` so the
    ``tl.dot`` lands on tensor-core-friendly tiles (DSV4-Flash:
    head_dim=128, num_heads=64).
    """
    batch_size, next_n, num_heads, head_dim = q_fp8.shape
    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kvcache_fp8, head_dim)
    _, block_size, _, _ = kv_values.shape
    # Ampere lacks the fp8e4nv type; the kernel decodes e4m3 from uint8.
    q = q_fp8.view(torch.uint8)
    kv_values = kv_values.view(torch.uint8)
    num_rows = batch_size * next_n
    token_count = max_seq_len
    logits = torch.empty(
        (num_rows, token_count),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    if num_rows == 0 or token_count == 0:
        return logits

    if seq_lens.dim() > 1:
        context_lens = seq_lens.squeeze(-1)
    else:
        context_lens = seq_lens
    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    block_n = 128
    grid = (num_rows, triton.cdiv(token_count, block_n))
    _fp8_paged_mqa_logits_rowwise_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weight,
        context_lens_2d,
        page_table,
        logits,
        0,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weight.stride(0),
        weight.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=64,
        BLOCK_H=8,
        num_warps=4,
    )
    return logits


def _fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Reference semantics (DeepGEMM test_attention), fp8 Q only.

    Not CUDA-graph compatible (``.item()`` per row); used only when the
    rowwise pre-conditions are unmet, for correctness checks.
    """
    batch_size, next_n, num_heads, head_dim = q.shape
    kv_values, kv_scales = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_kv, _, _ = kv_values.shape
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    lens_2d = context_lens.reshape(batch_size, -1)
    if lens_2d.shape[1] == 1 and next_n != 1:
        lens_2d = lens_2d.expand(batch_size, next_n)

    q_f32 = q.float()
    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(lens_2d[batch_idx, next_idx].item())
            if context_len <= 0:
                continue
            q_row = q_f32[batch_idx, next_idx]
            row_weights = weights[row]
            token_ids = torch.arange(context_len, device=q.device, dtype=torch.long)
            logical_blocks = token_ids // block_kv
            token_in_block = token_ids - logical_blocks * block_kv
            physical_blocks = block_tables[batch_idx, logical_blocks]
            kv_chunk = kv_values[physical_blocks, token_in_block, 0].float()
            scale_chunk = kv_scales[physical_blocks, token_in_block, 0].squeeze(-1)
            kv_chunk.mul_(scale_chunk[:, None])
            scores = torch.matmul(q_row, kv_chunk.T)
            scores.relu_()
            scores.mul_(row_weights[:, None])
            logits[row, :context_len] = scores.sum(dim=0)
    return logits


def fp8_paged_mqa_logits_sm8x(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Dispatch paged MQA logits to the rowwise Triton kernel, falling back
    to the torch reference when the shape pre-conditions are unmet."""
    num_heads, head_dim = q.shape[2], q.shape[3]
    if (
        head_dim % 64 == 0
        and num_heads % 4 == 0
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == head_dim + 4
    ):
        return fp8_paged_mqa_logits_rowwise_triton(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    from vllm.logger import init_logger

    init_logger(__name__).warning_once(
        "SM8x paged-MQA falling back to the torch reference path "
        "(q.dim=%s, head_dim=%s, num_heads=%s, kv_cache.dtype=%s, "
        "kv_cache.shape[-1]=%s). Not CUDA-graph compatible; expect a large "
        "per-step latency.",
        q.dim(),
        head_dim,
        num_heads,
        kv_cache.dtype,
        kv_cache.shape[-1] if kv_cache.dim() else None,
    )
    return _fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def fp8_mqa_logits_torch(
    q: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
    max_score_bytes: int = 64 * 1024 * 1024,
) -> torch.Tensor:
    """Reference MQA logits over an unpaged KV buffer (prefill indexer).

    Chunked over heads and keys so the transient ``[H, M, N]`` scores stay
    bounded. sglang has no sm8 non-paged kernel, so prefill runs on this
    torch path until a Triton prefill kernel is added.
    """
    k_f32 = k_values.float()
    k_f32.mul_(k_scales.reshape(-1, 1).float())
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q.shape
    seq_len_kv = k_f32.shape[0]
    logits = torch.zeros(
        (seq_len, seq_len_kv), device=q.device, dtype=torch.float32
    )
    score_elems_per_head = max(1, seq_len * seq_len_kv)
    max_heads = max_score_bytes // max(1, score_elems_per_head * 4)
    head_chunk_size = max(1, min(8, num_heads, max_heads))

    q_f32 = q.float()
    for head_start in range(0, num_heads, head_chunk_size):
        head_end = min(head_start + head_chunk_size, num_heads)
        q_chunk = q_f32[:, head_start:head_end, :].transpose(0, 1).contiguous()
        head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
        score_elems_per_key = max(1, seq_len * (head_end - head_start))
        max_keys = max_score_bytes // max(1, score_elems_per_key * 4)
        k_chunk_size = max(1, min(seq_len_kv, max_keys))
        for k_start in range(0, seq_len_kv, k_chunk_size):
            k_end = min(k_start + k_chunk_size, seq_len_kv)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            logits[:, k_start:k_end].add_(scores.sum(dim=0))

    if clean_logits:
        offsets = torch.arange(seq_len_kv, device=q.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        logits = logits.masked_fill(~valid, float("-inf"))
    return logits


def fp8_mqa_logits_sm8x(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    k_values, k_scales = kv
    return fp8_mqa_logits_torch(
        q,
        k_values,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits,
    )
