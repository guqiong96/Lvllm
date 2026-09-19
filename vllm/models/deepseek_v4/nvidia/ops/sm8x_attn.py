"""SM80/86 sparse-MLA attention path for DeepSeek-V4 / V4.1.

sglang-patch style: pure Triton, uint8 byte-decode via ``fp8_emulate`` (no
``fp8e4nv`` type anywhere), no CUDA sources. Mirrors the sglang final form:
single-pass combined dequant writing gathered ``(T, K, 512)`` bf16 with a
fused valid mask, then ``matmul_sparse_mla_attention_with_sink`` (bmm + Triton
sink finish), ~3x faster than a shmem-capped tilelang v1 on SM86.

Index spaces match the FlashMLA path (all indices are physical global slots
``phys_block * block_size + pos_in_block``; ``-1`` = invalid, clamped to slot 0
and excluded by the valid mask).
"""

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.utils.fp8_emulate import (
    e4m3fn_u8_to_f32,
)
from vllm.models.deepseek_v4.common.ops.sparse_mla_kernels import (
    matmul_sparse_mla_attention_with_sink,
)

from ._sm8x_guard import guard_slot_table

# Read from inside @triton.jit bodies, so they must be constexpr globals.
_FP8_DIM = tl.constexpr(448)
_TOKEN_BYTES = 576  # 448 fp8 + 64*2 bf16 rope
_SCALE_BYTES = 8
_QUANT_BLOCK = tl.constexpr(64)  # 7 e8m0 scale bytes per token


@triton.jit
def _combine_dequant_e8m0_fp8_row(
    addr_data,
    addr_scale,
    out_row_ptr,
    fp8_dim: tl.constexpr,
    quant_block: tl.constexpr,
):
    for qb in tl.static_range(fp8_dim // quant_block):
        offs = qb * quant_block + tl.arange(0, quant_block)
        raw = tl.load(addr_data + offs)
        vals = e4m3fn_u8_to_f32(raw)
        scale = tl.exp2(tl.load(addr_scale + qb).to(tl.float32) - 127.0)  # ue8m0
        tl.store(out_row_ptr + offs, (vals * scale).to(tl.bfloat16))
    # rope bf16 tail: [448:576] bytes
    d_offs = tl.arange(0, 64)
    rope = tl.load((addr_data + fp8_dim).to(tl.pointer_type(tl.bfloat16)) + d_offs)
    tl.store(out_row_ptr + fp8_dim + d_offs, rope)


@triton.jit
def _dequantize_combined_slots_kernel(
    out_ptr,  # (T, K_TOTAL, 512) bf16
    valid_ptr,  # (T, K_TOTAL) uint8
    swa_idx_ptr,
    swa_idx_stride,
    swa_lens_ptr,
    swa_cache_ptr,
    comp_idx_ptr,
    comp_idx_stride,
    comp_lens_ptr,
    comp_cache_ptr,
    swa_block_size,
    swa_block_stride,  # bytes per block (padded)
    comp_block_size,
    comp_block_stride,
    k_swa: tl.constexpr,
    k_comp: tl.constexpr,
    has_comp: tl.constexpr,
    write_valid: tl.constexpr,
    packed_584: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.program_id(1)
    out_row = out_ptr + (token_idx.to(tl.int64) * (k_swa + k_comp) + slot_idx) * 512

    if slot_idx < k_swa:
        slot = tl.load(swa_idx_ptr + token_idx * swa_idx_stride + slot_idx)
        valid = slot_idx < tl.load(swa_lens_ptr + token_idx)
        bs = swa_block_size
        stride = swa_block_stride
        cptr = swa_cache_ptr
    else:
        if has_comp:
            j = slot_idx - k_swa
            slot = tl.load(comp_idx_ptr + token_idx * comp_idx_stride + j)
            valid = j < tl.load(comp_lens_ptr + token_idx)
            bs = comp_block_size
            stride = comp_block_stride
            cptr = comp_cache_ptr
        else:
            slot = -1
            valid = False
            bs = swa_block_size
            stride = swa_block_stride
            cptr = swa_cache_ptr

    # -1 (or any negative) -> slot 0; the row is masked by `valid` and never
    # attended (dequant OOB lesson from the sglang port).
    slot = tl.where(valid, slot, tl.where(slot >= 0, slot, 0))
    blk = slot.to(tl.int64) // bs
    off = slot.to(tl.int64) % bs
    if packed_584:
        # SM120 FlashInfer packed pool: one 584B row per token
        # [0:448 fp8 | 448:576 rope | 576:584 ue8m0 scales] interleaved
        # in-row, vs the 576B block-pooled layout (scale tail after rows).
        addr_data = cptr + blk * stride + off * 584
        addr_scale = cptr + blk * stride + off * 584 + 576
    else:
        addr_data = cptr + blk * stride + off * 576
        addr_scale = cptr + blk * stride + bs * 576 + off * 8

    if valid:
        _combine_dequant_e8m0_fp8_row(
            addr_data, addr_scale, out_row, _FP8_DIM, _QUANT_BLOCK
        )
    else:
        zeros = tl.zeros((512,), tl.float32).to(tl.bfloat16)
        tl.store(out_row + tl.arange(0, 512), zeros)
    if write_valid:
        tl.store(valid_ptr + token_idx * (k_swa + k_comp) + slot_idx, valid.to(tl.uint8))


def dequantize_combined_slots(
    combined_kv: torch.Tensor,
    valid_buf: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_block_size: int,
    comp_cache: torch.Tensor | None,
    comp_indices: torch.Tensor | None,
    comp_lens: torch.Tensor | None,
    comp_block_size: int,
    packed_584: bool = False,
) -> None:
    assert swa_indices.dim() == 2, (
        f"swa_indices must be (tokens, width), got {tuple(swa_indices.shape)}; "
        "callers must drop the singleton query axis"
    )
    num_tokens, k_swa = swa_indices.shape
    if comp_indices is None:
        k_comp = 0
    else:
        k_comp = comp_indices.shape[-1]
    assert combined_kv.shape == (num_tokens, k_swa + k_comp, 512)
    block_stride_swa = _swa_block_stride(swa_cache, swa_block_size, packed_584)
    block_stride_comp = (
        _swa_block_stride(comp_cache, comp_block_size, packed_584)
        if comp_cache is not None
        else 0
    )
    # Catch a stale / out-of-range candidate slot (would fault with Xid31
    # FAULT_PDE) before launching, when VLLM_SM8X_GUARD=1. No-op otherwise.
    guard_slot_table(
        swa_indices, swa_lens, swa_cache.shape[0] * swa_block_size, "SWA"
    )
    if comp_indices is not None:
        guard_slot_table(
            comp_indices, comp_lens, comp_cache.shape[0] * comp_block_size, "comp"
        )
    _dequantize_combined_slots_kernel[(num_tokens, k_swa + k_comp)](
        combined_kv,
        valid_buf,
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        swa_cache,
        comp_indices if comp_indices is not None else swa_indices,
        comp_indices.stride(0) if comp_indices is not None else swa_indices.stride(0),
        comp_lens if comp_lens is not None else swa_lens,
        comp_cache if comp_cache is not None else swa_cache,
        swa_block_size=swa_block_size,
        swa_block_stride=block_stride_swa,
        comp_block_size=comp_block_size if comp_block_size else swa_block_size,
        comp_block_stride=block_stride_comp,
        k_swa=k_swa,
        k_comp=k_comp,
        has_comp=comp_indices is not None,
        write_valid=True,
        packed_584=packed_584,
        num_warps=4,
    )


def _swa_block_stride(cache: torch.Tensor, block_size: int, packed_584: bool = False) -> int:
    # cache: (num_blocks, block_size, 1, bytes) uint8, padded to 576 align.
    # Packed SM120 pools are exactly 584B/token with no 576-alignment pad.
    num_blocks = cache.shape[0]
    if num_blocks >= 2:
        return int(cache[1].data_ptr() - cache[0].data_ptr())
    if packed_584:
        return block_size * 584
    unpadded = block_size * (_TOKEN_BYTES + _SCALE_BYTES)
    return ((unpadded + _TOKEN_BYTES - 1) // _TOKEN_BYTES) * _TOKEN_BYTES


class Sm8xAttnBuffers:
    """CUDA-graph-friendly buffer cache; key MUST include num_candidates."""

    def __init__(self) -> None:
        self._bufs: dict = {}

    def get(self, num_tokens: int, k_total: int, device, dtype):
        key = (num_tokens, k_total, device)
        bufs = self._bufs.get(key)
        if bufs is None:
            combined = torch.empty(
                (num_tokens, k_total, 512), dtype=torch.bfloat16, device=device
            )
            valid = torch.empty(
                (num_tokens, k_total), dtype=torch.uint8, device=device
            )
            bufs = (combined, valid)
            self._bufs[key] = bufs
        return bufs


def sm8x_decode_attention(
    buffers: Sm8xAttnBuffers,
    q: torch.Tensor,  # (T, H_pad, 512)
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_block_size: int,
    comp_cache: torch.Tensor | None,
    comp_indices: torch.Tensor | None,  # (T, K) global slots
    comp_lens: torch.Tensor | None,
    comp_block_size: int,
    attn_sink: torch.Tensor,
    scale: float,
    output: torch.Tensor,  # (T, H_pad, 512)
    num_heads: int,
    packed_584: bool = False,
) -> None:
    # SWA indices arrive as the builder's (T, next_n, width) buffer; this path
    # wants rows of candidates, so drop the singleton query axis (mirrors the
    # comp_indices handling below). next_n > 1 must be flattened by the caller.
    if swa_indices.dim() == 3 and swa_indices.shape[1] == 1:
        swa_indices = swa_indices.squeeze(1)
    if comp_indices is not None and comp_indices.dim() == 3:
        comp_indices = comp_indices.squeeze(1)
    k_swa = swa_indices.shape[-1]
    k_comp = 0 if comp_indices is None else comp_indices.shape[-1]
    combined, valid = buffers.get(
        q.shape[0], k_swa + k_comp, q.device, torch.bfloat16
    )
    dequantize_combined_slots(
        combined,
        valid,
        swa_cache,
        swa_indices,
        swa_lens,
        swa_block_size,
        comp_cache,
        comp_indices,
        comp_lens,
        comp_block_size,
        packed_584=packed_584,
    )
    if q.shape[0] <= 16:
        candidate_block = 128
    else:
        candidate_block = None
    matmul_sparse_mla_attention_with_sink(
        q,
        combined,
        valid,
        scale,
        attn_sink,
        output,
        num_heads=num_heads,
        candidate_block_size=candidate_block,
    )
