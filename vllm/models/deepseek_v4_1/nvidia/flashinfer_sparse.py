# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 FlashInfer sparse MLA backend."""

from typing import TYPE_CHECKING, ClassVar, cast

import functools

import torch

import vllm.envs as envs
from vllm import envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.common.ops import (
    build_flashinfer_mixed_sparse_indices,
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4_1.common.ops.stat_probe import (
    int_probe_once,
    stat_probe_once,
)
from vllm.models.deepseek_v4_1.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}

logger = init_logger(__name__)


def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
    workspace = _flashinfer_dsv4_workspace_by_device.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _flashinfer_dsv4_workspace_by_device[device] = workspace
    return workspace


def _packed_block_span(pool: torch.Tensor) -> int:
    """Per-block stride of ``pool`` in tokens (``stride(0)//stride(-2)``): ==
    block_size for unpacked KV, larger when packed (#44577). Raises if not
    token-aligned."""
    block_stride = pool.stride(0)
    token_stride = pool.stride(-2)
    if block_stride % token_stride != 0:
        raise NotImplementedError(
            "FLASHINFER_MLA_SPARSE_DSV4 packed KV requires the per-block stride "
            f"({block_stride}) to be a multiple of the per-token stride "
            f"({token_stride}); this layout is not supported yet."
        )
    return block_stride // token_stride


_cache_probe_state: dict[str, int] = {}
_CACHE_PROBE_MAX_LOGS = 5


def _probe_cache_nan_rows(tag: str, cache: torch.Tensor, rows: torch.Tensor) -> None:
    """Read back exactly the cache rows the sparse kernel is about to consume.

    e4m3 NaN bytes in the 448B NoPE segment (0x7F ignoring sign) and fp16
    NaNs in the 128B RoPE segment separate "producer wrote garbage / page
    unwritten" from "kernel misreads a well-formed page". Out-of-range rows
    are counted before clamping (never silently absorbed).
    """
    if not envs.VLLM_DSV41_SM120_PATH_CHECKS:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    if _cache_probe_state.get(tag, 0) >= _CACHE_PROBE_MAX_LOGS:
        return
    if cache.shape[-1] != 584:  # non-fp8_ds_mla layout: nothing to classify
        return
    idx = rows.reshape(-1)
    idx = idx[idx >= 0]
    if idx.numel() == 0:
        return
    # NEVER materialize via reshape: a packed pool has stride(0) > rows per
    # block, so reshape would copy the whole multi-GiB pool. Gather by
    # (block, offset) instead.
    if cache.dim() == 4:
        n_blocks, blk_sz = cache.shape[0], cache.shape[1]
        n_rows = n_blocks * blk_sz
    else:
        n_rows = cache.shape[0]
    oob = int((idx >= n_rows).sum().item())
    _cache_probe_state[tag] = _cache_probe_state.get(tag, 0) + 1
    keep = idx[idx < n_rows][:4096]
    if keep.numel() == 0:
        logger.warning(
            "PATH_CACHE %s #%d: rows=%d uniq=0 oob=%d (all rows out of range)",
            tag,
            _cache_probe_state[tag],
            idx.numel(),
            oob,
        )
        return
    if cache.dim() == 4:
        sample = cache[
            torch.div(keep, blk_sz, rounding_mode="floor"), keep % blk_sz
        ]
        sample = sample.reshape(-1, cache.shape[-1])
    else:
        sample = cache.index_select(0, keep)
    nope = sample[:, :448].view(torch.uint8)
    nan_nope = int(((nope & 0x7F) == 0x7F).sum().item())
    rope = sample[:, 448:576].view(torch.float16)
    nan_rope = int(torch.isnan(rope).sum().item())
    scale_b = sample[:, 576:584].view(torch.uint8)
    nan_scale = int((scale_b == 0xFF).sum().item())
    logger.warning(
        "PATH_CACHE %s #%d: rows=%d sampled=%d oob=%d nan_nope=%d/%d "
        "nan_rope=%d/%d nan_scale=%d/%d",
        tag,
        _cache_probe_state[tag],
        idx.numel(),
        sample.shape[0],
        oob,
        nan_nope,
        sample.shape[0] * 448,
        nan_rope,
        sample.shape[0] * 64,
        nan_scale,
        sample.shape[0] * 8,
    )


def _sanitize_negative_indices(idx: torch.Tensor) -> torch.Tensor:
    """Replace -1 padding with a VALID index (last valid of the same row, or
    the first valid of the whole tensor when a row is entirely -1). Unlike a
    plain clamp_min(0) this never points at uninitialized pool row 0, so a
    NaN-free result cleanly proves the kernel honours topk_lens and vice
    versa. Sync-free and graph-capturable."""
    row_last = idx.amax(dim=-1, keepdim=True)
    any_valid = idx.amax()
    row_last = torch.where(row_last < 0, any_valid, row_last)
    return torch.where(idx >= 0, idx, row_last)


_emul_seen: dict[str, set] = {}
_EMUL_MAX = 12


def _cache_rows_as_vecs(cache: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """fp8_ds_mla 584B rows -> float32 (…, 512) vectors, dequantized with
    stored e8m0 scales (7 x 64B NoPE blocks; RoPE stored as bf16)."""
    valid = idx >= 0
    safe = idx.clamp_min(0)
    if cache.dim() == 4:
        flat = safe.reshape(-1)
        raw = cache[torch.div(flat, cache.shape[1], rounding_mode="floor"), flat % cache.shape[1]]
        raw = raw.reshape(*idx.shape, cache.shape[-1])
    else:
        raw = cache.index_select(0, safe.reshape(-1)).reshape(*idx.shape, cache.shape[-1])
    nope = raw[..., :448].to(torch.uint8)
    rope = raw[..., 448:576].view(torch.bfloat16).to(torch.float32)
    exp = raw[..., 576:584].to(torch.uint8)[..., :7].to(torch.int32) - 127
    vals = nope.to(torch.float32).reshape(*idx.shape, 7, 64)
    vals = torch.ldexp(vals, exp.unsqueeze(-1)).reshape(*idx.shape, 448)
    out = torch.cat([vals, rope], dim=-1)
    return torch.where(valid.unsqueeze(-1), out, torch.zeros((), device=out.device, dtype=out.dtype))


def _emulate_sparse_mla(
    tag, q, swa_idx, swa_lens, swa_cache, ex_idx, ex_lens, ex_cache,
    sm_scale, sink, real_rows, out_chunk,
) -> None:
    """Pure-torch double of the sparse-MLA kernel for the first rows.

    If the kernel says NaN where this says finite, the poison lives inside
    the kernel's read/compute path, not in the visible bytes."""
    if not envs.VLLM_DSV41_SM120_PATH_CHECKS:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    seen = _emul_seen.setdefault(tag, set())
    if len(seen) >= _EMUL_MAX:
        return
    n_rows = min(int(real_rows), q.shape[0], 6)
    emul_nan, kern_nan, details = [], [], ""
    total_valid = 0
    for r in range(n_rows):
        parts_v, parts_valid = [], []
        idx_r = swa_idx[r].reshape(-1)
        parts_v.append(_cache_rows_as_vecs(swa_cache, idx_r.unsqueeze(0))[0])
        parts_valid.append((idx_r >= 0) & (torch.arange(idx_r.shape[0], device=idx_r.device) < int(swa_lens[r])))
        if ex_idx is not None:
            e_r = ex_idx[r].reshape(-1)
            parts_v.append(_cache_rows_as_vecs(ex_cache, e_r.unsqueeze(0))[0])
            parts_valid.append((e_r >= 0) & (torch.arange(e_r.shape[0], device=e_r.device) < int(ex_lens[r])))
        kv = torch.cat(parts_v)
        kv_valid = torch.cat(parts_valid)
        total_valid += int(kv_valid.sum())
        scores = (kv.to(torch.float32) @ q[r].to(torch.float32).T) * sm_scale  # (W, H)
        scores = scores.masked_fill(~kv_valid.unsqueeze(-1), float("-inf"))
        sink_row = sink.to(torch.float32).unsqueeze(0)
        probs = torch.softmax(torch.cat([scores, sink_row], dim=0), dim=0)
        out_row = probs[:-1].to(torch.float32).T @ kv.to(torch.float32)
        em_n = bool(torch.isnan(out_row).any())
        if em_n:
            emul_nan.append(r)
        out_r = out_chunk[r] if r < out_chunk.shape[0] else None
        kr_n = out_r is not None and bool(torch.isnan(out_r).any())
        if kr_n:
            kern_nan.append(r)
        if r < 3:
            vsc = scores[kv_valid]
            nv = int(vsc.numel())
            fin_v = int(vsc.isfinite().sum()) if nv else 0
            pinf_v = int(torch.isposinf(vsc).sum()) if nv else 0
            details += (
                f" r{r}: vld={fin_v}/{int(kv_valid.sum())} posinf={pinf_v}"
                f" emN={int(em_n)} krN={int(kr_n)} sinkp={float(probs[-1, 0]):.3g};"
            )
    sig = (n_rows, tuple(emul_nan), tuple(kern_nan), total_valid)
    # capture/warmup garbage: no valid KV anywhere and nothing non-finite
    # tells us nothing new -> never consumes the tag budget.
    if total_valid == 0 and not emul_nan and not kern_nan:
        return
    if sig in seen:
        return
    seen.add(sig)
    logger.warning(
        "PATH_EMUL %s #%d: sampled=%d nvalid=%d emul_nan_rows=%s kernel_nan_rows=%s%s",
        tag, len(seen), n_rows, total_valid, emul_nan, kern_nan, details,
    )


# Sparse MLA h_q counts accepted natively (flashinfer>=0.6.14, #3545).
_SPARSE_MLA_SUPPORTED_Q_HEADS = (8, 16, 32, 64, 128)


def _pad_to_supported_q_heads(num_heads: int) -> int:
    for supported in _SPARSE_MLA_SUPPORTED_Q_HEADS:
        if num_heads <= supported:
            return supported
    raise ValueError(
        f"DeepseekV4 FlashInfer MLA Sparse does not support {num_heads} heads "
        "(sparse MLA kernel requires h_q in {8, 16, 32, 64, 128})."
    )


def _required_sm120_sparse_topk(vllm_config: VllmConfig, window_size: int) -> int:
    """Return the SM120 DSV4 SWA specialization needed by this model."""
    if not vllm_config.attention_config.use_non_causal:
        return window_size
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return window_size
    return get_dspark_swa_index_width(
        window_size,
        speculative_config.num_speculative_tokens,
    )


# The SM120 sparse-MLA op auto-selects: a standalone *decode* instantiation
# while num_tokens <= 64 (built only for the (heads, topk) pairs in
# flashinfer _DECODE_DSV4_DISPATCH), otherwise the paged prefill orchestrator,
# which asserts num_tokens > 64 in C++. A v4.1 prefill index row is
# ``window + max_image_tokens`` wide, so a VL variant hands 1152-wide rows to a
# short prefill (<= 64 tokens: a 2-token warmup batch, a chunk tail, a short
# prompt) and neither specialization matches -> "no decode kernel for this
# shape". Index rows are prefix-filled and -1 padded, so cutting the width to
# the smallest instantiated width that covers max(len) is lossless; the rare
# row that genuinely needs > 1024 candidates crosses the token threshold
# instead by replaying row 0 into scratch rows.
_SM120_DECODE_MAX_TOKENS = 64
_SM120_DECODE_TOPK_FALLBACK = (128, 192, 256, 512, 1024)
# topk widths compiled into sparse_mla_sm120_prefill.cu's
# dispatch_dsv4_single (the dual-cache variant additionally requires exactly
# 128, which the width snap below satisfies whenever no row exceeds the
# window; >128 candidate rows only exist inside image spans).
_SM120_PREFILL_TOPK_WIDTHS = (128, 192, 256, 512, 1024, 2048)


@functools.lru_cache(maxsize=None)
def _sm120_decode_topk_widths(num_heads: int) -> tuple[int, ...]:
    """topk widths FlashInfer instantiates for this head count (read from the
    kernel's own table so the two cannot drift)."""
    try:
        from flashinfer.mla._sparse_mla_sm120 import _DECODE_DSV4_DISPATCH

        widths = {
            int(topk)
            for heads, topk in _DECODE_DSV4_DISPATCH
            if int(heads) == num_heads
        }
    except Exception:  # pragma: no cover - older/newer flashinfer layout
        widths = set()
    return tuple(sorted(widths)) or _SM120_DECODE_TOPK_FALLBACK


def _pad_rows_with_row0(rows: torch.Tensor, pad: int) -> torch.Tensor:
    """Append ``pad`` scratch rows that replay row 0. Row 0 is a real, fully
    valid row, so every scratch row stays kernel-legal (no empty-row LSE) at the
    cost of one extra row of gather work on an already-tiny step."""
    if pad <= 0:
        return rows
    return torch.cat([rows, rows[:1].expand(pad, *rows.shape[1:])])


def _sm120_check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"SM120 sparse-MLA dispatch invariant broken: {msg}")


def _sm120_assert_rows_fit(
    indices: torch.Tensor,
    lens: torch.Tensor | None,
    width: int,
    *,
    needed: int,
    where: str,
) -> None:
    """Verify the premise that lets ``_snap_topk_width`` be lossless: index
    rows are prefix-filled and ``-1`` padded, so (a) no candidate lives in the
    columns being clipped, and (b) no row's ``len`` exceeds ``width``.

    Costs one device sync per call: gated by
    ``VLLM_DSV41_SM120_PATH_CHECKS`` (off on the steady-state path).
    """
    if not envs.VLLM_DSV41_SM120_PATH_CHECKS or indices.shape[0] == 0:
        return
    _sm120_check(needed <= width, f"{where}: needed {needed} > width {width}")
    tail = indices[..., width:]
    if tail.numel() and bool((tail != -1).any().item()):
        raise AssertionError(
            f"SM120 sparse-MLA dispatch invariant broken: {where} would drop "
            f"a live candidate -- index rows are not prefix-filled + -1 padded "
            f"at width {width} (row width {indices.shape[-1]})."
        )
    if lens is not None and lens.numel() and int(lens.max().item()) > width:
        raise AssertionError(
            f"SM120 sparse-MLA dispatch invariant broken: {where} has a row "
            f"longer ({int(lens.max().item())}) than the snapped width {width}."
        )


def _snap_topk_width(indices: torch.Tensor, width: int) -> torch.Tensor:
    """Clip/extend index columns to ``width``. Rows are prefix-filled and
    ``-1`` padded and ``-1`` is the kernel's invalid marker, so both
    directions are lossless provided every real candidate fits (checked by
    the caller's ``needed`` bound)."""
    cur = indices.shape[-1]
    if cur == width:
        return indices
    if cur > width:
        return indices[..., :width].contiguous()
    return torch.nn.functional.pad(indices, (0, width - cur), value=-1)


_geo_seen: set = set()
_GEO_MAX = 12


def _geo_probe(tag, q, lens, region, decision, in_w, out_w, pad):
    if not envs.VLLM_DSV41_SM120_PATH_CHECKS:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    sig = (tag, q.shape[0], region, decision, in_w, out_w, pad)
    if sig in _geo_seen or len(_geo_seen) >= _GEO_MAX:
        return
    _geo_seen.add(sig)
    ln_max = int(lens.max().item()) if lens.numel() else -1
    logger.warning(
        "PATH_GEO %s: rows=%d heads=%d region=%s decision=%s in_w=%d out_w=%d "
        "pad=%d lens_max=%d",
        tag, q.shape[0], q.shape[1], region, decision, in_w, out_w, pad, ln_max,
    )


def _sm120_prefill_dispatch_view(
    q: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    window_size: int,
    has_image: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    int,
]:
    """Reshape a short/wide prefill call so the SM120 dispatcher finds a kernel.

    Returns ``(q, indices, lens, out, copy_back, pad)``; ``copy_back`` is the
    destination view to copy ``out[:rows]`` into when ``pad`` scratch rows were
    added (the caller pads its own extra-segment rows with ``pad``).
    """
    rows = q.shape[0]
    in_w = swa_indices.shape[-1]
    if rows == 0:
        return q, swa_indices, swa_lens, out, None, 0
    if rows > _SM120_DECODE_MAX_TOKENS:
        # Paged prefill orchestrator region: its DSV4 specializations are
        # compiled for their own width list, so a wide VL row (window +
        # image) needs the same snap here as short rows do in the decode
        # table -- the 1152-wide rows crash the orchestrator too.
        widths = _SM120_PREFILL_TOPK_WIDTHS
        region = "prefill-orchestrator"
    else:
        widths = _sm120_decode_topk_widths(q.shape[1])
        region = "decode"
    if swa_indices.shape[-1] in widths:
        _geo_probe("GEO-PF", q, swa_lens, region, "native", in_w, swa_indices.shape[-1], 0)
        return q, swa_indices, swa_lens, out, None, 0
    if has_image:
        # Bidirectional image visibility widens rows data-dependently, so the
        # covered width has to come from the metadata itself.
        needed = int(swa_lens.max().item()) if swa_lens.numel() else 0
    else:
        # No image spans this step: len == min(pos + 1, window) <= window, so
        # the row bound is static and this stays sync-free.
        needed = window_size
    cover = [width for width in widths if width >= needed]
    if cover:
        _sm120_assert_rows_fit(
            swa_indices, swa_lens, cover[0], needed=needed, where=f"{region} snap"
        )
        swa_indices = _snap_topk_width(swa_indices, cover[0])
        _geo_probe("GEO-PF", q, swa_lens, region, "snap", in_w, cover[0], 0)
        return q, swa_indices, swa_lens, out, None, 0
    pad = _SM120_DECODE_MAX_TOKENS + 1 - rows
    if rows <= _SM120_DECODE_MAX_TOKENS:
        # The padded rows land in the *orchestrator*, so snap the width to
        # the prefill list (rows needing > 1024 candidates take the 2048
        # template) instead of leaving an unsupported width there.
        prefill_cover = [w for w in _SM120_PREFILL_TOPK_WIDTHS if w >= needed]
        if prefill_cover:
            _sm120_assert_rows_fit(
                swa_indices,
                swa_lens,
                prefill_cover[0],
                needed=needed,
                where="decode->orchestrator pad",
            )
            swa_indices = _snap_topk_width(swa_indices, prefill_cover[0])
    else:
        raise RuntimeError(
            "SM120 sparse-MLA prefill found no compiled specialization for "
            f"needed topk width {needed} (compiled widths: {widths})."
        )
    _sm120_check(pad > 0, f"pad must be positive, got {pad} for rows={rows}")
    _sm120_check(
        rows + pad == _SM120_DECODE_MAX_TOKENS + 1,
        f"padded rows {rows + pad} must cross the decode threshold "
        f">{_SM120_DECODE_MAX_TOKENS}",
    )
    q = _pad_rows_with_row0(q, pad)
    swa_indices = _pad_rows_with_row0(swa_indices, pad)
    swa_lens = _pad_rows_with_row0(swa_lens, pad)
    padded_out = torch.empty(
        (rows + pad, *out.shape[1:]), dtype=out.dtype, device=out.device
    )
    _geo_probe("GEO-PF", q, swa_lens, region, "pad->orch", in_w, swa_indices.shape[-1], pad)
    return q, swa_indices, swa_lens, padded_out, out, pad


class DeepseekV4FlashInferMLASparseBackend(DeepseekV4SparseMLABackend):
    """FlashInfer backend using the DSv4 sparse metadata/cache layout.

    Inherits the base and backend reuses its``DeepseekV4SparseMLAMetadataBuilder``
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Group-floor anchored (every worker publishes this list; see
        # Platform.group_capability_floor).
        floor = current_platform.group_capability_floor()
        if floor is not None and floor.to_int() // 10 == 12:
            # SM120 sparse-MLA decode is compiled for a 64-state page; v4.1
            # pins each cache group to 64 * compress_ratio tokens (see
            # deepseek_v4_1.attention._sm120_paged_block_size), so accept any
            # 64-multiple rather than pinning the global block size (which
            # would force a physical page split the BLHNC layout cannot do).
            return [MultipleOf(64)]
        return [128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV41"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [10, 12]

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if device_capability.major == 10:
            if kv_cache_dtype == "fp8_ds_mla":
                return (
                    "FLASHINFER_MLA_SPARSE_DSV4 SM10x uses the plain "
                    "per-tensor FP8 KV layout, not fp8_ds_mla"
                )
            if kv_cache_dtype not in (None, "auto", "bfloat16", "fp8", "fp8_e4m3"):
                return "kv_cache_dtype not supported"
            return None
        if device_capability.major == 12:
            if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla"):
                return "kv_cache_dtype not supported"
            from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

            if not has_flashinfer_sparse_mla_sm120():
                return (
                    "FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer's "
                    "sparse MLA decode API"
                )
            return None
        return "FLASHINFER_MLA_SPARSE_DSV4 requires SM10x or SM12x"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4FlashInferSparseMLAMetadataBuilder"]:
        return DeepseekV4FlashInferSparseMLAMetadataBuilder


class DeepseekV4FlashInferSparseMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    """Varlen-capable metadata builder for the FlashInfer sparse MLA backend."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekSparseSWAFlashInferMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    """SWA metadata for the FlashInfer sparse decode path (varlen decode)."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekSparseSWAFlashInferBackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAFlashInferMetadataBuilder]:
        return DeepseekSparseSWAFlashInferMetadataBuilder


class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):
    """FlashInfer TRTLLM-gen sparse MLA attention layer for SM100 DeepSeek V4."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    swa_backend_cls = DeepseekSparseSWAFlashInferBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return _pad_to_supported_q_heads(num_heads)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(
            self._o_proj_block_size
        )
        # Per-tensor FP8 scale buffers + precomputed scalar BMM scales. Only the
        # per-tensor FP8 cache path consumes these; bf16 reads ``self.scale``.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # TRTLLM-gen takes scalar scale args on a distinct C++ path vs
        # one-element tensors, so these are Python floats.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # The TRTLLM-gen kernel requires h_q in {64, 128}, so the output buffer
        # is allocated at the padded head count while q arrives at the local
        # head count; _forward pads q to match before the launcher.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: FlashInfer reads the cache directly and lazily
            # allocates its workspace, so nothing to reserve here.
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        # SWA-only layers have no compressed KV cache; consumers read the kv
        # source's cache.
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward(
            q=q,
            kv_cache=self_kv_cache,
            swa_k_cache=swa_kv_cache,
            swa_metadata=swa_metadata,
            attn_metadata=flashmla_metadata,
            swa_only=swa_only,
            output=output,
        )

    def _build_sparse_index_metadata(
        self,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the combined sparse-index tensors for the mixed batch.

        Returns ``(compressed_kv_cache, seq_lens, sparse_indices,
        sparse_topk_lens)``.
        """
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens

        assert swa_metadata.seq_lens is not None
        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.block_table is not None

        decode_swa_indices = swa_metadata.decode_swa_indices.reshape(
            num_decode_tokens, swa_metadata.decode_swa_width
        )
        decode_compressed_topk_lens = None
        decode_compressed_indices_are_local = False
        decode_is_valid_token = None

        if swa_only:
            assert self.topk_indices_buffer is not None
            compressed_kv_cache = swa_k_cache
            decode_compressed_indices = None
            prefill_topk_indices = self.topk_indices_buffer[
                num_decode_tokens:num_tokens, :0
            ]
            compressed_block_table = None
            compressed_block_size = swa_metadata.block_size
            top_k = 0
        else:
            assert kv_cache is not None
            assert attn_metadata is not None
            assert self.topk_indices_buffer is not None
            assert swa_metadata.is_valid_token is not None
            compressed_kv_cache = kv_cache
            compressed_block_table = attn_metadata.block_table[:num_reqs]
            compressed_block_size = attn_metadata.block_size // self.compress_ratio

            # Local indices filled by the index-source layer's indexer.
            if num_prefill_tokens > 0:
                prefill_topk_indices = self.topk_indices_buffer[
                    num_decode_tokens:num_tokens
                ]
                top_k = prefill_topk_indices.shape[-1]
            else:
                prefill_topk_indices = self.topk_indices_buffer[:0, :0]
                top_k = 0

            decode_compressed_indices_are_local = True
            decode_is_valid_token = swa_metadata.is_valid_token[:num_decode_tokens]
            if num_decode_tokens > 0:
                decode_compressed_indices = self.topk_indices_buffer[:num_decode_tokens]
            else:
                # Keep the logical width aligned with the mixed-batch case so
                # pure-prefill steps reuse the same Triton specialization.
                decode_compressed_indices = prefill_topk_indices[:0]

        query_start_loc = swa_metadata.query_start_loc[: num_reqs + 1]
        seq_lens = swa_metadata.seq_lens[:num_reqs]
        assert seq_lens.dtype == torch.int32
        # SWA-only layers all build the same mixed sparse indices, so the first
        # one caches them for the step; indexer layers depend on their own topk
        # indices and stay uncached.
        cached_sparse = (
            swa_metadata.flashinfer_sparse_index_cache.get("swa_only")
            if swa_only
            else None
        )
        if cached_sparse is None:
            swa_block_span = _packed_block_span(swa_k_cache)
            compressed_block_span = _packed_block_span(compressed_kv_cache)
            sparse_indices, sparse_topk_lens = build_flashinfer_mixed_sparse_indices(
                decode_swa_indices,
                decode_compressed_indices,
                decode_compressed_topk_lens,
                prefill_topk_indices[:num_prefill_tokens],
                query_start_loc,
                seq_lens,
                swa_metadata.token_to_req_indices[:num_tokens],
                swa_metadata.block_table[:num_reqs],
                swa_metadata.block_size,
                compressed_block_table,
                compressed_block_size,
                self.window_size,
                self.compress_ratio,
                top_k,
                decode_compressed_indices_are_local=decode_compressed_indices_are_local,
                decode_is_valid_token=decode_is_valid_token,
                swa_block_span=swa_block_span,
                compressed_block_span=compressed_block_span,
                prefill_left_visible=swa_metadata.prefill_left_visible,
                prefill_right_visible=swa_metadata.prefill_right_visible,
                # getattr for tests that bypass __init__ via object.__new__.
                max_image_tokens=getattr(self, "max_image_tokens", 0),
            )
            if swa_only:
                swa_metadata.flashinfer_sparse_index_cache["swa_only"] = (
                    sparse_indices,
                    sparse_topk_lens,
                )
        else:
            sparse_indices, sparse_topk_lens = cached_sparse
        return compressed_kv_cache, seq_lens, sparse_indices, sparse_topk_lens

    def _forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        assert self.kv_cache_torch_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens
        if num_tokens == 0:
            return

        (
            compressed_kv_cache,
            seq_lens,
            sparse_indices,
            sparse_topk_lens,
        ) = self._build_sparse_index_metadata(
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            swa_only=swa_only,
        )

        # CUDA graph execution can pad q/output past the scheduled token count;
        # restrict to the real tokens (the launcher validates sparse indices).
        query = q[:num_tokens]
        output = output[:num_tokens]
        bmm1_scale: float | torch.Tensor = self.scale
        bmm2_scale: float | torch.Tensor = 1.0
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert query.dtype == torch.float8_e4m3fn
            bmm1_scale = self._flashinfer_fp8_bmm1_scale
            bmm2_scale = self._flashinfer_fp8_bmm2_scale
        else:
            assert query.dtype == torch.bfloat16
            query = query.contiguous()

        # The TRTLLM-gen sparse-MLA kernel requires h_q in {64, 128}; zero-pad
        # the query heads to the allocated output head count. Padded heads attend
        # to the shared KV and are sliced off downstream (output is padded too).
        padded_heads = output.shape[1]
        if query.shape[1] < padded_heads:
            padded_query = query.new_zeros(
                (query.shape[0], padded_heads, query.shape[2])
            )
            padded_query[:, : query.shape[1], :] = query
            query = padded_query

        workspace = _get_flashinfer_dsv4_workspace(q.device)
        query_start_loc = swa_metadata.query_start_loc
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc is not None and query_start_loc_cpu is not None

        # Keep the TRTLLM-gen decode/prefill split: the launcher is tuned for
        # uniform-q batches, and this avoids flattening mixed batches into one call.
        if num_decode_tokens > 0:
            decode_cu = query_start_loc[: num_decodes + 1]
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[:num_decode_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[:num_decode_tokens],
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[:num_decode_tokens],
                seq_lens=seq_lens[:num_decodes],
                out=output[:num_decode_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=decode_cu,
                max_q_len=swa_metadata.max_decode_query_len,
            )

        if num_prefill_tokens > 0:
            # The prefill query view re-anchors at offset 0, so rebase the
            # cumulative query offsets to start at 0.
            prefill_cu = (
                query_start_loc[num_decodes : num_reqs + 1]
                - query_start_loc[num_decodes]
            )
            prefill_cu_cpu = query_start_loc_cpu[num_decodes : num_reqs + 1]
            prefill_lens_cpu = prefill_cu_cpu[1:] - prefill_cu_cpu[:-1]
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[num_decode_tokens:num_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                seq_lens=seq_lens[num_decodes:num_reqs],
                out=output[num_decode_tokens:num_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=prefill_cu,
                max_q_len=int(prefill_lens_cpu.max().item()),
            )


class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):
    """DeepSeek V4 sparse MLA attention through FlashInfer's SM120 kernels."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    swa_backend_cls = DeepseekSparseSWAFlashInferBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = True

    @staticmethod
    def _get_workspace(device: torch.device) -> torch.Tensor:
        return _get_flashinfer_dsv4_workspace(device)

    @staticmethod
    def _as_sparse_cache(kv_cache: torch.Tensor) -> torch.Tensor:
        if kv_cache.dtype == torch.float8_e4m3fn:
            kv_cache = kv_cache.view(torch.uint8)
        if kv_cache.dim() == 4:
            return kv_cache
        return kv_cache.unsqueeze(-2)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return _pad_to_supported_q_heads(num_heads)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config

        required_topk = _required_sm120_sparse_topk(vllm_config, self.window_size)
        if not has_flashinfer_sparse_mla_sm120_config(self.padded_heads, required_topk):
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires a FlashInfer "
                "DSV4 sparse MLA decode specialization for "
                f"(num_q_heads={self.padded_heads}, top_k={required_topk}). "
                "Install a FlashInfer build containing "
                "flashinfer-ai/flashinfer#4380."
            )
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(
            self._o_proj_block_size
        )
        # Per-tensor FP8 cache path scales.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # FlashInfer expects scalar scale arguments for this path.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def _reserve_empty_forward_workspace(self) -> None:
        self._get_workspace(
            torch.device("cuda", torch.accelerator.current_device_index())
        )

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Output may be padded to backend-supported head counts.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_empty_forward_workspace()
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        # SWA-only layers have no compressed KV cache; consumers read the kv
        # source's cache.
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _prepare_query(self, q: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert q.dtype == torch.float8_e4m3fn
            q = q.to(torch.bfloat16)
        else:
            assert q.dtype == torch.bfloat16
        padded_heads = output.shape[1]
        if q.shape[1] < padded_heads:
            padded_query = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_query[:, : q.shape[1], :] = q
            q = padded_query
        return q.contiguous()

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "Compressed-layer decode requires top-k indices from the indexer."
                )
            # Local indices filled by the index-source layer's indexer.
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            block_size = attn_metadata.block_size // self.compress_ratio
            global_indices, extra_sparse_lengths = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )
            extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        if envs.VLLM_DSV41_SM8X_CROSSCHECK:
            # A/B: same tensors, SM8x Triton dequant+attn on the packed 584B
            # pools instead of the SM120 CUDA kernel (cross-validation only).
            from vllm.models.deepseek_v4.nvidia.ops.sm8x_attn import (
                Sm8xAttnBuffers,
                sm8x_decode_attention,
            )

            if not hasattr(self, "_sm8x_buffers"):
                self._sm8x_buffers = Sm8xAttnBuffers()
            sm8x_decode_attention(
                self._sm8x_buffers,
                q,
                swa_cache,
                swa_indices,
                swa_lens,
                swa_metadata.block_size,
                extra_cache,
                extra_sparse_indices,
                extra_sparse_lengths,
                0 if swa_only else attn_metadata.block_size // self.compress_ratio,
                self.attn_sink,
                self.scale,
                output,
                self.n_local_heads,
                packed_584=True,
            )
            return
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache,
            workspace_buffer=self._get_workspace(q.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio == 0

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if not swa_only:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "Compressed-layer prefill requires top-k indices from the indexer."
                )
            if attn_metadata is None:
                raise RuntimeError("Compressed-layer prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError(
                    "Compressed-layer prefill request mapping is missing."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "Compressed-layer prefill validity metadata is missing."
                )
            # Local indices filled by the index-source layer's indexer.
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // self.compress_ratio
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )
            int_probe_once(f"FI-L{self.layer_id}.topk_local", local_topk_indices)
            int_probe_once(f"FI-L{self.layer_id}.topk_glob", extra_sparse_indices)
            int_probe_once(f"FI-L{self.layer_id}.topk_len", extra_sparse_lengths)

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        q_fp8_raw = q
        q = self._prepare_query(q, output)
        stat_probe_once(f"FI-L{self.layer_id}.q_fp8_raw", q_fp8_raw)
        stat_probe_once(f"FI-L{self.layer_id}.sink", self.attn_sink)
        stat_probe_once(f"FI-L{self.layer_id}.prefill_q", q)
        int_probe_once(
            f"FI-L{self.layer_id}.swa_idx",
            swa_metadata.prefill_swa_indices,
        )
        int_probe_once(
            f"FI-L{self.layer_id}.swa_lens",
            swa_metadata.prefill_swa_lens,
        )
        swa_kv_paged = self._as_sparse_cache(swa_k_cache)
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = self._as_sparse_cache(compressed_k_cache)
        _probe_cache_nan_rows(
            f"PG-L{self.layer_id}.swa",
            swa_kv_paged,
            swa_metadata.prefill_swa_indices,
        )
        if not swa_only and extra_kv_paged is not None:
            _probe_cache_nan_rows(
                f"PG-L{self.layer_id}.comp", extra_kv_paged, extra_sparse_indices
            )

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )

            q_chunk = q[query_start:query_end]
            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:
                raise RuntimeError(
                    "Compressed sparse MLA prefill requires compressed sparse indices."
                )
            out_chunk = output[query_start:query_end]
            (
                q_chunk,
                swa_indices_chunk,
                swa_lens_chunk,
                out_chunk,
                out_dst,
                pad_rows,
            ) = _sm120_prefill_dispatch_view(
                q_chunk,
                swa_indices_chunk,
                swa_lens_chunk,
                out_chunk,
                window_size=self.window_size,
                has_image=swa_metadata.prefill_left_visible is not None,
            )
            # DIAGNOSTIC (NaN hunt): sanitize -1 padding to VALID indices
            # after snap/assert so the prefix-filled invariant check still
            # sees the raw rows.
            swa_indices_chunk = _sanitize_negative_indices(swa_indices_chunk)
            if pad_rows and extra_sparse_indices_chunk is not None:
                # Scratch rows must exist in every segment of this call.
                extra_sparse_indices_chunk = _pad_rows_with_row0(
                    extra_sparse_indices_chunk, pad_rows
                )
                extra_sparse_lengths_chunk = _pad_rows_with_row0(
                    extra_sparse_lengths_chunk, pad_rows
                )
            # DIAGNOSTIC (NaN hunt): same -1 sanitization on the compressed
            # segment (post any width logic).
            if extra_sparse_indices_chunk is not None:
                extra_sparse_indices_chunk = _sanitize_negative_indices(
                    extra_sparse_indices_chunk
                )
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
                workspace_buffer=self._get_workspace(q.device),
                sparse_indices=swa_indices_chunk,
                compressed_kv_cache=extra_kv_paged,
                out=out_chunk,
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens_chunk,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_topk_lens=extra_sparse_lengths_chunk,
            )
            _emulate_sparse_mla(
                f"EM-L{self.layer_id}",
                q_chunk,
                swa_indices_chunk,
                swa_lens_chunk,
                swa_kv_paged,
                extra_sparse_indices_chunk,
                extra_sparse_lengths_chunk,
                extra_kv_paged,
                self.scale,
                self.attn_sink,
                out_dst.shape[0] if out_dst is not None else out_chunk.shape[0],
                out_chunk,
            )
            if out_dst is not None:
                _sm120_check(
                    out_dst.shape[0] == out_chunk.shape[0] - pad_rows,
                    f"copy-back region {out_dst.shape[0]} != padded rows "
                    f"{out_chunk.shape[0]} minus pad {pad_rows}",
                )
                out_dst.copy_(out_chunk[: out_dst.shape[0]])
