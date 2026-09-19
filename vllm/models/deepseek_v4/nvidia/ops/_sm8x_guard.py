# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side guards for the SM8x sparse-MLA Triton path.

The SM8x Triton kernels clamp only *negative* slots (``-1`` -> slot 0) and rely
on a ``valid`` mask; they carry **no upper bound** against the current KV pool.
A stale or out-of-range physical slot -- a freed page, or a block id that is
valid on one rank but past another rank's (smaller) pool under a heterogeneous
group floor -- is then read straight off the cache, and the GPU reports it as
``Xid31 FAULT_PDE VIRT_READ`` (an MMU read of an unmapped page-table entry).

These guards let us catch that *before* the kernel faults the device. When
``VLLM_SM8X_GUARD=1`` the Python launchers make one cheap host round-trip over
only the region the kernel will actually load (columns/slots below ``lens``), and
raise a catchable ``RuntimeError`` naming the offending slot instead of letting
the Triton kernel read a dangling page. Off by default (the check syncs).
Skipped while a CUDA graph stream is capturing, so coverage is eager/prefill
launches; captured decode replays no host code and cannot be guarded.

Used by ``sm8x_mqa.fp8_paged_mqa_logits_rowwise_triton`` (block table) and
``sm8x_attn.dequantize_combined_slots`` (candidate slot tables).
"""

import os

import torch

from vllm import envs

_counters: dict[str, int] = {}
_btw_logged: set[int] = set()


def _throttle(key: str, every: int) -> bool:
    """Check only every ``every``-th call for this site (sync tax control).

    ``every <= 1`` means *check every call* (zero throttle); ``every = N`` checks
    1-in-N. Note ``n % 1`` is always 0, so the ``every == 1`` case must be
    special-cased or ``VLLM_SM8X_GUARD_DECODE_EVERY=1`` silently disables the
    guard entirely (the ab61 false-negative).
    """
    n = _counters.get(key, 0) + 1
    _counters[key] = n
    return True if every <= 1 else n % every == 1


def _guard(table: torch.Tensor, loaded: torch.Tensor, limit: int, kind: str,
           every: int = 1) -> None:
    """``table`` values loaded where ``loaded`` is True must be < ``limit``.

    ``-1``/negative entries are the kernels' "invalid" sentinel (clamped to slot
    0, masked out), so they are ignored here; only positive out-of-range ids can
    actually fault. ``every`` rate-limits the host sync (each check drains the
    async CPU-MoE pipeline; decode launches check 1-in-N).
    """
    if not envs.VLLM_SM8X_GUARD or table.numel() == 0:
        return
    if torch.cuda.is_current_stream_capturing():
        # CUDA graph capture: a host round-trip is not allowed here (and the
        # slot tables hold capture-dummy data anyway). Guards therefore cover
        # eager/prefill launches only; graph replay is host-code-free.
        return
    if not _throttle(kind, every):
        return
    view = table[loaded]
    if view.numel() == 0:
        return
    mx = int(view.max())
    if mx >= limit:
        bad = table.ge(limit) & loaded
        coords = bad.nonzero()
        first = coords[0].tolist() if coords.numel() else [-1, -1]
        n_bad = int(bad.sum())
        raise RuntimeError(
            f"[VLLM_SM8X_GUARD] {kind} out of range: {n_bad} slot(s), max={mx} "
            f">= num_blocks/num_slots={limit} (table shape={tuple(table.shape)}, "
            f"first offending index={first}). This is a stale / freed physical "
            f"block (prefix-reuse use-after-free) or a mixed-floor numbering "
            f"mismatch -- reading it would raise Xid31 FAULT_PDE VIRT_READ."
        )


def _touch_pages(
    cache: tuple[torch.Tensor, torch.Tensor],  # (values, scale) block-strided views
    table: torch.Tensor,
    loaded: torch.Tensor,
) -> None:
    """First-touch every page the kernel is about to read, one step early.

    ab67 settled that a full-kernel memcheck stays silent on the mixed-TP4
    fault: the access is *inside* tracked allocations, so "invalid address"
    semantics cannot see it -- the page is mapped-but-unbacked (FAULT_PDE).
    This probe reads each about-to-be-read page via a plain copy instead, so
    a dead page faults on OUR host-visible call (with the bid + VA printed
    just before), not inside the Triton kernel. One line per touched block,
    so the last ``TOUCH`` line before the crash names the page.
    """
    vals, scale = cache
    if not _throttle(
        "touch", int(os.environ.get("VLLM_SM8X_GUARD_TOUCH_EVERY", "64"))
    ):
        return
    bids = [b for b in table[loaded].unique().tolist() if b >= 0]
    if not bids:
        return
    n = _counters.get("touch", 0) + 1
    _counters["touch"] = n
    dev = vals.device.index
    for bid in bids:
        va = vals[bid].data_ptr()
        print(
            f"[VLLM_SM8X_GUARD-TOUCH] dev={dev} step={n} bid={bid} "
            f"va=0x{va:x} bytes={vals[bid].numel() * vals.element_size()} "
            f"scale_va=0x{scale[bid].data_ptr():x}",
            flush=True,
        )
        vals[bid].flatten()
        scale[bid].flatten()


def guard_block_table(
    block_table: torch.Tensor,  # (B, max_blocks) physical block ids (or -1)
    context_lens_2d: torch.Tensor,  # (B, next_n) live context length per row
    block_size: int,
    num_blocks: int,
    cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Guard the paged-MQA logits block table (sm8x_mqa.py).

    The kernel reads ``block_table[b, offs_n // block_size]`` only for
    ``offs_n < context_len``, so the columns that can fault are those below
    ``ceil(context_len / block_size)`` per batch.
    """
    if not envs.VLLM_SM8X_GUARD or block_table.numel() == 0:
        return
    if torch.cuda.is_current_stream_capturing():
        # Same capture rule as _guard: the width probe below syncs, which is
        # illegal inside a CUDA graph capture (ab71: stream capture
        # invalidated at the first FULL_DECODE_ONLY run after ab65).
        return
    width = block_table.shape[1]
    cols = torch.arange(width, device=block_table.device).unsqueeze(0)
    allowed = (
        context_lens_2d.max(dim=1).values.to(torch.int64) + block_size - 1
    ) // block_size
    # The kernel indexes ``block_rank = offs_n // block_size`` up to
    # ``ceil(context_len/block_size)-1``; the ``loaded`` mask below only walks
    # the *existing* width columns, so it is blind to a request whose context
    # needs more columns than the block table physically has (block_rank >=
    # width => reads past the row / into the next row = PDE). Catch it directly.
    _dev = block_table.device
    _didx = _dev.index if _dev.index is not None else torch.cuda.current_device()
    _need = int(allowed.max()) if allowed.numel() else 0
    if _didx not in _btw_logged:
        _btw_logged.add(_didx)
        print(
            f"[VLLM_SM8X_GUARD-BTW] dev={_didx} block_table_width={width} "
            f"block_size={block_size} width_servable_tokens={width * block_size} "
            f"max_block_rank_now={max(_need - 1, 0)} max_ctx_now={_need * block_size}",
            flush=True,
        )
    if _need > width:
        raise RuntimeError(
            f"[VLLM_SM8X_GUARD] block_table width overrun: a row needs "
            f"{_need} block columns (context_len up to {_need * block_size}) but "
            f"the table is only width={width} wide (servable="
            f"{width * block_size} tokens, block_size={block_size}). The kernel's "
            f"``block_rank=offs_n//block_size`` reads past the row -> Xid31 "
            f"FAULT_PDE; guard's per-column check cannot see it. shape="
            f"{tuple(block_table.shape)}."
        )
    loaded = cols < allowed.unsqueeze(1)  # (B, max_blocks)
    decode_shaped = block_table.shape[0] * context_lens_2d.shape[1] <= 8
    _every = int(os.environ.get("VLLM_SM8X_GUARD_DECODE_EVERY", "64"))
    _guard(
        block_table,
        loaded,
        num_blocks,
        "paged-MQA block_table",
        every=_every if decode_shaped else 1,
    )
    if os.environ.get("VLLM_SM8X_GUARD_TOUCH") == "1" and cache is not None:
        _touch_pages(cache, block_table, loaded)


def guard_slot_table(
    slots: torch.Tensor,  # (tokens, K) absolute slots phys_block*block_size+pos (or -1)
    lens: torch.Tensor,  # (tokens,) number of valid candidate columns per token
    num_slots: int,
    kind: str,
) -> None:
    """Guard the sparse-MLA attention dequant gather (sm8x_attn.py)."""
    loaded = (
        torch.arange(slots.shape[1], device=slots.device).unsqueeze(0)
        < lens.to(torch.int64).unsqueeze(1)
    )
    _every = int(os.environ.get("VLLM_SM8X_GUARD_DECODE_EVERY", "64"))
    _guard(
        slots,
        loaded,
        num_slots,
        f"dequant {kind} slot table",
        every=_every if slots.shape[0] <= 8 else 1,
    )


_geom_logged: set[int] = set()


def geometry_probe(
    kv_values: torch.Tensor,  # (num_blocks, block_size, 1, head_dim) uint8 view
    kv_scale: torch.Tensor,  # (num_blocks, block_size, 1, s) float32 view
    page_table: torch.Tensor,  # (B, max_blocks) physical block ids (or -1)
    context_lens_2d: torch.Tensor,  # (B, next_n) live context length
) -> None:
    """Bound the rowwise kernel's *strided byte span*, not just the block id.

    ``guard_block_table`` only checks ``block_id < num_blocks``. That is blind to
    two mechanisms that both reproduce the mixed-TP4 Xid31 (5060Ti-first,
    prefix-cache-on *and* -off, guard-silent): (a) a high ``block_id`` whose
    ``block_id * page_stride + in-page offset`` runs past the real storage, and
    (b) an over-committed pool on the smaller rank (KV sized to a larger rank's
    budget, top pages mapped-but-faulted). This probes the exact end address the
    ``tl.load`` reaches for both the values and the scale plane against the
    physical ``untyped_storage().nbytes()``, and logs the full per-rank geometry
    once. Off unless ``VLLM_SM8X_GUARD=1``.
    """
    if not envs.VLLM_SM8X_GUARD or kv_values.numel() == 0:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    storage = kv_values.untyped_storage()
    stor_bytes = storage.nbytes()
    kv_es = kv_values.element_size()
    sc_es = kv_scale.element_size()
    kv_b0 = kv_values.stride(0) * kv_es  # bytes
    kv_b1 = kv_values.stride(1) * kv_es
    kv_bd = kv_values.stride(3) * kv_es
    sc_b0 = kv_scale.stride(0) * sc_es
    sc_b1 = kv_scale.stride(1) * sc_es
    sc_bd = kv_scale.stride(3) * sc_es
    kv_off = kv_values.data_ptr() - storage.data_ptr()
    sc_off = kv_scale.data_ptr() - storage.data_ptr()
    block_size = kv_values.shape[1]

    didx = kv_values.device.index
    if didx is None:
        didx = torch.cuda.current_device()
    if didx not in _geom_logged:
        _geom_logged.add(didx)
        try:
            free_b, tot_b = torch.cuda.mem_get_info(didx)
            alloc_b = torch.cuda.memory_allocated(didx)
        except Exception:  # pragma: no cover - diagnostics only
            free_b = tot_b = alloc_b = -1
        print(
            f"[VLLM_SM8X_GUARD-GEOM] dev={didx} num_blocks={kv_values.shape[0]} "
            f"block_size={block_size} page_stride_b={kv_b0} kv_off={kv_off} "
            f"scale_off={sc_off} scale_tok_b={sc_b1} storage_b={stor_bytes} "
            f"kv_view_tail_b={kv_off + (kv_values.shape[0]-1)*kv_b0 + (block_size-1)*kv_b1 + (kv_values.shape[3]-1)*kv_bd} "
            f"sc_view_tail_b={sc_off + (kv_values.shape[0]-1)*sc_b0 + (block_size-1)*sc_b1 + (kv_scale.shape[3]-1)*sc_bd} "
            f"cuda_free_b={free_b} cuda_total_b={tot_b} torch_alloc_b={alloc_b}",
            flush=True,
        )

    cols = torch.arange(page_table.shape[1], device=page_table.device)
    allowed = (
        context_lens_2d.max(dim=1).values.to(torch.int64) + block_size - 1
    ) // block_size
    loaded = cols.unsqueeze(0) < allowed.unsqueeze(1)
    if not bool(loaded.any()):
        return
    mx = int(page_table[loaded].clamp(min=0).max())
    kv_span = kv_off + mx * kv_b0 + (block_size - 1) * kv_b1 + (kv_values.shape[3] - 1) * kv_bd
    sc_span = sc_off + mx * sc_b0 + (block_size - 1) * sc_b1 + (kv_scale.shape[3] - 1) * sc_bd
    if kv_span >= stor_bytes or sc_span >= stor_bytes:
        raise RuntimeError(
            f"[VLLM_SM8X_GUARD] strided span overrun: max_block_idx={mx} "
            f"(num_blocks={kv_values.shape[0]}) kv_span_b={kv_span} "
            f"scale_span_b={sc_span} >= storage_b={stor_bytes} "
            f"(page_stride_b={kv_b0} scale_page_stride_b={sc_b0} block_size="
            f"{block_size} kv_off={kv_off} scale_off={sc_off} dev={didx}). id is "
            f"in-range but the strided address faults => pool over-commit on this "
            f"rank or a page_stride/view geometry mismatch (would Xid31 FAULT_PDE)."
        )
