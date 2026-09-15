# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logic/precision checks for the SM120 support functions added for DSV4.1.

All CPU-runnable: these cover the pure-python plumbing around the SM120
sparse-MLA kernels (page sizing, topk-width snapping, short-prefill row
padding, the fp4-indexer arch gate) -- not the CUDA kernels themselves.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse as fis
from vllm.models.deepseek_v4_1.attention import _sm120_paged_block_size
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLABackend
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

SM120 = DeviceCapability(12, 0)
SM121 = DeviceCapability(12, 1)
SM86 = DeviceCapability(8, 6)


def _patch_capability(monkeypatch, cap: DeviceCapability):
    # patch on the class: is_device_capability_family resolves through `cls`
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: cap),
    )


# ---------------------------------------------------------------------------
# _sm120_paged_block_size: block = 64 * tokens_per_state on SM120/121, else None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("ratio", "expect"), [(1, 64), (2, 128), (4, 256), (0, 64)])
def test_paged_block_size_sm120(monkeypatch, ratio, expect):
    _patch_capability(monkeypatch, SM120)
    assert _sm120_paged_block_size(ratio) == expect
    _patch_capability(monkeypatch, SM121)
    assert _sm120_paged_block_size(ratio) == expect


@pytest.mark.parametrize("cap", [SM86, DeviceCapability(9, 0), DeviceCapability(10, 0)])
def test_paged_block_size_off_sm120_is_none(monkeypatch, cap):
    _patch_capability(monkeypatch, cap)
    assert _sm120_paged_block_size(2) is None


# ---------------------------------------------------------------------------
# kernel block-size admission: MultipleOf(64) on SM120/121, unchanged elsewhere
# ---------------------------------------------------------------------------


def _mods(sizes):
    assert len(sizes) == 1
    assert isinstance(sizes[0], MultipleOf)
    assert sizes[0].base == 64


def test_sparse_mla_block_sizes_sm120_multiple_of_64(monkeypatch):
    _patch_capability(monkeypatch, SM120)
    _mods(DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes())
    _patch_capability(monkeypatch, SM86)
    assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [128]
    _patch_capability(monkeypatch, DeviceCapability(9, 0))
    assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [64]


def test_flashinfer_sparse_block_sizes_sm120_multiple_of_64(monkeypatch):
    _patch_capability(monkeypatch, SM120)
    _mods(fis.DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes())
    _patch_capability(monkeypatch, DeviceCapability(10, 0))
    assert (
        fis.DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
        == [128]
    )


# ---------------------------------------------------------------------------
# _sm120_decode_topk_widths: read from the kernel's own dispatch table
# ---------------------------------------------------------------------------


def _fake_decode_dispatch(monkeypatch, table):
    mod = ModuleType("flashinfer.mla._sparse_mla_sm120")
    mod._DECODE_DSV4_DISPATCH = table
    monkeypatch.setitem(sys.modules, "flashinfer.mla._sparse_mla_sm120", mod)


@pytest.fixture(autouse=True)
def _clear_width_cache():
    fis._sm120_decode_topk_widths.cache_clear()
    yield
    fis._sm120_decode_topk_widths.cache_clear()


def test_decode_topk_widths_from_table(monkeypatch):
    _fake_decode_dispatch(monkeypatch, frozenset({(32, 128), (32, 512), (32, 192), (16, 1024)}))
    assert fis._sm120_decode_topk_widths(32) == (128, 192, 512)
    assert fis._sm120_decode_topk_widths(16) == (1024,)
    assert fis._sm120_decode_topk_widths(8) == fis._SM120_DECODE_TOPK_FALLBACK


def test_decode_topk_widths_falls_back_on_broken_table(monkeypatch):
    _fake_decode_dispatch(monkeypatch, "not-a-table")
    assert fis._sm120_decode_topk_widths(32) == fis._SM120_DECODE_TOPK_FALLBACK


def test_real_decode_topk_widths_is_sorted_positive():
    widths = fis._sm120_decode_topk_widths(32)
    assert widths and all(isinstance(w, int) and w > 0 for w in widths)
    assert list(widths) == sorted(widths)


# ---------------------------------------------------------------------------
# _snap_topk_width / _pad_rows_with_row0: losslessness of the two rewrites
# ---------------------------------------------------------------------------


def _prefix_filled_rows(rows: int, width: int, lens: list[int]) -> torch.Tensor:
    """Index rows exactly as the metadata builder emits them: valid ids in a
    prefix, -1 padding for the rest."""
    idx = torch.full((rows, width), -1, dtype=torch.int32)
    for i, ln in enumerate(lens):
        idx[i, :ln] = torch.arange(ln, dtype=torch.int32)
    return idx


def test_snap_clip_is_lossless_for_prefix_rows():
    lens = [1, 40, 63, 64]
    idx = _prefix_filled_rows(4, 1152, lens)
    snapped = fis._snap_topk_width(idx, 128)
    assert snapped.shape == (4, 128)
    for i, ln in enumerate(lens):
        assert torch.equal(snapped[i, :ln], idx[i, :ln])
        assert (snapped[i, ln:] == -1).all()


def test_snap_pad_extends_with_minus_one_and_is_lossless():
    lens = [2, 33]
    idx = _prefix_filled_rows(2, 64, lens)
    snapped = fis._snap_topk_width(idx, 128)
    assert snapped.shape == (2, 128)
    assert torch.equal(snapped[:, :64], idx)
    assert (snapped[:, 64:] == -1).all()


def test_snap_equal_width_is_identity():
    idx = _prefix_filled_rows(2, 128, [5, 100])
    assert fis._snap_topk_width(idx, 128) is idx


def test_pad_rows_replays_row0():
    rows = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    padded = fis._pad_rows_with_row0(rows, 2)
    assert padded.shape == (5, 4)
    assert torch.equal(padded[:3], rows)
    assert torch.equal(padded[3], rows[0])
    assert torch.equal(padded[4], rows[0])


# ---------------------------------------------------------------------------
# _sm120_prefill_dispatch_view: the full dispatch matrix
# ---------------------------------------------------------------------------

DECODE_W = (128, 192, 256, 512, 1024)


@pytest.fixture()
def stub_decode_widths(monkeypatch):
    monkeypatch.setattr(fis, "_sm120_decode_topk_widths", lambda _heads: DECODE_W)


def _call(q, idx, lens, out, *, window, has_image=False):
    return fis._sm120_prefill_dispatch_view(
        q, idx, lens, out, window_size=window, has_image=has_image
    )


def _mk(rows, width, heads=32):
    q = torch.randn(rows, heads, 1, dtype=torch.bfloat16)
    idx = torch.full((rows, width), -1, dtype=torch.int32)
    lens = torch.full((rows,), 1, dtype=torch.int32)
    out = torch.zeros(rows, heads, 1, dtype=torch.bfloat16)
    return q, idx, lens, out


def test_dispatch_zero_rows_passthrough(stub_decode_widths):
    q, idx, lens, out = _mk(0, 128)
    assert _call(q, idx, lens, out, window=128) == (q, idx, lens, out, None, 0)


def test_dispatch_short_row_already_compiled_width_is_identity(stub_decode_widths):
    q, idx, lens, out = _mk(2, 256)
    r = _call(q, idx, lens, out, window=128)
    assert r[1] is idx and r[4] is None and r[5] == 0


def test_dispatch_wide_row_already_compiled_width_is_identity(stub_decode_widths):
    q, idx, lens, out = _mk(65, 2048)
    r = _call(q, idx, lens, out, window=128)
    assert r[1] is idx and r[4] is None and r[5] == 0


def test_dispatch_short_wide_row_snaps_to_decode_width(stub_decode_widths):
    # 1152-wide VL row on a short (<= 64 token) prefill: no decode kernel for
    # 1152, no image -> static bound window=128 -> snap to smallest cover (128).
    q, idx, lens, out = _mk(2, 1152)
    lens = torch.full((2,), 128, dtype=torch.int32)
    r = _call(q, idx, lens, out, window=128)
    snapped, copy_back, pad = r[1], r[4], r[5]
    assert snapped.shape[-1] == 128 and copy_back is None and pad == 0
    # lossless: every valid candidate (prefix < lens <= window) survived
    assert torch.equal(snapped[:, :128], idx[:, :128])


def test_dispatch_image_rows_snap_from_metadata_bound(stub_decode_widths):
    q, idx, lens, out = _mk(8, 1152)
    lens = torch.tensor([10, 300, 299, 1, 1, 1, 1, 1], dtype=torch.int32)
    r = _call(q, idx, lens, out, window=128, has_image=True)
    snapped, copy_back, pad = r[1], r[4], r[5]
    # needed = max(lens) = 300 -> smallest compiled width >= 300 is 512
    assert snapped.shape[-1] == 512 and copy_back is None and pad == 0
    for i in range(8):
        ln = int(lens[i])
        assert torch.equal(snapped[i, :ln], idx[i, :ln])
        assert (snapped[i, ln:] == -1).all()


def test_dispatch_short_row_beyond_decode_list_pads_into_orchestrator(
    stub_decode_widths,
):
    # window 2048 exceeds every decode width (max 1024): no decode cover, so
    # pad to 65 rows (crosses the 64-token threshold) and snap on the
    # prefill list (2048 covers 2048).
    rows = 4
    q, idx, lens, out = _mk(rows, 1152)
    lens = torch.full((rows,), 2048, dtype=torch.int32)
    r = _call(q, idx, lens, out, window=2048)
    q2, idx2, lens2, out2, copy_back, pad = r
    assert pad == 65 - rows
    assert q2.shape[0] == 65 and idx2.shape[0] == 65 and lens2.shape[0] == 65
    assert idx2.shape[-1] == 2048  # snapped on the prefill list
    assert torch.equal(idx2[:rows, :1152], idx)  # pad-snap is lossless
    assert (idx2[:rows, 1152:] == -1).all()
    assert copy_back is out and copy_back.shape[0] == rows
    # scratch rows replay row 0 (kernel-legal: row 0 is a valid row)
    assert torch.equal(idx2[rows:], idx2[:1].expand(pad, 2048))
    assert torch.equal(q2[rows], q[0])
    assert int(lens2[rows]) == int(lens[0])
    # the caller's copy-back recovers exactly the real rows region
    out2.fill_(-1.0)
    copy_back.copy_(out2[:rows])
    assert (out == -1.0).all()


def test_dispatch_wide_row_snaps_to_prefill_width(stub_decode_widths):
    # > 64 tokens: orchestrator region; 1152 is not compiled there either ->
    # snap to 128 (smallest prefill width >= window) with NO row padding.
    q, idx, lens, out = _mk(100, 1152)
    lens = torch.full((100,), 128, dtype=torch.int32)
    r = _call(q, idx, lens, out, window=128)
    snapped, copy_back, pad = r[1], r[4], r[5]
    assert snapped.shape == (100, 128) and copy_back is None and pad == 0


def test_dispatch_wide_row_no_specialization_raises(stub_decode_widths):
    q, idx, lens, out = _mk(100, 4096)
    with pytest.raises(RuntimeError, match="no compiled specialization"):
        _call(q, idx, lens, out, window=4096)


# ---------------------------------------------------------------------------
# the path assertions themselves (must actually fire, not be no-ops)
# ---------------------------------------------------------------------------


def test_assertion_fires_when_clip_would_drop_a_candidate(stub_decode_widths):
    # prefix-fill premise violated: a live candidate sits at column 200
    q, idx, lens, out = _mk(2, 1152)
    idx[1, 200] = 7
    with pytest.raises(AssertionError, match="would drop"):
        _call(q, idx, lens, out, window=128)


def test_path_checks_env_can_disable(monkeypatch, stub_decode_widths):
    monkeypatch.setattr(envs, "VLLM_DSV41_SM120_PATH_CHECKS", False)
    q, idx, lens, out = _mk(2, 1152)
    idx[1, 200] = 7  # would trip the lossless check if it were on
    snapped = _call(q, idx, lens, out, window=128)[1]
    assert snapped.shape[-1] == 128  # silently clipped, old behavior


# ---------------------------------------------------------------------------
# numerical equivalence: snap/pad are no-ops for any kernel obeying the
# (-1 invalid, prefix length) contract -- checked against a torch reference
# ---------------------------------------------------------------------------


def _ref_row_outputs(
    idx: torch.Tensor, lens: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Reference: out[i] = sum over valid columns of weights[i, idx[i, j]].
    Valid == (id >= 0) and (col < lens[i]) -- exactly the SM120 kernel
    contract the snap/pad transforms claim not to disturb."""
    rows, width = idx.shape
    cols = torch.arange(width).unsqueeze(0)
    valid = (idx >= 0) & (cols < lens.unsqueeze(1))
    safe = idx.clamp(min=0)
    contrib = torch.gather(weights[:rows], 1, safe.unsqueeze(-1).expand(-1, -1, 4))
    return (contrib * valid.unsqueeze(-1)).sum(1)


@pytest.mark.parametrize(
    ("cur_width", "window", "rows", "has_image"),
    [
        (1152, 128, 2, False),  # clip snap
        (64, 64, 2, False),  # pad snap (indices narrower than the table)
        (1152, 128, 8, True),  # metadata-bound snap
        (1152, 2048, 4, False),  # pad rows + prefill snap
    ],
)
def test_transform_is_numerically_identical_to_reference(
    stub_decode_widths, cur_width, window, rows, has_image
):
    torch.manual_seed(0)
    q = torch.randn(rows, 32, 1, dtype=torch.bfloat16)
    lens = torch.randint(1, window + 1, (rows,), dtype=torch.int32).clamp(
        max=cur_width
    )
    idx = _prefix_filled_rows(rows, cur_width, lens.tolist())
    out = torch.zeros(rows, 32, 1, dtype=torch.bfloat16)
    # candidate weights only exist for columns the kernel may consult
    weights = torch.randn(rows, max(cur_width, 2048), 4)
    ref = _ref_row_outputs(idx, lens, weights)

    q2, idx2, lens2, out2, copy_back, pad = _call(
        q, idx, lens, out, window=window, has_image=has_image
    )
    if copy_back is not None:
        # scratch rows only replay row 0 (content checked elsewhere); the
        # reference compares the real rows, which is what gets copied back.
        idx2, lens2 = idx2[:rows], lens2[:rows]
    got = _ref_row_outputs(idx2, lens2, weights)
    # permuting/scaling by -1 padding and scratch rows cannot move the result
    assert torch.equal(ref, got)


def test_dispatch_padded_rows_cross_decode_threshold(stub_decode_widths):
    # invariant: after padding we are ALWAYS on the > 64-token side (the
    # decode region is where padding was illegal).
    for rows in (1, 2, 8, 33, 64):
        q, idx, lens, out = _mk(rows, 1152)
        r = _call(q, idx, lens, out, window=8192 - 64)
        pad = r[5]
        assert rows + pad == 65


# ---------------------------------------------------------------------------
# dsa_indexer_uses_fp4 arch gate (sm120 fp4 indexer un-lock)
# ---------------------------------------------------------------------------


def _fake_cfg(kv_dtype):
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda _default: kv_dtype
        )
    )


@pytest.mark.parametrize("cap", [DeviceCapability(10, 0), SM120, SM121])
def test_fp4_indexer_gate_allows_blackwell(monkeypatch, cap):
    _patch_capability(monkeypatch, cap)
    assert dsa_indexer_uses_fp4(_fake_cfg("mxfp4")) is True


@pytest.mark.parametrize("cap", [SM86, DeviceCapability(9, 0)])
def test_fp4_indexer_gate_fail_closed_older_arch(monkeypatch, cap):
    _patch_capability(monkeypatch, cap)
    with pytest.raises(ValueError, match="requires Blackwell"):
        dsa_indexer_uses_fp4(_fake_cfg("mxfp4"))


@pytest.mark.parametrize("cap", [SM86, SM120])
def test_fp8_indexer_never_gated(monkeypatch, cap):
    _patch_capability(monkeypatch, cap)
    assert dsa_indexer_uses_fp4(_fake_cfg("fp8")) is False


def test_fp4_indexer_gate_unknown_dtype_raises(monkeypatch):
    _patch_capability(monkeypatch, SM120)
    with pytest.raises(ValueError, match="not supported"):
        dsa_indexer_uses_fp4(_fake_cfg("int4"))


# ---------------------------------------------------------------------------
# wo_a einsum input fingerprint (SF post-load must have run)
# ---------------------------------------------------------------------------


def _wo_a_probe():
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        _check_wo_a_einsum_inputs,
    )

    return _check_wo_a_einsum_inputs


def test_wo_a_sf_fingerprint_accepts_transformed_inputs():
    chk = _wo_a_probe()
    wo_a = SimpleNamespace(prefix="t.wo_a.ok")
    w = torch.zeros(2, 4, 256, dtype=torch.float8_e4m3fn)
    sf = torch.zeros(2, 1, 8, dtype=torch.int32)  # grouped packed UE8M0
    act = torch.zeros(3, 2, 1, dtype=torch.int32)
    chk(wo_a, w, sf, act, True)  # no raise
    chk(wo_a, w, sf.float(), act.float(), False)  # sm90 fp32 layout, no raise


def test_wo_a_sf_fingerprint_rejects_raw_ckpt_scale():
    chk = _wo_a_probe()
    wo_a = SimpleNamespace(prefix="t.wo_a.raw")
    w = torch.zeros(2, 4, 256, dtype=torch.float8_e4m3fn)
    raw = torch.zeros(8, 2, dtype=torch.uint8)  # 2D ckpt scale, never transformed
    act = torch.zeros(3, 2, 1, dtype=torch.int32)
    with pytest.raises(AssertionError, match="RAW ckpt scale"):
        chk(wo_a, w, raw, act, True)


def test_wo_a_sf_fingerprint_rejects_unreshaped_weight():
    chk = _wo_a_probe()
    wo_a = SimpleNamespace(prefix="t.wo.a.w2d")
    w = torch.zeros(8, 256, dtype=torch.float8_e4m3fn)  # not grouped 3D
    sf = torch.zeros(2, 1, 8, dtype=torch.int32)
    act = torch.zeros(3, 2, 1, dtype=torch.int32)
    with pytest.raises(AssertionError, match="grouped"):
        chk(wo_a, w, sf, act, True)


# ---------------------------------------------------------------------------
# #56659 python-side stride guard (packed == pre/post-fix equivalence)
# ---------------------------------------------------------------------------


def _pack_operands():
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        _pack_sm120_operands,
    )

    return _pack_sm120_operands


def test_stride_guard_canonicalizes_strided_A():
    pack = _pack_operands()
    base = torch.randn(16, 64)
    a_view = base[:, :32]  # stride (64,1) for shape (16,32): leading != width
    a, _ = pack(a_view, torch.randn(4, 8))
    assert a.stride() == (32, 1)
    assert torch.equal(a, a_view)


def test_stride_guard_canonicalizes_strided_bt():
    pack = _pack_operands()
    weight = torch.randn(64, 24)  # [K, N] storage
    b_t = weight.T[4:36, :]  # sliced transpose view, leading stride != rows
    _, b = pack(torch.randn(2, 2), b_t)
    assert b.stride(0) == 1 and b.stride(1) == b.shape[0]


def test_stride_guard_packed_inputs_untouched():
    pack = _pack_operands()
    a = torch.randn(5, 16)  # packed row-major
    b_t = torch.randn(16, 4).T  # packed col-major view
    a2, b2 = pack(a, b_t)
    assert a2 is a and b2 is b_t


# ---------------------------------------------------------------------------
# MXFP8 BMM arch gate (relaxed to consumer Blackwell)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [DeviceCapability(10, 0), SM120, SM121])
def test_mxfp8_bmm_gate_allows_blackwell(monkeypatch, cap):
    from vllm.model_executor.kernels.linear.mxfp8 import deep_gemm as dg

    monkeypatch.setattr(dg, "is_deep_gemm_supported", lambda: True)
    _patch_capability(monkeypatch, cap)
    ok, _ = dg.DeepGemmMxfp8BmmLinearKernel.is_supported()
    assert ok


@pytest.mark.parametrize("cap", [SM86, DeviceCapability(9, 0)])
def test_mxfp8_bmm_gate_fail_closed_elsewhere(monkeypatch, cap):
    from vllm.model_executor.kernels.linear.mxfp8 import deep_gemm as dg

    monkeypatch.setattr(dg, "is_deep_gemm_supported", lambda: True)
    _patch_capability(monkeypatch, cap)
    ok, reason = dg.DeepGemmMxfp8BmmLinearKernel.is_supported()
    assert not ok and "Blackwell" in reason


def test_mxfp8_bmm_gate_requires_deepgemm(monkeypatch):
    from vllm.model_executor.kernels.linear.mxfp8 import deep_gemm as dg

    monkeypatch.setattr(dg, "is_deep_gemm_supported", lambda: False)
    _patch_capability(monkeypatch, SM120)
    ok, reason = dg.DeepGemmMxfp8BmmLinearKernel.is_supported()
    assert not ok and "supported CUDA device" in reason
