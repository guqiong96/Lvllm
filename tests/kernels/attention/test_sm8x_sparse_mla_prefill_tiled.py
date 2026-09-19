# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiled SM8x sparse-MLA prefill parity.

The tiled path (query chunk x topk chunk x head block) must match the
per-(token, head) kernel it replaces, which defines the contract: flat
gathered-row indices, valid-first with -1 padding, per-query lens, attn-sink
merged with zero value.
"""

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

D = 512


def _make_case(T, K, H, H_pad, seed, lens=None, device="cuda"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, H_pad, D, device=device, generator=g).clamp(-3, 3).bfloat16()
    R = 4 * T + 8
    kv = torch.randn(R, D, device=device, generator=g).bfloat16()
    if lens is None:
        lens = torch.randint(0, K + 1, (T,), device=device, generator=g, dtype=torch.int32)
    idx = torch.full((T, K), -1, device=device, dtype=torch.int32)
    for t in range(T):
        n = int(lens[t])
        if n:
            idx[t, :n] = torch.randint(
                0, R, (n,), device=device, generator=g, dtype=torch.int32
            )
    sink = torch.randn(H, device=device, generator=g) - 1.0
    return q, kv, idx, lens, sink


def _ref(q, kv, idx, lens, sink, scale, H):
    # fp32 reference: softmax over valid rows plus a zero-value sink pseudo-token.
    T = q.shape[0]
    out = torch.zeros(T, H, D, device=q.device, dtype=torch.float32)
    for t in range(T):
        n = int(lens[t])
        logits = torch.full((H, n + 1), float("-inf"), device=q.device)
        if n:
            rows = idx[t, :n].long()
            logits[:, :n] = (q[t, :H].float() @ kv[rows].float().T) * scale
        logits[:, n] = sink.float().clamp(min=-1e30)
        p = torch.softmax(logits, dim=-1)[:, :n].unsqueeze(-1)
        out[t] = (p * kv[idx[t, :n].long()].float()).sum(1) if n else 0.0
    return out


@pytest.mark.parametrize("H,H_pad", [(16, 16), (4, 8), (8, 64)])
@requires_cuda
def test_matches_reference_and_zeroes_padded_heads(H, H_pad):
    from vllm.models.deepseek_v4.nvidia.ops.sm8x_sparse_mla_prefill import (
        sm8x_tiled_sparse_mla_prefill,
    )

    q, kv, idx, lens, sink = _make_case(T=300, K=700, H=H, H_pad=H_pad, seed=7)
    out = torch.zeros(q.shape, device=q.device, dtype=torch.bfloat16)
    sm8x_tiled_sparse_mla_prefill(
        q=q, kv_flat=kv, indices=idx, lens=lens, scale=0.0442,
        attn_sink=sink, output=out, num_heads=H,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(out[:, H:]) == 0
    if H == 16:  # multihead variant vs fp32 reference
        ref = _ref(q, kv, idx, lens, sink, 0.0442, H)
        assert (out[:, :H].float() - ref).abs().max().item() < 2e-2


@requires_cuda
def test_matches_legacy_kernel():
    from vllm.models.deepseek_v4.common.ops.sparse_mla_kernels import (
        sparse_mla_fwd_with_sink,
    )
    from vllm.models.deepseek_v4.nvidia.ops.sm8x_sparse_mla_prefill import (
        sm8x_tiled_sparse_mla_prefill,
    )

    H = 16
    q, kv, idx, lens, sink = _make_case(T=1024, K=512, H=H, H_pad=H, seed=11)
    out_old = torch.zeros(q.shape, device=q.device, dtype=torch.bfloat16)
    out_new = torch.zeros(q.shape, device=q.device, dtype=torch.bfloat16)
    sparse_mla_fwd_with_sink(
        q=q, kv=kv, indices=idx, topk_length=lens, scale=0.0442,
        attn_sink=sink, output=out_old, num_heads=H,
    )
    sm8x_tiled_sparse_mla_prefill(
        q=q, kv_flat=kv, indices=idx, lens=lens, scale=0.0442,
        attn_sink=sink, output=out_new, num_heads=H,
    )
    torch.cuda.synchronize()
    diff = (out_new.float() - out_old.float()).abs()
    assert diff.max().item() < 5e-3


@requires_cuda
def test_all_empty_lens_is_sink_only_zero():
    from vllm.models.deepseek_v4.nvidia.ops.sm8x_sparse_mla_prefill import (
        sm8x_tiled_sparse_mla_prefill,
    )

    H = 16
    q, kv, idx, _, sink = _make_case(
        T=64, K=64, H=H, H_pad=H, seed=3,
        lens=torch.zeros(64, device="cuda", dtype=torch.int32),
    )
    out = torch.full((64, H, D), 7.0, device=q.device, dtype=torch.bfloat16)
    sm8x_tiled_sparse_mla_prefill(
        q=q, kv_flat=kv, indices=idx, lens=torch.zeros_like(idx[:, 0]),
        scale=0.0442, attn_sink=sink, output=out, num_heads=H,
    )
    torch.cuda.synchronize()
    assert out.abs().max().item() == 0.0


def test_env_escape_hatch(monkeypatch):
    from vllm.models.deepseek_v4.nvidia.ops.sm8x_sparse_mla_prefill import (
        sm8x_tiled_prefill_enabled,
    )

    monkeypatch.delenv("VLLM_DSV4_SM8X_TILED_PREFILL", raising=False)
    assert sm8x_tiled_prefill_enabled()
    monkeypatch.setenv("VLLM_DSV4_SM8X_TILED_PREFILL", "0")
    assert not sm8x_tiled_prefill_enabled()
