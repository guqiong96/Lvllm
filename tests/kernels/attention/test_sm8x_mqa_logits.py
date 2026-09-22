# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM8x non-paged (prefill indexer) MQA-logits Triton path.

The Triton kernel replaced the chunked torch reference as the fast path;
these check numerical parity with the reference (which defines semantics),
the -inf window contract the windowed top-k consumers rely on, and the
dtype-gated dispatch/fallback in ``fp8_mqa_logits_sm8x``.
"""

import pytest
import torch

requires_fp8 = pytest.mark.skipif(
    not (torch.cuda.is_available() and hasattr(torch, "float8_e4m3fn")),
    reason="CUDA with float8_e4m3fn support is required",
)

H, D = 64, 128


def _make(M, N, seed=0):
    from vllm.utils.torch_utils import set_random_seed

    set_random_seed(seed)
    dev = "cuda"
    q = torch.randn(M, H, D, device=dev).clamp(-4, 4).to(torch.float8_e4m3fn)
    k = torch.randn(N, D, device=dev).clamp(-4, 4).to(torch.float8_e4m3fn)
    ks = torch.zeros(M, device=dev, dtype=torch.int32)
    ke = torch.arange(1, M + 1, device=dev, dtype=torch.int32).clamp(max=N)
    w = torch.rand(M, H, device=dev) + 0.1
    scale = torch.rand(N, 1, device=dev) + 0.5
    return q, (k, scale), w, ks, ke


def _window(M, N, ks, ke):
    n = torch.arange(N, device="cuda")[None, :]
    return (n >= ks[:, None]) & (n < ke[:, None])


@requires_fp8
@pytest.mark.parametrize(("M", "N"), [(64, 64), (512, 300), (2048, 2048)])
def test_prefill_triton_matches_torch_reference(M, N):
    from vllm.models.deepseek_v4.nvidia.ops.sm8x_mqa import (
        fp8_mqa_logits_sm8x_triton,
        fp8_mqa_logits_torch,
    )

    q, (k, scale), w, ks, ke = _make(M, N)
    ref = fp8_mqa_logits_torch(q, k, scale, w, ks, ke, clean_logits=False)
    out = fp8_mqa_logits_sm8x_triton(q, (k, scale), w, ks, ke)

    vin = _window(M, N, ks, ke)
    rel = ((out - ref).abs()[vin] / (ref[vin].abs() + 1e-3)).max().item()
    assert rel < 5e-2, f"in-window maxrel {rel:.2e}"
    assert torch.isinf(out[~vin]).all() and (out[~vin] < 0).all()


@requires_fp8
def test_prefill_reuses_provided_logits_buffer():
    from vllm.models.deepseek_v4.nvidia.ops.sm8x_mqa import (
        fp8_mqa_logits_sm8x_triton,
        fp8_mqa_logits_torch,
    )

    M, N = 256, 300
    q, (k, scale), w, ks, ke = _make(M, N)
    fresh = fp8_mqa_logits_sm8x_triton(q, (k, scale), w, ks, ke)
    buf = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
    out = fp8_mqa_logits_sm8x_triton(q, (k, scale), w, ks, ke, logits=buf)
    assert out.data_ptr() == buf.data_ptr()
    assert torch.equal(out, fresh)
    with pytest.raises(ValueError):
        fp8_mqa_logits_sm8x_triton(q, (k, scale), w, ks, ke, logits=buf[:1])

    ref_fresh = fp8_mqa_logits_torch(q, k, scale, w, ks, ke, clean_logits=True)
    ref_buf = fp8_mqa_logits_torch(
        q, k, scale, w, ks, ke, clean_logits=True, out=buf
    )
    assert ref_buf.data_ptr() == buf.data_ptr()
    assert torch.equal(ref_buf, ref_fresh)


@requires_fp8
def test_empty_shapes():
    from vllm.models.deepseek_v4.nvidia.ops.sm8x_mqa import (
        fp8_mqa_logits_sm8x_triton,
    )

    q, kv, w, ks, ke = _make(16, 16)
    assert fp8_mqa_logits_sm8x_triton(q[:0], kv, w[:0], ks[:0], ke[:0]).shape == (0, 16)


@requires_fp8
def test_dispatch_uses_triton_for_fp8_and_falls_back_otherwise(monkeypatch):
    from vllm.models.deepseek_v4.nvidia.ops import sm8x_mqa

    q, kv, w, ks, ke = _make(32, 32)
    calls = []
    monkeypatch.setattr(
        sm8x_mqa,
        "fp8_mqa_logits_sm8x_triton",
        lambda *a: calls.append("triton") or torch.empty((32, 32), device="cuda"),
    )
    sm8x_mqa.fp8_mqa_logits_sm8x(q, kv, w, ks, ke)
    assert calls == ["triton"]

    # Non-fp8 K (e.g. a bf16 debug buffer) must not enter the uint8 kernel.
    kv_bf16 = (kv[0].float().to(torch.bfloat16), kv[1])
    sm8x_mqa.fp8_mqa_logits_sm8x(q, kv_bf16, w, ks, ke)
    assert calls == ["triton"]
