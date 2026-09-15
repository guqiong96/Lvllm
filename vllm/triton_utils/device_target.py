# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep Triton's per-device compile target aligned with the device it keys.

``JITFunction.device_caches`` is a ``defaultdict`` whose factory
(``create_binder``) captures ``driver.active.get_current_target()`` — the
*ambient* device — while the entry is keyed by whichever device index is
touched first. On a heterogeneous machine, an entry created for device B
while device A is ambient permanently compiles and launches B's kernels with
A's ``sm_XX``: ``fp8e4nv not supported in this architecture`` /
``no kernel image is available`` / silently wrong codegen. Binding under
``torch.cuda.device(key)`` makes factory and key agree.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALLED = False


def _device_scope(device: Any):
    if not isinstance(device, int) or not torch.cuda.is_available():
        return nullcontext()
    if 0 <= device < torch.cuda.device_count():
        return torch.cuda.device(device)
    return nullcontext()


class _DeviceAlignedCaches(dict):
    """Lazily binds one ``(kernel_cache, key_cache, target, backend, binder)``
    tuple per device, with the ambient device set to the key being bound."""

    def __init__(self, kernel: Any) -> None:
        super().__init__()
        self._kernel = kernel

    def __missing__(self, device: Any) -> Any:
        with _device_scope(device):
            value = self._kernel.create_binder()
        self[device] = value
        bound = value[2].arch
        if isinstance(device, int) and isinstance(bound, int):
            major, minor = torch.cuda.get_device_capability(device)
            real = major * 10 + minor
            if bound != real:
                logger.warning_once(
                    "Triton compile target for device %s is sm_%s but the "
                    "device is sm_%s; check TRITON_OVERRIDE_ARCH.",
                    device,
                    bound,
                    real,
                )
        return value


def install_device_aligned_triton_targets() -> None:
    """Idempotent monkeypatch of ``JITFunction.__init__`` (import-time safe,
    touches no CUDA API)."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from triton.runtime.jit import JITFunction
    except Exception:  # pragma: no cover - triton unavailable/placeholder
        return

    if getattr(JITFunction, "_vllm_device_aligned_caches", False):
        _INSTALLED = True
        return

    orig_init = JITFunction.__init__

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self.device_caches = _DeviceAlignedCaches(self)

    JITFunction.__init__ = __init__  # type: ignore[method-assign]
    JITFunction._vllm_device_aligned_caches = True  # type: ignore[attr-defined]
    _INSTALLED = True


def bound_compile_arch(jit_kernel: Any, device_index: int | None = None) -> Any:
    """The arch Triton's ``run()``/``_do_compile()`` will use for one kernel.

    Read from the kernel's own per-device cache entry (the object
    ``_do_compile`` unpacks), so it reports the real compile target even when
    an ambient target probe would answer for a different card.
    """
    if jit_kernel is None or not torch.cuda.is_available():
        return None
    try:
        from triton import knobs
        from triton.runtime import driver

        from vllm.model_executor.layers.quantization.utils.fp8_emulate import (
            arch_to_int,
        )

        override = knobs.runtime.override_arch
        if override:
            return arch_to_int(override)
        while hasattr(jit_kernel, "fn") and not hasattr(jit_kernel, "device_caches"):
            jit_kernel = jit_kernel.fn
        device = (
            driver.active.get_current_device()
            if device_index is None
            else device_index
        )
        return jit_kernel.device_caches[device][2].arch
    except Exception:  # pragma: no cover - diagnostics must never raise
        return None
