# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep CuTeDSL's compile target aligned with the device this process runs on.

CuTeDSL names its compile arch from ``CUTE_DSL_ARCH``, and when that is unset it
calls ``get_compute_capability_major_minor()`` — whose default is
``cuDeviceGet(0)``, the first *visible* device, never this worker's device (the
DSL's other device queries do use ``cuCtxGetDevice()``; this one path does not).
A TP group spanning mixed architectures therefore compiles every rank for
dev0's arch: on a 3090 + 5060 Ti box the sm120 ranks pass the fp8 gate, emit
``cvt.e4m3x2.f32``, and ptxas aborts because the module says ``.target sm_86``.

Pinning is per process and hardware-anchored: env var for DSL objects built
later, live singleton for one already built. A ``CUTE_DSL_ARCH`` that was in the
environment when this module loaded is treated as user intent (cross-compile)
and only cross-checked against the device; a capability we cannot read leaves
the DSL's own detection alone rather than inventing a target.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CUTEDSL_ARCH_ENV = "CUTE_DSL_ARCH"

# Set before this module can pin anything: a value we write later must not be
# mistaken for the user's own override by the next pin() in this process.
_ENV_AT_IMPORT = os.environ.get(_CUTEDSL_ARCH_ENV)


def cutedsl_arch(device_index: int | None = None) -> str | None:
    """CuTeDSL-style arch string (``sm_86``, ``sm_120a``) for ``device_index``.

    Mirrors the DSL's own naming rule (``a`` suffix at major >= 9). ``None``
    means "cannot tell", which callers must treat as *do not pin*.
    """
    if not torch.cuda.is_available():
        return None
    if device_index is None:
        device_index = torch.cuda.current_device()
    try:
        major, minor = torch.cuda.get_device_capability(device_index)
    except Exception:  # pragma: no cover - unreadable device
        return None
    return f"sm_{major}{minor}" + ("a" if major >= 9 else "")


def _built_dsls() -> list[Any]:
    """DSL singletons that exist in this process (empty until cutlass is used).

    Walking ``BaseDSL`` subclasses keeps this working across DSL versions that
    rename or add a frontend; classes that cannot be materialized are skipped.
    """
    try:
        from cutlass.base_dsl.dsl import BaseDSL
    except Exception:  # pragma: no cover - cutlass not installed/imported
        return []

    dsls: list[Any] = []
    seen: set[type] = set()
    stack = list(BaseDSL.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        try:
            dsl = cls._get_dsl()  # type: ignore[attr-defined]
        except Exception:  # abstract or not constructible with no arguments
            continue
        if getattr(dsl, "envar", None) is not None:
            dsls.append(dsl)
    return dsls


def pin_cutedsl_arch(device_index: int | None = None) -> str | None:
    """Force CuTeDSL's compile arch to the current device. Idempotent.

    Call it once per process before the first ``cute.compile`` (worker startup,
    while cutlass is still unimported) and again at compile time, which is also
    what catches a DSL object already built with dev0's answer.
    """
    from vllm.model_executor.layers.quantization.utils.fp8_emulate import (
        arch_to_int,
    )

    device_arch = cutedsl_arch(device_index)

    if _ENV_AT_IMPORT:
        target = _ENV_AT_IMPORT
        if device_arch is not None and arch_to_int(target) != arch_to_int(device_arch):
            logger.warning_once(
                "%s=%s but the current device is %s; CuTeDSL will compile for "
                "%s. Unset it to use the device's own architecture.",
                _CUTEDSL_ARCH_ENV,
                target,
                device_arch,
                target,
            )
    elif device_arch is None:
        return None  # fail open to the DSL's own detection
    else:
        target = device_arch
        os.environ[_CUTEDSL_ARCH_ENV] = target

    pinned = False
    for dsl in _built_dsls():
        if dsl.envar.arch == target:
            continue
        pinned = True
        dsl.envar.arch = target
        if dsl.envar.arch != target:  # pragma: no cover - setter moved
            raise RuntimeError(
                f"CuTeDSL reports arch {dsl.envar.arch!r} after pinning to "
                f"{target!r}; the DSL's arch plumbing changed. Set "
                f"{_CUTEDSL_ARCH_ENV}={target} explicitly."
            )
    if pinned:
        logger.info_once(
            "Pinned CuTeDSL compile target to %s (device %s): its own detection "
            "answers for visible device 0, which is the wrong card on a "
            "mixed-arch TP group.",
            target,
            torch.cuda.current_device() if device_index is None else device_index,
        )
    return target
