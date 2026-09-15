"""fp8 e4m3 bit helpers for GPUs whose Triton has no ``fp8e4nv`` type.

Triton before Ada (SM89) cannot even name ``tl.float8e4nv``: both a pointer to
it and a value cast fail during ``to_ir`` with ``type fp8e4nv not supported in
this architecture``. Kernels gated by :func:`fp8_native_supported` take raw
uint8 pointers instead and use these helpers to decode an e4m3 byte, round a
float to the nearest e4m3 value, or encode a float to an e4m3 byte — entirely
in int/fp32 so the fp8 type is never materialized. SM89+ keeps its native
hardware path unchanged.

Bit-exact reference: ``torch.tensor(x).to(torch.float8_e4m3fn)`` (RNE,
saturating to 448, NaN -> NaN byte). Ported from the sglang sm80 work
(``kernels/ops/quantization/fp8_emulate.py`` plus the robust encode that
replaces the buggy ``tl.log2`` exponent version).
"""

import re
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_ARCH_RE = re.compile(r"^sm_?(\d+)$|^\s*(\d+)\s*$")


def arch_to_int(arch: object) -> int | None:
    """Normalize anything that names an SM arch (int, ``sm_86``, ``86``) to int."""
    if arch is None or isinstance(arch, bool):
        return None
    if isinstance(arch, int):
        return arch
    if isinstance(arch, str):
        match = _ARCH_RE.match(arch.strip())
        if match:
            return int(match.group(1) or match.group(2))
    return None


@lru_cache(maxsize=8)
def _fp8_native_supported_indexed(device_index: int) -> bool:
    if not torch.cuda.is_available():
        return True
    return torch.cuda.get_device_capability(device_index) >= (8, 9)


def fp8_native_supported(device_index: int | None = None) -> bool:
    """Whether Triton can form ``fp8e4nv`` pointers / casts (SM89+).

    Cached per resolved device index: mixed-arch TP groups must not answer for
    the wrong card. ``None`` means "the ambient device" and is resolved before
    caching — caching it under the literal ``None`` key would freeze the first
    caller's answer for every later device.
    """
    if not torch.cuda.is_available():
        return True
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _fp8_native_supported_indexed(device_index)


@lru_cache(maxsize=8)
def _triton_target_arch(device_index: int) -> object:
    """The arch Triton would compile for on ``device_index``, as an int or None.

    Read through Triton itself (``driver.active.get_current_target()``, plus
    ``TRITON_OVERRIDE_ARCH`` which ``parse_options`` prefers), so the answer
    cannot diverge from what ptxas actually receives. An override Triton cannot
    parse is reported as None (unknown), not as "supported".
    """
    from triton import knobs
    from triton.runtime import driver

    override = knobs.runtime.override_arch
    if override:
        return arch_to_int(override)
    with torch.cuda.device(device_index):
        return arch_to_int(driver.active.get_current_target().arch)


def fp8_triton_target_arch(device_index: int | None = None) -> object:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _triton_target_arch(device_index)


def fp8_triton_target_supported(device_index: int | None = None) -> bool:
    """Whether a Triton launch on ``device_index`` can materialize fp8e4nv.

    Index-mapping bugs (NVML vs CUDA enumeration, visibility order, ambient
    device, ``TRITON_OVERRIDE_ARCH``) cannot make this diverge from what
    actually compiles, because it asks the compiler's own question. Unknown
    archs fail closed (bf16/emulation is always numerically legal here).
    """
    arch = fp8_triton_target_arch(device_index)
    return arch is not None and arch >= 89


def fp8_triton_kernel_target_supported(jit_kernel: object) -> bool:
    """Whether launching ``jit_kernel`` right now can materialize fp8e4nv.

    Both views of the compile target must agree: the ambient one (what
    ``run()`` keys its device cache by) and the one already baked into this
    kernel's cache entry (what ``_do_compile()`` compiles with).
    """
    if not fp8_triton_target_supported():
        return False
    if not torch.cuda.is_available():
        return True
    from vllm.triton_utils.device_target import bound_compile_arch

    arch = arch_to_int(bound_compile_arch(jit_kernel))
    return arch is not None and arch >= 89


@triton.jit
def e4m3fn_u8_to_f32(u):
    """Decode an ``e4m3fn`` byte (1-4-3, exp bias 7) to f32. NaN (S.1111.111)
    decodes to +/-480 (never stored by a saturating quantizer)."""
    ui = u.to(tl.int32)
    sign = (ui >> 7) & 1
    exp = (ui >> 3) & 0xF
    man = ui & 0x7
    mant = man.to(tl.float32) * 0.125
    val = tl.where(
        exp != 0,
        tl.exp2((exp - 7).to(tl.float32)) * (1.0 + mant),
        0.015625 * mant,
    )
    return tl.where(sign != 0, -val, val)


@triton.jit
def round_to_e4m3fn_f32(x):
    """Round ``x`` to the nearest ``e4m3fn`` value, returned as f32 — the
    value cast ``x.to(tl.float8e4nv).to(tl.float32)`` without that type.

    RNE via ``rint`` on the quantized mantissa. The exponent comes from the
    IEEE bits (exact at binade boundaries, unlike ``log2``), clamped to the
    e4m3 minimum normal exponent (-6) so the subnormal range uses a fixed
    2**-9 grid. ``|x|`` saturates to 448 first.
    """
    a = tl.minimum(tl.abs(x), 448.0)
    bits = a.to(tl.int32, bitcast=True)
    e32 = (bits >> 23) & 0xFF
    is_sub = bits < 0x00800000
    e = tl.where(is_sub, -126, e32 - 127)
    e_step = tl.maximum(e, -6)
    step = tl.exp2(e_step.to(tl.float32) - 3.0)
    q = libdevice.rint(a / step)
    mag = q * step
    return tl.where(x < 0, -mag, mag)


@triton.jit
def f32_to_e4m3fn_u8(x):
    """Encode ``x`` (f32) to an ``e4m3fn`` byte (uint8). Saturating: |x| > 448
    -> +/-448 (0x7E/0xFE); NaN -> 0x7F. Exponent from IEEE bit patterns, not
    ``tl.log2`` (which rounds up just below powers of two and corrupts values
    like 127.99999 -> 0.0156; real bug in the vllm-ds4 original).
    """
    a = tl.minimum(tl.abs(x), 448.0)
    bits = a.to(tl.int32, bitcast=True)
    e32 = (bits >> 23) & 0xFF
    e_b = tl.where(e32 == 0, -126, e32 - 127)  # unbiased exponent of |a|
    normal = e_b >= -6
    # normal path: q in [8, 16), carry into the next binade at q == 16.
    step = tl.exp2(tl.maximum(e_b, -6).to(tl.float32) - 3.0)
    q = libdevice.rint(a / step)
    carry = q >= 16.0
    e_c = tl.where(carry, e_b + 1, e_b)
    q = tl.where(carry, 8.0, q)
    qi = q.to(tl.int32)
    mag_byte = tl.where(
        normal,
        ((e_c + 7) << 3) | (qi & 0x7),
        tl.where(qi >= 8, 0x08 | (qi - 8), qi),
    )
    is_nan = x != x
    # sign from the IEEE bit (preserves -0.0 / -NaN, unlike x < 0)
    sb = tl.where(x.to(tl.int32, bitcast=True) < 0, 0x80, 0)
    out = tl.where(is_nan, 0x7F, mag_byte) | sb
    return out.to(tl.uint8)
