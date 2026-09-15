# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.fp8_emulate import (
    fp8_native_supported,
    fp8_triton_target_arch,
)
from vllm.model_executor.utils import replace_parameter
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    _FUSED_INV_ROPE_FP8_QUANT_KERNEL,
    fp8_quant_target_supported,
    fused_inv_rope_fp8_quant,
)
from vllm.triton_utils.device_target import bound_compile_arch
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum
from vllm.v1.attention.ops.flashmla import sm8x_sparse_mla_enabled

logger = init_logger(__name__)

_wo_a_sf_logged: set[str] = set()


def _check_wo_a_einsum_inputs(
    wo_a: nn.Module, weight: torch.Tensor, weight_scale: torch.Tensor, o_scale: torch.Tensor,
    tma_aligned: bool,
) -> None:
    """Once per wo_a layer: verify the fp8_einsum contract the SM120 native
    path relies on -- the DeepGEMM block kernel's post-load must have run the
    grouped SF transform (sglang ``_setup_fp8_wo_a_scales`` equivalent).
    A raw ckpt scale (2D, e8m0/uint8 bytes) means the SF post-load never ran
    (wrong linear-kernel selection) and the einsum reads scales from the
    wrong place -- silent per-layer garbage from token 0 on."""
    name = getattr(wo_a, "prefix", None) or str(id(wo_a))
    first = name not in _wo_a_sf_logged
    if first:
        _wo_a_sf_logged.add(name)
        logger.info(
            "wo_a einsum inputs [%s]: weight %s %s scale %s %s str=%s; "
            "act scale %s %s",
            name,
            tuple(weight.shape),
            weight.dtype,
            tuple(weight_scale.shape),
            weight_scale.dtype,
            tuple(weight_scale.stride()),
            tuple(o_scale.shape),
            o_scale.dtype,
        )
    if not envs.VLLM_DSV41_SM120_PATH_CHECKS:
        return
    assert weight.ndim == 3, (
        f"wo_a weight must be the grouped [G,R,D] view for fp8_einsum, got "
        f"{tuple(weight.shape)} -- is_bmm post-load did not reshape"
    )
    assert weight_scale.ndim >= 3 and weight_scale.dtype not in (
        torch.float8_e8m0fnu,
        torch.uint8,
    ), (
        f"wo_a scale looks like the RAW ckpt scale {tuple(weight_scale.shape)} "
        f"{weight_scale.dtype}: the DeepGEMM SF post-load "
        "(deepgemm_post_process_fp8_weight_block is_bmm=True) never ran, so "
        "fp8_einsum would read scales from the wrong layout."
    )
    if tma_aligned:
        assert o_scale.dtype == torch.int32, (
            f"activation scales must be packed-int32 UE8M0 on SM100+, got "
            f"{o_scale.dtype}"
        )


def _sm8x_emulate_weight(weight: torch.Tensor) -> bool:
    """Whether an FP8-stacked weight must be emulated on its owning card.

    Mixed-arch TP hazard: ``weight.device.index`` (or the ambient device) can
    report a native-FP8 sibling while this very process compiles for an
    SM8x card. ``sm8x_sparse_mla_enabled()`` is per-process and was already
    trusted to route this model's attention through the Triton path, so treat
    it as a hard override.
    """
    if weight.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    native = fp8_native_supported(weight.device.index)
    if not native:
        return True
    if sm8x_sparse_mla_enabled():
        logger.info_once(
            "SM8x: per-device FP8 probe (%s) disagrees with this process's "
            "SM8x routing (dev=%s cur=%s); forcing emulation.",
            tuple(weight.shape),
            weight.device.index,
            torch.cuda.current_device() if torch.cuda.is_available() else -1,
        )
        return True
    return False


def _wo_a_bf16_from_fp8(layer: torch.nn.Module) -> torch.Tensor | None:
    """Dequantize an FP8 ``wo_a`` layer's weight to bf16, or None if the
    weight is not in a plain FP8 (or uint8-carried FP8) layout."""
    weight_fp8 = layer.weight
    if weight_fp8.dtype == torch.uint8:
        # Some ue8m0 checkpoints carry e4m3 bytes in uint8 storage.
        weight_fp8 = weight_fp8.view(torch.float8_e4m3fn)
    elif weight_fp8.dtype != torch.float8_e4m3fn:
        return None
    weight = weight_fp8.data
    scale = getattr(layer, "weight_scale_inv", None)
    if scale is None:
        scale = getattr(layer, "weight_scale", None)
    if scale is None:
        dequant = weight.float()
    else:
        if scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
            # The param stores raw ue8m0 exponent bytes as uint8 (mirrors
            # fp8_utils' ``ws.dtype in (e8m0fnu, uint8)`` gate): bitcast to
            # e8m0 and upcast to powers of two, never multiply the byte.
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                _upcast_e8m0_to_fp32,
            )

            scale = _upcast_e8m0_to_fp32(scale)
        block = getattr(layer, "weight_block_size", None)
        if block is None or scale.numel() == 1:
            dequant = weight.float() * scale.float().view(-1, 1)
        else:
            block_m, block_k = block
            scale = scale.float()
            scale = scale.repeat_interleave(block_m, dim=0)
            scale = scale.repeat_interleave(block_k, dim=1)
            dequant = weight.float() * scale[: weight.shape[0], : weight.shape[1]]
    return dequant.to(torch.bfloat16).contiguous()


def _sm8x_dequant_wo_a_layer(layer: torch.nn.Module) -> torch.Tensor:
    dequant = _wo_a_bf16_from_fp8(layer)
    assert dequant is not None, "SM8x wo_a dequant expects an FP8 plain layout"
    replace_parameter(layer, "weight", dequant)
    for attr in ("weight_scale", "weight_scale_inv"):
        if getattr(layer, attr, None) is not None:
            delattr(layer, attr)
    return dequant


def sm8x_dequant_wo_a_at_load(wo_a: nn.Module) -> None:
    """SM8x wo_a exemption from the block-FP8 kernel's load-time repack.

    ``wo_a`` (``is_bmm``) is consumed as a raw grouped bf16 stack by
    ``deep_gemm_fp8_o_proj``, not through ``apply``. On SM8x the selected
    block-FP8 kernel would repack for ``scaled_mm`` (Marlin), which the bmm
    reader cannot use and the fp8 einsum cannot run, so dequantize once at
    load and skip the kernel's post-load entirely (cpu_sparse packs the
    same way for its own backend). SM90+ keeps its path untouched.
    """
    quant_method = wo_a.quant_method
    orig_pwal = quant_method.process_weights_after_loading

    logger.info_once(
        "SM8x: wo_a load-time dequant patch installed on %s.",
        wo_a.prefix,
    )

    def _dequant_and_skip(layer: torch.nn.Module) -> None:
        if _wo_a_bf16_from_fp8(layer) is None:
            logger.info_once(
                "SM8x: wo_a weight was not plain FP8 at post-load for %s; "
                "deferring to the kernel's own repack.",
                layer.prefix,
            )
            orig_pwal(layer)
            return
        logger.info_once(
            "SM8x: dequantizing wo_a block-FP8 weights to bf16 at load "
            "(grouped bmm path; no Marlin repack, no fp8 einsum)."
        )
        _sm8x_dequant_wo_a_layer(layer)

    setattr(quant_method, "process_weights_after_loading", _dequant_and_skip)


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization.
    """
    weight = wo_a.weight
    from vllm.models.deepseek_v4_1.common.ops.stat_probe import stat_probe_once

    stat_probe_once(
        f"attn_o[{getattr(wo_a, 'prefix', id(wo_a))}]", o
    )
    if _sm8x_emulate_weight(weight):
        # Safety net for the load-time hook: if FP8 wo_a survived post-load
        # (hook missed), dequantize lazily on the first eager run so later
        # trace/capture sees a stable bf16 tensor.
        logger.info_once(
            "SM8x: wo_a still FP8 at forward (shape=%s, quant_method=%s); "
            "dequantizing to bf16 lazily.",
            tuple(weight.shape),
            type(getattr(wo_a, "quant_method", None)).__name__,
        )
        weight = _sm8x_dequant_wo_a_layer(wo_a)
    use_fp8 = weight.dtype == torch.float8_e4m3fn
    # Two independent reasons to stay on bf16: the card hosting wo_a cannot
    # name fp8e4nv (hardware fact), or the target this launch would actually
    # compile for cannot (binding/override hazards on heterogeneous boxes).
    if use_fp8 and not fp8_quant_target_supported(weight.device.index):
        # Hard gate on the target the QUANTIZE launch actually compiles with
        # (hardware capability + ambient target + this kernel's bound entry),
        # so a fp8e4nv store can never reach ptxas on a card that cannot
        # express it.
        if torch.cuda.is_available():
            logger.info_once(
                "SM8x: final hard-gate dequant (weight dev=%s cur=%s/%s "
                "target=%s bound=%s native(cur)=%s native(idx=%s)=%s sm8sig=%s "
                "env=%s/%s).",
                weight.device,
                torch.cuda.current_device(),
                torch.cuda.get_device_name(),
                fp8_triton_target_arch(weight.device.index),
                bound_compile_arch(
                    _FUSED_INV_ROPE_FP8_QUANT_KERNEL.kernel, weight.device.index
                ),
                fp8_native_supported(),
                weight.device.index,
                fp8_native_supported(weight.device.index),
                sm8x_sparse_mla_enabled(),
                os.environ.get("CUDA_DEVICE_ORDER"),
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        weight = _sm8x_dequant_wo_a_layer(wo_a)
        use_fp8 = False
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=einsum_recipe[2],
        tma_aligned_scales=tma_aligned_scales,
        quantize=use_fp8,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        _check_wo_a_einsum_inputs(
            wo_a, weight, weight_scale, o_scale, tma_aligned_scales
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    else:
        grouped_weight = weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    stat_probe_once(f"wo_a_z[{getattr(wo_a, 'prefix', id(wo_a))}]", z)
    return wo_b(z.flatten(1))
