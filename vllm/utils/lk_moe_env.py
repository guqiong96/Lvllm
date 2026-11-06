# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized environment registry for the lk_moe integration.

This is the single source of truth for every ``LK_*`` switch in vLLM.  It
mirrors the ``sglang.srt.environ`` / ``sglang.srt.utils.common`` lk_moe
convention:

* every knob is declared once as an ``Env*`` descriptor (``.get()``,
  ``.is_set()``, ``.set()``, ``.override()``);
* historical ``LVLLM_*`` spellings are migrated to their ``LK_*`` names at
  import time, so existing launch scripts keep working;
* call sites consume a single snapshot from :func:`get_lk_moe_config`, or
  the small predicate helpers below -- never ``os.getenv`` directly.

The module is intentionally stdlib-only so it can be imported very early.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import warnings
from contextlib import contextmanager
from typing import Any


logger = logging.getLogger(__name__)


class EnvField:
    """Base descriptor for one environment variable."""

    def __init__(self, default: Any):
        self.default = default

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError

    def _resolve_default(self) -> Any:
        return self.default() if callable(self.default) else self.default

    def get(self) -> Any:
        value = os.getenv(self.name)
        if value is None:
            return self._resolve_default()
        try:
            return self.parse(value)
        except ValueError as exc:
            default = self._resolve_default()
            warnings.warn(
                f'Invalid value for {self.name}: {exc}, using default "{default}"'
            )
            return default

    def is_set(self) -> bool:
        return self.name in os.environ

    def set(self, value: Any) -> None:
        os.environ[self.name] = str(value)

    def override(self, value: Any):
        @contextmanager
        def _ctx():
            backup = os.environ.get(self.name)
            self.set(value)
            try:
                yield
            finally:
                if backup is None:
                    os.environ.pop(self.name, None)
                else:
                    os.environ[self.name] = backup

        return _ctx()

    def clear(self) -> None:
        os.environ.pop(self.name, None)

    def __bool__(self) -> bool:
        raise RuntimeError(
            f"Please use `envs.{self.name}.get()` instead of `envs.{self.name}`"
        )


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ("true", "1", "yes", "y"):
            return True
        if value in ("false", "0", "no", "n"):
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f'"{value}" is not a valid integer value') from exc


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f'"{value}" is not a valid float value') from exc


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


# Historical LVLLM_* names are auto-renamed to the unified LK_* names once at
# import, so existing scripts keep working (both set -> LK_* wins).
_LK_ENV_RENAMES = {
    "LVLLM_MOE_NUMA_ENABLED": "LK_MOE_HYBRID_ENABLED",
    "LVLLM_ENABLE_NUMA_INTERLEAVE": "LK_NUMA_INTERLEAVE",
    "LVLLM_GPU_PREFILL_MIN_BATCH_SIZE": "LK_GPU_PREFILL_MIN_TOKENS",
    "LVLLM_GPU_PREFETCH_WINDOW": "LK_GPU_PREFETCH_WINDOW",
    "LVLLM_GPU_RESIDENT_MOE_LAYERS": "LK_GPU_RESIDENT_MOE_LAYERS",
    "LVLLM_GPU_RESIDENT_MOE_LAYERS_SPEC": "LK_GPU_RESIDENT_MOE_LAYERS_SPEC",
    "LVLLM_GPU_RESIDENT_MOE_LAYERS_MTP": "LK_GPU_RESIDENT_MOE_LAYERS_MTP",
    "LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK": "LK_GPU_RESIDENT_MOE_LAYERS_DSPARK",
    "LVLLM_ENABLE_MOE_LAYERWISE_LOAD": "LK_MOE_LAYERWISE_LOAD",
    "LVLLM_GPU_RESERVE_GB": "LK_GPU_RESERVE_GB",
}


def migrate_lk_env_vars() -> None:
    for legacy, current in _LK_ENV_RENAMES.items():
        value = os.environ.pop(legacy, None)
        if value is None:
            continue
        if current not in os.environ:
            os.environ[current] = value
        logger.warning(
            "lk_moe: env %s was renamed to %s%s",
            legacy,
            current,
            " (renamed value dropped, both were set)"
            if current in os.environ and os.environ[current] != value
            else "",
        )


class LKMoEEnvs:
    """Every lk_moe knob, declared once (env var name == attribute name)."""

    # Master switch and CPU/NUMA tier.
    LK_MOE_HYBRID_ENABLED = EnvBool(False)
    LK_NUMA_INTERLEAVE = EnvBool(True)
    LK_MOE_LAYERWISE_LOAD = EnvBool(False)
    LK_THREAD_BINDING = EnvStr("NUMA_NODE")
    LK_POWER_SAVING = EnvBool(False)

    # Layer placement.
    LK_GPU_RESIDENT_MOE_LAYERS = EnvStr(None)
    LK_GPU_RESIDENT_MOE_LAYERS_SPEC = EnvStr(None)
    LK_GPU_RESIDENT_MOE_LAYERS_MTP = EnvStr(None)
    LK_GPU_RESIDENT_MOE_LAYERS_DSPARK = EnvStr(None)
    LK_POOL_LAYERS = EnvStr(None)

    # GPU prefill tier.
    LK_GPU_PREFILL_MIN_TOKENS = EnvInt(0)
    LK_GPU_PREFILL_ENGINE = EnvStr("cutlass")
    LK_GPU_PREFILL_SUB_M = EnvInt(0)
    LK_GPU_PREFETCH_WINDOW = EnvInt(1)
    LK_PP_DEBUG = EnvBool(False)

    # Expert pool (CPU/GPU staging membership).
    LK_POOL_SLOTS = EnvInt(0)
    LK_POOL_WINDOW = EnvInt(32)
    LK_POOL_INFLIGHT = EnvInt(16)
    LK_POOL_DECODE = EnvStr(None)
    LK_POOL_LOG = EnvInt(0)
    LK_ROUTE_PROBE = EnvBool(False)

    # Draft verify / mailbox / misc.
    LK_VERIFY_RESIDENT = EnvInt(0)
    LK_VERIFY_SKIP_MISS = EnvInt(0)
    LK_HF_MBOX = EnvInt(1)
    LK_GPU_RESERVE_GB = EnvFloat(0.0)


envs = LKMoEEnvs()

migrate_lk_env_vars()

_gpu_prefill_min_tokens_override: int | None = None


@dataclasses.dataclass(frozen=True)
class LKMoEConfig:
    """Immutable snapshot of every lk_moe knob at one point in time."""

    enabled: bool
    numa_interleave: bool
    layerwise_load: bool
    thread_binding: str
    power_saving: bool
    gpu_resident_layers: str | None
    gpu_resident_layers_spec: str | None
    gpu_resident_layers_mtp: str | None
    gpu_resident_layers_dspark: str | None
    pool_layers: str | None
    prefill_min_tokens: int
    prefill_engine: str
    prefill_sub_m: int
    prefetch_window: int
    prefill_debug: bool
    pool_slots: int
    pool_window: int
    pool_inflight: int
    pool_decode: bool | None
    pool_log: int
    route_probe: bool
    verify_resident: int
    verify_skip_miss: int
    hf_mbox: int
    gpu_reserve_gb: float

    @property
    def gpu_prefill_enabled(self) -> bool:
        return self.enabled and self.prefill_min_tokens > 0

    @property
    def pool_enabled(self) -> bool:
        return self.enabled and self.pool_slots > 0

    @property
    def super_chunk_size(self) -> int:
        return self.prefill_sub_m if self.gpu_prefill_enabled else 0

    @property
    def prefill_engine_choice(self) -> str:
        return "cutlass" if self.prefill_engine.strip().lower() == "cutlass" else "triton"


def _pool_decode_default() -> bool | None:
    value = envs.LK_POOL_DECODE.get()
    if value is None:
        return None
    return EnvBool(False).parse(value)


def get_lk_moe_config() -> LKMoEConfig:
    """Return the single centralized snapshot consumed by vLLM code.

    The registry itself stays live (``envs.*.get()`` reads the process
    environment); callers should take one snapshot per layer construction so
    a layer never observes a half-updated environment.
    """
    return LKMoEConfig(
        enabled=envs.LK_MOE_HYBRID_ENABLED.get(),
        numa_interleave=envs.LK_NUMA_INTERLEAVE.get(),
        layerwise_load=envs.LK_MOE_LAYERWISE_LOAD.get(),
        thread_binding=envs.LK_THREAD_BINDING.get(),
        power_saving=envs.LK_POWER_SAVING.get(),
        gpu_resident_layers=envs.LK_GPU_RESIDENT_MOE_LAYERS.get(),
        gpu_resident_layers_spec=envs.LK_GPU_RESIDENT_MOE_LAYERS_SPEC.get(),
        gpu_resident_layers_mtp=envs.LK_GPU_RESIDENT_MOE_LAYERS_MTP.get(),
        gpu_resident_layers_dspark=envs.LK_GPU_RESIDENT_MOE_LAYERS_DSPARK.get(),
        pool_layers=envs.LK_POOL_LAYERS.get(),
        prefill_min_tokens=get_gpu_prefill_min_tokens(),
        prefill_engine=envs.LK_GPU_PREFILL_ENGINE.get(),
        prefill_sub_m=envs.LK_GPU_PREFILL_SUB_M.get(),
        prefetch_window=envs.LK_GPU_PREFETCH_WINDOW.get(),
        prefill_debug=envs.LK_PP_DEBUG.get(),
        pool_slots=envs.LK_POOL_SLOTS.get(),
        pool_window=max(8, envs.LK_POOL_WINDOW.get()),
        pool_inflight=envs.LK_POOL_INFLIGHT.get(),
        pool_decode=_pool_decode_default(),
        pool_log=envs.LK_POOL_LOG.get(),
        route_probe=envs.LK_ROUTE_PROBE.get(),
        verify_resident=envs.LK_VERIFY_RESIDENT.get(),
        verify_skip_miss=envs.LK_VERIFY_SKIP_MISS.get(),
        hf_mbox=envs.LK_HF_MBOX.get(),
        gpu_reserve_gb=envs.LK_GPU_RESERVE_GB.get(),
    )


@contextmanager
def lk_pool_pump_parked():
    """Park lk_moe's expert-pool pump around CUDA-graph launches.

    The pump issues cudaMemcpyAsync from its own thread; overlapping that with
    cuGraphLaunch deadlocks inside libcuda on P2P-patched drivers (launch
    thread waits on a driver condvar while the copy threads wait on libcuda's
    rwlock). Capture must not race with those API calls either.
    """
    if not is_lk_pool_enabled():
        yield
        return
    import lk_moe

    # Pool is on, so these hooks are part of the lk_moe contract: a build that
    # lacks them is incompatible and must fail loudly here instead of silently
    # skipping the park (which would bring the cuGraphLaunch/copy deadlock
    # back).
    pause = lk_moe.pool_capture_pause
    resume = lk_moe.pool_capture_resume
    pause()
    try:
        yield
    finally:
        resume()


def is_lk_moe_feature_enabled() -> bool:
    return get_lk_moe_config().enabled


def is_numa_interleave_enabled() -> bool:
    return get_lk_moe_config().numa_interleave


def get_gpu_prefill_min_tokens() -> int:
    if _gpu_prefill_min_tokens_override is not None:
        return _gpu_prefill_min_tokens_override
    return envs.LK_GPU_PREFILL_MIN_TOKENS.get()


def get_gpu_prefill_min_batch_size() -> int:
    return get_gpu_prefill_min_tokens()


def get_gpu_prefill_engine() -> str:
    return get_lk_moe_config().prefill_engine_choice


def get_gpu_prefetch_window() -> int:
    return envs.LK_GPU_PREFETCH_WINDOW.get()


def get_super_chunk_size() -> int:
    return get_lk_moe_config().super_chunk_size


def get_pool_slots() -> int:
    return envs.LK_POOL_SLOTS.get()


def get_pool_window() -> int:
    return max(8, envs.LK_POOL_WINDOW.get())


def get_pool_inflight() -> int:
    return envs.LK_POOL_INFLIGHT.get()


def get_verify_resident() -> int:
    return envs.LK_VERIFY_RESIDENT.get()


def get_verify_skip_miss() -> int:
    return envs.LK_VERIFY_SKIP_MISS.get()


def get_hf_mbox() -> int:
    return envs.LK_HF_MBOX.get()


def get_pool_log() -> int:
    return envs.LK_POOL_LOG.get()


def get_gpu_reserve_gb() -> float:
    return envs.LK_GPU_RESERVE_GB.get()


def is_lk_gpu_prefill_enabled() -> bool:
    return get_lk_moe_config().gpu_prefill_enabled


def is_lk_moe_use_gpu_prefill() -> bool:
    return is_lk_gpu_prefill_enabled()


def is_lk_pool_enabled() -> bool:
    return get_lk_moe_config().pool_enabled


def is_lk_pool_decode_enabled() -> bool:
    config = get_lk_moe_config()
    if not config.enabled:
        return False
    if config.pool_decode is not None:
        return config.pool_decode
    return config.pool_enabled


def lk_route_probe_enabled() -> bool:
    config = get_lk_moe_config()
    return config.pool_enabled and config.route_probe


def is_lk_prefill_debug() -> bool:
    return envs.LK_PP_DEBUG.get()


def enabled_layerwise_load() -> bool:
    return envs.LK_MOE_LAYERWISE_LOAD.get()


def disable_lk_moe_gpu_prefill() -> int:
    """Temporarily force the GPU-prefill tier off; returns the old value."""
    global _gpu_prefill_min_tokens_override
    origin = get_gpu_prefill_min_tokens()
    _gpu_prefill_min_tokens_override = 0
    return origin


def enable_lk_moe_gpu_prefill(value: int) -> int:
    global _gpu_prefill_min_tokens_override
    _gpu_prefill_min_tokens_override = value
    return value


def reset_lk_moe_gpu_prefill_override() -> None:
    global _gpu_prefill_min_tokens_override
    _gpu_prefill_min_tokens_override = None


def get_model_type_from_layer_name(layer_name: str) -> str:
    if not layer_name:
        return "main"
    if layer_name.startswith("stages."):
        return "dspark"
    if layer_name.startswith("mtp.") or ".mtp." in layer_name:
        return "mtp"
    return "main"


def is_lk_moe_mtp_layer(layer_name: str) -> bool:
    return get_model_type_from_layer_name(layer_name) == "mtp"


def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """Extract the layer index from a module name (vLLM-compatible)."""
    int_vals = [int(name) for name in layer_name.split(".") if name.isdecimal()]
    if not int_vals:
        raise ValueError(f"Cannot extract layer index from {layer_name!r}")
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )
        return int_vals[0]
    assert len(int_vals) <= 2, (
        f"layer name {layer_name} should contain at most two integers"
    )
    return int_vals[0] * num_attn_module + int_vals[1] if len(int_vals) == 2 else int_vals[0]


def get_gpu_resident_env_var(model_type: str = "main") -> str | None:
    if model_type in ("dspark", "mtp"):
        spec = envs.LK_GPU_RESIDENT_MOE_LAYERS_SPEC.get()
        if spec is not None:
            return spec
        role = (
            envs.LK_GPU_RESIDENT_MOE_LAYERS_MTP.get()
            if model_type == "mtp"
            else envs.LK_GPU_RESIDENT_MOE_LAYERS_DSPARK.get()
        )
        if role is not None:
            return role
    return envs.LK_GPU_RESIDENT_MOE_LAYERS.get()


def parse_layer_set(layer_set: str | None) -> set[int] | None:
    if layer_set is None:
        return None
    layers: set[int] = set()
    for part in layer_set.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                continue
            if start <= end:
                layers.update(range(start, end + 1))
        else:
            try:
                layers.add(int(part))
            except ValueError:
                continue
    return layers


def is_lk_moe_gpu_resident_layer(
    layer_name: str | int, model_type: str | None = None
) -> bool:
    config = get_lk_moe_config()
    if not config.enabled:
        return True
    if isinstance(layer_name, str):
        if model_type is None:
            model_type = get_model_type_from_layer_name(layer_name)
        if model_type != "main":
            return True
        layer_id = extract_layer_index(layer_name)
    else:
        layer_id = layer_name
        if model_type not in (None, "main"):
            return True

    resident = parse_layer_set(get_gpu_resident_env_var(model_type or "main"))
    return bool(resident and layer_id in resident)


def is_lk_moe_gpu_prefill_layer(layer_name: str) -> bool:
    return (
        is_lk_gpu_prefill_enabled()
        and not is_lk_moe_gpu_resident_layer(layer_name)
        and not is_lk_moe_mtp_layer(layer_name)
    )


def is_lk_moe_cpu_layer(layer_name: str) -> bool:
    if not is_lk_moe_feature_enabled():
        return False
    return (
        not is_lk_moe_gpu_resident_layer(layer_name)
        and not is_lk_moe_gpu_prefill_layer(layer_name)
        and not is_lk_moe_mtp_layer(layer_name)
    )


def is_lk_pool_layer(layer_name: str, model_type: str | None = None) -> bool:
    if model_type is None:
        model_type = get_model_type_from_layer_name(layer_name)
    if model_type != "main":
        return False
    if is_lk_moe_gpu_resident_layer(layer_name, model_type):
        return False
    return is_lk_pool_enabled()


def fill_lk_moe_config(cfg: Any, layer_name: str | None = None) -> Any:
    """Apply the centralized env snapshot to an ``lk_moe.MOEConfigV2``.

    This is the one integration point between vLLM and the lk_moe env surface.
    Membership mirrors SGLang's ``_fill_lk_cfg``: ``pool`` is the infra tier
    (cache OR prefill staging), ``pool_cache`` marks real pool slots (main
    layers only), and ``prefill_gpu`` gates the GPU-prefill tier.
    """
    config = get_lk_moe_config()
    if layer_name is not None:
        model_type = get_model_type_from_layer_name(layer_name)
        is_resident = is_lk_moe_gpu_resident_layer(layer_name, model_type)
        is_prefill = is_lk_moe_gpu_prefill_layer(layer_name)
        is_pool = is_lk_pool_layer(layer_name, model_type)
        try:
            cfg.layer_id = extract_layer_index(layer_name)
        except ValueError:
            cfg.layer_id = 0
        cfg.pool = bool(is_pool or is_prefill)
        cfg.pool_cache = bool(is_pool)
        cfg.prefill_gpu = bool(is_prefill)
        cfg.use_gpu_prefill = bool(
            is_prefill and not is_resident and get_gpu_prefill_engine() == "cutlass"
        )
        cfg.role_draft = model_type != "main"
        cfg.pool_decode = bool(is_lk_pool_decode_enabled() and is_pool)
    cfg.pool_slots = config.pool_slots
    cfg.pool_window = config.pool_window
    cfg.pool_inflight = config.pool_inflight
    cfg.pool_log = config.pool_log
    cfg.verify_resident = config.verify_resident
    cfg.verify_skip_miss = config.verify_skip_miss
    cfg.hf_mbox = config.hf_mbox
    cfg.prefill_min_tokens = config.prefill_min_tokens
    cfg.prefill_window = config.prefetch_window
    cfg.super_batch_size = config.super_chunk_size
    cfg.prefill_dbg = config.prefill_debug
    cfg.route_probe = lk_route_probe_enabled()
    return cfg
