# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    HAS_OPAQUE_TYPE,
    ModuleName,
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.envs import is_lk_moe_gpu_resident_layer, get_gpu_prefetch_window, get_gpu_prefill_min_batch_size, is_lk_moe_use_gpu_prefill, is_lk_moe_quant_on_gpu
def get_layer_from_name(layer_name: str) -> torch.nn.Module:
    forward_context: ForwardContext = get_forward_context()
    if layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    return forward_context.no_compile_layers[layer_name]


# On torch >= 2.11, layer_name is a hoisted ModuleName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | ModuleName
else:
    _layer_name_type = ModuleName if HAS_OPAQUE_TYPE else str


def _resolve_layer_name(layer_name: str | ModuleName) -> str:
    return layer_name.value if isinstance(layer_name, ModuleName) else layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's 'forward_dispatch' method.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    layer_name = layer.layer_name
    if layer.should_use_gpu_prefill(hidden_states):
        moe_prefetch(layer, layer_name, hidden_states, forward_context, get_gpu_prefetch_window())
        moe_wait_prefetch(layer, hidden_states, forward_context)
        # TODO(bnell): this can be removed after MK migration is complete.
        layer.ensure_moe_quant_config_init()
        fused_output = layer.runner.forward_dispatch(
            layer, hidden_states, router_logits, shared_experts_input
        )
        moe_cleanup(layer, layer_name, hidden_states, forward_context)
    else: 
        layer.ensure_moe_quant_config_init()
        fused_output = layer.runner.forward_dispatch(
            layer, hidden_states, router_logits, shared_experts_input
        )
    return fused_output


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    layer = get_layer_from_name(_resolve_layer_name(layer_name)) 
    layer_name = layer.layer_name
    if layer.should_use_gpu_prefill(hidden_states): 
        moe_prefetch(layer, layer_name, hidden_states, forward_context, get_gpu_prefetch_window())
        moe_wait_prefetch(layer, hidden_states, forward_context)
        # TODO(bnell): this can be removed after MK migration is complete.
        layer.ensure_moe_quant_config_init()
        shared_out, fused_out = layer.runner.forward_dispatch(
            layer, hidden_states, router_logits, shared_experts_input
        )
        moe_cleanup(layer, layer_name, hidden_states, forward_context)
    else:
        layer.ensure_moe_quant_config_init()
        shared_out, fused_out = layer.runner.forward_dispatch(
            layer, hidden_states, router_logits, shared_experts_input
        )
    return shared_out, fused_out


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Output shapes:
    # - fused_out: same as hidden_states (routed experts use transformed size)
    # - shared_out: same as shared_experts_input if provided, else same as
    #               hidden_states
    # (For latent MoE: shared experts use original hidden_size, not latent size)
    fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],  # is this still true?
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class DefaultMoERunner(MoERunner):
    """
    Default implementation of the MoE runner for executing Mixture of Experts layers.

    This class provides a comprehensive implementation for running MoE computations
    with support for:
    - Expert routing and token dispatching
    - Shared experts computation with optional parallel execution using CUDA streams
    - Data parallel (DP) chunking for large batch processing
    - Tensor model parallel and expert parallel operations
    - Various quantization methods and custom operators
    - Both monolithic and decomposed expert execution paths

    The runner handles the complete MoE forward pass including routing tokens to
    experts, executing expert computations, and combining results. It supports
    advanced features like overlapped execution of shared experts and optimized
    kernels for different parallel execution modes.

    Eventually, this class will be split up and specialized for different
    configurations, e.g. the presence or absence of shared experts, a gate, etc.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_input_transform: torch.nn.Module | None,
        gate: torch.nn.Module | None,
        shared_experts: torch.nn.Module | None,
        quant_method: FusedMoEMethodBase,
        reduce_results: bool,
        enable_dbo: bool,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.gate = gate
        self.quant_method = quant_method
        self.reduce_results = reduce_results
        self.enable_dbo = enable_dbo

        self.shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            self.shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                # Note: For now we must pass quant_method along to SharedExperts so it
                # can property determine where the shared experts are supposed to be
                # called, i.e. by a MK or by the MoERunner.
                # Once the MK can be created upfront, we can just pass in the proper
                # flags derived from the quant_method's MK.
                reduce_results=reduce_results,
                quant_method=quant_method,
                enable_dbo=enable_dbo,
            )

        # Chunked all2all staging tensor
        # These need to exist ahead of time due to CUDAgraph construction
        # needing a fixed buffer address.
        self.use_dp_chunking = self.moe_config.moe_parallel_config.use_dp_chunking
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None
        self._maybe_init_dp_chunking()

        # Needed for string -> FusedMoE layer lookup in custom ops.
        self.layer_name = layer.layer_name

        self.forward_entry, self.forward_impl = self._select_forward(layer)

    def _select_forward(self, layer: torch.nn.Module) -> tuple[Callable, Callable]:
        # Select implementation based on presence of DP chunking.
        forward_impl_fn = (
            self._forward_impl_chunked if self.use_dp_chunking else self._forward_impl
        )

        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped forward_impl.
            return (
                _moe_forward if self.shared_experts is None else _moe_forward_shared,
                forward_impl_fn,
            )

        return (
            torch.ops.vllm.moe_forward
            if self.shared_experts is None
            else torch.ops.vllm.moe_forward_shared,
            forward_impl_fn,
        )

    # TODO(bnell): temporary hack, do not call this method.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        if self.shared_experts is not None:
            self.shared_experts._quant_method = quant_method
        self.quant_method = quant_method

    def is_internal_router(self) -> bool:
        return self.gate is not None

    def _maybe_init_dp_chunking(self):
        if not self.use_dp_chunking:
            return

        assert self.batched_hidden_states is None
        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (2, moe.max_num_tokens, self.moe_config.num_logical_experts)
        else:
            states_shape = (moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (moe.max_num_tokens, self.moe_config.num_logical_experts)

        device = torch.accelerator.current_device_index()
        self.batched_hidden_states = torch.zeros(
            states_shape,
            dtype=moe.in_dtype,
            device=device,
        )

        self.batched_router_logits = torch.zeros(
            logits_shape,
            dtype=moe.router_logits_dtype,
            device=device,
        )

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        return (
            self.quant_method.moe_kernel is not None
            and self.quant_method.moe_kernel.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def apply_routed_input_transform(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by FusedMoE.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        TODO: For latent MoE bandwidth optimization, fc2_latent_proj could be
        moved inside SharedFusedMoE to all-reduce on the smaller latent
        dimension.

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self.shared_experts is not None else None,
        )

    def _maybe_reduce_output(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        trunc_sizes: list[int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        def trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            return x[..., :trunc_size]

        def reduce_and_trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            return trunc(self.maybe_all_reduce_tensor_model_parallel(x), trunc_size)

        if (
            not self.moe_config.is_sequence_parallel
            and not self.use_dp_chunking
            and self.reduce_results
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
        ):
            func = reduce_and_trunc
        else:
            func = trunc

        if isinstance(states, tuple):
            return tuple(
                [func(s, trunc_size) for s, trunc_size in zip(states, trunc_sizes)]
            )
        else:
            assert len(trunc_sizes) == 1
            return func(states, trunc_sizes[0])

    def _encode_layer_name(self) -> str | ModuleName:
        if HAS_OPAQUE_TYPE:
            return ModuleName(self.layer_name)
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim = hidden_states.shape[-1]
        if (
            not self.quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        if self.shared_experts is not None:
            orig_hidden_dims = [shared_experts_hidden_dim, transformed_hidden_dim]
        else:
            orig_hidden_dims = [transformed_hidden_dim]

        return hidden_states, orig_hidden_dims

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        if self.shared_experts is not None:
            assert shared_experts_input is not None
            self.shared_experts.apply(shared_experts_input, order)

    def _apply_quant_method(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # Run this before quant_method to avoid inplace issues.
        # TODO(bnell): probably not needed anymore since inplace is
        # disabled when shared experts are present.
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        local_topk_ids = layer.global_to_local_expert_ids(topk_ids) if layer.use_ep else topk_ids
        if self.quant_method.is_monolithic:
            if not layer.is_gpu_resident_layer and not layer.should_use_gpu_prefill(hidden_states):
                fused_out = layer.forward_lk( 
                    hidden_states,
                    topk_weights, 
                    local_topk_ids
                )
            else:
                from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod 
                if layer.is_gpu_prefill_layer and not isinstance(self.quant_method, UnquantizedFusedMoEMethod):
                    fused_out = torch.ops.vllm._fused_moe_gguf(
                        hidden_states,
                        layer.w13_weight.data,
                        layer.w2_weight.data,
                        topk_weights,
                        local_topk_ids,
                        2,
                        2,
                        layer.activation.value,
                    )
                else:
                    fused_out = self.quant_method.apply_monolithic(
                        layer=layer,
                        x=hidden_states,
                        router_logits=router_logits,
                    )
        else:
            if not layer.is_gpu_resident_layer and not layer.should_use_gpu_prefill(hidden_states):
                fused_out = layer.forward_lk(
                    hidden_states,
                    topk_weights, 
                    local_topk_ids
                )
            else:
                from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod 
                if layer.is_gpu_prefill_layer and not isinstance(self.quant_method, UnquantizedFusedMoEMethod):
                    fused_out = torch.ops.vllm._fused_moe_gguf(
                        hidden_states,
                        layer.w13_weight.data,
                        layer.w2_weight.data,
                        topk_weights,
                        local_topk_ids,
                        2,
                        2,
                        layer.activation.value,
                    )
                else:
            
                    # Passing shared_experts_input in case SharedExpertsOrder is
                    # NO_OVERLAP or MK_INTERNAL_OVERLAPPED.
                    fused_out = self.quant_method.apply(
                        layer=layer,
                        x=hidden_states,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        shared_experts_input=shared_experts_input,
                    )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        return (
            self.shared_experts.output if self.shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _allocate_dp_chunking_outputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        assert self.use_dp_chunking

        # Assert the inputs are of the proper type and shape.
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None

        assert self.batched_hidden_states.dtype == hidden_states.dtype, (
            f"{self.batched_hidden_states.dtype} == {hidden_states.dtype}"
        )
        assert self.batched_router_logits.dtype == router_logits.dtype, (
            f"{self.batched_router_logits.dtype} == {router_logits.dtype}"
        )

        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == router_logits.size(-1)

        final_fused_hidden_states = torch.empty_like(hidden_states)
        if self.shared_experts is not None:
            final_shared_hidden_states = torch.empty_like(hidden_states)
        else:
            final_shared_hidden_states = None

        return final_shared_hidden_states, final_fused_hidden_states

    def _maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor | None,
    ):
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.shared_experts is not None:
            self.shared_experts.maybe_sync_shared_experts_stream(shared_experts_input)

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
        )

    def _maybe_dispatch(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if self.do_naive_dispatch_combine:
            hidden_states, router_logits = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )

        # NOTE: Similar with DP, PCP also needs dispatch and combine. For
        # simplicity, AgRsAll2All was added separately for PCP here. Maybe
        # we should modify All2AllManager abstraction to better support PCP.
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return hidden_states, router_logits

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if self.do_naive_dispatch_combine:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(
                hidden_states,
                dim=0,
            )

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.
        or
        - A tuple of (shared experts output, new hidden_states).

        Calling sequence
        - forward
          - self.forward_entry (_moe_forward or _moe_forward_shared custom op)
            - forward_dispatch
              - forward_impl (_forward_impl or _forward_impl_chunked)

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reasons:
        1. the chunking loop in _forward_impl_chunked cannot be compiled by
           torch.compile
        2. pytorch cannot handle union types in custom op signatures so _moe_forward
           and _moe_forward_shared must be split.

        If _forward_impl_chunked can be implemented via torch.scan we can potentially
        get rid of _moe_forward and _moe_forward_shared and collapse the whole sequence
        into the 'forward' method.
        """

        # Apply transform for routed experts (e.g., latent projection for latent MoE)
        hidden_states, shared_experts_input = self.apply_routed_input_transform(
            hidden_states
        )

        hidden_states, og_hidden_dims = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        fused_output = self.forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            self._encode_layer_name(),
        )

        return self._maybe_reduce_output(fused_output, og_hidden_dims)

    def forward_dispatch(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # TODO(bnell): this can be removed after MK migration is complete.
        layer.ensure_moe_quant_config_init()

        # Sync aux and main stream for shared expert multi-stream overlap.
        self._maybe_sync_shared_experts_stream(shared_experts_input)

        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.EXTERNAL,
        )

        with self._sequence_parallel_context():
            return self.forward_impl(
                layer,
                hidden_states,
                router_logits,
                shared_experts_input,
            )

    def _slice_and_copy_input(
        self,
        out_slice: torch.Tensor,
        orig: torch.Tensor | None,
        start: int,
        end: int,
    ) -> torch.Tensor:
        assert orig is not None
        slice_size = end - start
        orig_slice = orig[start:end, :]
        if self.enable_dbo:
            assert out_slice.dim() == 3
            batch_buffer_idx = dbo_current_ubatch_id()
            out_slice = out_slice[batch_buffer_idx, :]

        assert out_slice.size(0) >= slice_size
        out_slice = out_slice[:slice_size, :]
        out_slice.copy_(orig_slice, non_blocking=True)
        return out_slice

    def _forward_impl_chunked(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        final_shared_hidden_states, final_fused_hidden_states = (
            self._allocate_dp_chunking_outputs(hidden_states, router_logits)
        )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.moe_config.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.moe_config.sp_size
            )

        num_tokens = hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            chunk_sizes = ctx.dp_metadata.chunked_sizes(
                self.moe_config.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            )
            with chunk_sizes:
                hidden_states_chunk = self._slice_and_copy_input(
                    self.batched_hidden_states,
                    hidden_states,
                    chunk_start,
                    chunk_end,
                )

                router_logits_chunk = self._slice_and_copy_input(
                    self.batched_router_logits,
                    router_logits,
                    chunk_start,
                    chunk_end,
                )

                shared_experts_input_chunk = (
                    shared_experts_input[chunk_start:chunk_end, :]
                    if shared_experts_input is not None
                    else None
                )

                shared_output_chunk, hidden_states_chunk = self._apply_quant_method(
                    layer=layer,
                    hidden_states=hidden_states_chunk,
                    router_logits=router_logits_chunk,
                    shared_experts_input=shared_experts_input_chunk,
                )

                # Store outputs
                # TODO(bnell): document when chunk_start >= num_tokens
                if chunk_start < num_tokens:
                    final_fused_hidden_states[chunk_start:chunk_end, :].copy_(
                        hidden_states_chunk, non_blocking=True
                    )
                    if self.shared_experts is not None:
                        assert shared_output_chunk is not None
                        assert final_shared_hidden_states is not None
                        final_shared_hidden_states[chunk_start:chunk_end, :].copy_(
                            shared_output_chunk, non_blocking=True
                        )

        if self.shared_experts is None:
            return final_fused_hidden_states
        else:
            assert final_shared_hidden_states is not None
            return (final_shared_hidden_states, final_fused_hidden_states)

    def _forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # TODO(bnell): parts of the dispatch/combine steps will go away once
        # #32567 lands and the remaining kernels are made MKs.  The PCP
        # code will probably remain
        hidden_states, router_logits = self._maybe_dispatch(
            layer,
            hidden_states,
            router_logits,
        )

        shared_output, hidden_states = self._apply_quant_method(
            layer=layer,
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=shared_experts_input,
        )

        return self._maybe_combine(
            shared_output,
            hidden_states,
        )
        

from vllm.envs import extract_layer_index    

from typing import Dict, Optional, List
from vllm.envs import extract_layer_index  
 

def create_cpu_weights(layer, is_fp8: bool, is_wna16: bool, is_regular: bool) -> Dict[str, torch.Tensor]: 
    pin_memory = is_pin_memory_available()
    cpu_weights = {}
    
    if is_fp8 or is_wna16: 
        param_names = ["w13_weight", "w2_weight"]
        for param_name in param_names:
            if param_name == "w13_weight":
                E = layer.local_num_experts
                N = layer.intermediate_size_per_partition * 2 if layer.has_gate_proj else layer.intermediate_size_per_partition
                K = layer.hidden_size
                shape = (E, N, K * 18 // 32)
            elif param_name == "w2_weight":
                E = layer.local_num_experts
                N = layer.hidden_size
                K = layer.intermediate_size_per_partition
                shape = (E, N, K * 18 // 32)
            
            weight_cpu = torch.zeros(
                shape,
                dtype=torch.uint8,
                device="cpu",
                requires_grad=False,
                pin_memory=pin_memory
            ).contiguous()
            
        
            cpu_weights[param_name] = weight_cpu
            logger.debug(f"Created {param_name} with shape {shape} for FP8/WNA16 layer")
            
    elif is_regular: 
        w13_shape = (layer.local_num_experts, 
                    layer.intermediate_size_per_partition * 2 if layer.has_gate_proj else layer.intermediate_size_per_partition,    
                    layer.hidden_size)
        w13_buffer_size = w13_shape[0] * w13_shape[1] * w13_shape[2] * 2 

        w13_weight_cpu = torch.zeros(
            w13_buffer_size,
            dtype=torch.uint8,
            device="cpu",
            requires_grad=False,
            pin_memory=pin_memory,
        ).contiguous() 
        
        
        cpu_weights['w13_weight'] = w13_weight_cpu
        logger.debug(f"Created w13_weight with shape {w13_shape} for regular layer")
            
        w2_shape = (layer.local_num_experts,
                    layer.hidden_size,
                    layer.intermediate_size_per_partition)
        w2_buffer_size = w2_shape[0] * w2_shape[1] * w2_shape[2] * 2
        w2_weight_cpu = torch.zeros(
            w2_buffer_size,
            dtype=torch.uint8,
            device="cpu",
            requires_grad=False,
            pin_memory=pin_memory,
        ).contiguous() 
        
        cpu_weights['w2_weight'] = w2_weight_cpu
        logger.debug(f"Created w2_weight with shape {w2_shape} for regular layer")
    
    return cpu_weights
 
def moe_prepare_gpu_prefill(layer, forward_context: ForwardContext, device: torch.device):
    
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE   
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsWNA16MarlinMoEMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsWNA16MoEMethod  
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod  
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsW8A8Fp8MoEMethod
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    if layer.is_gpu_prefill_layer: 
        batch_key = id(forward_context.batch_descriptor)
        logger.debug(f"batch_key={batch_key}, forward_context={id(forward_context)}")
        batch_id = getattr(forward_context, '_prefetch_batch_id', None)
        is_temporary = False
         
        if batch_id is not None:
            stored_batch_key = getattr(forward_context, '_prefetch_batch_key', None)
            if stored_batch_key != batch_key:
                logger.warning(f"Batch key mismatch! stored={stored_batch_key}, current={batch_key}, "
                       f"resetting batch_id from {batch_id} to None")
                batch_id = None
        
        if batch_id is None:
            with FusedMoE._batch_lock: 
                batch_id = getattr(forward_context, '_prefetch_batch_id', None)
                if batch_id is not None:
                    stored_batch_key = getattr(forward_context, '_prefetch_batch_key', None)
                    if stored_batch_key != batch_key:
                        batch_id = None
                
                if batch_id is None:
                    for bid, in_use in FusedMoE._batch_usage.items():
                        if not in_use:
                            batch_id = bid
                            FusedMoE._batch_usage[bid] = True
                            forward_context._prefetch_batch_id = batch_id
                            forward_context._prefetch_batch_key = batch_key  
                            break
                    
                    if batch_id is None:
                        batch_id = -1  
                        is_temporary = True
                        forward_context._prefetch_batch_id = batch_id
                        forward_context._prefetch_batch_key = batch_key 
         
        with torch.no_grad():

            batch_key = id(forward_context.batch_descriptor)

            prefetch_stream = forward_context._prefetch_streams[batch_key]
            prefetch_events = forward_context._prefetch_events
        
            with torch.cuda.stream(prefetch_stream):
                
                param_names = [
                    "w13_weight",
                    "w2_weight", 
                ] 
                
                if is_temporary:
                    cpu_weights = create_cpu_weights(layer)
                else:
                    cpu_weights = {}
                 
                for param_name in param_names:
                    if is_temporary:
                        weight_cpu = cpu_weights[param_name].contiguous()
                        weight_gpu = torch.zeros_like(weight_cpu, device=device, memory_format=torch.contiguous_format)
                    else:
                        weight_cpu = FusedMoE._cpu_weights_placeholder[batch_id][param_name].contiguous()
                        weight_gpu = FusedMoE._gpu_weights_placeholder[batch_id][param_name].contiguous()
                     
                    
                    is_fp8 =  isinstance(layer.quant_method, Fp8MoEMethod) or isinstance(layer.quant_method, CompressedTensorsW8A8Fp8MoEMethod)
                    is_wna16 = (isinstance(layer.quant_method, CompressedTensorsWNA16MarlinMoEMethod) or isinstance(layer.quant_method, CompressedTensorsWNA16MoEMethod))
                    is_regular = isinstance(layer.quant_method, UnquantizedFusedMoEMethod)
                    if is_fp8 or is_wna16:
                        if param_name == "w13_weight":
                            E = layer.local_num_experts
                            N = layer.intermediate_size_per_partition * 2 if layer.has_gate_proj else layer.intermediate_size_per_partition
                            K = layer.hidden_size
                            shape = (E, N, K * 18 // 32)
                        elif param_name == "w2_weight":
                            E = layer.local_num_experts
                            N = layer.hidden_size
                            K = layer.intermediate_size_per_partition
                            shape = (E, N, K * 18 // 32) 
                        
                        total_elements = shape[0] * shape[1] * shape[2]
                        weight_buffer = weight_cpu[:total_elements].reshape(shape)
                        weight_buffer_gpu = weight_gpu.view(torch.uint8)[:total_elements].reshape(shape)
                        layer.lk_moe.collectWeight(
                            param_name,
                            weight_buffer.data_ptr()
                        ) 
                    elif is_regular:
                        if param_name == "w13_weight":
                            E = layer.local_num_experts
                            N = layer.intermediate_size_per_partition * 2 if layer.has_gate_proj else layer.intermediate_size_per_partition
                            K = layer.hidden_size
                            shape = (E, N, K)
                        elif param_name == "w2_weight":
                            E = layer.local_num_experts
                            N = layer.hidden_size
                            K = layer.intermediate_size_per_partition
                            shape = (E, N, K)
                        weight_buffer = weight_cpu.view(layer.moe_config.in_dtype).reshape(shape)
                        weight_buffer_gpu = weight_gpu.view(layer.moe_config.in_dtype).reshape(shape)
                        if param_name == "w13_weight":
                            if layer.has_gate_proj:
                                layer.lk_moe.collect_weights(
                                    True,  
                                    0,
                                    0,
                                    weight_buffer.data_ptr(),  
                                    0  # 0   gate  
                                )
                                
                                layer.lk_moe.collect_weights(
                                    True,  
                                    0,
                                    0,
                                    weight_buffer.data_ptr(),  
                                    1  # 1   up
                                )
                            else:
                                layer.lk_moe.collect_weights(
                                    True,  
                                    0,
                                    0,
                                    weight_buffer.data_ptr(),  
                                    1  # 1   up
                                )
                        elif param_name == "w2_weight":
                            layer.lk_moe.collect_weights(
                                True,  
                                0,
                                0,
                                weight_buffer.data_ptr(),  
                                2  # w2
                            )
                        else:
                            raise ValueError(f"Unsupported param_name {param_name} for layer")
                    weight_buffer_gpu.copy_(weight_buffer, non_blocking=True)
                    weight_buffer_gpu.record_stream(prefetch_stream) 
                    setattr(layer, param_name, torch.nn.Parameter(weight_buffer_gpu, requires_grad=False))
                
                layer_id = id(layer)
                event = torch.cuda.Event()
                event.record(prefetch_stream)
                batch_key = id(forward_context.batch_descriptor)
                prefetch_events[(layer_id, batch_key)] = event

        
def moe_clean_gpu_prefill(layer, forward_context: ForwardContext):   
    with torch.no_grad():   
        param_names = ["w13_weight", "w2_weight"]
        
        for param_name in param_names:
            if hasattr(layer, param_name):
                setattr(layer, param_name, None)
       
        if hasattr(forward_context, '_prefetch_batch_id'):
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE  
            with FusedMoE._batch_lock:
                batch_id = forward_context._prefetch_batch_id
                if batch_id >= 0:
                    FusedMoE._batch_usage[batch_id] = False
            
            delattr(forward_context, '_prefetch_batch_id')
            if hasattr(forward_context, '_prefetch_batch_key'):
                delattr(forward_context, '_prefetch_batch_key')
  

def moe_cleanup(layer, layer_name: str, hidden_states: torch.Tensor, forward_context: ForwardContext): 
    
    if not layer.should_use_gpu_prefill(hidden_states):
        return
    
    
    layer_idx = extract_layer_index(layer_name)
    batch_key = id(forward_context.batch_descriptor)
     
    if not hasattr(forward_context, '_batch_prefetch_states'):
        return
    if batch_key not in forward_context._batch_prefetch_states:
        return
    
    batch_state = forward_context._batch_prefetch_states[batch_key]
    state = batch_state['state']
     
    keys_to_clean = [k for k in state.keys() if k <= layer_idx]
    
    for k in keys_to_clean: 
        candidate_name = layer_name.replace(f".{layer_idx}.", f".{k}.")
        if is_lk_moe_gpu_resident_layer(candidate_name):
            del state[k]
            continue  
        layer_obj = forward_context.no_compile_layers.get(candidate_name)
        if layer_obj:
            if hasattr(forward_context, '_prefetch_events'):
                layer_id = id(layer_obj)
                batch_key = id(forward_context.batch_descriptor)
                key = (layer_id, batch_key)
                if key in forward_context._prefetch_events:
                    forward_context._prefetch_events[key].wait()
                    del forward_context._prefetch_events[key]
            moe_clean_gpu_prefill(layer_obj, forward_context)
        del state[k]

def moe_prefetch(layer, layer_name: str, hidden_states: torch.Tensor, 
                 forward_context: ForwardContext, gpu_prefetch_window: int): 
    
    if not layer.should_use_gpu_prefill(hidden_states):
        return
    
    if not hasattr(forward_context, '_prefetch_streams'):
        forward_context._prefetch_streams = {}
        
    if not hasattr(forward_context, '_prefetch_events'):
        forward_context._prefetch_events = {}  
        
    if not hasattr(forward_context, '_batch_prefetch_states'):
        forward_context._batch_prefetch_states = {}
    
    layer_idx = extract_layer_index(layer_name)
    batch_key = id(forward_context.batch_descriptor) 
    
    if batch_key not in forward_context._prefetch_streams:
        forward_context._prefetch_streams[batch_key] = torch.cuda.Stream()
     
    if batch_key not in forward_context._batch_prefetch_states:
        forward_context._batch_prefetch_states[batch_key] = {
            'state': {},  # layer_idx -> prefetch_count
            'called_layers': set()
        }
    
    batch_state = forward_context._batch_prefetch_states[batch_key]
    state = batch_state['state']
    called_layers = batch_state['called_layers']
     
    if layer_idx == 0:
        state.clear()
        called_layers.clear()
     
    if layer_name in called_layers:
        return
    
    called_layers.add(layer_name)  
            
    active_prefetches = 0
    for k in state.keys():
        candidate_name = layer_name.replace(f".{layer_idx}.", f".{k}.")
        if not is_lk_moe_gpu_resident_layer(candidate_name):
            active_prefetches += 1
     
    available_slots = gpu_prefetch_window - active_prefetches
    
    layer_count = len(forward_context.no_compile_layers)
    if available_slots > 0: 
        prefetch_candidates = [] 
        for offset in range(0, layer_count): 
            candidate_idx = layer_idx + offset
            candidate_name = layer_name.replace(f".{layer_idx}.", f".{candidate_idx}.")
            
            if candidate_name not in forward_context.no_compile_layers:
                break  
             
            if is_lk_moe_gpu_resident_layer(candidate_name):
                continue
             
            if candidate_idx not in state and len(prefetch_candidates) < available_slots:
                candidate_layer = forward_context.no_compile_layers.get(candidate_name)
                if candidate_layer:
                    prefetch_candidates.append((candidate_idx, candidate_layer))
         
        for idx, layer_obj in prefetch_candidates:
            moe_prepare_gpu_prefill(layer_obj, forward_context, torch.cuda.current_device())
            state[idx] = 1
               
    
                    
def moe_wait_prefetch(layer, hidden_states: torch.Tensor, forward_context: ForwardContext):
 
    if not hasattr(forward_context, '_prefetch_events'):
        return
    
    if not layer.should_use_gpu_prefill(hidden_states):
        return
    
    layer_id = id(layer)
    batch_key = id(forward_context.batch_descriptor)
    prefetch_events = forward_context._prefetch_events
    key = (layer_id, batch_key)
    
    if key in prefetch_events:
        prefetch_events[key].wait()
        del prefetch_events[key] 
    current_stream = torch.cuda.current_stream() 
    if hasattr(layer, 'w13_weight') and layer.w13_weight is not None:
        layer.w13_weight.record_stream(current_stream)
    if hasattr(layer, 'w2_weight') and layer.w2_weight is not None:
        layer.w2_weight.record_stream(current_stream)
        
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.current_stream().synchronize()
