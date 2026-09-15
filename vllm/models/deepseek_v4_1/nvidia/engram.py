# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA Engram DP sharding, shared host storage, and asynchronous prefetch."""

import glob
import json
import mmap
import os
import re
import tempfile
import weakref
from contextlib import ExitStack

import numpy as np
import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_engram_dp_group,
    get_engram_dp_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4_1.common.engram import (
    DEAD_ID,
    EngramLayout,
    _engram_head_shard_weight_loader,
    _engram_select_rows,
)
from vllm.models.deepseek_v4_1.common.engram import (
    Engram as BaseEngram,
)
from vllm.models.deepseek_v4_1.common.engram import (
    ParallelEngramEmbedding as BaseParallelEngramEmbedding,
)
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

_page_cache_dropped_phases: set[str] = set()


def _mapped_file_identities() -> set[tuple[int, int]]:
    """(st_dev, inode) pairs of regular files currently mapped by this
    process. The maps dev column is ``major:minor`` in hex."""
    identities: set[tuple[int, int]] = set()
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                if len(fields) < 6 or not fields[5].startswith("/"):
                    continue
                major, _, minor = fields[3].partition(":")
                identities.add(
                    (
                        os.makedev(int(major, 16), int(minor, 16)),
                        int(fields[4]),
                    )
                )
    except (OSError, ValueError):
        pass
    return identities


def _open_file_identities() -> set[tuple[int, int]]:
    """(st_dev, st_ino) of every file currently open by any process on this
    machine: another rank's mmap or prefetch thread reading a checkpoint file
    looks idle to us, but evicting under it buys everyone a second pass of
    synchronous 4 KB page faults."""
    identities: set[tuple[int, int]] = set()
    try:
        pids = os.listdir("/proc")
    except OSError:
        return identities
    for pid in pids:
        if not pid.isdigit():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if not target.startswith("/") or target.endswith(" (deleted)"):
                continue
            try:
                stat = os.stat(target)
            except OSError:
                continue
            identities.add((stat.st_dev, stat.st_ino))
    return identities


def drop_checkpoint_page_cache(model_dir: str) -> tuple[int, int]:
    """Drop cached checkpoint pages with posix_fadvise(DONTNEED); return
    (files, bytes). Files still mapped by this process or open by anyone
    (mmap users, other ranks' prefetchers) are skipped: evicting a file
    mid-read buys a second pass of synchronous 4 KB page faults (the
    double-read storm seen with per-file hooks during loading).
    """
    in_use = _mapped_file_identities() | _open_file_identities()
    files = num_bytes = 0
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        try:
            stat = os.stat(path)
            if (stat.st_dev, stat.st_ino) in in_use:
                continue
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            files += 1
            num_bytes += stat.st_size
        finally:
            os.close(fd)
    return files, num_bytes


def _drop_page_cache_once(phase: str) -> None:
    """One-shot checkpoint page-cache drop per phase (sglang port): a single
    time point per process, so pre-faulting the host table and the loads that
    follow it find free memory instead of thrashing a full page cache.
    """
    if not envs.VLLM_ENGRAM_DROP_PAGE_CACHE or phase in _page_cache_dropped_phases:
        return
    _page_cache_dropped_phases.add(phase)
    try:
        model_dir = get_current_vllm_config().model_config.model
    except (AssertionError, AttributeError, ValueError):
        return
    if not model_dir or not os.path.isdir(model_dir):
        return
    files, num_bytes = drop_checkpoint_page_cache(model_dir)
    if files:
        logger.info(
            "Engram host table: dropped the page cache of %d checkpoint "
            "files (%.1f GiB) before %s",
            files,
            num_bytes / 2**30,
            "pre-faulting the host table"
            if phase == "allocate"
            else "filling the table",
        )


def load_deferred_engram_tables(
    root_module: torch.nn.Module,
    model_dir: str,
    patterns: tuple[re.Pattern, ...],
) -> None:
    """Dedicated second pass for the weights the main sweep skipped via
    ``weight_load_skip_patterns`` (the Engram host tables): drop the
    checkpoint page cache once, re-read only those tensors, stream them
    through the model's normal loader (so the usual TP/DP-sharded
    weight_loaders run and record the fills), then flush the pinned host
    tables. Runs strictly after all regular load + after-processing, so
    the huge table faults into freed memory as the last load step.
    """
    if not envs.VLLM_ENGRAM_DEFER_HOST_FILL:
        return
    from safetensors import safe_open

    from vllm.model_executor.models.utils import AutoWeightsLoader

    index = os.path.join(model_dir, "model.safetensors.index.json")
    by_file: dict[str, list[str]] = {}
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        for name, shard in weight_map.items():
            if any(p.search(name) for p in patterns):
                by_file.setdefault(os.path.join(model_dir, shard), []).append(name)
    else:
        for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            by_file[shard] = []

    wanted = sum(len(names) for names in by_file.values())
    if not any(
        isinstance(m, ParallelEngramEmbedding) for m in root_module.modules()
    ):
        return
    if wanted == 0:
        raise RuntimeError(
            "Engram deferred load: the main sweep skipped the host tables but "
            f"no matching tensors were found under {model_dir}."
        )
    logger.info(
        "Engram host table: deferred load pass starting over %d file(s)",
        len(by_file),
    )
    _drop_page_cache_once("fill")

    def gen():
        for shard, names in sorted(by_file.items()):
            with safe_open(shard, framework="pt") as sf:
                if not names:  # no index: match against this file's keys
                    names = [n for n in sf.keys() if any(p.search(n) for p in patterns)]
                for name in names:
                    yield name, sf.get_tensor(name)

    # The generator yields raw HF checkpoint names, so resolve them with the
    # model's *checkpoint* mapper (its normal pass-1 mapper).  Under the VL
    # wrapper the live ``hf_to_vllm_mapper`` is deliberately identity (the
    # outer wrapper already re-rooted names), so fall back to the stashed
    # checkpoint mapper when present.
    mapper = getattr(root_module, "_checkpoint_hf_to_vllm_mapper", None)
    if mapper is None:
        mapper = getattr(root_module, "hf_to_vllm_mapper", None)
    loader = AutoWeightsLoader(root_module)
    loader.load_weights(gen(), mapper=mapper)
    for module in root_module.modules():
        flush = getattr(module, "flush_deferred_host_fills", None)
        if callable(flush):
            flush()


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(
        dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max()
    )


def gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False
) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike).
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full(
            (slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID
        )
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0)


class DPSharedEngramStorage:
    """Registered host weights shared by a node-local DP group with one writer."""

    def __init__(
        self, num_rows: int, dim: int, block_size: int, group: GroupCoordinator
    ) -> None:
        self.group = group
        weight_bytes = num_rows * dim
        storage = self._allocate(weight_bytes + weight_bytes // block_size)
        self.weight = storage[:weight_bytes].view(torch.float8_e4m3fn).view(-1, dim)
        self.weight_scale_inv = storage[weight_bytes:].view(-1, dim // block_size)
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None

    def _allocate(self, num_bytes: int) -> torch.Tensor:
        """Map and register one physical allocation across a node-local DP group."""
        from vllm.distributed.device_communicators.shm_broadcast import (
            check_shm_free_space,
        )

        _drop_page_cache_once("allocate")
        group = self.group
        with ExitStack() as stack:
            path, error = None, None
            if group.rank_in_group == 0:
                try:
                    check_shm_free_space(num_bytes)
                    backing_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(
                            prefix="vllm_engram_", dir="/dev/shm"
                        )
                    )
                    backing_file.truncate(num_bytes)
                    path = backing_file.name
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            path, error = group.broadcast_object((path, error))
            if error is not None:
                raise RuntimeError(
                    "Engram shared-memory creation failed on EDP rank 0: " + error
                )

            mapping = owner = tensor = finalizer = None
            stage = "open"
            try:
                try:
                    with open(path, "r+b") as file:
                        stage = "mmap"
                        mapping = mmap.mmap(
                            file.fileno(), num_bytes, flags=mmap.MAP_SHARED
                        )
                    stage = "cudaHostRegister"
                    owner = np.frombuffer(mapping, dtype=np.uint8)
                    pointer = owner.ctypes.data
                    tensor = torch.from_numpy(owner)
                    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
                    if result.value != 0:
                        raise RuntimeError(f"cudaHostRegister failed: {result}")
                    finalizer = weakref.finalize(
                        owner, self._unregister, mapping, pointer
                    )
                    finalizer.atexit = False  # type: ignore[misc]
                    # The UVA helper otherwise allocates a private pinned copy.
                    if not tensor.is_pinned():
                        raise RuntimeError(
                            "CUDA did not recognize the shared Engram registration"
                        )
                except Exception as exc:
                    error = f"{stage}: {type(exc).__name__}: {exc}"

                errors: list[str | None] = [None] * group.world_size
                # Also fences peer mappings before the leader unlinks the file.
                torch.distributed.all_gather_object(
                    errors, error, group=group.cpu_group
                )
                failures = "; ".join(
                    f"EDP rank {rank}: {error}"
                    for rank, error in enumerate(errors)
                    if error is not None
                )
                if failures:
                    raise RuntimeError(
                        "Engram shared-memory initialization failed: " + failures
                    )
                assert tensor is not None
                return tensor
            except Exception:
                if finalizer is not None:
                    finalizer()
                tensor = owner = None
                if mapping is not None:
                    mapping.close()
                raise

    @staticmethod
    def _unregister(mapping: mmap.mmap, pointer: int) -> None:
        # Torch storage retains the numpy owner, including through cached UVA views.
        # Keep its mmap alive until CUDA has released the registration.
        result = torch.cuda.cudart().cudaHostUnregister(pointer)
        if result.value != 0:
            logger.warning("Engram cudaHostUnregister failed: %s", result)

    def load_weight(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        if self.group.rank_in_group == 0:
            _engram_head_shard_weight_loader(param, loaded_weight)
        # Read order may differ across ranks. Equal load counts ensure all shared
        # weights are ready after the last weight-loader call returns.
        torch.distributed.barrier(group=self.group.cpu_group)

    def get_views(
        self, weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (weight.data_ptr(), scales.data_ptr()) != (
            self.weight.data_ptr(),
            self.weight_scale_inv.data_ptr(),
        ):
            raise RuntimeError("Shared Engram parameter storage must not be replaced")
        if self._views is None:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv),
            )
        return self._views


class ParallelEngramEmbedding(BaseParallelEngramEmbedding):
    """Extend TP lookup with DP head sharding or shared, CPU-offloaded TP slices."""

    _shared_memory: DPSharedEngramStorage | None = None

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
    ) -> None:
        self.cpu_offload = cpu_offload
        self.dp_shared_memory = dp_shared_memory
        self.dp_size = get_engram_dp_size()
        if dp_shared_memory:
            if not cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            if self.dp_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires a node-local Engram DP "
                    f"group with size > 1; effective Engram DP size is {self.dp_size}. "
                    "Check that the node layout and rank placement allow complete "
                    "DP replicas to be co-located."
                )
            self.dp_size = 1
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        # True while the host table is only a lazy placeholder; the real
        # page-locked buffer is created in flush_deferred_host_fills().
        self._host_table_deferred = False
        # (param, checkpoint-view) pairs recorded when host fills are deferred
        # past the checkpoint sweep (VLLM_ENGRAM_DEFER_HOST_FILL). The views
        # keep their mmap alive through reference counting, so filling later
        # is safe; the page cache just gets dropped first.
        self._pending_host_fills: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        super().__init__(num_embeddings, dim, head_sizes, block_size)
        if cpu_offload:
            # Constant dummy values avoid randomizing huge CPU lookup tables.
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            # The ue8m0 encoding of scale 1.0 is exponent byte 127.
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})
            logger.info(
                "Engram table offloaded to pinned host memory: %d rows x %d, "
                "%.2f GiB %s",
                self.part_num_embeddings,
                dim,
                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,
                "shared across DP replicas" if dp_shared_memory else "per rank",
            )

    def _get_shard_info(self) -> tuple[int, int]:
        if self.dp_size == 1:
            return super()._get_shard_info()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dp_shared_memory:
            group = get_engram_dp_group()
            assert group is not None
            storage = DPSharedEngramStorage(
                self.part_num_embeddings, self.dim, self.block_size, group
            )
            self._shared_memory = storage
            self._weight_loader = storage.load_weight
            self._wrap_weight_loader_with_drop()
            return storage.weight, storage.weight_scale_inv
        if not self.cpu_offload:
            return super()._allocate_weights()
        if envs.VLLM_ENGRAM_DEFER_HOST_FILL:
            # Do not commit or page-lock ~189 GiB at construction. Hand back a
            # lazy, untouched CPU shape (device="cpu" so a surrounding meta
            # device context cannot make it a meta tensor; ~0 resident RSS) and
            # record the true pinned buffer's shape. flush_deferred_host_fills
            # () creates the real page-locked tensor right after the page-cache
            # drop, so the table no longer sits allocated across the whole
            # expert sweep -- matching sglang's allocate-late peak.
            self._host_table_deferred = True
            self._wrap_weight_loader_with_drop()
            return (
                torch.empty(
                    self.part_num_embeddings,
                    self.dim,
                    dtype=torch.float8_e4m3fn,
                    device="cpu",
                ),
                torch.empty(
                    self.part_num_embeddings,
                    self.dim // self.block_size,
                    dtype=torch.uint8,
                    device="cpu",
                ),
            )
        _drop_page_cache_once("allocate")
        # Model initialization may be inside a CUDA device context.
        weights = (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                device="cpu",
                pin_memory=True,
            ),
            torch.empty(
                self.part_num_embeddings,
                self.dim // self.block_size,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            ),
        )
        self._wrap_weight_loader_with_drop()
        return weights

    def _wrap_weight_loader_with_drop(self) -> None:
        """Drop the page cache once (per process) when the host-table fill
        starts, at the one time point where earlier checkpoint pages are no
        longer needed and only the still-mapped source file is protected.

        With VLLM_ENGRAM_DEFER_HOST_FILL the fill is instead recorded and run
        after the checkpoint sweep ends (``flush_deferred_host_fills``), so
        the huge pinned-table pages are touched with an empty page cache
        instead of competing with the loader for memory.
        """
        base_loader = self._weight_loader
        self._host_fill_loader = base_loader

        def load_with_drop(
            param: torch.nn.Parameter, loaded_weight: torch.Tensor
        ) -> None:
            if envs.VLLM_ENGRAM_DEFER_HOST_FILL:
                self._pending_host_fills.append((param, loaded_weight))
                return
            _drop_page_cache_once("fill")
            base_loader(param, loaded_weight)

        self._weight_loader = load_with_drop

    def flush_deferred_host_fills(self) -> None:
        """Run the host-table fills that were deferred during the checkpoint
        sweep: one page-cache drop, then pre-fault the pinned table into the
        freed memory. Idempotent; safe to call from every post-load hook."""
        pending = getattr(self, "_pending_host_fills", None)
        if not pending and not self._host_table_deferred:
            return
        if pending:
            logger.info(
                "Engram host table: flushing %d deferred fills (of %d rows) after "
                "the checkpoint sweep",
                len(pending),
                self.part_num_embeddings,
            )
        _drop_page_cache_once("fill")
        if self._host_table_deferred:
            # The drop just freed the checkpoint page cache; take that memory
            # now for the page-locked table, in place of the lazy placeholder
            # handed back at construction, so CUDA's UVA helper never has to
            # make a private pinned copy of it.
            self.weight.data = torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                device="cpu",
                pin_memory=True,
            )
            self.weight_scale_inv.data = torch.empty(
                self.part_num_embeddings,
                self.dim // self.block_size,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            self._host_table_deferred = False
            self._views = None
            self._view_src = None
        fill = self._host_fill_loader
        while pending:
            param, loaded_weight = pending.pop(0)
            fill(param, loaded_weight)

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_pending_host_fills", None) or self._host_table_deferred:
            # A post-load hook did not flush the deferred fills (or the table is
            # still a lazy placeholder); do it at first use so a lookup never
            # reads an empty or un-pinned table.
            self.flush_deferred_host_fills()
        if self._shared_memory is not None:
            return self._shared_memory.get_views(self.weight, self.weight_scale_inv)
        if not self.cpu_offload:
            return super()._storage()
        src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr())
        if self._view_src != src:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight.data),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),
            )
            self._view_src = src
        assert self._views is not None
        return self._views

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.dp_size == 1:
            return super().forward(indices)
        num_tokens = indices.shape[0]
        indices = gather_engram_hashes(indices)
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        out = _gather_engram_rows(out, num_tokens)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
        return out[:, : self.n_hash_cols]


def _gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows


class Engram(BaseEngram):
    """NVIDIA Engram with asynchronous offload and node-local DP lookup."""

    _prefetch_stream: torch.cuda.Stream | None = None

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        engram_config = get_current_vllm_config().engram_config
        assert engram_config is not None
        return ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload,
            dp_shared_memory=engram_config.dp_shared_memory,
        )

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        super()._init_staging(max_tokens * self.embed_tokens.dp_size, head_dim)
        if self.embed_tokens.cpu_offload:
            self._prefetch_stream = torch.cuda.Stream(device=self.staged_rows.device)

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Prefetch local shared rows or the DP group's gathered hash IDs."""
        if self._prefetch_stream is None:
            return super().prepare_embeddings(hash_ids)
        rows = self.staged_rows[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        self._start_prefetch(hash_ids, rows, self._prefetch_stream)

    @eager_break_during_capture
    def _start_prefetch(
        self, hash_ids: torch.Tensor, rows: torch.Tensor, stream: torch.cuda.Stream
    ) -> None:
        # Eager boundaries let the lookup span piecewise graph segments.
        stream.wait_stream(torch.cuda.current_stream())
        # Keep temporary hash storage alive until lookup finishes reading it.
        hash_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            self.embed_tokens.lookup(hash_ids, rows, background=True)

    @eager_break_during_capture
    def _finish_prefetch(self, stream: torch.cuda.Stream) -> None:
        torch.cuda.current_stream().wait_stream(stream)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if self._prefetch_stream is not None:
            self._finish_prefetch(self._prefetch_stream)
        if self.embed_tokens.dp_size > 1:
            slot = engram_gathered_num_tokens()
            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]
            return _gather_engram_rows(staged, num_tokens)
        return super()._ready_rows(num_tokens)
