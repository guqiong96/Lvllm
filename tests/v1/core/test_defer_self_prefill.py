# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the async double-schedule self-prefill defer.

When the next prefill chunk is allocated before the in-flight chunk settles
(async scheduling), allocation can fail while the only preemption victim is
the request itself. The scheduler must defer one step (the in-flight chunk's
recyclable-group blocks free on settle) instead of preempting self, which
resets `num_computed_tokens` and recomputes the whole prefill forever.

Shape: single sliding-window group (the recyclable group whose window-plus-
in-flight term caused the real 4-GPU 16k incident). Pool 330 blocks: chunk 1
allocates 256; chunk 2 needs 256 more but only 74 are free while chunk 1 is
still in flight; after settle the window tail (64) frees and chunk 2 fits.
"""

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

BLOCK = 16
WINDOW = 1024
CHUNK = 4096
PROMPT = 8192
NUM_BLOCKS = 330


def _make_scheduler():
    spec = SlidingWindowSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=WINDOW,
    )
    return create_scheduler(
        async_scheduling=True,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK,
        max_num_batched_tokens=CHUNK,
        max_model_len=PROMPT,
        kv_cache_spec=spec,
    )


def _model_output(scheduler_output, req_id, sampled):
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[sampled],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_tight_pool_double_schedule_defers_instead_of_self_preempt(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    scheduler = _make_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=PROMPT, block_size=BLOCK)
    scheduler.add_request(req)

    out1 = scheduler.schedule()
    assert out1.total_num_scheduled_tokens > 0, "chunk 1 must be admitted"
    assert req.num_computed_tokens > 0
    assert req.num_in_flight_tokens > 0

    # Async pattern: second schedule() before update_from_output. Chunk 2's
    # allocation fails against the still-held in-flight window and the only
    # victim is the request itself => defer, state untouched.
    out2 = scheduler.schedule()
    assert req.request_id not in out2.num_scheduled_tokens
    assert len(out2.preempted_req_ids) == 0
    assert req.status != RequestStatus.PREEMPTED
    assert req in scheduler.running
    computed_after_defer = req.num_computed_tokens

    # Settle chunk 1; the freed window blocks let chunk 2 in on the next step.
    # Intermediate prefill chunks sample nothing (the async placeholder
    # bookkeeping asserts on a spurious token).
    mid1 = req.num_computed_tokens < req.num_tokens
    scheduler.update_from_output(
        out1, _model_output(out1, req.request_id, [] if mid1 else [0])
    )
    out3 = scheduler.schedule()
    assert (
        out3.total_num_scheduled_tokens > 0
    ), "chunk 2 must be scheduled once chunk 1 settled"
    assert req.num_computed_tokens >= computed_after_defer
    assert req.status != RequestStatus.PREEMPTED


def test_defer_is_narrow(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    # The defer fires only on (victim == self) and (is_prefill_chunk) and
    # (in-flight > 0). Force allocation to fail and pin each direction.
    scheduler = _make_scheduler()
    real_alloc = scheduler.kv_cache_manager.allocate_slots
    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", lambda *a, **k: None
    )

    (req,) = create_requests(num_requests=1, num_tokens=PROMPT, block_size=BLOCK)
    scheduler.add_request(req)
    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", real_alloc)
    out = scheduler.schedule()  # chunk 1 schedules normally
    assert out.total_num_scheduled_tokens > 0
    assert req.num_in_flight_tokens > 0

    # In-flight prefill chunk, alloc forced to fail => defer, no preempt.
    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", lambda *a, **k: None
    )
    out2 = scheduler.schedule()
    assert out2.total_num_scheduled_tokens == 0
    assert len(out2.preempted_req_ids) == 0
    assert req.status != RequestStatus.PREEMPTED
    monkeypatch.undo()
    mid = req.num_computed_tokens < req.num_tokens
    scheduler.update_from_output(out, _model_output(out, req.request_id, [] if mid else [0]))
