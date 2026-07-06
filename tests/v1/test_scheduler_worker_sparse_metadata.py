# SPDX-License-Identifier: Apache-2.0
"""P2: scheduler-side sparse decode metadata and zombie request behavior."""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    RequestTracker,
    WorkerRetrieveState,
)
from tests.v1.connector_test_utils import make_worker_impl


@dataclass
class StubCachedRequestData:
    req_ids: list[str]
    new_token_ids: list[list[int]]
    new_block_ids: list[list[int]]
    resumed_req_ids: set[str] = field(default_factory=set)


@dataclass
class StubSchedulerOutput:
    finished_req_ids: set[str]
    scheduled_new_reqs: list
    scheduled_cached_reqs: StubCachedRequestData
    num_scheduled_tokens: dict[str, int]


def _make_scheduler_impl() -> LMCacheConnectorV1Impl:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl.enable_sparse_attention = True
    impl.config = MagicMock()
    impl.config.save_decode_cache = False
    impl.config.priority_limit = None
    impl.kv_role = "kv_both"
    impl.force_skip_save = False
    impl._block_size = 16
    impl._lmcache_chunk_size = 256
    impl._decode_window_save_window_size = 0
    impl._discard_partial_chunks = True
    impl._request_trackers = {}
    impl._unfinished_requests = {}
    impl.load_specs = {}
    impl._requests_priority = {}
    return impl


def _make_vllm_request(
    req_id: str,
    prompt_len: int,
    num_computed: int,
    decode_token: int,
) -> SimpleNamespace:
    prompt = list(range(prompt_len))
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=prompt_len,
        prompt_token_ids=prompt,
        num_computed_tokens=num_computed,
        all_token_ids=prompt + [decode_token],
    )


class TestBuildConnectorMetaSparseSyntheticLoadSpec:
    def test_sparse_decode_steps_synthesize_load_spec_and_sparse_tokens(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "sparse-req"
        prompt_len = 128
        vllm_req = _make_vllm_request(req_id, prompt_len, prompt_len + 1, 999)

        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=list(range(8)),
            num_saved_tokens=prompt_len,
        )
        tracker.is_decode_phase = True
        tracker.sparse_token_ids = list(range(prompt_len))
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[999]],
                new_block_ids=[[8]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        assert len(meta.requests) == 1
        req_meta = meta.requests[0]

        assert req_meta.is_sparse_decode
        assert req_meta.load_spec is not None
        assert req_meta.load_spec.can_load is True
        assert req_meta.load_spec.lmcache_cached_tokens == prompt_len
        assert req_meta.token_ids == list(range(prompt_len))
        assert req_meta.cached_keys == []

    def test_multi_step_sparse_decode_reuses_tracker_sparse_token_ids(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "sparse-req"
        prompt_len = 64
        sparse_tokens = list(range(prompt_len))
        vllm_req = _make_vllm_request(req_id, prompt_len, prompt_len + 2, 1001)

        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)) + [1000],
            allocated_block_ids=list(range(10)),
            num_saved_tokens=prompt_len,
        )
        tracker.is_decode_phase = True
        tracker.sparse_token_ids = sparse_tokens.copy()
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[1001]],
                new_block_ids=[[10]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = meta.requests[0]
        assert req_meta.token_ids == sparse_tokens
        assert tracker.sparse_token_ids == sparse_tokens

    def test_decode_window_sparse_load_uses_committed_window_end(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256

        req_id = "sparse-window"
        prompt_len = 356
        prompt = list(range(prompt_len))
        decode_tokens = list(range(10_000, 10_158))
        all_tokens = prompt + decode_tokens
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=prompt_len,
            prompt_token_ids=prompt,
            num_computed_tokens=512,
            all_token_ids=all_tokens,
        )

        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=all_tokens[:512],
            allocated_block_ids=list(range(40)),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True
        tracker.sparse_token_ids = all_tokens[:256]
        tracker.sparse_slot_mapping = [torch.arange(256)]
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[all_tokens[512]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = next(req for req in meta.requests if req.is_sparse_decode)

        assert req_meta.load_spec is not None
        assert req_meta.load_spec.can_load is True
        assert req_meta.load_spec.lmcache_cached_tokens == 256
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.can_save is True
        assert req_meta.save_spec.skip_leading_tokens == 256
        assert req_meta.decode_window_start == 256
        assert req_meta.decode_window_end == 512
        assert req_meta.token_ids == all_tokens[:512]
        assert req_meta.cached_tensors is tracker.cached_tensors
        assert req_meta.cached_chunk_ptrs_npu is tracker.cached_chunk_ptrs_npu
        assert tracker.sparse_token_ids == all_tokens[:256]
        assert req_meta.slot_mapping[0].numel() == 512
        assert tracker.sparse_slot_mapping[0].numel() == 512

        impl.update_connector_output(
            SimpleNamespace(completed_decode_window_saves={req_id: 512})
        )
        vllm_req.num_computed_tokens = 513
        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[all_tokens[513]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = next(req for req in meta.requests if req.is_sparse_decode)

        assert req_meta.load_spec is not None
        assert req_meta.load_spec.can_load is True
        assert req_meta.load_spec.lmcache_cached_tokens == 512
        assert req_meta.token_ids == all_tokens[:512]
        assert tracker.sparse_token_ids == all_tokens[:512]
        assert req_meta.slot_mapping[0].numel() == 512
        assert tracker.sparse_slot_mapping[0].numel() == 512


class TestZombieRequestInMetadata:
    def test_finished_request_still_in_metadata_keeps_worker_state(self) -> None:
        impl = make_worker_impl()
        impl._worker_retrieve_state = {
            "zombie": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
            "active": WorkerRetrieveState(metadata_warm=True),
        }

        impl._prune_worker_retrieve_state({"zombie", "active"})
        assert "zombie" in impl._worker_retrieve_state
        assert "active" in impl._worker_retrieve_state

    def test_request_removed_from_metadata_prunes_zombie_state(self) -> None:
        impl = make_worker_impl()
        engine = MagicMock()
        impl._manager = SimpleNamespace(lmcache_engine=engine)
        impl._worker_retrieve_state = {
            "zombie": WorkerRetrieveState(metadata_warm=True),
            "active": WorkerRetrieveState(metadata_warm=True),
        }

        impl._prune_worker_retrieve_state({"active"})

        assert "zombie" not in impl._worker_retrieve_state
        assert "active" in impl._worker_retrieve_state
        engine.lookup_unpin.assert_called_once_with("zombie")

    def test_scheduler_clears_tracker_on_finished_but_worker_needs_prune(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "finished-req"
        impl._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            prompt_len=32,
            token_ids=list(range(32)),
            allocated_block_ids=[0],
        )
        impl._unfinished_requests[req_id] = SimpleNamespace(request_id=req_id)

        scheduler_output = StubSchedulerOutput(
            finished_req_ids={req_id},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[],
                new_token_ids=[],
                new_block_ids=[],
            ),
            num_scheduled_tokens={},
        )

        impl.build_connector_meta(scheduler_output)
        assert req_id not in impl._request_trackers
        assert req_id not in impl._unfinished_requests


class TestDecodeWindowSaveMetadata:
    def test_decode_window_save_includes_prefill_tail_at_boundary(self) -> None:
        impl = _make_scheduler_impl()
        impl.enable_sparse_attention = False
        impl._decode_window_save_window_size = 256

        req_id = "decode-window"
        prompt_len = 356
        prompt = list(range(prompt_len))
        decode_tokens = list(range(10_000, 10_156))
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=prompt_len,
            prompt_token_ids=prompt,
            num_computed_tokens=prompt_len + 155,
            all_token_ids=prompt + decode_tokens,
        )
        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=prompt + decode_tokens[:-1],
            allocated_block_ids=list(range(32)),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_tokens[-1]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 1
        req_meta = meta.requests[0]
        assert req_meta.decode_window_start == 256
        assert req_meta.decode_window_end == 512
        assert req_meta.decode_window_size == 256
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.skip_leading_tokens == 256
        assert req_meta.token_ids == prompt + decode_tokens
        assert len(req_meta.slot_mapping[0]) == 512
        assert req_meta.cached_tensors is tracker.cached_tensors
        assert req_meta.cached_chunk_ptrs_npu is tracker.cached_chunk_ptrs_npu
        assert tracker.num_saved_tokens == 512
        assert tracker.decode_window_save_next_start == 512
        assert tracker.decode_window_save_committed_end == 256

    def test_decode_window_save_waits_until_boundary(self) -> None:
        impl = _make_scheduler_impl()
        impl.enable_sparse_attention = False
        impl._decode_window_save_window_size = 256

        req_id = "decode-window"
        prompt_len = 356
        prompt = list(range(prompt_len))
        decode_tokens = list(range(10_000, 10_155))
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=prompt_len,
            prompt_token_ids=prompt,
            num_computed_tokens=prompt_len + 154,
            all_token_ids=prompt + decode_tokens,
        )
        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=prompt + decode_tokens[:-1],
            allocated_block_ids=list(range(32)),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_tokens[-1]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert meta.requests == []
        assert tracker.num_saved_tokens == 256
        assert tracker.decode_window_save_next_start == 256
