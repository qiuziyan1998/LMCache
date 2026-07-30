# SPDX-License-Identifier: Apache-2.0
"""Tests for worker-local sparse decode retrieve state cache."""

# Standard
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_mod
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
    WorkerRetrieveState,
)
from lmcache.v1.cache_engine import LayerwiseStoreResult, LMCacheEngine
from lmcache.v1.gpu_connector.sparse import PreparedSparseSource
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from tests.v1.connector_test_utils import (
    make_sparse_req_meta,
    make_worker_connector,
)


def _make_impl() -> LMCacheConnectorV1Impl:
    impl = object.__new__(LMCacheConnectorV1Impl)
    impl._worker_retrieve_state = {}
    impl._worker_retrieve_registry_version = 0
    impl._worker_retrieve_last_prune_key = None
    impl._layerwise_save_storers = {}
    impl._deferred_latent_pending = set()
    impl.kv_role = "kv_both"
    impl._late_finished_sending = set()
    return impl


def _make_shared_engine(
    *,
    rank0: bool,
    generation: int = 9,
) -> LMCacheEngine:
    engine = object.__new__(LMCacheEngine)
    engine.enable_shared_cpu_cache = True
    engine.shared_cpu_cache_generation = generation
    engine._shared_cpu_request_leases = {}
    engine.metadata = SimpleNamespace(
        is_first_rank=lambda: rank0,
        world_size=1,
    )
    engine.store_location = "LocalCPUBackend"
    engine.config = SimpleNamespace(extra_config={})
    engine.lookup_unpin = MagicMock()
    return engine


def _make_group_order_impl(
    kv_caches: dict[str, torch.Tensor],
) -> LMCacheConnectorV1Impl:
    impl = object.__new__(LMCacheConnectorV1Impl)
    impl.config = SimpleNamespace(dsa_two_groups=True)
    impl.kv_caches = kv_caches
    impl.lmcache_engine = SimpleNamespace(
        metadata=SimpleNamespace(kv_layer_groups_manager=KVLayerGroupsManager())
    )
    return impl


def _make_store_request(
    impl: LMCacheConnectorV1Impl,
    *,
    token_count: int,
    start: int,
    end: int,
    key: str,
    tensor: str,
    decode_window: tuple[int, int] | None = None,
) -> tuple[ReqMeta, LayerwiseStoreResult]:
    request = ReqMeta(
        req_id="req-1",
        token_ids=[0] * token_count,
        is_sparse_decode=False,
        is_decode_window_save=decode_window is not None,
        decode_window_start=decode_window[0] if decode_window else None,
        decode_window_end=decode_window[1] if decode_window else None,
        decode_window_size=(
            decode_window[1] - decode_window[0]
            if decode_window
            else None
        ),
    )
    result = LayerwiseStoreResult(
        request_id=request.req_id,
        starts=[start],
        ends=[end],
        keys=[[key]],
        memory_objs=[[f"mem-{key}"]],
        tensors=[[tensor]],
    )
    return request, result


def _make_request(*, resumed: bool = False) -> ReqMeta:
    return ReqMeta(
        req_id="req-1",
        token_ids=[1, 2, 3],
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=3,
            can_load=True,
        ),
        is_sparse_decode=True,
        resumed_from_preemption=resumed,
    )


def _bind_worker_state(impl: LMCacheConnectorV1Impl, request: ReqMeta):
    return impl._worker_retrieve_state_for_request(request)


class TestWorkerRetrieveState:
    @staticmethod
    def _deep_snapshot(state: WorkerRetrieveState) -> dict[str, object]:
        return {
            "frontier": state.token_count,
            "kv_group0_cached_ranges": {
                "count": 1,
                "first_start": 0,
                "last_end": 256,
            },
            "kv_group1_cached_ranges": {
                "count": 0,
                "first_start": None,
                "last_end": None,
            },
            "kv_group0_cache_present": True,
            "kv_group1_cache_present": False,
            "scope_token": None,
            "scope_token_present": False,
            "shared_request_active": False,
            "shared_generation": 0,
            "pointer_cache_generation": 0,
        }

    def test_deep_retrieve_state_requires_both_gates_and_is_bounded_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        impl = _make_impl()
        state = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
        )
        request = _make_request()
        request.token_ids = [0] * 512
        request.load_spec.lmcache_cached_tokens = 512
        events = []
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
        monkeypatch.delenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", raising=False)
        monkeypatch.setattr(
            adapter_mod,
            "_mtp_dw_event",
            lambda stage, **fields: events.append({"stage": stage, **fields}),
        )

        impl._trace_deep_retrieve_state(
            request,
            state,
            self._deep_snapshot(state),
            invalidated=True,
            invalidation_reason="load_frontier_advanced",
            post_state=None,
            prior_state_rebound=False,
            kv_group0_retriever_present=True,
            kv_group1_retriever_present=False,
        )
        assert events == []
        assert not hasattr(impl, "_mtp_dw_deep_retrieve_transitions")

        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
        for _ in range(2):
            impl._trace_deep_retrieve_state(
                request,
                state,
                self._deep_snapshot(state),
                invalidated=True,
                invalidation_reason="load_frontier_advanced",
                post_state=None,
                prior_state_rebound=False,
                kv_group0_retriever_present=True,
                kv_group1_retriever_present=False,
            )

        assert len(events) == 1
        assert events[0]["stage"] == "deep"
        assert events[0]["event"] == "retrieve_state"
        assert events[0]["prior_frontier"] == 256
        assert events[0]["frontier"] == 512
        assert events[0]["prior_kv_group0_cached_ranges"]["last_end"] == 256
        assert events[0]["invalidation_reason"] == "load_frontier_advanced"
        assert events[0]["window_start"] == 256
        assert events[0]["window_end"] == 512

    def test_deep_retrieve_state_records_first_frontier_without_prior_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        impl = _make_impl()
        request = _make_request()
        request.load_spec.lmcache_cached_tokens = 256
        post_state = WorkerRetrieveState(token_count=256)
        events = []
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
        monkeypatch.setattr(
            adapter_mod,
            "_mtp_dw_event",
            lambda stage, **fields: events.append({"stage": stage, **fields}),
        )

        impl._trace_deep_retrieve_state(
            request,
            None,
            None,
            invalidated=False,
            invalidation_reason=None,
            post_state=post_state,
            prior_state_rebound=False,
            kv_group0_retriever_present=True,
            kv_group1_retriever_present=False,
        )

        assert len(events) == 1
        assert events[0]["prior_frontier"] == 0
        assert events[0]["prior_state_present"] is False
        assert events[0]["post_state_present"] is True
        assert events[0]["stale_state_retained"] is False

    def test_deep_retrieve_state_reports_stale_state_without_blocking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        impl = _make_impl()
        state = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
        )
        request = _make_request()
        request.token_ids = [0] * 512
        request.load_spec.lmcache_cached_tokens = 512
        events = []
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
        monkeypatch.setattr(
            adapter_mod,
            "_mtp_dw_event",
            lambda stage, **fields: events.append({"stage": stage, **fields}),
        )

        impl._trace_deep_retrieve_state(
            request,
            state,
            self._deep_snapshot(state),
            invalidated=False,
            invalidation_reason=None,
            post_state=state,
            prior_state_rebound=True,
            kv_group0_retriever_present=True,
            kv_group1_retriever_present=False,
        )

        assert [event["stage"] for event in events] == ["fail", "deep"]
        assert events[0]["invariant"] == "stale_retrieve_state"
        assert events[1]["stale_state_retained"] is True

    def test_deep_retrieve_state_does_not_flag_rebound_state_as_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        impl = _make_impl()
        prior_state = WorkerRetrieveState(token_count=256)
        rebound_state = WorkerRetrieveState(token_count=512)
        request = _make_request()
        request.token_ids = [0] * 512
        request.load_spec.lmcache_cached_tokens = 512
        events = []
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
        monkeypatch.setattr(
            adapter_mod,
            "_mtp_dw_event",
            lambda stage, **fields: events.append({"stage": stage, **fields}),
        )

        impl._trace_deep_retrieve_state(
            request,
            prior_state,
            self._deep_snapshot(prior_state),
            invalidated=False,
            invalidation_reason=None,
            post_state=rebound_state,
            prior_state_rebound=False,
            kv_group0_retriever_present=True,
            kv_group1_retriever_present=False,
        )

        assert [event["stage"] for event in events] == ["deep"]
        assert events[0]["stale_state_retained"] is False
        assert events[0]["post_kv_group0_cache_present"] is False

    def test_deep_group_presence_includes_pointer_only_cache(self) -> None:
        state = WorkerRetrieveState(
            cached_chunk_ptrs_npu=[torch.tensor([1], dtype=torch.long)],
            cached_shared_handles_indexer=[["handle"]],
        )

        assert LMCacheConnectorV1Impl._deep_retrieve_group_cache_present(state, 0)
        assert LMCacheConnectorV1Impl._deep_retrieve_group_cache_present(state, 1)

        empty_state = WorkerRetrieveState(
            cached_keys=[[]],
            cached_chunk_ptrs_npu=[None],
            cached_shared_handles_indexer=[[]],
        )
        assert not LMCacheConnectorV1Impl._deep_retrieve_group_cache_present(
            empty_state, 0
        )
        assert not LMCacheConnectorV1Impl._deep_retrieve_group_cache_present(
            empty_state, 1
        )

    def test_deep_range_summary_uses_only_complete_ranges(self) -> None:
        assert LMCacheConnectorV1Impl._deep_retrieve_range_summary([0], []) == {
            "count": 0,
            "first_start": None,
            "last_end": None,
        }
    def test_failed_block_reporting_ignores_unmapped_tokens(self):
        impl = _make_impl()
        impl._block_size = 16

        missing_blocks = impl.record_failed_blocks(
            "req-1",
            torch.tensor([True, True, True, True]),
            torch.tensor([True, False, True, False]),
            torch.tensor([0, 16, 32]),
        )

        assert missing_blocks == {1}

    def test_sparse_decode_load_tokens_reuses_full_prefix_list(self):
        tokens = [1, 2, 3, 4]

        full = LMCacheConnectorV1Impl._load_tokens_for_retrieve(
            tokens,
            4,
            is_sparse_decode=True,
        )
        longer = LMCacheConnectorV1Impl._load_tokens_for_retrieve(
            tokens,
            8,
            is_sparse_decode=True,
        )
        partial = LMCacheConnectorV1Impl._load_tokens_for_retrieve(
            tokens,
            2,
            is_sparse_decode=True,
        )

        assert full is tokens
        assert longer is tokens
        assert partial == [1, 2]
        assert partial is not tokens

    def test_sparse_decode_load_mask_uses_none_for_full_lmcache_prefix(self):
        req = _make_request()
        req.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4,
            can_load=True,
        )
        req.decode_token_mask = torch.ones(4, dtype=torch.bool)

        mask = LMCacheConnectorV1Impl._load_token_mask_for_retrieve(
            req,
            4,
            lmcache_chunk_size=256,
        )

        assert mask is None
        assert req.decode_token_mask is None

    def test_dsa_kv_metadata_group_order_is_semantic(self):
        impl = _make_group_order_impl(
            {
                "model.layers.0.self_attn.indexer.k_cache": (
                    torch.empty((1, 8, 1, 4), dtype=torch.uint8),
                ),
                "model.layers.0.self_attn.attn.k_cache": torch.empty(
                    (1, 8, 512), dtype=torch.bfloat16
                ),
            }
        )

        impl._refresh_kvcaches_list()
        impl._build_kv_layer_groups()

        groups = impl.lmcache_engine.metadata.kv_layer_groups_manager.kv_layer_groups
        assert len(groups) == 2
        assert groups[0].dtype == torch.bfloat16
        assert groups[0].layer_names == ["model.layers.0.self_attn.attn.k_cache"]
        assert groups[1].dtype == torch.uint8
        assert groups[1].layer_names == [
            "model.layers.0.self_attn.indexer.k_cache"
        ]

    def test_shared_cpu_cache_state_tracks_skipped_index(self):
        state = WorkerRetrieveState(
            shared_latent_status="present",
            shared_index_status="skipped",
            shared_generation=3,
            pointer_cache_generation=3,
            shared_request_active=True,
            request_scope_token="req-1:3",
        )

        assert state.shared_latent_status == "present"
        assert state.shared_index_status == "skipped"
        assert state.shared_generation == 3
        assert state.pointer_cache_generation == 3
        assert state.shared_request_active is True
        assert state.request_scope_token == "req-1:3"

    def test_sparse_decode_index_materialization_policy_for_shared_cpu_kv_both(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_materialize_index_on_decode_cold=True,
        )
        request = _make_request()

        assert (
            impl._sparse_decode_requires_index_materialization(
                request,
                shared_cpu_enabled=True,
            )
            is True
        )

    def test_sparse_decode_index_materialization_policy_for_non_shared_kv_both(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=False,
            shared_cpu_materialize_index_on_decode_cold=True,
        )
        request = _make_request()

        assert (
            impl._sparse_decode_requires_index_materialization(
                request,
                shared_cpu_enabled=False,
            )
            is False
        )

    def test_sparse_decode_index_materialization_policy_for_consumer(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_consumer"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_materialize_index_on_decode_cold=True,
        )
        request = _make_request()

        assert (
            impl._sparse_decode_requires_index_materialization(
                request,
                shared_cpu_enabled=True,
            )
            is True
        )

    def test_sparse_decode_index_materialization_policy_for_kv_both_disagg_metadata(
        self,
    ):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_materialize_index_on_decode_cold=True,
        )
        request = _make_request()
        request.disagg_spec = object()

        assert (
            impl._sparse_decode_requires_index_materialization(
                request,
                shared_cpu_enabled=True,
            )
            is True
        )

    def test_bind_rejects_skipped_index_for_shared_cpu_kv_both_sparse_decode(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=3,
            shared_cpu_materialize_index_on_decode_cold=True,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["latent-view"]],
            cached_chunk_ptrs_npu=[torch.tensor([1], dtype=torch.long)],
            metadata_warm=True,
            shared_latent_status="present",
            shared_index_status="skipped",
            shared_generation=3,
            pointer_cache_generation=3,
            shared_request_active=True,
            request_scope_token="req-1:3:3",
        )

        with pytest.raises(RuntimeError, match="invalid DSA index"):
            _bind_worker_state(impl, request)

    def test_record_shared_state_preserves_skipped_index_without_index_objs(self):
        impl = _make_impl()
        register = MagicMock()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            register_shared_cpu_sparse_request=register,
        )
        state = WorkerRetrieveState(shared_index_status="skipped")
        request = _make_request()
        state.cached_memory_objs = [["latent-view"]]
        state.cached_shared_handles = [["latent-handle"]]
        state.cached_chunk_ptrs_npu = ["latent-ptrs"]

        impl._record_shared_worker_retrieve_state(state, request)

        assert state.shared_latent_status == "present"
        assert state.shared_index_status == "skipped"
        assert state.pointer_cache_generation == 7
        register.assert_called_once_with(
            "req-1",
            owned_groups={0: state.cached_memory_objs},
        )

    def test_record_shared_state_uses_request_skipped_index_marker(self):
        impl = _make_impl()
        register = MagicMock()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            register_shared_cpu_sparse_request=register,
        )
        state = WorkerRetrieveState()
        request = _make_request()
        state.cached_memory_objs = [
            ["latent-view-layer0"],
            ["latent-view-layer1"],
        ]
        state.cached_shared_handles = [
            ["latent-handle-layer0"],
            ["latent-handle-layer1"],
        ]
        state.cached_chunk_ptrs_npu = [
            "latent-ptrs-layer0",
            "latent-ptrs-layer1",
        ]
        request.shared_index_skipped = True

        impl._record_shared_worker_retrieve_state(state, request)

        assert state.shared_latent_status == "present"
        assert state.shared_index_status == "skipped"
        assert state.shared_request_active is True
        assert state.pointer_cache_generation == 7
        register.assert_called_once()

    def test_record_shared_state_adopts_only_appended_chunks(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.num_layers = 1
        register = MagicMock()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            register_shared_cpu_sparse_request=register,
        )
        state = WorkerRetrieveState(
            cached_starts=[0, 256],
            cached_ends=[256, 512],
            cached_memory_objs=[["old", "new"]],
            cached_chunk_ptrs_npu=[torch.tensor([1, 2])],
            token_count=512,
            shared_index_status="skipped",
        )
        request = _make_request()
        request.token_ids = [0] * 512
        request.load_spec = LoadSpec(0, 512, True)

        impl._record_shared_worker_retrieve_state(
            state,
            request,
            previous_token_count=256,
        )

        register.assert_called_once_with(
            "req-1",
            owned_groups={0: state.cached_memory_objs},
            append_from={0: 1},
        )

    def test_record_shared_state_rejects_skipped_index_with_index_objs(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            shared_cpu_materialize_index_on_decode_cold=True,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState()
        request = _make_request()
        state.cached_memory_objs = [["latent-view"]]
        state.cached_shared_handles = [["latent-handle"]]
        state.cached_chunk_ptrs_npu = ["latent-ptrs"]
        state.cached_memory_objs_indexer = [["stale-index-view"]]
        state.cached_shared_handles_indexer = [["stale-index-handle"]]
        request.shared_index_skipped = True

        with pytest.raises(RuntimeError, match="kv_group=1"):
            impl._record_shared_worker_retrieve_state(state, request)

    def test_record_shared_state_rejects_missing_required_index_group(self):
        impl = _make_impl()
        impl.num_layers = 2
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            shared_cpu_materialize_index_on_decode_cold=True,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState()
        request = _make_request()
        state.cached_memory_objs = [
            ["latent-view-layer0"],
            ["latent-view-layer1"],
        ]
        state.cached_shared_handles = [
            ["latent-handle-layer0"],
            ["latent-handle-layer1"],
        ]
        state.cached_chunk_ptrs_npu = [
            "latent-ptrs-layer0",
            "latent-ptrs-layer1",
        ]
        request.shared_index_skipped = True

        with pytest.raises(RuntimeError, match="materialized DSA index"):
            impl._record_shared_worker_retrieve_state(state, request)

    def test_record_shared_state_rejects_partial_latent_group(self):
        impl = _make_impl()
        impl.num_layers = 2
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState()
        request = _make_request()
        state.cached_memory_objs = [["latent-view-layer0"], []]
        state.cached_shared_handles = [["latent-handle-layer0"], []]
        state.cached_chunk_ptrs_npu = ["latent-ptrs-layer0", None]

        with pytest.raises(RuntimeError, match="incomplete MLA latent"):
            impl._record_shared_worker_retrieve_state(state, request)

    def test_record_shared_state_rejects_tail_only_prefix_coverage(self):
        impl = _make_impl()
        impl.num_layers = 1
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState(token_count=512)
        request = _make_request()
        request.token_ids = [0] * 512
        request.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=512,
            can_load=True,
        )
        state.cached_starts = [256]
        state.cached_ends = [512]
        state.cached_memory_objs = [["latent-view"]]
        state.cached_shared_handles = [["latent-handle"]]
        state.cached_chunk_ptrs_npu = [torch.tensor([111], dtype=torch.long)]

        with pytest.raises(RuntimeError, match="non-contiguous prefix coverage"):
            impl._record_shared_worker_retrieve_state(state, request)

    def test_record_shared_state_rejects_partial_required_index_group(self):
        impl = _make_impl()
        impl.num_layers = 2
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            shared_cpu_materialize_index_on_decode_cold=True,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState()
        request = _make_request()
        state.cached_memory_objs = [
            ["latent-view-layer0"],
            ["latent-view-layer1"],
        ]
        state.cached_shared_handles = [
            ["latent-handle-layer0"],
            ["latent-handle-layer1"],
        ]
        state.cached_chunk_ptrs_npu = [
            "latent-ptrs-layer0",
            "latent-ptrs-layer1",
        ]
        state.cached_memory_objs_indexer = [["index-view-layer0"], []]
        state.cached_shared_handles_indexer = [["index-handle-layer0"], []]
        state.cached_chunk_ptrs_npu_indexer = ["index-ptrs-layer0", None]

        with pytest.raises(RuntimeError, match="complete materialized DSA index"):
            impl._record_shared_worker_retrieve_state(state, request)

    def test_record_shared_state_accepts_complete_required_index_group(self):
        impl = _make_impl()
        impl.num_layers = 2
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        register = MagicMock()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            shared_cpu_materialize_index_on_decode_cold=True,
            register_shared_cpu_sparse_request=register,
        )
        state = WorkerRetrieveState()
        request = _make_request()
        state.cached_memory_objs = [
            ["latent-view-layer0"],
            ["latent-view-layer1"],
        ]
        state.cached_shared_handles = [
            ["latent-handle-layer0"],
            ["latent-handle-layer1"],
        ]
        state.cached_chunk_ptrs_npu = [
            "latent-ptrs-layer0",
            "latent-ptrs-layer1",
        ]
        state.cached_memory_objs_indexer = [
            ["index-view-layer0"],
            ["index-view-layer1"],
        ]
        state.cached_shared_handles_indexer = [
            ["index-handle-layer0"],
            ["index-handle-layer1"],
        ]
        state.cached_chunk_ptrs_npu_indexer = [
            "index-ptrs-layer0",
            "index-ptrs-layer1",
        ]

        impl._record_shared_worker_retrieve_state(state, request)

        assert state.shared_latent_status == "present"
        assert state.shared_index_status == "present"
        assert state.shared_request_active is True
        assert state.token_count == request.load_spec.lmcache_cached_tokens
        assert state.request_scope_token == "req-1:7:3"
        register.assert_called_once_with(
            "req-1",
            owned_groups={
                0: state.cached_memory_objs,
                1: state.cached_memory_objs_indexer,
            },
        )

    def test_record_shared_state_rejects_short_latent_pointer_tensor(self):
        impl = _make_impl()
        impl.num_layers = 1
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState()
        request = _make_request()
        state.cached_starts = [0, 256]
        state.cached_ends = [256, 512]
        state.cached_memory_objs = [["latent-view-0", "latent-view-1"]]
        state.cached_shared_handles = [["latent-handle-0", "latent-handle-1"]]
        state.cached_chunk_ptrs_npu = [torch.tensor([111], dtype=torch.long)]

        with pytest.raises(RuntimeError, match="pointer-cache install"):
            impl._record_shared_worker_retrieve_state(state, request)

    def test_record_shared_state_rejects_indexer_shorter_than_latent_prefix(self):
        impl = _make_impl()
        impl.num_layers = 1
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            shared_cpu_materialize_index_on_decode_cold=True,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState()
        request = _make_request()
        state.cached_starts = [0, 256]
        state.cached_ends = [256, 512]
        state.cached_memory_objs = [["latent-view-0", "latent-view-1"]]
        state.cached_shared_handles = [["latent-handle-0", "latent-handle-1"]]
        state.cached_chunk_ptrs_npu = [torch.tensor([111, 222], dtype=torch.long)]
        state.cached_starts_indexer = [0]
        state.cached_ends_indexer = [256]
        state.cached_memory_objs_indexer = [["index-view-0"]]
        state.cached_shared_handles_indexer = [["index-handle-0"]]
        state.cached_chunk_ptrs_npu_indexer = [
            torch.tensor([333], dtype=torch.long)
        ]

        with pytest.raises(RuntimeError, match="materialized DSA index"):
            impl._record_shared_worker_retrieve_state(state, request)

    def test_shared_cpu_config_value_reads_engine_extra_config(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            config=SimpleNamespace(
                extra_config={
                    "shared_cpu_materialize_index_on_decode_cold": False,
                }
            )
        )

        assert impl._shared_cpu_materialize_index_on_decode_cold() is False

    def test_apply_extra_config_refreshes_vllm_capacity_hints(self):
        impl = _make_impl()
        config = SimpleNamespace(
            extra_config={
                "vllm_max_model_len": 1,
                "vllm_max_num_seqs": 2,
                "vllm_max_num_batched_tokens": 3,
            }
        )
        vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
            model_config=SimpleNamespace(max_model_len=20000),
            scheduler_config=SimpleNamespace(
                max_num_seqs=32,
                max_num_batched_tokens=4096,
            ),
        )

        impl._apply_extra_config(config, vllm_config)

        assert config.extra_config["vllm_max_model_len"] == 20000
        assert config.extra_config["vllm_max_num_seqs"] == 32
        assert config.extra_config["vllm_max_num_batched_tokens"] == 4096

    def test_release_shared_cpu_cache_state_invalidates_request_scope(self):
        state = WorkerRetrieveState(
            req_id="req-1",
            cached_memory_objs=[["view"]],
            cached_tensors=[["tensor"]],
            cached_shared_handles=[["handle"]],
            shared_latent_status="present",
            shared_index_status="skipped",
            indexer_npu_resident=True,
            indexer_npu_materialization_pending=True,
            shared_request_active=True,
            request_scope_token="req-1:4",
        )
        engine = SimpleNamespace(
            release_shared_cpu_sparse_request=MagicMock(),
        )

        LMCacheConnectorV1Impl._release_shared_worker_retrieve_state(
            state,
            engine,
        )

        engine.release_shared_cpu_sparse_request.assert_called_once_with("req-1")
        assert state.cached_shared_handles == []
        assert state.shared_index_status == "missing"
        assert state.indexer_npu_resident is False
        assert state.indexer_npu_materialization_pending is False
        assert state.shared_generation == 0
        assert state.pointer_cache_generation == 0
        assert state.shared_request_active is False
        assert state.request_scope_token is None
        assert state.cached_memory_objs == []
        assert state.cached_tensors == []

    def test_release_shared_cpu_cache_state_unregisters_engine_request(self):
        state = WorkerRetrieveState(
            req_id="req-1",
            shared_request_active=True,
            request_scope_token="req-1:4",
        )
        engine = SimpleNamespace(
            release_shared_cpu_sparse_request=MagicMock(),
        )

        LMCacheConnectorV1Impl._release_shared_worker_retrieve_state(
            state,
            engine,
        )

        engine.release_shared_cpu_sparse_request.assert_called_once_with("req-1")
        assert state.shared_request_active is False

    def test_finished_worker_request_releases_request_owned_cache_state(self):
        storer_closed: list[bool] = []

        def storer():
            try:
                yield
            finally:
                storer_closed.append(True)

        layerwise_storer = storer()
        next(layerwise_storer)
        engine = MagicMock()
        impl = _make_impl()
        impl._manager = SimpleNamespace(lmcache_engine=engine)
        storer_key = ("req-1", "normal_save", 0, 0, 0)
        impl._layerwise_save_storers = {storer_key: layerwise_storer}
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(
                req_id="req-1",
                cached_memory_objs=[["view"]],
                shared_request_active=True,
            )
        }

        impl._release_finished_worker_requests({"req-1"})

        assert impl._layerwise_save_storers == {}
        assert impl._worker_retrieve_state == {}
        assert storer_closed == [True]
        engine.release_shared_cpu_sparse_request.assert_called_once_with("req-1")
        engine.lookup_unpin.assert_called_once_with("req-1")

    def test_finished_worker_request_releases_engine_lease_without_state(self):
        engine = MagicMock()
        impl = _make_impl()
        impl._manager = SimpleNamespace(lmcache_engine=engine)

        impl._release_finished_worker_requests({"req-1"})

        engine.release_shared_cpu_sparse_request.assert_called_once_with("req-1")

    def test_get_finished_releases_worker_state_after_save(self):
        impl = _make_impl()
        impl._wait_for_save_done = True
        impl._release_finished_worker_requests = MagicMock()

        assert impl.get_finished(set()) == (None, None)
        impl._release_finished_worker_requests.assert_not_called()

        result = impl.get_finished({"req-1"})

        assert result == (None, None)
        impl._release_finished_worker_requests.assert_called_once_with({"req-1"})

    def test_get_finished_defers_cleanup_until_wait_for_save(self):
        request = SimpleNamespace(
            req_id="req-1",
            token_ids=[1, 2, 3],
            save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
        )
        metadata = LMCacheConnectorMetadata(requests=[request])
        impl = _make_impl()
        impl._parent = SimpleNamespace(
            _get_connector_metadata=lambda: metadata,
        )
        impl.kv_role = "kv_both"
        impl._wait_for_save_done = False
        impl._finished_req_ids_waiting_for_save = set()
        impl._release_finished_worker_requests = MagicMock()
        impl._wait_for_save_impl = MagicMock()

        assert impl.get_finished({"req-1"}) == (None, None)
        impl._release_finished_worker_requests.assert_not_called()
        assert impl._finished_req_ids_waiting_for_save == {"req-1"}

        impl.wait_for_save()

        impl._wait_for_save_impl.assert_called_once()
        (save_context,) = impl._wait_for_save_impl.call_args.args
        assert save_context == {}
        impl._release_finished_worker_requests.assert_called_once_with({"req-1"})
        assert impl._finished_req_ids_waiting_for_save == set()

    def test_wait_for_save_failure_marks_step_complete(self):
        impl = _make_impl()
        impl._parent = SimpleNamespace(
            _get_connector_metadata=lambda: LMCacheConnectorMetadata()
        )
        impl._wait_for_save_done = False
        impl._finished_req_ids_waiting_for_save = set()
        impl._wait_for_save_impl = MagicMock(
            side_effect=RuntimeError("save failed")
        )
        with pytest.raises(RuntimeError, match="save failed"):
            impl.wait_for_save()

        assert impl._wait_for_save_done is True

    def test_get_finished_without_pending_save_releases_immediately(self):
        metadata = LMCacheConnectorMetadata(requests=[])
        impl = _make_impl()
        impl._parent = SimpleNamespace(
            _get_connector_metadata=lambda: metadata,
        )
        impl._wait_for_save_done = False
        impl._finished_req_ids_waiting_for_save = set()
        impl._release_finished_worker_requests = MagicMock()

        assert impl.get_finished({"req-1"}) == (None, None)
        impl._release_finished_worker_requests.assert_called_once_with({"req-1"})

    def test_save_transfers_active_shared_state_without_releasing(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=False)
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["old-k"]],
            cached_starts=[0],
            cached_ends=[256],
            shared_latent_status="present",
            shared_generation=8,
            pointer_cache_generation=8,
            shared_request_active=True,
            request_scope_token="req-1:8:256",
        )
        request = _make_request()
        state = impl._worker_retrieve_state["req-1"]
        state.cached_keys = [["new-k"]]

        impl._publish_worker_retrieve_state(
            state,
            request,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
        )

        state = impl._worker_retrieve_state["req-1"]
        assert state.shared_request_active is True
        assert state.pointer_cache_generation == 8
        assert state.request_scope_token == "req-1:8:256"
        assert state.shared_validation_signature is None

    def test_publish_refreshes_existing_validation_signature(self):
        impl = _make_impl()
        impl.num_layers = 1
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = _make_shared_engine(rank0=False, generation=7)
        old_state = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["old-latent-view"]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            cached_shared_handles=[["old-handle"]],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=7,
            pointer_cache_generation=7,
            shared_request_active=True,
            request_scope_token="req-1:7:256",
        )
        request = _make_request()
        request.token_ids = [0] * 256
        request.load_spec.lmcache_cached_tokens = 256
        old_signature = impl._shared_worker_validation_signature(
            old_state,
            request,
            current_generation=7,
            pointer_generation=7,
            materialize_index=False,
        )
        old_state.shared_validation_signature = old_signature
        impl._worker_retrieve_state["req-1"] = old_state

        fresh = _make_request()
        fresh.token_ids = [0] * 256
        fresh.load_spec.lmcache_cached_tokens = 256
        fresh_state = old_state
        fresh_state.cached_keys = [["k0"]]
        fresh_state.cached_starts = [0]
        fresh_state.cached_ends = [256]
        fresh_state.cached_memory_objs = [["new-latent-view"]]
        fresh_state.cached_chunk_ptrs_npu = [
            torch.tensor([222], dtype=torch.long)
        ]
        fresh_state.cached_shared_handles = [["new-handle"]]

        impl._publish_worker_retrieve_state(
            fresh_state,
            fresh,
            location="local",
            metadata_warm=True,
            token_count=256,
        )

        new_state = impl._worker_retrieve_state["req-1"]
        assert new_state is old_state
        assert new_state.shared_validation_signature is not None
        assert new_state.shared_validation_signature != old_signature
        assert new_state.cached_memory_objs == [["new-latent-view"]]

    def test_save_registers_rank0_shared_backing_for_cleanup(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = True

            def is_valid(self):
                return True

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        impl = _make_impl()
        engine = _make_shared_engine(rank0=True, generation=9)
        impl.lmcache_engine = engine
        backing_obj = FakeMemObj()
        request = _make_request()
        state = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[[backing_obj]],
            cached_tensors=[["tensor"]],
            cached_chunk_ptrs_npu=["latent-ptrs"],
            cached_shared_handles=[["handle"]],
        )

        impl._publish_worker_retrieve_state(
            state,
            request,
            location="mixed",
            metadata_warm=True,
            token_count=256,
        )

        state = impl._worker_retrieve_state["req-1"]
        assert engine.shared_cpu_rank0_request_object_ids("req-1", 0) == {
            id(backing_obj)
        }
        assert state.cached_shared_handles == [["handle"]]
        assert state.shared_latent_status == "present"
        assert state.shared_generation == 9
        assert state.pointer_cache_generation == 9
        assert state.shared_request_active is True

        LMCacheConnectorV1Impl._release_shared_worker_retrieve_state(
            state,
            engine,
        )
        assert backing_obj.unpinned == 1
        assert backing_obj.released == 1
        assert state.cached_memory_objs == []

    def test_tp1_shared_state_cleanup_keeps_local_cpu_hot_cache_reference(self):
        class BorrowedLocalCPUObj:
            def __init__(self):
                self.ref_count = 1
                self.pin_count = 0
                self.valid = True

            @property
            def is_pinned(self):
                return self.pin_count > 0

            def is_valid(self):
                return self.valid

            def ref_count_up(self):
                self.ref_count += 1

            def ref_count_down(self):
                self.ref_count -= 1
                if self.ref_count == 0 and self.pin_count == 0:
                    self.valid = False

            def pin(self):
                self.pin_count += 1
                return True

            def unpin(self):
                self.pin_count -= 1
                if self.ref_count == 0 and self.pin_count == 0:
                    self.valid = False
                return True

            @property
            def tensor(self):
                return object() if self.valid else None

        impl = _make_impl()
        engine = _make_shared_engine(rank0=True, generation=9)
        impl.lmcache_engine = engine
        borrowed_obj = BorrowedLocalCPUObj()
        hot_cache = {"k": borrowed_obj}
        request, store_result = _make_store_request(
            impl,
            token_count=256,
            start=0,
            end=256,
            key="k",
            tensor="tensor",
        )
        store_result.memory_objs = [[borrowed_obj]]
        store_result.tensors = [[borrowed_obj.tensor]]

        impl._promote_layerwise_store_result(request, store_result)

        seeded_state = impl._worker_retrieve_state["req-1"]
        assert seeded_state.shared_request_active is False
        assert engine.shared_cpu_rank0_request_object_ids("req-1", 0) == {
            id(borrowed_obj)
        }
        assert borrowed_obj.ref_count == 2
        assert borrowed_obj.pin_count == 1

        impl._retain_shared_store_seed_state(seeded_state)
        assert borrowed_obj.ref_count == 2
        assert borrowed_obj.pin_count == 1

        request.is_sparse_decode = True
        request.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=256,
            can_load=True,
        )
        seeded_state.cached_chunk_ptrs_npu = ["latent-ptrs"]
        impl._publish_worker_retrieve_state(
            seeded_state,
            request,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
        )

        state = impl._worker_retrieve_state["req-1"]
        assert state.shared_request_active is True
        assert engine.shared_cpu_rank0_request_object_ids("req-1", 0) == {
            id(borrowed_obj)
        }
        assert borrowed_obj.ref_count == 2
        assert borrowed_obj.pin_count == 1

        impl._drop_worker_retrieve_state("req-1")

        assert borrowed_obj.ref_count == 1
        assert borrowed_obj.pin_count == 0
        assert borrowed_obj.valid is True
        assert hot_cache["k"].tensor is not None

        second_hit = hot_cache["k"]
        second_hit.ref_count_up()
        assert second_hit.ref_count == 2
        assert second_hit.tensor is not None
        second_hit.ref_count_down()
        assert second_hit.ref_count == 1
        assert second_hit.valid is True

    def test_store_seed_prepare_failure_rolls_back_request_ownership(self):
        class BorrowedObj:
            def __init__(self):
                self.ref_count = 1
                self.pin_count = 0

            @property
            def is_pinned(self):
                return self.pin_count > 0

            def ref_count_up(self):
                self.ref_count += 1

            def ref_count_down(self):
                self.ref_count -= 1

            def pin(self):
                self.pin_count += 1
                return True

            def unpin(self):
                self.pin_count -= 1
                return True

        impl = _make_impl()
        impl._latent_kvcaches = [object()]
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=9,
            store_location="LocalCPUBackend",
            metadata=SimpleNamespace(is_first_rank=lambda: True),
        )
        borrowed_obj = BorrowedObj()
        request, store_result = _make_store_request(
            impl,
            token_count=256,
            start=0,
            end=256,
            key="k",
            tensor="tensor",
        )
        store_result.memory_objs = [[borrowed_obj]]
        impl._refresh_prepared_sparse_sources = MagicMock(
            side_effect=RuntimeError("prepare failed")
        )

        with pytest.raises(RuntimeError, match="prepare failed"):
            impl._promote_layerwise_store_result(request, store_result)

        assert impl._worker_retrieve_state == {}
        assert borrowed_obj.ref_count == 1
        assert borrowed_obj.pin_count == 0

    def test_drop_then_publish_releases_passive_shared_views(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0

            def ref_count_down(self):
                self.released += 1

        impl = _make_impl()
        engine = _make_shared_engine(rank0=False, generation=10)
        impl.lmcache_engine = engine
        old_view = FakeMemObj()
        engine.register_shared_cpu_sparse_request(
            "req-1",
            owned_groups={0: [[old_view]]},
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            req_id="req-1",
            cached_keys=[["old-k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[[old_view]],
            shared_latent_status="present",
            shared_generation=10,
            pointer_cache_generation=10,
            shared_request_active=True,
            request_scope_token="req-1:10:256",
        )
        new_view = FakeMemObj()
        request = _make_request()
        state = WorkerRetrieveState(
            cached_keys=[["new-k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[[new_view]],
            cached_chunk_ptrs_npu=["latent-ptrs"],
            cached_shared_handles=[["new-handle"]],
        )

        impl._drop_worker_retrieve_state("req-1")
        impl._publish_worker_retrieve_state(
            state,
            request,
            location="mixed",
            metadata_warm=True,
            token_count=256,
        )

        state = impl._worker_retrieve_state["req-1"]
        assert old_view.released == 1
        assert new_view.released == 0
        assert engine._shared_cpu_request_leases["req-1"].object_ids(0) == {
            id(new_view)
        }
        assert state.shared_generation == 10
        impl._drop_worker_retrieve_state("req-1")

    def test_register_failure_releases_unstored_rank0_shared_objects(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = True

            def is_valid(self):
                return True

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        def fail_register(*_args, **_kwargs):
            raise RuntimeError("capacity registry failed")

        impl = _make_impl()
        engine = _make_shared_engine(rank0=True)
        engine.register_shared_cpu_sparse_request = fail_register
        impl.lmcache_engine = engine
        backing_obj = FakeMemObj()
        request = _make_request()
        state = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[[backing_obj]],
            cached_chunk_ptrs_npu=["latent-ptrs"],
            cached_shared_handles=[["handle"]],
        )

        with pytest.raises(RuntimeError, match="capacity registry failed"):
            impl._publish_worker_retrieve_state(
                state,
                request,
                location="mixed",
                metadata_warm=True,
                token_count=256,
            )

        assert impl._worker_retrieve_state == {}
        assert backing_obj.unpinned == 1
        assert backing_obj.released == 1

    def test_record_shared_state_rejects_missing_pointer_cache(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0

            def ref_count_down(self):
                self.released += 1

        impl = _make_impl()
        impl.lmcache_engine = _make_shared_engine(rank0=False)
        passive_view = FakeMemObj()
        request = _make_request()
        state = WorkerRetrieveState(
            cached_memory_objs=[[passive_view]],
            cached_shared_handles=[["handle"]],
        )

        with pytest.raises(RuntimeError, match="pointer-cache install"):
            impl._publish_worker_retrieve_state(
                state,
                request,
                location="mixed",
                metadata_warm=True,
                token_count=256,
            )
        assert impl._worker_retrieve_state == {}
        assert passive_view.released == 1

    def test_save_failure_releases_unstored_rank0_shared_objects(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = True

            def is_valid(self):
                return True

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        impl = _make_impl()
        impl.lmcache_engine = _make_shared_engine(rank0=True)
        backing_obj = FakeMemObj()
        request = _make_request()
        state = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[[backing_obj]],
            cached_shared_handles=[["handle"]],
        )

        with pytest.raises(RuntimeError, match="pointer-cache install"):
            impl._publish_worker_retrieve_state(
                state,
                request,
                location="mixed",
                metadata_warm=True,
                token_count=256,
            )

        assert "req-1" not in impl._worker_retrieve_state
        assert backing_obj.unpinned == 1
        assert backing_obj.released == 1

    def test_save_failure_does_not_mutate_old_shared_state(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0

            def ref_count_down(self):
                self.released += 1

        impl = _make_impl()
        engine = _make_shared_engine(rank0=False)
        impl.lmcache_engine = engine
        old_view = FakeMemObj()
        engine.register_shared_cpu_sparse_request(
            "req-1",
            owned_groups={0: [[old_view]]},
        )
        old_state = WorkerRetrieveState(
            req_id="req-1",
            cached_keys=[["old-k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[[old_view]],
            shared_latent_status="present",
            shared_generation=9,
            pointer_cache_generation=9,
            shared_request_active=True,
            request_scope_token="req-1:9:256",
        )
        impl._worker_retrieve_state["req-1"] = old_state
        new_view = FakeMemObj()
        request = _make_request()
        state = WorkerRetrieveState(
            cached_keys=[["new-k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[[new_view]],
            cached_shared_handles=[["new-handle"]],
        )

        with pytest.raises(RuntimeError, match="pointer-cache install"):
            impl._publish_worker_retrieve_state(
                state,
                request,
                location="mixed",
                metadata_warm=True,
                token_count=256,
            )

        assert impl._worker_retrieve_state["req-1"] is old_state
        assert engine._shared_cpu_request_leases["req-1"].object_ids(0) == {
            id(old_view)
        }
        assert old_state.pointer_cache_generation == 9
        assert old_view.released == 0
        assert new_view.released == 1
        engine.release_shared_cpu_sparse_request("req-1")

    def test_failed_in_place_refresh_drops_old_shared_state(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0

            def ref_count_down(self):
                self.released += 1

        impl = _make_impl()
        engine = _make_shared_engine(rank0=False)
        impl.lmcache_engine = engine
        old_view = FakeMemObj()
        new_view = FakeMemObj()
        engine.register_shared_cpu_sparse_request(
            "req-1",
            owned_groups={0: [[old_view]]},
        )
        state = WorkerRetrieveState(
            req_id="req-1",
            cached_keys=[["old-k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[[old_view]],
            cached_chunk_ptrs_npu=["old-ptrs"],
            shared_latent_status="present",
            shared_generation=9,
            pointer_cache_generation=9,
            shared_request_active=True,
            request_scope_token="req-1:9:256",
        )
        impl._worker_retrieve_state["req-1"] = state

        # Retrieval updates the bound state in place before publication.
        state.cached_keys = [["new-k"]]
        state.cached_memory_objs = [[new_view]]
        state.cached_chunk_ptrs_npu = []

        with pytest.raises(RuntimeError, match="pointer-cache install"):
            impl._publish_worker_retrieve_state(
                state,
                _make_request(),
                location="mixed",
                metadata_warm=True,
                token_count=256,
            )

        assert "req-1" not in impl._worker_retrieve_state
        assert "req-1" not in engine._shared_cpu_request_leases
        assert old_view.released == 1
        assert new_view.released == 1

    def test_dense_prefix_prime_interleaves_latent_and_indexer(self):
        order = []

        def _retriever(label):
            order.append(f"{label}0")
            yield None
            order.append(f"{label}1")
            yield None

        latent = _retriever("latent")
        index = _retriever("index")

        LMCacheConnectorV1Impl._prime_dense_prefix_retrievers(latent, index)

        assert order == ["latent0", "index0", "latent1", "index1"]

    def test_wait_for_layer_load_forwards_target_slot_row_by_request_id(self):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]

        captured = []

        def _retriever():
            payload = yield None
            captured.append(payload)
            yield torch.ones(4, dtype=torch.bool)

        retriever = _retriever()
        next(retriever)
        impl.layerwise_retrievers = [(retriever, None)]

        selected_tokens = torch.tensor(
            [[10, 11, 12, 13], [18831, 18814, 18810, 18651]],
            dtype=torch.int32,
        )
        target_slot_mapping = torch.tensor(
            [[100, 101, 102, 103], [900, 901, 902, 903]],
            dtype=torch.long,
        )

        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            token_start_index=[0, 0],
            request_ids=["other-req", "req-1"],
            target_slot_mapping=target_slot_mapping,
        )

        assert len(captured) == 1
        selected_payload, token_start, target_payload = captured[0]
        assert token_start is None
        assert torch.equal(selected_payload, selected_tokens[1])
        assert torch.equal(target_payload, target_slot_mapping[1])

    def test_wait_for_layer_load_reordered_rows_forward_payload_event(
        self, monkeypatch
    ):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]

        sentinel_event = object()
        event_calls = []

        def _record_event(*values):
            event_calls.append(values)
            return sentinel_event

        monkeypatch.setattr(
            adapter_mod,
            "_dsa_record_payload_event_if_needed",
            _record_event,
        )

        captured = []

        def _retriever():
            payload = yield None
            captured.append(payload)
            yield torch.ones(4, dtype=torch.bool)

        retriever = _retriever()
        next(retriever)
        impl.layerwise_retrievers = [(retriever, None)]

        selected_tokens = torch.tensor(
            [
                [10, 11, 12, 13],
                [100, 101, 102, 103],
                [18831, 18814, 18810, 18651],
            ],
            dtype=torch.int32,
        )
        target_slot_mapping = torch.tensor(
            [
                [500, 501, 502, 503],
                [700, 701, 702, 703],
                [900, 901, 902, 903],
            ],
            dtype=torch.long,
        )

        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            token_start_index=[0, 0, 0],
            request_ids=["req-1", "other-req", "req-1"],
            target_slot_mapping=target_slot_mapping,
        )

        assert len(event_calls) == 1
        assert len(captured) == 1
        payload = captured[0]
        assert payload["payload_event"] is sentinel_event
        assert torch.equal(
            payload["selected_token_ids"], selected_tokens[[0, 2]]
        )
        assert torch.equal(
            payload["target_slot_mapping"], target_slot_mapping[[0, 2]]
        )

    def test_wait_for_layer_load_contiguous_mtp_rows_use_view_payload(
        self, monkeypatch
    ):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]
        monkeypatch.setattr(
            adapter_mod,
            "_dsa_record_payload_event_if_needed",
            lambda *values: (_ for _ in ()).throw(
                AssertionError("contiguous MTP rows must not record an event")
            ),
        )

        captured = []

        def _retriever():
            payload = yield None
            captured.append(payload)
            yield torch.ones(4, dtype=torch.bool)

        retriever = _retriever()
        next(retriever)
        impl.layerwise_retrievers = [(retriever, None)]

        selected_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [10, 11, 12, 13],
                [20, 21, 22, 23],
            ],
            dtype=torch.int32,
        )
        target_slot_mapping = torch.tensor(
            [
                [100, 101, 102, 103],
                [900, 901, 902, 903],
                [904, 905, 906, 907],
            ],
            dtype=torch.long,
        )

        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            token_start_index=[0, 0, 0],
            request_ids=["other-req", "req-1", "req-1"],
            target_slot_mapping=target_slot_mapping,
        )

        selected_payload, token_start, target_payload = captured[0]
        assert token_start is None
        assert torch.equal(selected_payload, selected_tokens[1:3])
        assert torch.equal(target_payload, target_slot_mapping[1:3])
        assert selected_payload.data_ptr() == selected_tokens[1].data_ptr()
        assert target_payload.data_ptr() == target_slot_mapping[1].data_ptr()

    def test_wait_for_layer_load_ordered_sparse_rows_use_view_payload(
        self, monkeypatch
    ):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]
        monkeypatch.setattr(
            adapter_mod,
            "_dsa_record_payload_event_if_needed",
            lambda *values: (_ for _ in ()).throw(
                AssertionError("ordered sparse rows must not record payload events")
            ),
        )

        captured = []

        def _retriever():
            payload = yield None
            captured.append(payload)
            yield torch.ones(4, dtype=torch.bool)

        retriever = _retriever()
        next(retriever)
        impl.layerwise_retrievers = [(retriever, None)]

        selected_tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
        target_slot_mapping = torch.tensor([[900, 901, 902, 903]], dtype=torch.long)

        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            token_start_index=[0],
            request_ids=["req-1"],
            target_slot_mapping=target_slot_mapping,
        )

        selected_payload, token_start, target_payload = captured[0]
        assert token_start is None
        assert torch.equal(selected_payload, selected_tokens[0])
        assert torch.equal(target_payload, target_slot_mapping[0])
        assert selected_payload.data_ptr() == selected_tokens.data_ptr()
        assert target_payload.data_ptr() == target_slot_mapping.data_ptr()

    def test_request_union_payload_is_forwarded_as_one_sparse_row(self):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]
        captured = []

        def _retriever():
            payload = yield None
            captured.append(payload)
            yield torch.ones(3, dtype=torch.bool)

        retriever = _retriever()
        next(retriever)
        impl.layerwise_retrievers = [(retriever, None)]
        selected = torch.tensor([[5, 7, 9, 0]], dtype=torch.int32)
        targets = torch.tensor([[100, 101, 102, 0]], dtype=torch.long)
        counts = torch.tensor([3], dtype=torch.int32)

        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected,
            request_ids=["req-1"],
            target_slot_mapping=targets,
            selected_token_counts=counts,
        )

        assert len(captured) == 1
        assert captured[0]["selected_token_counts"].item() == 3
        assert captured[0]["selected_token_ids"].shape == (4,)

    def test_retrieve_stats_combine_mtp_rows_and_reset_each_window(
        self, monkeypatch
    ):
        monkeypatch.setenv(
            adapter_mod.RETRIEVE_STATS_INTERVAL_SECONDS_ENV,
            "10",
        )
        timestamps = iter((100.0, 105.0, 111.0, 112.0))
        monkeypatch.setattr(
            adapter_mod.time,
            "monotonic",
            lambda: next(timestamps),
        )
        log_records = []
        monkeypatch.setattr(
            adapter_mod.logger,
            "info",
            lambda message, *args: log_records.append((message, args)),
        )

        impl, _, _ = make_worker_connector([], use_layerwise=True)
        impl._record_sparse_retrieve_stats(
            torch.zeros((2, 4), dtype=torch.int32),
            torch.tensor([3, 4], dtype=torch.int32),
            row_count=2,
        )
        impl._record_sparse_retrieve_stats(
            torch.zeros(4, dtype=torch.int32),
            torch.tensor(5, dtype=torch.int32),
            row_count=1,
        )
        impl._record_sparse_retrieve_stats(
            torch.zeros((2, 4), dtype=torch.int32),
            torch.tensor([2, 6], dtype=torch.int32),
            row_count=2,
        )

        assert len(log_records) == 1
        message, args = log_records[0]
        assert message.startswith("[LMCacheRetrieveStats]")
        assert args[1:4] == (3, 5, 20)
        assert args[4] == pytest.approx(20 / 3)

        impl._record_sparse_retrieve_stats(
            torch.zeros(4, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
            row_count=1,
        )
        assert len(log_records) == 1
        assert impl._retrieve_stats_request_count == 1
        assert impl._retrieve_stats_row_count == 1
        assert impl._retrieve_stats_token_count == 4

    def test_retrieve_stats_default_off_does_not_read_counts(self, monkeypatch):
        monkeypatch.delenv(
            adapter_mod.RETRIEVE_STATS_INTERVAL_SECONDS_ENV,
            raising=False,
        )
        impl, _, _ = make_worker_connector([], use_layerwise=True)
        impl._record_sparse_retrieve_stats(
            selected_tokens=None,
            selected_token_counts=object(),
            row_count=2,
        )
        assert impl._retrieve_stats_interval_seconds == 0
        assert impl._retrieve_stats_request_count == 0

    def test_wait_for_layer_load_passes_all_request_rows_to_retrieve_stats(self):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]
        stats_inputs = []
        impl._record_sparse_retrieve_stats = (
            lambda selected, counts, rows: stats_inputs.append(
                (selected, counts, rows)
            )
        )

        def _retriever():
            yield None
            yield torch.ones(4, dtype=torch.bool)

        retriever = _retriever()
        next(retriever)
        impl.layerwise_retrievers = [(retriever, None)]
        selected = torch.tensor(
            [[1, 2, 0, 0], [3, 4, 5, 0]],
            dtype=torch.int32,
        )
        targets = torch.tensor(
            [[100, 101, 0, 0], [102, 103, 104, 0]],
            dtype=torch.long,
        )
        counts = torch.tensor([2, 3], dtype=torch.int32)

        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected,
            request_ids=["req-1", "req-1"],
            target_slot_mapping=targets,
            selected_token_counts=counts,
        )

        assert len(stats_inputs) == 1
        selected_per_request, counts_per_request, row_count = stats_inputs[0]
        assert torch.equal(selected_per_request, selected)
        assert torch.equal(counts_per_request, counts)
        assert row_count == 2

    def test_wait_for_layer_load_routes_exact_batch_rows_per_request(self):
        requests = [
            make_sparse_req_meta(req_id, token_count=4) for req_id in ("req-0", "req-1")
        ]
        impl, _, _ = make_worker_connector(requests, use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True, True]
        impl._layerwise_sparse_req_ids = [request.req_id for request in requests]
        captured = []
        events = []

        @contextmanager
        def _defer_consumer_wait():
            events.append(("enter", impl.current_layer))
            try:
                yield
            finally:
                events.append(("exit", impl.current_layer))

        impl.lmcache_engine.gpu_connector = SimpleNamespace(
            defer_sparse_load_consumer_wait=_defer_consumer_wait
        )

        def _retriever():
            payload = yield None
            captured.append(payload)
            events.append(("submit", impl._layerwise_sparse_req_ids[len(captured) - 1]))
            yield torch.ones(4, dtype=torch.bool)

        retrievers = [_retriever(), _retriever()]
        for retriever in retrievers:
            next(retriever)
        impl.layerwise_retrievers = [(retriever, None) for retriever in retrievers]

        selected_tokens = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int32,
        )
        target_slot_mapping = torch.tensor(
            [[100, 101, 102, 103], [200, 201, 202, 203]],
            dtype=torch.long,
        )
        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            request_ids=["req-1", "req-0"],
            target_slot_mapping=target_slot_mapping,
        )

        assert len(captured) == 2
        for row, (
            selected_payload,
            token_start,
            target_payload,
        ) in zip((1, 0), captured, strict=True):
            assert token_start is None
            assert torch.equal(selected_payload, selected_tokens[row])
            assert torch.equal(target_payload, target_slot_mapping[row])
        assert impl.current_layer == 1
        assert events == [
            ("enter", 0),
            ("submit", "req-0"),
            ("submit", "req-1"),
            ("exit", 0),
        ]

    def test_sparse_decode_attn_wait_drives_latent_and_indexer(self):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = True
        impl._indexer_layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]
        impl._layerwise_sparse_req_ids = ["req-1"]

        captured = []

        def _retriever(label):
            payload = yield None
            captured.append((label, payload))
            yield torch.ones(4, dtype=torch.bool)

        latent = _retriever("latent")
        indexer = _retriever("indexer")
        next(latent)
        next(indexer)
        impl.layerwise_retrievers = [(latent, indexer)]

        selected_tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
        target_slot_mapping = torch.tensor([[900, 901, 902, 903]], dtype=torch.long)
        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            token_start_index=[0],
            request_ids=["req-1"],
            target_slot_mapping=target_slot_mapping,
        )

        assert [label for label, _ in captured] == ["latent", "indexer"]
        selected_payload, token_start, target_payload = captured[0][1]
        assert token_start is None
        assert torch.equal(selected_payload, selected_tokens[0])
        assert torch.equal(target_payload, target_slot_mapping[0])
        assert captured[1][1] == (None, 0)
        assert impl.current_layer == 1

        impl.wait_for_layer_load("model.layers.0.self_attn.indexer.k_cache")

        assert [label for label, _ in captured] == ["latent", "indexer"]
        assert impl.current_layer == 1

    def test_sparse_decode_indexer_wait_can_prime_before_attn(self):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = True
        impl._indexer_layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]
        impl._layerwise_sparse_req_ids = ["req-1"]
        impl._layerwise_sparse_indexer_sent_layers = set()

        captured = []

        def _retriever(label):
            payload = yield None
            captured.append((label, payload))
            yield torch.ones(4, dtype=torch.bool)

        latent = _retriever("latent")
        indexer = _retriever("indexer")
        next(latent)
        next(indexer)
        impl.layerwise_retrievers = [(latent, indexer)]

        impl.wait_for_layer_load("model.layers.0.self_attn.indexer.k_cache")

        assert captured == [("indexer", (None, 0))]
        assert impl.current_layer == 0

        selected_tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
        target_slot_mapping = torch.tensor([[900, 901, 902, 903]], dtype=torch.long)
        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            token_start_index=[0],
            request_ids=["req-1"],
            target_slot_mapping=target_slot_mapping,
        )

        assert [label for label, _ in captured] == ["indexer", "latent"]
        selected_payload, token_start, target_payload = captured[1][1]
        assert token_start is None
        assert torch.equal(selected_payload, selected_tokens[0])
        assert torch.equal(target_payload, target_slot_mapping[0])
        assert impl.current_layer == 1

    def test_sparse_decode_resident_indexer_wait_is_noop(self):
        req = make_sparse_req_meta("req-1", token_count=4)
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = True
        impl._indexer_layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_retriever_is_sparse = [True]
        impl._layerwise_sparse_req_ids = ["req-1"]
        impl._layerwise_sparse_indexer_sent_layers = set()

        captured = []

        def _retriever():
            payload = yield None
            captured.append(payload)
            yield torch.ones(4, dtype=torch.bool)

        latent = _retriever()
        next(latent)
        impl.layerwise_retrievers = [(latent, None)]

        impl.wait_for_layer_load("model.layers.0.self_attn.indexer.k_cache")

        assert captured == []
        assert impl.current_layer == 0

        selected_tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
        target_slot_mapping = torch.tensor([[900, 901, 902, 903]], dtype=torch.long)
        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            token_start_index=[0],
            request_ids=["req-1"],
            target_slot_mapping=target_slot_mapping,
        )

        selected_payload, token_start, target_payload = captured[0]
        assert token_start is None
        assert torch.equal(selected_payload, selected_tokens[0])
        assert torch.equal(target_payload, target_slot_mapping[0])
        assert impl.current_layer == 1

    def test_dense_prefix_two_group_wait_advances_after_both_groups(self):
        req = ReqMeta(
            req_id="req-1",
            token_ids=[1, 2, 3, 4],
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=4,
                can_load=True,
            ),
            is_sparse_decode=False,
        )
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = True
        impl._indexer_layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
        impl.current_layer = 0
        impl.num_layers = 2
        impl._layerwise_requests = [req]
        impl._layerwise_retriever_is_sparse = [False]

        captured = []

        def _retriever(label):
            while True:
                captured.append(label)
                yield torch.ones(4, dtype=torch.bool)

        impl.layerwise_retrievers = [
            (_retriever("latent"), _retriever("indexer"))
        ]

        impl.wait_for_layer_load("model.layers.0.self_attn.attn")

        assert captured == ["latent"]
        assert impl.current_layer == 0

        impl.wait_for_layer_load("model.layers.0.self_attn.indexer.k_cache")

        assert captured == ["latent", "indexer"]
        assert impl.current_layer == 1

    def test_bind_keeps_scheduler_metadata_payload_free(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=False)
        request = _make_request()
        assert not hasattr(request, "cached_keys")

        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            location="local",
            token_count=256,
        )

        bound = _bind_worker_state(impl, request)
        assert bound is impl._worker_retrieve_state["req-1"]
        assert bound.cached_keys == [["layer0-key"]]
        assert bound.cached_starts == [0]
        assert bound.cached_ends == [256]
        assert not hasattr(request, "cached_keys")

    def test_bind_rejects_stale_shared_generation(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=2,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            shared_latent_status="present",
            shared_generation=1,
            shared_request_active=True,
        )

        with pytest.raises(RuntimeError, match="generation mismatch"):
            _bind_worker_state(impl, request)

    def test_bind_rejects_stale_pointer_cache_generation(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=2,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            shared_latent_status="present",
            shared_generation=2,
            pointer_cache_generation=1,
            shared_request_active=True,
        )

        with pytest.raises(RuntimeError, match="pointer-cache generation"):
            _bind_worker_state(impl, request)

    def test_bind_rejects_missing_shared_pointer_cache(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=2,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["latent-view"]],
            metadata_warm=True,
            shared_latent_status="present",
            shared_generation=2,
            pointer_cache_generation=2,
            shared_request_active=True,
            request_scope_token="req-1:2:3",
        )

        with pytest.raises(RuntimeError, match="missing MLA latent"):
            _bind_worker_state(impl, request)

    def test_shared_pointer_cache_entry_requires_actual_coverage(self):
        covers = LMCacheConnectorV1Impl._shared_pointer_cache_entry_covers

        assert not covers(None, 1)
        assert not covers([], 1)
        assert not covers(torch.empty(0, dtype=torch.long), 1)
        assert covers(torch.tensor([123], dtype=torch.long), 1)
        assert not covers(torch.tensor([123], dtype=torch.long), 2)
        assert covers(torch.tensor([123, 456], dtype=torch.long), 2)
        assert covers("fake-ptr", 1)

    def test_bind_rejects_shared_request_scope_mismatch(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=2,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["latent-view"]],
            cached_chunk_ptrs_npu=["latent-ptrs"],
            metadata_warm=True,
            shared_latent_status="present",
            shared_generation=2,
            pointer_cache_generation=2,
            shared_request_active=True,
            request_scope_token="req-1:2:2",
        )

        with pytest.raises(RuntimeError, match="request scope mismatch"):
            _bind_worker_state(impl, request)

    def test_bind_shared_scope_uses_lmcache_hit_tokens(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=2,
        )
        request = _make_request()
        request.token_ids = [1, 2, 3, 4]
        request.load_spec.lmcache_cached_tokens = 3
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["latent-view"]],
            cached_chunk_ptrs_npu=["latent-ptrs"],
            metadata_warm=True,
            shared_latent_status="present",
            shared_generation=2,
            pointer_cache_generation=2,
            shared_request_active=True,
            request_scope_token="req-1:2:3",
        )

        assert _bind_worker_state(impl, request) is not None

    def test_bind_shared_hot_state_reuses_validation_signature(self, monkeypatch):
        impl = _make_impl()
        impl.num_layers = 2
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=2,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"], ["layer1-key"]],
            cached_starts=[0],
            cached_ends=[3],
            cached_memory_objs=[["latent-view-layer0"], ["latent-view-layer1"]],
            cached_chunk_ptrs_npu=[
                torch.tensor([111], dtype=torch.long),
                torch.tensor([222], dtype=torch.long),
            ],
            metadata_warm=True,
            shared_latent_status="present",
            shared_generation=2,
            pointer_cache_generation=2,
            shared_request_active=True,
            request_scope_token="req-1:2:3",
        )

        calls = []
        prefix_checks = []
        original = LMCacheConnectorV1Impl._missing_shared_layer_cache_coverage
        original_prefix = LMCacheConnectorV1Impl._cached_ranges_cover_prefix

        def counting_coverage(layers, expected_layers, required_chunks):
            calls.append((expected_layers, required_chunks))
            return original(layers, expected_layers, required_chunks)

        def counting_prefix(starts, ends, token_count):
            prefix_checks.append((list(starts), list(ends), token_count))
            return original_prefix(starts, ends, token_count)

        monkeypatch.setattr(
            LMCacheConnectorV1Impl,
            "_missing_shared_layer_cache_coverage",
            staticmethod(counting_coverage),
        )
        monkeypatch.setattr(
            LMCacheConnectorV1Impl,
            "_cached_ranges_cover_prefix",
            classmethod(lambda cls, starts, ends, token_count: counting_prefix(
                starts,
                ends,
                token_count,
            )),
        )

        assert _bind_worker_state(impl, request) is not None
        assert calls == [(2, 1)]
        assert prefix_checks == [([0], [3], 3)]

        calls.clear()
        prefix_checks.clear()
        assert _bind_worker_state(impl, request) is not None
        assert calls == []
        assert prefix_checks == []

        state = impl._worker_retrieve_state["req-1"]
        state.cached_memory_objs = [list(layer) for layer in state.cached_memory_objs]
        calls.clear()
        prefix_checks.clear()
        assert _bind_worker_state(impl, request) is not None
        assert calls == [(2, 1)]
        assert prefix_checks == [([0], [3], 3)]

    def test_bind_rejects_incomplete_shared_latent_state(self):
        impl = _make_impl()
        impl.num_layers = 2
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=2,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["latent-view-layer0"], []],
            cached_chunk_ptrs_npu=["latent-ptrs-layer0", None],
            metadata_warm=True,
            shared_latent_status="present",
            shared_generation=2,
            pointer_cache_generation=2,
            shared_request_active=True,
            request_scope_token="req-1:2:3",
        )

        with pytest.raises(RuntimeError, match="incomplete MLA latent"):
            _bind_worker_state(impl, request)

    def test_bind_rejects_missing_strict_shared_index(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=3,
            shared_cpu_materialize_index_on_decode_cold=True,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            shared_latent_status="present",
            shared_index_status="missing",
            shared_generation=3,
            shared_request_active=True,
            request_scope_token="req-1:3:3",
        )

        with pytest.raises(RuntimeError, match="invalid DSA index"):
            _bind_worker_state(impl, request)

    def test_bind_warm_shared_state_does_not_walk_index_metadata(self):
        impl = _make_impl()
        impl.num_layers = 2
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=3,
            shared_cpu_materialize_index_on_decode_cold=True,
        )
        request = _make_request()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"], ["layer1-key"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[
                ["latent-view-layer0"],
                ["latent-view-layer1"],
            ],
            cached_chunk_ptrs_npu=[
                "latent-ptrs-layer0",
                "latent-ptrs-layer1",
            ],
            cached_memory_objs_indexer=[["index-view-layer0"], []],
            cached_chunk_ptrs_npu_indexer=["index-ptrs-layer0", None],
            metadata_warm=True,
            shared_latent_status="present",
            shared_index_status="present",
            shared_generation=3,
            pointer_cache_generation=3,
            shared_request_active=True,
            request_scope_token="req-1:3:3",
        )

        assert _bind_worker_state(impl, request) is not None

    def test_save_then_bind_round_trip(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=False)
        request = _make_request()
        state = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_shared_handles=[["handle"]],
        )

        impl._publish_worker_retrieve_state(
            state,
            request,
            location="local",
            metadata_warm=True,
            token_count=256,
        )

        fresh = _make_request()
        assert not hasattr(fresh, "cached_keys")
        bound = _bind_worker_state(impl, fresh)
        assert bound is state
        assert bound.cached_keys == [["k"]]
        assert bound.cached_shared_handles == [["handle"]]
        assert not hasattr(fresh, "cached_keys")

    def test_finalize_sparse_state_uses_lmcache_hit_token_count(self):
        impl = _make_impl()
        captured = {}

        def capture_save(
            state, request, *, location, metadata_warm, token_count
        ):
            captured["state"] = state
            captured["request"] = request
            captured["location"] = location
            captured["metadata_warm"] = metadata_warm
            captured["token_count"] = token_count

        impl._publish_worker_retrieve_state = capture_save
        request = _make_request()
        request.token_ids = [1, 2, 3, 4]
        request.load_spec.lmcache_cached_tokens = 3
        state = WorkerRetrieveState(cached_keys=[["k"]], metadata_warm=True)
        impl._worker_retrieve_state[request.req_id] = state

        impl._finalize_worker_retrieve_state_from_metadata(
            SimpleNamespace(requests=[request])
        )

        assert captured["state"] is state
        assert captured["request"] is request
        assert captured["token_count"] == 3

    def test_scheduler_metadata_excludes_worker_cache_payload(self):
        tracker = RequestTracker(
            req_id="req-1",
            prompt_len=512,
            token_ids=list(range(512)),
            allocated_block_ids=list(range(4)),
            allocated_block_ids_indexer=list(range(4)),
        )
        cache_fields = {
            "cached_keys",
            "cached_starts",
            "cached_ends",
            "cached_memory_objs",
            "cached_tensors",
            "cached_chunk_dev_ptrs",
            "cached_chunk_ptrs_npu",
            "cached_shared_handles",
            "cached_keys_indexer",
            "cached_starts_indexer",
            "cached_ends_indexer",
            "cached_memory_objs_indexer",
            "cached_tensors_indexer",
            "cached_chunk_dev_ptrs_indexer",
            "cached_chunk_ptrs_npu_indexer",
            "cached_shared_handles_indexer",
        }

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=128,
            lmcache_chunk_size=256,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=256,
                can_load=True,
            ),
            dsa_two_groups=True,
        )

        assert req_meta is not None
        assert cache_fields.isdisjoint(vars(tracker))
        assert cache_fields.isdisjoint(vars(req_meta))

    def test_sparse_reqmeta_does_not_allocate_full_true_decode_mask(self):
        tracker = RequestTracker(
            req_id="req-1",
            prompt_len=512,
            token_ids=list(range(512)),
            allocated_block_ids=list(range(4)),
        )
        tracker.is_decode_phase = True

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=128,
            lmcache_chunk_size=256,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=512,
                can_load=True,
            ),
            is_sparse_decode=True,
        )

        assert req_meta is not None
        assert req_meta.decode_token_mask is None
        assert tracker.sparse_decode_token_mask is None
        assert req_meta.decode_ret_mask is not None
        assert req_meta.decode_ret_mask.numel() == 512

    def test_sparse_decode_indexer_reuses_request_ret_mask(self):
        req = make_sparse_req_meta("req-1", token_count=512)
        req.decode_ret_mask = torch.zeros(512, dtype=torch.bool)
        req.indexer_slot_mapping = [torch.arange(512, dtype=torch.long)]
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = True
        impl.num_layers = 1
        impl.kv_caches = {
            "model.layers.0.self_attn.attn.k_cache": torch.zeros(1),
            "model.layers.0.self_attn.indexer.k_cache": torch.zeros(1),
        }
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_requests = []
        impl._layerwise_retriever_is_sparse = []
        impl._layerwise_sparse_req_ids = []
        impl._stats_monitor = SimpleNamespace(
            update_interval_vllm_hit_tokens=lambda *_args: None,
            update_interval_prompt_tokens=lambda *_args: None,
        )

        captured_kwargs = []

        class _FakeSharedEngine:
            enable_shared_cpu_cache = True
            shared_cpu_cache_generation = 1
            config = SimpleNamespace(
                extra_config={"shared_cpu_materialize_index_on_decode_cold": True}
            )

            def retrieve_layer_head_token_wise(self, tokens, mask, **kwargs):
                captured_kwargs.append(kwargs)

                def _retriever():
                    yield None
                    yield torch.ones(len(tokens), dtype=torch.bool)

                return _retriever()

        impl.lmcache_engine = _FakeSharedEngine()

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        assert len(captured_kwargs) == 2
        assert captured_kwargs[0]["kv_group"] == 0
        assert captured_kwargs[1]["kv_group"] == 1
        assert captured_kwargs[0]["ret_mask"] is req.decode_ret_mask
        assert captured_kwargs[1]["ret_mask"] is req.decode_ret_mask

    def test_sparse_decode_frontier_growth_refreshes_indexer_metadata_only(self):
        req = make_sparse_req_meta("req-1", token_count=512)
        req.indexer_slot_mapping = [torch.arange(512, dtype=torch.long)]
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = True
        impl.num_layers = 1
        impl.kv_caches = {
            "model.layers.0.self_attn.attn.k_cache": torch.zeros(1),
            "model.layers.0.self_attn.indexer.k_cache": torch.zeros(1),
        }
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_requests = []
        impl._layerwise_retriever_is_sparse = []
        impl._layerwise_sparse_req_ids = []
        impl._stats_monitor = SimpleNamespace(
            update_interval_vllm_hit_tokens=lambda *_args: None,
            update_interval_prompt_tokens=lambda *_args: None,
        )
        state = WorkerRetrieveState(
            req_id=req.req_id,
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[[torch.zeros(256)]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            cached_keys_indexer=[["ik0"]],
            cached_starts_indexer=[0],
            cached_ends_indexer=[256],
            cached_memory_objs_indexer=[["im0"]],
            cached_tensors_indexer=[[torch.zeros(256)]],
            cached_chunk_ptrs_npu_indexer=[
                torch.tensor([222], dtype=torch.long)
            ],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_index_status="present",
            indexer_npu_resident=True,
            shared_generation=1,
            pointer_cache_generation=1,
            shared_request_active=True,
            request_scope_token="req-1:1:256",
            prepared_sparse_sources={
                0: PreparedSparseSource(layers=(), total_tokens=256),
                1: PreparedSparseSource(layers=(), total_tokens=256),
            },
        )
        impl._worker_retrieve_state[req.req_id] = state
        captured_kwargs = []

        class _FakeSharedEngine:
            enable_shared_cpu_cache = True
            shared_cpu_cache_generation = 1
            config = SimpleNamespace(
                extra_config={"shared_cpu_materialize_index_on_decode_cold": True}
            )

            def retrieve_layer_head_token_wise(self, tokens, mask, **kwargs):
                captured_kwargs.append(kwargs)

                def _retriever():
                    yield kwargs.get("ret_mask")
                    while True:
                        yield kwargs.get("ret_mask")

                return _retriever()

        impl.lmcache_engine = _FakeSharedEngine()

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        assert len(captured_kwargs) == 2
        assert captured_kwargs[0]["kv_group"] == 0
        assert "sparse_metadata_only" not in captured_kwargs[0]
        assert captured_kwargs[1]["kv_group"] == 1
        assert captured_kwargs[1]["sparse_metadata_only"] is True
        assert "prepared_sparse_source" not in captured_kwargs[1]
        assert state.indexer_npu_resident is True
        impl._drain_layerwise_retrievers()

    def test_sparse_decode_start_uses_minimal_prepared_kwargs(self):
        req = make_sparse_req_meta("req-1", token_count=256)
        state = WorkerRetrieveState(
            cached_keys=[["layer-key"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_tensors=[[torch.zeros(256)]],
            cached_chunk_ptrs_npu=[torch.tensor([123], dtype=torch.long)],
        )
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = False
        impl.num_layers = 1
        impl.kv_caches = {
            "model.layers.0.self_attn.attn.k_cache": torch.zeros(1),
        }
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_requests = []
        impl._layerwise_retriever_is_sparse = []
        impl._layerwise_sparse_req_ids = []
        impl._stats_monitor = SimpleNamespace(
            update_interval_vllm_hit_tokens=lambda *_args: None,
            update_interval_prompt_tokens=lambda *_args: None,
        )
        captured_kwargs = []

        class _FakeEngine:
            enable_shared_cpu_cache = False

            def retrieve_layer_head_token_wise(self, tokens, mask, **kwargs):
                captured_kwargs.append(kwargs)

                def _retriever():
                    yield kwargs.get("ret_mask")
                    while True:
                        yield kwargs.get("ret_mask")

                return _retriever()

        impl.lmcache_engine = _FakeEngine()
        impl._publish_worker_retrieve_state(
            state,
            req,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
        )

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        assert len(captured_kwargs) == 1
        kwargs = captured_kwargs[0]
        assert set(kwargs) <= {
            "kvcaches",
            "slot_mapping",
            "sync",
            "kv_group",
            "prepared_sparse_source",
            "ret_mask",
            "req_id",
            "lmcache_cached_tokens",
        }
        prepared_source = kwargs["prepared_sparse_source"]
        assert prepared_source.total_tokens == 256
        assert prepared_source.chunk_token_counts == (256,)
        assert prepared_source.pointer_device == torch.device("cpu")
        assert not hasattr(req, "cached_keys")
        assert impl._worker_retrieve_state[req.req_id] is state
        impl._drain_layerwise_retrievers()

    def test_sparse_warm_ref_reuses_worker_metadata(self):
        req = make_sparse_req_meta("req-1", token_count=256)
        req.load_spec.vllm_cached_tokens = 128
        req.token_ids = []
        req.slot_mapping = []
        req.sparse_warm_ref = True
        ret_mask = torch.zeros(256, dtype=torch.bool)
        state = WorkerRetrieveState(
            cached_keys=[["layer-key"]],
            cached_starts=[0],
            cached_ends=[512],
            cached_tensors=[[torch.zeros(512)]],
            cached_chunk_ptrs_npu=[torch.tensor([123], dtype=torch.long)],
            slot_mapping=torch.arange(256, dtype=torch.long),
            decode_ret_mask=ret_mask,
        )
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = False
        impl.num_layers = 1
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_requests = []
        impl._layerwise_retriever_is_sparse = []
        impl._layerwise_sparse_req_ids = []
        impl._stats_monitor = SimpleNamespace(
            update_interval_vllm_hit_tokens=lambda *_args: None,
            update_interval_prompt_tokens=lambda *_args: None,
        )
        calls = []

        class _FakeEngine:
            enable_shared_cpu_cache = False

            def retrieve_layer_head_token_wise(self, tokens, mask, **kwargs):
                calls.append((tokens, mask, kwargs))

                def _retriever():
                    yield kwargs.get("ret_mask")
                    while True:
                        yield kwargs.get("ret_mask")

                return _retriever()

        impl.lmcache_engine = _FakeEngine()
        impl._publish_worker_retrieve_state(
            state,
            req,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=512,
        )

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        tokens, mask, kwargs = calls[0]
        assert tokens == []
        assert mask is None
        assert kwargs["slot_mapping"] is state.slot_mapping
        assert kwargs["ret_mask"] is ret_mask
        assert kwargs["prepared_sparse_source"].total_tokens == 512

        impl.wait_for_layer_load("model.layers.0.self_attn.attn")

        assert state.token_count == 512
        assert state.prepared_sparse_sources[0].total_tokens == 512

    def test_sparse_warm_ref_accepts_shared_prepared_superset(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
        )
        request = make_sparse_req_meta("req-1", token_count=256)
        request.token_ids = []
        request.slot_mapping = []
        request.sparse_warm_ref = True
        source = PreparedSparseSource(layers=(), total_tokens=512)
        state = WorkerRetrieveState(
            req_id="req-1",
            cached_keys=[["k0"]],
            metadata_warm=True,
            token_count=512,
            slot_mapping=torch.arange(256, dtype=torch.long),
            prepared_sparse_sources={0: source},
            shared_latent_status="present",
            shared_index_status="skipped",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:512",
        )
        impl._worker_retrieve_state["req-1"] = state

        assert impl._worker_retrieve_state_for_warm_ref(request) is state

    def test_sparse_warm_ref_requires_matching_worker_state(self):
        req = make_sparse_req_meta("req-1", token_count=256)
        req.token_ids = []
        req.slot_mapping = []
        req.sparse_warm_ref = True
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=False)
        impl.config.dsa_two_groups = False
        impl.num_layers = 1
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_requests = []
        impl._layerwise_retriever_is_sparse = []
        impl._layerwise_sparse_req_ids = []
        impl._stats_monitor = SimpleNamespace(
            update_interval_vllm_hit_tokens=lambda *_args: None,
            update_interval_prompt_tokens=lambda *_args: None,
        )

        with pytest.raises(RuntimeError, match="no matching prepared worker state"):
            impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

    def test_sparse_decode_defers_growing_state_until_layers_are_loaded(self):
        req = make_sparse_req_meta("req-1", token_count=512)
        state = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_tensors=[[torch.zeros(256)]],
            cached_chunk_ptrs_npu=[torch.tensor([123], dtype=torch.long)],
        )
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = False
        impl.num_layers = 1
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_requests = []
        impl._layerwise_retriever_is_sparse = []
        impl._layerwise_sparse_req_ids = []
        impl._stats_monitor = SimpleNamespace(
            update_interval_vllm_hit_tokens=lambda *_args: None,
            update_interval_prompt_tokens=lambda *_args: None,
        )
        impl._publish_worker_retrieve_state(
            state,
            req,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
        )

        class _FakeEngine:
            enable_shared_cpu_cache = False

            def retrieve_layer_head_token_wise(self, tokens, mask, **kwargs):
                def _retriever():
                    kwargs["cached_keys"][0].append("k1")
                    kwargs["cached_starts"].append(256)
                    kwargs["cached_ends"].append(512)
                    yield kwargs.get("ret_mask")
                    kwargs["cached_tensors"][0].append(torch.zeros(256))
                    kwargs["cached_chunk_ptrs_npu"][0] = torch.tensor(
                        [123, 456], dtype=torch.long
                    )
                    yield kwargs.get("ret_mask")

                return _retriever()

        impl.lmcache_engine = _FakeEngine()

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        assert state.token_count == 256
        assert state.prepared_sparse_sources[0].total_tokens == 256

        impl.wait_for_layer_load("model.layers.0.self_attn.attn")

        assert state.token_count == 512
        assert state.prepared_sparse_sources[0].total_tokens == 512

    def test_start_load_kv_aggregates_step_setup(self):
        sparse = make_sparse_req_meta("sparse", token_count=4)
        skipped = make_sparse_req_meta("skipped", can_load=False, token_count=4)
        dense = make_sparse_req_meta("dense", token_count=4)
        dense.is_sparse_decode = False
        sparse.load_spec.vllm_cached_tokens = 1
        skipped.load_spec.vllm_cached_tokens = 2
        dense.load_spec.vllm_cached_tokens = 3

        impl, _, _ = make_worker_connector(
            [sparse, skipped, dense], use_layerwise=True
        )
        impl.config.dsa_two_groups = False
        impl.num_layers = 1
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_requests = []
        impl._layerwise_retriever_is_sparse = []
        impl._layerwise_sparse_req_ids = []
        stats = SimpleNamespace(
            update_interval_vllm_hit_tokens=MagicMock(),
            update_interval_prompt_tokens=MagicMock(),
        )
        staging = MagicMock()
        calls = []

        class _FakeEngine:
            enable_shared_cpu_cache = False
            gpu_connector = SimpleNamespace(
                set_layerwise_staging_concurrency=staging
            )

            def retrieve_layer_head_token_wise(self, tokens, mask, **kwargs):
                calls.append(("sparse", kwargs["sync"]))

                def _retriever():
                    yield None
                    yield torch.ones(len(tokens), dtype=torch.bool)

                return _retriever()

            def retrieve_layer(self, tokens, mask, **kwargs):
                calls.append(("dense", kwargs["sync"]))

                def _retriever():
                    yield None
                    yield None
                    yield torch.ones(len(tokens), dtype=torch.bool)

                return _retriever()

        impl._stats_monitor = stats
        impl.lmcache_engine = _FakeEngine()

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        stats.update_interval_vllm_hit_tokens.assert_called_once_with(6)
        stats.update_interval_prompt_tokens.assert_called_once_with(12)
        staging.assert_called_once_with(2)
        assert calls == [("sparse", False), ("dense", True)]
        impl._drain_layerwise_retrievers()

    def test_start_load_kv_aborts_partial_sparse_batch(self):
        requests = [
            make_sparse_req_meta("req-1", token_count=4),
            make_sparse_req_meta("req-2", token_count=4),
        ]
        for request in requests:
            request.indexer_slot_mapping = [torch.arange(4, dtype=torch.long)]
        impl, _, _ = make_worker_connector(
            requests,
            use_layerwise=True,
            kv_role="kv_consumer",
        )
        impl.config.dsa_two_groups = True
        impl.num_layers = 1
        impl.kv_caches = {
            "model.layers.0.self_attn.attn.k_cache": torch.zeros(1),
            "model.layers.0.self_attn.indexer.k_cache": torch.zeros(1),
        }
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_requests = []
        impl._layerwise_retriever_is_sparse = []
        impl._layerwise_sparse_req_ids = []
        impl._stats_monitor = SimpleNamespace(
            update_interval_vllm_hit_tokens=lambda *_args: None,
            update_interval_prompt_tokens=lambda *_args: None,
        )
        closed = []
        indexer_calls = 0

        class _FailingEngine:
            enable_shared_cpu_cache = False

            def __init__(self):
                self.unpinned = []

            def lookup_unpin(self, req_id):
                self.unpinned.append(req_id)

            def retrieve_layer_head_token_wise(self, tokens, mask, **kwargs):
                nonlocal indexer_calls
                req_id = kwargs["req_id"]
                kv_group = kwargs["kv_group"]
                if kv_group == 1:
                    indexer_calls += 1
                    if indexer_calls == 2:
                        raise RuntimeError("index setup failed")

                def _retriever():
                    try:
                        yield None
                        while True:
                            yield torch.ones(len(tokens), dtype=torch.bool)
                    finally:
                        closed.append((req_id, kv_group))

                return _retriever()

        engine = _FailingEngine()
        impl._manager.lmcache_engine = engine

        with pytest.raises(RuntimeError, match="index setup failed"):
            impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        assert set(closed) == {
            ("req-1", 0),
            ("req-1", 1),
            ("req-2", 0),
        }
        assert engine.unpinned == ["req-1", "req-2"]
        assert impl.layerwise_retrievers == []
        assert impl._layerwise_requests == []
        assert impl._layerwise_retriever_is_sparse == []
        assert impl._layerwise_sparse_req_ids == []
        assert impl._worker_retrieve_state == {}

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))
        assert impl._layerwise_sparse_req_ids == ["req-1", "req-2"]
        impl._drain_layerwise_retrievers()

    def test_start_load_kv_without_attention_skips_step_setup(self):
        impl = _make_impl()
        impl._parent = SimpleNamespace(_get_connector_metadata=MagicMock())

        impl.start_load_kv(SimpleNamespace(attn_metadata=None))

        impl._parent._get_connector_metadata.assert_not_called()
        assert impl.current_layer == 0
        assert impl._wait_for_save_done is False

    def test_drain_layerwise_retrievers_closes_all_on_failure(self):
        impl = _make_impl()
        closed = []

        def _retriever(name, *, fail):
            try:
                yield
                if fail:
                    raise RuntimeError("drain failed")
                yield
            finally:
                closed.append(name)

        primary = _retriever("primary", fail=True)
        secondary = _retriever("secondary", fail=False)
        next(primary)
        next(secondary)
        impl.layerwise_retrievers = [(primary, secondary)]
        impl._layerwise_requests = [object()]
        impl._layerwise_retriever_is_sparse = [False]
        impl._layerwise_sparse_req_ids = ["req-1"]
        impl._layerwise_waited_groups = {0}
        impl._layerwise_sparse_indexer_sent_layers = {0}
        impl._layerwise_required_wait_groups_cache = (0,)

        with pytest.raises(RuntimeError, match="drain failed"):
            impl._drain_layerwise_retrievers()

        assert closed == ["primary", "secondary"]
        assert impl.layerwise_retrievers == []
        assert impl._layerwise_requests == []
        assert impl._layerwise_retriever_is_sparse == []
        assert impl._layerwise_sparse_req_ids == []
        assert impl._layerwise_waited_groups == set()
        assert impl._layerwise_sparse_indexer_sent_layers == set()
        assert impl._layerwise_required_wait_groups_cache is None

    def test_sparse_retrieve_failure_drops_partially_extended_state(self):
        impl = _make_impl()
        impl._release_request_lookup_pins = MagicMock()
        release_unowned = MagicMock()
        release_request = MagicMock()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            release_shared_cpu_unowned_objects=release_unowned,
            release_shared_cpu_sparse_request=release_request,
        )
        impl._drain_layerwise_retrievers = MagicMock()
        state = WorkerRetrieveState(
            req_id="req-1",
            cached_memory_objs=[["old", "new"]],
            shared_request_active=True,
        )
        impl._worker_retrieve_state["req-1"] = state
        request = _make_request()

        with pytest.raises(RuntimeError, match="index group failed"):
            with impl._sparse_retrieve_state_guard([request]):
                raise RuntimeError("index group failed")

        release_unowned.assert_called_once()
        release_request.assert_called_once_with("req-1")
        impl._drain_layerwise_retrievers.assert_called_once()
        assert "req-1" not in impl._worker_retrieve_state

    def test_sparse_retrieve_cancellation_uses_the_same_abort(self):
        impl = _make_impl()
        impl._abort_layerwise_retrieve_step = MagicMock()
        request = _make_request()

        with pytest.raises(KeyboardInterrupt):
            with impl._sparse_retrieve_state_guard([request]):
                raise KeyboardInterrupt

        impl._abort_layerwise_retrieve_step.assert_called_once_with([request])

    def test_layerwise_abort_releases_dense_and_sparse_requests(self):
        impl = _make_impl()
        impl._drop_worker_retrieve_state = MagicMock()
        impl._drain_layerwise_retrievers = MagicMock()
        dense = _make_request()
        dense.is_sparse_decode = False
        sparse = _make_request()
        sparse.req_id = "req-2"

        impl._abort_layerwise_retrieve_step([dense, sparse])

        assert [
            invocation.args
            for invocation in impl._drop_worker_retrieve_state.call_args_list
        ] == [("req-1",), ("req-2",)]
        impl._drain_layerwise_retrievers.assert_called_once_with(
            finish_dense=False
        )

    def test_store_results_remain_local_to_their_operation(self):
        impl = _make_impl()
        normal, normal_result = _make_store_request(
            impl,
            token_count=512,
            start=0,
            end=512,
            key="normal",
            tensor="normal-tensor",
        )
        first_window, first_result = _make_store_request(
            impl,
            token_count=256,
            start=0,
            end=256,
            key="window-0",
            tensor="window-tensor-0",
            decode_window=(0, 256),
        )
        second_window, second_result = _make_store_request(
            impl,
            token_count=512,
            start=256,
            end=512,
            key="window-1",
            tensor="window-tensor-1",
            decode_window=(256, 512),
        )

        operation_keys = {
            impl._layerwise_save_storer_key(request)
            for request in (normal, first_window, second_window)
        }
        assert len(operation_keys) == 3
        assert len({id(normal_result), id(first_result), id(second_result)}) == 3

    def test_empty_store_result_still_validates_request_identity(self):
        impl = _make_impl()
        request = ReqMeta(req_id="req-1", token_ids=[])
        result = LayerwiseStoreResult(request_id="req-other")

        with pytest.raises(RuntimeError, match="store result request mismatch"):
            impl._promote_layerwise_store_result(request, result)

    def test_sparse_decode_store_result_is_not_promoted(self):
        impl = _make_impl()
        request, result = _make_store_request(
            impl,
            token_count=256,
            start=0,
            end=256,
            key="decode",
            tensor="decode-tensor",
        )
        request.is_sparse_decode = True

        impl._promote_layerwise_store_result(request, result)

        assert request.req_id not in impl._worker_retrieve_state

    def test_store_seed_merges_chunked_prefill_hot_cache(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl._latent_kvcaches = [object()]
        impl._manager = SimpleNamespace(
            lmcache_engine=SimpleNamespace(
                storage_manager=None,
                store_location="LocalCPUBackend",
            ),
        )

        first, first_result = _make_store_request(
            impl,
            token_count=4096,
            start=0,
            end=4096,
            key="k0",
            tensor="t0",
        )
        second, second_result = _make_store_request(
            impl,
            token_count=8192,
            start=4096,
            end=8192,
            key="k1",
            tensor="t1",
        )
        impl._promote_layerwise_store_result(first, first_result)
        impl._promote_layerwise_store_result(second, second_result)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0, 4096]
        assert state.cached_ends == [4096, 8192]
        assert state.cached_keys == [["k0", "k1"]]
        assert state.cached_tensors == [["t0", "t1"]]
        assert state.token_count == 8192

    def test_store_seed_full_chunk_replaces_partial_at_same_start(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl._latent_kvcaches = [object()]
        impl._manager = SimpleNamespace(
            lmcache_engine=SimpleNamespace(
                enable_shared_cpu_cache=False,
                storage_manager=None,
                store_location="LocalCPUBackend",
            ),
        )
        partial, partial_result = _make_store_request(
            impl,
            token_count=100,
            start=0,
            end=100,
            key="partial-key",
            tensor="partial-tensor",
        )
        full, full_result = _make_store_request(
            impl,
            token_count=256,
            start=0,
            end=256,
            key="full-key",
            tensor="full-tensor",
        )

        impl._promote_layerwise_store_result(partial, partial_result)
        impl._promote_layerwise_store_result(full, full_result)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_keys == [["full-key"]]
        assert state.cached_tensors == [["full-tensor"]]
        assert state.token_count == 256

    def test_store_seed_partial_does_not_replace_full_at_same_start(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl._latent_kvcaches = [object()]
        impl._manager = SimpleNamespace(
            lmcache_engine=SimpleNamespace(
                enable_shared_cpu_cache=False,
                storage_manager=None,
                store_location="LocalCPUBackend",
            ),
        )
        full, full_result = _make_store_request(
            impl,
            token_count=256,
            start=0,
            end=256,
            key="full-key",
            tensor="full-tensor",
        )
        stale_partial, partial_result = _make_store_request(
            impl,
            token_count=100,
            start=0,
            end=100,
            key="partial-key",
            tensor="partial-tensor",
        )

        impl._promote_layerwise_store_result(full, full_result)
        impl._promote_layerwise_store_result(stale_partial, partial_result)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_keys == [["full-key"]]
        assert state.cached_tensors == [["full-tensor"]]

    def test_decode_save_merge_extends_pointer_cache_and_scope_token(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
            retain_shared_cpu_store_seed=MagicMock(),
        )
        old_ptrs = torch.tensor([111], dtype=torch.long)
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[old_ptrs],
            cached_shared_handles=[["h0"]],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved, store_result = _make_store_request(
            impl,
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        store_result.chunk_dev_ptrs = [[222]]
        store_result.chunk_ptrs = [torch.tensor([222], dtype=torch.long)]

        impl._promote_layerwise_store_result(saved, store_result)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0, 256]
        assert state.cached_ends == [256, 512]
        assert state.cached_chunk_dev_ptrs == [[111, 222]]
        assert state.cached_chunk_ptrs_npu[0].tolist() == [111, 222]
        # Decode-save does not broadcast shm handles. The next retrieve must
        # republish because handle coverage still reflects only the old chunk.
        assert state.cached_shared_handles == [["h0"]]
        assert state.token_count == 512
        assert state.request_scope_token == "req-1:5:512"
        assert state.shared_validation_signature is None

    def test_decode_window_save_tail_only_does_not_seed_warm_state(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl._latent_kvcaches = [object()]
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
        )
        saved, store_result = _make_store_request(
            impl,
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
            decode_window=(256, 512),
        )

        impl._promote_layerwise_store_result(saved, store_result)

        assert "req-1" not in impl._worker_retrieve_state

    def test_decode_window_save_shared_cpu_preserves_state_for_suffix_refresh(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.num_layers = 1
        impl._completed_decode_window_saves = {}
        impl._decode_window_save_completed_groups = set()
        impl._decode_window_save_expected_start = {}
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
            retain_shared_cpu_store_seed=MagicMock(),
        )
        old_ptrs = torch.tensor([111], dtype=torch.long)
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[old_ptrs],
            cached_shared_handles=[["h0"]],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved, store_result = _make_store_request(
            impl,
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
            decode_window=(256, 512),
        )
        saved.save_spec = SaveSpec(
            256,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        store_result.chunk_dev_ptrs = [[222]]
        store_result.chunk_ptrs = [torch.tensor([222], dtype=torch.long)]

        impl._record_decode_window_save_group_completed(saved, 0, store_result)
        impl._promote_layerwise_store_result(saved, store_result)
        impl._mark_decode_window_save_completed(saved)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_chunk_dev_ptrs == [[111]]
        assert state.cached_chunk_ptrs_npu[0].tolist() == [111]
        assert state.token_count == 256
        assert state.request_scope_token == "req-1:5:256"
        assert impl.get_completed_decode_window_saves() == {"req-1": 512}

        next_sparse = _make_request()
        next_sparse.token_ids = [0] * 512
        next_sparse.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=512,
            can_load=True,
        )
        assert not impl._should_invalidate_worker_retrieve_state(next_sparse, 512)

    def test_decode_window_save_shared_cpu_two_groups_preserve_state(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.num_layers = 1
        impl.kv_role = "kv_both"
        impl._completed_decode_window_saves = {}
        impl._decode_window_save_completed_groups = set()
        impl._decode_window_save_expected_start = {}
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
            retain_shared_cpu_store_seed=MagicMock(),
            config=SimpleNamespace(
                extra_config={"shared_cpu_materialize_index_on_decode_cold": True}
            ),
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            cached_shared_handles=[["h0"]],
            cached_keys_indexer=[["ik0"]],
            cached_starts_indexer=[0],
            cached_ends_indexer=[256],
            cached_memory_objs_indexer=[["im0"]],
            cached_tensors_indexer=[["it0"]],
            cached_chunk_dev_ptrs_indexer=[[333]],
            cached_chunk_ptrs_npu_indexer=[
                torch.tensor([333], dtype=torch.long)
            ],
            cached_shared_handles_indexer=[["ih0"]],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_index_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved, latent_result = _make_store_request(
            impl,
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
            decode_window=(256, 512),
        )
        saved.save_spec = SaveSpec(
            256,
            True,
            can_save_latent=True,
            can_save_indexer=True,
        )
        latent_result.chunk_dev_ptrs = [[222]]
        latent_result.chunk_ptrs = [torch.tensor([222], dtype=torch.long)]
        index_result = LayerwiseStoreResult(
            request_id=saved.req_id,
            kv_group=1,
            starts=[256],
            ends=[512],
            keys=[["ik1"]],
            memory_objs=[["im1"]],
            tensors=[["it1"]],
            chunk_dev_ptrs=[[444]],
            chunk_ptrs=[torch.tensor([444], dtype=torch.long)],
        )

        impl._record_decode_window_save_group_completed(saved, 0, latent_result)
        impl._record_decode_window_save_group_completed(saved, 1, index_result)
        impl._promote_layerwise_store_result(saved, latent_result)
        impl._promote_layerwise_store_result(saved, index_result)
        impl._mark_decode_window_save_completed(saved)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_chunk_ptrs_npu[0].tolist() == [111]
        assert state.cached_starts_indexer == [0]
        assert state.cached_ends_indexer == [256]
        assert state.cached_chunk_ptrs_npu_indexer[0].tolist() == [333]
        assert state.token_count == 256
        assert state.request_scope_token == "req-1:5:256"
        assert impl.get_completed_decode_window_saves() == {"req-1": 512}

        next_sparse = _make_request()
        next_sparse.token_ids = [0] * 512
        next_sparse.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=512,
            can_load=True,
        )
        assert not impl._should_invalidate_worker_retrieve_state(next_sparse, 512)

    def test_decode_save_merge_rejects_missing_pointer_cache(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
            retain_shared_cpu_store_seed=MagicMock(),
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved, store_result = _make_store_request(
            impl,
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        store_result.chunk_dev_ptrs = [[222]]
        # Missing per-layer NPU pointer tensor must not be hidden by a fresh
        # scope token; the next decode would otherwise look warm but be unable
        # to launch the shared sparse direct path.
        store_result.chunk_ptrs = [None]

        with pytest.raises(RuntimeError, match="incomplete shared CPU MLA"):
            impl._promote_layerwise_store_result(saved, store_result)

        state = impl._worker_retrieve_state["req-1"]
        assert state.request_scope_token == "req-1:5:256"
        assert state.pointer_cache_generation == 5
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_chunk_dev_ptrs == [[111]]
        assert state.cached_chunk_ptrs_npu[0].tolist() == [111]
        assert state.token_count == 256

    def test_decode_window_save_missing_pointer_cache_does_not_advance_state(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.num_layers = 1
        impl._completed_decode_window_saves = {}
        impl._decode_window_save_completed_groups = set()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
            retain_shared_cpu_store_seed=MagicMock(),
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved, store_result = _make_store_request(
            impl,
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
            decode_window=(256, 512),
        )
        saved.save_spec = SaveSpec(
            256,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        store_result.chunk_dev_ptrs = [[222]]
        store_result.chunk_ptrs = [None]

        impl._record_decode_window_save_group_completed(saved, 0, store_result)
        impl._promote_layerwise_store_result(saved, store_result)
        impl._mark_decode_window_save_completed(saved)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_chunk_dev_ptrs == [[111]]
        assert state.cached_chunk_ptrs_npu[0].tolist() == [111]
        assert state.token_count == 256
        assert impl.get_completed_decode_window_saves() == {}

    def test_decode_window_save_does_not_advertise_short_pointer_table(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.num_layers = 1
        impl._completed_decode_window_saves = {}
        impl._decode_window_save_completed_groups = set()
        impl._decode_window_save_expected_start = {}
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
            retain_shared_cpu_store_seed=MagicMock(),
        )
        old_chunk_count = 73
        old_end = old_chunk_count * 256
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[[f"k{i}" for i in range(old_chunk_count)]],
            cached_starts=[i * 256 for i in range(old_chunk_count)],
            cached_ends=[(i + 1) * 256 for i in range(old_chunk_count)],
            cached_memory_objs=[[f"m{i}" for i in range(old_chunk_count)]],
            cached_tensors=[[f"t{i}" for i in range(old_chunk_count)]],
            cached_chunk_dev_ptrs=[list(range(old_chunk_count))],
            cached_chunk_ptrs_npu=[
                torch.arange(old_chunk_count, dtype=torch.long)
            ],
            metadata_warm=True,
            token_count=old_end,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token=f"req-1:5:{old_end}",
        )

        saved, store_result = _make_store_request(
            impl,
            token_count=old_end + 256,
            start=old_end,
            end=old_end + 256,
            key="k-new",
            tensor="t-new",
            decode_window=(old_end, old_end + 256),
        )
        saved.save_spec = SaveSpec(
            old_end,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        store_result.chunk_dev_ptrs = [[999]]
        store_result.chunk_ptrs = [None]

        impl._record_decode_window_save_group_completed(saved, 0, store_result)
        impl._promote_layerwise_store_result(saved, store_result)
        impl._mark_decode_window_save_completed(saved)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_ends[-1] == old_end
        assert state.cached_chunk_ptrs_npu[0].numel() == old_chunk_count
        assert state.token_count == old_end
        assert impl.get_completed_decode_window_saves() == {}

    def test_decode_window_save_completion_requires_matching_window_range(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.num_layers = 1
        impl._completed_decode_window_saves = {}
        impl._decode_window_save_completed_groups = set()
        impl._decode_window_save_expected_start = {}
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
        )

        saved, store_result = _make_store_request(
            impl,
            token_count=512,
            start=0,
            end=256,
            key="wrong-window",
            tensor="wrong-tensor",
            decode_window=(256, 512),
        )
        saved.save_spec = SaveSpec(
            256,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        store_result.chunk_dev_ptrs = [[111]]
        store_result.chunk_ptrs = [torch.tensor([111], dtype=torch.long)]

        impl._record_decode_window_save_group_completed(saved, 0, store_result)
        impl._mark_decode_window_save_completed(saved)

        assert impl.get_completed_decode_window_saves() == {}

    def test_decode_window_save_out_of_order_window_does_not_merge_state(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.num_layers = 1
        impl._completed_decode_window_saves = {}
        impl._decode_window_save_completed_groups = set()
        impl._decode_window_save_expected_start = {"req-1": 256}
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
            retain_shared_cpu_store_seed=MagicMock(),
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved, store_result = _make_store_request(
            impl,
            token_count=768,
            start=512,
            end=768,
            key="future",
            tensor="future-tensor",
            decode_window=(512, 768),
        )
        saved.save_spec = SaveSpec(
            512,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        store_result.chunk_dev_ptrs = [[333]]
        store_result.chunk_ptrs = [torch.tensor([333], dtype=torch.long)]

        impl._record_decode_window_save_group_completed(saved, 0, store_result)
        impl._promote_layerwise_store_result(saved, store_result)
        impl._mark_decode_window_save_completed(saved)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_chunk_ptrs_npu[0].tolist() == [111]
        assert impl.get_completed_decode_window_saves() == {}

    def test_decode_save_merge_rejects_latent_growth_without_indexer_growth(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.num_layers = 1
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
            retain_shared_cpu_store_seed=MagicMock(),
            config=SimpleNamespace(
                extra_config={"shared_cpu_materialize_index_on_decode_cold": True}
            ),
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            cached_keys_indexer=[["ik0"]],
            cached_starts_indexer=[0],
            cached_ends_indexer=[256],
            cached_memory_objs_indexer=[["im0"]],
            cached_tensors_indexer=[["it0"]],
            cached_chunk_dev_ptrs_indexer=[[333]],
            cached_chunk_ptrs_npu_indexer=[
                torch.tensor([333], dtype=torch.long)
            ],
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_index_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved, store_result = _make_store_request(
            impl,
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        store_result.chunk_dev_ptrs = [[222]]
        store_result.chunk_ptrs = [torch.tensor([222], dtype=torch.long)]

        with pytest.raises(RuntimeError, match="incomplete shared CPU DSA index"):
            impl._promote_layerwise_store_result(saved, store_result)

        state = impl._worker_retrieve_state["req-1"]
        assert state.request_scope_token == "req-1:5:256"
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_ends_indexer == [256]
        assert state.token_count == 256

    def test_shared_indexer_selects_retrieve_mode_by_npu_and_cpu_state(self):
        request = _make_request()
        request.load_spec.lmcache_cached_tokens = 512
        state = WorkerRetrieveState(
            token_count=512,
            shared_request_active=True,
            shared_index_status="present",
            indexer_npu_resident=True,
        )

        assert (
            LMCacheConnectorV1Impl._shared_sparse_decode_indexer_retrieve_mode(
                request,
                state,
                512,
            )
            == adapter_mod.INDEXER_RETRIEVE_RESIDENT_SKIP
        )

        request.load_spec.lmcache_cached_tokens = 768
        assert (
            LMCacheConnectorV1Impl._shared_sparse_decode_indexer_retrieve_mode(
                request,
                state,
                768,
            )
            == adapter_mod.INDEXER_RETRIEVE_METADATA_ONLY
        )

        request.load_spec.lmcache_cached_tokens = 512
        state.shared_index_status = "missing"
        assert (
            LMCacheConnectorV1Impl._shared_sparse_decode_indexer_retrieve_mode(
                request,
                state,
                512,
            )
            == adapter_mod.INDEXER_RETRIEVE_FULL
        )

        state.shared_index_status = "present"
        state.indexer_npu_resident = False
        assert (
            LMCacheConnectorV1Impl._shared_sparse_decode_indexer_retrieve_mode(
                request,
                state,
                512,
            )
            == adapter_mod.INDEXER_RETRIEVE_FULL
        )

        state.indexer_npu_resident = True
        request.resumed_from_preemption = True
        assert (
            LMCacheConnectorV1Impl._shared_sparse_decode_indexer_retrieve_mode(
                request,
                state,
                512,
            )
            == adapter_mod.INDEXER_RETRIEVE_FULL
        )

    def test_full_indexer_materialization_commits_residency_at_finalization(self):
        impl = _make_impl()
        request = _make_request()
        request.is_sparse_decode = True
        state = WorkerRetrieveState(
            cached_keys=[["k"]],
            indexer_npu_materialization_pending=True,
        )
        impl._worker_retrieve_state[request.req_id] = state
        impl._prepared_sparse_sources_current = MagicMock(return_value=False)
        impl._shared_worker_retrieve_state_is_current = MagicMock(return_value=False)
        impl._publish_worker_retrieve_state = MagicMock()
        metadata = SimpleNamespace(requests=[request])

        impl._finalize_worker_retrieve_state_from_metadata(metadata)

        assert state.indexer_npu_resident is True
        assert state.indexer_npu_materialization_pending is False
        impl._publish_worker_retrieve_state.assert_called_once()

    def test_current_shared_state_skip_requires_validation_signature(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.num_layers = 1
        impl.kv_role = "kv_both"
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            config=SimpleNamespace(
                extra_config={"shared_cpu_materialize_index_on_decode_cold": True}
            ),
        )
        request = _make_request()
        request.load_spec.lmcache_cached_tokens = 512
        state = WorkerRetrieveState(
            req_id="req-1",
            cached_starts=[0, 256],
            cached_ends=[256, 512],
            cached_memory_objs=[["m0", "m1"]],
            cached_chunk_ptrs_npu=[torch.tensor([111, 222], dtype=torch.long)],
            cached_starts_indexer=[0, 256],
            cached_ends_indexer=[256, 512],
            cached_memory_objs_indexer=[["im0", "im1"]],
            cached_chunk_ptrs_npu_indexer=[
                torch.tensor([333, 444], dtype=torch.long)
            ],
            token_count=512,
            shared_latent_status="present",
            shared_index_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:512",
        )

        assert not impl._shared_worker_retrieve_state_is_current(
            state,
            request,
            512,
        )
        impl._validate_shared_worker_retrieve_state(state, request)
        assert impl._shared_worker_retrieve_state_is_current(
            state,
            request,
            512,
        )

        request.load_spec.lmcache_cached_tokens = 768
        assert not impl._shared_worker_retrieve_state_is_current(
            state,
            request,
            768,
        )

    def test_bind_shared_hot_state_rejects_short_pointer_tensor(self):
        impl = _make_impl()
        impl.num_layers = 1
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=3,
        )
        request = _make_request()
        request.load_spec.lmcache_cached_tokens = 512
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0", "k1"]],
            cached_starts=[0, 256],
            cached_ends=[256, 512],
            cached_memory_objs=[["latent-view-0", "latent-view-1"]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            metadata_warm=True,
            token_count=512,
            shared_latent_status="present",
            shared_generation=3,
            pointer_cache_generation=3,
            shared_request_active=True,
            request_scope_token="req-1:3:512",
        )

        with pytest.raises(RuntimeError, match="pointer-cache tensors"):
            _bind_worker_state(impl, request)

    def test_bind_shared_hot_state_rejects_gapped_prefix_coverage(self):
        impl = _make_impl()
        impl.num_layers = 1
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=3,
        )
        request = _make_request()
        request.token_ids = [0] * 768
        request.load_spec.lmcache_cached_tokens = 768
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0", "k2"]],
            cached_starts=[0, 512],
            cached_ends=[256, 768],
            cached_memory_objs=[["latent-view-0", "latent-view-2"]],
            cached_chunk_ptrs_npu=[torch.tensor([111, 333], dtype=torch.long)],
            metadata_warm=True,
            token_count=768,
            shared_latent_status="present",
            shared_generation=3,
            pointer_cache_generation=3,
            shared_request_active=True,
            request_scope_token="req-1:3:768",
        )

        with pytest.raises(RuntimeError, match="non-contiguous MLA"):
            _bind_worker_state(impl, request)

    def test_warm_kwargs_only_when_prefix_unchanged(self):
        impl = _make_impl()
        state = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            location="local",
            token_count=256,
        )

        warm = impl._sparse_decode_bootstrap_reuse_kwargs(256, state)
        assert warm["_retrieve_metadata_warm"] is True
        assert warm["cached_retrieve_location"] == "local"

        extended = impl._sparse_decode_bootstrap_reuse_kwargs(512, state)
        assert "_retrieve_metadata_warm" not in extended
        assert extended["cached_retrieve_location"] == "local"

    def test_tail_only_sparse_state_is_not_metadata_warm(self):
        impl = _make_impl()
        request = _make_request()
        request.token_ids = [0] * 512
        request.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=512,
            can_load=True,
        )
        state = WorkerRetrieveState(
            cached_keys=[["k1"]],
            cached_starts=[256],
            cached_ends=[512],
            metadata_warm=True,
            location="local",
            token_count=512,
        )

        warm = impl._sparse_decode_bootstrap_reuse_kwargs(512, state)

        assert "_retrieve_metadata_warm" not in warm
        assert warm["cached_retrieve_location"] == "local"
        impl._worker_retrieve_state["req-1"] = state
        assert impl._should_invalidate_worker_retrieve_state(request, 512)

    def test_invalidate_on_preemption_and_token_rollback(self):
        impl = _make_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
        )

        assert impl._should_invalidate_worker_retrieve_state(
            _make_request(resumed=True), 256
        )
        assert impl._should_invalidate_worker_retrieve_state(_make_request(), 128)

    def test_sparse_decode_selected_transfer_does_not_invalidate_full_prompt_cache(
        self,
    ):
        impl = _make_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[18879],
            metadata_warm=True,
            token_count=18879,
        )
        req = _make_request()
        req.token_ids = [0] * 18879
        assert not impl._should_invalidate_worker_retrieve_state(req, 18879)

    def test_sparse_decode_lmcache_hit_growth_invalidates(self):
        impl = _make_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
        )
        req = _make_request()
        req.token_ids = [0] * 512
        req.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=512,
            can_load=True,
        )

        assert impl._should_invalidate_worker_retrieve_state(req, 2048)

    def test_sparse_decode_aligned_lmcache_hit_growth_extends(self):
        impl = _make_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
        )
        req = _make_request()
        req.token_ids = [0] * 512
        req.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=512,
            can_load=True,
        )

        assert not impl._should_invalidate_worker_retrieve_state(req, 512)

    def test_sparse_decode_unaligned_lmcache_hit_growth_invalidates(self):
        impl = _make_impl()
        impl._lmcache_chunk_size = 256
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
        )
        req = _make_request()
        req.token_ids = [0] * 300
        req.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=300,
            can_load=True,
        )

        assert impl._should_invalidate_worker_retrieve_state(req, 300)

    def test_sparse_decode_prompt_shrink_invalidates(self):
        impl = _make_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[18879],
            metadata_warm=True,
            token_count=18879,
        )
        req = _make_request()
        req.token_ids = [0] * 4096
        assert impl._should_invalidate_worker_retrieve_state(req, 4096)

    def test_sparse_decode_shared_scope_mismatch_invalidates(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(shared_cpu_cache_generation=5)
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[18879],
            metadata_warm=True,
            token_count=18879,
            shared_request_active=True,
            request_scope_token="req-1:5:18879",
        )
        req = _make_request()
        req.token_ids = [0] * 18880

        assert impl._should_invalidate_worker_retrieve_state(req, 18880)

    def test_sparse_decode_shared_scope_invalidation_uses_retrieve_tokens(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(shared_cpu_cache_generation=5)
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[18879],
            metadata_warm=True,
            token_count=18879,
            shared_request_active=True,
            request_scope_token="req-1:5:18879",
        )
        req = _make_request()
        req.token_ids = [0] * 18880

        assert not impl._should_invalidate_worker_retrieve_state(req, 18879)

    def test_sparse_decode_invalidation_does_not_walk_index_metadata(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(shared_cpu_cache_generation=5)
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[512],
            cached_starts_indexer=[256],
            cached_ends_indexer=[512],
            metadata_warm=True,
            token_count=512,
            shared_request_active=True,
            shared_index_status="present",
            request_scope_token="req-1:5:512",
        )
        req = _make_request()
        req.token_ids = [0] * 512
        req.load_spec.lmcache_cached_tokens = 512

        assert not impl._should_invalidate_worker_retrieve_state(req, 512)

    def test_passive_shared_metadata_warm_skips_storage_probe(self):
        ensure_metadata = MagicMock(side_effect=AssertionError("should not probe"))
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            storage_manager=None,
            _is_passive=lambda: True,
            _ensure_retrieve_chunk_metadata=ensure_metadata,
        )

        location, metadata_warm = impl._warm_request_retrieve_metadata(
            WorkerRetrieveState(),
            _make_request(),
            torch.tensor([1, 2, 3]),
            torch.ones(3, dtype=torch.bool),
            kv_group=0,
            dsa_two_groups=False,
        )

        assert location is None
        assert metadata_warm is False
        ensure_metadata.assert_not_called()

    def test_prune_keeps_metadata_warm_states_until_request_finished(self):
        impl = _make_impl()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(
                metadata_warm=True, cached_keys=[["k"]]
            ),
            "req-2": WorkerRetrieveState(
                metadata_warm=True, cached_keys=[["k2"]]
            ),
        }
        impl._prune_worker_retrieve_state({"req-1"})
        assert set(impl._worker_retrieve_state) == {"req-1", "req-2"}

    def test_prune_skips_unchanged_registry(self):
        impl = _make_impl()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]])
        }

        impl._prune_worker_retrieve_state({"req-1"})
        pruned_states = impl._worker_retrieve_state
        impl._prune_worker_retrieve_state({"req-1"})

        assert impl._worker_retrieve_state is pruned_states

    def test_registry_change_invalidates_prune_key(self):
        impl = _make_impl()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]])
        }
        active_req_ids = {"req-1", "req-2"}
        impl._prune_worker_retrieve_state(active_req_ids)
        pruned_states = impl._worker_retrieve_state

        impl._set_worker_retrieve_state(
            "req-2",
            WorkerRetrieveState(metadata_warm=True, cached_keys=[["k2"]]),
        )
        impl._prune_worker_retrieve_state(active_req_ids)

        assert impl._worker_retrieve_state is not pruned_states
        assert set(impl._worker_retrieve_state) == active_req_ids

    def test_prune_releases_shared_scope_but_keeps_warm_metadata(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = True

            def is_valid(self):
                return True

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        engine = _make_shared_engine(rank0=True)
        impl = _make_impl()
        impl.lmcache_engine = engine
        backing = FakeMemObj()
        engine.register_shared_cpu_sparse_request(
            "req-2",
            owned_groups={0: [[backing]]},
        )
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
            "req-2": WorkerRetrieveState(
                req_id="req-2",
                cached_keys=[["k2"]],
                cached_starts=[0],
                cached_ends=[256],
                cached_memory_objs=[[backing]],
                cached_chunk_ptrs_npu=[torch.tensor([123], dtype=torch.long)],
                metadata_warm=True,
                shared_latent_status="present",
                shared_generation=9,
                pointer_cache_generation=9,
                shared_request_active=True,
                request_scope_token="req-2:9:256",
            ),
        }

        impl._prune_worker_retrieve_state({"req-1"})

        state = impl._worker_retrieve_state["req-2"]
        assert backing.unpinned == 1
        assert backing.released == 1
        assert "req-2" not in engine._shared_cpu_request_leases
        assert state.cached_keys == [["k2"]]
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_memory_objs == []
        assert state.cached_chunk_ptrs_npu == []
        assert state.shared_request_active is False
        assert state.metadata_warm is True

    def test_prune_drops_non_warm_finished_requests(self):
        impl = _make_impl()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
            "req-2": WorkerRetrieveState(),
        }
        impl._prune_worker_retrieve_state({"req-1"})
        assert set(impl._worker_retrieve_state) == {"req-1"}

    def test_drop_and_prune_release_lookup_pins(self):
        engine = MagicMock()
        impl = _make_impl()
        impl._manager = SimpleNamespace(lmcache_engine=engine)

        impl._drop_worker_retrieve_state("req-1")
        engine.lookup_unpin.assert_called_once_with("req-1")

        engine.lookup_unpin.reset_mock()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
            "req-2": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k2"]]),
        }
        impl._prune_worker_retrieve_state({"req-1"})
        engine.lookup_unpin.assert_not_called()

    def test_defer_lookup_unpin_for_active_sparse_decode(self):
        impl = _make_impl()
        request = _make_request()
        assert impl._should_defer_lookup_unpin_for_sparse_decode(request)

        finished = _make_request()
        finished.load_spec.can_load = False
        assert not impl._should_defer_lookup_unpin_for_sparse_decode(finished)

    def test_maybe_lookup_unpin_skips_active_sparse_decode(self):
        engine = MagicMock()
        impl = _make_impl()
        impl._manager = SimpleNamespace(lmcache_engine=engine)

        impl._maybe_lookup_unpin_for_request(_make_request())
        engine.lookup_unpin.assert_not_called()

        non_sparse = _make_request()
        non_sparse.is_sparse_decode = False
        impl._maybe_lookup_unpin_for_request(non_sparse)
        engine.lookup_unpin.assert_called_once_with("req-1")
