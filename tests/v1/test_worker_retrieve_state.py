# SPDX-License-Identifier: Apache-2.0
"""Tests for worker-local sparse decode retrieve state cache."""

# Standard
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_mod
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
    WorkerRetrieveState,
)
from lmcache.v1.gpu_connector.sparse import (
    PreparedSparseSource,
    PreparedSparseSourceLayer,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from tests.v1.connector_test_utils import (
    make_sparse_req_meta,
    make_worker_connector,
)


def _make_impl() -> LMCacheConnectorV1Impl:
    impl = object.__new__(LMCacheConnectorV1Impl)
    impl._worker_retrieve_state = {}
    return impl


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
    *,
    token_count: int,
    start: int,
    end: int,
    key: str,
    tensor: str,
) -> ReqMeta:
    return ReqMeta(
        req_id="req-1",
        token_ids=[0] * token_count,
        is_sparse_decode=False,
        cached_keys=[[key]],
        cached_starts=[start],
        cached_ends=[end],
        cached_memory_objs=[[f"mem-{key}"]],
        cached_tensors=[[tensor]],
    )


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


class TestWorkerRetrieveState:
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
        state = WorkerRetrieveState()
        state.shared_handles_by_group[0] = [["latent-handle"]]
        state.shared_views_by_group[0] = [["latent-view"]]
        state.shared_chunk_ptrs_npu_by_group[0] = [None]
        state.shared_latent_status = "present"
        state.shared_index_status = "skipped"
        state.shared_generation = 3
        state.pointer_cache_generation = 3
        state.shared_request_active = True
        state.request_scope_token = "req-1:3"

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
            impl._bind_worker_retrieve_state_to_request(request)

    def test_record_shared_state_preserves_skipped_index_without_index_objs(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState(shared_index_status="skipped")
        request = _make_request()
        request.cached_memory_objs = [["latent-view"]]
        request.cached_shared_handles = [["latent-handle"]]
        request.cached_chunk_ptrs_npu = ["latent-ptrs"]

        impl._record_shared_worker_retrieve_state(state, request)

        assert state.shared_latent_status == "present"
        assert state.shared_index_status == "skipped"
        assert state.pointer_cache_generation == 7
        assert state.shared_views_by_group == {0: [["latent-view"]]}
        assert state.shared_handles_by_group == {0: [["latent-handle"]]}

    def test_record_shared_state_uses_request_skipped_index_marker(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        state = WorkerRetrieveState()
        request = _make_request()
        request.cached_memory_objs = [
            ["latent-view-layer0"],
            ["latent-view-layer1"],
        ]
        request.cached_shared_handles = [
            ["latent-handle-layer0"],
            ["latent-handle-layer1"],
        ]
        request.cached_chunk_ptrs_npu = [
            "latent-ptrs-layer0",
            "latent-ptrs-layer1",
        ]
        request.shared_index_skipped = True

        impl._record_shared_worker_retrieve_state(state, request)

        assert state.shared_latent_status == "present"
        assert state.shared_index_status == "skipped"
        assert state.shared_request_active is True
        assert state.pointer_cache_generation == 7

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
        request.cached_memory_objs = [["latent-view"]]
        request.cached_shared_handles = [["latent-handle"]]
        request.cached_chunk_ptrs_npu = ["latent-ptrs"]
        request.cached_memory_objs_indexer = [["stale-index-view"]]
        request.cached_shared_handles_indexer = [["stale-index-handle"]]
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
        request.cached_memory_objs = [
            ["latent-view-layer0"],
            ["latent-view-layer1"],
        ]
        request.cached_shared_handles = [
            ["latent-handle-layer0"],
            ["latent-handle-layer1"],
        ]
        request.cached_chunk_ptrs_npu = [
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
        request.cached_memory_objs = [["latent-view-layer0"], []]
        request.cached_shared_handles = [["latent-handle-layer0"], []]
        request.cached_chunk_ptrs_npu = ["latent-ptrs-layer0", None]

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
        request.cached_starts = [256]
        request.cached_ends = [512]
        request.cached_memory_objs = [["latent-view"]]
        request.cached_shared_handles = [["latent-handle"]]
        request.cached_chunk_ptrs_npu = [torch.tensor([111], dtype=torch.long)]

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
        request.cached_memory_objs = [
            ["latent-view-layer0"],
            ["latent-view-layer1"],
        ]
        request.cached_shared_handles = [
            ["latent-handle-layer0"],
            ["latent-handle-layer1"],
        ]
        request.cached_chunk_ptrs_npu = [
            "latent-ptrs-layer0",
            "latent-ptrs-layer1",
        ]
        request.cached_memory_objs_indexer = [["index-view-layer0"], []]
        request.cached_shared_handles_indexer = [["index-handle-layer0"], []]
        request.cached_chunk_ptrs_npu_indexer = ["index-ptrs-layer0", None]

        with pytest.raises(RuntimeError, match="complete materialized DSA index"):
            impl._record_shared_worker_retrieve_state(state, request)

    def test_record_shared_state_accepts_complete_required_index_group(self):
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
        request.cached_memory_objs = [
            ["latent-view-layer0"],
            ["latent-view-layer1"],
        ]
        request.cached_shared_handles = [
            ["latent-handle-layer0"],
            ["latent-handle-layer1"],
        ]
        request.cached_chunk_ptrs_npu = [
            "latent-ptrs-layer0",
            "latent-ptrs-layer1",
        ]
        request.cached_memory_objs_indexer = [
            ["index-view-layer0"],
            ["index-view-layer1"],
        ]
        request.cached_shared_handles_indexer = [
            ["index-handle-layer0"],
            ["index-handle-layer1"],
        ]
        request.cached_chunk_ptrs_npu_indexer = [
            "index-ptrs-layer0",
            "index-ptrs-layer1",
        ]

        impl._record_shared_worker_retrieve_state(state, request)

        assert state.shared_latent_status == "present"
        assert state.shared_index_status == "present"
        assert state.shared_request_active is True
        assert state.token_count == request.load_spec.lmcache_cached_tokens
        assert state.request_scope_token == "req-1:7:3"
        assert state.shared_views_by_group == {
            0: request.cached_memory_objs,
            1: request.cached_memory_objs_indexer,
        }
        assert state.shared_handles_by_group == {
            0: request.cached_shared_handles,
            1: request.cached_shared_handles_indexer,
        }

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
        request.cached_starts = [0, 256]
        request.cached_ends = [256, 512]
        request.cached_memory_objs = [["latent-view-0", "latent-view-1"]]
        request.cached_shared_handles = [["latent-handle-0", "latent-handle-1"]]
        request.cached_chunk_ptrs_npu = [torch.tensor([111], dtype=torch.long)]

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
        request.cached_starts = [0, 256]
        request.cached_ends = [256, 512]
        request.cached_memory_objs = [["latent-view-0", "latent-view-1"]]
        request.cached_shared_handles = [["latent-handle-0", "latent-handle-1"]]
        request.cached_chunk_ptrs_npu = [torch.tensor([111, 222], dtype=torch.long)]
        request.cached_starts_indexer = [0]
        request.cached_ends_indexer = [256]
        request.cached_memory_objs_indexer = [["index-view-0"]]
        request.cached_shared_handles_indexer = [["index-handle-0"]]
        request.cached_chunk_ptrs_npu_indexer = [
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
        class FakeMemObj:
            def __init__(self, pinned: bool = False):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = pinned

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        passive_view = FakeMemObj()
        backing_obj = FakeMemObj(pinned=True)
        state = WorkerRetrieveState(
            shared_views_by_group={0: [[passive_view]]},
            cached_shared_handles=[["handle"]],
            rank0_backing_objs_by_group={0: [[backing_obj]]},
            shared_chunk_ptrs_npu_by_group={0: [None]},
            shared_latent_status="present",
            shared_index_status="skipped",
            shared_request_active=True,
            request_scope_token="req-1:4",
        )

        LMCacheConnectorV1Impl._release_shared_worker_retrieve_state(state)

        assert passive_view.released == 1
        assert backing_obj.unpinned == 1
        assert state.shared_views_by_group == {}
        assert state.cached_shared_handles == []
        assert state.rank0_backing_objs_by_group == {}
        assert state.shared_index_status == "missing"
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
        class FakeMemObj:
            def __init__(self, pinned: bool = False):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = pinned

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        storer_closed: list[bool] = []

        def storer():
            try:
                yield
            finally:
                storer_closed.append(True)

        layerwise_storer = storer()
        next(layerwise_storer)
        passive_view = FakeMemObj()
        backing_obj = FakeMemObj(pinned=True)
        engine = MagicMock()
        impl = _make_impl()
        impl._manager = SimpleNamespace(lmcache_engine=engine)
        impl._layerwise_save_storers = {("req-1", 0): layerwise_storer}
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(
                shared_views_by_group={0: [[passive_view]]},
                rank0_backing_objs_by_group={0: [[backing_obj]]},
            )
        }

        impl._release_finished_worker_requests({"req-1"})

        assert impl._layerwise_save_storers == {}
        assert impl._worker_retrieve_state == {}
        assert storer_closed == [True]
        assert passive_view.released == 1
        assert backing_obj.unpinned == 1
        assert backing_obj.released == 1
        engine.lookup_unpin.assert_called_once_with("req-1")

    def test_save_transfers_active_shared_state_without_releasing(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0

            def ref_count_down(self):
                self.released += 1

        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=False)
        view = FakeMemObj()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["old-k"]],
            cached_starts=[0],
            cached_ends=[256],
            shared_views_by_group={0: [[view]]},
            shared_latent_status="present",
            shared_generation=8,
            pointer_cache_generation=8,
            shared_request_active=True,
            request_scope_token="req-1:8:256",
        )
        request = _make_request()
        request.cached_keys = [["new-k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]

        impl._save_worker_retrieve_state_from_request(
            request,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
        )

        state = impl._worker_retrieve_state["req-1"]
        assert view.released == 0
        assert state.shared_views_by_group == {0: [[view]]}
        assert state.shared_request_active is True
        assert state.pointer_cache_generation == 8
        assert state.request_scope_token == "req-1:8:256"
        assert state.shared_validation_signature is None

    def test_save_replacement_does_not_reuse_old_validation_signature(self):
        impl = _make_impl()
        impl.num_layers = 1
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
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
        old_state.shared_validation_signature = (
            impl._shared_worker_validation_signature(
                old_state,
                request,
                current_generation=7,
                pointer_generation=7,
                materialize_index=False,
            )
        )
        impl._worker_retrieve_state["req-1"] = old_state

        fresh = _make_request()
        fresh.token_ids = [0] * 256
        fresh.load_spec.lmcache_cached_tokens = 256
        fresh.cached_keys = [["k0"]]
        fresh.cached_starts = [0]
        fresh.cached_ends = [256]
        fresh.cached_memory_objs = [["new-latent-view"]]
        fresh.cached_chunk_ptrs_npu = [torch.tensor([222], dtype=torch.long)]
        fresh.cached_shared_handles = [["new-handle"]]

        impl._save_worker_retrieve_state_from_request(
            fresh,
            location="local",
            metadata_warm=True,
            token_count=256,
        )

        new_state = impl._worker_retrieve_state["req-1"]
        assert new_state is not old_state
        assert new_state.shared_validation_signature is not None
        assert new_state.shared_validation_signature != (
            old_state.shared_validation_signature
        )
        assert new_state.cached_memory_objs == [["new-latent-view"]]

    def test_save_records_rank0_shared_backing_objs_for_cleanup(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = True

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=9,
            metadata=SimpleNamespace(is_first_rank=lambda: True),
        )
        backing_obj = FakeMemObj()
        request = _make_request()
        request.cached_keys = [["k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]
        request.cached_memory_objs = [[backing_obj]]
        request.cached_tensors = [["tensor"]]
        request.cached_chunk_ptrs_npu = ["latent-ptrs"]
        request.cached_shared_handles = [["handle"]]

        impl._save_worker_retrieve_state_from_request(
            request,
            location="mixed",
            metadata_warm=True,
            token_count=256,
        )

        state = impl._worker_retrieve_state["req-1"]
        assert state.shared_handles_by_group == {0: [["handle"]]}
        assert state.cached_shared_handles == [["handle"]]
        assert state.rank0_backing_objs_by_group == {0: [[backing_obj]]}
        assert state.shared_latent_status == "present"
        assert state.shared_generation == 9
        assert state.pointer_cache_generation == 9
        assert state.shared_request_active is True

        LMCacheConnectorV1Impl._release_shared_worker_retrieve_state(state)
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
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=9,
            store_location="LocalCPUBackend",
            lookup_unpin=lambda _req_id: None,
            metadata=SimpleNamespace(
                world_size=1,
                is_first_rank=lambda: True,
            ),
        )
        borrowed_obj = BorrowedLocalCPUObj()
        hot_cache = {"k": borrowed_obj}
        request = _make_store_request(
            token_count=256,
            start=0,
            end=256,
            key="k",
            tensor="tensor",
        )
        request.cached_memory_objs = [[borrowed_obj]]
        request.cached_tensors = [[borrowed_obj.tensor]]

        impl._maybe_seed_worker_retrieve_state_from_store(request)

        seeded_state = impl._worker_retrieve_state["req-1"]
        assert seeded_state.shared_request_active is False
        assert seeded_state.rank0_backing_objs_by_group == {0: [[borrowed_obj]]}
        assert borrowed_obj.ref_count == 2
        assert borrowed_obj.pin_count == 1

        impl._retain_rank0_store_seed_state(seeded_state)
        assert borrowed_obj.ref_count == 2
        assert borrowed_obj.pin_count == 1

        request.is_sparse_decode = True
        request.load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=256,
            can_load=True,
        )
        request.cached_chunk_ptrs_npu = ["latent-ptrs"]
        impl._save_worker_retrieve_state_from_request(
            request,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
        )

        state = impl._worker_retrieve_state["req-1"]
        assert state.shared_request_active is True
        assert state.rank0_backing_objs_by_group == {0: [[borrowed_obj]]}
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

    def test_save_releases_replaced_passive_shared_views(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0

            def ref_count_down(self):
                self.released += 1

        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=10,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        old_view = FakeMemObj()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["old-k"]],
            cached_starts=[0],
            cached_ends=[256],
            shared_views_by_group={0: [[old_view]]},
            shared_latent_status="present",
            shared_generation=9,
            pointer_cache_generation=9,
            shared_request_active=True,
            request_scope_token="req-1:9:256",
        )
        new_view = FakeMemObj()
        request = _make_request()
        request.cached_keys = [["new-k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]
        request.cached_memory_objs = [[new_view]]
        request.cached_chunk_ptrs_npu = ["latent-ptrs"]
        request.cached_shared_handles = [["new-handle"]]

        impl._save_worker_retrieve_state_from_request(
            request,
            location="mixed",
            metadata_warm=True,
            token_count=256,
        )

        state = impl._worker_retrieve_state["req-1"]
        assert old_view.released == 1
        assert new_view.released == 0
        assert state.shared_views_by_group == {0: [[new_view]]}
        assert state.shared_generation == 10

    def test_save_records_rank0_shared_request_in_engine_registry(self):
        class FakeMemObj:
            is_pinned = True

            def ref_count_down(self):
                pass

            def unpin(self):
                pass

        register = MagicMock()
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=9,
            metadata=SimpleNamespace(is_first_rank=lambda: True),
            register_shared_cpu_sparse_request=register,
        )
        request = _make_request()
        request.token_ids = [1] * 512
        request.cached_keys = [["k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]
        request.cached_memory_objs = [[FakeMemObj()]]
        request.cached_chunk_ptrs_npu = ["latent-ptrs"]
        request.cached_shared_handles = [["handle"]]

        impl._save_worker_retrieve_state_from_request(
            request,
            location="mixed",
            metadata_warm=True,
            token_count=256,
        )

        register.assert_called_once_with(
            "req-1",
            token_count=256,
            phase="sparse_decode_bootstrap",
        )

    def test_register_failure_releases_unstored_rank0_shared_objects(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = True

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        def fail_register(*_args, **_kwargs):
            raise RuntimeError("capacity registry failed")

        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=9,
            metadata=SimpleNamespace(is_first_rank=lambda: True),
            register_shared_cpu_sparse_request=fail_register,
        )
        backing_obj = FakeMemObj()
        request = _make_request()
        request.cached_keys = [["k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]
        request.cached_memory_objs = [[backing_obj]]
        request.cached_chunk_ptrs_npu = ["latent-ptrs"]
        request.cached_shared_handles = [["handle"]]

        with pytest.raises(RuntimeError, match="capacity registry failed"):
            impl._save_worker_retrieve_state_from_request(
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
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=9,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        passive_view = FakeMemObj()
        request = _make_request()
        request.cached_memory_objs = [[passive_view]]
        request.cached_shared_handles = [["handle"]]

        with pytest.raises(RuntimeError, match="pointer-cache install"):
            impl._save_worker_retrieve_state_from_request(
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

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=9,
            metadata=SimpleNamespace(is_first_rank=lambda: True),
        )
        backing_obj = FakeMemObj()
        request = _make_request()
        request.cached_keys = [["k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]
        request.cached_memory_objs = [[backing_obj]]
        request.cached_shared_handles = [["handle"]]

        with pytest.raises(RuntimeError, match="pointer-cache install"):
            impl._save_worker_retrieve_state_from_request(
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
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=9,
            metadata=SimpleNamespace(is_first_rank=lambda: False),
        )
        old_view = FakeMemObj()
        old_state = WorkerRetrieveState(
            cached_keys=[["old-k"]],
            cached_starts=[0],
            cached_ends=[256],
            shared_views_by_group={0: [[old_view]]},
            shared_latent_status="present",
            shared_generation=8,
            pointer_cache_generation=8,
            shared_request_active=True,
            request_scope_token="req-1:8:256",
        )
        impl._worker_retrieve_state["req-1"] = old_state
        new_view = FakeMemObj()
        request = _make_request()
        request.cached_keys = [["new-k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]
        request.cached_memory_objs = [[new_view]]
        request.cached_shared_handles = [["new-handle"]]

        with pytest.raises(RuntimeError, match="pointer-cache install"):
            impl._save_worker_retrieve_state_from_request(
                request,
                location="mixed",
                metadata_warm=True,
                token_count=256,
            )

        assert impl._worker_retrieve_state["req-1"] is old_state
        assert old_state.shared_views_by_group == {0: [[old_view]]}
        assert old_state.pointer_cache_generation == 8
        assert old_view.released == 0
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

    def test_bootstrap_wait_appends_partial_tail_to_explicit_sparse_load(self):
        req = make_sparse_req_meta("req-1", token_count=542)
        assert req.load_spec is not None
        req.load_spec.bootstrap_sample = True
        req.bootstrap_tail_token_indices = torch.tensor(
            [512, 513], dtype=torch.int32
        )
        req.bootstrap_tail_slot_mapping = torch.tensor(
            [2000, 2001], dtype=torch.long
        )
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._bootstrap_layerwise_req_ids = ["req-1"]
        impl._layerwise_retriever_is_sparse = [True]

        captured = []

        def _retriever():
            payload = yield None
            captured.append(payload)
            yield torch.ones(542, dtype=torch.bool)

        retriever = _retriever()
        next(retriever)
        impl.layerwise_retrievers = [(retriever, None)]

        selected_tokens = torch.tensor([[10, 11]], dtype=torch.int32)
        target_slot_mapping = torch.tensor([[900, 901]], dtype=torch.long)
        impl.wait_for_layer_load(
            "model.layers.0.self_attn.attn",
            selected_tokens=selected_tokens,
            token_start_index=[0],
            request_ids=["req-1"],
            target_slot_mapping=target_slot_mapping,
        )

        selected_payload, token_start, target_payload = captured[0]
        assert token_start is None
        assert torch.equal(
            selected_payload,
            torch.tensor([10, 11, 512, 513], dtype=torch.int32),
        )
        assert torch.equal(
            target_payload,
            torch.tensor([900, 901, 2000, 2001], dtype=torch.long),
        )

    def test_bootstrap_wait_hydrates_tail_with_no_selected_topk(self):
        req = make_sparse_req_meta("req-1", token_count=4)
        assert req.load_spec is not None
        req.load_spec.bootstrap_sample = True
        req.bootstrap_tail_token_indices = torch.tensor(
            [512, 513], dtype=torch.int32
        )
        req.bootstrap_tail_slot_mapping = torch.tensor(
            [2000, 2001], dtype=torch.long
        )
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.current_layer = 0
        impl.num_layers = 2
        impl._bootstrap_layerwise_req_ids = ["req-1"]
        impl._layerwise_retriever_is_sparse = [True]

        captured = []

        def _retriever():
            payload = yield None
            captured.append(payload)
            yield torch.ones(4, dtype=torch.bool)

        retriever = _retriever()
        next(retriever)
        impl.layerwise_retrievers = [(retriever, None)]

        impl.wait_for_layer_load("model.layers.0.self_attn.attn")

        selected_payload, token_start, target_payload = captured[0]
        assert token_start is None
        assert torch.equal(
            selected_payload,
            torch.tensor([0, 1, 2, 3, 512, 513], dtype=torch.int32),
        )
        assert torch.equal(
            target_payload,
            torch.tensor([0, 1, 2, 3, 2000, 2001], dtype=torch.long),
        )

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

    def test_bind_rehydrates_scheduler_empty_metadata(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=False)
        request = _make_request()
        assert request.cached_keys == []

        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            location="local",
            token_count=256,
        )

        bound = impl._bind_worker_retrieve_state_to_request(request)
        assert bound is not None
        assert request.cached_keys == [["layer0-key"]]
        assert request.cached_starts == [0]
        assert request.cached_ends == [256]

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
            impl._bind_worker_retrieve_state_to_request(request)

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
            impl._bind_worker_retrieve_state_to_request(request)

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
            impl._bind_worker_retrieve_state_to_request(request)

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
            impl._bind_worker_retrieve_state_to_request(request)

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

        assert impl._bind_worker_retrieve_state_to_request(request) is not None

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

        assert impl._bind_worker_retrieve_state_to_request(request) is not None
        assert calls == [(2, 1)]
        assert prefix_checks == [([0], [3], 3)]

        calls.clear()
        prefix_checks.clear()
        assert impl._bind_worker_retrieve_state_to_request(request) is not None
        assert calls == []
        assert prefix_checks == []

        state = impl._worker_retrieve_state["req-1"]
        state.cached_memory_objs = [list(layer) for layer in state.cached_memory_objs]
        calls.clear()
        prefix_checks.clear()
        assert impl._bind_worker_retrieve_state_to_request(request) is not None
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
            impl._bind_worker_retrieve_state_to_request(request)

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
            impl._bind_worker_retrieve_state_to_request(request)

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

        assert impl._bind_worker_retrieve_state_to_request(request) is not None

    def test_save_then_bind_round_trip(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=False)
        request = _make_request()
        request.cached_keys = [["k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]
        request.cached_shared_handles = [["handle"]]

        impl._save_worker_retrieve_state_from_request(
            request,
            location="local",
            metadata_warm=True,
            token_count=256,
        )

        fresh = _make_request()
        assert fresh.cached_keys == []
        impl._bind_worker_retrieve_state_to_request(fresh)
        assert fresh.cached_keys == [["k"]]
        assert fresh.cached_shared_handles == [["handle"]]

    def test_finalize_sparse_state_uses_lmcache_hit_token_count(self):
        impl = _make_impl()
        captured = {}

        def capture_save(request, *, location, metadata_warm, token_count):
            captured["request"] = request
            captured["location"] = location
            captured["metadata_warm"] = metadata_warm
            captured["token_count"] = token_count

        impl._save_worker_retrieve_state_from_request = capture_save
        request = _make_request()
        request.token_ids = [1, 2, 3, 4]
        request.load_spec.lmcache_cached_tokens = 3
        request.cached_keys = [["k"]]

        impl._finalize_worker_retrieve_state_from_metadata(
            SimpleNamespace(requests=[request])
        )

        assert captured["request"] is request
        assert captured["token_count"] == 3

    def test_request_tracker_round_trip_preserves_shared_handles(self):
        tracker = RequestTracker(
            req_id="req-1",
            prompt_len=512,
            token_ids=list(range(512)),
            allocated_block_ids=list(range(4)),
            allocated_block_ids_indexer=list(range(4)),
        )
        tracker.cached_keys = [["latent-k"]]
        tracker.cached_starts = [0]
        tracker.cached_ends = [256]
        tracker.cached_memory_objs = [["latent-mem"]]
        tracker.cached_shared_handles = [["latent-handle"]]
        tracker.cached_keys_indexer = [["index-k"]]
        tracker.cached_starts_indexer = [0]
        tracker.cached_ends_indexer = [256]
        tracker.cached_memory_objs_indexer = [["index-mem"]]
        tracker.cached_shared_handles_indexer = [["index-handle"]]

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
        assert req_meta.cached_shared_handles == [["latent-handle"]]
        assert req_meta.cached_shared_handles_indexer == [["index-handle"]]

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

    def test_bootstrap_reqmeta_maps_partial_tail_to_live_vllm_blocks(self):
        tracker = RequestTracker(
            req_id="req-1",
            prompt_len=542,
            token_ids=list(range(542)),
            allocated_block_ids=[11, 12, 13, 14, 15],
            skip_save=True,
        )
        tracker.seed_sparse_decode_tokens(tracker.token_ids)

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=128,
            lmcache_chunk_size=256,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=542,
                can_load=True,
                bootstrap_sample=True,
            ),
            is_sparse_decode=True,
        )

        assert req_meta is not None
        assert torch.equal(
            req_meta.bootstrap_tail_token_indices,
            torch.arange(512, 542, dtype=torch.int32),
        )
        assert torch.equal(
            req_meta.bootstrap_tail_slot_mapping,
            torch.arange(15 * 128, 15 * 128 + 30, dtype=torch.long),
        )

    def test_bootstrap_reqmeta_has_no_tail_for_chunk_aligned_prompt(self):
        tracker = RequestTracker(
            req_id="req-1",
            prompt_len=512,
            token_ids=list(range(512)),
            allocated_block_ids=[11, 12, 13, 14],
            skip_save=True,
        )
        tracker.seed_sparse_decode_tokens(tracker.token_ids)

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=128,
            lmcache_chunk_size=256,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=512,
                can_load=True,
                bootstrap_sample=True,
            ),
            is_sparse_decode=True,
        )

        assert req_meta is not None
        assert req_meta.bootstrap_tail_token_indices is None
        assert req_meta.bootstrap_tail_slot_mapping is None

    @pytest.mark.parametrize("can_parallel", [False, True])
    def test_sparse_decode_group_preflight_preserves_ret_mask_order(
        self, monkeypatch, can_parallel
    ):
        monkeypatch.setattr(
            adapter_mod,
            "SHARED_CPU_PARALLEL_GROUP_PREFLIGHT",
            True,
        )
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
        impl._layerwise_retriever_is_sparse = []
        impl._stats_monitor = SimpleNamespace(
            update_interval_vllm_hit_tokens=lambda *_args: None,
            update_interval_prompt_tokens=lambda *_args: None,
        )

        captured_kwargs = []
        prime_barrier = threading.Barrier(2)
        prime_threads = set()
        prime_threads_lock = threading.Lock()

        class _FakeSharedEngine:
            enable_shared_cpu_cache = True
            supports_parallel_shared_cpu_group_preflight = True
            shared_cpu_cache_generation = 1
            config = SimpleNamespace(
                extra_config={"shared_cpu_materialize_index_on_decode_cold": True}
            )

            def can_run_parallel_shared_cpu_group_preflight(self):
                return can_parallel

            def retrieve_layer_head_token_wise(self, tokens, mask, **kwargs):
                captured_kwargs.append(kwargs)

                def _retriever():
                    if can_parallel:
                        prime_barrier.wait(timeout=2)
                    with prime_threads_lock:
                        prime_threads.add(threading.get_ident())
                    kwargs["ret_mask"].fill_(kwargs["kv_group"] == 1)
                    yield kwargs["ret_mask"]
                    yield torch.ones(len(tokens), dtype=torch.bool)

                return _retriever()

        impl.lmcache_engine = _FakeSharedEngine()

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        assert len(captured_kwargs) == 2
        assert captured_kwargs[0]["kv_group"] == 0
        assert captured_kwargs[1]["kv_group"] == 1
        assert captured_kwargs[0]["ret_mask"] is req.decode_ret_mask
        if can_parallel:
            assert captured_kwargs[1]["ret_mask"] is not req.decode_ret_mask
        else:
            assert captured_kwargs[1]["ret_mask"] is req.decode_ret_mask
        assert torch.all(captured_kwargs[1]["ret_mask"])
        assert torch.all(req.decode_ret_mask)
        assert len(prime_threads) == (2 if can_parallel else 1)
        assert (
            captured_kwargs[0]["shared_cpu_request_preflight_state"]
            is captured_kwargs[1]["shared_cpu_request_preflight_state"]
        )

    def test_parallel_group_preflight_preserves_group_order_on_failure(self):
        prime_barrier = threading.Barrier(2)
        closed = []

        def make_retriever(group_id):
            try:
                yield from ()
                prime_barrier.wait(timeout=2)
                raise RuntimeError(f"group-{group_id}-failed")
            finally:
                closed.append(group_id)

        with pytest.raises(RuntimeError, match="group-0-failed"):
            adapter_mod._prime_sparse_retriever_pair(
                make_retriever(0),
                make_retriever(1),
                request_id="req-1",
                canonical_ret_mask=None,
                indexer_ret_mask=None,
            )

        assert sorted(closed) == [0, 1]

    def test_sparse_decode_start_uses_minimal_prepared_kwargs(self):
        req = make_sparse_req_meta("req-1", token_count=256)
        req.cached_keys = [["layer-key"]]
        req.cached_starts = [0]
        req.cached_ends = [256]
        req.cached_tensors = [[torch.zeros(256)]]
        req.cached_chunk_ptrs_npu = [torch.tensor([123], dtype=torch.long)]
        impl, _, _ = make_worker_connector([req], use_layerwise=True)
        impl.config.dsa_two_groups = False
        impl.num_layers = 1
        impl.kv_caches = {
            "model.layers.0.self_attn.attn.k_cache": torch.zeros(1),
        }
        impl._refresh_kvcaches_list()
        impl.layerwise_retrievers = []
        impl._layerwise_retriever_is_sparse = []
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
        impl._save_worker_retrieve_state_from_request(
            req,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
        )
        req.cached_keys = []
        req.cached_starts = []
        req.cached_ends = []
        req.cached_tensors = []
        req.cached_chunk_ptrs_npu = []

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
        }
        prepared_source = kwargs["prepared_sparse_source"]
        assert prepared_source.total_tokens == 256
        assert prepared_source.chunk_token_counts == (256,)
        assert prepared_source.pointer_device == torch.device("cpu")
        assert req.cached_keys == []
        assert req.cached_tensors == []
        impl._drain_layerwise_retrievers()

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

        first = _make_store_request(
            token_count=4096,
            start=0,
            end=4096,
            key="k0",
            tensor="t0",
        )
        second = _make_store_request(
            token_count=8192,
            start=4096,
            end=8192,
            key="k1",
            tensor="t1",
        )
        first.cached_shared_handles = [["h0"]]
        second.cached_shared_handles = [["h1"]]

        impl._maybe_seed_worker_retrieve_state_from_store(first)
        impl._maybe_seed_worker_retrieve_state_from_store(second)

        state = impl._worker_retrieve_state["req-1"]
        assert state.cached_starts == [0, 4096]
        assert state.cached_ends == [4096, 8192]
        assert state.cached_keys == [["k0", "k1"]]
        assert state.cached_tensors == [["t0", "t1"]]
        assert state.cached_shared_handles == [["h0", "h1"]]
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
        partial = _make_store_request(
            token_count=100,
            start=0,
            end=100,
            key="partial-key",
            tensor="partial-tensor",
        )
        full = _make_store_request(
            token_count=256,
            start=0,
            end=256,
            key="full-key",
            tensor="full-tensor",
        )

        impl._maybe_seed_worker_retrieve_state_from_store(partial)
        impl._maybe_seed_worker_retrieve_state_from_store(full)

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
        full = _make_store_request(
            token_count=256,
            start=0,
            end=256,
            key="full-key",
            tensor="full-tensor",
        )
        stale_partial = _make_store_request(
            token_count=100,
            start=0,
            end=100,
            key="partial-key",
            tensor="partial-tensor",
        )

        impl._maybe_seed_worker_retrieve_state_from_store(full)
        impl._maybe_seed_worker_retrieve_state_from_store(stale_partial)

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
            rank0_backing_objs_by_group={0: [["m0"]]},
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved = _make_store_request(
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        saved.cached_chunk_dev_ptrs = [[222]]
        saved.cached_chunk_ptrs_npu = [torch.tensor([222], dtype=torch.long)]

        impl._maybe_seed_worker_retrieve_state_from_store(saved)

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
        saved = _make_store_request(
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        saved.is_decode_window_save = True
        saved.decode_window_start = 256
        saved.decode_window_end = 512
        saved.decode_window_size = 256

        impl._maybe_seed_worker_retrieve_state_from_store(saved)

        assert "req-1" not in impl._worker_retrieve_state

    def test_decode_window_save_shared_cpu_defers_to_tail_refresh(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=False)
        impl.num_layers = 1
        impl._lmcache_chunk_size = 256
        impl._completed_decode_window_saves = {}
        impl._decode_window_save_completed_groups = set()
        impl._decode_window_save_expected_start = {}
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
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
            rank0_backing_objs_by_group={0: [["m0"]]},
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved = _make_store_request(
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        saved.is_decode_window_save = True
        saved.decode_window_start = 256
        saved.decode_window_end = 512
        saved.decode_window_size = 256
        saved.save_spec = SaveSpec(
            256,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        saved.cached_chunk_dev_ptrs = [[222]]
        saved.cached_chunk_ptrs_npu = [torch.tensor([222], dtype=torch.long)]

        impl._record_decode_window_save_group_completed(saved, 0)
        impl._maybe_seed_worker_retrieve_state_from_store(saved)
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
        assert (
            impl._shared_tail_refresh_prefix_chunks(state, 0, 512) == 1
        )

    def test_decode_window_save_shared_cpu_two_groups_refreshes_together(self):
        impl = _make_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.num_layers = 1
        impl._lmcache_chunk_size = 256
        impl.kv_role = "kv_both"
        impl._completed_decode_window_saves = {}
        impl._decode_window_save_completed_groups = set()
        impl._decode_window_save_expected_start = {}
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
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
            rank0_backing_objs_by_group={0: [["m0"]], 1: [["im0"]]},
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_index_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved = _make_store_request(
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        saved.is_decode_window_save = True
        saved.decode_window_start = 256
        saved.decode_window_end = 512
        saved.decode_window_size = 256
        saved.save_spec = SaveSpec(
            256,
            True,
            can_save_latent=True,
            can_save_indexer=True,
        )
        saved.cached_chunk_dev_ptrs = [[222]]
        saved.cached_chunk_ptrs_npu = [torch.tensor([222], dtype=torch.long)]
        saved.cached_keys_indexer = [["ik1"]]
        saved.cached_starts_indexer = [256]
        saved.cached_ends_indexer = [512]
        saved.cached_memory_objs_indexer = [["im1"]]
        saved.cached_tensors_indexer = [["it1"]]
        saved.cached_chunk_dev_ptrs_indexer = [[444]]
        saved.cached_chunk_ptrs_npu_indexer = [
            torch.tensor([444], dtype=torch.long)
        ]

        impl._record_decode_window_save_group_completed(saved, 0)
        impl._record_decode_window_save_group_completed(saved, 1)
        impl._maybe_seed_worker_retrieve_state_from_store(saved)
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
        assert (
            impl._shared_tail_refresh_prefix_chunks(state, 0, 512) == 1
        )
        assert (
            impl._shared_tail_refresh_prefix_chunks(state, 1, 512) == 1
        )

    def test_decode_save_merge_rejects_missing_pointer_cache(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=5,
            storage_manager=None,
            store_location="LocalCPUBackend",
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            rank0_backing_objs_by_group={0: [["m0"]]},
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved = _make_store_request(
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        saved.cached_chunk_dev_ptrs = [[222]]
        # Missing per-layer NPU pointer tensor must not be hidden by a fresh
        # scope token; the next decode would otherwise look warm but be unable
        # to launch the shared sparse direct path.
        saved.cached_chunk_ptrs_npu = [None]

        with pytest.raises(RuntimeError, match="incomplete shared CPU MLA"):
            impl._maybe_seed_worker_retrieve_state_from_store(saved)

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
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            rank0_backing_objs_by_group={0: [["m0"]]},
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved = _make_store_request(
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        saved.is_decode_window_save = True
        saved.decode_window_start = 256
        saved.decode_window_end = 512
        saved.decode_window_size = 256
        saved.save_spec = SaveSpec(
            256,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        saved.cached_chunk_dev_ptrs = [[222]]
        saved.cached_chunk_ptrs_npu = [None]

        impl._record_decode_window_save_group_completed(saved, 0)
        impl._maybe_seed_worker_retrieve_state_from_store(saved)
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
            rank0_backing_objs_by_group={
                0: [[f"m{i}" for i in range(old_chunk_count)]]
            },
            metadata_warm=True,
            token_count=old_end,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token=f"req-1:5:{old_end}",
        )

        saved = _make_store_request(
            token_count=old_end + 256,
            start=old_end,
            end=old_end + 256,
            key="k-new",
            tensor="t-new",
        )
        saved.is_decode_window_save = True
        saved.decode_window_start = old_end
        saved.decode_window_end = old_end + 256
        saved.decode_window_size = 256
        saved.save_spec = SaveSpec(
            old_end,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        saved.cached_chunk_dev_ptrs = [[999]]
        saved.cached_chunk_ptrs_npu = [None]

        impl._record_decode_window_save_group_completed(saved, 0)
        impl._maybe_seed_worker_retrieve_state_from_store(saved)
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

        saved = _make_store_request(
            token_count=512,
            start=0,
            end=256,
            key="wrong-window",
            tensor="wrong-tensor",
        )
        saved.is_decode_window_save = True
        saved.decode_window_start = 256
        saved.decode_window_end = 512
        saved.decode_window_size = 256
        saved.save_spec = SaveSpec(
            256,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        saved.cached_chunk_dev_ptrs = [[111]]
        saved.cached_chunk_ptrs_npu = [torch.tensor([111], dtype=torch.long)]

        impl._record_decode_window_save_group_completed(saved, 0)
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
        )
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k0"]],
            cached_starts=[0],
            cached_ends=[256],
            cached_memory_objs=[["m0"]],
            cached_tensors=[["t0"]],
            cached_chunk_dev_ptrs=[[111]],
            cached_chunk_ptrs_npu=[torch.tensor([111], dtype=torch.long)],
            rank0_backing_objs_by_group={0: [["m0"]]},
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved = _make_store_request(
            token_count=768,
            start=512,
            end=768,
            key="future",
            tensor="future-tensor",
        )
        saved.is_decode_window_save = True
        saved.decode_window_start = 512
        saved.decode_window_end = 768
        saved.decode_window_size = 256
        saved.save_spec = SaveSpec(
            512,
            True,
            can_save_latent=True,
            can_save_indexer=False,
        )
        saved.cached_chunk_dev_ptrs = [[333]]
        saved.cached_chunk_ptrs_npu = [torch.tensor([333], dtype=torch.long)]

        impl._record_decode_window_save_group_completed(saved, 0)
        impl._maybe_seed_worker_retrieve_state_from_store(saved)
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
            rank0_backing_objs_by_group={0: [["m0"]], 1: [["im0"]]},
            metadata_warm=True,
            token_count=256,
            shared_latent_status="present",
            shared_index_status="present",
            shared_generation=5,
            pointer_cache_generation=5,
            shared_request_active=True,
            request_scope_token="req-1:5:256",
        )

        saved = _make_store_request(
            token_count=512,
            start=256,
            end=512,
            key="k1",
            tensor="t1",
        )
        saved.cached_chunk_dev_ptrs = [[222]]
        saved.cached_chunk_ptrs_npu = [torch.tensor([222], dtype=torch.long)]

        with pytest.raises(RuntimeError, match="incomplete shared CPU DSA index"):
            impl._maybe_seed_worker_retrieve_state_from_store(saved)

        state = impl._worker_retrieve_state["req-1"]
        assert state.request_scope_token == "req-1:5:256"
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_ends_indexer == [256]
        assert state.token_count == 256

    def test_shared_indexer_resident_only_when_present_and_covered(self):
        request = _make_request()
        request.load_spec.lmcache_cached_tokens = 512
        state = WorkerRetrieveState(
            token_count=512,
            shared_request_active=True,
            shared_index_status="present",
        )

        assert LMCacheConnectorV1Impl._shared_sparse_decode_indexer_is_resident(
            request,
            state,
            512,
        )

        request.load_spec.lmcache_cached_tokens = 768
        assert not LMCacheConnectorV1Impl._shared_sparse_decode_indexer_is_resident(
            request,
            state,
            768,
        )

        request.load_spec.lmcache_cached_tokens = 512
        state.shared_index_status = "missing"
        assert not LMCacheConnectorV1Impl._shared_sparse_decode_indexer_is_resident(
            request,
            state,
            512,
        )

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
            impl._bind_worker_retrieve_state_to_request(request)

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
            impl._bind_worker_retrieve_state_to_request(request)

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
        assert (
            impl._worker_retrieve_state_invalidation_reason(
                _make_request(resumed=True), 256
            )
            == "resumed_from_preemption"
        )
        assert impl._should_invalidate_worker_retrieve_state(_make_request(), 128)
        assert (
            impl._worker_retrieve_state_invalidation_reason(_make_request(), 128)
            == "retrieve_token_count_shrank"
        )

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

    def test_prepared_sparse_source_reuses_chunk_aligned_prefix(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=True)
        first = torch.zeros(1)
        partial_tail = torch.ones(1)
        ptrs = torch.tensor([111, 222], dtype=torch.long)
        source = PreparedSparseSource(
            layers=(
                PreparedSparseSourceLayer(
                    tensors=(first, partial_tail),
                    chunk_ptrs_npu=ptrs,
                ),
            ),
            total_tokens=286,
            chunk_token_counts=(256, 30),
            pointer_device=torch.device("cpu"),
        )
        state = WorkerRetrieveState(
            shared_request_active=True,
            prepared_sparse_sources={0: source},
        )

        prefix = impl._prepared_sparse_source(state, 0, 256)

        assert prefix is not None
        assert prefix.total_tokens == 256
        assert prefix.chunk_token_counts == (256,)
        assert prefix.layers[0].tensors == (first,)
        assert prefix.layers[0].chunk_ptrs_npu.tolist() == [111]
        assert impl._prepared_sparse_source(state, 0, 256) is prefix

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
        assert (
            impl._worker_retrieve_state_invalidation_reason(req, 2048)
            == "lmcache_cached_prefix_grew"
        )

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
        assert (
            impl._worker_retrieve_state_invalidation_reason(req, 18880)
            == "shared_request_scope_changed"
        )

    def test_sparse_decode_prepared_token_growth_reports_exact_reason(self):
        impl = _make_impl()
        impl.lmcache_engine = SimpleNamespace(shared_cpu_cache_generation=5)
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            req_id="req-1",
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
            shared_request_active=True,
            shared_generation=5,
            request_scope_token="req-1:5:256",
            prepared_sparse_sources={0: SimpleNamespace(total_tokens=256)},
        )
        req = _make_request()
        req.token_ids = [0] * 512
        req.load_spec.lmcache_cached_tokens = 512

        assert (
            impl._worker_retrieve_state_invalidation_reason(req, 512)
            == "prepared_token_count_changed"
        )

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

    def test_get_finished_releases_worker_request_state(self):
        impl = _make_impl()
        impl._release_finished_worker_requests = MagicMock()

        assert impl.get_finished({"req-1"}) == (None, None)

        impl._release_finished_worker_requests.assert_called_once_with({"req-1"})

    def test_empty_connector_metadata_keeps_live_worker_state(self):
        impl, _, _ = make_worker_connector([], use_layerwise=True)
        impl._kvcaches_list = [torch.zeros(1)]
        impl.layerwise_retrievers = []
        impl._layerwise_retriever_is_sparse = []
        state = WorkerRetrieveState(
            req_id="req-1",
            shared_request_active=True,
            shared_generation=9,
            request_scope_token="req-1:9:256",
            token_count=256,
        )
        impl._worker_retrieve_state["req-1"] = state

        impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))

        assert impl._worker_retrieve_state["req-1"] is state
        assert state.shared_request_active

    def test_prune_releases_shared_scope_but_keeps_warm_metadata(self):
        class FakeMemObj:
            def __init__(self):
                self.released = 0
                self.unpinned = 0
                self.is_pinned = True

            def ref_count_down(self):
                self.released += 1

            def unpin(self):
                self.unpinned += 1
                self.is_pinned = False

        engine = SimpleNamespace(
            release_shared_cpu_sparse_request=MagicMock(),
        )
        impl = _make_impl()
        impl.lmcache_engine = engine
        backing = FakeMemObj()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
            "req-2": WorkerRetrieveState(
                req_id="req-2",
                cached_keys=[["k2"]],
                cached_starts=[0],
                cached_ends=[256],
                cached_memory_objs=[[backing]],
                cached_chunk_ptrs_npu=[torch.tensor([123], dtype=torch.long)],
                rank0_backing_objs_by_group={0: [[backing]]},
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
        engine.release_shared_cpu_sparse_request.assert_called_once_with("req-2")
        assert state.cached_keys == [["k2"]]
        assert state.cached_starts == [0]
        assert state.cached_ends == [256]
        assert state.cached_memory_objs == []
        assert state.cached_chunk_ptrs_npu == []
        assert state.shared_request_active is False
        assert state.metadata_warm is True
        assert (
            state.last_shared_scope_release_reason
            == "absent_from_current_connector_metadata"
        )
        assert state.last_shared_scope_release_token_count == 256
        assert (
            impl._prepared_sparse_source_miss_reason(
                state,
                0,
                256,
                shared_cpu_enabled=True,
            )
            == "shared_scope_inactive_after:absent_from_current_connector_metadata"
        )

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
