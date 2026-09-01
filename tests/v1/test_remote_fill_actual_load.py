# SPDX-License-Identifier: Apache-2.0
"""Actual-load tier-coherence tests for direct remote LocalCPU fill."""

# Standard
from collections import defaultdict
import logging
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    _RemoteFillMaterializationError,
)
from lmcache.v1.event_manager import EventStatus
from lmcache.v1.remote_fill import content_digest
from lmcache.v1.shared_cpu_cache import SharedHandleEnvelope
from lmcache.v1.storage_backend.storage_manager import StorageManager


class _TokenDatabase:
    def process_tokens(
        self,
        tokens=None,
        hashes=None,
        offsets=None,
        mask=None,
        request_configs=None,
        kv_group=0,
    ):
        del tokens, hashes, offsets, mask
        for chunk_index in range(2):
            start = chunk_index * 4
            yield (
                start,
                start + 4,
                self._make_key_by_hash(
                    0x100 + chunk_index,
                    request_configs,
                    kv_group=kv_group,
                ),
            )

    @staticmethod
    def _make_key_by_hash(
        chunk_hash,
        request_configs=None,
        kv_group=0,
        valid_tokens=None,
    ):
        del valid_tokens
        return CacheEngineKey(
            "model",
            1,
            0,
            chunk_hash,
            torch.bfloat16,
            request_configs,
            kv_group=kv_group,
        )


class _Stats:
    @staticmethod
    def on_lookup_request(_tokens):
        return object()

    @staticmethod
    def on_retrieve_request(_tokens):
        return object()

    @staticmethod
    def on_lookup_finished(_stats, _result):
        return None

    @staticmethod
    def on_retrieve_finished(_stats, _result):
        return None


class _PairStorageManager:
    def __init__(self, local_pairs=0, remote_pairs=0):
        self.local_pairs = local_pairs
        self.remote_pairs = remote_pairs
        self.local_page_hits = None
        self.pair_calls = []
        self.page_calls = []
        self.page_results = {}
        self.page_result_limits = {}
        self.page_errors = {}
        self.legacy_results = {}
        self.unpinned = []

    def batched_contains_two_group_layer_pages(
        self,
        group0,
        group1,
        search_range=None,
        pin=False,
        diagnostics=None,
    ):
        del diagnostics
        group0 = list(group0)
        group1 = list(group1)
        self.pair_calls.append((group0, group1, search_range, pin))
        local = search_range == ["LocalCPUBackend"]
        pairs = min(
            self.local_pairs if local else self.remote_pairs,
            len(group0),
        )
        location = "LocalCPUBackend" if local else "RemoteBackend"
        keys = [
            key
            for pair in zip(group0[:pairs], group1[:pairs], strict=True)
            for key in pair
        ]
        return pairs, {location: keys} if keys else {}

    def batched_contains_layer_pages(
        self,
        keys,
        search_range=None,
        pin=False,
    ):
        keys = list(keys)
        self.page_calls.append((keys, search_range, pin))
        if (
            search_range == ["LocalCPUBackend"]
            and self.local_page_hits is not None
        ):
            hits = min(int(self.local_page_hits), len(keys))
            return (
                hits,
                {"LocalCPUBackend": keys[:hits]} if hits else {},
            )
        error = self.page_errors.get(keys[0].kv_group)
        if error is not None:
            raise error
        limit = self.page_result_limits.get(keys[0].kv_group)
        if limit is not None:
            if limit <= 0:
                return 0, {}
            self.page_result_limits[keys[0].kv_group] = limit - 1
        result = self.page_results.get(keys[0].kv_group)
        if result is None:
            return 0, {}
        return 1, {result: keys}

    def batched_contains(self, keys, search_range=None, pin=False):
        del search_range, pin
        keys = list(keys)
        result = self.legacy_results.get(keys[0].kv_group)
        if result is None:
            return 0, {}
        return len(keys), {result: keys}

    def batched_unpin(self, keys, locations=None):
        self.unpinned.append((list(keys), locations))

    @staticmethod
    def touch_cache():
        return None


class _PairPageBackend:
    def __init__(self, raw_hits: int):
        self.raw_hits = raw_hits
        self.calls = []

    def batched_contains_layer_pages(self, keys, pin=False):
        keys = list(keys)
        self.calls.append((keys, pin))
        return min(self.raw_hits, len(keys))

    @staticmethod
    def batched_unpin(_keys):
        return None


class _PairPageManager:
    def __init__(self, backends):
        self.backends = backends

    def get_active_storage_backends(self, search_range=None):
        return [
            (name, backend)
            for name, backend in self.backends
            if search_range is None or name in search_range
        ]


def _engine(storage_manager):
    engine = object.__new__(LMCacheEngine)
    engine._init_failed = False
    engine._health_monitor = None
    engine.storage_manager = storage_manager
    engine.token_database = _TokenDatabase()
    engine.stats_monitor = _Stats()
    engine.retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]
    engine.lookup_pins = defaultdict(lambda: defaultdict(list))
    engine._remote_fill_lookup_plans = {}
    engine.use_layerwise = True
    engine.num_layers = 2
    engine.enable_shared_cpu_cache = True
    engine.save_only_first_rank = True
    engine.save_indexer_only_first_rank = True
    engine.metadata = SimpleNamespace(
        use_mla=True,
        world_size=2,
        worker_id=0,
        first_rank=0,
        is_first_rank=lambda: True,
    )
    engine.gpu_connector = object()
    engine.shared_cpu_cache_strict = False
    engine.shared_cpu_cache_generation = 1
    engine.config = SimpleNamespace(
        enable_remote_lmcache_store=True,
        dsa_two_groups=True,
        dsa_group1_load_mode="p2p_preferred",
        use_layerwise=True,
        enable_shared_cpu_cache=True,
        chunk_size=4,
        remote_url="mooncakestore://test",
        extra_config={
            "save_only_first_rank": True,
            "mooncake_page_first_multi_buffer": True,
            "mooncake_layer_merged_page_objects": True,
        },
    )
    return engine


def _local_full_hint(required_store_end: int = 8) -> dict:
    return {
        "lmcache.remote_fill_result": {
            "outcome": "LOCAL_FULL",
            "required_store_end": required_store_end,
            "destination_engine_epoch": 7,
        }
    }


def test_complete_local_two_group_prefix_is_pinned_as_pairs() -> None:
    storage = _PairStorageManager(local_pairs=2)
    engine = _engine(storage)

    assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 8

    assert len(storage.pair_calls) == 1
    group0, group1, search_range, pin = storage.pair_calls[0]
    assert [key.kv_group for key in group0] == [0, 0]
    assert [key.kv_group for key in group1] == [1, 1]
    assert search_range == ["LocalCPUBackend"]
    assert pin is True
    assert len(engine.lookup_pins["req"]["LocalCPUBackend"]) == 4
    assert {
        chunk.locations_by_group
        for chunk in engine._remote_fill_lookup_plans["req"].chunks
    } == {("LocalCPUBackend", "LocalCPUBackend")}
    chunk_plan = engine._remote_fill_retained_local_page_plan("req", 0, 8)
    assert chunk_plan is not None
    assert chunk_plan[0] == [(0, 4, 0x100), (4, 8, 0x101)]

    engine.lookup_unpin("req")
    assert "req" not in engine._remote_fill_lookup_plans
    assert "req" not in engine.lookup_pins
    assert len(storage.unpinned) == 1


def test_aborted_lookup_releases_pair_pins_and_plan() -> None:
    storage = _PairStorageManager(local_pairs=2)
    engine = _engine(storage)
    engine.event_manager = SimpleNamespace(
        get_event_status=lambda *_args: EventStatus.NOT_FOUND
    )

    assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 8
    engine.cleanup_memory_objs("req")

    assert "req" not in engine._remote_fill_lookup_plans
    assert "req" not in engine.lookup_pins
    assert len(storage.unpinned) == 1


def test_one_group_local_hole_uses_persistent_pairs_without_mixing() -> None:
    storage = _PairStorageManager(local_pairs=0, remote_pairs=2)
    engine = _engine(storage)

    assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 8

    assert [call[2] for call in storage.pair_calls] == [
        ["LocalCPUBackend"],
        ["RemoteBackend"],
    ]
    assert "LocalCPUBackend" not in engine.lookup_pins["req"]
    assert len(engine.lookup_pins["req"]["RemoteBackend"]) == 4
    assert {
        chunk.locations_by_group
        for chunk in engine._remote_fill_lookup_plans["req"].chunks
    } == {("RemoteBackend", "RemoteBackend")}


def test_persistent_direct_hbm_uses_remote_pair_proof_then_group0_overlay() -> None:
    storage = _PairStorageManager(local_pairs=2, remote_pairs=2)
    storage.local_page_hits = 1
    engine = _engine(storage)
    engine.config.dsa_group1_load_mode = "persistent_direct_hbm"
    engine.config.enable_remote_lmcache_store = False

    assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 8

    assert [call[2] for call in storage.pair_calls] == [["RemoteBackend"]]
    assert [call[3] for call in storage.pair_calls] == [False]
    assert len(storage.page_calls) == 1
    assert storage.page_calls[0][1:] == (["LocalCPUBackend"], True)
    chunks = engine._remote_fill_lookup_plans["req"].chunks
    assert [chunk.locations_by_group for chunk in chunks] == [
        ("LocalCPUBackend", "RemoteBackend"),
        ("RemoteBackend", "RemoteBackend"),
    ]
    candidates = list(engine.token_database.process_tokens())
    assert engine._remote_fill_retrieve_plan("req", candidates, 0) == [
        ("LocalCPUBackend", True),
        ("RemoteBackend", True),
    ]
    assert engine._remote_fill_retrieve_plan("req", candidates, 1) == [
        ("RemoteBackend", True),
        ("RemoteBackend", True),
    ]
    assert engine._remote_fill_retained_local_page_plan("req", 0, 4) == (
        [(0, 4, 0x100)],
        ["LocalCPUBackend"],
    )
    assert engine._remote_fill_retained_local_page_plan("req", 0, 8) is None
    assert engine._remote_fill_retained_local_page_plan("req", 1, 4) is None
    assert "RemoteBackend" not in engine.lookup_pins["req"]
    assert len(engine.lookup_pins["req"]["LocalCPUBackend"]) == 1


def test_persistent_direct_hbm_stops_at_remote_pair_hole_without_legacy_probe() -> None:
    storage = _PairStorageManager(remote_pairs=1)
    storage.local_page_hits = 0
    storage.page_results = {0: "RemoteBackend", 1: "RemoteBackend"}
    storage.legacy_results = {0: "RemoteBackend", 1: "RemoteBackend"}
    engine = _engine(storage)
    engine.config.dsa_group1_load_mode = "persistent_direct_hbm"

    assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 4

    assert len(storage.pair_calls) == 1
    assert storage.pair_calls[0][2] == ["RemoteBackend"]
    # The sole page lookup is the bounded G0 LocalCPU overlay; neither group
    # enters the legacy persistent per-layer fallback after the pair hole.
    assert len(storage.page_calls) == 1
    assert storage.page_calls[0][1] == ["LocalCPUBackend"]
    assert engine._remote_fill_lookup_plans["req"].chunks[0].locations_by_group == (
        "RemoteBackend",
        "RemoteBackend",
    )


def test_persistent_direct_hbm_requires_remote_backend_in_search_range() -> None:
    storage = _PairStorageManager(local_pairs=2, remote_pairs=2)
    engine = _engine(storage)
    engine.config.dsa_group1_load_mode = "persistent_direct_hbm"

    assert (
        engine.lookup(
            list(range(8)),
            search_range=["LocalCPUBackend"],
            lookup_id="req",
            pin=True,
        )
        == 0
    )
    assert storage.pair_calls == []
    assert "req" not in engine._remote_fill_lookup_plans


def test_persistent_direct_hbm_overlay_error_has_no_remote_pair_pins() -> None:
    storage = _PairStorageManager(remote_pairs=2)
    storage.page_errors = {0: RuntimeError("local overlay failed")}
    engine = _engine(storage)
    engine.config.dsa_group1_load_mode = "persistent_direct_hbm"

    assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 0

    assert storage.pair_calls[0][3] is False
    assert storage.unpinned == []
    assert "req" not in engine._remote_fill_lookup_plans
    assert engine.lookup_pins["req"] == {}


def test_groups_from_different_persistent_tiers_do_not_form_a_hit() -> None:
    storage = _PairStorageManager()
    storage.page_results = {0: "RemoteA", 1: "RemoteB"}
    engine = _engine(storage)
    engine.retrieve_locations = ["LocalCPUBackend", "RemoteA", "RemoteB"]

    assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 0

    assert "req" not in engine._remote_fill_lookup_plans
    assert engine.lookup_pins["req"] == {}
    assert {locations[0] for _, locations in storage.unpinned} == {
        "RemoteA",
        "RemoteB",
    }


def test_group1_lookup_error_releases_group0_and_recomputes(caplog) -> None:
    storage = _PairStorageManager()
    storage.page_results = {0: "RemoteBackend"}
    storage.page_errors = {1: RuntimeError("group1 lookup failed")}
    engine = _engine(storage)

    with caplog.at_level(logging.ERROR):
        assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 0

    assert "req" not in engine._remote_fill_lookup_plans
    assert engine.lookup_pins["req"] == {}
    assert len(storage.unpinned) == 1
    keys, locations = storage.unpinned[0]
    assert [key.kv_group for key in keys] == [0]
    assert locations == ["RemoteBackend"]
    assert '"code":"RF-D-002"' in caplog.text
    assert '"diagnostic_name":"decoder_paired_prefix_lookup_failure"' in caplog.text
    assert '"event":"remote_fill_lookup_failure"' in caplog.text
    assert '"action":"RECOMPUTE"' in caplog.text


def test_page_and_legacy_groups_share_one_persistent_chunk_plan() -> None:
    storage = _PairStorageManager()
    storage.page_results = {0: "RemoteBackend"}
    storage.page_result_limits = {0: 1}
    storage.legacy_results = {1: "RemoteBackend"}
    engine = _engine(storage)

    assert engine.lookup(list(range(8)), lookup_id="req", pin=True) == 4

    chunk = engine._remote_fill_lookup_plans["req"].chunks[0]
    assert chunk.locations_by_group == ("RemoteBackend", "RemoteBackend")
    assert chunk.page_by_group == (True, False)
    with pytest.raises(RuntimeError, match="do not match"):
        engine._remote_fill_retrieve_plan(
            "req",
            list(engine.token_database.process_tokens()),
            0,
        )
    first_candidate = [next(iter(engine.token_database.process_tokens()))]
    assert engine._remote_fill_retrieve_plan("req", first_candidate, 0) == [
        ("RemoteBackend", True)
    ]
    assert engine._remote_fill_retrieve_plan("req", first_candidate, 1) == [
        ("RemoteBackend", False)
    ]
    assert engine._remote_fill_retained_local_page_plan("req", 0, 4) is None
    with pytest.raises(RuntimeError, match="required frontier"):
        engine._remote_fill_retained_local_page_plan("req", 0, 5)


def test_missing_local_prefix_rechecks_and_falls_back_to_persistent() -> None:
    storage = _PairStorageManager(local_pairs=2)
    engine = _engine(storage)

    assert engine.lookup(list(range(8)), pin=False) == 8
    storage.local_pairs = 0
    storage.remote_pairs = 2

    assert engine.lookup(list(range(8)), lookup_id="actual", pin=True) == 8
    assert {
        chunk.locations_by_group
        for chunk in engine._remote_fill_lookup_plans["actual"].chunks
    } == {("RemoteBackend", "RemoteBackend")}
    assert "LocalCPUBackend" not in engine.lookup_pins["actual"]


def test_local_full_hint_records_retained_paired_actual_load(
    caplog, monkeypatch
) -> None:
    monkeypatch.setenv("LMCACHE_COLD_START_PERF", "1")
    storage = _PairStorageManager(local_pairs=2)
    engine = _engine(storage)

    with caplog.at_level(logging.INFO):
        assert (
            engine.lookup(
                list(range(8)),
                lookup_id="actual",
                pin=True,
                request_configs=_local_full_hint(),
            )
            == 8
        )

    snapshot = engine.remote_fill_actual_load_metrics_snapshot()
    assert snapshot.retained_at_load_total == 1
    assert snapshot.local_prefix_missing_at_load_total == 0
    assert snapshot.unexpected_remote_get_total == 0
    assert '"event":"remote_fill_actual_load"' in caplog.text
    assert '"outcome":"retained_at_load"' in caplog.text
    assert "[LMCACHE_COLD_PERF]" in caplog.text
    assert '"req_id":"actual"' in caplog.text
    assert '"clock_domain"' in caplog.text


def test_persistent_direct_hbm_group0_overlay_is_retained_actual_load() -> None:
    storage = _PairStorageManager(remote_pairs=2)
    storage.local_page_hits = 2
    engine = _engine(storage)
    engine.config.dsa_group1_load_mode = "persistent_direct_hbm"

    assert (
        engine.lookup(
            list(range(8)),
            lookup_id="actual",
            pin=True,
            request_configs=_local_full_hint(),
        )
        == 8
    )

    snapshot = engine.remote_fill_actual_load_metrics_snapshot()
    assert snapshot.retained_at_load_total == 1
    assert snapshot.local_prefix_missing_at_load_total == 0
    assert snapshot.unexpected_remote_get_total == 0


def test_local_full_hint_records_missing_local_prefix_and_remote_fallback(
    caplog, monkeypatch
) -> None:
    monkeypatch.setenv("LMCACHE_COLD_START_PERF", "1")
    storage = _PairStorageManager(local_pairs=0, remote_pairs=2)
    engine = _engine(storage)

    with caplog.at_level(logging.INFO):
        assert (
            engine.lookup(
                list(range(8)),
                lookup_id="actual",
                pin=True,
                request_configs=_local_full_hint(),
            )
            == 8
        )

    snapshot = engine.remote_fill_actual_load_metrics_snapshot()
    assert snapshot.retained_at_load_total == 0
    assert snapshot.local_prefix_missing_at_load_total == 1
    assert snapshot.unexpected_remote_get_total == 1
    assert '"event":"remote_fill_actual_load"' in caplog.text
    assert '"outcome":"local_prefix_missing_at_load"' in caplog.text
    assert '"event":"remote_fill_local_prefix_missing_at_load"' in caplog.text
    assert "remote_fill_evicted_before_load" not in caplog.text
    assert '"remote_suffix":true' in caplog.text
    group0, group1, _, _ = storage.pair_calls[0]
    expected_digest = content_digest(
        (
            tuple(key.without_layer().to_string() for key in group0),
            tuple(key.without_layer().to_string() for key in group1),
        )
    )
    assert '"local_lookup_attempted":true' in caplog.text
    assert '"local_pairs":0' in caplog.text
    assert '"required_pairs":2' in caplog.text
    assert f'"required_key_digest":"{expected_digest}"' in caplog.text


def test_local_full_hint_without_any_prefix_is_visible_and_recomputes(
    caplog,
) -> None:
    storage = _PairStorageManager(local_pairs=0, remote_pairs=0)
    engine = _engine(storage)

    with caplog.at_level(logging.WARNING):
        assert (
            engine.lookup(
                list(range(8)),
                lookup_id="actual",
                pin=True,
                request_configs=_local_full_hint(),
            )
            == 0
        )

    snapshot = engine.remote_fill_actual_load_metrics_snapshot()
    assert snapshot.retained_at_load_total == 0
    assert snapshot.local_prefix_missing_at_load_total == 1
    assert snapshot.unexpected_remote_get_total == 0
    assert '"code":"RF-D-001"' in caplog.text
    assert '"diagnostic_name":"decoder_retained_prefix_missing"' in caplog.text
    assert '"event":"remote_fill_fallback"' in caplog.text
    assert '"action":"PERSISTENT_FALLBACK_OR_RECOMPUTE"' in caplog.text


def test_rank0_materialization_failure_releases_pair_pins_and_recomputes() -> None:
    storage = _PairStorageManager(local_pairs=2)
    engine = _engine(storage)
    request_configs = _local_full_hint()

    assert (
        engine.lookup(
            list(range(8)),
            lookup_id="actual",
            pin=True,
            request_configs=request_configs,
        )
        == 8
    )

    def fail_materialization(**_kwargs):
        raise _RemoteFillMaterializationError(
            "hidden LocalCPU page is unavailable"
        )
        yield None

    engine._retrieve_layer_shared_rank0 = fail_materialization
    yielded = list(
        engine.retrieve_layer(
            list(range(8)),
            req_id="actual",
            kv_group=0,
            request_configs=request_configs,
        )
    )

    assert yielded[:-1] == [None] * (engine.num_layers + 1)
    assert not bool(yielded[-1].any())
    assert "actual" not in engine._remote_fill_lookup_plans
    assert "actual" not in engine.lookup_pins
    assert len(storage.unpinned) == 1


def test_rank0_gpu_consumer_failure_is_not_masked_as_recompute() -> None:
    storage = _PairStorageManager(local_pairs=2)
    engine = _engine(storage)
    request_configs = _local_full_hint()

    assert (
        engine.lookup(
            list(range(8)),
            lookup_id="actual",
            pin=True,
            request_configs=request_configs,
        )
        == 8
    )

    def fail_gpu_consumer(**_kwargs):
        raise RuntimeError("NPU load consumer failed")
        yield None

    engine._retrieve_layer_shared_rank0 = fail_gpu_consumer
    with pytest.raises(RuntimeError, match="NPU load consumer failed"):
        list(
            engine.retrieve_layer(
                list(range(8)),
                req_id="actual",
                kv_group=0,
                request_configs=request_configs,
            )
        )

    engine.lookup_unpin("actual")
    assert len(storage.unpinned) == 1


def test_rank0_retained_plan_mismatch_broadcasts_failure_and_recomputes() -> None:
    storage = _PairStorageManager(local_pairs=2)
    engine = _engine(storage)
    request_configs = _local_full_hint()
    broadcasts = []
    engine._broadcast_shared_envelope = broadcasts.append

    assert (
        engine.lookup(
            list(range(8)),
            lookup_id="actual",
            pin=True,
            request_configs=request_configs,
        )
        == 8
    )
    original_process_tokens = engine.token_database.process_tokens

    def mismatched_tokens(*args, **kwargs):
        for start, end, key in original_process_tokens(*args, **kwargs):
            yield start, end, CacheEngineKey(
                key.model_name,
                key.world_size,
                key.worker_id,
                key.chunk_hash + 1,
                key.dtype,
                key.request_configs,
                kv_group=key.kv_group,
            )

    engine.token_database.process_tokens = mismatched_tokens
    yielded = list(
        engine.retrieve_layer(
            list(range(8)),
            req_id="actual",
            kv_group=0,
            request_configs=request_configs,
        )
    )

    assert yielded[:-1] == [None] * (engine.num_layers + 1)
    assert not bool(yielded[-1].any())
    assert len(broadcasts) == 1
    assert broadcasts[0].status == "error"
    assert "actual" not in engine._remote_fill_lookup_plans
    assert "actual" not in engine.lookup_pins
    assert len(storage.unpinned) == 1


def test_rank0_error_envelope_returns_passive_recompute_mask_only_for_local_full(
    monkeypatch,
) -> None:
    import lmcache.v1.cache_engine as cache_engine_module

    engine = _engine(_PairStorageManager())
    engine.metadata.worker_id = 1
    engine.metadata.is_first_rank = lambda: False
    engine.shared_cpu_cache_generation = 9
    engine.shared_cpu_cache_passive_allocator = object()
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    error_envelope = SharedHandleEnvelope(
        request_id="actual",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="error",
        generation=9,
        handles=[],
        message="rank0 materialization failed",
    )
    engine._receive_shared_envelope = lambda: error_envelope
    yielded = list(
        engine.retrieve_layer(
            list(range(8)),
            req_id="actual",
            kv_group=0,
            request_configs=_local_full_hint(),
        )
    )

    assert len(yielded) == engine.num_layers + 2
    assert not bool(yielded[-1].any())

    ordinary_envelope = SharedHandleEnvelope(
        request_id="ordinary",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="error",
        generation=9,
        handles=[],
        message="ordinary materialization failed",
    )
    engine._receive_shared_envelope = lambda: ordinary_envelope
    with pytest.raises(ValueError, match="rank0 error envelope"):
        list(
            engine.retrieve_layer(
                list(range(8)),
                req_id="ordinary",
                kv_group=0,
            )
        )


def test_passive_view_construction_failure_closes_consumer_and_recomputes(
    monkeypatch,
) -> None:
    import lmcache.v1.cache_engine as cache_engine_module

    engine = _engine(_PairStorageManager())
    engine.metadata.worker_id = 1
    engine.metadata.is_first_rank = lambda: False
    engine.shared_cpu_cache_generation = 9
    closed = []

    class _FailingAllocator:
        @staticmethod
        def create_view(*_args, **_kwargs):
            raise ValueError("passive view construction failed")

    class _GPUConnector:
        @staticmethod
        def batched_to_gpu(*_args, **_kwargs):
            def consumer():
                try:
                    while True:
                        yield None
                finally:
                    closed.append(True)

            return consumer()

    engine.shared_cpu_cache_passive_allocator = _FailingAllocator()
    engine.gpu_connector = _GPUConnector()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([4]),
        torch.bfloat16,
        object(),
    )
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    envelope = SharedHandleEnvelope(
        request_id="actual",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=9,
        handles=[object()],
    )
    engine._receive_shared_envelope = lambda: envelope

    yielded = list(
        engine.retrieve_layer(
            list(range(8)),
            req_id="actual",
            kv_group=0,
            request_configs=_local_full_hint(),
        )
    )

    assert yielded[:-1] == [None] * (engine.num_layers + 1)
    assert not bool(yielded[-1].any())
    assert closed == [True]


def test_remote_fill_materialization_consensus_uses_all_rank_result() -> None:
    engine = _engine(_PairStorageManager())
    votes = []
    engine.collective_all_true_fn = lambda local_ready: (
        votes.append(local_ready) or False
    )

    assert not engine._remote_fill_all_ranks_materialized(
        True,
        req_id="actual",
        kv_group=0,
    )
    assert votes == [True]


def test_remote_fill_materialization_consensus_fails_closed_without_callback(
) -> None:
    engine = _engine(_PairStorageManager())

    assert not engine._remote_fill_all_ranks_materialized(
        True,
        req_id="actual",
        kv_group=0,
    )


def test_storage_manager_rejects_isolated_group_and_uses_next_pair_tier() -> None:
    group0 = [
        _TokenDatabase._make_key_by_hash(0x100 + index, kv_group=0).get_first_layer()
        for index in range(2)
    ]
    group1 = [
        _TokenDatabase._make_key_by_hash(0x100 + index, kv_group=1).get_first_layer()
        for index in range(2)
    ]
    isolated = _PairPageBackend(raw_hits=1)
    paired = _PairPageBackend(raw_hits=4)
    manager = _PairPageManager([("RemoteA", isolated), ("RemoteB", paired)])

    count, mapping = StorageManager.batched_contains_two_group_layer_pages(
        manager,
        group0,
        group1,
        pin=True,
    )

    assert count == 2
    assert list(mapping) == ["RemoteB"]
    assert [key.kv_group for key in mapping["RemoteB"]] == [0, 1, 0, 1]
    assert [pin for _, pin in isolated.calls] == [False]
    assert [pin for _, pin in paired.calls] == [False, True]
