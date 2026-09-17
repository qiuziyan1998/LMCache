# SPDX-License-Identifier: Apache-2.0
"""Per-group cardinality tests for DSA two-group (shared-indexer) models.

Covers the GLM-5.2 contract (79 LATENT / 22 INDEXER):
- Runtime and registered cardinality resolution with fail-closed behavior.
- LMCacheEngine.num_layers_for_group over registered/runtime sources.
- store_layer transfers exactly the group's layer rows (allocation batch
  size, key range 0..N-1, generator cadence) and fail-closes against a
  mismatched kvcaches list.
- retrieve_layer advances exactly the group's layer rows.
- lookup exact completeness checks each group over its own key range.
"""

# Standard
from types import SimpleNamespace
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1 import cache_engine as cache_engine_module
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.kv_layer_groups import (
    KVLayerGroupInfo,
    KVLayerGroupsManager,
    resolve_kv_group_num_layers,
)
from lmcache.v1.metadata import LMCacheMetadata


def _group(layer_names, shape):
    return KVLayerGroupInfo(
        layer_names=list(layer_names),
        layer_indices=list(range(len(layer_names))),
        shape=torch.Size(shape),
        dtype=torch.bfloat16,
    )


def _two_group_manager(latent_layers: int, indexer_layers: int):
    return KVLayerGroupsManager(
        kv_layer_groups=[
            _group(
                [f"latent-{i}" for i in range(latent_layers)],
                [1, latent_layers, 256, 512],
            ),
            _group(
                [f"indexer-{i}" for i in range(indexer_layers)],
                [1, indexer_layers, 256, 128],
            ),
        ]
    )


class TestResolveKvGroupNumLayers:
    def test_non_dsa_returns_model_count(self):
        assert (
            resolve_kv_group_num_layers(
                kv_group=0,
                dsa_two_groups=False,
                model_num_layers=79,
                registered_groups=[],
            )
            == 79
        )

    def test_registered_groups_win(self):
        groups = _two_group_manager(79, 22).kv_layer_groups
        assert (
            resolve_kv_group_num_layers(
                kv_group=0,
                dsa_two_groups=True,
                model_num_layers=79,
                registered_groups=groups,
                runtime=(79, 22),
            )
            == 79
        )
        assert (
            resolve_kv_group_num_layers(
                kv_group=1,
                dsa_two_groups=True,
                model_num_layers=79,
                registered_groups=groups,
                runtime=(79, 22),
            )
            == 22
        )

    def test_equal_runtime_groups_support_glm51(self):
        assert (
            resolve_kv_group_num_layers(
                kv_group=1,
                dsa_two_groups=True,
                model_num_layers=79,
                registered_groups=[],
                runtime=(79, 79),
            )
            == 79
        )

    def test_runtime_metadata_serves_scheduler(self):
        assert (
            resolve_kv_group_num_layers(
                kv_group=1,
                dsa_two_groups=True,
                model_num_layers=79,
                registered_groups=[],
                runtime=(79, 22),
            )
            == 22
        )

    def test_registered_and_runtime_disagree_fail_closed(self):
        groups = _two_group_manager(79, 22).kv_layer_groups
        with pytest.raises(ValueError, match="KV group metadata"):
            resolve_kv_group_num_layers(
                kv_group=1,
                dsa_two_groups=True,
                model_num_layers=79,
                registered_groups=groups,
                runtime=(79, 79),
            )

    @pytest.mark.parametrize("runtime", [None, (79,), (79, 0), (79, 22, 1)])
    def test_missing_or_invalid_runtime_fails_closed(self, runtime):
        with pytest.raises(ValueError, match="runtime"):
            resolve_kv_group_num_layers(
                kv_group=1,
                dsa_two_groups=True,
                model_num_layers=79,
                registered_groups=[],
                runtime=runtime,
            )

    def test_missing_runtime_rejects_registered_groups(self):
        with pytest.raises(ValueError, match="runtime"):
            resolve_kv_group_num_layers(
                kv_group=1,
                dsa_two_groups=True,
                model_num_layers=79,
                registered_groups=_two_group_manager(79, 22).kv_layer_groups,
            )

    def test_out_of_range_fail_closed(self):
        with pytest.raises(ValueError, match="out of range"):
            resolve_kv_group_num_layers(
                kv_group=2,
                dsa_two_groups=True,
                model_num_layers=79,
                registered_groups=[],
                runtime=(79, 22),
            )


def _make_metadata(
    manager: Optional[KVLayerGroupsManager],
    num_layers: int = 79,
    runtime: Optional[tuple[int, ...]] = None,
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="glm-5.2-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 1, 256, 1, 512),
        use_mla=True,
        chunk_size=256,
        kv_layer_groups_manager=manager or KVLayerGroupsManager(),
        runtime_kv_group_layer_counts=runtime,
    )


def _engine_for_cardinality(
    manager: Optional[KVLayerGroupsManager],
    runtime: Optional[tuple[int, ...]],
) -> LMCacheEngine:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 79
    engine.dsa_two_groups = True
    engine.metadata = _make_metadata(manager, runtime=runtime)
    return engine


class TestEngineNumLayersForGroup:
    @pytest.mark.parametrize("runtime", [None, (79,)])
    def test_init_rejects_invalid_runtime_before_resource_setup(
        self,
        monkeypatch,
        runtime,
    ):
        side_effects = []

        def record_side_effect(*_args, **_kwargs):
            side_effects.append(True)

        monkeypatch.setattr(
            LMCacheEngine,
            "_validate_shared_cpu_cache_contract",
            record_side_effect,
        )
        monkeypatch.setattr(
            LMCacheEngine,
            "_prepare_shared_cpu_cache_name",
            record_side_effect,
        )
        monkeypatch.setattr(
            cache_engine_module.multiprocessing,
            "set_start_method",
            record_side_effect,
        )
        monkeypatch.setattr(
            cache_engine_module.socket,
            "gethostname",
            record_side_effect,
        )
        monkeypatch.setattr(
            cache_engine_module.threading,
            "Condition",
            record_side_effect,
        )

        with pytest.raises(ValueError, match="runtime"):
            LMCacheEngine(
                SimpleNamespace(dsa_two_groups=True),
                _make_metadata(None, runtime=runtime),
                SimpleNamespace(),
                None,
                lambda *_args: None,
                lambda *_args: None,
            )

        assert side_effects == []

    def test_registered_two_groups(self):
        engine = _engine_for_cardinality(_two_group_manager(79, 22), (79, 22))
        assert engine.num_layers_for_group(0) == 79
        assert engine.num_layers_for_group(1) == 22

    def test_runtime_metadata_scheduler_source(self):
        engine = _engine_for_cardinality(None, (79, 22))
        assert engine.num_layers_for_group(0) == 79
        assert engine.num_layers_for_group(1) == 22

    def test_missing_runtime_metadata_fails_closed(self):
        engine = _engine_for_cardinality(None, None)
        with pytest.raises(ValueError, match="runtime"):
            engine.num_layers_for_group(1)

    def test_num_transfer_layers_validates_kvcaches(self):
        engine = _engine_for_cardinality(_two_group_manager(79, 22), (79, 22))
        assert (
            engine._num_transfer_layers_for_call(
                1, {"kvcaches": [object()] * 22}
            )
            == 22
        )
        with pytest.raises(ValueError, match="cardinality mismatch"):
            engine._num_transfer_layers_for_call(
                1, {"kvcaches": [object()] * 79}
            )


class _FakeLayerKey:
    """Chunk key recording the cardinality each split_layers call used."""

    def __init__(
        self,
        chunk_id: int,
        layer_id: Optional[int],
        deps=None,
        kv_group: int = 0,
    ):
        self.chunk_id = chunk_id
        self.layer_id = layer_id
        self.deps = deps
        self.kv_group = kv_group
        self.chunk_hash = chunk_id
        self.tags = ()

    def split_layers(self, num_layers: int):
        if self.deps is not None:
            self.deps.split_sizes.setdefault(self.kv_group, []).append(
                num_layers
            )
        return [
            _FakeLayerKey(self.chunk_id, i, self.deps, kv_group=self.kv_group)
            for i in range(num_layers)
        ]


class _FakeStoreDeps:
    """Minimal collaborators to drive store_layer/retrieve_layer."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.allocate_batch_sizes: list[int] = []
        self.put_keys: list[list] = []
        self.stored_keys_by_layer: dict[int, list] = {}
        self.split_sizes: dict[int, list[int]] = {}

    def make_key(self, chunk_id: int) -> _FakeLayerKey:
        return _FakeLayerKey(chunk_id, None, deps=self)


def _store_layer_engine(kv_group: int, group_layers: int):
    deps = _FakeStoreDeps(group_layers)
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 79
    engine.dsa_two_groups = True
    engine.metadata = _make_metadata(
        _two_group_manager(79, 22), runtime=(79, 22)
    )
    engine.kv_events_enabled = False
    engine.is_healthy = lambda: True
    engine._is_passive = lambda: False
    engine.is_frozen = lambda: False
    engine._get_req_id = lambda _kwargs: "req"
    engine._log_kvcache_for_check = lambda **_kwargs: None
    engine.stats_monitor = SimpleNamespace(
        on_store_request=lambda _n: "monitor-id",
        on_store_finished=lambda _m, _n: None,
        on_retrieve_request=lambda _n: "monitor-id",
        on_retrieve_finished=lambda _m, _n: None,
    )
    engine._layerwise_chunk_fully_stored = lambda *a, **k: False
    engine._shared_cpu_dtype_for_kv_group = lambda _g: torch.bfloat16
    engine._memory_format_for_kv_group = lambda _g: None
    engine.store_location = "LocalCPUBackend"
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: iter(
            [(0, 4, deps.make_key(0))]
        )
    )
    engine.config = SimpleNamespace(
        get_extra_config_value=lambda _k, default=None: default,
        chunk_size=256,
        dsa_two_groups=True,
        dsa_group1_load_mode="cpu_staged",
    )

    mem_obj = SimpleNamespace(
        get_size=lambda: 8,
        is_valid=lambda: True,
        ref_count_down=lambda: None,
    )

    class FakeStorageManager:
        @staticmethod
        def batched_allocate(_shape, _dtype, batch_size=None, **_kwargs):
            deps.allocate_batch_sizes.append(batch_size)
            return [mem_obj for _ in range(batch_size)]

        @staticmethod
        def batched_put(keys, _objs, location=None):
            engine_dep = deps
            for key in keys:
                engine_dep.stored_keys_by_layer.setdefault(
                    key.layer_id, []
                ).append(key)
            deps.put_keys.append(keys)

        @staticmethod
        def contains(_key, _locations):
            return "LocalCPUBackend"

        @staticmethod
        def batched_contains(keys, _locations, _pin):
            return len(keys), {}

    class FakeGPUConnector:
        @staticmethod
        def get_shape(num_tokens: int, kv_group: int = 0):
            layers = engine.num_layers_for_group(kv_group)
            return torch.Size([1, layers, num_tokens, 512])

        @staticmethod
        def batched_from_gpu(_objs, _starts, _ends, **_kwargs):
            def gen():
                yield
                for _ in range(group_layers):
                    yield

            return gen()

        @staticmethod
        def batched_to_gpu(_starts, _ends, **_kwargs):
            def consume():
                yield
                for _ in range(group_layers):
                    yield
                yield

            return consume()

    engine.storage_manager = FakeStorageManager()
    engine.gpu_connector = FakeGPUConnector()
    return engine, deps


class TestStoreLayerPerGroup:
    def test_indexer_group_transfers_22_rows(self, monkeypatch):
        monkeypatch.setattr(
            cache_engine_module, "assert_layerwise_gpu_connector", lambda _c: None
        )
        engine, deps = _store_layer_engine(kv_group=1, group_layers=22)
        monkeypatch.setattr(
            cache_engine_module, "CacheEngineKey", _FakeLayerKey
        )
        results = list(
            engine.store_layer(
                [1, 2, 3, 4],
                kv_group=1,
                req_id="req",
                kvcaches=[object()] * 22,
            )
        )
        # Cadence: 22 layer yields + 1 final store-result yield.
        assert len(results) == 23
        assert results[-1] is not None
        assert deps.allocate_batch_sizes == [22]
        stored_layers = sorted(deps.stored_keys_by_layer)
        assert stored_layers == list(range(22))

    def test_kvcaches_mismatch_fails_closed(self, monkeypatch):
        engine, deps = _store_layer_engine(kv_group=1, group_layers=22)
        monkeypatch.setattr(
            cache_engine_module, "CacheEngineKey", _FakeLayerKey
        )
        with pytest.raises(ValueError, match="cardinality mismatch"):
            list(
                engine.store_layer(
                    [1, 2, 3, 4],
                    kv_group=1,
                    req_id="req",
                    kvcaches=[object()] * 79,
                )
            )


class TestRetrieveLayerPerGroup:
    def test_indexer_group_advances_22_rows(self, monkeypatch):
        monkeypatch.setattr(
            cache_engine_module, "assert_layerwise_gpu_connector", lambda _c: None
        )
        engine, deps = _store_layer_engine(kv_group=1, group_layers=22)
        monkeypatch.setattr(
            cache_engine_module, "CacheEngineKey", _FakeLayerKey
        )

        class FakeStorageManager(engine.storage_manager.__class__):
            def __init__(self2):
                self2.layerwise_calls = 0

            def contains(self2, _key, _locations):
                return "LocalCPUBackend"

            def batched_contains(self2, keys, _locations, _pin):
                return len(keys), {}

            def layerwise_batched_get(self2, keys, location=None):
                self2.layerwise_calls += 1
                for layer_keys in keys:
                    mem_objs = [
                        SimpleNamespace(
                            is_valid=lambda: True,
                            ref_count_down=lambda: None,
                        )
                        for _ in layer_keys
                    ]
                    yield SimpleNamespace(
                        result=lambda objs=mem_objs: objs
                    )

        storage = FakeStorageManager()
        engine.storage_manager = storage
        engine._should_use_shared_layerwise_retrieve = lambda _g: False
        engine._is_passive = lambda: False
        engine._maybe_unpin_retrieved_objs = lambda _objs, _loc: None
        # token_database yields one chunk whose split must cover 22 rows.
        engine.token_database = SimpleNamespace(
            process_tokens=lambda **_kwargs: iter([(0, 4, deps.make_key(0))])
        )

        results = list(
            engine.retrieve_layer(
                [1, 2, 3, 4],
                kv_group=1,
                req_id="req",
                kvcaches=[object()] * 22,
            )
        )
        # Yield cadence for 22 rows: sum, 21 x None, trailing None, mask.
        assert len(results) == 24
        assert results[0] is not None
        assert int(results[0]) == 4
        assert results[-1].tolist() == [True, True, True, True]
        # layerwise_batched_get received exactly 22 layer rows.
        assert storage.layerwise_calls == 1
        assert deps.num_layers == 22


class TestLookupExactCompleteness:
    def test_indexer_keys_capped_at_group_layers(self, monkeypatch):
        engine, deps = _store_layer_engine(kv_group=1, group_layers=22)
        monkeypatch.setattr(
            cache_engine_module, "CacheEngineKey", _FakeLayerKey
        )
        contains_sizes: dict[int, int] = {}

        class FakeStorageManager:
            @staticmethod
            def batched_contains(keys, _locations, _pin):
                grouped: dict[int, int] = {}
                for key in keys:
                    grouped[int(getattr(key, "kv_group", 0))] = (
                        grouped.get(int(getattr(key, "kv_group", 0)), 0) + 1
                    )
                for group, count in grouped.items():
                    contains_sizes[group] = count
                return len(keys), {"LocalCPUBackend": list(keys)}

        engine.storage_manager = FakeStorageManager()
        engine.use_layerwise = True
        engine.is_healthy = lambda: True
        engine.stats_monitor = SimpleNamespace(
            on_lookup_request=lambda _n: None,
            on_lookup_finished=lambda _stats, _res: None,
        )
        engine.lookup_pins = {}
        engine._layerwise_lookup_kv_groups = lambda: [0, 1]
        chunk_keys = {
            0: [deps.make_key(0)],
            1: [deps.make_key(1)],
        }

        class FakeTokenDatabase:
            def process_tokens(self, **_kwargs):
                yield (0, 4, chunk_keys[0][0])
                yield (4, 8, chunk_keys[1][0])

            def _make_key_by_hash(self, _hash, _configs, kv_group=0, valid_tokens=None):
                return _FakeLayerKey(0, None, deps=deps, kv_group=kv_group)

        engine.token_database = FakeTokenDatabase()
        engine._sampled_scheduler_lookup = lambda *a, **k: 0

        monkeypatch.setattr(
            cache_engine_module, "mooncake_layer_pages_enabled", lambda _c: False
        )
        result = engine.lookup(tokens=[1, 2, 3, 4, 5, 6, 7, 8])
        assert result == 8
        # Each group was checked over its own cardinality, not 79.
        assert contains_sizes == {0: 79, 1: 22}
        assert deps.split_sizes == {0: [79, 79], 1: [22, 22]}


class TestKeyRange:
    def test_split_layers_never_exceeds_group_cardinality(self):
        engine, deps = _store_layer_engine(kv_group=1, group_layers=22)
        key = deps.make_key(0)
        split = key.split_layers(engine.num_layers_for_group(1))
        layer_ids = [k.layer_id for k in split]
        assert layer_ids == list(range(22))
        assert max(layer_ids) < 22


def test_cache_engine_key_layer_range():
    """LayerCacheEngineKey must produce group-local ids below 22 for GLM-5.2."""
    key = CacheEngineKey(
        model_name="m",
        world_size=1,
        worker_id=0,
        chunk_hash=123,
        dtype=torch.bfloat16,
    )
    split = key.split_layers(22)
    assert len(split) == 22
    assert [k.layer_id for k in split] == list(range(22))
