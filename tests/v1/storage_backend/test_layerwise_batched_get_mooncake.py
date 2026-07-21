# SPDX-License-Identifier: Apache-2.0
"""
Tests for layerwise_batched_get blocking path + Mooncake connector semantics.

Covers cold start (Mooncake-only data), warm path (CPU hot cache), write-back
after remote fetch, and store scenarios (CPU + Mooncake).
"""

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from types import SimpleNamespace
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager


class MockMemoryObj:
    """Minimal MemoryObj stand-in with ref-count tracking."""

    def __init__(self, obj_id: int):
        self.obj_id = obj_id
        self.ref_count = 1
        self.ref_count_down_calls = 0

    def ref_count_down(self):
        self.ref_count -= 1
        self.ref_count_down_calls += 1

    def ref_count_up(self):
        self.ref_count += 1

    def get_size(self) -> int:
        return 1024


class MockMooncakeConnection:
    """Mimics MooncakestoreConnector capability flags."""

    def support_batched_get(self) -> bool:
        return True

    def support_batched_get_non_blocking(self) -> bool:
        return False


class MockLocalCPUBackend(LocalCPUBackend):
    """LocalCPUBackend stand-in tracking hot cache and API usage."""

    def __init__(self):
        # Skip LocalCPUBackend.__init__ — tests only need isinstance() + duck typing.
        self.hot_cache: dict[CacheEngineKey, MockMemoryObj] = {}
        self.batched_submit_put_task_calls: list[
            tuple[list[CacheEngineKey], list[MockMemoryObj]]
        ] = []
        self.batched_get_non_blocking_calls: list[list[CacheEngineKey]] = []

    def get_allocator_backend(self):
        return self

    def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None):
        self.batched_submit_put_task_calls.append((list(keys), list(memory_objs)))
        for key, obj in zip(keys, memory_objs, strict=False):
            self.hot_cache[key] = obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec=None,
    ) -> list[MockMemoryObj]:
        self.batched_get_non_blocking_calls.append(list(keys))
        return [self.hot_cache[k] for k in keys if k in self.hot_cache]


class MockRemoteBackend:
    """RemoteBackend stand-in for layerwise blocking get tests."""

    def __init__(
        self,
        blocking_results: list[list[MockMemoryObj | None]],
        local_cpu_backend: MockLocalCPUBackend | None = None,
    ):
        self.connection = MockMooncakeConnection()
        self.local_cpu_backend = local_cpu_backend
        self.batched_get_blocking_calls: list[list[CacheEngineKey]] = []
        self.batched_submit_put_task_calls: list[list[CacheEngineKey]] = []
        self._blocking_results = blocking_results
        self._call_idx = 0
        self.put_future: Future = Future()
        self.put_future.set_result(None)

    def get_allocator_backend(self):
        if self.local_cpu_backend is None:
            raise RuntimeError("local_cpu_backend required")
        return self.local_cpu_backend

    def requires_put_completion(self) -> bool:
        return True

    def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None):
        self.batched_submit_put_task_calls.append(list(keys))
        return [self.put_future]

    def batched_get_blocking(self, keys: list[CacheEngineKey]):
        self.batched_get_blocking_calls.append(list(keys))
        if self._call_idx >= len(self._blocking_results):
            return [None] * len(keys)
        result = self._blocking_results[self._call_idx]
        self._call_idx += 1
        return result


def _make_layer_key(layer_id: int, chunk_hash: int = 0xabc) -> LayerCacheEngineKey:
    return LayerCacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
        layer_id=layer_id,
    )


def _make_chunk_keys(
    num_layers: int, num_chunks: int = 1
) -> list[list[LayerCacheEngineKey]]:
    """Layer-major keys: keys[layer_id][chunk_idx]."""
    keys_per_layer: list[list[LayerCacheEngineKey]] = []
    for layer_id in range(num_layers):
        layer_keys = [
            _make_layer_key(layer_id, chunk_hash=0x100 + chunk_idx)
            for chunk_idx in range(num_chunks)
        ]
        keys_per_layer.append(layer_keys)
    return keys_per_layer


@pytest.fixture
def event_manager():
    return EventManager()


@pytest.fixture
def base_config():
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=1.0,
        lmcache_instance_id="test_layerwise_mooncake",
        extra_config={"layerwise_use_blocking_get": False},
    )


@pytest.fixture
def scheduler_metadata():
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
        role="scheduler",
    )


def _build_manager(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    event_manager: EventManager,
    local_cpu: MockLocalCPUBackend,
    remote: MockRemoteBackend | None = None,
) -> StorageManager:
    manager = StorageManager(
        config=config,
        metadata=metadata,
        event_manager=event_manager,
    )
    backends: OrderedDict[str, object] = OrderedDict()
    backends["LocalCPUBackend"] = local_cpu
    if remote is not None:
        remote.local_cpu_backend = local_cpu
        backends["RemoteBackend"] = remote
    manager.storage_backends = backends
    return manager


def _run_layerwise_get(
    manager: StorageManager,
    keys: list[list[LayerCacheEngineKey]],
    location: str,
) -> list[list[MockMemoryObj]]:
    layers: list[list[MockMemoryObj]] = []
    for task in manager.layerwise_batched_get(keys, location=location):
        layers.append(task.result())
    return layers


class TestMooncakeConnectorCapabilities:
    def test_support_batched_get_non_blocking_is_false(self):
        from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
            MooncakestoreConnector,
        )

        class _Conn(MooncakestoreConnector):
            def __init__(self):
                pass

        conn = object.__new__(_Conn)
        assert conn.support_batched_get() is True
        assert conn.support_batched_get_non_blocking() is False

    def test_lmcache_config_exposes_protocol_fallback(self):
        from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
            MooncakeStoreConfig,
        )

        config = SimpleNamespace(
            extra_config={
                "local_hostname": "decoder",
                "metadata_server": "127.0.0.1:8080",
                "master_server_address": "127.0.0.1:50051",
                "protocol": "rdma",
                "protocol_fallback": "tcp",
                "device_name": "mlx5_0",
            }
        )

        mooncake_config = MooncakeStoreConfig.load_from_lmcache_config(config)

        assert mooncake_config.protocol == "rdma"
        assert mooncake_config.protocol_fallback == "tcp"
        assert mooncake_config.device_name == "mlx5_0"


class TestStorageManagerLayerwiseHelpers:
    def test_normalize_layerwise_prefix_releases_after_first_none(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        remote = MockRemoteBackend([])
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )

        o0, o1, o2 = MockMemoryObj(0), MockMemoryObj(1), MockMemoryObj(2)
        normalized = manager._normalize_layerwise_prefix([o0, o1, None, o2])

        assert normalized == [o0, o1]
        assert o2.ref_count_down_calls == 1

    def test_layerwise_prefers_blocking_for_mooncake(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        remote = MockRemoteBackend([])
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )
        assert manager._layerwise_get_prefers_blocking(
            "RemoteBackend", remote
        ) is True

    def test_layerwise_prefers_async_for_local_cpu(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, None
        )
        assert manager._layerwise_get_prefers_blocking(
            "LocalCPUBackend", local_cpu
        ) is False


class TestLayerwiseBatchedGetColdPath:
    """Data exists only in remote (Mooncake) — cold start via blocking get."""

    def test_mooncake_only_single_layer(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        objs = [MockMemoryObj(10), MockMemoryObj(11)]
        remote = MockRemoteBackend([objs])
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )
        keys = _make_chunk_keys(num_layers=1, num_chunks=2)

        result = _run_layerwise_get(manager, keys, location="RemoteBackend")

        assert len(result) == 1
        assert [o.obj_id for o in result[0]] == [10, 11]
        assert len(remote.batched_get_blocking_calls) == 1
        assert len(local_cpu.batched_get_non_blocking_calls) == 0

    def test_mooncake_only_multi_layer(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        remote = MockRemoteBackend(
            [
                [MockMemoryObj(0)],
                [MockMemoryObj(1)],
                [MockMemoryObj(2)],
            ]
        )
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )
        keys = _make_chunk_keys(num_layers=3, num_chunks=1)

        result = _run_layerwise_get(manager, keys, location="RemoteBackend")

        assert len(result) == 3
        assert [o.obj_id for o in result[0]] == [0]
        assert [o.obj_id for o in result[1]] == [1]
        assert [o.obj_id for o in result[2]] == [2]
        assert len(remote.batched_get_blocking_calls) == 3


class TestLayerwiseBatchedGetWriteBack:
    """Remote cold read should populate LocalCPUBackend hot cache."""

    def test_write_back_after_mooncake_layerwise_get(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        layer_objs = [MockMemoryObj(42)]
        remote = MockRemoteBackend([layer_objs])
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )
        keys = _make_chunk_keys(num_layers=1, num_chunks=1)

        _run_layerwise_get(manager, keys, location="RemoteBackend")

        assert len(local_cpu.batched_submit_put_task_calls) == 1
        put_keys, put_objs = local_cpu.batched_submit_put_task_calls[0]
        assert put_keys == keys[0]
        assert [o.obj_id for o in put_objs] == [42]
        assert keys[0][0] in local_cpu.hot_cache

    def test_warm_path_uses_cpu_async_after_write_back(
        self, base_config, scheduler_metadata, event_manager
    ):
        """Simulate: cold remote fetch → write-back → warm CPU read."""
        local_cpu = MockLocalCPUBackend()
        remote = MockRemoteBackend([[MockMemoryObj(99)]])
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )
        keys = _make_chunk_keys(num_layers=1, num_chunks=1)

        # Cold: Mooncake only
        _run_layerwise_get(manager, keys, location="RemoteBackend")
        assert len(remote.batched_get_blocking_calls) == 1

        # Warm: CPU hot cache (async path)
        result = _run_layerwise_get(manager, keys, location="LocalCPUBackend")
        assert len(remote.batched_get_blocking_calls) == 1
        assert len(local_cpu.batched_get_non_blocking_calls) == 1
        assert [o.obj_id for o in result[0]] == [99]


class TestLayerwiseBatchedGetCpuOnly:
    """Data only in CPU hot cache — warm path without remote."""

    def test_cpu_only_uses_async_get(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        keys = _make_chunk_keys(num_layers=2, num_chunks=1)
        for layer_keys in keys:
            for key in layer_keys:
                local_cpu.hot_cache[key] = MockMemoryObj(key.layer_id)

        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, None
        )
        result = _run_layerwise_get(manager, keys, location="LocalCPUBackend")

        assert len(result) == 2
        assert [o.obj_id for o in result[0]] == [0]
        assert [o.obj_id for o in result[1]] == [1]
        assert len(local_cpu.batched_get_non_blocking_calls) == 2


class TestLayerwiseBatchedGetCpuAndMooncake:
    """CPU and Mooncake both hold data; location selects backend."""

    def test_cpu_location_skips_remote(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        remote = MockRemoteBackend([[MockMemoryObj(1)]])
        keys = _make_chunk_keys(num_layers=1, num_chunks=1)

        local_cpu.hot_cache[keys[0][0]] = MockMemoryObj(100)
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )

        result = _run_layerwise_get(manager, keys, location="LocalCPUBackend")

        assert [o.obj_id for o in result[0]] == [100]
        assert len(remote.batched_get_blocking_calls) == 0
        assert len(local_cpu.batched_get_non_blocking_calls) == 1

    def test_remote_location_reads_mooncake_not_cpu(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        remote = MockRemoteBackend([[MockMemoryObj(7)]])
        keys = _make_chunk_keys(num_layers=1, num_chunks=1)

        local_cpu.hot_cache[keys[0][0]] = MockMemoryObj(100)
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )

        result = _run_layerwise_get(manager, keys, location="RemoteBackend")

        assert [o.obj_id for o in result[0]] == [7]
        assert len(remote.batched_get_blocking_calls) == 1


class TestBatchedPutCpuAndMooncake:
    """Store path: batched_put writes to allocator (CPU) and remote."""

    def test_batched_put_targets_cpu_and_remote(
        self, base_config, scheduler_metadata, event_manager
    ):
        local_cpu = MockLocalCPUBackend()
        remote = MockRemoteBackend([], local_cpu_backend=local_cpu)
        manager = _build_manager(
            base_config, scheduler_metadata, event_manager, local_cpu, remote
        )
        manager.allocator_backend = local_cpu

        keys = [_make_layer_key(0)]
        objs = [MockMemoryObj(1)]

        futures = manager.batched_put(keys, objs)

        assert len(local_cpu.batched_submit_put_task_calls) == 1
        assert local_cpu.batched_submit_put_task_calls[0][0] == keys
        assert len(remote.batched_submit_put_task_calls) == 1
        assert remote.batched_submit_put_task_calls[0] == keys
        assert futures == [remote.put_future]
        assert objs[0].ref_count_down_calls == 1


class TestMooncakeConnectorBatchedGetNonBlockingFallback:
    def test_delegates_to_batched_get_with_prefix_semantics(self):
        from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
            MooncakestoreConnector,
        )

        class _Conn(MooncakestoreConnector):
            async def batched_get(self, keys):
                return [
                    MockMemoryObj(1),
                    MockMemoryObj(2),
                    None,
                    MockMemoryObj(3),
                ]

        conn = object.__new__(_Conn)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                conn.batched_get_non_blocking("lid", [_make_layer_key(0)])
            )
        finally:
            loop.close()

        assert len(result) == 2
        assert result[0].obj_id == 1
        assert result[1].obj_id == 2


class TestMooncakeConnectorBatchedContains:
    def test_uses_batch_is_exist_with_prefix_semantics(self):
        from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
            MooncakestoreConnector,
        )

        class _Store:
            def __init__(self):
                self.keys = []

            def batch_is_exist(self, keys):
                self.keys = list(keys)
                return [1, 1, 0, 1]

        conn = object.__new__(MooncakestoreConnector)
        conn.store = _Store()
        keys = [_make_layer_key(layer_id) for layer_id in range(4)]

        conn.config = SimpleNamespace(experimental_sampled_layerwise_lookup=False)
        assert not conn.support_batched_contains()

        conn.config.experimental_sampled_layerwise_lookup = True
        assert conn.support_batched_contains()
        assert conn.batched_contains(keys) == 2
        assert conn.store.keys == [key.to_string() for key in keys]
