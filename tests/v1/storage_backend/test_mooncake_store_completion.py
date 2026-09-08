# SPDX-License-Identifier: Apache-2.0
"""Mooncake-specific remote store completion semantics."""

# Standard
from collections import OrderedDict
from collections.abc import Coroutine, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace
from unittest.mock import Mock
import asyncio
import ctypes
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.kv_layer_groups import (
    KVLayerGroupInfo,
    KVLayerGroupsManager,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, TensorMemoryAllocator
from lmcache.v1.storage_backend.connector import (
    mooncakestore_connector as mooncake_connector,
)
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager


class _MemoryObj:
    def __init__(self, size: int = 16, data_ptr: int = 123) -> None:
        self.ref_count = 1
        self.raw_tensor = object()
        self.data_ptr = data_ptr
        self.size = size

    def ref_count_up(self) -> None:
        self.ref_count += 1

    def ref_count_down(self) -> None:
        self.ref_count -= 1

    def get_size(self) -> int:
        return self.size

    def is_valid(self) -> bool:
        return self.ref_count > 0


class _Serializer:
    @staticmethod
    def serialize(memory_obj: _MemoryObj) -> _MemoryObj:
        memory_obj.ref_count_up()
        return memory_obj


class _Connection:
    def __init__(self, requires_completion: bool) -> None:
        self._requires_completion = requires_completion

    @staticmethod
    def support_batched_put() -> bool:
        return True

    def requires_put_completion(self) -> bool:
        return self._requires_completion

    @staticmethod
    async def batched_put(keys, memory_objs) -> None:
        return None


def _key(chunk_hash: int, kv_group: int = 0) -> CacheEngineKey:
    return CacheEngineKey("test", 1, 0, chunk_hash, torch.float16, kv_group=kv_group)


def _layer_key(
    chunk_hash: int, layer_id: int, kv_group: int = 0
) -> LayerCacheEngineKey:
    return LayerCacheEngineKey(
        "test",
        1,
        0,
        chunk_hash,
        torch.float16,
        layer_id=layer_id,
        kv_group=kv_group,
    )


def _group_manager(*num_layers: int) -> KVLayerGroupsManager:
    return KVLayerGroupsManager(
        kv_layer_groups=[
            KVLayerGroupInfo(
                layer_names=[f"group-{group}-layer-{layer}" for layer in range(size)],
                layer_indices=list(range(size)),
                shape=torch.Size([1, 4, 8]),
                dtype=torch.float16,
            )
            for group, size in enumerate(num_layers)
        ]
    )


def _configure_page_cardinality(
    connector: MooncakestoreConnector,
    model_num_layers: int,
    *,
    dsa_two_groups: bool = False,
    runtime: tuple[int, ...] | None = None,
    manager: KVLayerGroupsManager | None = None,
) -> None:
    backend = getattr(connector, "local_cpu_backend", SimpleNamespace())
    config = getattr(backend, "config", SimpleNamespace())
    config.dsa_two_groups = dsa_two_groups
    config.extra_config = {}
    metadata = getattr(backend, "metadata", SimpleNamespace())
    metadata.kv_shape = (model_num_layers,)
    metadata.kv_layer_groups_manager = manager or KVLayerGroupsManager()
    metadata.runtime_kv_group_layer_counts = runtime
    backend.config = config
    backend.metadata = metadata
    connector.local_cpu_backend = backend


def _make_remote_backend(requires_completion: bool) -> RemoteBackend:
    backend = object.__new__(RemoteBackend)
    backend.connection = _Connection(requires_completion)
    backend.local_cpu_backend = None
    backend.loop = object()
    backend.serializer = _Serializer()
    backend._mla_worker_id_as0_mode = False
    backend.put_tasks = set()
    backend.lock = threading.Lock()
    backend._inflight_gets = set()
    backend._closing = False
    return backend


def test_layer_page_timeout_releases_late_result(monkeypatch) -> None:
    page = _MemoryObj()

    class _Connection:
        @staticmethod
        async def batched_get_layer_pages(keys):
            return [page]

    class _LateFuture:
        callback = None
        complete = False

        def result(self, timeout=None):
            if not self.complete:
                raise TimeoutError
            return [page]

        def add_done_callback(self, callback):
            self.callback = callback

        def cancel(self):
            raise AssertionError("timed-out transfer must finish for safe cleanup")

    late = _LateFuture()

    def submit(coroutine, loop):
        coroutine.close()
        return late

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    backend = _make_remote_backend(True)
    backend.connection = _Connection()
    backend.loop = object()
    backend.config = SimpleNamespace(blocking_timeout_secs=0.01)
    backend._mla_worker_id_as0_mode = False

    with pytest.raises(TimeoutError):
        backend.batched_get_layer_pages([_layer_key(1, 0)])
    assert page.ref_count == 1
    late.complete = True
    assert late.callback is not None
    late.callback(late)
    assert page.ref_count == 0


def test_remote_backend_returns_only_required_completion(monkeypatch) -> None:
    source_futures: list[Future] = []

    def submit(coroutine, loop) -> Future:
        coroutine.close()
        future: Future = Future()
        source_futures.append(future)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    mooncake_future = _make_remote_backend(True).batched_submit_put_task(
        [_key(1)], [_MemoryObj()]
    )
    other_future = _make_remote_backend(False).batched_submit_put_task(
        [_key(2)], [_MemoryObj()]
    )

    assert mooncake_future == [source_futures[0]]
    assert other_future is None
    for future in source_futures:
        future.set_result(None)


@pytest.mark.parametrize("requires_completion", [False, True])
def test_instrumented_put_preserves_connector_failure_policy(
    requires_completion: bool,
) -> None:
    class _FailingConnector:
        @staticmethod
        async def batched_put(keys, memory_objs) -> None:
            raise RuntimeError("write failed")

        @staticmethod
        def requires_put_completion() -> bool:
            return requires_completion

    connector = object.__new__(InstrumentedRemoteConnector)
    connector._connector = _FailingConnector()
    connector._stats_monitor = SimpleNamespace(
        update_interval_remote_time_to_put=lambda value: None,
        update_interval_remote_write_metrics=lambda value: None,
    )
    connector.name = "test"
    memory_obj = _MemoryObj()
    operation = connector.batched_put([_key(1)], [memory_obj])

    if requires_completion:
        with pytest.raises(RuntimeError, match="write failed"):
            asyncio.run(operation)
    else:
        asyncio.run(operation)
    assert memory_obj.ref_count == 0


def test_instrumented_connector_delegates_layer_page_operations() -> None:
    keys = [_layer_key(1, 0)]
    pages = [_MemoryObj()]

    class _PageConnector:
        @staticmethod
        def batched_contains_layer_pages(
            actual_keys: list[LayerCacheEngineKey],
        ) -> int:
            assert actual_keys == keys
            return 1

        @staticmethod
        async def batched_get_layer_pages(
            actual_keys: list[LayerCacheEngineKey],
        ) -> list[_MemoryObj]:
            assert actual_keys == keys
            return pages

    connector = object.__new__(InstrumentedRemoteConnector)
    connector._connector = _PageConnector()

    assert connector.batched_contains_layer_pages(keys) == 1
    assert asyncio.run(connector.batched_get_layer_pages(keys)) == pages


def test_mooncake_requires_put_completion() -> None:
    connector = object.__new__(MooncakestoreConnector)
    assert connector.requires_put_completion()


def test_mooncake_direct_pages_use_existing_page_keys() -> None:
    calls = []

    class _Store:
        @staticmethod
        def register_buffer(ptr, size):
            calls.append(("register", ptr, size))
            return 0

        @staticmethod
        def batch_put_from_multi_buffers(keys, ptrs, sizes, replica):
            calls.append(("put", keys, ptrs, sizes))
            return [0] * len(keys)

        @staticmethod
        def batch_is_exist(keys):
            calls.append(("exists", keys))
            return [1] * len(keys)

    class _Event:
        waited = False

        def synchronize(self):
            self.waited = True

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    _configure_page_cardinality(
        connector,
        3,
        dsa_two_groups=True,
        runtime=(3, 2),
        manager=_group_manager(3, 2),
    )
    connector._external_put_lock = asyncio.Lock()
    connector._external_buffers = {}
    connector._inflight_put_tasks = set()
    connector.store = _Store()
    connector.replica_config = object()
    connector.config = SimpleNamespace(transfer_timeout=5)
    owner = torch.empty(16, dtype=torch.uint8)
    event = _Event()
    page_key = _key(7, kv_group=1)

    asyncio.run(
        connector.batched_put_external_pages(
            [page_key],
            [[owner.data_ptr()]],
            [[owner.numel()]],
            (owner,),
            event,
            "request",
        )
    )

    assert event.waited
    put = next(call for call in calls if call[0] == "put")
    assert put[1] == ["__lmcache_page_v1__@2@test@1@0@7@half@1"]
    assert put[2:] == ([[owner.data_ptr()]], [[owner.numel()]])
    layer_key = _key(8, kv_group=1).get_layer(1)
    asyncio.run(
        connector.batched_put_external_pages(
            [layer_key],
            [[owner.data_ptr()]],
            [[owner.numel()]],
            (owner,),
            event,
            "request",
        )
    )
    assert [call for call in calls if call[0] == "put"][-1][1] == [
        layer_key.to_string()
    ]
    assert connector.batched_external_pages_exist([page_key]) == [True]
    exists = next(call for call in calls if call[0] == "exists")
    assert exists[1] == ["__lmcache_page_v1__@2@test@1@0@7@half@1"]


def test_instrumented_connector_delegates_direct_pages() -> None:
    recorded = []

    class _DirectConnector:
        @staticmethod
        async def batched_put_external_pages(*args) -> None:
            recorded.append(args)

    connector = object.__new__(InstrumentedRemoteConnector)
    connector._connector = _DirectConnector()
    connector._stats_monitor = SimpleNamespace(
        update_interval_remote_time_to_put=lambda value: None,
        update_interval_remote_write_metrics=lambda value: None,
    )
    asyncio.run(
        connector.batched_put_external_pages(
            [_key(1)], [[1]], [[2]], (), None, "request"
        )
    )
    assert recorded and recorded[0][-1] == "request"


def test_mooncake_zero_copy_metadata_reuses_homogeneous_group() -> None:
    def metadata_for_key(
        key: CacheEngineKey,
    ) -> tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int]:
        fmt = (
            MemoryFormat.KV_DSA_INDEX_FMT
            if key.kv_group == 1
            else MemoryFormat.KV_MLA_LATENT_FMT
        )
        return ([torch.Size([4])], [torch.float16], fmt, 8)

    metadata = Mock(side_effect=metadata_for_key)
    backend = SimpleNamespace(
        batched_allocate=Mock(
            side_effect=lambda *args, batch_size, **kwargs: [
                _MemoryObj() for _ in range(batch_size)
            ]
        ),
        allocate=Mock(side_effect=lambda *args: _MemoryObj()),
    )
    connector = object.__new__(MooncakestoreConnector)
    connector._metadata_for_raw_key = metadata
    connector.local_cpu_backend = backend
    connector._page_first_multi_buffer = True

    keys = [_layer_key(1, layer) for layer in range(3)]
    memory_objs, key_metadata, mode = connector._allocate_zero_copy_buffers(keys)

    assert metadata.call_count == 1
    assert all(value is key_metadata[0] for value in key_metadata)
    assert len(memory_objs) == len(keys)
    assert mode == "batched"
    assert backend.batched_allocate.call_args.kwargs["address_backed"] is True

    metadata.reset_mock()
    backend.batched_allocate.reset_mock()
    backend.allocate.reset_mock()
    keys[1].kv_group = 1
    _, key_metadata, mode = connector._allocate_zero_copy_buffers(keys)

    assert metadata.call_count == len(keys)
    assert backend.batched_allocate.call_count == 0
    assert backend.allocate.call_count == len(keys)
    assert [value[2] for value in key_metadata] == [
        MemoryFormat.KV_MLA_LATENT_FMT,
        MemoryFormat.KV_DSA_INDEX_FMT,
        MemoryFormat.KV_MLA_LATENT_FMT,
    ]
    assert mode == "individual"


def test_mooncake_batch_status_failure_is_not_silenced() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector.store = SimpleNamespace(batch_put_from=lambda *args: [0, -1])

    with pytest.raises(RuntimeError, match="batch_put_from failed"):
        asyncio.run(
            connector._batched_put_zero_copy(
                [_key(1), _key(2)], [_MemoryObj(), _MemoryObj()]
            )
        )


def test_mooncake_zero_copy_put_does_not_require_tensor_view() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector._page_first_multi_buffer = False
    connector.save_chunk_meta = False
    connector.store = SimpleNamespace(batch_put_from=lambda *args: [0])
    memory_obj = _MemoryObj()
    memory_obj.has_tensor_storage = True
    del memory_obj.raw_tensor

    asyncio.run(connector.batched_put([_key(1)], [memory_obj]))

    assert memory_obj.ref_count == 1


def test_mooncake_page_get_scatter_returns_layer_objects() -> None:
    class _PageStore:
        def batch_get_into_multi_buffers(self, page_keys, ptrs, sizes):
            assert page_keys == ["page-1", "page-2"]
            assert ptrs == [[100, 200], [300, 400]]
            assert sizes == [[16, 16], [16, 16]]
            return [32, 0]

    connector = object.__new__(MooncakestoreConnector)
    connector.store = _PageStore()
    memory_objs = [_MemoryObj(16, address) for address in (100, 200, 300, 400)]
    allocated = list(memory_objs)
    connector._allocate_zero_copy_buffers = lambda _keys: (
        memory_objs,
        [],
        "batched",
    )
    keys = [
        _layer_key(1, 0),
        _layer_key(2, 0),
        _layer_key(1, 1),
        _layer_key(2, 1),
    ]
    expected = [memory_objs[0], None, memory_objs[1], None]

    loaded = asyncio.run(
        connector._batch_get_pages(
            keys,
            [("page-1", [0, 2]), ("page-2", [1, 3])],
        )
    )

    assert loaded == expected
    assert [memory_obj.ref_count for memory_obj in allocated] == [1, 1, 0, 0]


class _BlockingGetStore:
    def __init__(self, allocations: list[MemoryObj]) -> None:
        self.allocations = allocations
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.block_pages = True
        self.outcome = "success"
        self.owned_at_exit: list[bool] = []

    @staticmethod
    def batch_is_exist(keys: list[str]) -> list[int]:
        return [1] * len(keys)

    def batch_get_into_multi_buffers(
        self, keys: list[str], ptrs: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        block = self.block and self.block_pages
        self._read(
            [ptr for page in ptrs for ptr in page],
            [size for page in sizes for size in page],
            block,
        )
        return [-1 if block and self.outcome == "miss" else sum(page) for page in sizes]

    def batch_get_into(
        self, keys: list[str], ptrs: list[int], sizes: list[int]
    ) -> list[int]:
        block = self.block and not self.block_pages
        self._read(ptrs, sizes, block)
        return [-1 if block and self.outcome == "miss" else size for size in sizes]

    def _read(self, ptrs: list[int], sizes: list[int], block: bool) -> None:
        destinations = [obj for obj in self.allocations if obj.data_ptr in ptrs]
        assert len(destinations) == len(ptrs)
        if block:
            self.entered.set()
            assert self.release.wait(5), "native read was not released"
        owned = all(obj.is_valid() and obj.get_ref_count() == 1 for obj in destinations)
        self.owned_at_exit.append(owned)
        assert owned, "native write outlived destination ownership"
        for ptr, size in zip(ptrs, sizes, strict=True):
            ctypes.memset(ptr, 7, size)
        if block and self.outcome == "error":
            raise RuntimeError("late native read failure")
        if block and self.outcome == "cancel":
            raise asyncio.CancelledError


@pytest.fixture
def page_get_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[MooncakestoreConnector]:
    """Exercise public reads with real allocator ownership and a native double."""
    allocator = TensorMemoryAllocator(torch.zeros(32768, dtype=torch.uint8))
    allocations: list[MemoryObj] = []
    releases: list[Mock] = []

    def allocate(
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat,
        **kwargs: object,
    ) -> list[MemoryObj]:
        objects = allocator.batched_allocate_address_backed(
            shapes, dtypes, batch_size, fmt
        )
        assert objects is not None
        for obj in objects:
            release = Mock(wraps=obj.ref_count_down)
            monkeypatch.setattr(obj, "ref_count_down", release)
            releases.append(release)
        allocations.extend(objects)
        return list(objects)

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector.meta_shapes = [torch.Size([8])]
    connector.meta_dtypes = [torch.float16]
    connector.meta_fmt = MemoryFormat.KV_MLA_LATENT_FMT
    connector.single_token_size = 4
    monkeypatch.setattr(connector, "_page_first_multi_buffer", True, raising=False)
    monkeypatch.setattr(connector, "_dsa_raw_token_dims", {}, raising=False)
    connector.__dict__.update(
        _inflight_gets=set(),
        _inflight_put_tasks=set(),
        _closing=False,
        _closed=False,
        _close_lock=asyncio.Lock(),
        _external_buffers={},
    )
    connector.registered_buffer_ptr = allocator.buffer.data_ptr()
    connector.local_cpu_backend = SimpleNamespace(
        memory_allocator=allocator,
        batched_allocate=allocate,
        allocations=allocations,
        releases=releases,
    )
    _configure_page_cardinality(connector, 2)
    connector.store = _BlockingGetStore(allocations)
    connector.store.unregister_buffer = Mock(return_value=0)
    connector.store.close = Mock()
    try:
        yield connector
    finally:
        connector.store.release.set()
        for obj in allocations:
            if obj.is_valid():
                obj.ref_count_down()


@pytest.mark.parametrize("mode", ["pages", "legacy", "mixed"])
@pytest.mark.parametrize("outcome", ["success", "miss", "error", "cancel"])
@pytest.mark.parametrize("cancel_count", [1, 2])
def test_mooncake_get_cancellation_drains_native_writes(
    page_get_connector: MooncakestoreConnector,
    mode: str,
    outcome: str,
    cancel_count: int,
) -> None:
    """Cancellation owns current buffers, not results of earlier public reads."""
    connector = page_get_connector
    backend = connector.local_cpu_backend
    store = connector.store
    keys = [_layer_key(1, layer) for layer in range(2)]
    if mode == "legacy":
        keys = keys[:1]
    elif mode == "mixed":
        keys.append(_layer_key(2, 0))
    store.block_pages = mode == "pages"

    async def run() -> None:
        earlier = await connector.batched_get([_layer_key(3, 0), _layer_key(3, 1)])
        assert all(obj is not None and obj.get_ref_count() == 1 for obj in earlier)
        store.block = True
        store.outcome = outcome
        task = asyncio.create_task(connector.batched_get(keys))
        try:
            assert await asyncio.to_thread(store.entered.wait, 5)
            assert len(backend.allocations) == len(earlier) + len(keys)
            allocated_bytes = backend.memory_allocator.total_allocated_size
            for _ in range(cancel_count):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert backend.memory_allocator.total_allocated_size == allocated_bytes
                assert all(obj.get_ref_count() == 1 for obj in backend.allocations)
                assert all(release.call_count == 0 for release in backend.releases)
        finally:
            store.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        assert all(store.owned_at_exit)
        assert [obj.get_ref_count() for obj in backend.allocations] == (
            [1, 1] + [0] * len(keys)
        )
        assert [release.call_count for release in backend.releases] == (
            [0, 0] + [1] * len(keys)
        )
        for obj in earlier:
            assert obj is not None
            obj.ref_count_down()
        assert backend.memory_allocator.total_allocated_size == 0
        assert all(release.call_count == 1 for release in backend.releases)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["pages", "legacy", "mixed"])
@pytest.mark.parametrize("outcome", ["success", "miss", "error", "cancel"])
def test_remote_get_timeout_releases_late_native_results(
    page_get_connector: MooncakestoreConnector, mode: str, outcome: str
) -> None:
    """A blocking timeout returns misses while native writes retain their buffers."""
    connector = page_get_connector
    local = connector.local_cpu_backend
    store = connector.store
    store.block = True
    store.block_pages = mode == "pages"
    store.outcome = outcome
    keys = [_layer_key(1, layer) for layer in range(2)]
    if mode == "legacy":
        keys = keys[:1]
    elif mode == "mixed":
        keys.append(_layer_key(2, 0))
    backend = _make_remote_backend(True)
    backend.connection = InstrumentedRemoteConnector(connector)
    backend.local_cpu_backend = local
    backend.config = SimpleNamespace(blocking_timeout_secs=0.05)
    backend.stats_monitor = Mock()
    backend.stats_monitor.get_current_retrieve_stats.return_value = None
    backend.deserializer = Mock()
    backend.__dict__.update(_mla_worker_id_as0_mode=False, _get_blocking_failed_count=0)

    async def run() -> None:
        backend.loop = asyncio.get_running_loop()
        task = asyncio.create_task(
            asyncio.to_thread(backend.batched_get_blocking, keys)
        )
        try:
            assert await asyncio.to_thread(store.entered.wait, 5)
            assert await asyncio.wait_for(task, 5) == [None] * len(keys)
            assert backend.get_blocking_failed_count == 1
            assert local.memory_allocator.total_allocated_size > 0
            assert all(obj.get_ref_count() == 1 for obj in local.allocations)
            assert all(release.call_count == 0 for release in local.releases)
            backend.deserializer.deserialize.assert_not_called()
        finally:
            store.release.set()

        async def wait_for_cleanup() -> None:
            while local.memory_allocator.total_allocated_size:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_for_cleanup(), 5)
        assert all(store.owned_at_exit)
        assert all(obj.get_ref_count() == 0 for obj in local.allocations)
        assert all(release.call_count == 1 for release in local.releases)

    asyncio.run(run())


def test_remote_get_timeout_releases_result_racing_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registering cleanup after the future completes must still release results."""
    memory_obj = _MemoryObj()
    future: Future = Future()
    result = future.result

    def timeout(timeout: float | None = None) -> list[_MemoryObj | None]:
        if timeout is not None:
            future.set_result([memory_obj, None])
            raise TimeoutError
        return result()

    monkeypatch.setattr(future, "result", timeout)
    cancel = Mock(wraps=future.cancel)
    monkeypatch.setattr(future, "cancel", cancel)

    def submit(coroutine: Coroutine, loop: object) -> Future:
        coroutine.close()
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    backend = _make_remote_backend(True)
    backend.connection = SimpleNamespace(
        support_batched_get=lambda: True,
        batched_get=lambda keys: asyncio.sleep(0),
    )
    backend.local_cpu_backend = object()
    backend.config = SimpleNamespace(blocking_timeout_secs=0.01)
    backend.stats_monitor = Mock()
    backend.stats_monitor.get_current_retrieve_stats.return_value = None
    backend.deserializer = Mock()
    monkeypatch.setattr(backend, "_get_blocking_failed_count", 0, raising=False)

    assert backend.batched_get_blocking([_key(1), _key(2)]) == [None, None]
    cancel.assert_not_called()
    backend.deserializer.deserialize.assert_not_called()
    assert memory_obj.ref_count == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_mooncake_close_drains_cancelled_reads_before_unregister(
    page_get_connector: MooncakestoreConnector, legacy: bool
) -> None:
    """Cancelling close must leave the read, registration and allocator live."""
    connector = page_get_connector
    local = connector.local_cpu_backend
    store = connector.store
    store.block = True
    store.block_pages = not legacy
    keys = [_layer_key(1, layer) for layer in range(1 if legacy else 2)]

    async def run() -> None:
        read = asyncio.create_task(connector.batched_get(keys))
        close = None
        try:
            assert await asyncio.to_thread(store.entered.wait, 5)
            # Executor-backed reads must survive even an all-task cancellation.
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
            close = asyncio.create_task(connector.close())
            await asyncio.sleep(0)
            assert not close.done()
            with pytest.raises(RuntimeError, match="closing"):
                await connector.batched_get(keys)
            close.cancel()
            with pytest.raises(asyncio.CancelledError):
                await close
            read.cancel()
            await asyncio.sleep(0)
            assert not read.done()
            store.unregister_buffer.assert_not_called()
            store.close.assert_not_called()
            assert all(release.call_count == 0 for release in local.releases)
        finally:
            store.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(read, 5)
            if close is not None:
                await asyncio.gather(close, return_exceptions=True)
        await connector.close()
        await connector.close()
        assert all(store.owned_at_exit)
        assert local.memory_allocator.total_allocated_size == 0
        assert all(release.call_count == 1 for release in local.releases)
        store.unregister_buffer.assert_called_once()
        store.close.assert_called_once()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["pages", "legacy", "mixed"])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
@pytest.mark.parametrize("recreate", [False, True])
def test_timeout_then_storage_close_drains_before_allocator(
    page_get_connector: MooncakestoreConnector,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    outcome: str,
    recreate: bool,
) -> None:
    """Global close and backend recreation cannot invalidate a native destination."""
    connector = page_get_connector
    local = connector.local_cpu_backend
    store = connector.store
    store.block = True
    store.block_pages = mode == "pages"
    store.outcome = outcome
    keys = [_layer_key(1, layer) for layer in range(2)]
    if mode == "legacy":
        keys = keys[:1]
    elif mode == "mixed":
        keys.append(_layer_key(2, 0))
    events: list[str] = []
    cpu = object.__new__(LocalCPUBackend)
    cpu.memory_allocator = local.memory_allocator
    cpu.batched_msg_sender = None
    cpu.clear = Mock()

    def unregister(ptr: int) -> int:
        assert all(store.owned_at_exit)
        assert all(release.call_count == 1 for release in local.releases)
        events.append("unregister")
        return 0

    def close_allocator() -> None:
        assert local.memory_allocator.total_allocated_size == 0
        assert all(release.call_count == 1 for release in local.releases)
        assert events[-1] == ("replacement" if recreate else "transport")
        events.append("allocator")

    store.unregister_buffer.side_effect = unregister
    store.close.side_effect = lambda: events.append("transport")
    monkeypatch.setattr(
        local.memory_allocator, "close", Mock(side_effect=close_allocator)
    )
    remote = _make_remote_backend(True)
    remote.connection = InstrumentedRemoteConnector(connector)
    remote.local_cpu_backend = cpu
    remote.config = SimpleNamespace(blocking_timeout_secs=0.05)
    remote.stats_monitor = Mock()
    remote.stats_monitor.get_current_retrieve_stats.return_value = None
    remote.deserializer = Mock()
    monkeypatch.setattr(remote, "_get_blocking_failed_count", 0, raising=False)
    closing = threading.Event()
    remote_close = remote.close

    def close_remote() -> None:
        closing.set()
        remote_close()

    monkeypatch.setattr(remote, "close", close_remote)
    manager = object.__new__(StorageManager)
    manager.manager_lock = threading.Lock()
    manager.storage_backends = OrderedDict(LocalCPUBackend=cpu, RemoteBackend=remote)
    manager.config = SimpleNamespace(local_cpu=True)
    manager.metadata = SimpleNamespace(role="scheduler")
    manager.lmcache_worker = None
    manager.loop = asyncio.new_event_loop()
    manager.thread = threading.Thread(target=manager.loop.run_forever)
    remote.loop = manager.loop
    replacement = Mock(spec=RemoteBackend)
    replacement.close.side_effect = lambda: events.append("replacement")
    create = Mock(return_value=OrderedDict(RemoteBackend=replacement))
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.CreateStorageBackends", create
    )
    manager.thread.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                assert remote.batched_get_blocking(keys) == [None] * len(keys)
                assert store.entered.wait(5)
                for operation in (manager.close_backend, manager.recreate_backend):
                    with pytest.raises(
                        RuntimeError,
                        match="LocalCPUBackend: dependent backends.*RemoteBackend",
                    ):
                        operation("LocalCPUBackend")
                    assert manager.storage_backends["LocalCPUBackend"] is cpu
                    assert manager.storage_backends["RemoteBackend"] is remote
                    local.memory_allocator.close.assert_not_called()
                    create.assert_not_called()
                if recreate:
                    recreated = executor.submit(
                        manager.recreate_backend, "RemoteBackend"
                    )
                    assert closing.wait(5)
                closed = executor.submit(manager.close)
                assert closing.wait(5)
                assert not closed.done()
                acquired = manager.manager_lock.acquire(blocking=False)
                if acquired:
                    manager.manager_lock.release()
                assert not acquired
                store.unregister_buffer.assert_not_called()
                store.close.assert_not_called()
                local.memory_allocator.close.assert_not_called()
                assert all(release.call_count == 0 for release in local.releases)
                count = len(local.allocations)
                assert remote.batched_get_blocking(keys) == [None] * len(keys)
                assert len(local.allocations) == count
                if recreate:
                    create.assert_not_called()
            finally:
                store.release.set()
            if recreate:
                recreated.result(timeout=5)
            closed.result(timeout=5)
        assert all(store.owned_at_exit)
        assert events == ["unregister", "transport"] + (
            ["replacement"] if recreate else []
        ) + ["allocator"]
        local.memory_allocator.close.assert_called_once()
        store.unregister_buffer.assert_called_once()
        store.close.assert_called_once()
        assert all(release.call_count == 1 for release in local.releases)
        assert not manager.thread.is_alive()
        assert manager.storage_backends == {}
        with pytest.raises(KeyError):
            manager.recreate_backend("RemoteBackend")
    finally:
        store.release.set()
        if manager.loop.is_running():
            manager.loop.call_soon_threadsafe(manager.loop.stop)
        manager.thread.join(timeout=5)
        manager.loop.close()


def test_storage_close_keeps_allocator_and_loop_live_on_remote_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsafe remote drain cannot be silently followed by allocator teardown."""
    remote = _make_remote_backend(True)
    remote.loop = Mock()
    remote.loop.is_running.return_value = True
    remote.connection = SimpleNamespace(close=lambda: asyncio.sleep(0))
    failed: Future = Future()
    failed.set_exception(RuntimeError("unsafe drain"))

    def submit(coroutine: Coroutine, loop: object) -> Future:
        coroutine.close()
        return failed

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    cpu = Mock(spec=LocalCPUBackend)
    manager = object.__new__(StorageManager)
    manager.manager_lock = threading.Lock()
    manager.storage_backends = OrderedDict(LocalCPUBackend=cpu, RemoteBackend=remote)
    manager.loop = remote.loop
    manager.thread = Mock()

    with pytest.raises(RuntimeError, match="unsafe drain"):
        manager.close()
    cpu.close.assert_not_called()
    manager.loop.stop.assert_not_called()
    manager.loop.call_soon_threadsafe.assert_not_called()
    manager.thread.join.assert_not_called()
    assert list(manager.storage_backends) == ["LocalCPUBackend", "RemoteBackend"]


def test_remote_close_propagates_late_result_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed late-result release remains a barrier to allocator shutdown."""
    monkeypatch.setattr(RemoteBackend, "_read_cleanups", {})
    memory_obj = _MemoryObj()
    release = Mock(side_effect=RuntimeError("release failed"))
    monkeypatch.setattr(memory_obj, "ref_count_down", release)
    late: Future = Future()

    def submit(coroutine: Coroutine, loop: object) -> Future:
        coroutine.close()
        return late

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    remote = _make_remote_backend(True)
    remote.connection = SimpleNamespace(
        support_batched_get=lambda: True,
        batched_get=lambda keys: asyncio.sleep(0),
        close=Mock(),
    )
    cpu = Mock(spec=LocalCPUBackend)
    remote.local_cpu_backend = cpu
    remote.config = SimpleNamespace(blocking_timeout_secs=0.001)
    remote.stats_monitor = Mock()
    remote.stats_monitor.get_current_retrieve_stats.return_value = None
    remote.deserializer = Mock()
    monkeypatch.setattr(remote, "_get_blocking_failed_count", 0, raising=False)
    assert remote.batched_get_blocking([_key(1)]) == [None]

    # A pending cleanup barrier also protects a CPU whose owner is no longer listed.
    manager = object.__new__(StorageManager)
    manager.manager_lock = threading.Lock()
    manager.storage_backends = OrderedDict(LocalCPUBackend=cpu)
    for operation in (manager.close_backend, manager.recreate_backend):
        with pytest.raises(RuntimeError, match="outstanding remote read cleanup"):
            operation("LocalCPUBackend")
        cpu.close.assert_not_called()
        assert manager.storage_backends["LocalCPUBackend"] is cpu
    late.set_result([memory_obj])

    for _ in range(2):
        with pytest.raises(RuntimeError, match="release failed"):
            remote.close()
    remote.connection.close.assert_not_called()
    release.assert_called_once()

    manager = object.__new__(StorageManager)
    manager.manager_lock = threading.Lock()
    manager.storage_backends = OrderedDict(LocalCPUBackend=cpu, RemoteBackend=remote)
    manager.config = SimpleNamespace(local_cpu=True)
    manager.loop = Mock()
    manager.thread = Mock()
    # This existing best-effort API swallows the failed remote close and removes it.
    assert manager.close_backend("RemoteBackend")
    assert list(manager.storage_backends) == ["LocalCPUBackend"]
    for operation in (manager.close_backend, manager.recreate_backend):
        with pytest.raises(RuntimeError, match="outstanding remote read cleanup"):
            operation("LocalCPUBackend")
        cpu.close.assert_not_called()
        assert manager.storage_backends["LocalCPUBackend"] is cpu
    with pytest.raises(RuntimeError, match="release failed"):
        manager.close()
    cpu.close.assert_not_called()
    manager.loop.call_soon_threadsafe.assert_not_called()


def test_mooncake_layer_page_get_allocates_one_object_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PageStore:
        def __init__(self) -> None:
            self.args = None

        def batch_get_into_multi_buffers(self, *args):
            self.args = args
            return [sum(sizes) for sizes in args[2]]

    class _Backend:
        def __init__(self) -> None:
            self.allocator = TensorMemoryAllocator(
                torch.zeros(16384, dtype=torch.uint8)
            )
            self.submitted = None

        def batched_allocate_layer_pages(self, *args):
            return self.allocator.batched_allocate_layer_pages(*args)

        def batched_submit_layer_pages(self, keys, pages):
            self.submitted = (keys, pages)

    connector = object.__new__(MooncakestoreConnector)
    connector.__dict__.update(_inflight_gets=set(), _closing=False)
    connector._layer_merged_pages = True
    connector._page_first_multi_buffer = True
    connector.local_cpu_backend = _Backend()
    _configure_page_cardinality(
        connector,
        3,
        dsa_two_groups=True,
        runtime=(3, 2),
        manager=_group_manager(3, 2),
    )
    connector.store = _PageStore()
    metadata_calls = []

    def metadata_for_raw_key(key):
        metadata_calls.append(key)
        return (
            [torch.Size([8])],
            [torch.float16],
            MemoryFormat.KV_MLA_LATENT_FMT,
            16,
        )

    connector._metadata_for_raw_key = metadata_for_raw_key
    keys = [_layer_key(chunk_hash, 0, kv_group=1) for chunk_hash in (1, 2)]
    events = []
    monkeypatch.setattr(
        mooncake_connector, "cold_start_perf_enabled", lambda: True
    )
    monkeypatch.setattr(
        mooncake_connector,
        "cold_start_perf_log",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    pages = asyncio.run(connector.batched_get_layer_pages(keys))

    assert len(pages) == 2
    assert connector.store.args[0] == [
        "__lmcache_page_v1__@2@test@1@0@1@half@1",
        "__lmcache_page_v1__@2@test@1@0@2@half@1",
    ]
    assert connector.store.args[2] == [[16, 16], [16, 16]]
    assert connector.store.args[1] == [
        [page.layer_data_ptr(0), page.layer_data_ptr(1)] for page in pages
    ]
    assert len(metadata_calls) == 1
    submitted_keys, submitted_pages = connector.local_cpu_backend.submitted
    assert submitted_pages == pages
    assert submitted_keys == [keys[0].without_layer(), keys[1].without_layer()]
    event, fields = events.pop()
    assert event == "mooncake_page_get"
    assert fields["layout"] == "layer_merged"
    assert fields["kv_group"] == 1
    assert fields["kv_groups"] == [1]
    assert fields["pages"] == 2
    assert fields["submitted_pages"] == 2
    assert fields["completed_pages"] == 2
    assert fields["layers"] == 2
    assert fields["buffers"] == 4
    assert fields["bytes"] == 64
    assert fields["status"] == "ok"
    assert all(
        fields[name] >= 0
        for name in (
            "metadata_ms",
            "allocation_ms",
            "buffer_setup_ms",
            "transfer_ms",
            "publish_ms",
        )
    )
    for page in pages:
        page.ref_count_down()


def test_mooncake_page_grouping_serializes_each_page_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    _configure_page_cardinality(connector, 3)
    keys = [
        _layer_key(chunk_hash, layer_id)
        for layer_id in range(3)
        for chunk_hash in (1, 2)
    ]
    page_key = Mock(wraps=mooncake_connector.mooncake_page_key)
    monkeypatch.setattr(mooncake_connector, "mooncake_page_key", page_key)

    groups, legacy_indices = connector._complete_page_groups(keys)

    assert legacy_indices == []
    assert len(groups) == 2
    assert sorted(indices for _, indices in groups) == [[0, 2, 4], [1, 3, 5]]
    assert page_key.call_count == 2


def test_mooncake_page_cardinality_global_fallback(
) -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    _configure_page_cardinality(connector, 3)

    assert connector._page_keys_for([_layer_key(1, 0)]) == [
        "__lmcache_page_v1__@3@test@1@0@1@half@0"
    ]


def test_mooncake_page_cardinality_rejects_missing_dsa_runtime() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    _configure_page_cardinality(connector, 3, dsa_two_groups=True)

    with pytest.raises(ValueError, match="runtime"):
        connector._page_keys_for([_layer_key(1, 0, kv_group=1)])


def test_mooncake_page_cardinality_reads_registered_groups_live() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    manager = KVLayerGroupsManager()
    _configure_page_cardinality(
        connector,
        3,
        dsa_two_groups=True,
        runtime=(3, 2),
        manager=manager,
    )
    key = _layer_key(1, 0, kv_group=1)

    assert connector._page_keys_for([key]) == [
        "__lmcache_page_v1__@2@test@1@0@1@half@1"
    ]

    manager.kv_layer_groups = _group_manager(3, 2).kv_layer_groups

    assert connector._page_keys_for([key]) == [
        "__lmcache_page_v1__@2@test@1@0@1@half@1"
    ]


def test_mooncake_page_cardinality_uses_runtime_metadata() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    _configure_page_cardinality(
        connector,
        79,
        dsa_two_groups=True,
        runtime=(79, 22),
    )

    assert connector._page_keys_for([_layer_key(1, 0, kv_group=1)]) == [
        "__lmcache_page_v1__@22@test@1@0@1@half@1"
    ]


def test_mooncake_unequal_page_groups_use_representative_cardinality() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    _configure_page_cardinality(
        connector,
        3,
        dsa_two_groups=True,
        runtime=(3, 2),
        manager=_group_manager(3, 2),
    )
    keys = [
        _layer_key(1, 0, kv_group=0),
        _layer_key(1, 0, kv_group=1),
        _layer_key(1, 1, kv_group=0),
        _layer_key(1, 1, kv_group=1),
        _layer_key(1, 2, kv_group=0),
        _layer_key(2, 0, kv_group=1),
    ]

    groups, legacy_indices = connector._complete_page_groups(keys)

    assert groups == [
        ("__lmcache_page_v1__@3@test@1@0@1@half@0", [0, 2, 4]),
        ("__lmcache_page_v1__@2@test@1@0@1@half@1", [1, 3]),
    ]
    assert legacy_indices == [5]


def test_mooncake_page_alias_requires_complete_batch() -> None:
    class _Store:
        @staticmethod
        def batch_is_exist(keys):
            return [int(key.startswith("__lmcache_page_v1__")) for key in keys]

        @staticmethod
        def is_exist(key):
            return int(key.startswith("__lmcache_page_v1__"))

    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    _configure_page_cardinality(connector, 2)
    connector.store = _Store()
    keys = [_layer_key(1, layer_id) for layer_id in range(2)]

    assert connector.batched_contains(keys[:1]) == 0
    assert connector.batched_contains(keys) == 2
    assert connector.batched_contains_layer_pages(keys[:1]) == 1
    assert not asyncio.run(connector.exists(keys[0]))

    del _Store.batch_is_exist
    assert connector.batched_contains_layer_pages(keys[:1]) == 1


def test_mooncake_page_put_keeps_partial_tail_in_legacy_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PageStore:
        def __init__(self) -> None:
            self.page_args = None
            self.legacy_args = None

        def batch_put_from_multi_buffers(self, *args):
            self.page_args = args
            return [0]

        def batch_put_from(self, *args):
            self.legacy_args = args
            return [0, 0]

    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=4)
    )
    _configure_page_cardinality(connector, 2)
    connector._metadata_for_raw_key = lambda _key: ([], [], None, 4)
    connector.store = _PageStore()
    events = []
    monkeypatch.setattr(
        mooncake_connector, "cold_start_perf_enabled", lambda: True
    )
    monkeypatch.setattr(
        mooncake_connector,
        "cold_start_perf_log",
        lambda _logger, event, **fields: events.append((event, fields)),
    )
    keys = [
        _layer_key(1, 0),
        _layer_key(2, 0),
        _layer_key(1, 1),
        _layer_key(2, 1),
    ]
    memory_objs = [
        _MemoryObj(16, 100),
        _MemoryObj(8, 200),
        _MemoryObj(16, 300),
        _MemoryObj(8, 400),
    ]

    asyncio.run(connector._batched_put_zero_copy(keys, memory_objs))

    assert connector.store.page_args[1] == [[100, 300]]
    assert connector.store.page_args[2] == [[16, 16]]
    assert connector.store.legacy_args[1] == [200, 400]
    assert connector.store.legacy_args[2] == [8, 8]
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)
    event, fields = events[0]
    assert event == "mooncake_page_put"
    assert fields["pages"] == 1
    assert fields["buffers"] == 2
    assert fields["bytes"] == 32
    assert fields["kv_groups"] == [0]
    assert fields["first_page_key"] == connector.store.page_args[0][0]
    assert fields["last_page_key"] == connector.store.page_args[0][0]
    assert fields["legacy_objects"] == 2


def test_mooncake_page_put_selects_each_layer_buffer() -> None:
    class _PageStore:
        def __init__(self) -> None:
            self.page_args = None
            self.ref_count_during_put = None

        def batch_put_from_multi_buffers(self, *args):
            self.page_args = args
            self.ref_count_during_put = page.get_ref_count()
            return [0]

    allocator = TensorMemoryAllocator(torch.zeros(16384, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        [torch.Size([8])],
        [torch.float16],
        batch_size=1,
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert pages is not None
    page = pages[0]
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=4)
    )
    _configure_page_cardinality(connector, 2)
    connector._metadata_for_raw_key = lambda _key: ([], [], None, 4)
    connector.store = _PageStore()
    keys = [_layer_key(1, layer_id) for layer_id in range(2)]

    asyncio.run(connector._batched_put_zero_copy(keys, [page, page]))

    assert connector.store.page_args[1] == [
        [page.layer_data_ptr(layer_id) for layer_id in range(2)]
    ]
    assert connector.store.page_args[2] == [[page.layer_size] * 2]
    assert connector.store.ref_count_during_put == 2
    assert page.get_ref_count() == 1
    page.ref_count_down()


def test_mooncake_timeout_keeps_source_buffer_until_native_put_exits() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector.config = SimpleNamespace(transfer_timeout=0.01)
    connector._inflight_put_tasks = set()
    memory_obj = _MemoryObj()
    release = threading.Event()

    def blocking_put() -> int:
        release.wait()
        return 0

    async def run() -> None:
        try:
            with pytest.raises(TimeoutError, match="timed out"):
                await connector._run_blocking_put(
                    "put_from", blocking_put, (), [memory_obj]
                )
            assert memory_obj.ref_count == 2
        finally:
            release.set()
        while connector._inflight_put_tasks:
            await asyncio.sleep(0.001)

    asyncio.run(run())
    assert memory_obj.ref_count == 1
