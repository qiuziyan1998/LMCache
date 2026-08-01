# SPDX-License-Identifier: Apache-2.0
"""Mooncake-specific remote store completion semantics."""

# Standard
from concurrent.futures import Future
from types import ModuleType, SimpleNamespace
import asyncio
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_process import (
    _run_mooncake_process_worker,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


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


def _key(chunk_hash: int) -> CacheEngineKey:
    return CacheEngineKey("test", 1, 0, chunk_hash, torch.float16)


def _layer_key(chunk_hash: int, layer_id: int) -> LayerCacheEngineKey:
    return LayerCacheEngineKey(
        "test",
        1,
        0,
        chunk_hash,
        torch.float16,
        layer_id=layer_id,
    )


def _make_remote_backend(requires_completion: bool) -> RemoteBackend:
    backend = object.__new__(RemoteBackend)
    backend.connection = _Connection(requires_completion)
    backend.loop = object()
    backend.serializer = _Serializer()
    backend._mla_worker_id_as0_mode = False
    backend.put_tasks = set()
    backend.lock = threading.Lock()
    return backend


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


def test_mooncake_requires_put_completion() -> None:
    connector = object.__new__(MooncakestoreConnector)
    assert connector.requires_put_completion()


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


def test_mooncake_page_get_scatter_returns_layer_objects() -> None:
    class _PageStore:
        def batch_get_into_multi_buffers(self, page_keys, ptrs, sizes):
            assert len(page_keys) == 1
            assert ptrs == [[100, 300]]
            assert sizes == [[16, 16]]
            return [32]

    connector = object.__new__(MooncakestoreConnector)
    connector._page_num_layers = 2
    connector.store = _PageStore()
    memory_objs = [_MemoryObj(16, 100), _MemoryObj(16, 300)]
    connector._allocate_zero_copy_buffers = lambda _keys: (
        memory_objs,
        [],
        "batched",
    )
    keys = [_layer_key(1, 0), _layer_key(1, 1)]

    loaded = asyncio.run(
        connector._batch_get_pages(keys, [("page-key", [0, 1])])
    )

    assert loaded == {0: memory_objs[0], 1: memory_objs[1]}
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)


def test_mooncake_page_put_keeps_partial_tail_in_legacy_layout() -> None:
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
    connector._page_num_layers = 2
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=4)
    )
    connector._metadata_for_raw_key = lambda _key: ([], [], None, 4)
    connector.store = _PageStore()
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


def test_mooncake_page_get_process_isolation_uses_shared_offsets() -> None:
    class _ProcessClient:
        def __init__(self) -> None:
            self.calls = []
            self.pid = 1234

        def get_pages(
            self,
            page_keys,
            page_offsets,
            page_sizes,
            *,
            timeout,
        ):
            self.calls.append(
                (page_keys, page_offsets, page_sizes, timeout)
            )
            return [32]

    class _ParentStore:
        @staticmethod
        def batch_get_into_multi_buffers(*_args):
            raise AssertionError('parent Mooncake store must not transfer')

    buffer = SimpleNamespace(
        data_ptr=lambda: 1000,
        numel=lambda: 4096,
    )
    process_client = _ProcessClient()
    connector = object.__new__(MooncakestoreConnector)
    connector.config = SimpleNamespace(transfer_timeout=7)
    connector.store = _ParentStore()
    connector._cold_process_transfer_client = process_client
    connector.local_cpu_backend = SimpleNamespace(
        memory_allocator=SimpleNamespace(buffer=buffer)
    )
    memory_objs = [_MemoryObj(16, 1016), _MemoryObj(16, 1048)]
    connector._allocate_zero_copy_buffers = lambda _keys: (
        memory_objs,
        [],
        'batched',
    )
    keys = [_layer_key(1, 0), _layer_key(1, 1)]

    loaded = asyncio.run(
        connector._batch_get_pages_process_isolated(
            keys,
            [('page-key', [0, 1])],
        )
    )

    assert loaded == {0: memory_objs[0], 1: memory_objs[1]}
    assert process_client.calls == [
        (['page-key'], [[16, 48]], [[16, 16]], 7.0)
    ]
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)


def test_mooncake_process_worker_reconstructs_shared_slab_pointers(
    monkeypatch,
) -> None:
    class _Connection:
        def __init__(self) -> None:
            self.messages = iter(
                [
                    ('get_pages', ['page'], [[16, 48]], [[16, 16]]),
                    ('close',),
                ]
            )
            self.sent = []
            self.closed = False

        def recv(self):
            return next(self.messages)

        def send(self, value) -> None:
            self.sent.append(value)

        def close(self) -> None:
            self.closed = True

    class _Mapping:
        ptr = 1000
        size = 4096

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _Store:
        def __init__(self) -> None:
            self.get_calls = []
            self.unregistered = []
            self.closed = False

        def setup(self, *_args) -> None:
            return None

        def register_buffer(self, ptr, size) -> int:
            assert (ptr, size) == (1000, 4096)
            return 0

        def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
            self.get_calls.append((keys, ptrs, sizes))
            return [32]

        def unregister_buffer(self, ptr) -> None:
            self.unregistered.append(ptr)

        def close(self) -> None:
            self.closed = True

    mapping = _Mapping()
    store = _Store()
    mooncake_module = ModuleType('mooncake')
    mooncake_store_module = ModuleType('mooncake.store')
    mooncake_store_module.MooncakeDistributedStore = lambda: store
    monkeypatch.setitem(sys.modules, 'mooncake', mooncake_module)
    monkeypatch.setitem(sys.modules, 'mooncake.store', mooncake_store_module)
    monkeypatch.setattr(
        'lmcache.v1.storage_backend.connector.mooncakestore_process.'
        'SharedSlabMapping.attach',
        lambda **_kwargs: mapping,
    )
    connection = _Connection()
    config = MooncakeStoreConfig(
        local_hostname='host',
        metadata_server='meta',
        global_segment_size=1,
        local_buffer_size=0,
        protocol='tcp',
        device_name='npu',
        master_server_address='master',
        transfer_timeout=30,
        storage_root_dir='',
    )

    _run_mooncake_process_worker(
        connection,
        config.__dict__,
        'slab',
        4096,
        0,
    )

    assert store.get_calls == [
        (['page'], [[1016, 1048]], [[16, 16]])
    ]
    assert connection.sent[0][0] == 'ready'
    assert connection.sent[1] == ('result', [32])
    assert connection.sent[2] == ('closed', None)
    assert store.unregistered == [1000]
    assert store.closed
    assert mapping.closed
    assert connection.closed


def test_remote_backend_dispatches_process_isolated_get(monkeypatch) -> None:
    calls = []
    memory_obj = _MemoryObj()

    class _GetConnection:
        @staticmethod
        def support_batched_get() -> bool:
            return True

        def batched_get(self, _keys):
            calls.append('normal')

            async def result():
                return [memory_obj]

            return result()

        def batched_get_process_isolated(self, _keys):
            calls.append('isolated')

            async def result():
                return [memory_obj]

            return result()

    def submit(coroutine, _loop) -> Future:
        future: Future = Future()
        future.set_result(asyncio.run(coroutine))
        return future

    monkeypatch.setattr(asyncio, 'run_coroutine_threadsafe', submit)
    backend = object.__new__(RemoteBackend)
    backend.local_cpu_backend = object()
    backend.connection = _GetConnection()
    backend._mla_worker_id_as0_mode = False
    backend.loop = object()
    backend.config = SimpleNamespace(blocking_timeout_secs=1)
    backend.stats_monitor = SimpleNamespace(
        update_interval_remote_time_to_get_sync=lambda _value: None,
        get_current_retrieve_stats=lambda: None,
    )
    backend.deserializer = SimpleNamespace(deserialize=lambda value: value)
    backend._get_blocking_failed_count = 0

    result = backend.batched_get_blocking_process_isolated([_key(1)])

    assert result == [memory_obj]
    assert calls == ['isolated']


def test_instrumented_connector_forwards_process_isolated_get() -> None:
    memory_obj = _MemoryObj()

    class _Underlying:
        async def batched_get_process_isolated(self, keys):
            assert keys == [_key(1)]
            return [memory_obj]

    connector = object.__new__(InstrumentedRemoteConnector)
    connector._connector = _Underlying()
    connector._stats_monitor = SimpleNamespace(
        update_interval_remote_time_to_get=lambda _value: None,
        update_interval_remote_read_metrics=lambda _value: None,
    )

    result = asyncio.run(
        connector.batched_get_process_isolated([_key(1)])
    )

    assert result == [memory_obj]
