# SPDX-License-Identifier: Apache-2.0
"""Mooncake-specific remote store completion semantics."""

# Standard
from concurrent.futures import Future, TimeoutError
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import asyncio
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import (
    MOONCAKE_VALID_TOKENS_TAG,
    mooncake_page_key,
)
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.storage_backend.connector import (
    mooncakestore_connector as mooncake_connector,
)
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


def _make_mooncake_connector(
    monkeypatch: pytest.MonkeyPatch,
    extra_config: dict,
    calls: list,
) -> tuple[MooncakestoreConnector, asyncio.AbstractEventLoop]:
    class _Store:
        def setup(self, *args):
            calls.append(("setup", args))
            return 0

        def register_buffer(self, ptr, size):
            calls.append(("register", ptr, size))
            return 0

        def unregister_buffer(self, ptr):
            calls.append(("unregister", ptr))
            return 0

        def close(self):
            calls.append(("close",))

    package = ModuleType("mooncake")
    package.__path__ = []  # type: ignore[attr-defined]
    store_module = ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = _Store
    store_module.ReplicateConfig = type("ReplicateConfig", (), {})
    package.store = store_module
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setattr(
        mooncake_connector.NUMADetector,
        "get_numa_mapping",
        lambda _config: None,
    )
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        extra_config={
            "local_hostname": "configured-host",
            "metadata_server": "metadata",
            "master_server_address": "master",
            "protocol": "ascend",
            **extra_config,
        },
    )
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(1, 2, 8, 1, 1),
        chunk_size=8,
    )
    local_cpu = SimpleNamespace(
        config=config,
        metadata=metadata,
        memory_allocator=SimpleNamespace(),
    )
    loop = asyncio.new_event_loop()
    return MooncakestoreConnector("", 0, "", loop, local_cpu, config), loop


def test_mooncake_reuses_vllm_engine_and_registration_registry(monkeypatch) -> None:
    calls = []

    class _Engine:
        @staticmethod
        def get_rpc_port():
            return 12345

        @staticmethod
        def get_engine():
            return "native-engine"

    global_te = SimpleNamespace(
        get_transfer_engine=lambda hostname, device_name: (
            calls.append(("engine", hostname, device_name)) or _Engine()
        ),
        register_buffer=lambda ptrs, sizes: calls.append(
            ("shared-register", ptrs, sizes)
        ),
        adopt_registered_buffer=lambda ptr, size, register=None: (
            calls.append(("shared-adopt", ptr, size))
            or (register() if register is not None else 0) == 0
        ),
        release_adopted_buffer=lambda ptr, size, unregister=None: (
            calls.append(("shared-release", ptr, size))
            or (unregister() if unregister is not None else None)
        ),
    )
    modules = {
        "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine": (
            SimpleNamespace(global_te=global_te)
        ),
        "vllm.utils.network_utils": SimpleNamespace(get_ip=lambda: "10.0.0.1"),
    }
    monkeypatch.setattr(mooncake_connector, "import_module", modules.__getitem__)

    connector, loop = _make_mooncake_connector(
        monkeypatch,
        {"mooncake_reuse_vllm_transfer_engine": True},
        calls,
    )
    tensor = torch.empty(16, dtype=torch.uint8)
    owner = SimpleNamespace(
        device=SimpleNamespace(type="npu"),
        untyped_storage=tensor.untyped_storage,
    )
    cpu_owner = torch.empty(8, dtype=torch.uint8)
    try:
        connector._register_external_owners((owner,))
        connector._register_external_owners((owner,))
        connector._register_external_owners((cpu_owner,))
        asyncio.run(connector.close())
    finally:
        loop.close()

    setup = next(call for call in calls if call[0] == "setup")
    assert setup[1][0] == "10.0.0.1:12345"
    assert setup[1][-1] == "native-engine"
    assert [call[0] for call in calls].count("shared-register") == 1
    assert [call[0] for call in calls].count("shared-adopt") == 0
    assert [call[0] for call in calls].count("shared-release") == 0
    assert [call[0] for call in calls].count("register") == 1
    assert [call[0] for call in calls].count("unregister") == 1


def test_mooncake_shared_engine_tracks_cpu_slab_registration(monkeypatch) -> None:
    calls = []

    class _Engine:
        @staticmethod
        def get_rpc_port():
            return 12345

        @staticmethod
        def get_engine():
            return "native-engine"

    global_te = SimpleNamespace(
        get_transfer_engine=lambda *_args, **_kwargs: _Engine(),
        adopt_registered_buffer=lambda ptr, size, register=None: (
            calls.append(("adopt", ptr, size))
            or (register() if register is not None else 0) == 0
        ),
        release_adopted_buffer=lambda ptr, size, unregister=None: (
            calls.append(("release", ptr, size))
            or (unregister() if unregister is not None else None)
        ),
    )
    modules = {
        "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine": (
            SimpleNamespace(global_te=global_te)
        ),
        "vllm.utils.network_utils": SimpleNamespace(get_ip=lambda: "10.0.0.1"),
    }
    monkeypatch.setattr(mooncake_connector, "import_module", modules.__getitem__)
    buffer = torch.empty(16, dtype=torch.uint8)
    connector, loop = _make_mooncake_connector(
        monkeypatch,
        {"mooncake_reuse_vllm_transfer_engine": True},
        calls,
    )
    connector.local_cpu_backend.memory_allocator = SimpleNamespace(
        pin_allocator=SimpleNamespace(buffer=buffer)
    )
    connector._register_cpu_buffer()
    try:
        connector._unregister_cpu_buffer()
    finally:
        loop.close()

    assert calls[-2:] == [
        ("release", buffer.data_ptr(), buffer.numel()),
        ("unregister", buffer.data_ptr()),
    ]


def test_mooncake_close_retries_cpu_slab_after_live_transfer() -> None:
    calls = []
    releases = iter((RuntimeError("in use"), None))

    def release(ptr, size, unregister=None):
        calls.append(("release", ptr, size))
        error = next(releases)
        if error is not None:
            raise error
        assert unregister is not None
        return unregister()

    connector = object.__new__(MooncakestoreConnector)
    connector._inflight_put_tasks = set()
    connector.registered_buffer_ptr = 0x1000
    connector.registered_buffer_size = 0x1000
    connector._shared_cpu_buffer_adopted = True
    connector._shared_global_te = SimpleNamespace(
        release_adopted_buffer=release,
    )
    connector._external_buffers = {}
    connector._shared_external_buffers = {}
    connector.store = SimpleNamespace(
        unregister_buffer=lambda ptr: calls.append(("unregister", ptr)) or 0,
        close=lambda: calls.append(("close",)),
    )

    with pytest.raises(RuntimeError, match="still in use"):
        asyncio.run(connector.close())
    assert connector.registered_buffer_ptr == 0x1000
    assert ("close",) not in calls

    asyncio.run(connector.close())
    assert connector.registered_buffer_ptr is None
    assert calls[-2:] == [("unregister", 0x1000), ("close",)]


def test_mooncake_shared_cpu_adoption_failure_is_fail_closed(monkeypatch) -> None:
    calls = []

    class _Engine:
        @staticmethod
        def get_rpc_port():
            return 12345

        @staticmethod
        def get_engine():
            return "native-engine"

    def reject_adoption(_ptr, _size, _register=None):
        raise RuntimeError("foreign owner")

    modules = {
        "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine": (
            SimpleNamespace(global_te=SimpleNamespace(
                get_transfer_engine=lambda *_args, **_kwargs: _Engine(),
                adopt_registered_buffer=reject_adoption,
            ))
        ),
        "vllm.utils.network_utils": SimpleNamespace(get_ip=lambda: "10.0.0.1"),
    }
    monkeypatch.setattr(mooncake_connector, "import_module", modules.__getitem__)
    buffer = torch.empty(16, dtype=torch.uint8)
    connector, loop = _make_mooncake_connector(
        monkeypatch,
        {"mooncake_reuse_vllm_transfer_engine": True},
        calls,
    )
    connector.local_cpu_backend.memory_allocator = SimpleNamespace(
        pin_allocator=SimpleNamespace(buffer=buffer)
    )
    with pytest.raises(RuntimeError, match="foreign owner"):
        connector._register_cpu_buffer()
    loop.close()

    assert ("register", buffer.data_ptr(), buffer.numel()) not in calls


def test_mooncake_shared_cpu_requires_adoption_api(monkeypatch) -> None:
    calls = []

    class _Engine:
        get_rpc_port = staticmethod(lambda: 12345)
        get_engine = staticmethod(lambda: "native-engine")

    modules = {
        "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine": (
            SimpleNamespace(global_te=SimpleNamespace(
                get_transfer_engine=lambda *_args, **_kwargs: _Engine(),
            ))
        ),
        "vllm.utils.network_utils": SimpleNamespace(get_ip=lambda: "10.0.0.1"),
    }
    monkeypatch.setattr(mooncake_connector, "import_module", modules.__getitem__)
    connector, loop = _make_mooncake_connector(
        monkeypatch,
        {"mooncake_reuse_vllm_transfer_engine": True},
        calls,
    )
    connector.local_cpu_backend.memory_allocator = SimpleNamespace(
        pin_allocator=SimpleNamespace(buffer=torch.empty(16, dtype=torch.uint8))
    )

    with pytest.raises(RuntimeError, match="external registration API"):
        connector._register_cpu_buffer()
    loop.close()

    assert not any(call[0] == "register" for call in calls)
    assert connector.registered_buffer_ptr is None


def test_mooncake_shared_engine_is_strictly_opt_in(monkeypatch) -> None:
    calls = []
    connector, loop = _make_mooncake_connector(monkeypatch, {}, calls)
    try:
        asyncio.run(connector.close())
    finally:
        loop.close()

    setup = next(call for call in calls if call[0] == "setup")
    assert len(setup[1]) == 7


def test_mooncake_shared_engine_rejects_non_ascend_protocol(monkeypatch) -> None:
    with pytest.raises(ValueError, match="requires protocol=ascend"):
        _make_mooncake_connector(
            monkeypatch,
            {
                "protocol": "tcp",
                "mooncake_reuse_vllm_transfer_engine": True,
            },
            [],
        )


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


def _make_remote_backend(
    requires_completion: bool,
    *,
    remote_fill_enabled: bool = False,
) -> RemoteBackend:
    backend = object.__new__(RemoteBackend)
    backend.config = SimpleNamespace(
        enable_remote_lmcache_store=remote_fill_enabled,
    )
    backend.connection = _Connection(requires_completion)
    backend.loop = object()
    backend.serializer = _Serializer()
    backend._mla_worker_id_as0_mode = False
    backend.put_tasks = set()
    backend._single_put_futures = {}
    backend._single_put_callbacks = {}
    backend.lock = threading.Lock()
    return backend


def test_remote_fill_missing_connection_fails_required_puts(monkeypatch) -> None:
    monkeypatch.setenv(
        "LMCACHE_REMOTE_FILL_H0_QUALIFICATION",
        "mooncake-sync-write-visible-v1",
    )
    backend = _make_remote_backend(True, remote_fill_enabled=True)
    backend.connection = None

    single = backend.submit_put_task(_key(1), _MemoryObj())
    batch = backend.batched_submit_put_task([_key(2)], [_MemoryObj()])

    with pytest.raises(ConnectionError, match="required remote storage"):
        single.result()
    assert batch is not None
    with pytest.raises(ConnectionError, match="required remote storage"):
        batch[0].result()


def test_legacy_missing_connection_remains_best_effort(monkeypatch) -> None:
    monkeypatch.delenv("LMCACHE_REMOTE_FILL_H0_QUALIFICATION", raising=False)
    backend = _make_remote_backend(False, remote_fill_enabled=True)
    backend.connection = None

    single = backend.submit_put_task(_key(1), _MemoryObj())
    batch = backend.batched_submit_put_task([_key(2)], [_MemoryObj()])

    assert single.result() is None
    assert batch is None


@pytest.mark.parametrize("partial", [False, True])
def test_duplicate_single_key_put_joins_real_pending_future(
    monkeypatch, partial: bool
) -> None:
    submitted: list[Future] = []

    class _SingleConnection:
        @staticmethod
        async def put(key, memory_obj) -> None:
            return None

    def submit(coroutine, loop) -> Future:
        coroutine.close()
        future: Future = Future()
        submitted.append(future)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    backend = _make_remote_backend(True)
    backend.connection = _SingleConnection()
    first_callback: list[CacheEngineKey] = []
    second_callback: list[CacheEngineKey] = []
    key = (
        CacheEngineKey(
            "test",
            1,
            0,
            7,
            torch.float16,
            {MOONCAKE_VALID_TOKENS_TAG: 3},
        )
        if partial
        else _key(7)
    )

    first = backend.submit_put_task(key, _MemoryObj(), first_callback.append)
    second = backend.submit_put_task(key, _MemoryObj(), second_callback.append)

    assert first is second
    assert not first.done()
    assert len(submitted) == 1
    assert backend.exists_in_put_tasks(key)

    submitted[0].set_result(None)

    assert first.result() is None
    assert first_callback == [key]
    assert second_callback == [key]
    assert not backend.exists_in_put_tasks(key)
    assert backend._single_put_futures == {}
    assert backend._single_put_callbacks == {}


def test_duplicate_batched_put_joins_real_physical_writer(monkeypatch) -> None:
    submitted: list[Future] = []

    class _BatchConnection(_Connection):
        @staticmethod
        async def put(key, memory_obj) -> None:
            raise AssertionError("same-key join must not submit individual puts")

    def submit(coroutine, loop) -> Future:
        del loop
        coroutine.close()
        future: Future = Future()
        submitted.append(future)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    backend = _make_remote_backend(True)
    backend.connection = _BatchConnection(True)
    keys = [_key(11), _key(12)]
    first_callbacks: list[CacheEngineKey] = []
    second_callbacks: list[CacheEngineKey] = []

    first = backend.batched_submit_put_task(
        keys,
        [_MemoryObj(), _MemoryObj()],
        on_complete_callback=first_callbacks.append,
    )
    second = backend.batched_submit_put_task(
        keys,
        [_MemoryObj(), _MemoryObj()],
        on_complete_callback=second_callbacks.append,
    )

    assert first is not None and second is not None
    assert len(submitted) == 1
    assert all(future is first[0] for future in second)
    assert not first[0].done()

    submitted[0].set_result(None)

    assert first[0].result() is None
    assert first_callbacks == keys
    assert second_callbacks == keys
    assert backend._single_put_futures == {}
    assert backend._single_put_callbacks == {}
    assert not any(backend.exists_in_put_tasks(key) for key in keys)


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
    backend = object.__new__(RemoteBackend)
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


def test_mooncake_page_put_partitions_latent_placement_and_restores_order() -> None:
    class _ReplicateConfig:
        pass

    calls = []

    class _Store:
        @staticmethod
        def batch_put_from_multi_buffers(keys, ptrs, sizes, replica):
            calls.append((keys, ptrs, sizes, replica))
            return [int(key) + 10 for key in keys]

    connector = object.__new__(MooncakestoreConnector)
    connector.store = _Store()
    connector._replicate_config_cls = _ReplicateConfig
    connector.replica_config = _ReplicateConfig()
    connector.replica_config.replica_num = 3
    keys = [
        CacheEngineKey(
            "test",
            1,
            0,
            index,
            torch.float16,
            request_configs=(
                {"lmcache.mooncake_preferred_segment": segment}
                if segment is not None
                else {}
            ),
            kv_group=kv_group,
        )
        for index, kv_group, segment in (
            (0, 0, "segment-b"),
            (1, 1, "ignored-for-indexer"),
            (2, 0, None),
            (3, 0, "segment-a"),
            (4, 0, "segment-b"),
        )
    ]

    result = connector._batch_put_multi_buffers_by_segment(
        keys,
        [str(index) for index in range(5)],
        [[index] for index in range(5)],
        [[1] for _ in range(5)],
    )

    statuses, segments, preferred, default, batches = result
    assert statuses == [10, 11, 12, 13, 14]
    assert (segments, preferred, default, batches) == (
        ["segment-a", "segment-b"],
        3,
        2,
        3,
    )
    assert calls[0][0] == ["0", "4"]
    assert calls[0][3].preferred_segment == "segment-b"
    assert calls[1][0] == ["1", "2"]
    assert calls[1][3] is connector.replica_config
    assert calls[2][0] == ["3"]
    assert calls[2][3].preferred_segment == "segment-a"
    assert calls[0][3] is not calls[2][3]
    assert calls[0][3].replica_num == calls[2][3].replica_num == 1


def test_mooncake_page_put_fast_paths_reuse_input_arrays() -> None:
    class _ReplicateConfig:
        pass

    calls = []

    class _Store:
        @staticmethod
        def batch_put_from_multi_buffers(keys, ptrs, sizes, replica):
            calls.append((keys, ptrs, sizes, replica))
            return None

    connector = object.__new__(MooncakestoreConnector)
    connector.store = _Store()
    connector._replicate_config_cls = _ReplicateConfig
    connector.replica_config = _ReplicateConfig()
    connector.replica_config.replica_num = 2
    connector.replica_config.preferred_segment = "local-segment"
    store_keys = ["0", "1", "2", "3", "4"]
    ptrs = [[10], [20], [30], [40], [50]]
    sizes = [[1], [1], [1], [1], [1]]
    default_keys = [
        _key(0),
        CacheEngineKey(
            "test",
            1,
            0,
            1,
            torch.float16,
            request_configs={"lmcache.mooncake_preferred_segment": 123},
            kv_group=0,
        ),
        CacheEngineKey(
            "test",
            1,
            0,
            2,
            torch.float16,
            request_configs={"lmcache.mooncake_preferred_segment": ""},
            kv_group=0,
        ),
        CacheEngineKey(
            "test",
            1,
            0,
            3,
            torch.float16,
            request_configs={"lmcache.mooncake_preferred_segment": "   "},
            kv_group=0,
        ),
        CacheEngineKey(
            "test",
            1,
            0,
            4,
            torch.float16,
            request_configs={"lmcache.mooncake_preferred_segment": "ignored"},
            kv_group=1,
        ),
    ]

    default_result = connector._batch_put_multi_buffers_by_segment(
        default_keys, store_keys, ptrs, sizes
    )
    assert default_result == (None, [], 0, 5, 1)
    assert calls[-1][0] is store_keys
    assert calls[-1][1] is ptrs
    assert calls[-1][2] is sizes
    assert calls[-1][3] is connector.replica_config
    assert connector.replica_config.preferred_segment == "local-segment"

    hinted_keys = [
        CacheEngineKey(
            "test",
            1,
            0,
            index,
            torch.float16,
            request_configs={
                "lmcache.mooncake_preferred_segment": "decoder-segment"
            },
            kv_group=0,
        )
        for index in range(5)
    ]
    hinted_result = connector._batch_put_multi_buffers_by_segment(
        hinted_keys, store_keys, ptrs, sizes
    )
    assert hinted_result == (None, ["decoder-segment"], 5, 0, 1)
    assert calls[-1][0] is store_keys
    assert calls[-1][1] is ptrs
    assert calls[-1][2] is sizes
    assert calls[-1][3].preferred_segment == "decoder-segment"
    assert calls[-1][3].replica_num == 1
    assert connector.replica_config.preferred_segment == "local-segment"


def test_mooncake_page_put_mixed_none_statuses_remain_aligned() -> None:
    class _ReplicateConfig:
        pass

    class _Store:
        @staticmethod
        def batch_put_from_multi_buffers(keys, ptrs, sizes, replica):
            if replica is connector.replica_config:
                return None
            return [7] * len(keys)

    connector = object.__new__(MooncakestoreConnector)
    connector.store = _Store()
    connector._replicate_config_cls = _ReplicateConfig
    connector.replica_config = _ReplicateConfig()
    connector.replica_config.replica_num = 1
    hinted = CacheEngineKey(
        "test",
        1,
        0,
        1,
        torch.float16,
        request_configs={
            "lmcache.mooncake_preferred_segment": "decoder-segment"
        },
        kv_group=0,
    )

    statuses, *_ = connector._batch_put_multi_buffers_by_segment(
        [_key(0), hinted], ["0", "1"], [[10], [20]], [[1], [1]]
    )
    assert statuses == [0, 7]


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
    connector._page_num_layers = 2
    connector._external_put_lock = asyncio.Lock()
    connector._external_buffers = {}
    connector._inflight_put_tasks = set()
    connector.store = _Store()
    connector.replica_config = object()
    connector.config = SimpleNamespace(transfer_timeout=5)
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 1)
    owner = torch.empty(16, dtype=torch.uint8)
    events = (_Event(), _Event())

    asyncio.run(
        connector.batched_put_external_pages(
            [_key(7)],
            [[owner.data_ptr()]],
            [[owner.numel()]],
            (owner,),
            events,
            "request",
        )
    )

    assert all(event.waited for event in events)
    put = next(call for call in calls if call[0] == "put")
    assert put[1] == [mooncake_page_key(_key(7), 2)]
    assert put[2:] == ([[owner.data_ptr()]], [[owner.numel()]])
    layer_key = _key(8).get_layer(1)
    asyncio.run(
        connector.batched_put_external_pages(
            [layer_key],
            [[owner.data_ptr()]],
            [[8]],
            (owner,),
            events,
            "request",
        )
    )
    assert [call for call in calls if call[0] == "put"][-1][1] == [
        layer_key.to_string()
    ]
    with pytest.raises(ValueError, match="byte count mismatch"):
        asyncio.run(
            connector.batched_put_external_pages(
                [_key(9)],
                [[owner.data_ptr()]],
                [[owner.numel() - 1]],
                (owner,),
                events,
                "request",
            )
        )
    assert connector.batched_external_pages_exist([_key(7)]) == [True]
    exists = next(call for call in calls if call[0] == "exists")
    assert exists[1] == [mooncake_page_key(_key(7), 2)]


def test_mooncake_direct_page_put_uses_latent_segment_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReplicateConfig:
        pass

    calls = []

    class _Store:
        @staticmethod
        def register_buffer(ptr, size):
            return 0

        @staticmethod
        def batch_put_from_multi_buffers(keys, ptrs, sizes, replica):
            calls.append((keys, replica))
            return [0] * len(keys)

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector._external_put_lock = asyncio.Lock()
    connector._external_buffers = {}
    connector._inflight_put_tasks = set()
    connector.store = _Store()
    connector._replicate_config_cls = _ReplicateConfig
    connector.replica_config = _ReplicateConfig()
    connector.replica_config.replica_num = 1
    connector.config = SimpleNamespace(transfer_timeout=5)
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 1)
    owner = torch.empty(32, dtype=torch.uint8)
    hinted = CacheEngineKey(
        "test",
        1,
        0,
        10,
        torch.float16,
        request_configs={
            "lmcache.mooncake_preferred_segment": "decoder-segment"
        },
        kv_group=0,
    )
    events = []
    monkeypatch.setattr(
        mooncake_connector, "cold_start_perf_enabled", lambda: True
    )
    monkeypatch.setattr(
        mooncake_connector,
        "cold_start_perf_log",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    asyncio.run(
        connector.batched_put_external_pages(
            [hinted, _key(11)],
            [[owner.data_ptr()], [owner.data_ptr() + 16]],
            [[16], [16]],
            (owner,),
            None,
            "request",
        )
    )

    assert len(calls) == 2
    assert calls[0][1].preferred_segment == "decoder-segment"
    assert calls[1][1] is connector.replica_config
    event, fields = events[-1]
    assert event == "direct_npu_page_put"
    assert fields["preferred_segments"] == ["decoder-segment"]
    assert fields["preferred_pages"] == 1
    assert fields["default_pages"] == 1
    assert fields["placement_batches"] == 2


def test_mooncake_direct_page_put_attributes_preferred_failure_and_retries() -> None:
    class _ReplicateConfig:
        pass

    class _Store:
        def __init__(self) -> None:
            self.calls = []
            self.fail_preferred = True

        @staticmethod
        def register_buffer(ptr, size):
            return 0

        def batch_put_from_multi_buffers(self, keys, ptrs, sizes, replica):
            self.calls.append((keys, ptrs, sizes, replica))
            preferred = getattr(replica, "preferred_segment", None)
            if self.fail_preferred and preferred == "decoder-segment":
                return [-7] * len(keys)
            return [0] * len(keys)

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector._external_put_lock = asyncio.Lock()
    connector._external_buffers = {}
    connector._inflight_put_tasks = set()
    connector.store = _Store()
    connector._replicate_config_cls = _ReplicateConfig
    connector.replica_config = _ReplicateConfig()
    connector.replica_config.replica_num = 1
    connector.config = SimpleNamespace(transfer_timeout=5)
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 1)
    owner = torch.empty(32, dtype=torch.uint8)
    default = _key(30)
    preferred = CacheEngineKey(
        "test",
        1,
        0,
        31,
        torch.float16,
        request_configs={
            "lmcache.mooncake_preferred_segment": "decoder-segment"
        },
        kv_group=0,
    )
    keys = [default, preferred]
    store_keys = [mooncake_page_key(key, 2) for key in keys]
    ptrs = [[owner.data_ptr()], [owner.data_ptr() + 16]]
    sizes = [[16], [16]]

    with pytest.raises(RuntimeError, match="direct page put failed") as exc_info:
        asyncio.run(
            connector.batched_put_external_pages(
                keys, ptrs, sizes, (owner,), None, "request"
            )
        )

    assert exc_info.value.failed_pages == [store_keys[1]]
    assert connector.store.calls[0][:3] == (
        [store_keys[0]],
        [ptrs[0]],
        [sizes[0]],
    )
    assert connector.store.calls[1][:3] == (
        [store_keys[1]],
        [ptrs[1]],
        [sizes[1]],
    )

    connector.store.calls.clear()
    connector.store.fail_preferred = False
    placement = connector._batch_put_multi_buffers_by_segment(
        keys, store_keys, ptrs, sizes
    )

    assert placement[0] == [0, 0]
    assert store_keys == [
        mooncake_page_key(default, 2),
        mooncake_page_key(preferred, 2),
    ]
    assert ptrs == [[owner.data_ptr()], [owner.data_ptr() + 16]]
    assert sizes == [[16], [16]]
    assert connector.store.calls[0][:3] == (
        [store_keys[0]],
        [ptrs[0]],
        [sizes[0]],
    )
    assert connector.store.calls[1][:3] == (
        [store_keys[1]],
        [ptrs[1]],
        [sizes[1]],
    )


def test_mooncake_cancelled_direct_page_put_drains_native_transfer() -> None:
    entered = threading.Event()
    release = threading.Event()

    class _Store:
        @staticmethod
        def register_buffer(ptr, size):
            return 0

        @staticmethod
        def batch_put_from_multi_buffers(keys, ptrs, sizes, replica):
            entered.set()
            assert release.wait(5)
            return [0] * len(keys)

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector._external_put_lock = asyncio.Lock()
    connector._external_buffers = {}
    connector._inflight_put_tasks = set()
    connector.store = _Store()
    connector.replica_config = object()
    connector.config = SimpleNamespace(transfer_timeout=5)
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 1)
    owner = torch.empty(16, dtype=torch.uint8)

    async def cancel_put() -> None:
        task = asyncio.create_task(
            connector.batched_put_external_pages(
                [_key(12)],
                [[owner.data_ptr()]],
                [[16]],
                (owner,),
                None,
                "request",
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert connector._external_put_lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_put())


def test_mooncake_direct_page_put_has_native_hard_deadline() -> None:
    release = threading.Event()

    class _Store:
        @staticmethod
        def register_buffer(ptr, size):
            return 0

        @staticmethod
        def batch_put_from_multi_buffers(keys, ptrs, sizes, replica):
            assert release.wait(5)
            return [0] * len(keys)

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector._external_put_lock = asyncio.Lock()
    connector._external_buffers = {}
    connector._inflight_put_tasks = set()
    connector.store = _Store()
    connector.replica_config = object()
    connector.config = SimpleNamespace(
        transfer_timeout=0.01,
        remote_fill_native_hard_timeout_ms=20,
    )
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 1)
    owner = torch.empty(16, dtype=torch.uint8)
    timer = threading.Timer(0.1, release.set)
    timer.start()
    try:
        with pytest.raises(NativeExternalPageTransferUnknownError):
            asyncio.run(
                connector.batched_put_external_pages(
                    [_key(13)],
                    [[owner.data_ptr()]],
                    [[16]],
                    (owner,),
                    None,
                    "request",
                )
            )
    finally:
        release.set()
        timer.join()


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
        metadata=SimpleNamespace(chunk_size=4),
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

    assert metadata.call_count == 2
    assert backend.batched_allocate.call_count == 0
    assert backend.allocate.call_count == len(keys)
    assert [value[2] for value in key_metadata] == [
        MemoryFormat.KV_MLA_LATENT_FMT,
        MemoryFormat.KV_DSA_INDEX_FMT,
        MemoryFormat.KV_MLA_LATENT_FMT,
    ]
    assert mode == "individual"


@pytest.mark.parametrize(
    "fmt",
    (MemoryFormat.KV_MLA_LATENT_FMT, MemoryFormat.KV_DSA_INDEX_FMT),
)
def test_mooncake_partial_flat_buffer_uses_exact_token_bytes(fmt) -> None:
    backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=4),
        batched_allocate=Mock(return_value=[_MemoryObj()]),
        allocate=Mock(),
    )
    connector = object.__new__(MooncakestoreConnector)
    connector._metadata_for_raw_key = lambda _key: (
        [torch.Size([16])],
        [torch.float16],
        fmt,
        8,
    )
    connector.local_cpu_backend = backend
    connector._page_first_multi_buffer = True
    key = LayerCacheEngineKey(
        "test",
        1,
        0,
        9,
        torch.float16,
        {MOONCAKE_VALID_TOKENS_TAG: 3},
        layer_id=0,
        kv_group=1 if fmt == MemoryFormat.KV_DSA_INDEX_FMT else 0,
    )

    connector._allocate_zero_copy_buffers([key])

    assert backend.batched_allocate.call_args.args[0] == [torch.Size([12])]


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
    connector._page_num_layers = 2
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


@pytest.mark.parametrize("layout", ("page", "legacy"))
def test_mooncake_cancelled_get_drains_native_read(layout: str) -> None:
    entered = threading.Event()
    release = threading.Event()
    memory_obj = _MemoryObj()
    native_validity = []

    class _BlockingStore:
        def batch_get_into_multi_buffers(self, *_args):
            entered.set()
            native_validity.append(memory_obj.is_valid())
            assert release.wait(5)
            native_validity.append(memory_obj.is_valid())
            return [memory_obj.get_size()]

        def batch_get_into(self, *_args):
            entered.set()
            native_validity.append(memory_obj.is_valid())
            assert release.wait(5)
            native_validity.append(memory_obj.is_valid())
            return [memory_obj.get_size()]

    connector = object.__new__(MooncakestoreConnector)
    connector.store = _BlockingStore()
    connector._allocate_zero_copy_buffers = lambda _keys: (
        [memory_obj],
        [([torch.Size([8])], [torch.float16], MemoryFormat.KV_MLA_LATENT_FMT, 2)],
        "individual",
    )
    connector.meta_shapes = [torch.Size([8])]
    connector.meta_dtypes = [torch.float16]
    connector.meta_fmt = MemoryFormat.KV_MLA_LATENT_FMT

    async def cancel_get() -> None:
        if layout == "page":
            get = connector._batch_get_pages([_key(1)], [("page-1", [0])])
        else:
            get = connector._batch_get_into_legacy([_key(1)])
        task = asyncio.create_task(get)
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert memory_obj.is_valid()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_get())

    assert native_validity == [True, True]
    assert memory_obj.ref_count == 0


def test_mooncake_direct_external_get_registers_storage_and_validates_bytes() -> None:
    calls = []

    class _Store:
        def register_buffer(self, ptr, size):
            calls.append(("register", ptr, size))
            return 0

        def unregister_buffer(self, ptr):
            calls.append(("unregister", ptr))
            return 0

        def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
            calls.append(("get", keys, ptrs, sizes))
            return [sum(page_sizes) for page_sizes in sizes]

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 2)
    connector._external_buffers = {}
    connector._external_put_lock = asyncio.Lock()
    connector._inflight_put_tasks = set()
    connector.store = _Store()
    owner = torch.empty(32, dtype=torch.uint8)

    asyncio.run(
        connector.batched_get_external_pages(
            [_key(1)], [[owner.data_ptr()]], [[32]], (owner,), "request"
        )
    )
    asyncio.run(
        connector.batched_get_external_pages(
            [_key(1)], [[owner.data_ptr()]], [[32]], (owner,), "request"
        )
    )

    assert [call[0] for call in calls].count("register") == 1
    assert [call[0] for call in calls].count("get") == 2


def test_mooncake_cancelled_direct_external_get_drains_native_read() -> None:
    entered = threading.Event()
    release = threading.Event()

    class _Store:
        @staticmethod
        def batch_get_into_multi_buffers(keys, ptrs, sizes):
            entered.set()
            assert release.wait(5)
            return [sum(page_sizes) for page_sizes in sizes]

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.config = SimpleNamespace(transfer_timeout=5)
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 2)
    connector._register_external_owners = lambda owners: None
    connector._external_put_lock = asyncio.Lock()
    connector._inflight_put_tasks = set()
    connector.store = _Store()
    owner = torch.empty(32, dtype=torch.uint8)

    async def cancel_get() -> None:
        task = asyncio.create_task(
            connector.batched_get_external_pages(
                [_key(1)], [[owner.data_ptr()]], [[32]], (owner,), "request"
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert connector._external_put_lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not connector._external_put_lock.locked()

    asyncio.run(cancel_get())


def test_external_page_reuses_registered_cpu_allocator_storage() -> None:
    calls = []
    owner = torch.empty(64, dtype=torch.uint8)
    storage = owner.untyped_storage()
    connector = object.__new__(MooncakestoreConnector)
    connector.registered_buffer_ptr = storage.data_ptr()
    connector.registered_buffer_size = storage.nbytes()
    connector._external_buffers = {}
    connector.store = SimpleNamespace(
        register_buffer=lambda ptr, size: calls.append(("register", ptr, size)),
        unregister_buffer=lambda ptr: calls.append(("unregister", ptr)),
    )

    connector._register_external_owners((owner[8:16],))

    assert calls == []
    assert connector._external_buffers == {}


@pytest.mark.parametrize("pointer,size", ((99, 1), (119, 2)))
def test_direct_external_buffer_validation_rejects_partial_owner_overlap(
    pointer, size
) -> None:
    class _Storage:
        def __init__(self, pointer, size):
            self._pointer = pointer
            self._size = size

        def data_ptr(self):
            return self._pointer

        def nbytes(self):
            return self._size

    class _Owner:
        def __init__(self, pointer, size):
            self._storage = _Storage(pointer, size)

        def untyped_storage(self):
            return self._storage

    owners = (_Owner(100, 20), _Owner(105, 5))
    MooncakestoreConnector._validate_external_buffer_owners(
        [[110]], [[10]], owners
    )
    with pytest.raises(ValueError, match="outside registered storage"):
        MooncakestoreConnector._validate_external_buffer_owners(
            [[pointer]], [[size]], owners
        )


def test_external_page_size_cache_preserves_legacy_layer_layouts() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_num_layers = 2
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (
        None,
        None,
        None,
        2 + key.layer_id,
    )

    connector._external_page_key(_layer_key(1, 0), [16])
    connector._external_page_key(_layer_key(1, 1), [24])


@pytest.mark.parametrize("statuses", ([], [31]))
def test_mooncake_direct_external_get_rejects_partial_status(statuses) -> None:
    owner = torch.empty(32, dtype=torch.uint8)
    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 2)
    connector._register_external_owners = lambda owners: None
    connector._external_put_lock = asyncio.Lock()
    connector._inflight_put_tasks = set()
    connector.store = SimpleNamespace(
        batch_get_into_multi_buffers=lambda *args: statuses
    )

    with pytest.raises(RuntimeError, match="invalid status count|failed or was short"):
        asyncio.run(
            connector.batched_get_external_pages(
                [_key(1)], [[owner.data_ptr()]], [[32]], (owner,), "request"
            )
        )


def test_remote_direct_get_timeout_drains_before_next_registration() -> None:
    calls = []
    first_started = threading.Event()
    release_first = threading.Event()

    class _Store:
        def register_buffer(self, ptr, size):
            calls.append(("register", ptr, size))
            return 0

        def unregister_buffer(self, ptr):
            calls.append(("unregister", ptr))
            return 0

        def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
            calls.append(("get", keys))
            if not first_started.is_set():
                first_started.set()
                release_first.wait()
            return [sum(page_sizes) for page_sizes in sizes]

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.config = SimpleNamespace(transfer_timeout=60)
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=8)
    )
    connector._metadata_for_raw_key = lambda key: (None, None, None, 2)
    connector._external_buffers = {}
    connector._external_put_lock = asyncio.Lock()
    connector._inflight_put_tasks = set()
    connector.store = _Store()

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    backend = object.__new__(RemoteBackend)
    backend.connection = connector
    backend.loop = loop
    backend.config = SimpleNamespace(blocking_timeout_secs=0.01)
    backend._mla_worker_id_as0_mode = False
    owners = [torch.empty(32, dtype=torch.uint8) for _ in range(2)]
    errors = []

    def load(index):
        try:
            backend.batched_get_external_pages(
                [_key(index + 1)],
                [[owners[index].data_ptr()]],
                [[32]],
                (owners[index],),
                f"request-{index}",
            )
        except TimeoutError as error:
            errors.append(error)

    first = threading.Thread(target=load, args=(0,))
    second = threading.Thread(target=load, args=(1,))
    try:
        first.start()
        assert first_started.wait(timeout=1)
        backend.config.blocking_timeout_secs = 1
        second.start()
        second.join(timeout=0.05)
        assert second.is_alive()
        assert [call[0] for call in calls] == ["register", "get"]
        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)
        assert not first.is_alive() and not second.is_alive()
        assert len(errors) == 1
        assert [call[0] for call in calls] == [
            "register",
            "get",
            "unregister",
            "register",
            "get",
        ]
    finally:
        release_first.set()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1)
        loop.close()


@pytest.mark.parametrize("canonical_keys", [False, True])
def test_mooncake_layer_page_get_allocates_exact_full_and_tail_pages(
    monkeypatch: pytest.MonkeyPatch,
    canonical_keys: bool,
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
            self.metadata = SimpleNamespace(chunk_size=8)

        def batched_allocate_layer_pages(self, *args, **kwargs):
            return self.allocator.batched_allocate_layer_pages(*args, **kwargs)

        def batched_submit_layer_pages(self, keys, pages):
            self.submitted = (keys, pages)

    connector = object.__new__(MooncakestoreConnector)
    connector._layer_merged_pages = True
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.local_cpu_backend = _Backend()
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
    base_keys = [
        _key(1),
        CacheEngineKey(
            "test",
            1,
            0,
            2,
            torch.float16,
            {MOONCAKE_VALID_TOKENS_TAG: 3},
        ),
    ]
    keys = (
        base_keys
        if canonical_keys
        else [key.get_first_layer() for key in base_keys]
    )
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
        mooncake_page_key(key, 2) for key in base_keys
    ]
    assert connector.store.args[2] == [[16, 16], [6, 6]]
    assert connector.store.args[1] == [
        [page.layer_data_ptr(0), page.layer_data_ptr(1)] for page in pages
    ]
    assert len(metadata_calls) == 1
    submitted_keys, submitted_pages = connector.local_cpu_backend.submitted
    assert submitted_pages == pages
    assert submitted_keys == base_keys
    event, fields = events.pop()
    assert event == "mooncake_page_get"
    assert fields["layout"] == "layer_merged"
    assert fields["kv_group"] == 0
    assert fields["kv_groups"] == [0]
    assert fields["pages"] == 2
    assert fields["submitted_pages"] == 2
    assert fields["completed_pages"] == 2
    assert fields["layers"] == 2
    assert fields["buffers"] == 4
    assert fields["bytes"] == 44
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
    connector._page_num_layers = 3
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
    connector._page_num_layers = 2
    connector.store = _Store()
    keys = [_layer_key(1, layer_id) for layer_id in range(2)]

    assert connector.batched_contains(keys[:1]) == 0
    assert connector.batched_contains(keys) == 2
    assert connector.batched_contains_layer_pages(keys[:1]) == 1
    assert connector.batched_contains_layer_pages([keys[0].without_layer()]) == 1


def test_partial_page_lookup_falls_back_to_legacy_layer_keys() -> None:
    class _Store:
        seen = []

        @staticmethod
        def is_exist(key):
            return int(key.startswith("__lmcache_page_v1__"))

        @classmethod
        def batch_is_exist(cls, keys):
            cls.seen.append(keys)
            return [
                int(
                    not key.startswith("__lmcache_page_v1__")
                    and "internal.valid_tokens" not in key
                )
                for key in keys
            ]

    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.store = _Store()
    keys = [
        LayerCacheEngineKey(
            "test",
            1,
            0,
            9,
            torch.float16,
            {MOONCAKE_VALID_TOKENS_TAG: 3},
            layer_id=layer_id,
        )
        for layer_id in range(2)
    ]

    assert connector.batched_contains(keys) == 2
    assert len(_Store.seen) == 2
    assert all("internal.valid_tokens" not in key for key in _Store.seen[1])
    assert not asyncio.run(connector.exists(keys[0]))

    del _Store.batch_is_exist
    assert connector.batched_contains_layer_pages(keys[:1]) == 1


def test_mooncake_page_put_merges_exact_partial_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PageStore:
        def __init__(self) -> None:
            self.page_args = None
            self.legacy_args = None

        def batch_put_from_multi_buffers(self, *args):
            self.page_args = args
            return [0] * len(args[0])

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
    connector._metadata_for_raw_key = lambda _key: (
        [],
        [],
        MemoryFormat.KV_MLA_LATENT_FMT,
        4,
    )
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
    full_keys = [
        _layer_key(1, 0),
        _layer_key(1, 1),
    ]
    partial_keys = [
        LayerCacheEngineKey(
            "test",
            1,
            0,
            2,
            torch.float16,
            {MOONCAKE_VALID_TOKENS_TAG: 2},
            layer_id=layer_id,
        )
        for layer_id in range(2)
    ]
    keys = [full_keys[0], partial_keys[0], full_keys[1], partial_keys[1]]
    memory_objs = [
        _MemoryObj(16, 100),
        _MemoryObj(8, 200),
        _MemoryObj(16, 300),
        _MemoryObj(8, 400),
    ]

    asyncio.run(connector._batched_put_zero_copy(keys, memory_objs))

    assert connector.store.page_args[1] == [[100, 300], [200, 400]]
    assert connector.store.page_args[2] == [[16, 16], [8, 8]]
    assert connector.store.legacy_args is None
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)
    event, fields = events[0]
    assert event == "mooncake_page_put"
    assert fields["pages"] == 2
    assert fields["buffers"] == 4
    assert fields["bytes"] == 48
    assert fields["kv_groups"] == [0]
    assert fields["first_page_key"] == connector.store.page_args[0][0]
    assert fields["last_page_key"] == connector.store.page_args[0][-1]
    assert fields["legacy_objects"] == 0
    assert fields["preferred_segments"] == []
    assert fields["preferred_pages"] == 0
    assert fields["default_pages"] == 2
    assert fields["placement_batches"] == 1


def test_mooncake_cpu_page_put_uses_latent_segment_hint() -> None:
    class _ReplicateConfig:
        pass

    class _PageStore:
        def __init__(self) -> None:
            self.calls = []

        def batch_put_from_multi_buffers(self, keys, ptrs, sizes, replica):
            self.calls.append((keys, ptrs, sizes, replica))
            return [0] * len(keys)

    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector._replicate_config_cls = _ReplicateConfig
    connector.replica_config = _ReplicateConfig()
    connector.replica_config.replica_num = 1
    connector._inflight_put_tasks = set()
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=4)
    )
    connector._metadata_for_raw_key = lambda _key: (
        [],
        [],
        MemoryFormat.KV_MLA_LATENT_FMT,
        4,
    )
    connector.store = _PageStore()
    default = _key(20)
    hinted = CacheEngineKey(
        "test",
        1,
        0,
        21,
        torch.float16,
        request_configs={
            "lmcache.mooncake_preferred_segment": "decoder-segment"
        },
        kv_group=0,
    )
    keys = [
        default.get_layer(0),
        hinted.get_layer(0),
        default.get_layer(1),
        hinted.get_layer(1),
    ]
    memory_objs = [
        _MemoryObj(16, 100),
        _MemoryObj(16, 200),
        _MemoryObj(16, 300),
        _MemoryObj(16, 400),
    ]

    asyncio.run(connector._batched_put_zero_copy(keys, memory_objs))

    assert len(connector.store.calls) == 2
    assert connector.store.calls[0][3] is connector.replica_config
    assert connector.store.calls[1][3].preferred_segment == "decoder-segment"
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)


def test_mooncake_cpu_page_put_rejects_inconsistent_layer_hint() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=4)
    )
    connector._metadata_for_raw_key = lambda _key: (
        [],
        [],
        MemoryFormat.KV_MLA_LATENT_FMT,
        4,
    )
    native_calls = []
    connector.store = SimpleNamespace(
        batch_put_from_multi_buffers=lambda *args: native_calls.append(args)
    )
    base = _key(22)
    hinted = CacheEngineKey(
        "test",
        1,
        0,
        22,
        torch.float16,
        request_configs={
            "lmcache.mooncake_preferred_segment": "decoder-segment"
        },
        kv_group=0,
    )

    with pytest.raises(ValueError, match="inconsistent KV group"):
        asyncio.run(
            connector._batched_put_zero_copy(
                [base.get_layer(0), hinted.get_layer(1)],
                [_MemoryObj(16, 100), _MemoryObj(16, 200)],
            )
        )

    assert native_calls == []


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
        valid_tokens=[4],
        full_tokens=4,
    )
    assert pages is not None
    page = pages[0]
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
