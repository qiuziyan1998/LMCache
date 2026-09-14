# SPDX-License-Identifier: Apache-2.0
# Standard
from copy import deepcopy
from types import SimpleNamespace
import os
import random
import shlex
import subprocess
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import (
    mock_up_broadcast_fn,
    mock_up_broadcast_object_fn,
)
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus, EventType
import lmcache.v1.cache_engine as cache_engine_module

# Local
from .utils import (
    DummyLMCacheAsyncLookupServer,
    check_paged_kv_cache_equal,
    create_gpu_connector,
    dumb_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
    has_cufile,
    recover_engine_states,
)


def get_expected_count(token_len, save_unfull_chunk, chunk_size):
    """Calculate expected token count based on save_unfull_chunk setting.

    Args:
        token_len: Total token length
        save_unfull_chunk: Whether to save partial chunks
        chunk_size: Chunk size for alignment

    Returns:
        If save_unfull_chunk is True, returns token_len as-is.
        Otherwise, returns chunk-aligned count (rounded down).
    """
    if save_unfull_chunk:
        return token_len
    return (token_len // chunk_size) * chunk_size


@pytest.mark.parametrize(
    ("chunk_locations", "expected_cleanup_locations"),
    [
        (
            ("LocalCPUBackend", "LocalCPUBackend"),
            ["LocalCPUBackend"],
        ),
        (
            ("LocalCPUBackend", "RemoteBackend"),
            ["LocalCPUBackend", "RemoteBackend"],
        ),
    ],
)
def test_layerwise_retrieve_preserves_cleanup_locations(
    monkeypatch,
    chunk_locations,
    expected_cleanup_locations,
):
    class FakeKey:
        def __init__(self, chunk_id):
            self.chunk_id = chunk_id

        def split_layers(self, num_layers):
            return [SimpleNamespace(chunk_id=self.chunk_id) for _ in range(num_layers)]

    class FakeMemoryObj:
        def __init__(self, location, layer_id):
            self.location = location
            self.layer_id = layer_id
            self.ref_count_down_calls = 0

        def ref_count_down(self):
            self.ref_count_down_calls += 1

    class FakeStorageManager:
        def __init__(self):
            self.returned = []

        def contains(self, key, _retrieve_locations):
            return chunk_locations[key.chunk_id]

        def layerwise_batched_get(self, keys, location=None):
            for layer_id, layer_keys in enumerate(keys):
                mem_objs = [
                    FakeMemoryObj(location, layer_id) for _ in layer_keys
                ]
                self.returned.extend(mem_objs)
                yield SimpleNamespace(result=lambda objs=mem_objs: objs)

    class FakeGPUConnector:
        @staticmethod
        def batched_to_gpu(_starts, _ends, **_kwargs):
            def consume():
                yield
                for _ in range(2):
                    yield
                yield

            return consume()

    monkeypatch.setattr(cache_engine_module, "CacheEngineKey", FakeKey)
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.retrieve_locations = list(dict.fromkeys(chunk_locations))
    engine.storage_manager = FakeStorageManager()
    engine.gpu_connector = FakeGPUConnector()
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: iter(
            [(0, 1, FakeKey(0)), (1, 2, FakeKey(1))]
        )
    )
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _num_tokens: "monitor-id",
        on_retrieve_finished=lambda _monitor_id, _num_tokens: None,
    )
    engine.is_healthy = lambda: True
    engine._get_req_id = lambda _kwargs: "req"
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: False
    engine._is_passive = lambda: False
    cleanup_calls = []
    engine._maybe_unpin_retrieved_objs = (
        lambda mem_objs, location: cleanup_calls.append(
            (location, list(mem_objs))
        )
    )

    list(engine.retrieve_layer([1, 2]))

    assert [location for location, _ in cleanup_calls] == (
        expected_cleanup_locations
    )
    for location, mem_objs in cleanup_calls:
        assert mem_objs
        assert all(mem_obj.location == location for mem_obj in mem_objs)
    assert all(
        mem_obj.ref_count_down_calls == 1
        for mem_obj in engine.storage_manager.returned
    )


def test_layerwise_retrieve_forwards_per_layer_request(monkeypatch):
    class FakeKey:
        def split_layers(self, num_layers):
            return [SimpleNamespace(layer_id=i) for i in range(num_layers)]

    class FakeMemoryObj:
        def ref_count_down(self):
            pass

    class FakeStorageManager:
        @staticmethod
        def contains(_key, _locations):
            return "LocalCPUBackend"

        @staticmethod
        def layerwise_batched_get(keys, location=None):
            del location
            for _ in keys:
                obj = FakeMemoryObj()
                yield SimpleNamespace(result=lambda obj=obj: [obj])

    received = []

    class FakeGPUConnector:
        @staticmethod
        def batched_to_gpu(_starts, _ends, **_kwargs):
            payload = yield
            received.append(payload)
            payload = yield
            received.append(payload)
            yield
            yield

    monkeypatch.setattr(cache_engine_module, "CacheEngineKey", FakeKey)
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.storage_manager = FakeStorageManager()
    engine.gpu_connector = FakeGPUConnector()
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: iter([(0, 1, FakeKey())])
    )
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _num_tokens: "monitor-id",
        on_retrieve_finished=lambda _monitor_id, _num_tokens: None,
    )
    engine.is_healthy = lambda: True
    engine._get_req_id = lambda _kwargs: "req"
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: False
    engine._is_passive = lambda: False
    engine._maybe_unpin_retrieved_objs = lambda _objs, _location: None

    retriever = engine.retrieve_layer([1])
    assert int(next(retriever)) == 1
    first = {"slot_mapping": torch.tensor([10])}
    second = {"slot_mapping": torch.tensor([20])}
    assert retriever.send(first) is None
    assert retriever.send(second) is None
    next(retriever)

    assert received[0]["layer_request"] is first
    assert received[1]["layer_request"] is second


@pytest.mark.parametrize("prime_steps", [1, 2])
def test_layerwise_retrieve_close_releases_pending_memory_objs_once(
    monkeypatch,
    prime_steps,
):
    class FakeKey:
        def split_layers(self, num_layers):
            return [SimpleNamespace(layer_id=i) for i in range(num_layers)]

    class FakeMemoryObj:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.ref_count_down_calls = 0

        def ref_count_down(self):
            self.ref_count_down_calls += 1

    class FakeFuture:
        def __init__(self, memory_obj):
            self.memory_obj = memory_obj
            self.result_calls = 0

        def result(self):
            self.result_calls += 1
            return [self.memory_obj]

    class FakeStorageManager:
        def __init__(self):
            self.returned = []
            self.futures = []

        @staticmethod
        def contains(_key, _locations):
            return "LocalCPUBackend"

        def layerwise_batched_get(self, keys, location=None):
            del location
            for layer_id, _ in enumerate(keys):
                memory_obj = FakeMemoryObj(layer_id)
                future = FakeFuture(memory_obj)
                self.returned.append(memory_obj)
                self.futures.append(future)
                yield future

    consumer_closes = []

    class FakeGPUConnector:
        @staticmethod
        def batched_to_gpu(_starts, _ends, **_kwargs):
            try:
                yield
                yield
                yield
                yield
            finally:
                consumer_closes.append(True)

    monkeypatch.setattr(cache_engine_module, "CacheEngineKey", FakeKey)
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.storage_manager = FakeStorageManager()
    engine.gpu_connector = FakeGPUConnector()
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: iter([(0, 1, FakeKey())])
    )
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _num_tokens: "monitor-id",
        on_retrieve_finished=lambda _monitor_id, _num_tokens: None,
    )
    engine.is_healthy = lambda: True
    engine._get_req_id = lambda _kwargs: "req"
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: False
    engine._is_passive = lambda: False
    cleanup_calls = []
    engine._maybe_unpin_retrieved_objs = (
        lambda mem_objs, location: cleanup_calls.append(
            (location, list(mem_objs))
        )
    )

    retriever = engine.retrieve_layer([1])
    next(retriever)
    if prime_steps == 2:
        retriever.send(None)
    retriever.close()
    # Repeated close must not release an already relinquished reference again.
    retriever.close()

    assert len(engine.storage_manager.returned) == prime_steps
    assert all(
        memory_obj.ref_count_down_calls == 1
        for memory_obj in engine.storage_manager.returned
    )
    assert all(
        future.result_calls == 1
        for future in engine.storage_manager.futures
    )
    assert consumer_closes == [True]
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][0] == "LocalCPUBackend"
    assert cleanup_calls[0][1] == engine.storage_manager.returned


@pytest.mark.parametrize(
    "completion",
    ["normal", "abort", "send_failure", "terminal_sync_failure"],
)
def test_deferred_layerwise_retrieve_holds_sources_until_last_layer_entry(
    monkeypatch,
    completion,
):
    events = []

    class FakeKey:
        def split_layers(self, num_layers):
            return [SimpleNamespace(layer_id=i) for i in range(num_layers)]

    class FakeMemoryObj:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.ref_count_down_calls = 0

        def ref_count_down(self):
            self.ref_count_down_calls += 1
            events.append(("release", self.layer_id))

    class FakeStorageManager:
        def __init__(self):
            self.returned = []

        @staticmethod
        def contains(_key, _locations):
            return "LocalCPUBackend"

        def layerwise_batched_get(self, keys, location=None):
            del location
            for layer_id, _ in enumerate(keys):
                memory_obj = FakeMemoryObj(layer_id)
                self.returned.append(memory_obj)
                yield SimpleNamespace(result=lambda obj=memory_obj: [obj])

    class FakeGPUConnector:
        @staticmethod
        def batched_to_gpu(_starts, _ends, **_kwargs):
            synchronized = False
            try:
                payload = yield
                events.append(("h2d", 0, payload))
                if completion == "send_failure":
                    raise RuntimeError("H2D send failed after launch")
                payload = yield
                events.append(("h2d", 1, payload))
                # send(final layer) stops here without synchronizing.
                yield
                if completion == "terminal_sync_failure":
                    events.append(("consumer_sync_failed",))
                    raise RuntimeError("terminal load-stream sync failed")
                events.append(("consumer_sync",))
                synchronized = True
                yield
            finally:
                # Abort closes the consumer at the final gate. Model the
                # Ascend connector's synchronized close before host release.
                if not synchronized and completion == "abort":
                    events.append(("consumer_sync",))

    monkeypatch.setattr(cache_engine_module, "CacheEngineKey", FakeKey)
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.storage_manager = FakeStorageManager()
    engine.gpu_connector = FakeGPUConnector()
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: iter([(0, 1, FakeKey())])
    )
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _num_tokens: "monitor-id",
        on_retrieve_finished=lambda _monitor_id, _num_tokens: None,
    )
    engine.is_healthy = lambda: True
    engine._get_req_id = lambda _kwargs: "req"
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: False
    engine._is_passive = lambda: False
    engine._maybe_unpin_retrieved_objs = (
        lambda _objs, _location: events.append(("unpin",))
    )

    retriever = engine.retrieve_layer(
        [1],
        deferred_layerwise_get=True,
    )
    next(retriever)
    if completion == "send_failure":
        with pytest.raises(RuntimeError, match="H2D send failed after launch"):
            retriever.send({"slot_mapping": torch.tensor([0])})
        assert [event[0] for event in events] == ["h2d"]
        assert len(engine.storage_manager.returned) == 1
        assert engine.storage_manager.returned[0].ref_count_down_calls == 0
        assert [event[0] for event in events].count("unpin") == 0
        assert engine._unsafe_layerwise_retrieve_sources == (
            engine.storage_manager.returned
        )
        return
    retriever.send({"slot_mapping": torch.tensor([0])})
    # N-1 submits the final H2D and returns at the cleanup gate.
    assert retriever.send({"slot_mapping": torch.tensor([1])}) is None
    assert [event[0] for event in events] == ["h2d", "h2d"]
    assert all(
        memory_obj.ref_count_down_calls == 0
        for memory_obj in engine.storage_manager.returned
    )

    if completion == "abort":
        # Adapter abort resets/synchronizes all transfer state before close.
        events.append(("reset",))
        retriever.close()
        event_names = [event[0] for event in events]
        assert event_names.index("reset") < event_names.index("consumer_sync")
        assert event_names.index("consumer_sync") < event_names.index("release")
    elif completion == "normal":
        # The adapter's last-layer entry bank fence happens before this resume.
        events.append(("last_layer_wait",))
        ret_mask = retriever.send(None)
        assert bool(torch.all(ret_mask))
        event_names = [event[0] for event in events]
        assert event_names.index("last_layer_wait") < event_names.index(
            "consumer_sync"
        )
        assert event_names.index("consumer_sync") < event_names.index("release")
        retriever.close()
    else:
        events.append(("last_layer_wait",))
        with pytest.raises(RuntimeError, match="terminal load-stream sync failed"):
            retriever.send(None)
        assert all(
            memory_obj.ref_count_down_calls == 0
            for memory_obj in engine.storage_manager.returned
        )
        assert [event[0] for event in events].count("unpin") == 0
        assert engine._unsafe_layerwise_retrieve_sources == (
            engine.storage_manager.returned
        )
        return

    assert all(
        memory_obj.ref_count_down_calls == 1
        for memory_obj in engine.storage_manager.returned
    )
    assert [event[0] for event in events].count("unpin") == 1


def test_deferred_layerwise_store_persists_only_after_source_done(monkeypatch):
    class FakeKey:
        def split_layers(self, num_layers):
            return [SimpleNamespace(layer_id=i) for i in range(num_layers)]

    class FakeMemoryObj:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.valid = True

        @staticmethod
        def get_size():
            return 1

        def is_valid(self):
            return self.valid

        def ref_count_down(self):
            self.valid = False

    events = []

    class FakeFuture:
        def __init__(self, layer_id):
            self.layer_id = layer_id

        def result(self):
            events.append(("persist_done", self.layer_id))

    class FakeStorageManager:
        @staticmethod
        def batched_allocate(_shape, _dtype, batch_size, **_kwargs):
            return [FakeMemoryObj(i) for i in range(batch_size)]

        @staticmethod
        def batched_put(keys, _memory_objs, location=None):
            del location
            layer_id = keys[0].layer_id
            events.append(("persist_submit", layer_id))
            # Exceed the old in-forward threshold with one completed source.
            # None of these futures may be waited until the final drain.
            return [FakeFuture(layer_id) for _ in range(8)]

    commands = []

    class FakeGPUConnector:
        @staticmethod
        def get_shape(_num_tokens, kv_group=0):
            del kv_group
            return torch.Size([1])

        @staticmethod
        def batched_from_gpu(_memory_objs, _starts, _ends, **_kwargs):
            command = yield None
            commands.append(command)
            command = yield None
            events.append(("source_done", 0))
            commands.append(command)
            yield 0
            events.append(("source_done", 1))
            yield 1

    monkeypatch.setattr(cache_engine_module, "CacheEngineKey", FakeKey)
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = FakeStorageManager()
    engine.gpu_connector = FakeGPUConnector()
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: iter([(0, 1, FakeKey())])
    )
    engine.stats_monitor = SimpleNamespace(
        on_store_request=lambda _num_tokens: "monitor-id",
        on_store_finished=lambda _monitor_id, _num_tokens: None,
    )
    engine.config = SimpleNamespace(
        get_extra_config_value=lambda _name, default: default
    )
    engine.store_location = "LocalCPUBackend"
    engine.kv_events_enabled = False
    engine.is_healthy = lambda: True
    engine.is_frozen = lambda: False
    engine._is_passive = lambda: False
    engine._get_req_id = lambda _kwargs: "req"
    engine._log_kvcache_for_check = lambda **_kwargs: None
    engine._shared_cpu_dtype_for_kv_group = lambda _group: torch.float32
    engine._memory_format_for_kv_group = lambda _group: None
    engine._layerwise_chunk_fully_stored = lambda *_args, **_kwargs: False

    storer = engine.store_layer(
        [1],
        deferred_layerwise_put=True,
        req_id="req",
    )
    assert next(storer) is None
    first = {"slot_mapping": torch.tensor([10])}
    second = {"slot_mapping": torch.tensor([20])}
    # Pre-HCOM: only the D2H submission generator is advanced.
    assert storer.send(first) is None
    assert events == []
    # Post-HCOM: finish layer 0 and make the storer ready for layer 1.
    assert next(storer) is None
    assert events == []

    # Even when the GPU connector reports an old source buffer complete, the
    # pre-HCOM callback must not call storage_manager.batched_put.
    assert storer.send(second) is None
    assert events == [("source_done", 0)]
    # The explicit post-HCOM finish is the first point allowed to publish it.
    assert next(storer) is None
    assert events == [("source_done", 0), ("persist_submit", 0)]
    result = next(storer)

    assert result is not None
    assert result.request_id == "req"
    assert commands == [first, second]
    assert events == [
        ("source_done", 0),
        ("persist_submit", 0),
        ("source_done", 1),
        ("persist_submit", 1),
        *[("persist_done", 0)] * 8,
        *[("persist_done", 1)] * 8,
    ]


def test_deferred_layerwise_store_drains_multiple_rotating_banks(monkeypatch):
    class FakeKey:
        def split_layers(self, num_layers):
            return [SimpleNamespace(layer_id=i) for i in range(num_layers)]

    class FakeMemoryObj:
        valid = True

        @staticmethod
        def get_size():
            return 1

        def is_valid(self):
            return self.valid

        def ref_count_down(self):
            self.valid = False

    persisted = []

    class FakeStorageManager:
        @staticmethod
        def batched_allocate(_shape, _dtype, batch_size, **_kwargs):
            return [FakeMemoryObj() for _ in range(batch_size)]

        @staticmethod
        def batched_put(keys, _memory_objs, location=None):
            del location
            persisted.append(keys[0].layer_id)
            return []

    class FakeGPUConnector:
        @staticmethod
        def get_shape(_num_tokens, kv_group=0):
            del kv_group
            return torch.Size([1])

        @staticmethod
        def batched_from_gpu(_memory_objs, _starts, _ends, **_kwargs):
            yield None
            for _ in range(4):
                yield None
            yield 0
            yield 1
            yield 2
            yield 3

    monkeypatch.setattr(cache_engine_module, "CacheEngineKey", FakeKey)
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 4
    engine.storage_manager = FakeStorageManager()
    engine.gpu_connector = FakeGPUConnector()
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: iter([(0, 1, FakeKey())])
    )
    engine.stats_monitor = SimpleNamespace(
        on_store_request=lambda _num_tokens: "monitor-id",
        on_store_finished=lambda _monitor_id, _num_tokens: None,
    )
    engine.config = SimpleNamespace(
        get_extra_config_value=lambda _name, default: default
    )
    engine.store_location = "LocalCPUBackend"
    engine.kv_events_enabled = False
    engine.is_healthy = lambda: True
    engine.is_frozen = lambda: False
    engine._is_passive = lambda: False
    engine._get_req_id = lambda _kwargs: "req"
    engine._log_kvcache_for_check = lambda **_kwargs: None
    engine._shared_cpu_dtype_for_kv_group = lambda _group: torch.float32
    engine._memory_format_for_kv_group = lambda _group: None
    engine._layerwise_chunk_fully_stored = lambda *_args, **_kwargs: False

    storer = engine.store_layer(
        [1],
        deferred_layerwise_put=True,
        req_id="req",
    )
    assert next(storer) is None
    for layer in range(4):
        assert storer.send({"layer": layer}) is None
        assert next(storer) is None

    result = next(storer)
    assert result is not None
    assert result.request_id == "req"
    assert persisted == [0, 1, 2, 3]


@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_same_retrieve_store(save_unfull_chunk, autorelease_v1):
    device = "cuda"
    num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)

    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    original_retrieved_cache = deepcopy(retrieved_cache)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    # Check the kv cache and the retrieval buffer are not the same
    check_paged_kv_cache_equal(retrieved_cache, original_retrieved_cache, slot_mapping)
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(retrieved_cache, kv_cache, slot_mapping)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size, remote_url=None, save_unfull_chunk=save_unfull_chunk
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test retrieve empty """
    ret_mask = engine.retrieve(
        tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
    )
    recover_engine_states(engine)

    length = torch.sum(ret_mask)
    assert length == 0
    check_paged_kv_cache_equal(retrieved_cache, original_retrieved_cache, slot_mapping)
    """ test store """
    engine.store(tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    recover_engine_states(engine)

    """ Store is async. Need to wait for the store to finish """
    expected_count = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    timeout = 1.5
    start_time = time.time()
    while engine.lookup(tokens) < expected_count:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    ret_mask = engine.retrieve(
        tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
    )
    recover_engine_states(engine)

    length = torch.sum(ret_mask)
    assert length == expected_count
    check_paged_kv_cache_equal(retrieved_cache, kv_cache, slot_mapping[:expected_count])


@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("backend", ["cpu", "local_disk", "remote", "remote_cachegen"])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_retrieve_prefix(
    chunk_size, backend, save_unfull_chunk, lmserver_v1_process, autorelease_v1
):
    url = None
    remote_serde = None
    check_equality = True
    if "remote" in backend:
        url = lmserver_v1_process.server_url
        if backend == "remote_cachegen":
            backend = "remote"
            remote_serde = "cachegen"
            check_equality = False
        else:
            remote_serde = "naive"
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping_full = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping_full[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping_full[-new_num_tokens:], device=device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size,
        backend=backend,
        remote_url=url,
        remote_serde=remote_serde,
        save_unfull_chunk=save_unfull_chunk,
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    recover_engine_states(engine)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2 - t1}")
    """ Compute expected length """
    expected_length = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    """ Store is async. Need to wait for the store to finish """
    if backend == "cpu":
        timeout = 1
        search_range = "LocalCPUBackend"
    elif backend == "local_disk":
        timeout = 30
        search_range = "LocalDiskBackend"
    elif backend == "remote":
        timeout = 30
        search_range = "RemoteBackend"
    start_time = time.time()
    while engine.lookup(tokens, search_range=search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    # Get actual stored length - may be less than expected if is_last_prefill=False
    # even when save_unfull_chunk=True
    actual_stored_tokens = engine.lookup(torch.cat([tokens, new_tokens]))
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(
        torch.cat([tokens, new_tokens]),
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    recover_engine_states(engine)

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5 - t4}")

    # Use actual stored length (may be chunk-aligned even if save_unfull_chunk=True
    # if is_last_prefill=False)
    assert length == actual_stored_tokens

    if check_equality:
        check_paged_kv_cache_equal(
            kv_cache,
            retrieved_cache,
            torch.cat([slot_mapping, new_slot_mapping])[:actual_stored_tokens],
        )

    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize(
    "backend",
    ["cpu", "local_disk", "remote"],
)
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_store_offset(
    chunk_size, backend, save_unfull_chunk, lmserver_v1_process, autorelease_v1
):
    url = None
    if backend == "remote":
        url = lmserver_v1_process.server_url
    device = "cuda"
    num_tokens = 2000
    num_suffix_tokens = 500
    num_total_tokens = 3000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_total_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_total_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size,
        backend=backend,
        remote_url=url,
        save_unfull_chunk=save_unfull_chunk,
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    engine.store(
        tokens[:num_tokens],
        kvcaches=kv_cache,
        slot_mapping=slot_mapping[:num_tokens],
    )

    offset_chunk_cnt = num_tokens // chunk_size
    offset_length = offset_chunk_cnt * chunk_size
    mask = torch.ones(num_tokens + num_suffix_tokens, device=device)
    mask[:offset_length] = 0
    engine.store(
        tokens[: num_tokens + num_suffix_tokens],
        kvcaches=kv_cache,
        mask=mask,
        slot_mapping=slot_mapping[: num_tokens + num_suffix_tokens],
    )
    recover_engine_states(engine)

    """ Compute expected length """
    total_tokens = num_tokens + num_suffix_tokens
    expected_length = (total_tokens // chunk_size) * chunk_size
    """ Store is async. Need to wait for the store to finish """
    if backend == "cpu":
        timeout = 1
    elif backend == "local_disk":
        timeout = 30
    start_time = time.time()
    while engine.lookup(tokens[: num_tokens + num_suffix_tokens]) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(
        tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
    )
    recover_engine_states(engine)

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5 - t4}")

    assert length == expected_length
    check_paged_kv_cache_equal(
        kv_cache,
        retrieved_cache,
        slot_mapping[:expected_length],
    )

    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("chunk_size", [128])  # , 256])
@pytest.mark.parametrize(
    "backend",
    [
        # "cpu",
        "local_disk"
    ],
)
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_mixed_retrieve(chunk_size, backend, save_unfull_chunk, autorelease_v1):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    slot_mapping_full = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping_full[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping_full[-new_num_tokens:], device=device)

    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size, backend=backend, save_unfull_chunk=save_unfull_chunk
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    engine.store(new_tokens, kvcaches=kv_cache, slot_mapping=new_slot_mapping)
    recover_engine_states(engine)
    """ Store is async. Need to wait for the store to finish """
    expected_length = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    if backend == "cpu":
        timeout = 1
        search_range = "LocalCPUBackend"
    elif backend == "local_disk":
        timeout = 30
        search_range = "LocalDiskBackend"
    start_time = time.time()
    while engine.lookup(tokens, search_range=search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    # Check actual stored tokens for the combined tokens
    # When tokens are stored separately, the total may be chunk-aligned
    actual_stored_total = engine.lookup(
        torch.cat([tokens, new_tokens]), search_range=search_range
    )
    ret_mask = engine.retrieve(
        torch.cat([tokens, new_tokens]),
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    recover_engine_states(engine)
    length = torch.sum(ret_mask)
    # Use actual stored total (may be chunk-aligned even if save_unfull_chunk=True
    # if is_last_prefill=False)
    assert length == actual_stored_total
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[:length],
    )

    """Wait for store to finish"""
    expected_length = get_expected_count(new_num_tokens, save_unfull_chunk, chunk_size)
    start_time = time.time()
    while engine.lookup(new_tokens, search_range=search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test another retrieve """
    ret_mask = engine.retrieve(
        new_tokens, kvcaches=retrieved_cache, slot_mapping=new_slot_mapping
    )
    recover_engine_states(engine)
    length = torch.sum(ret_mask)
    assert length == expected_length
    check_paged_kv_cache_equal(
        retrieved_cache, kv_cache, new_slot_mapping[:expected_length]
    )

    """ insert the mixed kv cache """
    final_tokens = torch.cat([tokens, new_tokens])
    engine.store(
        final_tokens,
        kvcaches=kv_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    recover_engine_states(engine)

    """Wait until store finishes"""
    expected_length = get_expected_count(
        num_tokens + new_num_tokens, save_unfull_chunk, chunk_size
    )
    start_time = time.time()
    while (
        engine.lookup(torch.cat([tokens, new_tokens]), search_range=search_range)
        < expected_length
    ):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ should retrieve the mixed version """
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    ret_mask = engine.retrieve(
        final_tokens,
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    recover_engine_states(engine)
    length = torch.sum(ret_mask)
    assert length == expected_length

    # Only check chunk-aligned tokens when save_unfull_chunk=False
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping])[:expected_length],
    )
    """destroy local disk path"""
    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_store_kv_tensors_mask(save_unfull_chunk, autorelease_v1):
    device = "cuda"
    num_tokens = 1000
    new_num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )

    new_tokens = generate_tokens(new_num_tokens, device)
    final_tokens = torch.cat([tokens, new_tokens])

    slot_mapping_full = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping_full[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping_full[-new_num_tokens:], device=device)

    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size, save_unfull_chunk=save_unfull_chunk
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ Store some tokens with mask """
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    recover_engine_states(engine)
    """Wait until store finishes"""
    expected_count = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    timeout = 1
    start_time = time.time()
    while engine.lookup(tokens) < expected_count:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)

    prefix_length = engine.lookup(tokens)
    assert prefix_length == expected_count, (
        f"Expected {expected_count} prefix tokens, but got {prefix_length}"
    )
    """ Store more tokens """
    # Re-query prefix_length for final_tokens (original flow)
    prefix_length = engine.lookup(final_tokens)
    # Store requires mask False count to be chunk-aligned
    # When save_unfull_chunk=True, prefix_length may not be chunk-aligned,
    # so we need to round it down to chunk boundary for the mask
    num_falses_for_store = (prefix_length // chunk_size) * chunk_size
    kv_tensor_mask = torch.ones_like(final_tokens, dtype=torch.bool)
    kv_tensor_mask[:num_falses_for_store] = False

    engine.store(
        final_tokens,
        mask=kv_tensor_mask,
        kvcaches=kv_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    recover_engine_states(engine)
    """Wait until store finishes"""
    expected_final_count = get_expected_count(
        num_tokens + new_num_tokens, save_unfull_chunk, chunk_size
    )
    timeout = 1
    start_time = time.time()
    while engine.lookup(final_tokens) < expected_final_count:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)

    prefix_length = engine.lookup(final_tokens)
    assert prefix_length == expected_final_count, (
        f"Expected {expected_final_count} prefix tokens, but got {prefix_length}"
    )
    """ retrieve the whole cache """
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )
    ret_mask = engine.retrieve(
        final_tokens,
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    recover_engine_states(engine)
    length = torch.sum(ret_mask)
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[:length],
    )

    """ retrieve cache with some mask:
    """
    # Retrieve requires mask False count to be chunk-aligned
    # Original used chunk_size * 3 (768), which is tokens' chunk-aligned length
    # When save_unfull_chunk=True, we need to ensure chunk alignment
    num_falses = (num_tokens // chunk_size) * chunk_size
    mask = torch.ones_like(final_tokens, dtype=torch.bool)
    mask[:num_falses] = False
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )
    ret_mask = engine.retrieve(
        final_tokens,
        mask=mask,
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    recover_engine_states(engine)
    length = torch.sum(ret_mask)
    full_length = num_tokens + new_num_tokens
    expected_length = full_length - num_falses
    # When save_unfull_chunk=False, retrieved length may be chunk-aligned
    expected_retrieved_length = get_expected_count(
        expected_length, save_unfull_chunk, chunk_size
    )
    assert length == expected_retrieved_length

    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            retrieved_cache,
            kv_cache,
            torch.cat([slot_mapping, new_slot_mapping])[:full_length],
        )
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[num_falses : num_falses + length],
    )

    mask[: num_falses + 5] = False
    with pytest.raises(ValueError):
        engine.retrieve(
            final_tokens,
            mask=mask,
            kvcaches=retrieved_cache,
            slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
        )
        recover_engine_states(engine)


@pytest.mark.parametrize("chunk_size", [128])
@pytest.mark.parametrize(
    "backend",
    [
        "local_cpu_disk_remote",
    ],
)
@pytest.mark.parametrize(
    "retrieve_from",
    [
        "local_cpu",
        "local_disk",
        "remote",
    ],
)
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_hierarchy_retrieve(
    chunk_size,
    backend,
    retrieve_from,
    save_unfull_chunk,
    lmserver_v1_process,
    autorelease_v1,
):
    url = None
    if backend == "local_cpu_disk_remote":
        url = lmserver_v1_process.server_url
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )

    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )

    slot_mapping = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping[-new_num_tokens:], device=device)

    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size,
        backend=backend,
        remote_url=url,
        save_unfull_chunk=save_unfull_chunk,
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    recover_engine_states(engine)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2 - t1}")
    """ Compute expected length """
    expected_length = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    """ Store is async. Need to wait for the store to finish """
    timeout = 1
    start_time = time.time()
    while engine.lookup(tokens) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ Wait until disk save is finished """
    if retrieve_from in ["local_disk", "remote"]:
        engine.storage_manager.clear(locations=["LocalCPUBackend"])
        timeout = 30
        start_time = time.time()
        while (
            engine.lookup(tokens, search_range=["LocalDiskBackend"]) < expected_length
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
    """ Wait until remote save is finished """
    if retrieve_from == "remote":
        engine.storage_manager.clear(locations=["LocalCPUBackend"])
        # FIXME: change this `clear`
        engine.storage_manager.storage_backends["LocalDiskBackend"].dict.clear()
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens, search_range=["RemoteBackend"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
    """ test retrieve """
    t4 = time.perf_counter()
    # Get actual stored length
    actual_stored = engine.lookup(torch.cat([tokens, new_tokens]))
    ret_mask = engine.retrieve(
        torch.cat([tokens, new_tokens]),
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    recover_engine_states(engine)

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5 - t4}")

    # Use actual stored length for assertion
    assert length == actual_stored
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[:actual_stored],
    )

    """ Wait until disk save is finished before deleting the directory"""
    if backend in ["local_cpu_disk"]:
        engine.storage_manager.clear(locations=["LocalCPUBackend"])
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)

    if backend in ["local_cpu_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize(
    "backend",
    [
        "local_cpu_disk",
    ],
)
@pytest.mark.parametrize(
    "prefetch_from",
    [
        "local_disk",
    ],
)
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_prefetch_retrieve(
    backend, prefetch_from, save_unfull_chunk, autorelease_v1
):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    test_lookup_id = "test_lookup_id"

    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )

    slot_mapping = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping[-new_num_tokens:], device=device)

    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size,
        backend=backend,
        enable_async_loading=True,
        save_unfull_chunk=save_unfull_chunk,
    )

    async_lookup_server = DummyLMCacheAsyncLookupServer()
    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        ),
        async_lookup_server=async_lookup_server,
    )

    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    recover_engine_states(engine)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2 - t1}")
    """ Compute expected length """
    # For prefetch retrieve, we need to check what was actually stored
    # Since this test uses async operations, we check the actual lookup result
    expected_length = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    """ Wait for cpu store to finish """
    timeout = 1
    start_time = time.time()
    actual_lookup = engine.lookup(tokens)
    while actual_lookup < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ Delete cpu cache and wait until disk save finishes."""
    if prefetch_from == "local_disk":
        engine.storage_manager.clear(locations=["LocalCPUBackend"])
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.1)
    """ Wait until disk load (prefetch) finishes and delete disk cache"""
    engine.async_lookup_and_prefetch(
        lookup_id=test_lookup_id, tokens=torch.cat([tokens, new_tokens])
    )

    if prefetch_from == "local_disk":
        timeout = 60
        start_time = time.time()
        while (
            engine.event_manager.get_event_status(EventType.LOADING, test_lookup_id)
            != EventStatus.DONE
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
        engine.storage_manager.storage_backends["LocalDiskBackend"].dict.clear()
    """ test retrieve """
    t4 = time.perf_counter()

    # Get actual stored length for retrieve
    actual_stored = engine.lookup(torch.cat([tokens, new_tokens]))
    ret_mask = engine.retrieve(
        torch.cat([tokens, new_tokens])[:actual_stored],
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
        req_id=test_lookup_id,
    )
    recover_engine_states(engine)

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5 - t4}")

    assert length == actual_stored
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[:actual_stored],
    )

    if backend in ["local_cpu_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        "local_disk",
        "remote",
        "local_disk_remote",
        "local_cpu_disk_remote",
    ],
)
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_mem_leak(
    chunk_size, backend, save_unfull_chunk, lmserver_v1_process, autorelease_v1
):
    url = None
    if "remote" in backend:
        url = lmserver_v1_process.server_url

    device = "cuda"
    num_tokens = 2000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size,
        backend=backend,
        remote_url=url,
        save_unfull_chunk=save_unfull_chunk,
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    recover_engine_states(engine)

    expected_length = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    timeout = 30
    """Wait until cpu store finishes"""
    if "cpu" in backend:
        start_time = time.time()
        while engine.lookup(tokens, search_range=["LocalCPUBackend"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
    """Wait until disk store finishes"""
    if "disk" in backend:
        start_time = time.time()
        while (
            engine.lookup(tokens, search_range=["LocalDiskBackend"]) < expected_length
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)

    if "remote" in backend:
        start_time = time.time()
        while engine.lookup(tokens, search_range=["RemoteBackend"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)

    tensor_memory_allocator = (
        engine.storage_manager.allocator_backend.memory_allocator.pin_allocator
    )
    if "cpu" not in backend:
        assert tensor_memory_allocator.total_allocated_size == 0
    else:
        assert tensor_memory_allocator.total_allocated_size > 0

    if "disk" in backend:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        "local_disk",
    ],
)
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.no_shared_allocator
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_paged_retrieve_after_eviction(
    chunk_size, backend, save_unfull_chunk, autorelease_v1
):
    device = "cuda"
    # NOTE: The default backend cache size is 2 GB.
    # 10000 tokens ia around 1.3 GB so a second retrieve will cause an eviction.
    num_tokens = 10000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32)

    tokens_1 = generate_tokens(num_tokens, device)
    tokens_2 = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping_1 = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping_1 = torch.tensor(slot_mapping_1, device=device)
    slot_mapping_2 = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping_2 = torch.tensor(slot_mapping_2, device=device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size,
        backend=backend,
        save_unfull_chunk=save_unfull_chunk,
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    expected_length = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)

    engine.store(tokens_1, kvcaches=kv_cache, slot_mapping=slot_mapping_1)
    recover_engine_states(engine)

    timeout = 30
    if "disk" in backend:
        start_time = time.time()
        while (
            engine.lookup(tokens_1, search_range=["LocalDiskBackend"]) < expected_length
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)

    engine.store(tokens_2, kvcaches=kv_cache, slot_mapping=slot_mapping_2)
    recover_engine_states(engine)

    """Wait until cpu store finishes"""
    if "cpu" in backend:
        start_time = time.time()
        while (
            engine.lookup(tokens_2, search_range=["LocalCPUBackend"]) < expected_length
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
        assert (
            engine.lookup(tokens_1, search_range=["LocalCPUBackend"]) < expected_length
        )

    """Wait until disk store finishes"""
    if "disk" in backend:
        start_time = time.time()
        while (
            engine.lookup(tokens_2, search_range=["LocalDiskBackend"]) < expected_length
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
        assert (
            engine.lookup(tokens_1, search_range=["LocalDiskBackend"]) < expected_length
        )

    ret_mask = engine.retrieve(
        tokens_1,
        kvcaches=retrieved_cache,
        slot_mapping=slot_mapping_1,
    )
    recover_engine_states(engine)
    length = torch.sum(ret_mask)
    assert length < num_tokens

    ret_mask = engine.retrieve(
        tokens_2,
        kvcaches=retrieved_cache,
        slot_mapping=slot_mapping_2,
    )
    recover_engine_states(engine)
    length = torch.sum(ret_mask)
    assert length == expected_length

    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


def test_builder(autorelease_v1):
    instance_id = "test"
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256)
    cfg2 = LMCacheEngineConfig.from_legacy(chunk_size=512)
    connector = None
    should_be_none = LMCacheEngineBuilder.get(instance_id)
    assert should_be_none is None

    _engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            instance_id,
            cfg,
            dumb_metadata(),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    _engine2 = autorelease_v1(LMCacheEngineBuilder.get(instance_id))  # noqa

    with pytest.raises(ValueError):
        LMCacheEngineBuilder.get_or_create(
            instance_id,
            cfg2,
            dumb_metadata(),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_force_store_wait(autorelease_v1):
    device = "cuda"
    num_tokens = 10000
    num_blocks = 5000
    block_size = 16
    dtype = torch.bfloat16

    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)

    connector = create_gpu_connector(1024, 32)

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    num_requests = 8

    def generate_random_slot_mapping(num_blocks, block_size, num_tokens, device):
        slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
        return torch.tensor(slot_mapping, device=device)

    list_tokens = [generate_tokens(num_tokens, device) for _ in range(num_requests)]
    list_slot_mappings = [
        generate_random_slot_mapping(num_blocks, block_size, num_tokens, device)
        for _ in range(num_requests)
    ]

    homedir = os.environ.get("HOME", "/tmp")
    with tempfile.TemporaryDirectory(
        dir=homedir, ignore_cleanup_errors=True
    ) as temp_dir:
        cfg = LMCacheEngineConfig.from_defaults(
            local_cpu=False,
            max_local_cpu_size=2,  # small cpu buffer
            local_disk=temp_dir,
            max_local_disk_size=20,
            extra_config={"force_store_wait": True},
        )

        engine = autorelease_v1(
            LMCacheEngineBuilder.get_or_create(
                "test",
                cfg,
                dumb_metadata(kv_shape),
                connector,
                mock_up_broadcast_fn,
                mock_up_broadcast_object_fn,
            )
        )

        # Store kv cache into slow devices
        for t, s in zip(list_tokens, list_slot_mappings, strict=False):
            engine.store(t, kvcaches=kv_cache, slot_mapping=s)

        # Sleep 10 seconds for the last request
        time.sleep(20)

        # No KV cache should be skipped
        # With default save_unfull_chunk=False, we expect chunk-aligned count
        chunk_size = 256
        for t in list_tokens:
            expected_count = (len(t) // chunk_size) * chunk_size
            assert engine.lookup(t) == expected_count


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_builder_destroy(autorelease_v1):
    """Test the destroy method of LMCacheEngineBuilder"""
    instance_id = "test_destroy"
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256)
    connector = create_gpu_connector(1024, 32)

    # Verify instance doesn't exist initially
    should_be_none = LMCacheEngineBuilder.get(instance_id)
    assert should_be_none is None

    # Create an engine instance
    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        cfg,
        dumb_metadata(),
        connector,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )

    # Verify instance exists
    retrieved_engine = LMCacheEngineBuilder.get(instance_id)
    assert retrieved_engine is not None
    assert retrieved_engine is engine

    # Verify internal state is populated
    assert instance_id in LMCacheEngineBuilder._instances
    assert instance_id in LMCacheEngineBuilder._cfgs
    assert instance_id in LMCacheEngineBuilder._metadatas
    assert instance_id in LMCacheEngineBuilder._stat_loggers

    # Destroy the instance
    LMCacheEngineBuilder.destroy(instance_id)

    # Verify instance is completely removed
    should_be_none_after_destroy = LMCacheEngineBuilder.get(instance_id)
    assert should_be_none_after_destroy is None

    # Verify all internal state is cleaned up
    assert instance_id not in LMCacheEngineBuilder._instances
    assert instance_id not in LMCacheEngineBuilder._cfgs
    assert instance_id not in LMCacheEngineBuilder._metadatas
    assert instance_id not in LMCacheEngineBuilder._stat_loggers

    # Verify destroying non-existent instance doesn't raise error
    LMCacheEngineBuilder.destroy("non_existent_id")  # Should not raise


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_builder_destroy_multiple_instances(autorelease_v1):
    """Test destroying one instance doesn't affect others"""
    instance_id1 = "test_destroy_1"
    instance_id2 = "test_destroy_2"
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256)
    connector = create_gpu_connector(1024, 32)

    # Create two engine instances
    engine1 = LMCacheEngineBuilder.get_or_create(
        instance_id1,
        cfg,
        dumb_metadata(),
        connector,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )

    engine2 = LMCacheEngineBuilder.get_or_create(
        instance_id2,
        cfg,
        dumb_metadata(),
        connector,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )

    # Verify both instances exist
    assert LMCacheEngineBuilder.get(instance_id1) is engine1
    assert LMCacheEngineBuilder.get(instance_id2) is engine2

    # Destroy only the first instance
    LMCacheEngineBuilder.destroy(instance_id1)

    # Verify first instance is destroyed but second remains
    assert LMCacheEngineBuilder.get(instance_id1) is None
    assert LMCacheEngineBuilder.get(instance_id2) is engine2

    # Verify internal state for first instance is cleaned up
    assert instance_id1 not in LMCacheEngineBuilder._instances
    assert instance_id1 not in LMCacheEngineBuilder._cfgs
    assert instance_id1 not in LMCacheEngineBuilder._metadatas
    assert instance_id1 not in LMCacheEngineBuilder._stat_loggers

    # Verify internal state for second instance remains
    assert instance_id2 in LMCacheEngineBuilder._instances
    assert instance_id2 in LMCacheEngineBuilder._cfgs
    assert instance_id2 in LMCacheEngineBuilder._metadatas
    assert instance_id2 in LMCacheEngineBuilder._stat_loggers

    # Clean up second instance
    LMCacheEngineBuilder.destroy(instance_id2)


@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA for test_multi_device_backends",
)
@pytest.mark.skipif(
    not has_cufile(),
    reason="Requires NVIDIA cuFile (libcufile.so). "
    "Skipping on systems without GDS/cuFile (e.g., AMD ROCm).",
)
def test_multi_device_backends(save_unfull_chunk, autorelease_v1):
    """Test running GPU-related backend with local CPU backends
    together
    """
    device = "cuda"
    num_tokens = 2000
    chunk_size = 256  # Default chunk size for this test
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    connector = create_gpu_connector(1024, 32)
    metadata = dumb_metadata()
    metadata.model_name = "test-model"  # NOTE: Gds does not accept name with '_'

    tokens = generate_tokens(num_tokens, device)

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    original_retrieved_cache = deepcopy(retrieved_cache)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    # Check the kv cache and the retrieval buffer are not the same
    check_paged_kv_cache_equal(retrieved_cache, original_retrieved_cache, slot_mapping)
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(retrieved_cache, kv_cache, slot_mapping)

    homedir = os.environ.get("HOME", "/tmp")
    with tempfile.TemporaryDirectory(
        dir=homedir, ignore_cleanup_errors=True
    ) as temp_dir:
        cfg = LMCacheEngineConfig.from_dict(
            {
                "local_cpu": True,
                "max_local_cpu_size": 5,
                "gds_path": temp_dir,
                "cufile_buffer_size": 1024,
                "save_unfull_chunk": save_unfull_chunk,
                "extra_config": {
                    "use_direct_io": True,
                },
            }
        )

        connector = create_gpu_connector(1024, 32)

        engine = autorelease_v1(
            LMCacheEngineBuilder.get_or_create(
                "engine",
                cfg,
                metadata,
                connector,
                mock_up_broadcast_fn,
                mock_up_broadcast_object_fn,
            )
        )

        """ test store """
        engine.store(tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
        recover_engine_states(engine)
        time.sleep(3)  # wait a bit to finish the store

        """ Test lookup """
        expected_count = get_expected_count(len(tokens), save_unfull_chunk, chunk_size)
        ret = engine.lookup(tokens)
        assert ret == expected_count

        ret_cpu = engine.lookup(tokens, search_range=["LocalCPUBackend"])
        assert ret_cpu == expected_count

        ret_gds = engine.lookup(tokens, search_range=["GdsBackend"])
        assert ret_gds == expected_count

        """ Test retrieve """
        ret_mask = engine.retrieve(
            tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
        )
        recover_engine_states(engine)
        length = torch.sum(ret_mask)
        assert length == expected_count
        # Only check chunk-aligned tokens when save_unfull_chunk=False
        check_paged_kv_cache_equal(
            retrieved_cache, kv_cache, slot_mapping[:expected_count]
        )

        LMCacheEngineBuilder.destroy("engine")
