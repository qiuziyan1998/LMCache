# SPDX-License-Identifier: Apache-2.0
"""
Test cases for StorageManager.

This module tests the critical logic in prefetch_all_done_callback that handles:
1. Calculating the actual number of retrieved chunks based on batched_get_non_blocking
   results (not batched_async_contains results)
2. Handling chunk eviction between contains check and actual retrieval
3. Ensuring prefix-based continuity: if a tier retrieves fewer chunks than expected,
   all subsequent tiers are ignored
4. Properly cleaning up (ref_count_down) memory objects that won't be used due to
   discontinuity

Key scenarios tested:
- All chunks retrieved successfully from all tiers
- Middle tier partial retrieval (subsequent tiers ignored)
- First tier partial retrieval (all subsequent tiers ignored)
- Last chunk not being full size
- Single tier partial retrieval
"""

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from types import SimpleNamespace
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager


class MockMemoryObj:
    """Mock MemoryObj for testing."""

    def __init__(self, obj_id: int):
        self.obj_id = obj_id
        self.ref_count = 1
        self.ref_count_down_called = False

    def ref_count_down(self):
        self.ref_count -= 1
        self.ref_count_down_called = True

    def __repr__(self):
        return f"MockMemoryObj(id={self.obj_id}, ref_count={self.ref_count})"


class MockAsyncLookupServer:
    """Mock async lookup server for testing."""

    def __init__(self):
        self.responses = []

    def send_response_to_scheduler(self, lookup_id: str, retrieved_length: int):
        self.responses.append((lookup_id, retrieved_length))


def _keyed_results(
    *tiers: list[MockMemoryObj],
) -> list[list[tuple[str, MockMemoryObj]]]:
    return [
        [
            (f"tier-{tier_idx}-chunk-{chunk_idx}", obj)
            for chunk_idx, obj in enumerate(tier)
        ]
        for tier_idx, tier in enumerate(tiers)
    ]


@pytest.mark.parametrize(
    "fmt,shape,kv_group",
    (
        (MemoryFormat.KV_MLA_LATENT_FMT, torch.Size([16]), 0),
        (MemoryFormat.KV_DSA_INDEX_FMT, torch.Size([8]), 1),
    ),
)
def test_batched_put_layer_pages_uses_one_local_and_remote_page(
    monkeypatch, fmt, shape, kv_group
):
    allocator = TensorMemoryAllocator(torch.zeros(16384, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        shape,
        torch.float16,
        batch_size=2,
        num_layers=3,
        fmt=fmt,
        valid_tokens=[8, 3],
        full_tokens=8,
    )
    assert pages is not None
    assert pages[0].layer_tensor(0).numel() == shape.numel()
    assert pages[1].layer_tensor(0).numel() == shape.numel() * 3 // 8
    keys = [
        CacheEngineKey("model", 1, 0, 0, torch.float16, kv_group=kv_group),
        CacheEngineKey(
            "model",
            1,
            0,
            1,
            torch.float16,
            {"lmcache.tag.internal.valid_tokens": 3},
            kv_group=kv_group,
        ),
    ]
    local = object.__new__(LocalCPUBackend)
    remote = object.__new__(RemoteBackend)
    local.use_hot = True
    remote.connection = SimpleNamespace(batched_put_external_pages=lambda: None)
    local_calls = []
    remote_calls = []
    remote_futures = []

    def submit_local(page_keys, submitted_pages):
        local_calls.append((list(page_keys), list(submitted_pages)))
        for page in submitted_pages:
            page.ref_count_up()

    def submit_remote(page_keys, ptrs, sizes, owners, ready_event, req_id):
        remote_calls.append(
            (list(page_keys), ptrs, sizes, owners, ready_event, req_id)
        )
        future = Future()
        remote_futures.append(future)
        return future

    monkeypatch.setattr(local, "batched_submit_layer_pages", submit_local)
    monkeypatch.setattr(remote, "batched_submit_external_pages", submit_remote)
    manager = object.__new__(StorageManager)
    manager.storage_backends = OrderedDict(
        (("LocalCPUBackend", local), ("RemoteBackend", remote))
    )
    manager.config = SimpleNamespace(chunk_size=8)
    manager._bypassed_backends = set()
    manager._bypass_lock = threading.Lock()
    manager._freeze = False
    manager._freeze_lock = threading.Lock()

    futures = manager.batched_put_layer_pages(keys, pages, req_id="request")

    assert len(futures) == 1
    remote_keys, ptrs, sizes, owners, ready_event, req_id = remote_calls[0]
    assert remote_keys == keys
    assert ptrs == [
        [page.layer_data_ptr(layer) for layer in range(3)] for page in pages
    ]
    assert sizes == [[page.layer_size] * 3 for page in pages]
    assert len(owners) == 1
    assert ready_event is None
    assert req_id == "request"
    assert all(page.get_ref_count() == 1 for page in pages)
    assert local_calls == []
    remote_futures[0].set_result(None)
    assert local_calls == [(keys, pages)]
    assert futures[0].result() is None
    assert all(page.get_ref_count() == 1 for page in pages)


def test_batched_put_layer_pages_preserves_caller_refs_on_remote_error(monkeypatch):
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8]),
        torch.float16,
        batch_size=1,
        num_layers=2,
        fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=2,
        full_tokens=2,
    )
    assert pages is not None
    key = CacheEngineKey("model", 1, 0, 0, torch.float16, kv_group=1)
    local = object.__new__(LocalCPUBackend)
    remote = object.__new__(RemoteBackend)
    local.use_hot = True
    remote.connection = SimpleNamespace(batched_put_external_pages=lambda: None)

    def submit_local(submitted_keys, held):
        for page in held:
            page.ref_count_up()

    def fail_remote(*_args):
        raise RuntimeError("remote failed")

    monkeypatch.setattr(
        local,
        "batched_submit_layer_pages",
        submit_local,
    )
    monkeypatch.setattr(
        remote,
        "batched_submit_external_pages",
        fail_remote,
    )
    manager = object.__new__(StorageManager)
    manager.storage_backends = OrderedDict(
        (("LocalCPUBackend", local), ("RemoteBackend", remote))
    )
    manager.config = SimpleNamespace(chunk_size=2)
    manager._bypassed_backends = set()
    manager._bypass_lock = threading.Lock()
    manager._freeze = False
    manager._freeze_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="remote failed"):
        manager.batched_put_layer_pages([key], pages)

    # The failed durable write never becomes visible in LocalCPU.
    assert pages[0].get_ref_count() == 1
    pages[0].ref_count_down()
    assert pages[0].get_ref_count() == 0


def test_batched_put_layer_pages_hides_async_remote_failure(monkeypatch):
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8]),
        torch.float16,
        batch_size=1,
        num_layers=2,
        fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=2,
        full_tokens=2,
    )
    assert pages is not None
    key = CacheEngineKey("model", 1, 0, 0, torch.float16, kv_group=1)
    local = object.__new__(LocalCPUBackend)
    remote = object.__new__(RemoteBackend)
    local.use_hot = True
    remote.connection = SimpleNamespace(batched_put_external_pages=lambda: None)
    local_calls = []

    def submit_local(submitted_keys, held):
        local_calls.append((list(submitted_keys), list(held)))
        held[0].ref_count_up()

    future = Future()
    monkeypatch.setattr(local, "batched_submit_layer_pages", submit_local)
    monkeypatch.setattr(
        remote, "batched_submit_external_pages", lambda *_args: future
    )
    manager = object.__new__(StorageManager)
    manager.storage_backends = OrderedDict(
        (("LocalCPUBackend", local), ("RemoteBackend", remote))
    )
    manager.config = SimpleNamespace(chunk_size=2)
    manager._bypassed_backends = set()
    manager._bypass_lock = threading.Lock()
    manager._freeze = False
    manager._freeze_lock = threading.Lock()

    completions = manager.batched_put_layer_pages([key], pages)
    assert len(completions) == 1 and not completions[0].done()
    assert local_calls == []
    future.set_exception(RuntimeError("remote failed"))

    with pytest.raises(RuntimeError, match="remote failed"):
        completions[0].result()
    assert local_calls == []
    assert pages[0].get_ref_count() == 0


@pytest.mark.parametrize("local_keys", ({"a", "b"}, {"a"}))
def test_layerwise_remote_hint_prefers_only_complete_local_hit(
    monkeypatch, local_keys
):
    calls = []

    async def done():
        return []

    def get(name):
        calls.append(name)
        return done()

    local = SimpleNamespace(
        contains_all_exact=lambda keys: all(key in local_keys for key in keys),
        batched_get_non_blocking=lambda *_args: get("local"),
    )
    remote = SimpleNamespace(
        batched_get_non_blocking=lambda *_args: get("remote")
    )
    manager = object.__new__(StorageManager)
    manager.storage_backends = {
        "LocalCPUBackend": local,
        "RemoteBackend": remote,
    }
    manager.loop = object()
    manager._layerwise_get_prefers_blocking = lambda *_args: False

    def submit(coro, _loop):
        coro.close()
        return Future()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    next(
        manager.layerwise_batched_get(
            [["a", "b"]], location="RemoteBackend"
        )
    )

    assert calls == ["local" if len(local_keys) == 2 else "remote"]


@pytest.fixture
def event_manager():
    """Create an EventManager for testing."""
    return EventManager()


@pytest.fixture
def storage_manager_config():
    """Create a test configuration for StorageManager."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=False,
        lmcache_instance_id="test_instance",
    )
    return config


@pytest.fixture
def storage_manager_metadata():
    """Create test metadata for StorageManager."""
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
        role="scheduler",
    )
    return metadata


@pytest.fixture
def storage_manager(storage_manager_config, storage_manager_metadata, event_manager):
    """Create a StorageManager for testing."""
    manager = StorageManager(
        config=storage_manager_config,
        metadata=storage_manager_metadata,
        event_manager=event_manager,
    )
    # Mock the async lookup server
    manager.async_lookup_server = MockAsyncLookupServer()
    yield manager
    manager.close()


def _close_only_storage_manager(backends) -> StorageManager:
    manager = object.__new__(StorageManager)
    manager.storage_backends = OrderedDict(backends)
    manager.loop = SimpleNamespace(is_running=lambda: False)
    manager.thread = SimpleNamespace(is_alive=lambda: False)
    return manager


def test_close_orders_remote_dependency_before_local_cpu() -> None:
    calls = []

    class Backend:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    manager = _close_only_storage_manager(
        [
            ("LocalCPUBackend", Backend("local")),
            ("RemoteBackend", Backend("remote")),
        ]
    )

    manager.close()

    assert calls == ["remote", "local"]


def test_strict_remote_close_failure_retains_local_cpu_and_loop() -> None:
    calls = []

    class LocalBackend:
        def close(self) -> None:
            calls.append("local")

    class StrictRemoteBackend:
        @staticmethod
        def requires_strict_external_close() -> bool:
            return True

        @staticmethod
        def close() -> None:
            calls.append("remote")
            raise RuntimeError("external HBM ownership unknown")

    manager = _close_only_storage_manager(
        [
            ("LocalCPUBackend", LocalBackend()),
            ("RemoteBackend", StrictRemoteBackend()),
        ]
    )

    with pytest.raises(RuntimeError, match="ownership unknown"):
        manager.close()

    assert calls == ["remote"]
    assert list(manager.storage_backends) == [
        "LocalCPUBackend",
        "RemoteBackend",
    ]


class TestStorageManagerPrefetchCallback:
    """Test cases for StorageManager prefetch_all_done_callback."""

    def test_all_chunks_retrieved_successfully(self, storage_manager):
        """Test Case 1: All chunks retrieved successfully from all tiers."""
        # Setup: 5 chunks total (1280 tokens), distributed across 2 tiers
        # Tier 0: 3 chunks, Tier 1: 2 chunks
        cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280]
        tier_expected_chunks = [3, 2]

        # Create mock memory objects for all chunks
        tier0_objs = [MockMemoryObj(i) for i in range(3)]
        tier1_objs = [MockMemoryObj(i + 3) for i in range(2)]
        res = _keyed_results(tier0_objs, tier1_objs)

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_1", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_1", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: All 5 chunks should be counted, total 1280 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_1"
        assert retrieved_length == 1280

        # Verify: No memory objects should have ref_count_down called
        for obj in tier0_objs + tier1_objs:
            assert not obj.ref_count_down_called

    def test_middle_tier_partial_retrieval(self, storage_manager):
        """Test Case 2: Middle tier only got partial chunks, subsequent tier ignored."""
        # Setup: 7 chunks total (1792 tokens), distributed across 3 tiers
        # Tier 0: 3 chunks, Tier 1: 2 chunks, Tier 2: 2 chunks
        cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        tier_expected_chunks = [3, 2, 2]

        # Tier 0 got all 3, Tier 1 only got 1 (eviction), Tier 2 got all 2
        tier0_objs = [MockMemoryObj(i) for i in range(3)]
        tier1_objs = [MockMemoryObj(i + 3) for i in range(1)]  # Only 1 instead of 2
        tier2_objs = [MockMemoryObj(i + 5) for i in range(2)]  # Got all 2
        res = _keyed_results(tier0_objs, tier1_objs, tier2_objs)

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_2", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_2", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: Only 4 chunks counted (3 from tier0 + 1 from tier1)
        # Total: 1024 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_2"
        assert retrieved_length == 1024

        # Verify: Tier 0 and Tier 1 objects should NOT have ref_count_down called
        for obj in tier0_objs + tier1_objs:
            assert not obj.ref_count_down_called

        # Verify: All Tier 2 objects should have ref_count_down called
        for obj in tier2_objs:
            assert obj.ref_count_down_called

    def test_first_tier_partial_retrieval(self, storage_manager):
        """
        Test Case 3: First tier only got partial chunks,
        all subsequent tiers ignored.
        """
        # Setup: 7 chunks total (1792 tokens), distributed across 3 tiers
        # Tier 0: 3 chunks, Tier 1: 2 chunks, Tier 2: 2 chunks
        cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        tier_expected_chunks = [3, 2, 2]

        # Tier 0 only got 2 (eviction), Tier 1 got all 2, Tier 2 got all 2
        tier0_objs = [MockMemoryObj(i) for i in range(2)]  # Only 2 instead of 3
        tier1_objs = [MockMemoryObj(i + 3) for i in range(2)]  # Got all 2
        tier2_objs = [MockMemoryObj(i + 5) for i in range(2)]  # Got all 2
        res = _keyed_results(tier0_objs, tier1_objs, tier2_objs)

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_3", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_3", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: Only 2 chunks counted (2 from tier0)
        # Total: 512 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_3"
        assert retrieved_length == 512

        # Verify: Tier 0 objects should NOT have ref_count_down called
        for obj in tier0_objs:
            assert not obj.ref_count_down_called

        # Verify: All Tier 1 and Tier 2 objects should have ref_count_down called
        for obj in tier1_objs + tier2_objs:
            assert obj.ref_count_down_called

    def test_last_chunk_not_full(self, storage_manager):
        """Test with last chunk not being full size."""
        # Setup: 3 chunks with last chunk only 128 tokens (640 tokens total)
        # Tier 0: 2 chunks, Tier 1: 1 chunk
        cum_chunk_lengths_total = [0, 256, 512, 640]  # Last chunk is 128 tokens
        tier_expected_chunks = [2, 1]

        # All chunks retrieved successfully
        tier0_objs = [MockMemoryObj(i) for i in range(2)]
        tier1_objs = [MockMemoryObj(i + 2) for i in range(1)]
        res = _keyed_results(tier0_objs, tier1_objs)

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_4", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_4", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: All 3 chunks counted, total 640 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_4"
        assert retrieved_length == 640

        # Verify: No memory objects should have ref_count_down called
        for obj in tier0_objs + tier1_objs:
            assert not obj.ref_count_down_called

    def test_single_tier_partial_retrieval(self, storage_manager):
        """Test with single tier that only got partial chunks."""
        # Setup: 5 chunks total (1280 tokens), single tier
        cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280]
        tier_expected_chunks = [5]

        # Only got 3 chunks instead of 5
        tier0_objs = [MockMemoryObj(i) for i in range(3)]
        res = _keyed_results(tier0_objs)

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_5", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_5", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: Only 3 chunks counted, total 768 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_5"
        assert retrieved_length == 768

        # Verify: No memory objects should have ref_count_down called
        # (no remaining chunks in current tier, no subsequent tiers)
        for obj in tier0_objs:
            assert not obj.ref_count_down_called


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_async_lookup_setup_failure_sends_one_terminal_miss(
    storage_manager, monkeypatch, failure_type
):
    class FailingContainsBackend:
        async def batched_async_contains(self, *_args, **_kwargs):
            raise failure_type("contains failed")

    monkeypatch.setattr(
        storage_manager,
        "get_active_storage_backends",
        lambda search_range=None: [("failing", FailingContainsBackend())],
    )
    key = CacheEngineKey("model", 1, 0, 0, torch.float16)

    await storage_manager.async_lookup_and_prefetch(
        "setup-failed", [key], [0, 256], pin=True
    )
    await asyncio.sleep(0)

    assert storage_manager.async_lookup_server.responses == [("setup-failed", 0)]
    assert (
        storage_manager.event_manager.get_event_status(
            EventType.LOADING, "setup-failed"
        )
        == EventStatus.DONE
    )
    assert storage_manager.event_manager.get_event_future(
        EventType.LOADING, "setup-failed"
    ).result() == []


@pytest.mark.asyncio
async def test_async_lookup_task_failure_preserves_successful_cleanup_results(
    storage_manager, monkeypatch
):
    retained = MockMemoryObj(0)

    class Backend:
        def __init__(self, result=None, error=None):
            self.result = result
            self.error = error

        async def batched_async_contains(self, *_args, **_kwargs):
            return 1

        async def batched_get_non_blocking(self, *_args, **_kwargs):
            if self.error is not None:
                raise self.error
            return self.result

    backends = [
        ("successful", Backend(result=[retained])),
        ("failing", Backend(error=RuntimeError("get failed"))),
    ]
    monkeypatch.setattr(
        storage_manager,
        "get_active_storage_backends",
        lambda search_range=None: backends,
    )
    storage_manager.async_serializer = SimpleNamespace(
        run=lambda coro, _num_chunks: coro
    )
    keys = [
        CacheEngineKey("model", 1, 0, 0, torch.float16),
        CacheEngineKey("model", 1, 0, 1, torch.float16),
    ]

    await storage_manager.async_lookup_and_prefetch(
        "task-failed", keys, [0, 256, 512], pin=True
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert storage_manager.async_lookup_server.responses == [("task-failed", 0)]
    cleanup_result = storage_manager.event_manager.get_event_future(
        EventType.LOADING, "task-failed"
    ).result()
    assert cleanup_result == [[(keys[0], retained)], []]
