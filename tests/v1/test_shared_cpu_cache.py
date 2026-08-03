# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace
import asyncio
import sys

import pytest
import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    PagedTensorMemoryAllocator,
    TensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.shared_cpu_cache import (
    PassiveSharedViewAllocator,
    SharedChunkHandle,
    SharedCPUCacheError,
    SharedCPUCacheValidationError,
    SharedHandleBatch,
    SharedHandleEnvelope,
    SharedCPURequestLease,
    SharedSlabMapping,
    validate_shared_handle_batch,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUPrefixGetResult


class _LeaseMemoryObj:
    def __init__(self, *, pinned: bool = False) -> None:
        self.ref_count = 1
        self.pin_count = int(pinned)
        self.valid = True

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0

    def is_valid(self) -> bool:
        return self.valid

    def ref_count_up(self) -> None:
        self.ref_count += 1

    def ref_count_down(self) -> None:
        self.ref_count -= 1
        if self.ref_count == 0 and self.pin_count == 0:
            self.valid = False

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.pin_count -= 1
        return True


def test_shared_cpu_request_lease_retains_store_seed_once() -> None:
    memory_obj = _LeaseMemoryObj()
    lease = SharedCPURequestLease(
        request_id="req",
        generation=1,
        is_rank0=True,
    )

    lease.replace_groups({0: [[memory_obj]]}, retain=True)
    lease.replace_groups({0: [[memory_obj]]}, retain=True)

    assert memory_obj.ref_count == 2
    assert memory_obj.pin_count == 1
    lease.close()
    assert memory_obj.ref_count == 1
    assert memory_obj.pin_count == 0


def test_shared_cpu_request_lease_retain_failure_is_transactional() -> None:
    class FailingPinMemoryObj(_LeaseMemoryObj):
        def pin(self) -> bool:
            return False

    old_obj = _LeaseMemoryObj()
    new_obj = _LeaseMemoryObj()
    failing_obj = FailingPinMemoryObj()
    lease = SharedCPURequestLease(
        request_id="req",
        generation=1,
        is_rank0=True,
    )
    lease.replace_groups({0: [[old_obj]]}, retain=True)

    with pytest.raises(RuntimeError, match="pin"):
        lease.replace_groups(
            {0: [[old_obj, new_obj, failing_obj]]},
            retain=True,
        )

    assert lease.object_ids(0) == {id(old_obj)}
    assert (old_obj.ref_count, old_obj.pin_count) == (2, 1)
    assert (new_obj.ref_count, new_obj.pin_count) == (1, 0)
    assert (failing_obj.ref_count, failing_obj.pin_count) == (1, 0)

    lease.close()
    assert (old_obj.ref_count, old_obj.pin_count) == (1, 0)


def test_shared_cpu_request_lease_replaces_adopted_views() -> None:
    old_view = _LeaseMemoryObj()
    new_view = _LeaseMemoryObj()
    layers = [[old_view]]
    lease = SharedCPURequestLease(
        request_id="req",
        generation=1,
        is_rank0=False,
    )

    lease.replace_groups({0: layers}, retain=False)
    layers[0].append(new_view)
    assert lease.object_ids(0) == {id(old_view)}

    lease.replace_groups({0: [[new_view]]}, retain=False)
    assert old_view.ref_count == 0
    assert not old_view.valid
    assert new_view.ref_count == 1

    lease.close()
    lease.close()
    assert new_view.ref_count == 0
    assert not new_view.valid


def test_shared_cpu_request_lease_appends_adopted_suffix() -> None:
    old_views = [_LeaseMemoryObj(), _LeaseMemoryObj()]
    new_views = [_LeaseMemoryObj(), _LeaseMemoryObj()]
    lease = SharedCPURequestLease(
        request_id="req",
        generation=1,
        is_rank0=False,
    )
    lease.replace_groups({0: [[old_views[0]], [old_views[1]]]}, retain=False)

    lease.append_groups(
        {
            0: [
                [old_views[0], new_views[0]],
                [old_views[1], new_views[1]],
            ]
        },
        {0: 1},
    )

    assert lease.object_ids(0) == {id(obj) for obj in old_views + new_views}
    lease.close()
    assert all(not obj.valid for obj in old_views + new_views)


def test_shared_cpu_request_lease_append_alignment_failure_is_atomic() -> None:
    old_views = [_LeaseMemoryObj(), _LeaseMemoryObj()]
    new_views = [_LeaseMemoryObj(), _LeaseMemoryObj()]
    lease = SharedCPURequestLease(
        request_id="req",
        generation=1,
        is_rank0=False,
    )
    lease.replace_groups(
        {0: [[old_views[0]], [old_views[1], new_views[1]]]},
        retain=False,
    )

    with pytest.raises(ValueError, match="append-aligned"):
        lease.append_groups(
            {
                0: [
                    [old_views[0], new_views[0]],
                    [old_views[1], new_views[1]],
                ]
            },
            {0: 1},
        )

    assert lease.groups[0] == [
        [old_views[0]],
        [old_views[1], new_views[1]],
    ]


def test_shared_cpu_engine_registers_request_suffix_without_replacement() -> None:
    old_view = _LeaseMemoryObj()
    new_view = _LeaseMemoryObj()
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_generation = 1
    engine.metadata = SimpleNamespace(is_first_rank=lambda: False)
    engine._shared_cpu_request_leases = {}
    engine.register_shared_cpu_sparse_request(
        "req",
        owned_groups={0: [[old_view]]},
    )

    engine.register_shared_cpu_sparse_request(
        "req",
        owned_groups={0: [[old_view, new_view]]},
        append_from={0: 1},
    )

    assert engine._shared_cpu_request_leases["req"].groups == {
        0: [[old_view, new_view]]
    }
    assert old_view.ref_count == 1
    engine.release_shared_cpu_sparse_request("req")
    assert not old_view.valid and not new_view.valid


def test_stale_request_lease_does_not_claim_new_generation_objects() -> None:
    memory_obj = _LeaseMemoryObj()
    stale_lease = SharedCPURequestLease(
        request_id="req",
        generation=1,
        is_rank0=True,
    )
    stale_lease.replace_groups({0: [[memory_obj]]}, retain=True)

    # Simulate the independent pin/ref acquired for a generation-2 publication.
    memory_obj.ref_count_up()
    memory_obj.pin()
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_generation = 2
    engine.metadata = SimpleNamespace(is_first_rank=lambda: True)
    engine._shared_cpu_request_leases = {"req": stale_lease}

    assert engine.shared_cpu_rank0_request_object_ids("req", 0) == set()
    engine.release_shared_cpu_unowned_objects(
        "req",
        {0: [[memory_obj]]},
    )
    assert memory_obj.ref_count == 2
    assert memory_obj.pin_count == 1

    engine.release_shared_cpu_sparse_request("req")
    assert memory_obj.ref_count == 1
    assert memory_obj.pin_count == 0


class _FakeRemoteConnector(RemoteConnector):
    async def exists(self, key):  # pragma: no cover - not used by these tests
        return False

    def exists_sync(self, key):  # pragma: no cover - not used by these tests
        return False

    async def get(self, key):  # pragma: no cover - not used by these tests
        return None

    async def put(self, key, memory_obj):  # pragma: no cover - not used by these tests
        return None

    async def list(self):  # pragma: no cover - not used by these tests
        return []

    async def close(self):  # pragma: no cover - not used by these tests
        return None


def _make_key(kv_group: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=1234,
        dtype=torch.float16,
        kv_group=kv_group,
    )


def test_dense_retrieve_reuses_group0_chunk_plan_for_group1():
    class TokenDatabase:
        def __init__(self):
            self.calls = []

        def process_tokens(
            self,
            *,
            tokens=None,
            hashes=None,
            offsets=None,
            mask=None,
            request_configs=None,
            kv_group=0,
        ):
            self.calls.append((tokens, hashes, offsets, kv_group))
            if hashes is None:
                hashes = [101, 202]
                offsets = [2, 2]
            start = 0
            for chunk_hash, size in zip(hashes, offsets, strict=True):
                end = start + size
                yield start, end, replace(
                    _make_key(kv_group),
                    chunk_hash=chunk_hash,
                )
                start = end

    engine = object.__new__(LMCacheEngine)
    engine.token_database = TokenDatabase()
    state = {}
    kwargs = {"shared_cpu_request_preflight_state": state}

    group0 = list(
        engine._dense_retrieve_token_results([1, 2, 3, 4], None, None, 0, kwargs)
    )
    group1 = list(
        engine._dense_retrieve_token_results([9, 9, 9, 9], None, None, 1, kwargs)
    )

    assert [(start, end) for start, end, _ in group1] == [
        (start, end) for start, end, _ in group0
    ]
    assert [key.chunk_hash for _, _, key in group1] == [101, 202]
    assert engine.token_database.calls[1] == (
        None,
        [101, 202],
        [2, 2],
        1,
    )


def test_shared_page_first_uniform_location_uses_local_then_one_remote_probe():
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(
        extra_config={"mooncake_page_first_multi_buffer": True}
    )
    engine.retrieve_locations = None
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        cpu_lock=nullcontext(),
        hot_cache={},
    )
    calls = []

    def batched_contains(keys, search_range):
        calls.append((list(keys), search_range))
        return len(keys), {"RemoteBackend": list(keys)}

    remote = SimpleNamespace(
        connection=SimpleNamespace(support_batched_contains=lambda: True)
    )
    engine.storage_manager = SimpleNamespace(
        get_active_storage_backends=lambda search_range=None: iter(
            [("RemoteBackend", remote)]
        ),
        batched_contains=batched_contains,
    )
    keys_by_chunk = [
        replace(_make_key(), chunk_hash=chunk).split_layers(2)
        for chunk in (1, 2)
    ]

    assert (
        engine._shared_page_first_uniform_location(keys_by_chunk)
        == "RemoteBackend"
    )
    assert calls == [
        ([key for chunk in keys_by_chunk for key in chunk], ["RemoteBackend"])
    ]
    flat_keys = calls[0][0]
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        cpu_lock=nullcontext(),
        hot_cache=dict.fromkeys(flat_keys),
    )
    assert (
        engine._shared_page_first_uniform_location(keys_by_chunk)
        == "LocalCPUBackend"
    )
    assert len(calls) == 1
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        cpu_lock=nullcontext(),
        hot_cache={flat_keys[0]: object()},
    )
    assert (
        engine._shared_page_first_uniform_location(keys_by_chunk)
        == "RemoteBackend"
    )
    # A page-first probe must include every layer from the shortest common
    # LocalCPU prefix, even when another layer already has that object locally.
    assert calls[-1] == (flat_keys, ["RemoteBackend"])
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        cpu_lock=nullcontext(),
        hot_cache={flat_keys[-1]: object()},
    )
    assert engine._shared_page_first_uniform_location(keys_by_chunk) is None
    assert len(calls) == 2


@pytest.mark.parametrize(
    "local_hit,remote_hits,retrieval_backends,supports_batch",
    [
        (True, 2, ["RemoteBackend"], True),
        (False, 3, ["RemoteBackend"], True),
        (False, 4, ["LocalCPUBackend"], True),
        (False, 4, ["RemoteBackend", "LocalDiskBackend"], True),
        (False, 4, ["RemoteBackend"], False),
    ],
)
def test_shared_page_first_uniform_location_falls_back(
    local_hit, remote_hits, retrieval_backends, supports_batch
):
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(
        extra_config={"mooncake_page_first_multi_buffer": True}
    )
    engine.retrieve_locations = retrieval_backends
    keys_by_chunk = [
        replace(_make_key(), chunk_hash=chunk).split_layers(2)
        for chunk in (1, 2)
    ]
    flat_keys = [key for chunk in keys_by_chunk for key in chunk]
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        cpu_lock=nullcontext(),
        hot_cache={flat_keys[-1]: object()} if local_hit else {},
    )
    calls = []

    def batched_contains(keys, search_range):
        calls.append((list(keys), search_range))
        return remote_hits, {"RemoteBackend": list(keys[:remote_hits])}

    remote = SimpleNamespace(
        connection=SimpleNamespace(
            support_batched_contains=lambda: supports_batch
        )
    )
    active = [("RemoteBackend", remote)]
    if "LocalDiskBackend" in retrieval_backends:
        active.append(("LocalDiskBackend", object()))
    engine.storage_manager = SimpleNamespace(
        get_active_storage_backends=lambda search_range=None: iter(active),
        batched_contains=batched_contains,
    )

    assert engine._shared_page_first_uniform_location(keys_by_chunk) is None
    assert len(calls) == (
        1
        if not local_hit
        and supports_batch
        and retrieval_backends == ["RemoteBackend"]
        else 0
    )


def test_page_first_plan_uses_shortest_local_layer_prefix():
    locations = [
        ["LocalCPUBackend", "LocalCPUBackend", "RemoteBackend"],
        ["LocalCPUBackend", "RemoteBackend", "RemoteBackend"],
    ]

    assert LMCacheEngine._shared_page_first_common_prefix_plan(locations) == [
        ["LocalCPUBackend", "RemoteBackend", "RemoteBackend"],
        ["LocalCPUBackend", "RemoteBackend", "RemoteBackend"],
    ]
    assert (
        LMCacheEngine._shared_page_first_common_prefix_plan(
            [["RemoteBackend", "LocalCPUBackend"]]
        )
        is None
    )


def test_dense_shared_cache_adoption_transfers_ownership():
    engine = object.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.shared_cpu_cache_generation = 9
    engine._shared_cpu_request_leases = {}
    engine.metadata = SimpleNamespace(is_first_rank=lambda: False)
    memory_objs = [[_LeaseMemoryObj()], [_LeaseMemoryObj()]]
    keys = [[_make_key()], [_make_key()]]
    handles = [[object()], [object()]]
    caches = {
        "cached_keys": [],
        "cached_starts": [],
        "cached_ends": [],
        "cached_memory_objs": [],
        "cached_shared_handles": [],
    }

    adopted = engine._adopt_dense_shared_retrieve_cache(
        req_id="req-dense",
        starts=[0],
        ends=[256],
        keys_layer_major=keys,
        memory_objs=memory_objs,
        handles=handles,
        kv_group=0,
        kwargs={"_retain_shared_dense_cache": True, **caches},
    )

    assert adopted
    assert caches["cached_memory_objs"] == memory_objs
    assert caches["cached_shared_handles"] == handles
    assert engine.shared_cpu_rank0_request_object_ids("req-dense", 0) == set()
    assert engine._shared_cpu_request_leases["req-dense"].object_ids(0) == {
        id(obj) for layer in memory_objs for obj in layer
    }
    index_objs = [[_LeaseMemoryObj()], [_LeaseMemoryObj()]]
    index_caches = {
        "cached_keys": [],
        "cached_starts": [],
        "cached_ends": [],
        "cached_memory_objs": [],
        "cached_shared_handles": [],
    }
    assert engine._adopt_dense_shared_retrieve_cache(
        req_id="req-dense",
        starts=[0],
        ends=[256],
        keys_layer_major=keys,
        memory_objs=index_objs,
        handles=handles,
        kv_group=1,
        kwargs={"_retain_shared_dense_cache": True, **index_caches},
    )
    assert engine._shared_cpu_request_leases["req-dense"].object_ids() == {
        id(obj)
        for layers in (memory_objs, index_objs)
        for layer in layers
        for obj in layer
    }
    engine.release_shared_cpu_sparse_request("req-dense")
    assert all(
        not obj.valid
        for layers in (memory_objs, index_objs)
        for layer in layers
        for obj in layer
    )


def test_dense_shared_cache_adoption_falls_back_when_incomplete():
    engine = object.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.shared_cpu_cache_generation = 9
    engine._shared_cpu_request_leases = {}
    engine.metadata = SimpleNamespace(is_first_rank=lambda: False)
    caches = {
        "cached_keys": [],
        "cached_starts": [],
        "cached_ends": [],
        "cached_memory_objs": [],
        "cached_shared_handles": [],
    }

    assert not engine._adopt_dense_shared_retrieve_cache(
        req_id="req-incomplete",
        starts=[0],
        ends=[256],
        keys_layer_major=[[_make_key()]],
        memory_objs=[[_LeaseMemoryObj()]],
        handles=[[object()]],
        kv_group=0,
        kwargs={"_retain_shared_dense_cache": True, **caches},
    )
    assert "req-incomplete" not in engine._shared_cpu_request_leases
    assert not any(caches.values())


def test_dense_shared_cache_adoption_rolls_back_before_ownership_transfer():
    engine = object.__new__(LMCacheEngine)
    engine.num_layers = 1
    def fail_registration(*_args, **_kwargs):
        raise RuntimeError("registration failed")

    engine.register_shared_cpu_sparse_request = fail_registration
    memory_obj = _LeaseMemoryObj()
    caches = {
        "cached_keys": [],
        "cached_starts": [],
        "cached_ends": [],
        "cached_memory_objs": [],
        "cached_shared_handles": [],
    }

    with pytest.raises(RuntimeError, match="registration failed"):
        engine._adopt_dense_shared_retrieve_cache(
            req_id="req-fail",
            starts=[0],
            ends=[256],
            keys_layer_major=[[_make_key()]],
            memory_objs=[[memory_obj]],
            handles=[[object()]],
            kv_group=0,
            kwargs={"_retain_shared_dense_cache": True, **caches},
        )

    assert not any(caches.values())
    assert memory_obj.valid


class _FakeLayerwiseStorageManager:
    def __init__(
        self,
        present,
        block_mapping=None,
        remove_count=None,
        expose_local_cpu_backend=False,
    ):
        self.present = set(present)
        self.block_mapping = block_mapping
        self.remove_count = remove_count
        self.removed = []
        self.pinned = []
        self.unpinned = []
        self.storage_backends = (
            {"LocalCPUBackend": self} if expose_local_cpu_backend else {}
        )

    def contains(self, key, pin=False):
        return key in self.present

    def batched_contains(self, keys, search_range=None, pin=False):
        if search_range and "LocalCPUBackend" not in search_range:
            return 0, {}
        if self.block_mapping is not None:
            if pin:
                self.pinned.extend(keys)
            return len(keys), self.block_mapping
        hit = 0
        for key in keys:
            if key not in self.present:
                break
            hit += 1
        if pin:
            self.pinned.extend(keys[:hit])
        return hit, {"LocalCPUBackend": keys[:hit]} if hit else {}

    def batched_remove(self, keys, locations=None):
        self.removed.append((list(keys), locations))
        removed = sum(1 for key in keys if key in self.present)
        for key in keys:
            self.present.discard(key)
        return removed if self.remove_count is None else self.remove_count

    def batched_unpin(self, keys, locations=None):
        self.unpinned.append((list(keys), locations))

    def touch_cache(self):
        return None

    def get_active_storage_backends(self, location=None, search_range=None):
        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            if search_range and backend_name not in search_range:
                continue
            yield backend_name, backend


def test_layerwise_chunk_fully_stored_repairs_partial_cache() -> None:
    engine = object.__new__(LMCacheEngine)
    engine.retrieve_locations = ["LocalCPUBackend"]
    keys = _make_key().split_layers(4)

    engine.storage_manager = _FakeLayerwiseStorageManager(keys)
    assert (
        engine._layerwise_chunk_location_if_fully_stored(
            keys, req_id="req", kv_group=0, start=0, end=256
        )
        == "LocalCPUBackend"
    )
    assert engine._layerwise_chunk_fully_stored(
        keys, req_id="req", kv_group=0, start=0, end=256
    )
    assert engine.storage_manager.removed == []

    engine.storage_manager = _FakeLayerwiseStorageManager([])
    assert (
        engine._layerwise_chunk_location_if_fully_stored(
            keys, req_id="req", kv_group=0, start=0, end=256
        )
        is None
    )
    assert not engine._layerwise_chunk_fully_stored(
        keys, req_id="req", kv_group=0, start=0, end=256
    )
    assert engine.storage_manager.removed == []

    engine.storage_manager = _FakeLayerwiseStorageManager(
        keys[2:],
        expose_local_cpu_backend=True,
    )
    assert (
        engine._layerwise_chunk_location_if_fully_stored(
            keys, req_id="req", kv_group=0, start=0, end=256
        )
        is None
    )
    assert engine.storage_manager.removed == []
    assert not engine._layerwise_chunk_fully_stored(
        keys, req_id="req", kv_group=0, start=0, end=256
    )
    assert engine.storage_manager.removed == [(keys, None)]
    assert engine.storage_manager.present == set()

    engine.storage_manager = _FakeLayerwiseStorageManager(keys[:1])
    assert (
        engine._layerwise_chunk_location_if_fully_stored(
            keys, req_id="req", kv_group=0, start=0, end=256
        )
        is None
    )
    assert engine.storage_manager.removed == []
    assert not engine._layerwise_chunk_fully_stored(
        keys, req_id="req", kv_group=0, start=0, end=256
    )
    assert engine.storage_manager.removed == [(keys, ["LocalCPUBackend"])]

    engine.storage_manager = _FakeLayerwiseStorageManager(
        keys,
        block_mapping={
            "LocalCPUBackend": keys[:2],
            "LocalDiskBackend": keys[2:],
        },
    )
    assert (
        engine._layerwise_chunk_location_if_fully_stored(
            keys, req_id="req", kv_group=0, start=0, end=256
        )
        is None
    )
    assert engine.storage_manager.removed == []
    assert not engine._layerwise_chunk_fully_stored(
        keys, req_id="req", kv_group=0, start=0, end=256
    )
    assert engine.storage_manager.removed == [(keys, ["LocalCPUBackend"])]

    engine.storage_manager = _FakeLayerwiseStorageManager(
        keys[:1],
        remove_count=0,
    )
    with pytest.raises(ValueError, match="could not remove all existing layers"):
        engine._layerwise_chunk_fully_stored(
            keys, req_id="req", kv_group=0, start=0, end=256
        )


class _FakeLookupTokenDatabase:
    def process_tokens(
        self,
        tokens=None,
        hashes=None,
        offsets=None,
        request_configs=None,
    ):
        if tokens is not None:
            end = len(tokens)
        else:
            end = offsets[0]
        yield (
            0,
            end,
            self._make_key_by_hash(
                0xABC,
                request_configs,
                kv_group=0,
            ),
        )

    def _make_key_by_hash(self, chunk_hash, request_configs=None, kv_group=0):
        return CacheEngineKey(
            model_name="model",
            world_size=1,
            worker_id=0,
            chunk_hash=chunk_hash,
            dtype=torch.float16,
            request_configs=request_configs,
            kv_group=kv_group,
        )


class _FakeMultiChunkLookupTokenDatabase(_FakeLookupTokenDatabase):
    chunk_ends = (4, 8, 12, 14)

    def process_tokens(
        self,
        tokens=None,
        hashes=None,
        offsets=None,
        request_configs=None,
    ):
        del tokens, hashes, offsets
        start = 0
        for chunk_index, end in enumerate(self.chunk_ends):
            yield (
                start,
                end,
                self._make_key_by_hash(
                    0x100 + chunk_index,
                    request_configs,
                    kv_group=0,
                ),
            )
            start = end


class _FakeLookupStatsMonitor:
    def on_lookup_request(self, _num_tokens):
        return object()

    def on_lookup_finished(self, _stats, _result):
        return None


def _make_dsa_lookup_engine(present):
    engine = object.__new__(LMCacheEngine)
    engine._init_failed = False
    engine._health_monitor = None
    engine.storage_manager = _FakeLayerwiseStorageManager(present)
    engine.token_database = _FakeLookupTokenDatabase()
    engine.stats_monitor = _FakeLookupStatsMonitor()
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.lookup_pins = defaultdict(lambda: defaultdict(list))
    engine.use_layerwise = True
    engine.num_layers = 2
    engine.config = SimpleNamespace(dsa_two_groups=True)
    return engine


class _RecordingRemoteSampleStorageManager:
    def __init__(self, present, local_present=(), pin_present=None):
        self.present = set(present)
        self.pin_present = set(present if pin_present is None else pin_present)
        self.local_present = set(local_present)
        self.calls = []
        self.local_calls = []
        self.unpinned = []
        self.storage_backends = {"RemoteBackend": self}

    def contains(self, key, search_range=None, pin=False):
        self.local_calls.append((key, search_range, pin))
        if key in self.local_present and search_range == ["LocalCPUBackend"]:
            return "LocalCPUBackend"
        return None

    def batched_contains(self, keys, search_range=None, pin=False):
        keys = list(keys)
        local = search_range == ["LocalCPUBackend"]
        (self.local_calls if local else self.calls).append(
            (keys, search_range, pin)
        )
        hit = 0
        present = self.local_present if local else (
            self.pin_present if pin else self.present
        )
        for key in keys:
            if key not in present:
                break
            hit += 1
        location = "LocalCPUBackend" if local else "RemoteBackend"
        mapping = {location: keys[:hit]} if hit else {}
        return hit, mapping

    def batched_unpin(self, keys, locations=None):
        self.unpinned.append((list(keys), locations))

    def touch_cache(self):
        return None

    def get_active_storage_backends(self, location=None, search_range=None):
        for backend in ("LocalCPUBackend", "RemoteBackend"):
            if location and location != backend:
                continue
            if search_range and backend not in search_range:
                continue
            yield backend, self


def _sampled_keys_for_chunk(token_db, chunk_index, num_layers=4):
    sampled = []
    for kv_group in (0, 1):
        group_key = token_db._make_key_by_hash(
            0x100 + chunk_index,
            kv_group=kv_group,
        )
        layer_keys = group_key.split_layers(num_layers)
        sampled.extend((layer_keys[0], layer_keys[-1]))
    return sampled


def _layer_keys_for_chunk(token_db, chunk_index, kv_group, num_layers=4):
    return token_db._make_key_by_hash(
        0x100 + chunk_index,
        kv_group=kv_group,
    ).split_layers(num_layers)


def _make_sampled_lookup_engine(present, local_present=(), pin_present=None):
    engine = _make_dsa_lookup_engine([])
    engine.token_database = _FakeMultiChunkLookupTokenDatabase()
    engine.storage_manager = _RecordingRemoteSampleStorageManager(
        present, local_present, pin_present
    )
    engine.retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]
    engine.num_layers = 4
    engine.config.experimental_sampled_layerwise_lookup = True
    return engine


def test_sampled_lookup_uses_local_first_and_reverse_tail_probes() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    first_keys = _sampled_keys_for_chunk(token_db, 0)
    winner_keys = _sampled_keys_for_chunk(token_db, 2)
    engine = _make_sampled_lookup_engine([*first_keys, *winner_keys])

    assert engine.lookup(list(range(14)), lookup_id="req", pin=True) == 12

    calls = engine.storage_manager.calls
    assert [call[0] for call in calls[:3]] == [
        first_keys,
        _sampled_keys_for_chunk(token_db, 3),
        _sampled_keys_for_chunk(token_db, 2),
    ]
    assert all(call[1] == ["RemoteBackend"] for call in calls)
    assert engine.lookup_pins["req"]["RemoteBackend"] == []


def test_sampled_lookup_combines_local_and_remote_keys() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    local_keys = [
        key
        for chunk in range(3)
        for key in _layer_keys_for_chunk(token_db, chunk, 0)
    ]
    remote_keys = [
        *_sampled_keys_for_chunk(token_db, 0)[2:],
        *_sampled_keys_for_chunk(token_db, 2)[2:],
    ]
    engine = _make_sampled_lookup_engine(remote_keys, local_keys)

    assert engine.lookup(list(range(14)), lookup_id="req", pin=True) == 12

    assert engine.lookup_pins["req"]["LocalCPUBackend"] == local_keys
    assert engine.lookup_pins["req"]["RemoteBackend"] == []


def test_sampled_lookup_avoids_remote_when_samples_are_local() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    local_keys = [
        key
        for chunk in range(4)
        for group in (0, 1)
        for key in _layer_keys_for_chunk(token_db, chunk, group)
    ]
    engine = _make_sampled_lookup_engine([], local_keys)

    assert engine.lookup(list(range(14)), lookup_id="req", pin=True) == 14

    assert engine.storage_manager.calls == []
    assert engine.lookup_pins["req"]["LocalCPUBackend"] == local_keys


def test_sampled_lookup_does_not_trust_partial_local_layers() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    sampled = _sampled_keys_for_chunk(token_db, 0)
    engine = _make_sampled_lookup_engine([], sampled)

    assert engine.lookup(list(range(14)), lookup_id="req", pin=True) == 0
    assert engine.lookup_pins["req"] == {}


def test_sampled_lookup_rejects_local_prefix_holes() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    first = [
        key
        for group in (0, 1)
        for key in _layer_keys_for_chunk(token_db, 0, group)
    ]
    tail = [
        key
        for group in (0, 1)
        for key in _layer_keys_for_chunk(token_db, 3, group)
    ]
    engine = _make_sampled_lookup_engine([], [*first, *tail])

    assert engine.lookup(list(range(14)), lookup_id="req", pin=True) == 0
    assert engine.lookup_pins["req"] == {}
    assert engine.storage_manager.unpinned == [
        (first, ["LocalCPUBackend"]),
    ]


def test_sampled_lookup_rolls_back_local_pins_on_remote_race() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    pinned_local_keys = _layer_keys_for_chunk(token_db, 0, 0)
    partial_local = _layer_keys_for_chunk(token_db, 1, 0)[:1]
    remote_keys = [
        *_sampled_keys_for_chunk(token_db, 0)[2:],
        *_sampled_keys_for_chunk(token_db, 2),
    ]
    partial_remote = _sampled_keys_for_chunk(token_db, 1)[:2]
    engine = _make_sampled_lookup_engine(
        remote_keys,
        [*pinned_local_keys, *partial_local],
        pin_present=partial_remote[:1],
    )

    assert engine.lookup(list(range(14)), lookup_id="req", pin=True) == 0
    assert engine.lookup_pins["req"] == {}
    assert engine.storage_manager.unpinned == [
        (partial_local, ["LocalCPUBackend"]),
        (pinned_local_keys, ["LocalCPUBackend"]),
    ]


def test_sampled_lookup_returns_zero_after_first_chunk_miss() -> None:
    engine = _make_sampled_lookup_engine([])

    assert engine.lookup(list(range(14)), lookup_id="req", pin=False) == 0

    assert len(engine.storage_manager.calls) == 1
    assert engine.storage_manager.calls[0][1] == ["RemoteBackend"]


def test_sampled_lookup_uses_active_backends_when_range_is_unset() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    first_keys = _sampled_keys_for_chunk(token_db, 0)
    tail_keys = _sampled_keys_for_chunk(token_db, 3)
    engine = _make_sampled_lookup_engine([*first_keys, *tail_keys])
    engine.retrieve_locations = None

    assert engine.lookup(list(range(14)), pin=False) == 14
    assert engine.storage_manager.calls == [
        (first_keys, ["RemoteBackend"], False),
        (tail_keys, ["RemoteBackend"], False),
    ]


def test_sampled_lookup_respects_explicit_local_search_range() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    engine = _make_sampled_lookup_engine(_sampled_keys_for_chunk(token_db, 0))

    assert (
        engine.lookup(
            list(range(14)),
            search_range=["LocalCPUBackend"],
            pin=False,
        )
        == 0
    )
    assert engine.storage_manager.calls == []


def test_sampled_lookup_can_select_partial_tail_chunk() -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    first_keys = _sampled_keys_for_chunk(token_db, 0)
    tail_keys = _sampled_keys_for_chunk(token_db, 3)
    engine = _make_sampled_lookup_engine([*first_keys, *tail_keys])

    assert engine.lookup(list(range(14)), lookup_id="req", pin=False) == 14
    assert [call[0] for call in engine.storage_manager.calls] == [
        first_keys,
        tail_keys,
    ]


def test_sampled_lookup_without_remote_falls_back_to_local_cpu() -> None:
    token_db = _FakeLookupTokenDatabase()
    latent_layers = token_db._make_key_by_hash(0xABC, kv_group=0).split_layers(2)
    index_layers = token_db._make_key_by_hash(0xABC, kv_group=1).split_layers(2)
    engine = _make_dsa_lookup_engine([*latent_layers, *index_layers])
    engine.storage_manager.storage_backends = {
        "LocalCPUBackend": engine.storage_manager
    }
    engine.config.experimental_sampled_layerwise_lookup = True

    assert engine.lookup([1, 2, 3], lookup_id="req", pin=True) == 3
    assert engine.lookup_pins["req"]["LocalCPUBackend"] == [
        *latent_layers,
        *index_layers,
    ]


def test_layerwise_lookup_requires_dsa_index_group_before_hit() -> None:
    token_db = _FakeLookupTokenDatabase()
    latent_layers = token_db._make_key_by_hash(0xABC, kv_group=0).split_layers(2)
    index_layers = token_db._make_key_by_hash(0xABC, kv_group=1).split_layers(2)

    engine = _make_dsa_lookup_engine(latent_layers)
    assert engine.lookup([1, 2, 3], lookup_id="req", pin=True) == 0
    assert engine.lookup_pins["req"]["LocalCPUBackend"] == []
    assert engine.storage_manager.pinned == []

    engine = _make_dsa_lookup_engine([*latent_layers, *index_layers[:1]])
    assert engine.lookup([1, 2, 3], lookup_id="req", pin=True) == 0
    assert engine.lookup_pins["req"]["LocalCPUBackend"] == []
    assert engine.storage_manager.pinned == []

    engine = _make_dsa_lookup_engine([*latent_layers, *index_layers])
    assert engine.lookup([1, 2, 3], lookup_id="req", pin=True) == 3
    assert engine.lookup_pins["req"]["LocalCPUBackend"] == [
        *latent_layers,
        *index_layers,
    ]


class _RaceLayerwiseStorageManager:
    def __init__(self, full_latent, partial_index):
        self.full_latent = list(full_latent)
        self.partial_index = list(partial_index)
        self.unpinned = []

    def batched_contains(self, keys, search_range=None, pin=False):
        if not pin:
            return len(keys), {"LocalCPUBackend": list(keys)}
        kv_group = keys[0].kv_group if keys else 0
        if kv_group == 0:
            return len(keys), {"LocalCPUBackend": self.full_latent}
        return len(self.partial_index), {"LocalCPUBackend": self.partial_index}

    def batched_unpin(self, keys, locations=None):
        self.unpinned.append((list(keys), locations))

    def touch_cache(self):
        return None


def test_layerwise_lookup_unpins_current_partial_group_on_pin_race() -> None:
    token_db = _FakeLookupTokenDatabase()
    latent_layers = token_db._make_key_by_hash(0xABC, kv_group=0).split_layers(2)
    index_layers = token_db._make_key_by_hash(0xABC, kv_group=1).split_layers(2)
    engine = _make_dsa_lookup_engine([])
    engine.storage_manager = _RaceLayerwiseStorageManager(
        latent_layers,
        index_layers[:1],
    )

    assert engine.lookup([1, 2, 3], lookup_id="req", pin=True) == 0
    assert engine.lookup_pins["req"]["LocalCPUBackend"] == []
    assert engine.storage_manager.unpinned == [
        (index_layers[:1], ["LocalCPUBackend"]),
        (latent_layers, ["LocalCPUBackend"]),
    ]


class _FakeAsyncLookupServer:
    def __init__(self):
        self.responses = []

    def send_response_to_scheduler(self, lookup_id, num_hit_tokens):
        self.responses.append((lookup_id, num_hit_tokens))


async def _run_lookup_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


def _submit_lookup_inline(coro, _loop):
    asyncio.run(coro)
    return SimpleNamespace()


def _install_inline_async_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        "lmcache.v1.cache_engine.asyncio.to_thread",
        _run_lookup_inline,
    )
    monkeypatch.setattr(
        "lmcache.v1.cache_engine.asyncio.run_coroutine_threadsafe",
        _submit_lookup_inline,
    )


def test_async_lookup_prefetch_layerwise_fails_closed_with_zero_hit() -> None:

    engine = object.__new__(LMCacheEngine)
    async_lookup_server = _FakeAsyncLookupServer()
    engine.storage_manager = SimpleNamespace(
        async_lookup_server=async_lookup_server
    )
    engine.use_layerwise = True

    engine.async_lookup_and_prefetch(
        lookup_id="req",
        hashes=[123],
        offsets=[3],
        pin=True,
    )

    assert async_lookup_server.responses == [("req", 0)]


def test_async_sampled_layerwise_lookup_returns_remote_result(monkeypatch) -> None:
    token_db = _FakeMultiChunkLookupTokenDatabase()
    first_keys = _sampled_keys_for_chunk(token_db, 0)
    tail_keys = _sampled_keys_for_chunk(token_db, 3)
    engine = _make_sampled_lookup_engine([*first_keys, *tail_keys])
    async_lookup_server = _FakeAsyncLookupServer()
    engine.storage_manager.async_lookup_server = async_lookup_server
    engine.storage_manager.loop = object()
    _install_inline_async_lookup(monkeypatch)

    engine.async_lookup_and_prefetch(
        lookup_id="req",
        hashes=[0x100, 0x101, 0x102, 0x103],
        offsets=[4, 4, 4, 2],
        pin=False,
    )

    assert async_lookup_server.responses == [("req", 14)]


def test_async_sampled_lookup_without_remote_falls_back_to_local_cpu(
    monkeypatch,
) -> None:
    token_db = _FakeLookupTokenDatabase()
    latent_layers = token_db._make_key_by_hash(0xABC, kv_group=0).split_layers(2)
    index_layers = token_db._make_key_by_hash(0xABC, kv_group=1).split_layers(2)
    engine = _make_dsa_lookup_engine([*latent_layers, *index_layers])
    engine.storage_manager.storage_backends = {
        "LocalCPUBackend": engine.storage_manager
    }
    engine.config.experimental_sampled_layerwise_lookup = True
    async_lookup_server = _FakeAsyncLookupServer()
    engine.storage_manager.async_lookup_server = async_lookup_server
    engine.storage_manager.loop = object()
    _install_inline_async_lookup(monkeypatch)

    engine.async_lookup_and_prefetch(
        lookup_id="req",
        hashes=[0xABC],
        offsets=[3],
        pin=False,
    )

    assert async_lookup_server.responses == [("req", 3)]


class _CaptureTokenDatabase:
    def __init__(self):
        self.calls = []

    def process_tokens(self, *args, **kwargs):
        self.calls.append(kwargs.get("kv_group", 0))
        return iter(())


class _NoopStoreStats:
    def profile_process_tokens(self):
        return nullcontext()


class _NoopStatsMonitor:
    def on_store_request(self, _num_tokens):
        return _NoopStoreStats()


def test_base_non_layerwise_paths_pass_kv_group_to_key_generation() -> None:
    engine = object.__new__(LMCacheEngine)
    token_database = _CaptureTokenDatabase()
    engine.token_database = token_database
    engine.gpu_connector = object()
    engine.storage_manager = object()
    engine.stats_monitor = _NoopStatsMonitor()
    engine.is_healthy = lambda: True
    engine._is_passive = lambda: False
    engine.is_frozen = lambda: False
    engine._get_req_id = lambda _kwargs: "req"
    engine._log_kvcache_for_check = lambda **_kwargs: None

    list(
        engine.store(
            [1, 2, 3],
            kv_group=1,
            request_configs={"lmcache.tag.case": "base-store"},
        )
        or []
    )

    assert token_database.calls == [1]

    token_database.calls.clear()
    engine.storage_manager = SimpleNamespace(
        get_block_mapping=lambda _chunk_infos: {},
    )
    ret_mask = torch.zeros(3, dtype=torch.bool, device="cpu")
    chunks, size = engine._process_tokens_internal(
        [1, 2, 3],
        None,
        ret_mask,
        kv_group=1,
        request_configs={"lmcache.tag.case": "base-retrieve"},
    )

    assert chunks == []
    assert size == 0
    assert token_database.calls == [1]

    token_database.calls.clear()

    class _Future:
        def result(self):
            return []

    engine.event_manager = SimpleNamespace(
        get_event_future=lambda _event_type, _req_id: _Future(),
    )
    chunks, size = engine._async_process_tokens_internal(
        [1, 2, 3],
        None,
        ret_mask,
        req_id="req",
        kv_group=1,
        request_configs={"lmcache.tag.case": "base-async-retrieve"},
    )

    assert chunks == []
    assert size == 0
    assert token_database.calls == [1]


def _make_memory_obj(
    backing: torch.Tensor,
    *,
    offset: int = 128,
    logical_size: int = 16,
    physical_size: int = 64,
    kv_group: int = 0,
) -> TensorMemoryObj:
    raw = backing[offset : offset + logical_size]
    metadata = MemoryObjMetadata(
        shape=torch.Size([8]),
        dtype=torch.float16,
        address=offset,
        phy_size=physical_size,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT
        if kv_group == 0
        else MemoryFormat.KV_DSA_INDEX_FMT,
        cached_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        shapes=[torch.Size([8])],
        dtypes=[torch.float16],
    )
    return TensorMemoryObj(
        raw_data=raw,
        metadata=metadata,
        parent_allocator=None,
    )


def _make_engine_for_contract(*, use_layerwise: bool, sparse: bool, shared: bool):
    engine = object.__new__(LMCacheEngine)
    engine.metadata = SimpleNamespace(use_mla=True, world_size=2)
    engine.save_only_first_rank = True
    engine.enable_shared_cpu_cache = shared
    engine.dsa_two_groups = True
    engine.shared_cpu_cache_strict = True
    engine.config = SimpleNamespace(
        use_layerwise=use_layerwise,
        enable_sparse_attention=sparse,
        local_cpu=True,
        max_local_cpu_size=1,
        get_extra_config_value=lambda key, default=None: default,
    )
    return engine


class _FakeSharedShapeConnector:
    def get_shape(self, num_tokens: int, kv_group: int = 0) -> torch.Size:
        hidden = 1024 if kv_group == 0 else 128
        return torch.Size([num_tokens, hidden])


class _LegacyShapeConnector:
    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 1024])


class _MisleadingGroupShapeConnector:
    def get_shape(self, num_tokens: int, kv_group: int = 0) -> torch.Size:
        hidden = 1024 if kv_group == 0 else 4096
        return torch.Size([num_tokens, hidden])


def _make_engine_for_sparse_capacity(*, max_local_cpu_size: float):
    engine = object.__new__(LMCacheEngine)
    extra_config = {
        "vllm_max_model_len": 1024,
        "vllm_max_num_seqs": 32,
    }
    engine.config = SimpleNamespace(
        enable_sparse_attention=True,
        chunk_size=256,
        max_local_cpu_size=max_local_cpu_size,
        extra_config=extra_config,
        get_extra_config_value=lambda key, default=None: extra_config.get(
            key,
            default,
        ),
    )
    engine.metadata = SimpleNamespace(
        world_size=8,
        is_first_rank=lambda: True,
        max_model_len=1024,
        kv_dtype=torch.float16,
        get_dtypes=lambda: [torch.float16],
        get_shapes=lambda num_tokens: [torch.Size([num_tokens, 1024])],
    )
    engine.num_layers = 4
    engine.save_only_first_rank = True
    engine.enable_shared_cpu_cache = True
    engine.dsa_two_groups = True
    engine.shared_cpu_cache_strict = True
    engine.gpu_connector = _FakeSharedShapeConnector()
    engine._shared_cpu_request_leases = {}
    return engine


def test_shared_cpu_group1_shape_uses_metadata_when_connector_lacks_kv_group():
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(dsa_two_groups=True)
    engine.gpu_connector = _LegacyShapeConnector()
    engine.metadata = SimpleNamespace(
        use_mla=True,
        get_dtypes=lambda: [torch.float16, torch.uint8],
        get_shapes=lambda num_tokens: [
            torch.Size([num_tokens, 1024]),
            torch.Size([num_tokens, 128]),
        ],
    )

    shape, dtype, fmt = engine._expected_shared_cpu_chunk_metadata(
        kv_group=1,
        num_tokens=17,
    )

    assert shape == torch.Size([17, 128])
    assert dtype == torch.uint8
    assert fmt == MemoryFormat.KV_DSA_INDEX_FMT


def test_shared_cpu_group1_shape_prefers_metadata_over_connector():
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(dsa_two_groups=True)
    engine.gpu_connector = _MisleadingGroupShapeConnector()
    engine.metadata = SimpleNamespace(
        use_mla=True,
        get_dtypes=lambda: [torch.float16, torch.uint8],
        get_shapes=lambda num_tokens: [
            torch.Size([num_tokens, 1024]),
            torch.Size([num_tokens, 128]),
        ],
    )

    shape, _, _ = engine._expected_shared_cpu_chunk_metadata(
        kv_group=1,
        num_tokens=17,
    )

    assert shape == torch.Size([17, 128])


def test_shared_cpu_group1_shape_missing_metadata_fails_loudly():
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(dsa_two_groups=True)
    engine.gpu_connector = _LegacyShapeConnector()
    engine.metadata = SimpleNamespace(
        use_mla=True,
        get_dtypes=lambda: [torch.float16],
        get_shapes=lambda num_tokens: [torch.Size([num_tokens, 1024])],
    )

    with pytest.raises(ValueError, match="KV group shape metadata"):
        engine._expected_shared_cpu_chunk_metadata(
            kv_group=1,
            num_tokens=17,
        )


@pytest.mark.no_shared_allocator
def test_shared_cpu_size_override_wins_over_first_rank_size(monkeypatch):
    captured = {}

    class DummyMixedMemoryAllocator:
        def __init__(self, size, **kwargs):
            captured["size"] = size
            captured["kwargs"] = kwargs

    import lmcache.v1.cache_engine as cache_engine_module

    monkeypatch.setattr(
        cache_engine_module,
        "MixedMemoryAllocator",
        DummyMixedMemoryAllocator,
    )
    config = SimpleNamespace(
        extra_config={
            "save_only_first_rank": True,
            "enable_shared_cpu_cache": True,
            "shared_cpu_cache_size_gb": 3,
            "first_rank_max_local_cpu_size": 9,
        },
        gds_path=None,
        cufile_buffer_size=None,
        max_local_cpu_size=5,
        get_extra_config_value=lambda key, default=None: config.extra_config.get(
            key,
            default,
        ),
    )
    metadata = SimpleNamespace(use_mla=True, is_first_rank=lambda: True)

    allocator = LMCacheEngineBuilder._Create_memory_allocator(
        config,
        metadata,
        None,
    )

    assert isinstance(allocator, DummyMixedMemoryAllocator)
    assert captured["size"] == 3 * 1024**3


def test_shared_cpu_shm_capacity_preflight_reports_sigbus_risk(monkeypatch):
    engine = object.__new__(LMCacheEngine)
    engine.enable_shared_cpu_cache = True
    engine.shared_cpu_cache_name = "/lmcache-too-large"
    engine.metadata = SimpleNamespace(is_first_rank=lambda: True)
    engine.config = SimpleNamespace(
        max_local_cpu_size=2,
        get_extra_config_value=lambda key, default=None: default,
    )

    monkeypatch.setattr("os.path.isdir", lambda path: path == "/dev/shm")
    monkeypatch.setattr(
        "os.statvfs",
        lambda _path: SimpleNamespace(f_bavail=1, f_frsize=1024**3),
    )

    with pytest.raises(ValueError, match="SIGBUS"):
        engine._preflight_shared_cpu_shm_capacity()


class _FakeAddressManager:
    def __init__(self, free_bytes: int):
        self._free_bytes = free_bytes
        self.total_allocated_size = 0

    def get_free_size(self) -> int:
        return self._free_bytes


class _FakeLocalCPUBackend:
    def __init__(self, *, free_bytes: int, hot_cache: dict):
        self.hot_cache = hot_cache
        self.cpu_lock = nullcontext()
        pin_allocator = SimpleNamespace(
            address_manager=_FakeAddressManager(free_bytes),
        )
        self.memory_allocator = SimpleNamespace(
            buffer=torch.empty(1024, dtype=torch.uint8),
            pin_allocator=pin_allocator,
            align_bytes=64,
        )


class _FakeResolvableMemoryObj:
    def __init__(self):
        self.is_pinned = False
        self.ref_count_down_count = 0

    def is_valid(self):
        return self.ref_count_down_count == 0

    def pin(self):
        self.is_pinned = True

    def unpin(self):
        self.is_pinned = False

    def ref_count_down(self):
        self.ref_count_down_count += 1


class _FakeLayerwiseGPUConnector:
    def __init__(self):
        self.close_count = 0
        self.sent = []

    def batched_to_gpu(self, starts, ends, **kwargs):
        try:
            while True:
                mem_objs = yield
                if mem_objs is not None:
                    self.sent.append(mem_objs)
        finally:
            self.close_count += 1


class _FakePassiveSharedView:
    def __init__(self):
        self.ref_count_down_count = 0

    def is_valid(self):
        return self.ref_count_down_count == 0

    def ref_count_down(self):
        self.ref_count_down_count += 1


class _FakePassiveSharedAllocator:
    def __init__(self):
        self.views = []
        self.shm_name = "/lmcache-test"
        self.slab_size = 4096

    def create_view(self, *args, **kwargs):
        view = _FakePassiveSharedView()
        self.views.append(view)
        return view

    def create_batch_view(self, *args, **kwargs):
        return self.create_view()


def _make_passive_shared_retrieve_engine(
    *,
    kv_group: int,
    num_layers: int = 2,
    requests: tuple[tuple[str, int], ...] = (("req-1", 0),),
) -> LMCacheEngine:
    engine = object.__new__(LMCacheEngine)
    engine.gpu_connector = _FakeLayerwiseGPUConnector()
    engine.num_layers = num_layers
    engine.shared_cpu_cache_generation = 9
    engine.shared_cpu_cache_passive_allocator = _FakePassiveSharedAllocator()
    engine.metadata = SimpleNamespace(first_rank=0)
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_finished=lambda monitor_req_id, tokens: None
    )
    engine._expected_shared_cpu_chunk_metadata = lambda **kwargs: (
        torch.Size([4]),
        torch.float16,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    envelopes = iter(
        [
            SharedHandleEnvelope(
                request_id=req_id,
                phase="dense_prefix",
                request_ordinal=request_ordinal,
                layer_id=layer_id,
                kv_group=kv_group,
                status="ok",
                generation=9,
                handles=[object()],
            )
            for req_id, request_ordinal in requests
            for layer_id in range(num_layers)
        ]
    )
    engine._receive_shared_envelope = lambda: next(envelopes)
    return engine


def _make_passive_shared_retriever(
    engine: LMCacheEngine,
    *,
    req_id: str = "req-1",
    request_ordinal: int = 0,
    kv_group: int = 0,
):
    ret_mask = torch.zeros(4, dtype=torch.bool)
    keys_by_layer = _make_key().split_layers(engine.num_layers)
    retriever = engine._retrieve_layer_shared_passive(
        starts_all=[0],
        ends_all=[4],
        keys_layer_major=[[key] for key in keys_by_layer],
        ret_mask=ret_mask,
        monitor_req_id=123,
        req_id=req_id,
        kv_group=kv_group,
        kwargs={
            "shared_cpu_phase": "dense_prefix",
            "shared_cpu_request_ordinal": request_ordinal,
        },
    )
    return retriever, ret_mask


class _FakeGetBlockingLocalCPUBackend:
    def __init__(self, hot_obj):
        self.hot_obj = hot_obj

    def get_blocking(self, key):
        return self.hot_obj


def test_engine_contract_requires_shared_cache_for_dense_layerwise_tp():
    engine = _make_engine_for_contract(
        use_layerwise=True,
        sparse=False,
        shared=False,
    )

    with pytest.raises(ValueError, match="use_layerwise=true") as exc_info:
        engine._validate_shared_cpu_cache_contract()
    message = str(exc_info.value)
    assert "enable_shared_cpu_cache" in message
    assert "save_only_first_rank" in message
    assert "TP/world_size=2" in message
    assert "shared_cpu_cache_size_gb" in message


def test_engine_contract_requires_shared_cache_for_sparse_tp():
    engine = _make_engine_for_contract(
        use_layerwise=False,
        sparse=True,
        shared=False,
    )

    with pytest.raises(ValueError, match="enable_sparse_attention=true") as exc_info:
        engine._validate_shared_cpu_cache_contract()
    message = str(exc_info.value)
    assert "enable_shared_cpu_cache" in message
    assert "save_only_first_rank" in message
    assert "TP/world_size=2" in message


def test_engine_contract_requires_broadcast_object_fn_for_shared_tp():
    engine = _make_engine_for_contract(
        use_layerwise=True,
        sparse=False,
        shared=True,
    )
    engine.broadcast_object_fn = None

    with pytest.raises(ValueError, match="broadcast_object_fn"):
        engine._validate_shared_cpu_cache_contract()


def test_engine_contract_requires_index_materialization_for_strict_sparse():
    engine = _make_engine_for_contract(
        use_layerwise=True,
        sparse=True,
        shared=True,
    )
    engine.broadcast_object_fn = lambda obj, src=0: obj
    engine.config.get_extra_config_value = (
        lambda key, default=None: False
        if key == "shared_cpu_materialize_index_on_decode_cold"
        else default
    )

    with pytest.raises(ValueError, match="must materialize DSA index"):
        engine._validate_shared_cpu_cache_contract()


def test_rank0_post_init_broadcasts_startup_error_on_storage_failure(
    monkeypatch,
):
    engine = object.__new__(LMCacheEngine)
    engine.post_inited = False
    engine.enable_shared_cpu_cache = True
    engine.use_layerwise = False
    engine.save_only_first_rank = True
    engine.lmcache_worker = None
    engine.event_manager = object()
    engine.storage_manager = None
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_slab_size = None
    engine.shared_cpu_cache_generation = 0
    engine.metadata = SimpleNamespace(
        use_mla=True,
        world_size=2,
        worker_id=0,
        first_rank=0,
        is_first_rank=lambda: True,
    )
    engine.config = SimpleNamespace(
        get_lookup_server_worker_ids=lambda use_mla, world_size: [],
    )
    broadcasts = []
    engine.broadcast_object_fn = lambda payload, src: broadcasts.append(
        (payload, src)
    )

    def fail_storage_manager(*args, **kwargs):
        raise RuntimeError("stale shm segment")

    monkeypatch.setattr(
        "lmcache.v1.cache_engine.StorageManager",
        fail_storage_manager,
    )

    with pytest.raises(RuntimeError, match="stale shm segment"):
        engine.post_init()

    assert len(broadcasts) == 1
    envelope, src = broadcasts[0]
    assert src == 0
    assert envelope["status"] == "error"
    assert envelope["shm_name"] == "/lmcache-test"
    assert "StorageManager" in envelope["message"]
    assert "stale shm segment" in envelope["message"]


def test_sparse_capacity_preflight_fails_when_one_max_request_cannot_fit():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=0.001)

    with pytest.raises(ValueError, match="one maximum request cannot fit"):
        engine._report_shared_cpu_sparse_capacity_sanity()


def test_sparse_capacity_preflight_records_startup_estimate():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)

    engine._report_shared_cpu_sparse_capacity_sanity()

    estimate = engine.config.extra_config[
        "shared_cpu_sparse_startup_capacity_estimate"
    ]
    assert estimate["max_model_len"] == 1024
    assert estimate["max_num_seqs"] == 32
    assert estimate["kv_groups"] == [0, 1]
    assert estimate["one_max_request_bytes"] > 0
    assert estimate["configured_worst_case_bytes"] == (
        estimate["one_max_request_bytes"] * 32
    )


def test_shared_cpu_index_group_dtype_uses_single_dtype_metadata():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)

    assert engine._shared_cpu_dtype_for_kv_group(1) is torch.float16


def test_sparse_capacity_shape_helper_keeps_two_dim_token_shape():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    engine.num_layers = 256

    assert engine._shape_numel_without_layer_dim(torch.Size([256, 1024])) == (
        256 * 1024
    )


def test_runtime_capacity_skips_hot_cache_scan_when_free_space_is_sufficient():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    miss_key = _make_key()

    class _NoScanHotCache(dict):
        def items(self):
            raise AssertionError("hot cache must not be scanned")

    backend = _FakeLocalCPUBackend(
        free_bytes=1 << 30,
        hot_cache=_NoScanHotCache(),
    )
    engine._shared_local_cpu_backend = lambda: backend
    engine.config.chunk_size = 4

    details = engine._shared_cpu_runtime_capacity_details(
        req_id="req-fast",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=[[miss_key]],
        chunk_locations_layer_major=[["MooncakeStore"]],
        token_count=4,
        chunk_token_lengths=[1],
    )

    assert details["fits"] is True
    assert details["capacity_scan_skipped"] is True
    assert details["capacity_scan_skip_reason"] == "free_capacity_sufficient"
    assert details["available_after_eviction"] == 1 << 30


def test_runtime_capacity_details_exclude_required_hot_chunks_from_evictable():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    hot_key = _make_key()
    miss_key = CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=5678,
        dtype=torch.float16,
        kv_group=0,
    )
    other_key = CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=9999,
        dtype=torch.float16,
        kv_group=0,
    )
    hot_obj = _make_memory_obj(
        torch.empty(1024, dtype=torch.uint8),
        physical_size=64,
    )
    other_obj = _make_memory_obj(
        torch.empty(1024, dtype=torch.uint8),
        offset=256,
        physical_size=64,
    )
    backend = _FakeLocalCPUBackend(
        free_bytes=9000,
        hot_cache={hot_key: hot_obj, other_key: other_obj},
    )
    engine._shared_local_cpu_backend = lambda: backend
    engine._is_rank0_shared_mem_obj = lambda mem_obj: mem_obj in (
        hot_obj,
        other_obj,
    )
    engine.config.chunk_size = 4
    engine._shared_cpu_request_leases["req-old"] = SharedCPURequestLease(
        request_id="req-old",
        generation=0,
        is_rank0=True,
        active=True,
    )

    details = engine._shared_cpu_runtime_capacity_details(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=[[hot_key, miss_key]],
        chunk_locations_layer_major=[["LocalCPUBackend", "MooncakeStore"]],
        token_count=8,
        chunk_token_lengths=[1, 1],
    )

    expected_missing_bytes = engine._shared_cpu_estimated_physical_chunk_bytes(0)
    assert details["required_bytes"] == expected_missing_bytes
    assert details["available_after_eviction"] == 9064
    assert details["protected_hot_bytes"] == 64
    assert details["hot_chunk_count"] == 1
    assert details["non_shm_hot_chunk_count"] == 0
    assert details["active_sparse_requests"] == 2
    assert details["fits"] is True


def test_capacity_snapshot_reads_nested_pin_allocator_free_space():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    backend = _FakeLocalCPUBackend(free_bytes=768, hot_cache={})
    engine._shared_local_cpu_backend = lambda: backend

    snapshot = engine._shared_cpu_capacity_snapshot()

    assert snapshot["slab_bytes"] == 1024
    assert snapshot["free_bytes"] == 768
    assert snapshot["allocated_bytes"] == 0


def test_runtime_capacity_counts_non_shm_hot_hits_as_required_bytes():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    hot_key = _make_key()
    hot_obj = _make_memory_obj(
        torch.empty(1024, dtype=torch.uint8),
        physical_size=64,
    )
    backend = _FakeLocalCPUBackend(
        free_bytes=0,
        hot_cache={hot_key: hot_obj},
    )
    engine._shared_local_cpu_backend = lambda: backend
    engine._is_rank0_shared_mem_obj = lambda _mem_obj: False
    engine.config.chunk_size = 4

    details = engine._shared_cpu_runtime_capacity_details(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=[[hot_key]],
        chunk_locations_layer_major=[["LocalCPUBackend"]],
        token_count=1,
        chunk_token_lengths=[1],
    )

    expected_bytes = engine._shared_cpu_estimated_physical_chunk_bytes(
        0,
        num_tokens=1,
    )
    assert details["required_bytes"] == expected_bytes
    assert details["available_after_eviction"] == 0
    assert details["protected_hot_bytes"] == 0
    assert details["hot_chunk_count"] == 0
    assert details["non_shm_hot_chunk_count"] == 1
    assert details["fits"] is False


def test_runtime_capacity_details_report_failure_before_materialization():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    miss_key = _make_key()
    backend = _FakeLocalCPUBackend(free_bytes=0, hot_cache={})
    engine._shared_local_cpu_backend = lambda: backend
    engine.config.chunk_size = 4

    details = engine._shared_cpu_runtime_capacity_details(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=[[miss_key]],
        chunk_locations_layer_major=[["MooncakeStore"]],
        token_count=4,
        chunk_token_lengths=[1],
    )

    assert details[
        "required_bytes"
    ] == engine._shared_cpu_estimated_physical_chunk_bytes(0)
    assert details["available_after_eviction"] == 0
    assert details["fits"] is False


def test_runtime_capacity_uses_full_chunks_for_remote_fetch():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    engine._shared_local_cpu_backend = lambda: _FakeLocalCPUBackend(
        free_bytes=0, hot_cache={}
    )
    engine.config.chunk_size = 4
    calls = []
    engine._shared_cpu_estimated_physical_chunk_bytes = (
        lambda _group, num_tokens=None: calls.append(num_tokens)
        or (num_tokens or 4)
    )

    details = engine._shared_cpu_runtime_capacity_details(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=[[_make_key()], [_make_key()]],
        chunk_locations_layer_major=[["MooncakeStore"], ["MooncakeStore"]],
        chunk_token_lengths=[1],
    )

    assert details["required_bytes"] == 8
    assert calls == [None]


def test_rank0_resolver_rematerializes_non_shm_hot_cache_hit():
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = object()
    hot_obj = _FakeResolvableMemoryObj()
    materialized_obj = _FakeResolvableMemoryObj()
    backend = _FakeGetBlockingLocalCPUBackend(hot_obj)
    key = _make_key()
    materialized_from = []

    engine._shared_local_cpu_backend = lambda: backend
    engine._is_rank0_shared_mem_obj = lambda _obj: False
    engine._validate_rank0_shared_mem_obj = lambda *args, **kwargs: None

    def materialize_shared_copy(**kwargs):
        materialized_from.append(kwargs["src_obj"])
        return materialized_obj

    engine._materialize_shared_rank0_copy = materialize_shared_copy

    resolved = engine._resolve_shared_rank0_layer_mem_objs(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        layer_id=0,
        kv_group=0,
        keys_layer=[key],
        chunk_locations=["LocalCPUBackend"],
    )

    assert resolved == [materialized_obj]
    assert materialized_from == [hot_obj]
    assert hot_obj.ref_count_down_count == 1
    assert materialized_obj.is_pinned


def test_rank0_resolver_scatters_remote_suffix_in_token_order():
    keys = [replace(_make_key(), chunk_hash=0x200 + i) for i in range(4)]
    local_first = _FakeResolvableMemoryObj()
    remote_second = _FakeResolvableMemoryObj()
    remote_third = _FakeResolvableMemoryObj()
    remote_fourth = _FakeResolvableMemoryObj()
    staged_second = _FakeResolvableMemoryObj()
    staged_third = _FakeResolvableMemoryObj()
    staged_fourth = _FakeResolvableMemoryObj()

    class _MissingLocalBackend:
        def get_blocking(self, _key):
            return None

    class _RemoteStorageManager:
        def __init__(self):
            self.calls = []

        def batched_get(self, fetch_keys, location=None):
            self.calls.append((list(fetch_keys), location))
            return [remote_second, remote_third, remote_fourth]

    storage_manager = _RemoteStorageManager()
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = storage_manager
    engine._shared_local_cpu_backend = lambda: _MissingLocalBackend()
    shared_objs = {
        local_first,
        staged_second,
        staged_third,
        staged_fourth,
    }
    engine._is_rank0_shared_mem_obj = lambda obj: obj in shared_objs
    engine._validate_rank0_shared_mem_obj = lambda *args, **kwargs: None
    staged_by_source = {
        remote_second: staged_second,
        remote_third: staged_third,
        remote_fourth: staged_fourth,
    }
    engine._materialize_shared_rank0_copy = lambda **kwargs: staged_by_source[
        kwargs["src_obj"]
    ]

    resolved = engine._resolve_shared_rank0_layer_mem_objs(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        layer_id=0,
        kv_group=0,
        keys_layer=keys,
        local_prefix=LocalCPUPrefixGetResult(
            [local_first],
            [1, 2, 3],
            keys[1:],
        ),
    )

    assert resolved == [
        local_first,
        staged_second,
        staged_third,
        staged_fourth,
    ]
    assert storage_manager.calls == [(keys[1:], "RemoteBackend")]
    assert remote_second.ref_count_down_count == 1
    assert remote_third.ref_count_down_count == 1
    assert remote_fourth.ref_count_down_count == 1
    assert all(obj.is_pinned for obj in resolved)


def test_rank0_resolver_reuses_remote_object_already_in_shared_slab():
    keys = [replace(_make_key(), chunk_hash=0x280 + i) for i in range(2)]
    remote_objects = [_FakeResolvableMemoryObj(), _FakeResolvableMemoryObj()]

    class _NoHotLookupBackend:
        def get_blocking(self, _key):
            raise AssertionError("shared remote objects must not be looked up again")

    class _RemoteStorageManager:
        def batched_get(self, fetch_keys, location=None):
            assert fetch_keys == keys
            assert location == "RemoteBackend"
            return list(remote_objects)

    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = _RemoteStorageManager()
    engine._shared_local_cpu_backend = lambda: _NoHotLookupBackend()
    engine._is_rank0_shared_mem_obj = lambda obj: obj in remote_objects
    engine._validate_rank0_shared_mem_obj = lambda *args, **kwargs: None
    engine._materialize_shared_rank0_copy = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("shared remote objects must not be copied")
    )

    resolved = engine._resolve_shared_rank0_layer_mem_objs(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        layer_id=0,
        kv_group=0,
        keys_layer=keys,
        local_prefix=LocalCPUPrefixGetResult([], [0, 1], keys),
    )

    assert resolved == remote_objects
    assert all(obj.is_pinned for obj in resolved)
    assert all(obj.ref_count_down_count == 0 for obj in resolved)


def test_page_first_resolver_joins_common_local_prefix_and_remote_suffix():
    keys = [
        [replace(_make_key(), chunk_hash=layer * 10 + chunk) for chunk in range(3)]
        for layer in range(2)
    ]
    local = [
        [_FakeResolvableMemoryObj(), _FakeResolvableMemoryObj()],
        [_FakeResolvableMemoryObj()],
    ]
    remote = [
        [_FakeResolvableMemoryObj(), _FakeResolvableMemoryObj()],
        [_FakeResolvableMemoryObj(), _FakeResolvableMemoryObj()],
    ]
    prefixes = [
        LocalCPUPrefixGetResult(
            list(layer),
            list(range(len(layer), 3)),
            keys[layer_id][len(layer) :],
        )
        for layer_id, layer in enumerate(local)
    ]
    engine = object.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = object()
    engine._shared_local_cpu_backend = lambda: object()
    engine._is_rank0_shared_mem_obj = lambda _obj: True
    engine._validate_rank0_shared_mem_obj = lambda *_args, **_kwargs: None
    remote_calls = []

    def resolve_remote(**kwargs):
        remote_calls.append(kwargs)
        return remote

    engine._resolve_shared_rank0_remote_layers_windowed = resolve_remote

    resolved = engine._resolve_shared_rank0_page_first_layers(
        req_id="req-mixed",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=keys,
        local_prefix_layers=prefixes,
    )

    assert resolved == [
        [local[0][0], *remote[0]],
        [local[1][0], *remote[1]],
    ]
    assert local[0][1].ref_count_down_count == 1
    assert remote_calls[0]["keys_layer_major"] == [layer[1:] for layer in keys]
    assert remote_calls[0]["layers_per_batch"] == 2
    assert all(obj.is_pinned for obj in (local[0][0], local[1][0]))


def test_page_first_resolver_rolls_back_local_prefix_on_remote_failure():
    keys = [
        [replace(_make_key(), chunk_hash=layer * 10 + chunk) for chunk in range(2)]
        for layer in range(2)
    ]
    local = [[_FakeResolvableMemoryObj()] for _ in range(2)]
    prefixes = [
        LocalCPUPrefixGetResult(list(layer), [1], keys[layer_id][1:])
        for layer_id, layer in enumerate(local)
    ]
    engine = object.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = object()
    engine._shared_local_cpu_backend = lambda: object()
    engine._is_rank0_shared_mem_obj = lambda _obj: True
    engine._validate_rank0_shared_mem_obj = lambda *_args, **_kwargs: None
    engine._resolve_shared_rank0_remote_layers_windowed = lambda **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("remote failed"))

    with pytest.raises(RuntimeError, match="remote failed"):
        engine._resolve_shared_rank0_page_first_layers(
            req_id="req-mixed",
            phase="sparse_decode_bootstrap",
            kv_group=0,
            keys_layer_major=keys,
            local_prefix_layers=prefixes,
        )

    assert all(not obj.is_pinned for layer in local for obj in layer)
    assert all(obj.ref_count_down_count == 1 for layer in local for obj in layer)


def test_rank0_windowed_remote_resolver_batches_layers_and_preserves_order():
    keys_layer_major = [
        [
            replace(_make_key(), chunk_hash=0x400 + layer * 2 + chunk)
            for chunk in range(2)
        ]
        for layer in range(5)
    ]
    objects_by_key = {
        key: _FakeResolvableMemoryObj()
        for layer_keys in keys_layer_major
        for key in layer_keys
    }

    class _WindowedStorageManager:
        def __init__(self):
            self.calls = []

        def batched_get(self, fetch_keys, location=None):
            self.calls.append((list(fetch_keys), location))
            return [objects_by_key[key] for key in fetch_keys]

    storage_manager = _WindowedStorageManager()
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = storage_manager
    engine._is_rank0_shared_mem_obj = lambda obj: obj in objects_by_key.values()
    engine._validate_rank0_shared_mem_obj = lambda *args, **kwargs: None

    resolved = engine._resolve_shared_rank0_remote_layers_windowed(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=keys_layer_major,
        layers_per_batch=2,
    )

    assert [len(call[0]) for call in storage_manager.calls] == [4, 4, 2]
    assert all(call[1] == "RemoteBackend" for call in storage_manager.calls)
    assert resolved == [
        [objects_by_key[key] for key in layer_keys]
        for layer_keys in keys_layer_major
    ]
    assert all(obj.is_pinned for layer in resolved for obj in layer)


def test_rank0_windowed_remote_resolver_uses_cached_slab_context(monkeypatch):
    keys_layer_major = [
        [
            replace(_make_key(), chunk_hash=0x500 + layer * 2 + chunk)
            for chunk in range(2)
        ]
        for layer in range(3)
    ]
    root_allocator = TensorMemoryAllocator(torch.empty(32768, dtype=torch.uint8))
    objects = root_allocator.batched_allocate(
        torch.Size([8]),
        torch.float16,
        len(keys_layer_major) * 2,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert objects is not None
    objects_by_key = dict(
        zip(
            (key for layer_keys in keys_layer_major for key in layer_keys),
            objects,
            strict=True,
        )
    )
    fetched_by_key = dict(objects_by_key)
    fallback_key = keys_layer_major[1][0]
    foreign_obj = _FakeResolvableMemoryObj()
    fetched_by_key[fallback_key] = foreign_obj
    batch_pins = []
    monitor = SimpleNamespace(
        on_pin=lambda _obj: None,
        on_pin_many=lambda objs: batch_pins.append(tuple(objs)),
        on_unpin=lambda _obj: None,
    )
    monkeypatch.setattr(PinMonitor, "GetOrCreate", lambda _config=None: monitor)
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = SimpleNamespace(
        batched_get=lambda keys, location=None: [fetched_by_key[key] for key in keys]
    )
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.metadata = SimpleNamespace(
        use_mla=True,
        get_dtypes=lambda: [torch.float16],
    )
    engine.config = SimpleNamespace(dsa_two_groups=True)
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        memory_allocator=SimpleNamespace(
            shm_name="/lmcache-test",
            pin_allocator=root_allocator,
            buffer=root_allocator.buffer,
        )
    )
    metadata_checks = []

    def checked_metadata(mem_obj, context):
        metadata_checks.append(mem_obj)
        return LMCacheEngine._shared_rank0_object_metadata(mem_obj, context)

    engine._shared_rank0_object_metadata = checked_metadata
    engine._is_rank0_shared_mem_obj = lambda _obj: (_ for _ in ()).throw(
        AssertionError("cached-context path must not repeat shared-object checks")
    )
    engine._materialize_shared_rank0_copy = (
        lambda **kwargs: objects_by_key[kwargs["key"]]
    )
    stages = []
    engine._shared_rank0_resolver_timing_hook = (
        lambda stage, _elapsed: stages.append(stage)
    )

    resolved = engine._resolve_shared_rank0_remote_layers_windowed(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=keys_layer_major,
        layers_per_batch=3,
    )

    assert resolved == [
        [objects_by_key[key] for key in layer_keys]
        for layer_keys in keys_layer_major
    ]
    assert all(obj.is_pinned for obj in objects)
    assert len(batch_pins) == 1
    assert {id(obj) for obj in batch_pins[0]} == {id(obj) for obj in objects}
    assert metadata_checks == [foreign_obj, objects_by_key[fallback_key]]
    assert foreign_obj.ref_count_down_count == 1
    assert stages == [
        "windows",
        "results",
        "classification",
        "pinning",
        "scatter",
    ]
    for obj in objects:
        obj.unpin()
        obj.ref_count_down()


def test_rank0_windowed_cached_context_validation_failure_rolls_back(monkeypatch):
    keys = [[_make_key(), replace(_make_key(), chunk_hash=0x601)]]
    root_allocator = TensorMemoryAllocator(torch.empty(8192, dtype=torch.uint8))
    valid_batch = root_allocator.batched_allocate(
        torch.Size([8]),
        torch.float16,
        1,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    invalid_batch = root_allocator.batched_allocate(
        torch.Size([8]),
        torch.bfloat16,
        1,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert valid_batch is not None and invalid_batch is not None
    objects = [valid_batch[0], invalid_batch[0]]
    batch_pins = []
    monitor = SimpleNamespace(
        on_pin=lambda _obj: None,
        on_pin_many=lambda objs: batch_pins.append(tuple(objs)),
        on_unpin=lambda _obj: None,
    )
    monkeypatch.setattr(PinMonitor, "GetOrCreate", lambda _config=None: monitor)
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = SimpleNamespace(
        batched_get=lambda _keys, location=None: list(objects)
    )
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.metadata = SimpleNamespace(
        use_mla=True,
        get_dtypes=lambda: [torch.float16],
    )
    engine.config = SimpleNamespace(dsa_two_groups=True)
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        memory_allocator=SimpleNamespace(
            shm_name="/lmcache-test",
            pin_allocator=root_allocator,
            buffer=root_allocator.buffer,
        )
    )
    stages = []
    engine._shared_rank0_resolver_timing_hook = (
        lambda stage, _elapsed: stages.append(stage)
    )

    with pytest.raises(ValueError, match="dtype does not match"):
        engine._resolve_shared_rank0_remote_layers_windowed(
            req_id="req-1",
            phase="sparse_decode_bootstrap",
            kv_group=0,
            keys_layer_major=keys,
            layers_per_batch=1,
        )

    assert all(not obj.is_pinned and not obj.is_valid() for obj in objects)
    assert batch_pins == []
    assert stages[-1] == "rollback"


def test_rank0_windowed_failed_pin_rolls_back_all_objects():
    class _FailedPinMemoryObj(_FakeResolvableMemoryObj):
        def pin(self):
            return False

    objects = [_FakeResolvableMemoryObj(), _FailedPinMemoryObj()]
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = SimpleNamespace(
        batched_get=lambda _keys, location=None: objects
    )
    engine._is_rank0_shared_mem_obj = lambda _obj: True
    engine._validate_rank0_shared_mem_obj = lambda *_args, **_kwargs: None

    with pytest.raises(ValueError, match="failed to pin"):
        engine._resolve_shared_rank0_remote_layers_windowed(
            req_id="req-1",
            phase="sparse_decode_bootstrap",
            kv_group=0,
            keys_layer_major=[[_make_key(), _make_key()]],
            layers_per_batch=1,
        )

    assert all(not obj.is_pinned for obj in objects)
    assert all(obj.ref_count_down_count == 1 for obj in objects)


def test_rank0_windowed_fallback_verifies_successful_pin():
    class _NoopPinMemoryObj(_FakeResolvableMemoryObj):
        def pin(self):
            return True

    memory_obj = _NoopPinMemoryObj()
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = SimpleNamespace(
        batched_get=lambda _keys, location=None: [memory_obj]
    )
    engine._is_rank0_shared_mem_obj = lambda _obj: True

    def validate(mem_obj, *_args, _require_pinned=True, **_kwargs):
        if _require_pinned and not mem_obj.is_pinned:
            raise ValueError("must be pinned")

    engine._validate_rank0_shared_mem_obj = validate

    with pytest.raises(ValueError, match="must be pinned"):
        engine._resolve_shared_rank0_remote_layers_windowed(
            req_id="req-1",
            phase="sparse_decode_bootstrap",
            kv_group=0,
            keys_layer_major=[[_make_key()]],
            layers_per_batch=1,
        )

    assert not memory_obj.is_pinned
    assert memory_obj.ref_count_down_count == 1


def test_rank0_resolver_releases_prefetched_hits_on_alignment_error():
    keys = [replace(_make_key(), chunk_hash=0x300 + i) for i in range(2)]
    local_obj = _FakeResolvableMemoryObj()
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = object()
    engine._shared_local_cpu_backend = lambda: object()

    with pytest.raises(ValueError, match="not aligned"):
        engine._resolve_shared_rank0_layer_mem_objs(
            req_id="req-1",
            phase="sparse_decode_bootstrap",
            layer_id=0,
            kv_group=0,
            keys_layer=keys,
            local_prefix=LocalCPUPrefixGetResult(
                [local_obj],
                [0],
                [keys[0]],
            ),
        )

    assert local_obj.ref_count_down_count == 1


def test_rank0_handle_builder_rejects_partial_publication():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_generation = 3
    engine.metadata = SimpleNamespace(worker_id=0)
    backing = torch.arange(1024, dtype=torch.uint8)

    with pytest.raises(ValueError, match="partial layer handles"):
        engine._make_shared_handles_for_layer(
            req_id="req-1",
            phase="dense_prefix",
            keys_layer=[_make_key(), _make_key(kv_group=1)],
            mem_objs_layer=[_make_memory_obj(backing)],
            layer_id=0,
            kv_group=0,
        )


def test_rank0_handle_builder_validates_objects_before_publication():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_generation = 3
    engine.metadata = SimpleNamespace(worker_id=0)
    backing = torch.arange(1024, dtype=torch.uint8)

    def reject_publication(*_args, **_kwargs):
        raise ValueError("object is not shm-backed")

    engine._validate_rank0_shared_mem_obj = reject_publication

    with pytest.raises(ValueError, match="not shm-backed"):
        engine._make_shared_handles_for_layer(
            req_id="req-1",
            phase="dense_prefix",
            keys_layer=[_make_key()],
            mem_objs_layer=[_make_memory_obj(backing)],
            layer_id=0,
            kv_group=0,
        )

    handles = engine._make_shared_handles_for_layer(
        req_id="req-1",
        phase="dense_prefix",
        keys_layer=[_make_key()],
        mem_objs_layer=[_make_memory_obj(backing)],
        layer_id=0,
        kv_group=0,
        validate_memory_objs=False,
    )
    assert len(handles) == 1


def test_shared_chunk_handle_preserves_key_and_cached_positions():
    backing = torch.arange(1024, dtype=torch.uint8)
    key = _make_key()
    memory_obj = _make_memory_obj(backing)

    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=key,
        layer_id=2,
        kv_group=0,
        chunk_index=3,
        shm_name="/lmcache-test",
        memory_obj=memory_obj,
        generation=7,
        producer_rank=0,
    )

    encoded = handle.to_dict()
    assert set(encoded) == {
        "request_id",
        "phase",
        "key",
        "layer_id",
        "kv_group",
        "chunk_index",
        "shm_name",
        "offset",
        "physical_size",
        "logical_size",
        "shape",
        "dtype",
        "shapes",
        "dtypes",
        "fmt",
        "cached_positions",
        "generation",
        "producer_rank",
        "status",
    }
    assert encoded["key"] == key
    assert encoded["cached_positions"] == [0, 1, 2, 3]
    forbidden_fragments = (
        "ptr",
        "pointer",
        "data_ptr",
        "host",
        "device",
        "allocator",
        "parent",
        "object",
    )
    assert not any(
        fragment in field
        for field in encoded
        for fragment in forbidden_fragments
    )

    decoded = SharedChunkHandle.from_dict(encoded)
    assert decoded.key == key
    assert decoded.cached_positions == [0, 1, 2, 3]
    assert decoded.offset == 128
    assert decoded.logical_size == 16
    assert decoded.physical_size == 64


def test_shared_chunk_handle_uses_refreshed_partial_page_logical_size():
    full_shape = torch.Size([32, 8])
    partial_shape = torch.Size([19, 8])
    dtype = torch.bfloat16
    fmt = MemoryFormat.KV_T2D
    full_bytes = full_shape.numel() * dtype.itemsize
    partial_bytes = partial_shape.numel() * dtype.itemsize
    tensor_buffer = torch.zeros(full_bytes * 2, dtype=torch.uint8, device="cpu")
    allocator = PagedTensorMemoryAllocator(tensor_buffer, [full_shape], [dtype], fmt)

    full = allocator.allocate(full_shape, dtype, fmt)
    assert full is not None
    allocator.free(full)

    partial = allocator.allocate(partial_shape, dtype, fmt)
    assert partial is not None

    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=partial,
        generation=7,
        producer_rank=0,
    )

    assert handle.shape == partial_shape
    assert handle.logical_size == partial_bytes
    assert handle.logical_size == handle.shape.numel() * handle.dtype.itemsize

    allocator.free(partial)
    allocator.close()


def test_shared_chunk_handle_uses_refreshed_remote_partial_chunk_size():
    full_shape = torch.Size([2, 1, 256, 9, 8])
    partial_tokens = 147
    dtype = torch.bfloat16
    full_bytes = full_shape.numel() * dtype.itemsize
    single_token_size = full_bytes // full_shape[2]
    partial_bytes = partial_tokens * single_token_size
    tensor_buffer = torch.zeros(full_bytes * 2, dtype=torch.uint8, device="cpu")
    allocator = PagedTensorMemoryAllocator(tensor_buffer, [full_shape], [dtype])

    full = allocator.allocate(full_shape, dtype, MemoryFormat.KV_MLA_FMT)
    assert full is not None
    allocator.free(full)

    memory_obj = allocator.allocate(full_shape, dtype, MemoryFormat.KV_MLA_FMT)
    assert memory_obj is not None

    connector = object.__new__(_FakeRemoteConnector)
    connector.full_chunk_size_bytes = full_bytes
    connector.single_token_size = single_token_size
    memory_obj = connector.reshape_partial_chunk(memory_obj, partial_bytes)

    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=memory_obj,
        generation=7,
        producer_rank=0,
    )

    assert handle.shape[2] == partial_tokens
    assert handle.logical_size == partial_bytes
    assert handle.logical_size == handle.shape.numel() * handle.dtype.itemsize

    allocator.free(memory_obj)
    allocator.close()


def test_shared_chunk_handle_rejects_missing_required_field():
    backing = torch.arange(1024, dtype=torch.uint8)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=2,
        kv_group=0,
        chunk_index=3,
        shm_name="/lmcache-test",
        memory_obj=_make_memory_obj(backing),
        generation=7,
        producer_rank=0,
    )
    encoded = handle.to_dict()
    encoded.pop("cached_positions")

    with pytest.raises(SharedCPUCacheValidationError, match="cached_positions"):
        SharedChunkHandle.from_dict(encoded)


def test_shared_chunk_handle_rejects_pointer_private_fields():
    backing = torch.arange(1024, dtype=torch.uint8)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=2,
        kv_group=0,
        chunk_index=3,
        shm_name="/lmcache-test",
        memory_obj=_make_memory_obj(backing),
        generation=7,
        producer_rank=0,
    )
    encoded = handle.to_dict()
    encoded["host_ptr"] = 123456

    with pytest.raises(SharedCPUCacheValidationError, match="forbidden"):
        SharedChunkHandle.from_dict(encoded)


def test_shared_chunk_handle_reports_bad_payload_type_and_dtype():
    with pytest.raises(SharedCPUCacheValidationError, match="expected dict"):
        SharedChunkHandle.from_dict("not-a-dict")  # type: ignore[arg-type]

    backing = torch.arange(1024, dtype=torch.uint8)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=2,
        kv_group=0,
        chunk_index=3,
        shm_name="/lmcache-test",
        memory_obj=_make_memory_obj(backing),
        generation=7,
        producer_rank=0,
    )
    encoded = handle.to_dict()
    encoded["dtype"] = "torch.not_a_dtype"

    with pytest.raises(SharedCPUCacheValidationError, match="Unknown.*dtype"):
        SharedChunkHandle.from_dict(encoded)


def test_passive_allocator_creates_view_and_free_only_invalidates():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=11,
        producer_rank=0,
    )

    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=11,
    )
    view = allocator.create_view(
        handle,
        expected_request_id="req-1",
        expected_phase="sparse_decode_bootstrap",
        expected_layer_id=0,
        expected_kv_group=0,
        expected_chunk_index=0,
    )

    assert view.parent() is allocator
    assert view.metadata.address == handle.offset
    assert view.metadata.phy_size == handle.physical_size
    assert view.metadata.cached_positions.tolist() == [0, 1, 2, 3]
    assert torch.equal(view.raw_tensor, slab[128:144])

    view.ref_count_down()
    assert not view.is_valid()
    allocator.free(view)
    assert not view.is_valid()


def test_passive_allocator_rejects_bounds_generation_and_order_mismatch():
    slab = torch.arange(256, dtype=torch.uint8)
    source_obj = _make_memory_obj(
        slab,
        offset=128,
        logical_size=16,
        physical_size=256,
    )
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=1,
        kv_group=0,
        chunk_index=4,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=3,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="generation=2"):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=1,
            expected_kv_group=0,
            expected_chunk_index=4,
        )

    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )
    with pytest.raises(SharedCPUCacheValidationError, match="bounds"):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=1,
            expected_kv_group=0,
            expected_chunk_index=5,
        )


def test_passive_allocator_rejects_inconsistent_shape_size():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    bad_handle = replace(handle, shape=torch.Size([4]), shapes=[torch.Size([4])])
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="shape/dtype bytes"):
        allocator.create_view(
            bad_handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
        )


def test_passive_allocator_rejects_key_mismatch():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    key = _make_key()
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=key,
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="expected="):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_key=_make_key(kv_group=1),
        )


def test_passive_allocator_accepts_rank0_key_for_passive_rank():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key().get_first_layer(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )
    expected_key = CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=1,
        chunk_hash=1234,
        dtype=torch.float16,
        kv_group=0,
    ).get_first_layer()

    view = allocator.create_view(
        handle,
        expected_request_id="req-1",
        expected_phase="dense_prefix",
        expected_layer_id=0,
        expected_kv_group=0,
        expected_chunk_index=0,
        expected_key=expected_key,
        expected_producer_rank=0,
    )

    assert view.parent() is allocator
    view.ref_count_down()


def test_passive_allocator_rejects_producer_rank_mismatch():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=3,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="producer_rank=3"):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_producer_rank=0,
        )


def test_passive_allocator_rejects_cached_positions_mismatch():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="cached_positions"):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_cached_positions=[4, 5, 6, 7],
        )


def test_passive_allocator_allows_missing_cached_positions_when_expected():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    view = allocator.create_view(
        replace(handle, cached_positions=None),
        expected_request_id="req-1",
        expected_phase="dense_prefix",
        expected_layer_id=0,
        expected_kv_group=0,
        expected_chunk_index=0,
        expected_cached_positions=[0, 1, 2, 3],
    )

    assert view.metadata.cached_positions is None
    view.ref_count_down()


def test_passive_allocator_rejects_expected_metadata_mismatch():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="shape="):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_shape=torch.Size([4]),
        )
    with pytest.raises(SharedCPUCacheValidationError, match="dtype="):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_dtype=torch.float32,
        )
    with pytest.raises(SharedCPUCacheValidationError, match="fmt="):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        )


def test_passive_allocator_never_allocates():
    allocator = PassiveSharedViewAllocator(
        slab_tensor=torch.empty(16, dtype=torch.uint8),
        shm_name="/lmcache-test",
        generation=1,
    )
    with pytest.raises(SharedCPUCacheError):
        allocator.allocate(torch.Size([1]), torch.uint8)
    with pytest.raises(SharedCPUCacheError):
        allocator.batched_allocate(torch.Size([1]), torch.uint8, 2)


def test_shared_slab_owner_close_unlinks_without_detach(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "lmcache.c_ops",
        SimpleNamespace(
            unlink_shm=lambda shm_name: calls.append(("unlink", shm_name)),
            detach_shm_pinned_ptr=lambda ptr, size: calls.append(
                ("detach", ptr, size)
            ),
        ),
    )
    mapping = SharedSlabMapping(
        shm_name="/lmcache-owner-close",
        size=16,
        ptr=1234,
        tensor=torch.empty(16, dtype=torch.uint8),
        generation=7,
        owner=True,
    )

    mapping.close()
    mapping.close()

    assert calls == [("unlink", "/lmcache-owner-close")]


def test_shared_slab_preflight_reports_null_device_ptr(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "lmcache.c_ops",
        SimpleNamespace(get_device_ptr=lambda ptr: None),
    )
    mapping = SharedSlabMapping(
        shm_name="/lmcache-preflight",
        size=16,
        ptr=1234,
        tensor=torch.empty(16, dtype=torch.uint8),
        generation=7,
        owner=False,
    )

    with pytest.raises(SharedCPUCacheError, match="get_device_ptr returned None"):
        mapping.preflight_device_ptr()


def test_shared_slab_attach_falls_back_to_non_cuda_equivalents(monkeypatch):
    import lmcache

    fallback_ops = SimpleNamespace(
        attach_shm_pinned_ptr=lambda size, name, writable: 4321,
    )
    monkeypatch.delitem(sys.modules, "lmcache.c_ops", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "lmcache.non_cuda_equivalents",
        fallback_ops,
    )
    monkeypatch.setattr(
        lmcache,
        "non_cuda_equivalents",
        fallback_ops,
        raising=False,
    )
    monkeypatch.setattr(
        SharedSlabMapping,
        "_tensor_from_ptr",
        staticmethod(
            lambda _ptr, size: (torch.empty(size, dtype=torch.uint8), object())
        ),
    )

    mapping = SharedSlabMapping.attach(
        shm_name="/lmcache-no-cuda",
        size=16,
        generation=3,
        writable=False,
    )

    assert mapping.ptr == 4321
    assert mapping.owner is False


def test_shared_slab_attach_reports_null_host_ptr(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "lmcache.c_ops",
        SimpleNamespace(attach_shm_pinned_ptr=lambda size, name, writable: 0),
    )

    with pytest.raises(SharedCPUCacheError, match="returned 0"):
        SharedSlabMapping.attach(
            shm_name="/lmcache-attach-null",
            size=16,
            generation=3,
            writable=False,
        )


def test_shared_slab_attach_detaches_if_tensor_view_creation_fails(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "lmcache.c_ops",
        SimpleNamespace(
            attach_shm_pinned_ptr=lambda size, name, writable: 1234,
            detach_shm_pinned_ptr=lambda ptr, size: calls.append((ptr, size)),
        ),
    )

    def fail_tensor_from_ptr(_ptr, _size):
        raise RuntimeError("tensor view failed")

    monkeypatch.setattr(
        SharedSlabMapping,
        "_tensor_from_ptr",
        staticmethod(fail_tensor_from_ptr),
    )

    with pytest.raises(RuntimeError, match="tensor view failed"):
        SharedSlabMapping.attach(
            shm_name="/lmcache-attach-cleanup",
            size=16,
            generation=3,
            writable=False,
        )

    assert calls == [(1234, 16)]


def test_rank0_slab_rejects_empty_allocator_buffer():
    with pytest.raises(SharedCPUCacheError, match="invalid buffer size"):
        SharedSlabMapping.from_rank0_allocator(
            shm_name="/lmcache-rank0-empty",
            allocator_tensor=torch.empty(0, dtype=torch.uint8),
            generation=9,
        )


def test_rank0_startup_preflight_broadcasts_error_before_raising():
    engine = object.__new__(LMCacheEngine)
    broadcasts = []
    engine.enable_shared_cpu_cache = True
    engine.storage_manager = None
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_slab_size = None
    engine.shared_cpu_cache_generation = 0
    engine.metadata = SimpleNamespace(
        world_size=2,
        first_rank=0,
        worker_id=0,
        is_first_rank=lambda: True,
    )
    engine.broadcast_object_fn = lambda obj, rank: broadcasts.append((obj, rank))

    with pytest.raises(ValueError, match="requires StorageManager"):
        engine._post_init_shared_cpu_cache()

    assert broadcasts
    envelope, rank = broadcasts[-1]
    assert rank == 0
    assert envelope["status"] == "error"
    assert "requires StorageManager" in envelope["message"]
    assert envelope["shm_name"] == "/lmcache-test"


def test_rank0_startup_preflight_failure_closes_mapping_before_error_broadcast(
    monkeypatch,
):
    engine = object.__new__(LMCacheEngine)
    broadcasts = []
    closed = []

    class FakeMapping:
        def preflight_device_ptr(self):
            raise SharedCPUCacheError("preflight boom")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        "lmcache.v1.cache_engine.SharedSlabMapping.from_rank0_allocator",
        lambda **_kwargs: FakeMapping(),
    )
    engine.enable_shared_cpu_cache = True
    engine.shared_cpu_cache_strict = True
    engine.shared_cpu_cache_mapping = None
    engine.storage_manager = SimpleNamespace(
        local_cpu_backend=SimpleNamespace(
            memory_allocator=SimpleNamespace(
                buffer=torch.empty(16, dtype=torch.uint8),
                shm_name="/lmcache-startup-cleanup",
            )
        )
    )
    engine.shared_cpu_cache_name = None
    engine.shared_cpu_cache_slab_size = None
    engine.shared_cpu_cache_generation = 0
    engine.metadata = SimpleNamespace(
        world_size=2,
        first_rank=0,
        worker_id=0,
        is_first_rank=lambda: True,
    )
    engine.broadcast_object_fn = lambda obj, rank: broadcasts.append((obj, rank))

    with pytest.raises(SharedCPUCacheError, match="preflight boom"):
        engine._post_init_shared_cpu_cache()

    assert closed == [True]
    assert engine.shared_cpu_cache_mapping is None
    envelope, rank = broadcasts[-1]
    assert rank == 0
    assert envelope["status"] == "error"
    assert "preflight boom" in envelope["message"]


def test_receive_shared_envelope_reports_corrupt_payload():
    engine = object.__new__(LMCacheEngine)
    engine.metadata = SimpleNamespace(first_rank=0)
    engine.broadcast_object_fn = lambda obj, rank: {"status": "ok"}

    with pytest.raises(ValueError, match="corrupt envelope"):
        engine._receive_shared_envelope()


def test_skipped_index_envelope_round_trips_without_handles():
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=5,
        kv_group=1,
        status="skipped",
        generation=9,
        handles=[],
        message="index already resident by non-strict debug path",
    )

    encoded = envelope.to_dict()
    assert set(encoded) == {
        "request_id",
        "phase",
        "request_ordinal",
        "layer_id",
        "kv_group",
        "status",
        "generation",
        "handles",
        "message",
        "error_details",
    }
    decoded = SharedHandleEnvelope.from_dict(encoded)
    assert decoded.status == "skipped"
    assert decoded.kv_group == 1
    assert decoded.handles == []
    assert decoded.message is not None


def test_compact_shared_handle_batch_round_trip_and_view():
    batch = SharedHandleBatch(
        shm_name="/lmcache-test",
        producer_rank=0,
        num_layers=2,
        num_chunks=2,
        physical_sizes=[64, 32],
        chunk_hashes=[11, 22],
        offsets=[0, 64, 128, 192],
    )
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=9,
        handles=[],
        batch=batch,
    )
    decoded = SharedHandleEnvelope.from_dict(envelope.to_dict())
    assert decoded.batch == batch

    allocator = PassiveSharedViewAllocator(
        slab_tensor=torch.arange(256, dtype=torch.uint8),
        shm_name="/lmcache-test",
        generation=9,
    )
    view = allocator.create_batch_view(
        batch,
        layer_id=1,
        chunk_index=0,
        shape=torch.Size([8]),
        dtype=torch.float16,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        cached_positions=range(4),
    )
    assert view.metadata.address == 128
    assert view.metadata.phy_size == 64
    assert view.tensor.shape == torch.Size([8])
    tail = allocator.create_batch_view(
        batch,
        layer_id=1,
        chunk_index=1,
        shape=torch.Size([4]),
        dtype=torch.float16,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        cached_positions=range(2),
    )
    assert tail.metadata.address == 192
    assert tail.metadata.phy_size == 32
    with pytest.raises(SharedCPUCacheValidationError, match="physical sizes"):
        validate_shared_handle_batch(
            replace(batch, physical_sizes=[]),
            expected_shm_name=batch.shm_name,
            expected_producer_rank=0,
            expected_num_layers=2,
            expected_num_chunks=2,
            expected_chunk_hashes=batch.chunk_hashes,
            slab_size=256,
        )


def test_shared_envelope_rejects_missing_required_field():
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=5,
        kv_group=1,
        status="skipped",
        generation=9,
        handles=[],
        message="index already resident by non-strict debug path",
    )
    encoded = envelope.to_dict()
    encoded.pop("error_details")

    with pytest.raises(SharedCPUCacheValidationError, match="error_details"):
        SharedHandleEnvelope.from_dict(encoded)


def test_shared_envelope_rejects_pointer_private_fields():
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=5,
        kv_group=1,
        status="skipped",
        generation=9,
        handles=[],
        message="index already resident by non-strict debug path",
    )
    encoded = envelope.to_dict()
    encoded["device_ptr"] = 123456

    with pytest.raises(SharedCPUCacheValidationError, match="forbidden"):
        SharedHandleEnvelope.from_dict(encoded)


def test_shared_envelope_reports_bad_payload_type_status_and_handles():
    with pytest.raises(SharedCPUCacheValidationError, match="expected dict"):
        SharedHandleEnvelope.from_dict("not-a-dict")  # type: ignore[arg-type]

    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=5,
        kv_group=1,
        status="skipped",
        generation=9,
        handles=[],
        message="index already resident by non-strict debug path",
    )
    encoded = envelope.to_dict()
    encoded["status"] = "surprise"
    with pytest.raises(SharedCPUCacheValidationError, match="unsupported status"):
        SharedHandleEnvelope.from_dict(encoded)

    encoded = envelope.to_dict()
    encoded["handles"] = {"not": "a list"}
    with pytest.raises(SharedCPUCacheValidationError, match="handles must be a list"):
        SharedHandleEnvelope.from_dict(encoded)


def test_dense_prefix_zero_hit_broadcasts_skipped_not_miss():
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = SimpleNamespace()
    engine.gpu_connector = SimpleNamespace()
    engine.num_layers = 2
    engine.shared_cpu_cache_generation = 9
    engine.metadata = SimpleNamespace(first_rank=0, worker_id=0)
    broadcasts = []
    engine.broadcast_object_fn = lambda obj, rank: broadcasts.append(obj)
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_finished=lambda monitor_req_id, tokens: broadcasts.append(
            {"stats": int(tokens)}
        )
    )
    ret_mask = torch.zeros(8, dtype=torch.bool)

    yielded = list(
        engine._retrieve_layer_shared_rank0(
            starts=[],
            ends=[],
            keys_layer_major=[],
            chunk_locations_layer_major=[],
            location=None,
            ret_mask=ret_mask,
            monitor_req_id=123,
            req_id="req-1",
            kv_group=0,
            kwargs={"shared_cpu_phase": "dense_prefix"},
        )
    )

    envelopes = [item for item in broadcasts if "status" in item]
    assert [item["status"] for item in envelopes] == ["skipped", "skipped"]
    assert all(item["handles"] == [] for item in envelopes)
    assert torch.equal(yielded[-1], ret_mask)
    assert broadcasts[-1] == {"stats": 0}


@pytest.mark.parametrize(
    ("locations", "location", "page_first"),
    [
        ([["RemoteBackend"] * 2] * 2, "RemoteBackend", True),
        ([["LocalCPUBackend", "RemoteBackend"]] * 2, "mixed", True),
        ([["RemoteBackend", "LocalCPUBackend"]] * 2, "mixed", False),
    ],
)
def test_shared_dense_page_first_selects_safe_resolver(
    monkeypatch, locations, location, page_first
):
    import lmcache.v1.cache_engine as cache_engine_module

    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(
        extra_config={"mooncake_page_first_multi_buffer": True}
    )
    engine.storage_manager = SimpleNamespace()
    engine.gpu_connector = _FakeLayerwiseGPUConnector()
    engine.num_layers = 2
    engine.shared_cpu_cache_generation = 9
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_finished=lambda monitor_req_id, tokens: None
    )
    chunk_keys = [
        replace(_make_key(), chunk_hash=0x700 + chunk).split_layers(2)
        for chunk in range(2)
    ]
    keys_by_layer = [list(row) for row in zip(*chunk_keys, strict=True)]
    mem_objs = [
        [_FakeResolvableMemoryObj() for _ in range(2)] for _ in range(engine.num_layers)
    ]
    for mem_obj in (obj for layer in mem_objs for obj in layer):
        mem_obj.pin()
    page_calls = []

    def resolve_page_first(**kwargs):
        page_calls.append(kwargs)
        return mem_objs

    layer_calls = []

    def resolve_layer(**kwargs):
        layer_calls.append(kwargs)
        return mem_objs[kwargs["layer_id"]]

    engine._resolve_shared_rank0_page_first_layers = resolve_page_first
    engine._resolve_shared_rank0_layer_mem_objs = resolve_layer
    engine._make_shared_handles_for_layer = lambda **kwargs: [
        object() for _ in kwargs["mem_objs_layer"]
    ]
    engine._broadcast_shared_envelope = lambda _envelope: None

    list(
        engine._retrieve_layer_shared_rank0(
            starts=[0, 4],
            ends=[4, 8],
            keys_layer_major=keys_by_layer,
            chunk_locations_layer_major=locations,
            location=location,
            ret_mask=torch.ones(8, dtype=torch.bool),
            monitor_req_id=123,
            req_id="req-page-first",
            kv_group=0,
            kwargs={"shared_cpu_phase": "dense_prefix"},
        )
    )

    assert len(page_calls) == int(page_first)
    assert len(layer_calls) == (0 if page_first else engine.num_layers)
    if page_first:
        assert page_calls[0]["keys_layer_major"] == keys_by_layer
    assert engine.gpu_connector.sent == mem_objs
    assert all(
        obj.ref_count_down_count == 1 and not obj.is_pinned
        for layer in mem_objs
        for obj in layer
    )


def test_shared_dense_page_first_publishes_one_compact_batch(monkeypatch):
    import lmcache.v1.cache_engine as cache_engine_module

    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(
        extra_config={"mooncake_page_first_multi_buffer": True}
    )
    engine.storage_manager = SimpleNamespace()
    engine.gpu_connector = _FakeLayerwiseGPUConnector()
    engine.num_layers = 2
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_generation = 9
    engine.metadata = SimpleNamespace(first_rank=0, worker_id=0)
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_finished=lambda monitor_req_id, tokens: None
    )
    backing = torch.empty(192, dtype=torch.uint8)
    mem_objs = [
        [
            _make_memory_obj(backing, offset=0),
            _make_memory_obj(backing, offset=64, physical_size=32),
        ],
        [
            _make_memory_obj(backing, offset=96),
            _make_memory_obj(backing, offset=160, physical_size=32),
        ],
    ]
    for obj in (obj for layer in mem_objs for obj in layer):
        obj.pin()
    engine._resolve_shared_rank0_page_first_layers = lambda **_kwargs: mem_objs
    engine._make_shared_handles_for_layer = lambda **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("compact page-first path must not build handles"))
    broadcasts = []
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)
    chunk_keys = [
        replace(_make_key(), chunk_hash=0x900 + chunk).split_layers(2)
        for chunk in range(2)
    ]
    keys_by_layer = [list(row) for row in zip(*chunk_keys, strict=True)]

    list(
        engine._retrieve_layer_shared_rank0(
            starts=[0, 4],
            ends=[4, 8],
            keys_layer_major=keys_by_layer,
            chunk_locations_layer_major=[
                ["RemoteBackend", "RemoteBackend"]
                for _ in range(engine.num_layers)
            ],
            location="RemoteBackend",
            ret_mask=torch.ones(8, dtype=torch.bool),
            monitor_req_id=123,
            req_id="req-page-first",
            kv_group=0,
            kwargs={"shared_cpu_phase": "dense_prefix"},
        )
    )

    assert len(broadcasts) == 1
    assert broadcasts[0].batch is not None
    assert broadcasts[0].batch.physical_sizes == [64, 32]
    assert broadcasts[0].batch.offsets == [0, 64, 96, 160]
    assert engine.gpu_connector.sent == mem_objs


@pytest.mark.parametrize("kv_group", [0, 1])
@pytest.mark.parametrize("page_first", [False, True])
def test_shared_dense_rank0_retriever_releases_before_result_tail(
    monkeypatch, kv_group, page_first
):
    import lmcache.v1.cache_engine as cache_engine_module

    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(
        extra_config={"mooncake_page_first_multi_buffer": page_first}
    )
    engine.storage_manager = SimpleNamespace()
    engine.gpu_connector = _FakeLayerwiseGPUConnector()
    engine.num_layers = 2
    engine.shared_cpu_cache_generation = 9
    engine.metadata = SimpleNamespace(first_rank=0, worker_id=0)
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_finished=lambda monitor_req_id, tokens: None
    )
    mem_objs = [_FakeResolvableMemoryObj(), _FakeResolvableMemoryObj()]
    resolver_calls = {"layer": 0, "page_first": 0}

    def resolve_layer(**kwargs):
        resolver_calls["layer"] += 1
        mem_obj = mem_objs[kwargs["layer_id"]]
        # Match _resolve_shared_rank0_layer_mem_objs(), which returns an
        # ownership pin that the retriever releases after publication.
        mem_obj.pin()
        return [mem_obj]

    engine._resolve_shared_rank0_layer_mem_objs = resolve_layer

    def resolve_page_first(**kwargs):
        resolver_calls["page_first"] += 1
        for mem_obj in mem_objs:
            mem_obj.pin()
        return [[mem_obj] for mem_obj in mem_objs]

    engine._resolve_shared_rank0_page_first_layers = resolve_page_first
    engine._make_shared_handles_for_layer = lambda **kwargs: [object()]
    broadcasts = []
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)
    ret_mask = torch.ones(4, dtype=torch.bool)
    keys_by_layer = [[_make_key()], [_make_key()]]
    chunk_location = "RemoteBackend" if page_first else "LocalCPUBackend"

    retriever = engine._retrieve_layer_shared_rank0(
        starts=[0],
        ends=[4],
        keys_layer_major=keys_by_layer,
        chunk_locations_layer_major=[[chunk_location], [chunk_location]],
        location=chunk_location,
        ret_mask=ret_mask,
        monitor_req_id=123,
        req_id="req-1",
        kv_group=kv_group,
        kwargs={"shared_cpu_phase": "dense_prefix"},
    )

    yielded = [next(retriever) for _ in range(engine.num_layers + 1)]

    assert resolver_calls == (
        {"layer": 0, "page_first": 1}
        if page_first
        else {"layer": 2, "page_first": 0}
    )
    assert yielded[0].item() == 4
    assert yielded[1] is None
    assert yielded[2] is None
    assert [item.layer_id for item in broadcasts] == [0, 1]
    assert engine.gpu_connector.sent == [[mem_objs[0]], [mem_objs[1]]]
    assert [mem.ref_count_down_count for mem in mem_objs] == [0, 0]
    assert all(mem.is_pinned for mem in mem_objs)
    assert engine.gpu_connector.close_count == 1

    assert torch.equal(next(retriever), ret_mask)
    assert [mem.ref_count_down_count for mem in mem_objs] == [1, 1]
    assert all(not mem.is_pinned for mem in mem_objs)
    with pytest.raises(StopIteration):
        next(retriever)
    assert [mem.ref_count_down_count for mem in mem_objs] == [1, 1]


@pytest.mark.parametrize("kv_group", [0, 1])
def test_shared_dense_passive_retriever_releases_before_result_tail(
    monkeypatch, kv_group
):
    import lmcache.v1.cache_engine as cache_engine_module

    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    engine = _make_passive_shared_retrieve_engine(kv_group=kv_group)
    retriever, ret_mask = _make_passive_shared_retriever(
        engine,
        kv_group=kv_group,
    )

    yielded = [next(retriever) for _ in range(engine.num_layers + 1)]

    assert yielded[0].item() == 4
    assert yielded[1] is None
    assert yielded[2] is None
    assert engine.gpu_connector.sent == [
        [engine.shared_cpu_cache_passive_allocator.views[0]],
        [engine.shared_cpu_cache_passive_allocator.views[1]],
    ]
    assert [
        view.ref_count_down_count
        for view in engine.shared_cpu_cache_passive_allocator.views
    ] == [0, 0]
    assert engine.gpu_connector.close_count == 1

    assert torch.equal(next(retriever), ret_mask)
    assert [
        view.ref_count_down_count
        for view in engine.shared_cpu_cache_passive_allocator.views
    ] == [1, 1]
    with pytest.raises(StopIteration):
        next(retriever)


def test_shared_dense_passive_compact_batch_preserves_layerwise_consumption(
    monkeypatch,
):
    import lmcache.v1.cache_engine as cache_engine_module

    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    engine = _make_passive_shared_retrieve_engine(kv_group=0)
    receive_count = 0
    batch = SharedHandleBatch(
        shm_name="/lmcache-test",
        producer_rank=0,
        num_layers=2,
        num_chunks=1,
        physical_sizes=[64],
        chunk_hashes=[_make_key().chunk_hash],
        offsets=[0, 64],
    )

    def receive():
        nonlocal receive_count
        receive_count += 1
        return SharedHandleEnvelope(
            request_id="req-1",
            phase="dense_prefix",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
            status="ok",
            generation=9,
            handles=[],
            batch=batch,
        )

    engine._receive_shared_envelope = receive
    retriever, ret_mask = _make_passive_shared_retriever(engine)
    yielded = list(retriever)

    assert receive_count == 1
    assert engine.gpu_connector.sent == [
        [engine.shared_cpu_cache_passive_allocator.views[0]],
        [engine.shared_cpu_cache_passive_allocator.views[1]],
    ]
    assert torch.equal(yielded[-1], ret_mask)


def test_shared_dense_passive_views_remain_request_owned(monkeypatch):
    import lmcache.v1.cache_engine as cache_engine_module

    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    engine = _make_passive_shared_retrieve_engine(
        kv_group=0,
        requests=(("req-a", 0), ("req-b", 1)),
    )
    first, first_mask = _make_passive_shared_retriever(
        engine, req_id="req-a", request_ordinal=0
    )
    second, _ = _make_passive_shared_retriever(
        engine, req_id="req-b", request_ordinal=1
    )
    next(first)
    next(first)
    next(second)
    assert next(first) is None

    views = engine.shared_cpu_cache_passive_allocator.views
    assert [view.ref_count_down_count for view in views] == [0, 0, 0]

    assert torch.equal(next(first), first_mask)
    assert [view.ref_count_down_count for view in views] == [1, 1, 0]

    second.close()
    assert [view.ref_count_down_count for view in views] == [1, 1, 1]
    assert engine.gpu_connector.close_count == 2


def test_strict_shared_envelope_rejects_miss_before_view_creation():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_generation = 9
    engine.shared_cpu_cache_strict = True
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="miss",
        generation=9,
        handles=[],
        message="missing required dense prefix chunk",
    )

    with pytest.raises(ValueError, match="strict mode received miss envelope"):
        engine._validate_shared_layerwise_envelope(
            envelope,
            req_id="req-1",
            phase="dense_prefix",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
        )


def test_shared_envelope_rejects_request_ordinal_mismatch():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_generation = 9
    engine.shared_cpu_cache_strict = True
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=2,
        layer_id=0,
        kv_group=0,
        status="skipped",
        generation=9,
        handles=[],
    )

    with pytest.raises(ValueError, match="request_ordinal=2"):
        engine._validate_shared_layerwise_envelope(
            envelope,
            req_id="req-1",
            phase="dense_prefix",
            request_ordinal=1,
            layer_id=0,
            kv_group=0,
        )


def test_shared_envelope_rejects_status_handle_mismatch():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_generation = 9
    engine.shared_cpu_cache_strict = True
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=9,
        handles=[],
    )

    with pytest.raises(ValueError, match="ok envelope"):
        engine._validate_shared_layerwise_envelope(
            envelope,
            req_id="req-1",
            phase="dense_prefix",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
        )

    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=_make_memory_obj(torch.arange(1024, dtype=torch.uint8)),
        generation=9,
        producer_rank=0,
    )
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="skipped",
        generation=9,
        handles=[handle],
    )

    with pytest.raises(ValueError, match="must not carry handles"):
        engine._validate_shared_layerwise_envelope(
            envelope,
            req_id="req-1",
            phase="dense_prefix",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
        )
