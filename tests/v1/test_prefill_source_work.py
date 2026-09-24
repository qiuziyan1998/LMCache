# SPDX-License-Identifier: Apache-2.0
"""Public prefill retrieval performs source work only for the new suffix."""

# Standard
from collections import deque
from dataclasses import replace
from threading import Lock
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1 import cache_engine as engine_module
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import LayerPageSource, MemoryFormat
from lmcache.v1.prefill_metadata import PrefillMetadataCache
from lmcache.v1.prefill_sources import SharedPrefillSources
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator, SharedHandleBatch
from lmcache.v1.token_database import ChunkedTokenDatabase


class HostConsumer:
    def __init__(self) -> None:
        self.appended: list[int] = []

    def supports_dense_sparse_cache_retention(self) -> bool:
        return True

    def prepare_layerwise_prefill_source_pointers(
        self, sources: list, host_rows: list, device_rows: list, **kwargs: Any
    ) -> None:
        if not host_rows:
            host_rows.extend([] for _ in sources)
        source = sources[0]
        self.appended.append(
            len(source.pages) + len(source.suffix)
            if isinstance(source, LayerPageSource)
            else len(source)
        )
        for layer, source in enumerate(sources):
            if isinstance(source, LayerPageSource):
                host_rows[layer].extend(
                    obj.layer_data_ptr(layer) for obj in source.pages
                )
                host_rows[layer].extend(obj.data_ptr for obj in source.suffix)
            else:
                host_rows[layer].extend(obj.data_ptr for obj in source)
        device_rows[:] = [None] * len(sources)

    def batched_to_gpu(self, *args: Any, **kwargs: Any):
        while True:
            yield None

    def synchronize_dense_load_stream(self) -> None:
        pytest.fail("Successful retained loads must not synchronize")


class PageBackend:
    def __init__(self) -> None:
        self.hot_cache: dict = {}
        self.cpu_lock = Lock()
        self.get_counts: list[int] = []
        self.contains_count = 0

    def contains(self, key):
        self.contains_count += 1
        return key in self.hot_cache

    def get_blocking(self, key):
        self.get_counts.append(1)
        obj = self.hot_cache.get(key)
        if obj is not None:
            obj.ref_count_up()
        return obj

    def batched_get_layer_page_prefix(self, keys: list):
        self.get_counts.append(len(keys))
        pages = [self.hot_cache[key] for key in keys]
        for page in pages:
            page.ref_count_up()
        return pages, len(pages)


class SharedEngine(LMCacheEngine):
    def __init__(self, *, passive: bool, layers: int = 3) -> None:
        self.config = SimpleNamespace(chunk_size=4, save_unfull_chunk=True)
        self.num_layers = layers
        self.metadata = SimpleNamespace(
            first_rank=0, worker_id=int(passive), is_first_rank=lambda: not passive
        )
        self.gpu_connector = HostConsumer()
        self.shared_cpu_cache_generation = 7
        self.shared_cpu_cache_name = "/source-work"
        self.allocator = PassiveSharedViewAllocator(
            slab_tensor=torch.zeros(16384, dtype=torch.uint8),
            shm_name=self.shared_cpu_cache_name,
            generation=7,
        )
        self.shared_cpu_cache_passive_allocator = self.allocator if passive else None
        self.backend = PageBackend()
        self.backend.memory_allocator = SimpleNamespace(
            shm_name=self.shared_cpu_cache_name,
            pin_allocator=self.allocator,
            buffer=self.allocator.buffer,
        )
        self.storage_manager = SimpleNamespace(
            storage_backends={"LocalCPUBackend": self.backend}
        )
        self.stats_monitor = SimpleNamespace(
            on_retrieve_request=lambda _: 1, on_retrieve_finished=lambda *_: None
        )
        self._shared_cpu_request_leases = {}
        self.incoming = deque()
        self.outgoing = []
        self.validated: list[int] = []
        self.location_counts: list[int] = []
        self.token_database = object.__new__(ChunkedTokenDatabase)
        self.token_database.config = self.config
        self.token_database.chunk_size = 4
        self.token_database.hash_func = hash
        self.token_database.mooncake_payload_layout = "test-layout"
        self.token_database.save_only_first_rank = False
        self.token_database.metadata = SimpleNamespace(
            model_name="model",
            world_size=8,
            worker_id=0,
            get_dtypes=lambda: [torch.float16, torch.float16],
        )
        self.caches = {
            name: []
            for name in (
                "cached_keys",
                "cached_starts",
                "cached_ends",
                "cached_memory_objs",
                "cached_shared_handles",
                "cached_chunk_dev_ptrs",
                "cached_chunk_ptrs_npu",
            )
        }
        self.call_kwargs = dict(
            self.caches,
            req_id="r",
            _prefill_metadata_cache=PrefillMetadataCache(),
            _retain_shared_dense_cache=True,
            _reuse_shared_dense_prefix=True,
            deferred_layerwise_get=True,
            prefill_dma_block_ids_by_bank=((0,), (1,)),
        )

    def is_healthy(self) -> bool:
        return True

    def _is_passive(self) -> bool:
        return self.metadata.worker_id != 0

    def _should_use_shared_layerwise_retrieve(self, kv_group: int) -> bool:
        return True

    def _remote_fill_pair_lookup_enabled(self) -> bool:
        return False

    def _remote_fill_retrieve_plan(self, *args: Any):
        return None

    def _shared_local_cpu_backend(self):
        return self.backend

    def _shared_rank0_object_context(self, kv_group: int):
        return engine_module._SharedRank0ObjectContext(
            self.allocator,
            self.allocator.slab_size,
            torch.float16,
            MemoryFormat.KV_MLA_LATENT_FMT,
        )

    def _expected_shared_cpu_chunk_metadata(self, *, kv_group: int, num_tokens: int):
        return torch.Size([num_tokens]), torch.float16, MemoryFormat.KV_MLA_LATENT_FMT

    def _validate_rank0_shared_mem_obj(self, obj, **kwargs: Any) -> None:
        self.validated.append(id(obj))
        super()._validate_rank0_shared_mem_obj(obj, **kwargs)

    def _shared_page_first_location_plan(self, keys):
        self.location_counts.append(len(keys))
        return super()._shared_page_first_location_plan(keys)

    def _broadcast_shared_envelope(self, envelope) -> None:
        self.outgoing.append(envelope)

    def _receive_shared_envelope(self):
        return self.incoming.popleft()

    def populate(self, total: int, group: int, *, pages: bool = True) -> None:
        for start, end, key in self.token_database.process_tokens(
            tokens=list(range(total)),
            kv_group=group,
            request_configs=self.call_kwargs.get("request_configs"),
        ):
            if (key if pages else key.get_layer(0)) in self.backend.hot_cache:
                continue
            if not pages:
                for layer in range(self.num_layers):
                    batch = SharedHandleBatch(
                        shm_name=self.shared_cpu_cache_name,
                        producer_rank=0,
                        num_layers=self.num_layers,
                        num_chunks=1,
                        physical_sizes=[(end - start) * 2],
                        chunk_hashes=[key.chunk_hash],
                        offsets=[len(self.backend.hot_cache) * 128] * self.num_layers,
                    )
                    self.backend.hot_cache[key.get_layer(layer)] = (
                        self.allocator.create_batch_view(
                            batch,
                            layer_id=layer,
                            chunk_index=0,
                            shape=torch.Size([end - start]),
                            dtype=torch.float16,
                            fmt=MemoryFormat.KV_MLA_LATENT_FMT,
                            cached_positions=range(start, end),
                        )
                    )
                continue
            batch = SharedHandleBatch(
                shm_name=self.shared_cpu_cache_name,
                producer_rank=0,
                num_layers=self.num_layers,
                num_chunks=1,
                physical_sizes=[(end - start) * 2],
                chunk_hashes=[key.chunk_hash],
                offsets=[],
                page_offsets=[group * 8192 + len(self.backend.hot_cache) * 128],
                page_physical_sizes=[(end - start) * 2 * self.num_layers],
            )
            self.backend.hot_cache[key] = self.allocator.create_page_view(
                batch,
                chunk_index=0,
                shape=torch.Size([end - start]),
                dtype=torch.float16,
                fmt=MemoryFormat.KV_MLA_LATENT_FMT,
                cached_positions=range(start, end),
            )

    def retrieve(self, total: int, group: int = 0):
        return list(
            self.retrieve_layer(list(range(total)), kv_group=group, **self.call_kwargs)
        )


@pytest.fixture
def shared_engines(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        PinMonitor,
        "GetOrCreate",
        lambda: SimpleNamespace(
            on_pin_many=lambda _: None, on_pin=lambda _: None, on_unpin=lambda _: None
        ),
    )
    monkeypatch.setattr(engine_module, "assert_layerwise_gpu_connector", lambda _: None)
    monkeypatch.setattr(engine_module, "mooncake_page_layout_enabled", lambda _: True)
    monkeypatch.setattr(engine_module, "mooncake_layer_pages_enabled", lambda _: True)
    return SharedEngine(passive=False), SharedEngine(passive=True)


@pytest.mark.parametrize("group", [0, 1])
def test_public_retrieve_only_resolves_and_constructs_new_sources(
    shared_engines, monkeypatch: pytest.MonkeyPatch, group: int
) -> None:
    rank0, passive = shared_engines
    view_calls = []
    create = passive.allocator.create_page_view

    def count_view(*args, **kwargs):
        view_calls.append(kwargs["chunk_index"])
        return create(*args, **kwargs)

    monkeypatch.setattr(passive.allocator, "create_page_view", count_view)
    first = None
    for total in (8, 12, 16, 16):
        rank0.populate(total, group)
        assert rank0.retrieve(total, group)[-1].all()
        passive.incoming.extend(rank0.outgoing)
        rank0.outgoing.clear()
        assert passive.retrieve(total, group)[-1].all()
        if first is None:
            first = passive.caches["cached_memory_objs"][0][0]
        assert passive.caches["cached_memory_objs"][0][0] is first
    assert rank0.backend.get_counts == [2, 1, 1]
    assert rank0.location_counts == [2, 1, 1]
    assert len(rank0.validated) == 4
    assert view_calls == [0, 1, 2, 3]
    assert rank0.gpu_connector.appended == [2, 1, 1, 0]
    assert passive.gpu_connector.appended == [2, 1, 1, 0]
    pages = list(rank0.backend.hot_cache.values())
    assert all(page.get_ref_count() == 2 for page in pages)
    rank0.release_shared_cpu_sparse_request("r")
    passive.release_shared_cpu_sparse_request("r")
    assert all(
        page.get_ref_count() == 1 and page.metadata.pin_count == 0 for page in pages
    )
    assert not first.is_valid()
    for page in pages:
        page.ref_count_down()


def test_promoted_partial_tail_avoids_backend_borrow_and_preserves_old_source(
    shared_engines,
) -> None:
    rank0, passive = shared_engines
    rank0.populate(6, 0)
    rank0.retrieve(6)
    passive.incoming.extend(rank0.outgoing)
    rank0.outgoing.clear()
    passive.retrieve(6)
    old_tail = rank0.caches["cached_memory_objs"][0][1]
    passive_tail = passive.caches["cached_memory_objs"][0][1]
    lease = rank0.get_shared_cpu_request_lease("r")
    old_location_view = lease.prefill_locations[0].key_rows()[0]
    old_key = old_location_view[1]

    rank0.populate(8, 0)
    plan = rank0.call_kwargs["_prefill_metadata_cache"].prepare(
        rank0.token_database, list(range(8)), kv_group=0, num_layers=rank0.num_layers
    )
    replacement = rank0.backend.hot_cache[plan.base_keys[1]]
    rank0.caches["cached_starts"][1:] = plan.starts[1:]
    rank0.caches["cached_ends"][1:] = plan.ends[1:]
    for layer in range(rank0.num_layers):
        rank0.caches["cached_keys"][layer][1:] = plan.keys_layer_major[layer][1:]
        rank0.caches["cached_memory_objs"][layer][1:] = [replacement]
        rank0.caches["cached_chunk_dev_ptrs"][layer][1:] = [
            replacement.layer_data_ptr(layer)
        ]
    rank0.retain_shared_cpu_store_seed(
        "r", {0: rank0.caches["cached_memory_objs"]}, append_from={0: 1}
    )
    assert old_location_view[1] is old_key
    assert lease.prefill_locations[0].valid_chunks == 1
    assert old_tail.get_ref_count() == 2 and old_tail.metadata.pin_count == 1
    assert replacement.get_ref_count() == 2

    rank0.retrieve(8)
    passive.incoming.extend(rank0.outgoing)
    rank0.outgoing.clear()
    passive.retrieve(8)
    assert rank0.backend.get_counts == [2]
    assert rank0.location_counts == [2, 1]
    assert len(rank0.validated) == 3
    assert rank0.gpu_connector.appended == [2, 0]
    assert passive.gpu_connector.appended == [2, 1]
    assert old_tail.get_ref_count() == 2
    assert passive_tail.is_valid()
    assert passive.caches["cached_memory_objs"][0][1] is not passive_tail

    # Backend eviction must not invalidate an in-flight historical tail.
    old_tail.ref_count_down()
    assert old_tail.is_valid() and old_tail.get_ref_count() == 1
    rank0.release_shared_cpu_sparse_request("r")
    passive.release_shared_cpu_sparse_request("r")
    assert not old_tail.is_valid() and not passive_tail.is_valid()
    for page in rank0.backend.hot_cache.values():
        if page is not old_tail:
            assert page.get_ref_count() == 1 and page.metadata.pin_count == 0
            page.ref_count_down()


@pytest.mark.parametrize("change", ["keys", "generation", "release"])
def test_source_proof_invalidates_on_key_epoch_or_request_change(
    shared_engines, change: str
) -> None:
    rank0, passive = shared_engines
    rank0.populate(8, 0)
    rank0.retrieve(8)
    passive.incoming.extend(rank0.outgoing)
    rank0.outgoing.clear()
    passive.retrieve(8)
    first = passive.caches["cached_memory_objs"][0][0]
    if change == "keys":
        for engine in (rank0, passive):
            engine.call_kwargs["request_configs"] = {"lmcache.tag.schema": "new"}
        rank0.populate(8, 0)
    elif change == "generation":
        for engine in (rank0, passive):
            engine.shared_cpu_cache_generation += 1
        passive.shared_cpu_cache_passive_allocator = PassiveSharedViewAllocator(
            slab_tensor=torch.zeros(16384, dtype=torch.uint8),
            shm_name=passive.shared_cpu_cache_name,
            generation=8,
        )
    else:
        rank0.release_shared_cpu_sparse_request("r")
        passive.release_shared_cpu_sparse_request("r")
    rank0.retrieve(8)
    passive.incoming.extend(rank0.outgoing)
    rank0.outgoing.clear()
    passive.retrieve(8)
    assert rank0.backend.get_counts == [2, 2]
    assert passive.caches["cached_memory_objs"][0][0] is not first
    assert rank0.gpu_connector.appended == [2, 2]
    assert passive.gpu_connector.appended == [2, 2]
    old_lease = rank0.get_shared_cpu_request_lease("r")
    rank0.release_shared_cpu_sparse_request("r")
    passive.release_shared_cpu_sparse_request("r")
    assert not old_lease.prefill_sources and not old_lease.prefill_locations
    assert not first.is_valid()
    for page in rank0.backend.hot_cache.values():
        assert page.get_ref_count() == 1 and page.metadata.pin_count == 0
        page.ref_count_down()


def test_failed_suffix_retention_rolls_back_only_new_borrow(
    shared_engines, monkeypatch: pytest.MonkeyPatch
) -> None:
    rank0, _ = shared_engines
    rank0.populate(8, 0)
    rank0.retrieve(8)
    lease = rank0.get_shared_cpu_request_lease("r")
    original = [list(row) for row in lease.groups[0]]
    rank0.populate(12, 0)
    suffix = list(rank0.backend.hot_cache.values())[-1]
    monkeypatch.setattr(suffix, "pin", lambda: False)
    with pytest.raises(RuntimeError, match="pin"):
        rank0.retain_shared_cpu_store_seed(
            "r", {0: [row + [suffix] for row in original]}, append_from={0: 2}
        )
    assert lease.groups[0] == original
    assert suffix.get_ref_count() == 1 and suffix.metadata.pin_count == 0
    assert lease.prefill_sources[0].valid_chunks == 2
    rank0.release_shared_cpu_sparse_request("r")
    for page in rank0.backend.hot_cache.values():
        assert page.get_ref_count() == 1 and page.metadata.pin_count == 0
        page.ref_count_down()


def test_local_only_legacy_sources_remain_incremental_with_partial_tail(
    shared_engines, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_module, "mooncake_page_layout_enabled", lambda _: False)
    monkeypatch.setattr(engine_module, "mooncake_layer_pages_enabled", lambda _: False)
    rank0, passive = shared_engines
    calls = []
    create = passive.allocator.create_batch_view

    def count_view(*args, **kwargs):
        calls.append((kwargs["layer_id"], kwargs["chunk_index"]))
        return create(*args, **kwargs)

    monkeypatch.setattr(passive.allocator, "create_batch_view", count_view)
    for total in (6, 8, 10, 10):
        rank0.populate(total, 0, pages=False)
        rank0.retrieve(total)
        passive.incoming.extend(rank0.outgoing)
        rank0.outgoing.clear()
        passive.retrieve(total)
    expected = 4 * rank0.num_layers
    assert len(rank0.backend.get_counts) == expected
    assert rank0.backend.contains_count == expected
    assert len(rank0.validated) == expected
    assert len(calls) == expected
    assert rank0.gpu_connector.appended == [2, 1, 1, 0]
    assert passive.gpu_connector.appended == [2, 1, 1, 0]
    rank0.release_shared_cpu_sparse_request("r")
    passive.release_shared_cpu_sparse_request("r")
    for obj in rank0.backend.hot_cache.values():
        assert obj.get_ref_count() == 1 and obj.metadata.pin_count == 0
        obj.ref_count_down()


def test_compact_prefix_checks_earlier_tail_offsets_before_later_shape_change() -> None:
    old = SharedHandleBatch(
        shm_name="/slab",
        producer_rank=0,
        num_layers=2,
        num_chunks=3,
        physical_sizes=[8, 8, 4],
        chunk_hashes=[1, 2, 3],
        offsets=[64, 80, 96, 128, 144, 160],
    )
    state = SharedPrefillSources((), old, (0, 4, 8), (4, 8, 10), (), 3)
    changed = replace(
        old, offsets=[256, 80, 96, 128, 144, 160], physical_sizes=[8, 8, 8]
    )
    assert state.wire_prefix(changed, 3) == 0
