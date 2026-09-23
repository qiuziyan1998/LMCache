# SPDX-License-Identifier: Apache-2.0
"""Reference and metadata contracts for incremental shared prefill sources."""

# Standard
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryObj
from lmcache.v1.shared_cpu_cache import (
    PassiveSharedViewAllocator,
    SharedChunkHandle,
    SharedCPUCacheValidationError,
    SharedCPURequestLease,
    SharedHandleBatch,
)


def make_key(chunk_hash: int = 101) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.float16,
    )


def make_allocator() -> PassiveSharedViewAllocator:
    return PassiveSharedViewAllocator(
        slab_tensor=torch.zeros(4096, dtype=torch.uint8),
        shm_name="/prefill-test",
        generation=9,
    )


def make_batch(*, page: bool) -> SharedHandleBatch:
    return SharedHandleBatch(
        shm_name="/prefill-test",
        producer_rank=0,
        num_layers=2,
        num_chunks=2,
        physical_sizes=[64, 64],
        chunk_hashes=[101, 102],
        offsets=[] if page else [0, 64, 128, 192],
        page_offsets=[0, 128] if page else [],
        page_physical_sizes=[128, 128] if page else [],
    )


def create_compact(
    allocator: PassiveSharedViewAllocator,
    batch: SharedHandleBatch,
    *,
    page: bool,
    previous: TensorMemoryObj | None = None,
    **overrides: Any,
) -> TensorMemoryObj:
    values = dict(
        chunk_index=0,
        shape=torch.Size([1, 4, 8]),
        dtype=torch.float16,
        fmt=MemoryFormat.KV_T2D,
        cached_positions=range(4),
        previous=previous,
    )
    values.update(overrides)
    if page:
        return allocator.create_page_view(batch, **values)
    return allocator.create_batch_view(batch, layer_id=0, **values)


@pytest.mark.parametrize("page", [False, True])
def test_compact_prefix_borrows_same_objects_and_builds_only_suffix(
    page: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = make_allocator()
    batch = make_batch(page=page)
    old = create_compact(allocator, batch, page=page)
    old_positions = old.metadata.cached_positions
    allocations = []
    original_arange = torch.arange

    def record_arange(*args: Any, **kwargs: Any) -> torch.Tensor:
        allocations.append(args)
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", record_arange)
    for _ in range(3):
        assert create_compact(allocator, batch, page=page, previous=old) is old
    assert old.metadata.cached_positions is old_positions
    assert old.metadata.ref_count == 1
    suffix = create_compact(
        allocator,
        batch,
        page=page,
        chunk_index=1,
        cached_positions=range(4, 8),
    )
    assert suffix is not old
    assert allocations == [(4, 8, 1)]
    assert torch.equal(suffix.metadata.cached_positions, original_arange(4, 8))
    old.ref_count_down()
    suffix.ref_count_down()
    assert not old.is_valid() and not suffix.is_valid()


@pytest.mark.parametrize("page", [False, True])
@pytest.mark.parametrize(
    "change",
    ["hash", "offset", "shape", "dtype", "format", "range", "generation", "invalid"],
)
def test_compact_descriptor_change_never_reuses_old_view(
    page: bool, change: str
) -> None:
    allocator = make_allocator()
    batch = make_batch(page=page)
    old = create_compact(allocator, batch, page=page)
    values: dict[str, Any] = {}
    if change == "hash":
        batch = replace(batch, chunk_hashes=[999, 102])
    elif change == "offset":
        batch = (
            replace(batch, page_offsets=[256, 384])
            if page
            else replace(batch, offsets=[256, 320, 384, 448])
        )
    elif change == "shape":
        values["shape"] = torch.Size([2, 4, 4])
    elif change == "dtype":
        values.update(dtype=torch.float32, shape=torch.Size([1, 4, 4]))
    elif change == "format":
        values.update(fmt=MemoryFormat.KV_2TD, shape=torch.Size([4, 1, 8]))
    elif change == "range":
        values["cached_positions"] = range(4, 8)
    elif change == "generation":
        allocator.generation = 10
    else:
        old.ref_count_down()
    current = create_compact(allocator, batch, page=page, previous=old, **values)
    assert current is not old
    if old.is_valid():
        old.ref_count_down()
    current.ref_count_down()


@pytest.mark.parametrize("page", [False, True])
@pytest.mark.parametrize("change", ["shm", "bounds"])
def test_compact_reuse_still_rejects_wrong_slab_or_bounds(
    page: bool, change: str
) -> None:
    allocator = make_allocator()
    batch = make_batch(page=page)
    old = create_compact(allocator, batch, page=page)
    if change == "shm":
        batch = replace(batch, shm_name="/other-slab")
    else:
        batch = (
            replace(batch, page_offsets=[4096, 4096])
            if page
            else replace(batch, offsets=[4096] * 4)
        )
    with pytest.raises(SharedCPUCacheValidationError):
        create_compact(allocator, batch, page=page, previous=old)
    assert old.is_valid()
    assert old.metadata.ref_count == 1
    old.ref_count_down()


def test_plain_handle_reuses_after_validation_and_invalidates_changed_metadata() -> (
    None
):
    allocator = make_allocator()
    handle = SharedChunkHandle(
        request_id="req",
        phase="dense_prefix",
        key=make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/prefill-test",
        offset=0,
        physical_size=64,
        logical_size=64,
        shape=torch.Size([4, 8]),
        dtype=torch.float16,
        fmt=MemoryFormat.KV_T2D,
        generation=9,
        producer_rank=0,
        cached_positions=list(range(4)),
    )
    expected = dict(
        expected_request_id="req",
        expected_phase="dense_prefix",
        expected_layer_id=0,
        expected_kv_group=0,
        expected_cached_positions=range(4),
    )
    old = allocator.create_view(handle, **expected)
    assert allocator.create_view(handle, previous=old, **expected) is old
    current = allocator.create_view(
        replace(handle, offset=64), previous=old, **expected
    )
    assert current is not old
    for changed in (
        replace(handle, generation=10),
        replace(handle, shm_name="/other-slab"),
        replace(handle, cached_positions=[1, 2, 3, 4]),
    ):
        with pytest.raises(SharedCPUCacheValidationError):
            allocator.create_view(changed, previous=old, **expected)
    old.ref_count_down()
    current.ref_count_down()


class CountedSource:
    """Minimal observable reference owner for the public lease contract."""

    def __init__(self, *, pinned: bool = False) -> None:
        self.ref_count = 1
        self.pin_count = int(pinned)

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0

    def is_valid(self) -> bool:
        return self.ref_count > 0

    def ref_count_up(self) -> None:
        self.ref_count += 1

    def ref_count_down(self) -> None:
        assert self.ref_count > 0
        self.ref_count -= 1

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        assert self.pin_count > 0
        self.pin_count -= 1
        return True


def test_replaced_partial_tail_survives_store_seed_promotion_until_close() -> None:
    prefix, tail, replacement, suffix = [CountedSource() for _ in range(4)]
    lease = SharedCPURequestLease("req", 9, True)
    lease.replace_groups({0: [[prefix, tail]]}, retain=True)
    # The replacement arrives with a resolver-owned ref and pin.
    replacement.ref_count_up()
    replacement.pin()
    lease.replace_groups(
        {0: [[prefix, replacement]]}, retain=False, preserve_replaced=True
    )
    assert lease.object_ids(0) == {id(prefix), id(replacement)}
    assert lease.object_ids() == {id(prefix), id(tail), id(replacement)}
    lease.replace_groups({0: [[prefix, replacement, suffix]]}, retain=True)
    lease.replace_groups({0: [[prefix, replacement, suffix]]}, retain=True)
    assert [
        (obj.ref_count, obj.pin_count) for obj in (prefix, tail, replacement, suffix)
    ] == [(2, 1)] * 4
    lease.close()
    lease.close()
    assert [
        (obj.ref_count, obj.pin_count) for obj in (prefix, tail, replacement, suffix)
    ] == [(1, 0)] * 4


class AdoptionEngine(LMCacheEngine):
    """Expose the dense-adoption boundary for focused refcount fault tests."""

    def __init__(self, *, rank0: bool) -> None:
        self.num_layers = 1
        self.gpu_connector = SimpleNamespace(
            supports_dense_sparse_cache_retention=lambda: True
        )
        self.shared_cpu_cache_generation = 9
        self.metadata = SimpleNamespace(is_first_rank=lambda: rank0)
        self._shared_cpu_request_leases = {}

    def adopt(self, sources: list[CountedSource], caches: dict[str, Any]) -> bool:
        return self._adopt_dense_shared_retrieve_cache(
            req_id="req",
            starts=[i * 4 for i in range(len(sources))],
            ends=[(i + 1) * 4 for i in range(len(sources))],
            keys_layer_major=[[make_key(i) for i in range(len(sources))]],
            memory_objs=[sources],
            handles=[[object() for _ in sources]],
            kv_group=0,
            kwargs={
                "_retain_shared_dense_cache": True,
                "_reuse_shared_dense_prefix": True,
                "deferred_layerwise_get": True,
                "prefill_dma_block_ids_by_bank": {},
                **caches,
            },
        )


def make_caches(sources: list[CountedSource]) -> dict[str, Any]:
    return {
        "cached_keys": [],
        "cached_starts": [],
        "cached_ends": [],
        "cached_memory_objs": [],
        "cached_shared_handles": [],
        "cached_chunk_dev_ptrs": [[id(obj) for obj in sources]],
        "cached_chunk_ptrs_npu": [None],
    }


def test_rank0_promoted_sources_do_not_gain_duplicate_adopted_references() -> None:
    engine = AdoptionEngine(rank0=True)
    prefix, suffix = CountedSource(), CountedSource()
    engine.retain_shared_cpu_store_seed("req", {0: [[prefix, suffix]]})
    caches = make_caches([prefix, suffix])
    for _ in range(2):
        for obj in (prefix, suffix):
            obj.ref_count_up()
            obj.pin()
        assert engine.adopt([prefix, suffix], caches)
        assert [(obj.ref_count, obj.pin_count) for obj in (prefix, suffix)] == [
            (2, 1)
        ] * 2
    engine.release_shared_cpu_sparse_request("req")
    assert [(obj.ref_count, obj.pin_count) for obj in (prefix, suffix)] == [(1, 0)] * 2


def test_failed_incremental_adoption_keeps_previous_ownership_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = AdoptionEngine(rank0=False)
    prefix, suffix = CountedSource(), CountedSource()
    caches = make_caches([prefix])
    assert engine.adopt([prefix], caches)
    previous_metadata = caches["cached_memory_objs"][0]

    def reject_registration(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("registration failed")

    monkeypatch.setattr(
        engine, "register_shared_cpu_sparse_request", reject_registration
    )
    caches["cached_chunk_dev_ptrs"][0].append(id(suffix))
    with pytest.raises(RuntimeError, match="registration failed"):
        engine.adopt([prefix, suffix], caches)
    assert caches["cached_memory_objs"][0] is previous_metadata
    assert caches["cached_memory_objs"] == [[prefix]]
    assert prefix.ref_count == suffix.ref_count == 1
    suffix.ref_count_down()  # The failed caller still owns only its new suffix.
    engine.release_shared_cpu_sparse_request("req")
    assert prefix.ref_count == suffix.ref_count == 0
