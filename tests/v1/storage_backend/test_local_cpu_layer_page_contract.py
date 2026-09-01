# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterator, Sequence
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    MemoryFormat,
    TensorMemoryAllocator,
)
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import (
    ExternalTwoGroupCommitResult,
    LayerPageBatchPutResult,
    LocalCPUBackend,
)


class _RecordingSender:
    def __init__(self) -> None:
        self.operations: list[tuple[OpType, int]] = []

    def add_kv_op(self, *, op_type: OpType, key: int) -> None:
        self.operations.append((op_type, key))


def _page_key(index: int, group: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=index,
        dtype=torch.bfloat16,
        kv_group=group,
    )


@pytest.fixture
def page_backend() -> Iterator[
    tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]]
]:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        local_cpu=True,
        lmcache_instance_id="layer_page_contract",
    )
    PinMonitor.GetOrCreate(config)
    allocator = TensorMemoryAllocator(torch.zeros(2 * 1024 * 1024, dtype=torch.uint8))
    backend = LocalCPUBackend(config=config, memory_allocator=allocator)
    allocated_pages: list[LayerPageMemoryObj] = []

    def allocate(count: int) -> list[LayerPageMemoryObj]:
        pages = backend.batched_allocate_layer_pages(
            [torch.Size([8])],
            [torch.bfloat16],
            batch_size=count,
            num_layers=2,
            fmt=MemoryFormat.KV_DSA_INDEX_FMT,
            valid_tokens=8,
            full_tokens=8,
            eviction=False,
        )
        assert pages is not None
        allocated_pages.extend(pages)
        return pages

    yield backend, allocate

    for page in allocated_pages:
        while page.is_valid() and page.metadata.pin_count > 0:
            page.unpin()
    for key in backend.get_keys():
        backend.remove(key)
    for page in allocated_pages:
        while page.is_valid() and page.get_ref_count() > 0:
            page.ref_count_down()
    allocator.close()
    LMCStatsMonitor.unregister_all_metrics()
    LMCStatsMonitor.DestroyInstance()
    PinMonitor.DestroyInstance()


def test_two_group_prefix_rejects_mismatched_lengths(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, _ = page_backend

    with pytest.raises(ValueError, match="aligned key counts"):
        backend.batched_contains_two_group_prefix([_page_key(0, 0)], [], pin=False)

    with pytest.raises(ValueError, match="identical logical page pairs"):
        backend.batched_contains_two_group_prefix(
            [_page_key(0, 0)], [_page_key(1, 1)], pin=False
        )


def test_two_group_prefix_accepts_group_specific_schema_and_dtype(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, _ = page_backend
    common = {"lmcache.tag.payload": "layout-v3"}
    group0 = CacheEngineKey(
        "test_model",
        1,
        0,
        17,
        torch.float16,
        common,
        kv_group=0,
    )
    group1 = CacheEngineKey(
        "test_model",
        1,
        0,
        17,
        torch.bfloat16,
        {**common, "lmcache.tag.dsa_idx": "v2"},
        kv_group=1,
    )

    assert backend.batched_contains_two_group_prefix([group0], [group1]) == 0


def test_two_group_prefix_stops_at_first_hole(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    group0 = [_page_key(index, 0) for index in range(3)]
    group1 = [_page_key(index, 1) for index in range(3)]
    stored_keys = [group0[0], group1[0], group0[1], group0[2], group1[2]]

    backend.batched_submit_layer_pages_if_absent(stored_keys, allocate(5))

    assert backend.batched_contains_two_group_prefix(group0, group1) == 1


def test_two_group_prefix_pins_both_groups_atomically(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    group0 = [_page_key(index, 0) for index in range(2)]
    group1 = [_page_key(index, 1) for index in range(2)]
    interleaved = [key for pair in zip(group0, group1, strict=True) for key in pair]
    pages = allocate(4)
    backend.batched_submit_layer_pages_if_absent(interleaved, pages)

    assert backend.batched_contains_two_group_prefix(group0, group1, pin=True) == 2
    assert [page.metadata.pin_count for page in pages] == [1, 1, 1, 1]

    backend.touch_cache()
    backend.batched_unpin(interleaved)
    assert [page.metadata.pin_count for page in pages] == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("keys_factory", "pages_factory", "message"),
    [
        (
            lambda keys: keys[:1],
            lambda pages: pages,
            "aligned key counts",
        ),
        (
            lambda keys: [keys[0], keys[0]],
            lambda pages: pages,
            "unique keys",
        ),
        (
            lambda keys: keys,
            lambda pages: [pages[0], pages[0]],
            "unique pages",
        ),
    ],
)
def test_put_if_absent_validates_complete_batch(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
    keys_factory: Callable[[list[CacheEngineKey]], Sequence[CacheEngineKey]],
    pages_factory: Callable[[list[LayerPageMemoryObj]], Sequence[LayerPageMemoryObj]],
    message: str,
) -> None:
    backend, allocate = page_backend
    keys = [_page_key(0, 0), _page_key(0, 1)]
    pages = allocate(2)

    with pytest.raises(ValueError, match=message):
        backend.batched_submit_layer_pages_if_absent(
            keys_factory(keys), pages_factory(pages)
        )

    assert not backend.contains_any_exact(keys)


def test_put_if_absent_rejects_noncanonical_key_and_nonpage(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    page = allocate(1)[0]
    layer_key = LayerCacheEngineKey("test_model", 1, 0, 1, torch.bfloat16, layer_id=0)

    with pytest.raises(ValueError, match="layer-independent"):
        backend.batched_submit_layer_pages_if_absent([layer_key], [page])
    with pytest.raises(ValueError, match="layer-page objects"):
        backend.batched_submit_layer_pages_if_absent(
            [_page_key(0, 0)],
            [object()],  # type: ignore[list-item]
        )


def test_put_if_absent_rejects_invalid_page(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    page = allocate(1)[0]
    page.ref_count_down()
    assert not page.is_valid()

    with pytest.raises(ValueError, match="invalid pages"):
        backend.batched_submit_layer_pages_if_absent([_page_key(0, 0)], [page])


def test_put_if_absent_existing_entry_wins_without_notification(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    sender = _RecordingSender()
    backend.batched_msg_sender = sender  # type: ignore[assignment]
    key = _page_key(0, 0)
    original, duplicate = allocate(2)

    first = backend.batched_submit_layer_pages_if_absent([key], [original])
    assert sender.operations == [(OpType.ADMIT, key.chunk_hash)]
    sender.operations.clear()
    duplicate_ref_count = duplicate.get_ref_count()
    second = backend.batched_submit_layer_pages_if_absent([key], [duplicate])
    winner = backend.get_blocking(key)

    assert first == LayerPageBatchPutResult((key,), ())
    assert second == LayerPageBatchPutResult((), (key,))
    assert winner is original
    assert duplicate.get_ref_count() == duplicate_ref_count
    assert sender.operations == []
    assert winner is not None
    winner.ref_count_down()


def test_put_if_absent_is_not_partially_visible(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, allocate = page_backend
    keys = [_page_key(index, index % 2) for index in range(4)]
    pages = allocate(4)
    policy_entered = threading.Event()
    allow_policy = threading.Event()
    reader_started = threading.Event()
    original_update = backend.cache_policy.update_on_put_many

    def delayed_policy_update(updated_keys: Sequence[CacheEngineKey]) -> None:
        policy_entered.set()
        assert allow_policy.wait(timeout=5)
        original_update(updated_keys)

    monkeypatch.setattr(
        backend.cache_policy, "update_on_put_many", delayed_policy_update
    )

    def observe_complete_batch() -> bool:
        reader_started.set()
        return backend.contains_all_exact(keys)

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(
            backend.batched_submit_layer_pages_if_absent, keys, pages
        )
        assert policy_entered.wait(timeout=5)
        reader = executor.submit(observe_complete_batch)
        assert reader_started.wait(timeout=5)
        try:
            assert not reader.done()
        finally:
            allow_policy.set()

        assert writer.result(timeout=5) == LayerPageBatchPutResult(tuple(keys), ())
        assert reader.result(timeout=5)


def test_put_if_absent_rolls_back_policy_exception(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, allocate = page_backend
    keys = [_page_key(index, index % 2) for index in range(4)]
    pages = allocate(4)
    original_ref_counts = [page.get_ref_count() for page in pages]

    def fail_policy_update(_keys: Sequence[CacheEngineKey]) -> None:
        raise RuntimeError("proven policy failure")

    monkeypatch.setattr(backend.cache_policy, "update_on_put_many", fail_policy_update)

    with pytest.raises(RuntimeError, match="proven policy failure"):
        backend.batched_submit_layer_pages_if_absent(keys, pages)

    assert not backend.contains_any_exact(keys)
    assert [page.get_ref_count() for page in pages] == original_ref_counts


def test_put_if_absent_empty_and_disabled_are_noops(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    assert backend.batched_submit_layer_pages_if_absent([], []) == (
        LayerPageBatchPutResult((), ())
    )

    page = allocate(1)[0]
    initial_ref_count = page.get_ref_count()
    backend.use_hot = False
    result = backend.batched_submit_layer_pages_if_absent([_page_key(0, 0)], [page])

    assert result == LayerPageBatchPutResult((), ())
    assert page.get_ref_count() == initial_ref_count
    assert not backend.contains_any_exact([_page_key(0, 0)])


def test_external_commit_rejects_mismatched_groups(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, _ = page_backend

    with pytest.raises(ValueError, match="aligned key counts"):
        backend.commit_external_two_group_prefix_if_absent([_page_key(0, 0)], [], {})
    with pytest.raises(ValueError, match="Group 0/1"):
        backend.commit_external_two_group_prefix_if_absent(
            [_page_key(0, 1)], [_page_key(0, 0)], {}
        )
    with pytest.raises(ValueError, match="identical logical page pairs"):
        backend.commit_external_two_group_prefix_if_absent(
            [_page_key(0, 0)], [_page_key(1, 1)], {}
        )


def test_external_group0_commit_rejects_wrong_group_and_foreign_ready_key(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend

    with pytest.raises(ValueError, match="requires Group 0 keys"):
        backend.commit_external_group0_prefix_if_absent([_page_key(0, 1)], {})

    required = [_page_key(0, 0)]
    with pytest.raises(ValueError, match="belong to the required prefix"):
        backend.commit_external_group0_prefix_if_absent(
            required,
            {_page_key(1, 0): allocate(1)[0]},
        )


def test_external_group0_commit_combines_existing_and_ready_coverage(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    required = [_page_key(index, 0) for index in range(2)]
    existing, duplicate, new_page = allocate(3)
    backend.batched_submit_layer_pages_if_absent([required[0]], [existing])

    result = backend.commit_external_group0_prefix_if_absent(
        required,
        {required[0]: duplicate, required[1]: new_page},
    )

    assert result == ExternalTwoGroupCommitResult(
        committed=True,
        inserted_keys=(required[1],),
        existing_keys=(required[0],),
        redundant_pages=(duplicate,),
    )
    assert backend.contains_all_exact(required)


def test_external_group0_missing_page_is_atomic(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    required = [_page_key(index, 0) for index in range(2)]
    ready_page = allocate(1)[0]

    result = backend.commit_external_group0_prefix_if_absent(
        required,
        {required[0]: ready_page},
    )

    assert not result.committed
    assert result.missing_keys == (required[1],)
    assert result.redundant_pages == (ready_page,)
    assert not backend.contains_any_exact(required)


def test_external_commit_missing_reservation_inserts_nothing(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    group0 = [_page_key(index, 0) for index in range(2)]
    group1 = [_page_key(index, 1) for index in range(2)]
    ready_keys = [group0[0], group1[0], group0[1]]
    ready_pages = allocate(3)
    original_ref_counts = [page.get_ref_count() for page in ready_pages]

    result = backend.commit_external_two_group_prefix_if_absent(
        group0,
        group1,
        dict(zip(ready_keys, ready_pages, strict=True)),
    )

    assert result == ExternalTwoGroupCommitResult(
        committed=False,
        inserted_keys=(),
        existing_keys=(),
        redundant_pages=tuple(ready_pages),
        missing_keys=(group1[1],),
    )
    assert result.lock_wait_seconds >= 0
    assert result.lock_hold_seconds >= 0
    assert not backend.contains_any_exact([*group0, *group1])
    assert [page.get_ref_count() for page in ready_pages] == original_ref_counts


def test_external_commit_combines_existing_and_ready_coverage(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    group0 = [_page_key(index, 0) for index in range(2)]
    group1 = [_page_key(index, 1) for index in range(2)]
    required = [key for pair in zip(group0, group1, strict=True) for key in pair]
    existing_page, duplicate_page, *new_pages = allocate(5)
    backend.batched_submit_layer_pages_if_absent([group0[0]], [existing_page])
    duplicate_ref_count = duplicate_page.get_ref_count()
    ready = {
        group0[0]: duplicate_page,
        group1[0]: new_pages[0],
        group0[1]: new_pages[1],
        group1[1]: new_pages[2],
    }

    result = backend.commit_external_two_group_prefix_if_absent(group0, group1, ready)
    winner = backend.get_blocking(group0[0])

    assert result == ExternalTwoGroupCommitResult(
        committed=True,
        inserted_keys=(group1[0], group0[1], group1[1]),
        existing_keys=(group0[0],),
        redundant_pages=(duplicate_page,),
    )
    assert result.lock_wait_seconds >= 0
    assert result.lock_hold_seconds >= 0
    assert backend.contains_all_exact(required)
    assert winner is existing_page
    assert duplicate_page.get_ref_count() == duplicate_ref_count
    assert winner is not None
    winner.ref_count_down()


def test_missing_prefix_without_trace_is_not_misattributed(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, _ = page_backend
    group0 = [_page_key(0, 0)]
    group1 = [_page_key(0, 1)]
    diagnostics: dict[str, object] = {}

    assert (
        backend.batched_contains_two_group_prefix(
            group0,
            group1,
            diagnostics=diagnostics,
        )
        == 0
    )
    assert diagnostics["retention_trace_status"] == "no_exact_key_match"
    assert diagnostics["retention_attribution_status"] == "trace_not_matched"
    assert "retention_attributed_cause" not in diagnostics


def test_external_commit_retention_trace_identifies_explicit_remove(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_COLD_START_PERF", "1")
    backend, allocate = page_backend
    group0 = [_page_key(0, 0)]
    group1 = [_page_key(0, 1)]
    pages = allocate(2)

    result = backend.commit_external_two_group_prefix_if_absent(
        group0,
        group1,
        {group0[0]: pages[0], group1[0]: pages[1]},
    )
    assert result.retention_trace_id is not None
    assert backend.remove(group0[0])

    diagnostics: dict[str, object] = {}
    assert (
        backend.batched_contains_two_group_prefix(
            group0,
            group1,
            diagnostics=diagnostics,
        )
        == 0
    )
    assert diagnostics["retention_trace_status"] == "matched"
    assert diagnostics["retention_trace_id"] == result.retention_trace_id
    assert diagnostics["local_first_hole_pair"] == 0
    assert diagnostics["local_first_hole_group0_state"] == "absent"
    assert diagnostics["local_first_hole_group1_state"] == "committed_page"
    assert diagnostics["retention_attribution_status"] == "attributed"
    assert diagnostics["retention_attributed_cause"] == "explicit_remove"
    assert diagnostics["retention_attributed_operation"] == "remove"
    assert diagnostics["retention_attributed_pair"] == 0
    assert diagnostics["retention_attributed_group"] == 0
    assert diagnostics["retention_attributed_removed_committed_page"] is True
    assert "retention_attributed_callsite" in diagnostics


def test_external_commit_retention_trace_identifies_layer_page_allocation_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_COLD_START_PERF", "1")
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        local_cpu=True,
        lmcache_instance_id="layer_page_retention_trace",
    )
    allocator = TensorMemoryAllocator(torch.zeros(3 * 4096, dtype=torch.uint8))
    backend = LocalCPUBackend(config=config, memory_allocator=allocator)
    group0 = [_page_key(0, 0)]
    group1 = [_page_key(0, 1)]
    committed = backend.batched_allocate_layer_pages(
        [torch.Size([8])],
        [torch.bfloat16],
        batch_size=2,
        num_layers=2,
        fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=8,
        full_tokens=8,
        eviction=False,
    )
    assert committed is not None
    result = backend.commit_external_two_group_prefix_if_absent(
        group0,
        group1,
        {group0[0]: committed[0], group1[0]: committed[1]},
    )
    assert result.retention_trace_id is not None
    for page in committed:
        page.ref_count_down()

    allocated = backend.batched_allocate_layer_pages(
        [torch.Size([8])],
        [torch.bfloat16],
        batch_size=2,
        num_layers=2,
        fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        busy_loop=False,
        valid_tokens=8,
        full_tokens=8,
        eviction=True,
    )
    assert allocated is not None
    try:
        diagnostics: dict[str, object] = {}
        assert (
            backend.batched_contains_two_group_prefix(
                group0,
                group1,
                diagnostics=diagnostics,
            )
            == 0
        )
        assert diagnostics["retention_trace_status"] == "matched"
        assert diagnostics["retention_attribution_status"] == "attributed"
        assert diagnostics["retention_attributed_cause"] == "layer_page_allocate_evict"
        assert diagnostics["retention_attributed_removed_committed_page"] is True
        assert "retention_attributed_callsite" in diagnostics
        assert diagnostics["retention_mutation_causes"] == {
            "layer_page_allocate_evict": 1
        }
    finally:
        for key in backend.get_keys():
            backend.remove(key)
        for page in allocated:
            if page.is_valid():
                page.ref_count_down()
        allocator.close()
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()
        PinMonitor.DestroyInstance()


def test_capacity_reclaim_is_bounded_and_skips_pinned_pages() -> None:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        local_cpu=True,
        lmcache_instance_id="bounded_capacity_reclaim",
    )
    PinMonitor.GetOrCreate(config)
    allocator = TensorMemoryAllocator(torch.zeros(4 * 4096, dtype=torch.uint8))
    backend = LocalCPUBackend(config=config, memory_allocator=allocator)
    keys = [_page_key(0, 0), _page_key(1, 0)]
    pages = backend.batched_allocate_layer_pages(
        [torch.Size([8])],
        [torch.bfloat16],
        batch_size=2,
        num_layers=2,
        fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=8,
        full_tokens=8,
        eviction=False,
    )
    assert pages is not None
    backend.batched_submit_layer_pages(keys, pages)
    for page in pages:
        page.ref_count_down()
    assert pages[1].pin()

    try:
        assert not backend.reclaim_evictable_capacity(
            2 * 4096,
            min_free_bytes=2 * 4096,
            min_free_ratio=0,
            num_layers=2,
            cause="test_capacity_reclaim",
        )
        assert backend.get_keys() == keys

        pages[1].unpin()
        assert backend.reclaim_evictable_capacity(
            4096,
            min_free_bytes=2 * 4096,
            min_free_ratio=0,
            num_layers=2,
            cause="test_capacity_reclaim",
        )
        assert backend.get_keys() == keys[1:]
        assert backend.get_allocator_capacity_bytes()[0] == 3 * 4096
    finally:
        if pages[1].metadata.pin_count:
            pages[1].unpin()
        for key in backend.get_keys():
            backend.remove(key)
        allocator.close()
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()
        PinMonitor.DestroyInstance()


def test_external_commit_rejects_wrong_kind_existing_winner(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    group0 = [_page_key(0, 0)]
    group1 = [_page_key(0, 1)]
    ready_pages = allocate(2)
    invalid_winner = object()
    with backend.cpu_lock:
        backend.hot_cache[group0[0]] = invalid_winner  # type: ignore[assignment]

    result = backend.commit_external_two_group_prefix_if_absent(
        group0,
        group1,
        dict(zip((group0[0], group1[0]), ready_pages, strict=True)),
    )

    assert not result.committed
    assert result.missing_keys == (group0[0],)
    assert not backend.contains_any_exact(group1)
    with backend.cpu_lock:
        assert backend.hot_cache.pop(group0[0]) is invalid_winner


def test_external_commit_rejects_static_metadata_mismatch_existing_winner(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
) -> None:
    backend, allocate = page_backend
    group0 = [_page_key(0, 0)]
    group1 = [_page_key(0, 1)]
    existing, ready_group0, ready_group1 = allocate(3)
    backend.batched_submit_layer_pages_if_absent(group0, [existing])
    existing.valid_tokens = 7
    existing.meta.valid_tokens = 7

    result = backend.commit_external_two_group_prefix_if_absent(
        group0,
        group1,
        {group0[0]: ready_group0, group1[0]: ready_group1},
    )

    assert not result.committed
    assert result.missing_keys == (group0[0],)
    assert not backend.contains_any_exact(group1)
    existing.valid_tokens = 8
    existing.meta.valid_tokens = 8


def test_external_commit_is_not_partially_visible(
    page_backend: tuple[LocalCPUBackend, Callable[[int], list[LayerPageMemoryObj]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, allocate = page_backend
    group0 = [_page_key(index, 0) for index in range(2)]
    group1 = [_page_key(index, 1) for index in range(2)]
    required = [key for pair in zip(group0, group1, strict=True) for key in pair]
    pages = allocate(4)
    ready = dict(zip(required, pages, strict=True))
    policy_entered = threading.Event()
    allow_policy = threading.Event()
    reader_started = threading.Event()
    original_update = backend.cache_policy.update_on_put_many

    def delayed_policy_update(updated_keys: Sequence[CacheEngineKey]) -> None:
        policy_entered.set()
        assert allow_policy.wait(timeout=5)
        original_update(updated_keys)

    monkeypatch.setattr(
        backend.cache_policy, "update_on_put_many", delayed_policy_update
    )

    def observe_complete_batch() -> bool:
        reader_started.set()
        return backend.contains_all_exact(required)

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(
            backend.commit_external_two_group_prefix_if_absent,
            group0,
            group1,
            ready,
        )
        assert policy_entered.wait(timeout=5)
        reader = executor.submit(observe_complete_batch)
        assert reader_started.wait(timeout=5)
        try:
            assert not reader.done()
        finally:
            allow_policy.set()

        assert writer.result(timeout=5) == ExternalTwoGroupCommitResult(
            committed=True,
            inserted_keys=tuple(required),
            existing_keys=(),
            redundant_pages=(),
        )
        assert reader.result(timeout=5)
