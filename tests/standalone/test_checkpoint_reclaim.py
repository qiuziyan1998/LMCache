# SPDX-License-Identifier: Apache-2.0
"""Exercise LocalCPU reclamation with counted cache ownership and capacity."""

from collections import OrderedDict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from threading import Lock
from types import SimpleNamespace as NS
import ast
import logging
import threading

import pytest

ROOT = Path(__file__).resolve().parents[2]


class LayerKey(tuple):
    def split_layers(self, count):
        return [LayerKey((self[0], layer)) for layer in range(count)]


class Page:
    def __init__(self, pool, size=10, *, pins=0, refs=1):
        self.pool, self.size, self.pins, self.refs = pool, size, pins, refs

    @property
    def can_evict(self):
        return self.pins == 0 and self.refs == 1

    def get_physical_size(self):
        return self.size

    def ref_count_down(self):
        assert self.refs > 0
        self.refs -= 1
        if not self.refs:
            self.pool.free += self.size


class LayerPage(Page):
    pass


@pytest.mark.parametrize("counts", [(3, 1), (79, 22)])
def test_fragmented_reclaim_keeps_per_group_layer_counts(counts):
    class GroupKey(LayerKey):
        @property
        def kv_group(self):
            return self[2]

        def split_layers(self, count):
            return [GroupKey((self[0], layer, self.kv_group)) for layer in range(count)]

    obj, pool, _ = backend()
    required = 10 * sum(counts)
    pool.free, pool.total = required, 4096
    pages = []
    for group, count in enumerate(counts):
        for layer in range(count):
            page = Page(pool)
            pages.append(page)
            obj.hot_cache[GroupKey(("chunk", layer, group))] = page
    assert obj.reclaim_evictable_capacity(
        required, min_free_bytes=0, min_free_ratio=0, num_layers=counts,
        cause="checkpoint", max_scan_entries=sum(counts), allocation_failed=True,
    )
    assert not obj.hot_cache
    assert all(page.refs == 0 for page in pages)
    assert pool.free == 2 * required


def backend():
    ns = dict(
        islice=islice,
        serving_perf_enabled=lambda: False,
        LayerCacheEngineKey=LayerKey,
        LayerPageMemoryObj=LayerPage,
    )
    paths = [
        (
            ROOT / "lmcache/v1/storage_backend/local_cpu_backend.py",
            {
                "reclaim_evictable_capacity",
                "_pop_bounded_reclaim_locked",
                "_pop_layer_page_evict_candidate_locked",
                "try_touch_layer_pages",
            },
        ),
        (
            ROOT / "lmcache/v1/storage_backend/cache_policy/lru.py",
            {"get_evict_candidates", "update_on_hit"},
        ),
    ]
    for path, names in paths:
        nodes = [
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, *nodes], type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            ns,
        )
    policy = type(
        "LRUCachePolicy",
        (),
        {
            "get_evict_candidates": ns["get_evict_candidates"],
            "update_on_hit": ns["update_on_hit"],
            "update_chunk_hash_dict": lambda self, key: None,
            "update_on_force_evict": lambda self, key: None,
        },
    )
    ns["LRUCachePolicy"] = policy
    cls = type("Backend", (), {name: ns[name] for name in paths[0][1]})
    obj = cls()
    pool = NS(free=0, total=1000)
    obj.get_allocator_capacity_bytes = lambda: (pool.free, pool.total)
    obj.cpu_lock, obj.cache_policy, obj.hot_cache, obj.use_hot = (
        Lock(),
        policy(),
        OrderedDict(),
        True,
    )
    obj.stats_monitor = NS(
        update_local_cpu_evict_metrics=lambda n: None,
        update_local_cpu_evict_failed_count=lambda n: None,
    )
    removed = []
    obj._record_external_retention_mutation_locked = lambda key, **kw: removed.append(
        key
    )
    return obj, pool, removed


def reclaim(obj, size, limit=16):
    return obj.reclaim_evictable_capacity(
        size,
        min_free_bytes=0,
        min_free_ratio=0,
        num_layers=2,
        cause="checkpoint_capacity_reclaim",
        max_scan_entries=limit,
    )


def test_reclaim_skips_pinned_and_borrowed_pages_and_stops_at_target():
    obj, pool, removed = backend()
    obj.hot_cache.update(
        [
            (1, LayerPage(pool, pins=1)),
            (2, LayerPage(pool, refs=2)),
            (3, LayerPage(pool)),
            (4, LayerPage(pool)),
            (5, LayerPage(pool)),
        ]
    )
    assert reclaim(obj, 20)
    assert removed == [3, 4] and pool.free == 20
    assert list(obj.hot_cache) == [1, 2, 5]


def test_insufficient_candidate_window_does_not_evict_anything():
    obj, pool, removed = backend()
    obj.hot_cache.update((i, LayerPage(pool)) for i in range(4))
    assert not reclaim(obj, 30, limit=2)
    assert not removed and len(obj.hot_cache) == 4 and pool.free == 0


def test_busy_cache_lock_refuses_without_waiting():
    obj, pool, removed = backend()
    obj.hot_cache[1] = LayerPage(pool)
    obj.cpu_lock.acquire()
    try:
        assert not reclaim(obj, 10)
    finally:
        obj.cpu_lock.release()
    assert not removed


@pytest.mark.parametrize("borrowed", [False, True])
def test_legacy_chunk_requires_every_layer_to_be_evictable(borrowed):
    obj, pool, removed = backend()
    keys = [LayerKey(("chunk", layer)) for layer in range(2)]
    obj.hot_cache[keys[0]] = Page(pool)
    obj.hot_cache[keys[1]] = Page(pool, refs=2 if borrowed else 1)
    assert reclaim(obj, 10) is (not borrowed)
    assert removed == ([] if borrowed else keys)


def test_legacy_chunk_cannot_expand_beyond_scan_window():
    obj, pool, removed = backend()
    for layer in range(2):
        obj.hot_cache[LayerKey(("chunk", layer))] = Page(pool)
    assert not reclaim(obj, 10, limit=1)
    assert not removed


def test_layer_major_legacy_entries_can_reclaim_one_complete_chunk():
    obj, pool, removed = backend()
    for layer in range(2):
        for chunk in ("a", "b", "c"):
            obj.hot_cache[LayerKey((chunk, layer))] = Page(pool)
    # The sibling is farther away in LRU order, but checking this complete
    # two-layer chunk fits the two-entry inspection budget.
    assert reclaim(obj, 20, limit=2)
    assert removed == [LayerKey(("a", 0)), LayerKey(("a", 1))]


def test_full_scan_remote_fill_behavior_remains_available():
    obj, pool, removed = backend()
    obj.hot_cache.update(
        [(1, LayerPage(pool, pins=1)), (2, LayerPage(pool)), (3, LayerPage(pool))]
    )
    assert obj.reclaim_evictable_capacity(
        10,
        min_free_bytes=10,
        min_free_ratio=0,
        num_layers=2,
        cause="remote_fill_capacity_reclaim",
    )
    assert removed == [2, 3] and pool.free == 20


def test_other_policy_preserves_bounded_refusal_instead_of_changing_victims():
    obj, pool, removed = backend()
    obj.cache_policy = object()
    obj.hot_cache[1] = LayerPage(pool)
    assert not reclaim(obj, 10)
    assert not removed


def test_no_eviction_if_capacity_already_available():
    obj, pool, removed = backend()
    pool.free = 20
    obj.cpu_lock.acquire()
    try:
        assert reclaim(obj, 10)
    finally:
        obj.cpu_lock.release()
    assert not removed


@pytest.mark.parametrize("failed", [False, True])
def test_post_reclaim_capacity_race_is_not_reported_as_success(failed):
    obj, pool, removed = backend()
    obj.hot_cache[1] = LayerPage(pool)
    # Another allocator consumes the newly freed capacity before the final check.
    obj.get_allocator_capacity_bytes = lambda: (0, pool.total)
    operation = failed_allocation_reclaim if failed else reclaim
    assert not operation(obj, 10)
    assert removed == [1]


@pytest.mark.parametrize("failed", [False, True])
def test_candidate_scan_does_not_visit_beyond_the_budget(failed):
    obj, pool, removed = backend()
    visited = []

    class CountedCache(OrderedDict):
        def __iter__(self):
            for key in super().__iter__():
                visited.append(key)
                yield key

    obj.hot_cache = CountedCache((i, LayerPage(pool, pins=1)) for i in range(20))
    operation = failed_allocation_reclaim if failed else reclaim
    assert not operation(obj, 10, limit=3)
    assert visited == [0, 1, 2] and not removed


def test_checkpoint_touch_preserves_owners_and_lookup_list():
    obj, pool, _ = backend()
    obj.hot_cache.update((i, LayerPage(pool, refs=2, pins=i)) for i in range(4))
    obj.hot_cache[4] = Page(pool)
    obj.keys_in_request = ["unrelated"]
    assert obj.try_touch_layer_pages([3, 2, 99, 1, 0, 4])
    assert list(obj.hot_cache) == [4, 3, 2, 1, 0]
    assert obj.keys_in_request == ["unrelated"]
    assert [(obj.hot_cache[i].refs, obj.hot_cache[i].pins) for i in range(4)] == [
        (2, i) for i in range(4)
    ]


def test_checkpoint_touch_does_not_wait_or_override_other_policies():
    obj, pool, _ = backend()
    obj.hot_cache.update((i, LayerPage(pool)) for i in range(3))
    obj.cpu_lock.acquire()
    try:
        assert not obj.try_touch_layer_pages([2, 1, 0])
    finally:
        obj.cpu_lock.release()
    obj.cache_policy = object()
    assert not obj.try_touch_layer_pages([2, 1, 0])
    assert list(obj.hot_cache) == [0, 1, 2]


def failed_allocation_reclaim(obj, size, limit=16):
    return obj.reclaim_evictable_capacity(
        size,
        min_free_bytes=0,
        min_free_ratio=0,
        num_layers=2,
        cause="checkpoint_capacity_reclaim",
        max_scan_entries=limit,
        allocation_failed=True,
    )


def test_allocation_failure_evicts_one_large_idle_page_despite_free_byte_total():
    obj, pool, removed = backend()
    pool.free = 100
    obj.hot_cache.update(
        [
            ("pinned", LayerPage(pool, 90, pins=1)),
            ("borrowed", LayerPage(pool, 90, refs=2)),
            ("small", LayerPage(pool, 10)),
            ("large", LayerPage(pool, 80)),
            ("later", LayerPage(pool, 90)),
        ]
    )
    assert failed_allocation_reclaim(obj, 60)
    assert removed == ["large"] and pool.free == 180
    assert obj.hot_cache["pinned"].pins == 1
    assert obj.hot_cache["borrowed"].refs == 2


def test_fragmentation_reclaim_can_free_less_than_request_to_allow_coalescing():
    obj, pool, removed = backend()
    pool.free = 60  # Three separated 20-byte free spans; request needs 50.
    obj.hot_cache["between_free_spans"] = LayerPage(pool, 35)
    assert failed_allocation_reclaim(obj, 50)
    assert removed == ["between_free_spans"]


def test_fragmentation_scan_and_eviction_are_bounded():
    obj, pool, removed = backend()
    pool.free = 100
    obj.hot_cache.update((i, LayerPage(pool, 20)) for i in range(10))
    obj.hot_cache["outside_budget"] = LayerPage(pool, 100)
    assert failed_allocation_reclaim(obj, 30, limit=4)
    assert removed == [0, 1]  # No whole-cache purge while seeking a large page.


def test_failed_allocation_reclaim_keeps_sources_when_capacity_is_impossible():
    obj, pool, removed = backend()
    pool.free = 10
    obj.hot_cache[0] = LayerPage(pool, 20)
    assert not failed_allocation_reclaim(obj, 50)
    assert removed == []


def test_failed_allocation_reclaim_never_waits_on_cache_lock():
    obj, pool, removed = backend()
    pool.free = 100
    obj.hot_cache[0] = LayerPage(pool, 80)
    obj.cpu_lock.acquire()
    try:
        # Capacity is only advisory; the caller still must retry allocation.
        assert failed_allocation_reclaim(obj, 60)
    finally:
        obj.cpu_lock.release()
    assert removed == []


def test_failed_allocation_reclaim_rejects_unbounded_mode():
    obj, _, _ = backend()
    with pytest.raises(ValueError, match="reclaim request"):
        failed_allocation_reclaim(obj, 60, limit=None)


def test_failed_allocation_reclaim_preserves_other_policy_and_legacy_owners():
    obj, pool, removed = backend()
    pool.free = 100
    keys = [LayerKey(("chunk", i)) for i in range(2)]
    obj.hot_cache[keys[0]] = Page(pool, 60)
    obj.hot_cache[keys[1]] = Page(pool, 60, refs=2)
    assert failed_allocation_reclaim(obj, 60)
    assert removed == []
    obj.hot_cache["large"] = LayerPage(pool, 80)
    obj.cache_policy = object()
    assert failed_allocation_reclaim(obj, 60)
    assert removed == []


def test_real_address_allocator_retries_fragmented_pool_after_page_eviction():
    # Only the sorted container is a fixture; allocation/coalescing are production.
    class SortedBlocks(list):
        def __init__(self, *, key):
            self.key = key

        def add(self, value):
            self.append(value)
            self.sort(key=self.key)

        def bisect_left(self, value):
            return next(
                (i for i, old in enumerate(self) if self.key(old) >= self.key(value)),
                len(self),
            )

    def synchronized(name):
        def decorate(method):
            def call(self, *args, **kwargs):
                with getattr(self, name):
                    return method(self, *args, **kwargs)

            return call

        return decorate

    path = ROOT / "lmcache/v1/memory_management.py"
    nodes = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name in {"FreeBlock", "AddressManager"}
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ns = dict(
        dataclass=dataclass,
        threading=threading,
        SortedList=SortedBlocks,
        synchronized=synchronized,
        _lmcache_nvtx_annotate=lambda f: f,
        logger=logging.getLogger(__name__),
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    manager = ns["AddressManager"](200, align_bytes=1)
    spans = [manager.allocate(n) for n in (40, 10, 70, 10, 40, 30)]
    manager.free(*spans[0])
    manager.free(*spans[4])
    assert manager.get_capacity_bytes() == (80, 200)
    with pytest.raises(RuntimeError, match="no enough memory"):
        manager.batched_allocate(60, 1)
    obj, pool, removed = backend()
    obj.get_allocator_capacity_bytes = manager.get_capacity_bytes

    class AllocatedPage(LayerPage):
        def ref_count_down(self):
            self.refs -= 1
            if self.refs == 0:
                manager.free(*spans[2])

    obj.hot_cache["victim"] = AllocatedPage(pool, 70)
    assert failed_allocation_reclaim(obj, 60)
    assert removed == ["victim"]
    assert manager.batched_allocate(60, 1) == [(50, 60)]


def test_failed_allocation_frees_removed_owners_outside_cache_lock():
    obj, pool, removed = backend()
    pool.free = 100

    class CheckedPage(LayerPage):
        def ref_count_down(self):
            assert not obj.cpu_lock.locked()
            assert "victim" not in obj.hot_cache
            super().ref_count_down()

    obj.hot_cache["victim"] = CheckedPage(pool, 80)
    assert failed_allocation_reclaim(obj, 60)
    assert removed == ["victim"]
