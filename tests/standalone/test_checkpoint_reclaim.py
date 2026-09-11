# SPDX-License-Identifier: Apache-2.0
"""Exercise LocalCPU reclamation with counted cache ownership and capacity."""

import ast
from collections import OrderedDict
from itertools import islice
from pathlib import Path
from threading import Lock
from types import SimpleNamespace as NS

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
            },
        ),
        (
            ROOT / "lmcache/v1/storage_backend/cache_policy/lru.py",
            {"get_evict_candidates"},
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


def test_post_reclaim_capacity_race_is_not_reported_as_success():
    obj, pool, removed = backend()
    obj.hot_cache[1] = LayerPage(pool)
    # Another allocator consumes the newly freed capacity before the final check.
    obj.get_allocator_capacity_bytes = lambda: (0, pool.total)
    assert not reclaim(obj, 10)
    assert removed == [1]


def test_candidate_scan_does_not_visit_beyond_the_budget():
    obj, pool, removed = backend()
    visited = []

    class CountedCache(OrderedDict):
        def __iter__(self):
            for key in super().__iter__():
                visited.append(key)
                yield key

    obj.hot_cache = CountedCache((i, LayerPage(pool, pins=1)) for i in range(20))
    assert not reclaim(obj, 10, limit=3)
    assert visited == [0, 1, 2] and not removed
