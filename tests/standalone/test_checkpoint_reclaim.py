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


@pytest.mark.parametrize("limit", [None, 16])
def test_reclaim_skips_pinned_and_borrowed_pages_and_stops_at_target(limit):
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
    assert reclaim(obj, 20, limit=limit)
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


@pytest.mark.parametrize("limit", [None, 16])
@pytest.mark.parametrize("borrowed", [False, True])
def test_legacy_chunk_requires_every_layer_to_be_evictable(borrowed, limit):
    obj, pool, removed = backend()
    keys = [LayerKey(("chunk", layer)) for layer in range(2)]
    obj.hot_cache[keys[0]] = Page(pool)
    obj.hot_cache[keys[1]] = Page(pool, refs=2 if borrowed else 1)
    assert reclaim(obj, 10, limit=limit) is (not borrowed)
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


def test_remote_fill_retains_the_requested_free_capacity_floor():
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


@pytest.mark.parametrize("limit", [None, 16])
def test_no_eviction_if_capacity_already_available(limit):
    obj, pool, removed = backend()
    class NoScan(OrderedDict):
        def __iter__(self):
            raise AssertionError("no-pressure path must not scan")

        def items(self):
            raise AssertionError("no-pressure path must not scan")

        def values(self):
            raise AssertionError("no-pressure path must not scan")

    obj.hot_cache = NoScan()
    pool.free = 20
    obj.cpu_lock.acquire()
    try:
        assert reclaim(obj, 10, limit=limit)
    finally:
        obj.cpu_lock.release()
    assert not removed


@pytest.mark.parametrize("limit", [None, 16])
def test_post_reclaim_capacity_race_is_not_reported_as_success(limit):
    obj, pool, removed = backend()
    obj.hot_cache[1] = LayerPage(pool)
    # Another allocator consumes the newly freed capacity before the final check.
    obj.get_allocator_capacity_bytes = lambda: (0, pool.total)
    assert not reclaim(obj, 10, limit=limit)
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


def test_unbounded_lru_checks_each_candidate_once_and_stops_early():
    obj, pool, removed = backend()
    checked = []

    class CountedPage(LayerPage):
        @property
        def can_evict(self):
            checked.append(self)
            return super().can_evict

    pages = [CountedPage(pool, pins=int(i < 100)) for i in range(1000)]
    obj.hot_cache.update(enumerate(pages))
    assert reclaim(obj, 30, limit=None)
    assert removed == [100, 101, 102]
    assert checked == pages[:103]
    assert pool.free == 30


def test_unbounded_lru_does_not_refuse_victims_beyond_checkpoint_scan_limit():
    obj, pool, removed = backend()
    obj.hot_cache.update((i, LayerPage(pool, pins=1)) for i in range(100))
    obj.hot_cache[100] = LayerPage(pool)
    assert reclaim(obj, 10, limit=None)
    assert removed == [100]


def test_unbounded_shortfall_preserves_cache_order_references_and_capacity():
    obj, pool, removed = backend()
    pages = [LayerPage(pool), LayerPage(pool, pins=1), LayerPage(pool, refs=2)]
    obj.hot_cache.update(enumerate(pages))
    assert not reclaim(obj, 11, limit=None)
    assert list(obj.hot_cache.items()) == list(enumerate(pages))
    assert [page.refs for page in pages] == [1, 1, 2]
    assert not removed and pool.free == 0


@pytest.mark.parametrize("blocked_layer", [0, 1])
def test_unbounded_lru_skips_blocked_legacy_group_for_later_page(blocked_layer):
    obj, pool, removed = backend()
    keys = [LayerKey(("chunk", layer)) for layer in range(2)]
    for layer, key in enumerate(keys):
        obj.hot_cache[key] = Page(pool, pins=int(layer == blocked_layer))
    obj.hot_cache["later"] = LayerPage(pool)
    assert reclaim(obj, 10, limit=None)
    assert removed == ["later"] and list(obj.hot_cache) == keys
    assert all(page.refs == 1 for page in obj.hot_cache.values())


def test_unbounded_legacy_siblings_are_not_counted_twice():
    obj, pool, removed = backend()
    keys = [LayerKey((chunk, layer)) for layer in range(2) for chunk in ("a", "b")]
    obj.hot_cache.update((key, Page(pool)) for key in keys)
    assert not reclaim(obj, 41, limit=None)
    assert not removed
    assert reclaim(obj, 21, limit=None)
    assert removed == [keys[0], keys[2], keys[1], keys[3]]
    assert pool.free == 40


def test_non_lru_unbounded_reclaim_preserves_policy_victim_order():
    obj, pool, removed = backend()

    class NewestFirst(type(obj.cache_policy)):
        def get_evict_candidates(self, cache, num_candidates=1):
            eligible = [key for key in reversed(cache) if cache[key].can_evict]
            return eligible[:num_candidates]

    obj.cache_policy = NewestFirst()
    obj.hot_cache.update((i, LayerPage(pool)) for i in range(3))
    assert reclaim(obj, 20, limit=None)
    assert removed == [2, 1] and list(obj.hot_cache) == [0]


@pytest.mark.parametrize("hot", [False, True])
def test_empty_cache_cannot_reclaim_capacity(hot):
    obj, pool, removed = backend()
    obj.use_hot = hot
    assert not reclaim(obj, 10, limit=None)
    assert not removed and pool.free == 0


def test_unbounded_release_is_outside_cache_lock_and_preserves_ratio_floor():
    obj, pool, removed = backend()

    class ReleasedPage(LayerPage):
        def ref_count_down(self):
            assert not obj.cpu_lock.locked()
            super().ref_count_down()

    pool.free = 90
    obj.hot_cache.update((i, ReleasedPage(pool)) for i in range(10))
    assert obj.reclaim_evictable_capacity(
        20, min_free_bytes=5, min_free_ratio=0.1, num_layers=2,
        cause="remote_fill_capacity_reclaim",
    )
    assert removed == [0, 1, 2] and pool.free == 120


def test_perf_log_distinguishes_selected_bytes_from_total_cache(monkeypatch):
    obj, pool, _ = backend()
    obj.hot_cache.update((i, LayerPage(pool)) for i in range(10))
    scope = obj.reclaim_evictable_capacity.__func__.__globals__
    logs = []
    monkeypatch.setitem(scope, "serving_perf_enabled", lambda: True)
    monkeypatch.setitem(scope, "time", NS(perf_counter=lambda: 1.0))
    monkeypatch.setitem(scope, "logger", object())
    monkeypatch.setitem(scope, "serving_perf_log", lambda *args, **kw: logs.append(kw))
    assert reclaim(obj, 20, limit=None)
    assert logs[0]["evictable_bytes"] == logs[0]["evicted_bytes"] == 20
    assert logs[0]["evictable_bytes_scope"] == "eligible_candidates"
