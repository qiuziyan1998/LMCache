# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission preparation: actual engine, pages, and LocalCPU ownership."""

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
import asyncio
import gc
import os

import pytest
import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    _RemoteFillChunkLookupPlan,
    _RemoteFillLookupPlan,
    _SharedRank0ObjectContext,
)
from lmcache.v1.config import LMCacheEngineConfig, prefill_cpu_prefetch_enabled
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager


@pytest.fixture
def setup():
    config = LMCacheEngineConfig.from_defaults(chunk_size=4)
    PinMonitor.GetOrCreate(config)
    slab = torch.zeros(65536, dtype=torch.uint8)
    allocator = TensorMemoryAllocator(slab)
    local = LocalCPUBackend(config=config, memory_allocator=allocator)
    remote_calls = []
    allocated = []
    cancelled = set()
    engine_class = LMCacheEngine
    if os.getenv("LMCACHE_TEST_ASCEND_PREFETCH") == "1":
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        engine_class = AscendLMCacheEngine
    engine = object.__new__(engine_class)
    engine.config = SimpleNamespace(chunk_size=4)
    engine.lookup_pins = defaultdict(lambda: defaultdict(list))
    engine._remote_fill_lookup_plans = {}
    engine._remote_fill_pair_lookup_enabled = lambda: True
    engine._shared_local_cpu_backend = lambda: local
    engine.num_layers_for_group = lambda group: (2, 3)[group]
    engine._expected_shared_cpu_chunk_metadata = (
        lambda kv_group, num_tokens, layer_id=0: (
            torch.Size([num_tokens * (layer_id + 2 if kv_group else 2)]),
            torch.uint8,
            MemoryFormat.KV_DSA_INDEX_FMT,
        )
    )
    if engine_class is not LMCacheEngine:
        # Exercise Ascend's real group/mixed-C8 metadata override on CPU pages.
        del engine._expected_shared_cpu_chunk_metadata
        engine.dsa_two_groups = True
        engine.metadata = SimpleNamespace(
            use_mla=True,
            indexer_c8_layout=SimpleNamespace(
                mixed=True, layer_bytes=lambda tokens, layer: tokens * (layer + 2)
            ),
            kv_layer_groups_manager=SimpleNamespace(
                kv_layer_groups=[SimpleNamespace(hidden_dim_size=2, dtype=torch.uint8)]
            ),
        )
        engine._memory_format_for_kv_group = lambda group: MemoryFormat.KV_DSA_INDEX_FMT
    engine._shared_rank0_object_context = lambda group: _SharedRank0ObjectContext(
        allocator,
        slab.numel(),
        torch.uint8,
        MemoryFormat.KV_DSA_INDEX_FMT,
    )

    def make_key(i, group, n):
        return CacheEngineKey(
            "model",
            1,
            0,
            i,
            torch.uint8,
            {"lmcache.tag.internal.valid_tokens": str(n)} if n < 4 else None,
            kv_group=group,
        )

    def group_key(k, kv_group, request_configs):
        from lmcache.v1.mooncake_layout import mooncake_valid_tokens

        return make_key(k.chunk_hash, kv_group, mooncake_valid_tokens(k, 4))

    engine._lookup_key_for_kv_group = group_key

    def allocate(k, n):
        layout = [
            engine._expected_shared_cpu_chunk_metadata(
                kv_group=k.kv_group, num_tokens=n, layer_id=layer
            )
            for layer in range(engine.num_layers_for_group(k.kv_group))
        ]
        pages = allocator.batched_allocate_layer_pages(
            [x[0] for x in layout],
            [x[1] for x in layout],
            1,
            len(layout),
            layout[0][2],
            valid_tokens=n,
            full_tokens=n,
        )
        assert pages
        page = pages[0]
        for layer in range(page.num_layers):
            page.layer_tensor(layer).fill_(k.chunk_hash + layer + k.kv_group)
        allocated.append(page)
        return page

    def submit(keys):
        from lmcache.v1.mooncake_layout import mooncake_valid_tokens

        remote_calls.append(keys)
        pages = [allocate(k, mooncake_valid_tokens(k, 4)) for k in keys]
        local.batched_submit_layer_pages(keys, pages)
        future = Future()
        future.set_result(pages)
        return future

    manager = SimpleNamespace(
        storage_backends={
            "RemoteBackend": SimpleNamespace(submit_layer_pages_get=submit)
        },
        async_lookup_server=SimpleNamespace(
            is_lookup_cancelled=lambda req: req in cancelled
        ),
        get_active_storage_backends=lambda search_range=None: [
            ("LocalCPUBackend", local)
        ],
        batched_unpin=lambda keys, locations: local.batched_unpin(keys),
    )
    manager.batched_contains_two_group_layer_pages = (
        lambda *a, **kw: StorageManager.batched_contains_two_group_layer_pages(
            manager, *a, **kw
        )
    )
    manager.async_lookup_server.send_lookup_complete = lambda req: None
    engine.storage_manager = manager

    def plan(tail=4, hot=0):
        keys = [
            [make_key(i, group, 4 if i == 0 else tail) for i in range(2)]
            for group in (0, 1)
        ]
        chunks = tuple(
            _RemoteFillChunkLookupPlan(
                i * 4,
                i * 4 + (4 if i == 0 else tail),
                i,
                tuple(
                    "LocalCPUBackend" if i < hot else "RemoteBackend" for _ in (0, 1)
                ),
                (True, True),
            )
            for i in range(2)
        )
        engine._remote_fill_lookup_plans["req"] = _RemoteFillLookupPlan(chunks)
        candidates = [
            (c.start, c.end, k) for c, k in zip(chunks, keys[0], strict=False)
        ]
        engine.token_database = SimpleNamespace(
            process_tokens=lambda **kw: iter(candidates)
        )
        for group in (0, 1):
            for i, k in enumerate(keys[group][:hot]):
                page = allocate(k, 4 if i == 0 else tail)
                local.batched_submit_layer_pages([k], [page])
                page.ref_count_down()
        if hot:
            count, mapping = manager.batched_contains_two_group_layer_pages(
                *[[k.get_first_layer() for k in g[:hot]] for g in keys],
                ["LocalCPUBackend"],
                True,
            )
            engine.lookup_pins["req"].update(mapping)
        return keys, 4 + tail

    yield engine, local, remote_calls, allocated, cancelled, plan
    engine._release_lookup_pins("req")
    for page in allocated:
        while page.is_valid() and page.metadata.pin_count > 0:
            page.unpin()
    local.clear()
    for page in allocated:
        while page.is_valid() and page.get_ref_count() > 0:
            page.ref_count_down()
    allocator.close()


@pytest.mark.parametrize("hot", [0, 1, 2])
@pytest.mark.parametrize("tail", [1, 3, 4])
def test_prepared_prefix_is_pinned_and_foreground_needs_no_remote(setup, hot, tail):
    engine, local, calls, pages, _, plan = setup
    keys, end = plan(tail, hot)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert engine._prefetch_prefill_cpu_prefix("req", end) == end
        assert len(calls) == (0 if hot == 2 else 2)
        assert all(
            c.locations_by_group == ("LocalCPUBackend",) * 2
            for c in engine._remote_fill_lookup_plans["req"].chunks
        )
        assert all(p.get_ref_count() == 1 and p.metadata.pin_count == 1 for p in pages)
        assert all(not p.can_evict for p in pages)
        # The actual exact-plan resolver must use LocalCPU and its owned refs.
        for group in (0, 1):
            engine._resolve_shared_rank0_layer_mem_objs = lambda **kw: []
            result, count = engine._resolve_shared_rank0_exact_layer_pages(
                req_id="req",
                phase="prefill",
                kv_group=group,
                keys_layer_major=[[k.get_first_layer() for k in keys[group]]]
                * engine.num_layers_for_group(group),
                page_chunks=2,
                base_page_keys=keys[group],
                chunk_locations=["LocalCPUBackend"] * 2,
            )
            engine._release_shared_retrieve_objs(result[0], unpin=True)
            assert count == 2
        assert len(calls) == (0 if hot == 2 else 2)
        engine._release_lookup_pins("req")
        assert all(p.metadata.pin_count == 0 and p.get_ref_count() == 1 for p in pages)
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("failure", ["cancel", "capacity", "bad_layout", "short"])
def test_failure_never_publishes_local_plan_and_releases_new_ownership(setup, failure):
    engine, local, calls, pages, cancelled, plan = setup
    _, end = plan()
    original = engine.storage_manager.storage_backends[
        "RemoteBackend"
    ].submit_layer_pages_get

    def submit(keys):
        if keys[0].kv_group == 1 and failure == "capacity":
            raise MemoryError("capacity")
        f = original(keys)
        if failure == "cancel":
            cancelled.add("req")
        if failure == "bad_layout":
            f.result()[0].valid_tokens += 1
        if failure == "short":
            removed = f.result().pop()
            removed.ref_count_down()
        return f

    engine.storage_manager.storage_backends[
        "RemoteBackend"
    ].submit_layer_pages_get = submit
    with pytest.raises((ValueError, RuntimeError, MemoryError)):
        engine._prefetch_prefill_cpu_prefix("req", end)
    assert all(
        c.locations_by_group == ("RemoteBackend",) * 2
        for c in engine._remote_fill_lookup_plans["req"].chunks
    )
    assert all(p.metadata.pin_count == 0 and p.get_ref_count() == 1 for p in pages)


def test_pending_native_read_keeps_owners_until_completion(setup):
    engine, local, calls, pages, cancelled, plan = setup
    _, end = plan()
    entered = Event()
    completion = Future()
    original = engine.storage_manager.storage_backends[
        "RemoteBackend"
    ].submit_layer_pages_get

    def submit(keys):
        ready = original(keys)
        completion.pages = ready.result()
        entered.set()
        return completion

    engine.storage_manager.storage_backends[
        "RemoteBackend"
    ].submit_layer_pages_get = submit
    with ThreadPoolExecutor(1) as pool:
        job = pool.submit(engine._prefetch_prefill_cpu_prefix, "req", end)
        assert entered.wait(2)
        cancelled.add("req")
        assert not job.done()
        assert all(p.get_ref_count() == 2 for p in pages)
        completion.set_result(completion.pages)
        with pytest.raises(RuntimeError, match="cancelled"):
            job.result(2)
    assert all(p.get_ref_count() == 1 for p in pages)


def test_new_knob_defaults_on_but_d_and_sync_paths_are_unchanged():
    c = LMCacheEngineConfig.from_defaults()
    assert c.enable_prefill_cpu_prefetch is True
    assert not prefill_cpu_prefetch_enabled(c)
    for k, v in dict(
        enable_async_loading=True,
        pd_role="sender",
        use_layerwise=True,
        enable_shared_cpu_cache=True,
        dsa_two_groups=True,
        enable_remote_lmcache_store=True,
    ).items():
        setattr(c, k, v)
    assert prefill_cpu_prefetch_enabled(c)
    c.pd_role = "receiver"
    assert not prefill_cpu_prefetch_enabled(c)


@pytest.mark.parametrize("cancel", [False, True])
def test_async_entry_waits_for_preparation_without_blocking_loop(
    setup, monkeypatch, cancel
):
    engine, local, calls, pages, cancelled, plan = setup
    _, end = plan()
    engine.use_layerwise = True
    engine.config = SimpleNamespace(
        chunk_size=4,
        enable_async_loading=True,
        enable_prefill_cpu_prefetch=True,
        pd_role="sender",
        use_layerwise=True,
        enable_shared_cpu_cache=True,
        dsa_two_groups=True,
        enable_remote_lmcache_store=True,
    )
    engine._sampled_scheduler_lookup_requested = lambda: True
    engine.lookup = lambda **kw: end
    submitted = []
    replies = []
    server = engine.storage_manager.async_lookup_server
    lookup_finished = []
    server.send_lookup_complete = lookup_finished.append
    server.send_response_to_scheduler = lambda req, result: replies.append(
        (req, result)
    )
    entered = Event()
    release = Event()
    native = engine.storage_manager.storage_backends[
        "RemoteBackend"
    ].submit_layer_pages_get

    def read(keys):
        assert lookup_finished == ["req"]
        entered.set()
        assert release.wait(3)
        return native(keys)

    engine.storage_manager.storage_backends[
        "RemoteBackend"
    ].submit_layer_pages_get = read
    monkeypatch.setattr(
        asyncio,
        "run_coroutine_threadsafe",
        lambda coro, loop: submitted.append(loop.create_task(coro)),
    )

    async def run():
        engine.storage_manager.loop = asyncio.get_running_loop()
        engine.async_lookup_and_prefetch("req", hashes=[0, 1], offsets=[4, 4], pin=True)
        assert await asyncio.to_thread(entered.wait, 2)
        assert replies == []
        # Foreground event-loop work runs while the background payload read waits.
        progressed = []
        engine.storage_manager.loop.call_soon(progressed.append, True)
        await asyncio.sleep(0)
        assert progressed == [True]
        if cancel:
            cancelled.add("req")
        release.set()
        await submitted[0]

    asyncio.run(run())
    assert replies == [("req", 0 if cancel else end)]
    if cancel:
        assert "req" not in engine.lookup_pins
        assert "req" not in engine._remote_fill_lookup_plans
        assert all(p.metadata.pin_count == 0 and p.get_ref_count() == 1 for p in pages)


def test_legacy_plan_falls_back_without_fetching(setup):
    engine, _, calls, _, _, plan = setup
    _, end = plan()
    original = engine._remote_fill_lookup_plans["req"]
    engine._remote_fill_lookup_plans["req"] = _RemoteFillLookupPlan(
        tuple(replace(c, page_by_group=(True, False)) for c in original.chunks)
    )
    assert engine._prefetch_prefill_cpu_prefix("req", end) == end
    assert calls == []


def test_cancel_before_background_start_issues_no_lookup(setup, monkeypatch):
    engine, _, calls, _, cancelled, _ = setup
    cancelled.add("req")
    engine.use_layerwise = True
    engine.config = SimpleNamespace(
        enable_async_loading=True,
        enable_prefill_cpu_prefetch=True,
        pd_role="sender",
        use_layerwise=True,
        enable_shared_cpu_cache=True,
        dsa_two_groups=True,
        enable_remote_lmcache_store=True,
    )
    engine._sampled_scheduler_lookup_requested = lambda: False
    engine.lookup = lambda **kw: pytest.fail("cancelled lookup must not run")
    replies = []
    engine.storage_manager.async_lookup_server.send_response_to_scheduler = (
        lambda req, result: replies.append(result)
    )
    monkeypatch.setattr(
        asyncio, "run_coroutine_threadsafe", lambda coro, loop: asyncio.run(coro)
    )
    engine.storage_manager.loop = None
    engine.async_lookup_and_prefetch("req", pin=True)
    assert replies == [0] and calls == []


def test_preparation_stops_key_construction_at_proven_prefix(setup):
    engine, _, _, _, _, plan = setup
    _, end = plan()
    original = engine.token_database.process_tokens

    def tokens(**kwargs):
        chunks = list(original(**kwargs))
        yield from chunks
        yield end, end + 4, chunks[-1][2]
        raise AssertionError("preparation walked the uncached suffix")

    engine.token_database.process_tokens = tokens
    assert engine._prefetch_prefill_cpu_prefix("req", end) == end


def test_preparation_resolves_slab_context_once_per_group(setup):
    engine, _, _, _, _, plan = setup
    _, end = plan()
    original = engine._shared_rank0_object_context
    calls = []

    def context(group):
        calls.append(group)
        return original(group)

    engine._shared_rank0_object_context = context
    assert engine._prefetch_prefill_cpu_prefix("req", end) == end
    assert calls == [0, 1]


def test_prefetch_pins_actual_cache_winners_and_retires_duplicate_reads(setup):
    engine, local, _, pages, _, plan = setup
    _, end = plan(hot=2)
    original = engine._remote_fill_lookup_plans["req"]
    engine._remote_fill_lookup_plans["req"] = _RemoteFillLookupPlan(
        tuple(
            replace(c, locations_by_group=("RemoteBackend",) * 2)
            for c in original.chunks
        )
    )
    winners = list(pages)
    assert engine._prefetch_prefill_cpu_prefix("req", end) == end
    assert all(p.get_ref_count() == 1 and p.metadata.pin_count == 1 for p in winners)
    assert all(not p.is_valid() for p in pages[len(winners) :])
    assert set(map(id, local.hot_cache.values())) == set(map(id, winners))


def test_mismatched_retained_prefix_is_rejected_before_io(setup):
    engine, _, calls, _, _, plan = setup
    _, end = plan()
    original = engine._remote_fill_lookup_plans["req"]
    engine._remote_fill_lookup_plans["req"] = _RemoteFillLookupPlan(
        tuple(replace(c, chunk_hash=c.chunk_hash + 100) for c in original.chunks)
    )
    with pytest.raises(ValueError, match="match retained prefix"):
        engine._prefetch_prefill_cpu_prefix("req", end)
    assert calls == []
