# SPDX-License-Identifier: Apache-2.0
"""Exercise production page retrieval with a native put holding the last page.

Only allocator storage and native I/O are substituted. The asyncio put cleanup,
page allocation/retry, and get publication are the production methods.
"""

# Standard
import ast
import asyncio
import logging
from pathlib import Path
import threading
import time
from types import MethodType, SimpleNamespace as NS
from typing import cast

# Third Party
import pytest


ROOT = Path(__file__).resolve().parents[2]


def methods(relative, names, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = ast.parse("from __future__ import annotations")
    module.body.extend(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class Key:
    kv_group = 0
    dtype = "test"


class Page:
    layer_size = 8

    def __init__(self):
        self.refs = 1
        self.filled = False

    def ref_count_up(self):
        self.refs += 1

    def ref_count_down(self):
        self.refs -= 1

    def layer_data_ptr(self, layer):
        return 100 + 8 * layer

    def is_valid(self):
        return self.refs > 0


def setup(*, pressure=False, failure=None, timeout=0.3):
    namespace = dict(
        asyncio=asyncio, time=time, cast=cast,
        logger=logging.getLogger(__name__),
        serving_perf_enabled=lambda: False,
        audit_enabled=lambda: False,
        LayerCacheEngineKey=type("LayerKey", (), {}),
        mooncake_valid_tokens=lambda key, size: size,
    )
    methods("lmcache/v1/storage_backend/local_cpu_backend.py", {
        "batched_allocate_layer_pages", "async_batched_allocate_layer_pages",
    }, namespace)
    methods("lmcache/v1/storage_backend/connector/mooncakestore_connector.py", {
        "batched_get_layer_pages", "_run_blocking_put", "_await_native_task",
    }, namespace)
    retained, page = Page(), Page()
    state = NS(attempts=0, evictions=0, calls=0, published=[],
               native_started=threading.Event(), native_release=threading.Event(),
               native_done=threading.Event(), retained=retained, page=page)

    def allocate(*args, **kwargs):
        state.attempts += 1
        if pressure and state.attempts == 1:
            # A native put completes while allocation is executing. Its Python
            # completion callback needs this same event loop to release refs.
            state.native_release.set()
            assert state.native_done.wait(2)
        if failure == "allocation" or (pressure and retained.refs > 1):
            return None
        return [page]

    def evict(*args, **kwargs):
        state.evictions += 1
        return [], []

    backend = NS(
        metadata=NS(chunk_size=1024), config=NS(blocking_timeout_secs=timeout),
        use_hot=True, cpu_lock=threading.Lock(),
        memory_allocator=NS(batched_allocate_layer_pages=allocate),
        _pop_layer_page_evict_candidate_locked=evict,
        stats_monitor=NS(update_local_cpu_evict_failed_count=lambda n: None,
                         update_local_cpu_evict_metrics=lambda n: None),
        batched_submit_layer_pages=lambda keys, rows: state.published.extend(rows),
    )
    for name in ("batched_allocate_layer_pages", "async_batched_allocate_layer_pages"):
        if name in namespace:
            setattr(backend, name, MethodType(namespace[name], backend))

    def get(*args):
        state.calls += 1
        if failure == "native":
            raise RuntimeError("native get failed")
        if failure == "short_read":
            return [8]  # Two layers require 16 bytes.
        page.filled = True
        return [16]

    connector = NS(
        _layer_merged_pages=True, _inflight_put_tasks=set(),
        _page_keys_for=lambda keys: ["page"], _page_num_layers_for=lambda key: 2,
        _metadata_for_raw_key=lambda key: ([(4,)], ["test"], 0, 16),
        config=NS(transfer_timeout=2), local_cpu_backend=backend,
        store=NS(batch_get_into_multi_buffers=get),
    )
    for name in ("batched_get_layer_pages", "_run_blocking_put"):
        setattr(connector, name, MethodType(namespace[name], connector))
    return connector, state


def test_native_put_completion_can_release_pages_while_get_waits():
    connector, state = setup(pressure=True)

    def put():
        state.native_started.set()
        assert state.native_release.wait(2)
        state.native_done.set()

    async def run():
        task = asyncio.create_task(connector._run_blocking_put(
            "test", put, (), [state.retained],
        ))
        try:
            assert await asyncio.to_thread(state.native_started.wait, 2)
            assert state.retained.refs == 2
            pages = await connector.batched_get_layer_pages([Key()])
            assert pages == [state.page], (
                "get starved native-put cleanup on its own event loop; "
                f"native_done={state.native_done.is_set()} refs={state.retained.refs}"
            )
            assert state.retained.refs == 1
            assert state.page.filled and state.published == pages
        finally:
            state.native_release.set()
            await task

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["allocation", "short_read", "native"])
def test_failed_get_never_publishes_unfilled_pages(failure):
    connector, state = setup(failure=failure, timeout=0.02)
    if failure == "allocation":
        assert asyncio.run(connector.batched_get_layer_pages([Key()])) == []
        assert state.calls == 0
    else:
        with pytest.raises(RuntimeError, match="expected|native get failed"):
            asyncio.run(connector.batched_get_layer_pages([Key()]))
        assert state.page.refs == 0
    assert not state.published and not state.page.filled


def test_successful_allocation_does_not_yield_or_retry(monkeypatch):
    connector, state = setup()

    async def forbidden(*args):
        pytest.fail("A successful allocation must not sleep")

    monkeypatch.setattr(asyncio, "sleep", forbidden)
    assert asyncio.run(connector.batched_get_layer_pages([Key()])) == [state.page]
    assert state.attempts == 1 and state.evictions == 0


def test_cancellation_during_allocation_never_submits_get():
    connector, state = setup(failure="allocation", timeout=60)

    async def run():
        task = asyncio.create_task(connector.batched_get_layer_pages([Key()]))
        await asyncio.sleep(0)  # Let allocation reach its cooperative wait.
        assert state.attempts > 0 and not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.calls == 0 and not state.published

    asyncio.run(run())


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("failure", ["allocation", "short_read", "native"])
def test_resolver_stops_instead_of_returning_unfilled_history(exact, failure):
    from test_prefill_page_source import resolver

    class LayerKey(Key):
        chunk = 0

        def split_layers(self, count):
            return [(0, layer) for layer in range(count)]

    connector, state = setup(failure=failure, timeout=0)
    engine, _, _, _ = resolver(set())
    engine._release_shared_retrieve_objs = lambda rows, **kw: [
        obj.ref_count_down() for obj in rows
    ]
    remote = engine.storage_manager.storage_backends["RemoteBackend"]
    remote.batched_get_layer_pages = lambda keys: asyncio.run(
        connector.batched_get_layer_pages(keys)
    )
    keys = [LayerKey()]
    message = {
        "allocation": "result count is inconsistent",
        "short_read": "expected",
        "native": "native get failed",
    }[failure]
    with pytest.raises((ValueError, RuntimeError), match=message):
        engine._resolve_shared_rank0_layer_pages(
            req_id="failed-load", phase="prefill", kv_group=0,
            keys_layer_major=[keys, keys], page_chunks=1, base_page_keys=keys,
            exact_chunk_locations=["RemoteBackend"] if exact else None,
        )
    assert not state.published and not state.page.filled
