# SPDX-License-Identifier: Apache-2.0
"""Request and page ownership must not depend on automatic cyclic GC."""

# Standard
from collections.abc import Iterator
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
import asyncio
import gc
import logging
import threading
import time
import weakref

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_mod
from lmcache.v1.storage_backend.connector import (
    mooncakestore_connector as mooncake_mod,
)
from lmcache.v1.storage_backend import remote_backend as remote_mod
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


@pytest.fixture
def gc_disabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    logger = logging.Logger("gc-disabled-cleanup")
    logger.addHandler(logging.NullHandler())
    for module in (adapter_mod, mooncake_mod):
        monkeypatch.setattr(module, "serving_perf_enabled", lambda: False)
        monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(remote_mod, "logger", logger)
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        gc.collect()


class _Request:
    def __init__(self) -> None:
        self.req_id = "request"
        self.load_spec = SimpleNamespace(dsa_group1_direct_hbm=True)
        self.token_ids = list(range(1024))


def _failed_future(owner: object) -> Future:
    future = Future()
    try:
        try:
            raise ValueError("native failure")
        except ValueError as cause:
            raise RuntimeError("load failure") from cause
    except RuntimeError as error:
        future.set_exception(error)
    # The stored traceback references this frame, its future and the owner.
    return future


def _failed_adapter() -> tuple[Any, _Request, Future, Future]:
    impl = object.__new__(adapter_mod.LMCacheConnectorV1Impl)
    impl.lmcache_engine = SimpleNamespace(
        remote_fill_requires_paired_restart=lambda: False
    )
    impl._invalid_block_ids = set()
    impl._release_request_lookup_pins = lambda _req_id: None
    request = _Request()
    latent = _failed_future(request)
    indexer = _failed_future(request)
    impl._get_cold_load_coordinator().futures = {
        request.req_id: (1, latent, request, {7}, 0.0, indexer)
    }
    impl._get_cold_load_coordinator().last_latent_future = latent
    return impl, request, latent, indexer


def test_failed_cold_load_releases_frames_without_gc(gc_disabled: None) -> None:
    def complete() -> list[weakref.ReferenceType]:
        impl, request, latent, indexer = _failed_adapter()
        refs = [weakref.ref(obj) for obj in (impl, request, latent, indexer)]
        assert impl._drain_dsa_cold_load_futures() == {"request"}
        assert impl._invalid_block_ids == {7}
        assert not impl._get_cold_load_coordinator().futures
        for future in (latent, indexer):
            error = future.exception()
            assert error.__traceback__ is None
            assert error.__cause__.__traceback__ is None
        return refs

    refs = [ref for _ in range(50) for ref in complete()]
    assert not gc.isenabled()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("mode", ["pending", "unknown_dma", "failed_fence"])
def test_unretired_cold_load_preserves_failure_owners(
    gc_disabled: None, mode: str
) -> None:
    impl, request, latent, indexer = _failed_adapter()
    if mode == "pending":
        indexer = Future()
        impl._get_cold_load_coordinator().futures[request.req_id] = (
            1,
            latent,
            request,
            {7},
            0.0,
            indexer,
        )
    elif mode == "unknown_dma":
        impl.lmcache_engine.remote_fill_requires_paired_restart = lambda: True
    else:
        request.load_spec.dsa_group1_direct_hbm = False

        def fail_fence() -> None:
            raise RuntimeError("fence failed")

        impl._synchronize_dsa_cold_dense_load = fail_fence

    if mode == "unknown_dma":
        with pytest.raises(RuntimeError, match="load failure"):
            impl._drain_dsa_cold_load_futures()
    else:
        assert impl._drain_dsa_cold_load_futures() is None
    assert request.req_id in impl._get_cold_load_coordinator().futures
    assert latent.exception().__traceback__ is not None
    assert impl._invalid_block_ids == set()


class _Page:
    layer_size = 16
    data_ptr = 1234

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.refs = 1
        self.releases = 0

    def layer_data_ptr(self, layer: int) -> int:
        if self.mode == "pointer":
            raise RuntimeError("pointer setup failed")
        return 1234 + layer * self.layer_size

    def is_valid(self) -> bool:
        return self.refs > 0

    def ref_count_up(self) -> None:
        self.refs += 1

    def get_size(self) -> int:
        return self.layer_size

    def ref_count_down(self) -> None:
        assert self.refs > 0
        self.refs -= 1
        self.releases += 1


class _PageBackend:
    metadata = SimpleNamespace(chunk_size=8)

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.pages = [_Page(mode), _Page(mode)]
        self.published = False

    def batched_allocate_layer_pages(self, *args: Any, **kwargs: Any) -> list[_Page]:
        return self.pages

    def batched_submit_layer_pages(self, keys: list[Any], pages: list[_Page]) -> None:
        if self.mode == "publish":
            # A partial publication owns one extra cache reference.
            pages[0].refs += 1
            raise RuntimeError("publication failed")
        self.published = True


def _page_connector(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> tuple[Any, list[SimpleNamespace]]:
    connector = object.__new__(mooncake_mod.MooncakestoreConnector)
    connector._layer_merged_pages = True
    connector._page_num_layers = 2
    connector.local_cpu_backend = _PageBackend(mode)
    connector._page_keys_for = lambda keys: ["page0", "page1"]
    connector._metadata_for_raw_key = lambda key: ([None], [None], None, None)
    monkeypatch.setattr(mooncake_mod, "mooncake_valid_tokens", lambda *_: 8)

    def native_get(
        keys: list[Any], ptrs: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        if mode == "native":
            raise RuntimeError("native failure")
        return [0, 0] if mode == "status" else [sum(s) for s in sizes]

    connector.store = SimpleNamespace(batch_get_into_multi_buffers=native_get)
    keys = [SimpleNamespace(kv_group=0, dtype=None) for _ in range(2)]
    return connector, keys


@pytest.mark.parametrize(
    "mode", ["pointer", "submit", "native", "status", "publish", "success"]
)
@pytest.mark.parametrize("perf_enabled", [False, True])
def test_page_get_releases_caller_references_without_gc(
    gc_disabled: None, monkeypatch: pytest.MonkeyPatch, mode: str, perf_enabled: bool
) -> None:
    monkeypatch.setattr(mooncake_mod, "serving_perf_enabled", lambda: perf_enabled)
    connector, keys = _page_connector(monkeypatch, mode)

    async def run() -> None:
        if mode == "submit":

            def fail_submit(coro: Any) -> None:
                coro.close()
                raise RuntimeError("task submission failed")

            monkeypatch.setattr(asyncio, "create_task", fail_submit)
        if mode == "success":
            pages = await connector.batched_get_layer_pages(keys)
            assert connector.local_cpu_backend.published
            assert all(page.refs == 1 for page in pages)
            for page in pages:
                page.ref_count_down()
        else:
            with pytest.raises(RuntimeError):
                await connector.batched_get_layer_pages(keys)

    asyncio.run(run())
    pages = connector.local_cpu_backend.pages
    assert [page.releases for page in pages] == [1, 1]
    assert [page.refs for page in pages] == ([1, 0] if mode == "publish" else [0, 0])
    assert not gc.isenabled()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("mode", ["success", "failure", "cancelled"])
def test_remote_put_callbacks_release_frames_without_gc(
    gc_disabled: None, batched: bool, mode: str
) -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    refs = []
    notifications = []

    async def put(key: Any, owner: _Page) -> None:
        if mode == "failure":
            raise RuntimeError("remote put failed")
        if mode == "cancelled":
            raise asyncio.CancelledError()

    async def batched_put(keys: Any, owners: list[_Page]) -> None:
        await put(keys[0], owners[0])

    backend = object.__new__(RemoteBackend)
    backend.connection = SimpleNamespace(
        put=put, batched_put=batched_put, support_batched_put=lambda: True
    )
    backend.loop = loop
    backend.config = SimpleNamespace(enable_remote_lmcache_store=True)
    backend._mla_worker_id_as0_mode = False
    backend.lock = threading.Lock()
    backend.put_tasks = set()
    backend._single_put_futures = {}
    backend._single_put_callbacks = {}
    backend._put_failed_count = 0
    backend.serializer = SimpleNamespace(serialize=lambda owner: owner)

    def submit() -> Future:
        owner = _Page("success")
        refs.append(weakref.ref(owner))
        if batched:
            futures = backend.batched_submit_put_task(
                ["key"], [owner], on_complete_callback=notifications.append
            )
            assert futures is not None
            return futures[0]
        return backend.submit_put_task(
            "key", owner, on_complete_callback=notifications.append
        )

    async def settle() -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    try:
        for _ in range(20):
            future = submit()
            # Inspect terminal errors without adding test frames to them.
            error = future.exception(timeout=3)
            assert (error is None) == (mode == "success")
            del error, future
            asyncio.run_coroutine_threadsafe(settle(), loop).result(3)
        assert notifications == ["key"] * 20
        assert not backend.put_tasks
        assert not backend._single_put_futures
        assert not backend._single_put_callbacks
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)
        loop.close()
    assert len(refs) == 20
    assert all(ref() is None for ref in refs)
    assert not gc.isenabled()


def test_repeated_page_get_cancellation_retains_native_destinations(
    gc_disabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector, keys = _page_connector(monkeypatch, "success")
    entered = threading.Event()
    release = threading.Event()
    native_finished = threading.Event()

    def native_get(
        keys: list[Any], ptrs: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        entered.set()
        assert release.wait(5), "test failed to release native work"
        native_finished.set()
        return [sum(s) for s in sizes]

    connector.store.batch_get_into_multi_buffers = native_get

    async def run() -> None:
        task = asyncio.create_task(connector.batched_get_layer_pages(keys))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not native_finished.is_set()
            assert not task.done()
            assert all(page.refs == 1 for page in connector.local_cpu_backend.pages)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert native_finished.is_set()
    assert not connector.local_cpu_backend.published
    assert all(page.refs == 0 for page in connector.local_cpu_backend.pages)
    assert all(page.releases == 1 for page in connector.local_cpu_backend.pages)


@pytest.mark.parametrize("mode", ["success", "native", "status", "publish"])
def test_page_connector_drops_retired_transfer_frames(
    gc_disabled: None, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    def complete() -> list[weakref.ReferenceType]:
        connector, keys = _page_connector(monkeypatch, mode)
        refs = [weakref.ref(connector)] + [
            weakref.ref(page) for page in connector.local_cpu_backend.pages
        ]

        async def run() -> None:
            try:
                pages = await connector.batched_get_layer_pages(keys)
            except RuntimeError:
                assert mode != "success"
            else:
                assert mode == "success"
                for page in pages:
                    page.ref_count_down()

        asyncio.run(run())
        return refs

    refs = [ref for _ in range(20) for ref in complete()]
    assert all(ref() is None for ref in refs)
    assert not gc.isenabled()


@pytest.mark.parametrize("operation", ["get", "put", "blocking_put"])
@pytest.mark.parametrize("mode", ["success", "failure", "late_failure"])
def test_native_connector_drops_terminal_transfer_frames(
    gc_disabled: None, operation: str, mode: str
) -> None:
    def complete() -> list[weakref.ReferenceType]:
        connector = object.__new__(mooncake_mod.MooncakestoreConnector)
        connector.config = SimpleNamespace(
            transfer_timeout=0.01 if mode == "late_failure" else 3
        )
        connector.save_chunk_meta = False
        connector._page_first_multi_buffer = True
        connector._external_put_lock = asyncio.Lock()
        connector._inflight_put_tasks = set()
        connector._external_native_deadline = lambda: time.perf_counter() + 3
        connector._validate_external_buffer_owners = lambda *_: None
        connector._register_external_owners = lambda *_: None
        connector._external_page_key = lambda *_: "page"
        owner = _Page("success")
        owner.device = SimpleNamespace(type="cpu")

        def native(*args: Any) -> Any:
            if mode == "late_failure":
                time.sleep(0.03)
            if mode != "success":
                raise RuntimeError("native failed")
            return ([0], 0, 0, 0, 0) if operation == "put" else [32]

        connector.store = SimpleNamespace(batch_get_into_multi_buffers=native)
        connector._batch_put_multi_buffers_by_segment = native
        refs = [weakref.ref(owner), weakref.ref(connector)]

        async def run() -> None:
            try:
                if operation == "blocking_put":
                    await connector._run_blocking_put("put", native, (), [owner])
                elif operation == "put":
                    await connector.batched_put_external_pages(
                        ["key"], [[1234]], [[32]], (owner,), (), "request"
                    )
                else:
                    await connector.batched_get_external_pages(
                        ["key"], [[1234]], [[32]], (owner,), "request"
                    )
            except (RuntimeError, TimeoutError):
                assert mode != "success"
            else:
                assert mode == "success"
            # Blocking puts retain owners past the soft timeout until native
            # completion. Drain the test's actual background tasks too.
            if connector._inflight_put_tasks:
                await asyncio.gather(
                    *tuple(connector._inflight_put_tasks), return_exceptions=True
                )
            await asyncio.sleep(0)
            assert not connector._inflight_put_tasks
            assert owner.refs == 1

        asyncio.run(run())
        return refs

    refs = [ref for _ in range(10) for ref in complete()]
    assert all(ref() is None for ref in refs)
    assert not gc.isenabled()


@pytest.mark.parametrize("legacy", [False, True])
def test_page_fallback_native_failure_drops_frames(
    gc_disabled: None, legacy: bool
) -> None:
    def complete() -> list[weakref.ReferenceType]:
        connector = object.__new__(mooncake_mod.MooncakestoreConnector)
        pages = [_Page("success")]
        connector.meta_shapes = connector.meta_dtypes = connector.meta_fmt = True
        connector._allocate_zero_copy_buffers = lambda _: (
            pages,
            [(None, None, None, 1)],
            "test",
        )
        connector._has_zero_copy_storage = lambda _: True

        def native(*args: Any) -> None:
            raise RuntimeError("native failed")

        connector.store = SimpleNamespace(
            batch_get_into=native, batch_get_into_multi_buffers=native
        )
        refs = [weakref.ref(connector), weakref.ref(pages[0])]
        if legacy:
            result = asyncio.run(connector._batch_get_into_legacy(["key"]))
        else:
            result = asyncio.run(connector._batch_get_pages(["key"], [("page", [0])]))
        assert result == [None]
        assert pages[0].refs == 0
        assert pages[0].releases == 1
        return refs

    refs = [ref for _ in range(20) for ref in complete()]
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize(
    "mode", ["complete", "unknown", "cancelled_native", "native_timeout"]
)
def test_external_native_drain_preserves_deadline_and_ownership(
    gc_disabled: None, mode: str
) -> None:
    async def run() -> None:
        native = asyncio.get_running_loop().create_future()
        waiter = asyncio.create_task(
            mooncake_mod._wait_external_native_until_hard_deadline(
                native,
                deadline=time.perf_counter() + (0.05 if mode == "unknown" else 3),
                operation="get",
            )
        )
        await asyncio.sleep(0)
        for _ in range(3):
            waiter.cancel()
            await asyncio.sleep(0)
        assert not waiter.done()
        assert not native.cancelled()
        if mode == "complete":
            native.set_result(7)
            assert await waiter == 7
        elif mode == "native_timeout":
            native.set_exception(TimeoutError("native returned timeout"))
            with pytest.raises(TimeoutError, match="native returned timeout"):
                await waiter
        else:
            if mode == "cancelled_native":
                native.cancel()
            with pytest.raises(
                mooncake_mod.NativeExternalPageTransferUnknownError
            ) as caught:
                await waiter
            assert caught.value.terminal_future is native
            if mode == "unknown":
                assert not native.done()
                native.set_result(7)

    asyncio.run(run())
    assert not gc.isenabled()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("timeout", [0.0, 120.0])
def test_native_wait_handles_already_terminal_future(
    gc_disabled: None, outcome: str, timeout: float
) -> None:
    async def run() -> None:
        future = asyncio.get_running_loop().create_future()
        if outcome == "success":
            future.set_result(7)
            assert await mooncake_mod._await_native_task(future, timeout=timeout) == 7
        else:
            if outcome == "cancelled":
                future.cancel()
                expected_error = asyncio.CancelledError
            else:
                future.set_exception(RuntimeError("native failure"))
                expected_error = RuntimeError
            with pytest.raises(expected_error):
                await mooncake_mod._await_native_task(future, timeout=timeout)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["failure", "late_failure", "late_success"])
def test_remote_page_handoff_drops_failed_and_late_future_frames(
    gc_disabled: None, mode: str
) -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    release = threading.Event()
    refs = []
    releases = []

    class Page(_Page):
        def ref_count_down(self) -> None:
            super().ref_count_down()
            releases.append(1)

    async def retrieve(keys: list[Any]) -> list[Page]:
        page = Page("success")
        refs.append(weakref.ref(page))
        if mode.startswith("late"):
            while not release.is_set():
                await asyncio.sleep(0.001)
        if mode != "late_success":
            page.ref_count_down()
            raise RuntimeError("page retrieval failed")
        return [page]

    backend = object.__new__(RemoteBackend)
    backend.connection = SimpleNamespace(batched_get_layer_pages=retrieve)
    backend.loop = loop
    backend.config = SimpleNamespace(
        blocking_timeout_secs=0.01 if mode.startswith("late") else 3
    )
    backend._mla_worker_id_as0_mode = False

    async def settle() -> None:
        for _ in range(1000):
            if len(releases) == 20:
                # Let coroutine-to-future completion callbacks finish too.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return
            await asyncio.sleep(0.001)
        raise AssertionError("late page work did not settle")

    try:
        expected_error = TimeoutError if mode.startswith("late") else RuntimeError
        for _ in range(20):
            try:
                backend.batched_get_layer_pages([])
            except expected_error:
                pass
            else:
                pytest.fail("expected page retrieval failure or timeout")
        release.set()
        asyncio.run_coroutine_threadsafe(settle(), loop).result(3)
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)
        loop.close()
    assert len(releases) == 20
    assert len(refs) == 20
    assert all(ref() is None for ref in refs)
    assert not gc.isenabled()


def test_cold_coordinator_preserves_subclass_and_direct_base_poll() -> None:
    class Derived(adapter_mod.LMCacheConnectorV1Impl):
        def _drain_dsa_cold_load_futures(self):
            self.polls += 1
            return super()._drain_dsa_cold_load_futures()

    impl = object.__new__(Derived)
    impl.polls = 0
    coordinator = impl._get_cold_load_coordinator()
    assert impl._drain_dsa_cold_load_futures() is None
    assert impl.polls == 1
    assert coordinator.futures == {}
    assert adapter_mod.LMCacheConnectorV1Impl._drain_dsa_cold_load_futures(impl) is None
    assert impl.polls == 1


def test_idle_cold_coordinator_does_not_retain_adapter_without_gc(
    gc_disabled: None,
) -> None:
    impl = object.__new__(adapter_mod.LMCacheConnectorV1Impl)
    adapter_ref = weakref.ref(impl)
    coordinator = impl._get_cold_load_coordinator()
    assert impl._drain_dsa_cold_load_futures() is None
    del impl
    assert adapter_ref() is None
    assert coordinator.poll() is None
