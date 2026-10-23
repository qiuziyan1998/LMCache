# SPDX-License-Identifier: Apache-2.0
"""Real IPC/shared-memory coverage; native SDK operations are CPU stand-ins."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace
import ctypes
import gc
import json
import os
import socket
import subprocess
import sys
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.rpc.mooncake_transport import (
    RemoteMooncakeStore,
    WorkerMooncakeTransport,
)
from lmcache.v1.rpc.storage_protocol import content_digest, pack_uints
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.storage_service import StorageService

NONCE = b"native-test"


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


class Store:
    def __init__(self):
        self.values = {}
        self.calls = []
        self.entered, self.release = threading.Event(), threading.Event()

    def get_hostname(self):
        return "host"

    def batch_is_exist(self, keys):
        return [int(key in self.values) for key in keys]

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, replica):
        self.calls.append((keys, ptrs, sizes, replica.preferred_segment))
        for key, row, lengths in zip(keys, ptrs, sizes, strict=True):
            self.values[key] = b"".join(
                ctypes.string_at(p, n) for p, n in zip(row, lengths, strict=True)
            )
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        if "blocked" in keys:
            self.entered.set()
            assert self.release.wait(10)
        results = []
        for key, row, lengths in zip(keys, ptrs, sizes, strict=True):
            value = self.values.get(key)
            if value is None:
                results.append(-1)
                continue
            assert len(value) <= sum(lengths)
            start = 0
            for ptr, size in zip(row, lengths, strict=True):
                chunk = value[start : start + size]
                ctypes.memmove(ptr, chunk, len(chunk))
                start += size
            results.append(len(value))
        return results

    def batch_get_into(self, keys, ptrs, sizes):
        return self.batch_get_into_multi_buffers(
            keys, [[p] for p in ptrs], [[n] for n in sizes]
        )

    def put_parts(self, key, header, view):
        self.values[key] = header + bytes(view)
        return 0


class ProbeTransport(WorkerMooncakeTransport):
    def __init__(self, runtime_config):
        self.slab = SharedMemory(name=runtime_config["name"])
        mapping = SimpleNamespace(
            ptr=ctypes.addressof(ctypes.c_char.from_buffer(self.slab.buf)),
            size=self.slab.size,
            shm_name=self.slab.name,
            generation=1,
        )
        super().__init__(
            Store(),
            mapping,
            instance="cache",
            epoch="epoch",
            nonce=NONCE,
            session="host:123",
            replica_factory=SimpleNamespace,
            max_bytes=self.slab.size,
            close_native=self.slab.close,
        )

    def bootstrap_descriptor(self):
        return self.descriptor()

    def _execute(self, request):
        if request.method == "stop":
            return {"status": "STOPPED"}
        return super()._execute(request)


@contextmanager
def transport(tmp_path, *, child=False, max_work=16, slab_size=8192):
    slab = SharedMemory(create=True, size=slab_size)
    second = SharedMemory(name=slab.name)
    work, life = endpoint(), endpoint()
    info = dict(work=work, lifecycle=life, nonce=NONCE.hex())
    base = ctypes.addressof(ctypes.c_char.from_buffer(slab.buf))
    worker = None
    process = None
    service = None
    remote = None
    try:
        if child:
            config = dict(
                runtime_factory="tests.v1.test_worker_mooncake_transport:ProbeTransport",
                runtime_config=dict(name=slab.name),
                work_endpoint=work,
                lifecycle_endpoint=life,
                cache_instance_id="cache",
                service_epoch="epoch",
                sessions={"storage-owner": NONCE.hex()},
            )
            filename = tmp_path / "native.json"
            filename.write_text(json.dumps(config))
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2]))
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lmcache.v1.storage_service",
                    "--bootstrap",
                    str(filename),
                    "--watch-parent-stdin",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            from queue import Queue

            ready = Queue()
            threading.Thread(
                target=lambda: ready.put(process.stdout.readline()), daemon=True
            ).start()
            line = ready.get(timeout=15)
            assert line, process.stderr.read()
            registration = json.loads(line)["bootstrap_descriptor"]
        else:
            mapping = SimpleNamespace(
                ptr=ctypes.addressof(ctypes.c_char.from_buffer(second.buf)),
                size=second.size,
                shm_name=second.name,
                generation=1,
            )
            assert mapping.ptr != base
            worker = WorkerMooncakeTransport(
                Store(),
                mapping,
                instance="cache",
                epoch="epoch",
                nonce=NONCE,
                session="host:123",
                replica_factory=SimpleNamespace,
                max_bytes=slab.size,
                close_native=lambda: None,
            )
            service = StorageService(
                worker,
                work,
                life,
                cache_instance_id="cache",
                service_epoch="epoch",
                sessions={"storage-owner": NONCE},
                max_work=max_work,
            )
            service.start()
            registration = worker.descriptor()
        remote = RemoteMooncakeStore(
            info,
            instance="cache",
            epoch="epoch",
            base=base,
            size=slab.size,
            generation=1,
            timeout=5,
            max_bytes=slab.size,
        )
        remote.bind(registration, slab.name)
        yield remote, worker, slab
    finally:
        if worker is not None:
            worker.store.release.set()
        if process is not None and remote is not None and process.poll() is None:
            future = remote.client.request(
                "stop",
                {},
                request_id="stop",
                request_attempt="epoch",
                load_generation=1,
                purpose="stop",
            )
            result = future.result(3)
            assert result.value == {"status": "STOPPED"}
            remote.client.acknowledge_terminal(
                future.identity,
                {
                    "terminal_result_digest": content_digest(
                        (result.status, result.value, result.error)
                    )
                },
            ).result(3)
        if remote is not None:
            remote.client.close()
        if service is not None:
            service.close()
        if process is not None:
            process.stdin.close()
            try:
                process.wait(timeout=10)
                assert process.returncode == 0, process.stderr.read()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
        second.close()
        slab.close()
        slab.unlink()


@pytest.mark.parametrize("child", [False, True])
def test_shared_payload_roundtrip_and_partial_ranges(tmp_path, child):
    with transport(tmp_path, child=child) as (remote, worker, slab):
        slab.buf[:16] = b"0123456789abcdef"
        replica = SimpleNamespace(preferred_segment="destination")
        base = remote.base
        assert remote.batch_put_from_multi_buffers(
            ["full", "tail"],
            [[base, base + 4], [base + 8, base + 12]],
            [[4, 4], [2, 4]],
            replica,
        ) == [0, 0]
        assert remote.batch_is_exist(["full", "missing", "tail"]) == [1, 0, 1]
        assert remote.batch_get_into_multi_buffers(
            ["full", "tail", "missing"],
            [[base + 32], [base + 48, base + 52], [base + 64]],
            [[8], [2, 4], [8]],
        ) == [8, 6, -1]
        assert bytes(slab.buf[32:40]) == b"01234567"
        assert bytes(slab.buf[48:50]) == b"89"
        assert bytes(slab.buf[52:56]) == b"cdef"
        assert (
            remote.export_address(base + 7, 8) == remote.registration["host_base"] + 7
        )
        if worker:
            assert worker.store.calls[0][2] == [[8], [2, 4]]  # contiguous same-key only
            assert worker.registry.snapshot()["operations"] == 0


@pytest.mark.parametrize("offset,size", [(-1, 1), (8191, 2), (8192, 1), (0, 0)])
def test_bad_operands_rejected_before_native_submission(tmp_path, offset, size):
    with transport(tmp_path) as (remote, worker, slab):
        with pytest.raises(ValueError):
            remote.batch_get_into(["bad"], [remote.base + offset], [size])
        assert worker.registry.snapshot()["operations"] == 0
        assert worker.store.calls == []


def test_blocked_transfer_keeps_lifecycle_and_lookup_serviceable(tmp_path):
    with transport(tmp_path) as (remote, worker, slab), ThreadPoolExecutor(1) as pool:
        worker.store.values["blocked"] = b"ok"
        job = pool.submit(remote.batch_get_into, ["blocked"], [remote.base], [2])
        assert worker.store.entered.wait(2)
        assert remote.batch_is_exist(["blocked"]) == [1]
        assert not job.done()
        worker.store.release.set()
        assert job.result(2) == [2]
        assert bytes(slab.buf[:2]) == b"ok"


def test_timeout_retains_physical_owner_until_explicit_recovery(tmp_path):
    class Owner:
        refs = 1

        def ref_count_up(self):
            self.refs += 1

        def ref_count_down(self):
            self.refs -= 1

    with transport(tmp_path) as (remote, worker, slab):
        owner = Owner()
        remote.timeout = 0.1
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with pytest.raises(NativeExternalPageTransferUnknownError):
                remote.invoke(
                    "batch_get_into", (["blocked"], [remote.base], [8]), [owner]
                )
            assert owner.refs == 2
            with pytest.raises(RuntimeError, match="unknown"):
                remote.close()
            worker.store.release.set()
        finally:
            if was_enabled:
                gc.enable()


def test_stale_registration_refused_and_metadata_payload_not_marshaled(tmp_path):
    with transport(tmp_path) as (remote, worker, slab):
        with pytest.raises(ValueError):
            remote.bind(dict(remote.registration, slab_generation=2), slab.name)
        slab.buf[:4] = b"data"
        owner = SimpleNamespace(data_ptr=remote.base)
        assert (
            remote.invoke(
                "put_parts", ("meta", b"header", memoryview(slab.buf)[:4]), [owner]
            )
            == 0
        )
        assert worker.store.values["meta"] == b"headerdata"
        assert remote.batch_get_into(["meta"], [remote.base + 64], [10]) == [10]
        assert bytes(slab.buf[64:74]) == b"headerdata"


def test_metadata_connector_uses_shared_destinations(tmp_path, monkeypatch):
    import torch
    from lmcache.v1.memory_management import TensorMemoryAllocator
    from tests.v1.storage_backend.test_mooncake_store_completion import (
        _make_mooncake_connector,
        _key,
    )

    with transport(tmp_path) as (remote, worker, slab):
        allocator = TensorMemoryAllocator(
            torch.frombuffer(slab.buf, dtype=torch.uint8), align_bytes=64
        )
        connector, loop = _make_mooncake_connector(
            monkeypatch, {}, [], native_io=remote
        )
        connector.local_cpu_backend = SimpleNamespace(allocate=allocator.allocate)
        source = allocator.allocate(
            connector.meta_shapes, connector.meta_dtypes, connector.meta_fmt
        )
        source.tensor.fill_(3)
        try:
            loop.run_until_complete(
                connector._put_with_metadata(_key(1).to_string(), source)
            )
            restored = loop.run_until_complete(
                connector._batch_get_buffer([_key(1), _key(2)])
            )
            assert restored[1] is None
            torch.testing.assert_close(restored[0].tensor, source.tensor)
            restored[0].ref_count_down()
        finally:
            source.ref_count_down()
            loop.run_until_complete(connector.close())
            loop.close()
            del source, restored, allocator, connector


def test_worker_rejects_invalid_wire_bounds_and_replay_is_idempotent(tmp_path):
    with transport(tmp_path) as (remote, worker, slab):
        # Bypass the M-side range checks to exercise the actual W boundary.
        for generation, offset in [(2, 0), (1, slab.size)]:
            payload = dict(
                keys=["x"],
                generation=generation,
                api="batch_get_into",
                counts=[1],
                offsets=pack_uints([offset]),
                sizes=pack_uints([1]),
            )
            future = remote.client.request(
                "native_get",
                payload,
                request_id="test",
                request_attempt="epoch",
                load_generation=1,
                purpose="test",
            )
            result = future.result(2)
            assert result.status == "FAILED_SAFE"
            remote.client.acknowledge_terminal(
                future.identity,
                {
                    "terminal_result_digest": content_digest(
                        (result.status, result.value, result.error)
                    )
                },
            ).result(2)
        slab.buf[:4] = b"once"
        payload = dict(
            keys=["x"],
            generation=1,
            api="batch_put_from_multi_buffers",
            counts=[1],
            offsets=pack_uints([0]),
            sizes=pack_uints([4]),
            segment="host",
        )
        future = remote.client.request(
            "native_put",
            payload,
            request_id="test",
            request_attempt="epoch",
            load_generation=1,
            purpose="test",
        )
        result = future.result(2)
        # Replay the same public request without a second SDK call.
        from lmcache.v1.rpc.storage_protocol import StorageRequest

        request = StorageRequest(
            future.identity,
            "native_put",
            payload,
            content_digest(("native_put", payload)),
            NONCE,
        )
        assert worker.handle(request).status == "COMPLETED"
        assert len(worker.store.calls) == 1
        remote.client.acknowledge_terminal(
            future.identity,
            {
                "terminal_result_digest": content_digest(
                    (result.status, result.value, result.error)
                )
            },
        ).result(2)
        assert worker.registry.snapshot()["operations"] == 0


def test_proxy_shutdown_has_no_gc_dependent_client_cycle(tmp_path):
    with transport(tmp_path) as (remote, worker, slab):
        remote.batch_is_exist(["x"])
        remote.shutdown()
        # The context manager itself retains remote; verify the callback is weak.
        callback = remote.client._failure_callback
        assert all(cell.cell_contents is not remote for cell in callback.__closure__)


def test_busy_job_is_cancelled_and_acked_without_native_submission(tmp_path):
    with (
        transport(tmp_path, max_work=1) as (remote, worker, slab),
        ThreadPoolExecutor(1) as pool,
    ):
        worker.store.values["blocked"] = b"ok"
        job = pool.submit(remote.batch_get_into, ["blocked"], [remote.base], [2])
        assert worker.store.entered.wait(2)
        try:
            with pytest.raises(ValueError):
                remote.batch_get_into(["not-submitted"], [remote.base + 4], [2])
            assert remote.is_healthy()
        finally:
            worker.store.release.set()
        assert job.result(2) == [2]
        assert worker.registry.snapshot()["operations"] == 0


def test_queued_job_does_not_submit_after_native_uncertainty(tmp_path):
    from threading import BoundedSemaphore, Event
    from unittest.mock import Mock

    class ObservedSlot:
        def __init__(self):
            self.slot = BoundedSemaphore(1)
            self.waiting = Event()
            self.enters = 0

        def __enter__(self):
            self.enters += 1
            if self.enters == 2:
                self.waiting.set()
            self.slot.acquire()

        def __exit__(self, *args):
            self.slot.release()

    with transport(tmp_path) as (remote, worker, slab), ThreadPoolExecutor(2) as pool:
        remote.timeout = 0.3
        remote._slots = slot = ObservedSlot()
        worker.store.batch_get_into = Mock(wraps=worker.store.batch_get_into)
        first = pool.submit(remote.batch_get_into, ["blocked"], [remote.base], [1])
        assert worker.store.entered.wait(2)
        second = pool.submit(remote.batch_get_into, ["later"], [remote.base + 8], [1])
        assert slot.waiting.wait(2)
        try:
            with pytest.raises(NativeExternalPageTransferUnknownError):
                first.result(2)
            with pytest.raises(RuntimeError) as caught:
                second.result(2)
            assert not isinstance(caught.value, NativeExternalPageTransferUnknownError)
        finally:
            worker.store.release.set()
        # Drain the real W executor before checking which SDK calls occurred.
        remote.client.close()
        assert worker.store.batch_get_into.call_count == 1


def test_split_native_reply_cardinality_is_checked_before_concatenation(tmp_path):
    with transport(tmp_path) as (remote, worker, slab):
        calls = []

        def short_reply(keys, ptrs, sizes):
            calls.append(len(keys))
            return [1] * (len(keys) - 1 if len(calls) == 1 else len(keys))

        worker.store.batch_get_into = short_reply
        with pytest.raises(ValueError, match="result count"):
            remote.batch_get_into(
                [str(i) for i in range(257)],
                [remote.base + i for i in range(257)],
                [1] * 257,
            )
        assert calls == [256]


@pytest.mark.parametrize("indexer", [False, True])
def test_exact_layer_pages_cross_native_boundary_without_padding(tmp_path, indexer):
    import torch
    from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator

    with transport(tmp_path, slab_size=1024 * 1024) as (remote, worker, slab):
        raw = torch.frombuffer(slab.buf, dtype=torch.uint8)
        raw.fill_(0xA5)
        allocator = TensorMemoryAllocator(raw)
        widths = ([256] * 6 + [130] * 16) if indexer else [576]
        count = 22 if indexer else 79
        dtype = torch.uint8 if indexer else torch.bfloat16
        pages = allocator.batched_allocate_layer_pages(
            [torch.Size([4, n]) for n in widths],
            [dtype] * len(widths),
            2,
            count,
            MemoryFormat.KV_MLA_FMT,
            valid_tokens=[4, 1],
            full_tokens=4,
        )
        assert pages is not None
        try:
            pointers = [
                [page.layer_data_ptr(i) for i in range(count)] for page in pages
            ]
            sizes = [[page.layer_size_bytes(i) for i in range(count)] for page in pages]
            for index, page in enumerate(pages):
                worker.store.values[str(index)] = (
                    bytes([0x30 + index]) * page.get_size()
                )
            assert remote.invoke(
                "batch_get_into_multi_buffers", (["0", "1"], pointers, sizes), pages
            ) == [p.get_size() for p in pages]
            for index, page in enumerate(pages):
                begin = page.data_ptr - remote.base
                assert bool(
                    (raw[begin : begin + page.get_size()] == 0x30 + index).all()
                )
                assert bool(
                    (
                        raw[begin + page.get_size() : begin + page.get_physical_size()]
                        == 0xA5
                    ).all()
                )
                assert page.get_ref_count() == 1
        finally:
            for page in pages:
                page.ref_count_down()
            assert allocator.num_active_allocations == 0
            del page, pages, allocator, raw


def test_metadata_get_known_failure_drops_task_cycle_with_gc_disabled(
    tmp_path, monkeypatch
):
    import asyncio
    from weakref import ref
    import torch
    from lmcache.v1.memory_management import TensorMemoryAllocator
    from tests.v1.storage_backend.test_mooncake_store_completion import (
        _make_mooncake_connector,
        _key,
    )

    with transport(tmp_path) as (remote, worker, slab):
        allocator = TensorMemoryAllocator(
            torch.frombuffer(slab.buf, dtype=torch.uint8), align_bytes=64
        )
        connector, loop = _make_mooncake_connector(
            monkeypatch, {}, [], native_io=remote
        )
        references = []

        def allocate(*args):
            obj = allocator.allocate(*args)
            references.append(ref(obj))
            return obj

        connector.local_cpu_backend = SimpleNamespace(allocate=allocate)
        # SDK completion is known, but a malformed reply is a safe load failure.
        worker.store.batch_get_into = lambda *args: []

        async def load():
            try:
                await connector._batch_get_buffer([_key(1)])
            except ValueError:
                pass

        enabled = gc.isenabled()
        gc.disable()
        try:
            asyncio.run(load())
            assert allocator.num_active_allocations == 0
            assert all(item() is None for item in references)
        finally:
            asyncio.run(connector.close())
            loop.close()
            if enabled:
                gc.enable()
            # Only clear failed-test cycles after checking GC-off behavior above.
            gc.collect()
            connector = allocator = None
