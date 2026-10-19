# SPDX-License-Identifier: Apache-2.0
"""Real CPU subprocess/socket tests; fake I/O cannot establish NPU fencing."""

# Standard
from concurrent.futures import TimeoutError
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Queue
from typing import Any, Iterator
import json
import os
import socket
import subprocess
import sys
import threading
import time

# Third Party
import pytest
import zmq

# First Party
from lmcache.v1.rpc.storage_client import StorageClient, StorageUnavailableError
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    StorageResult,
    content_digest,
    decode_message,
    encode_message,
)
from lmcache.v1.storage_service import StorageService

NONCE = b"cpu-test-storage-session"


class ProbeRuntime:
    """CPU-only owner used to exercise transport ordering and shared bytes."""

    def __init__(self) -> None:
        self.slab = SharedMemory(create=True, size=64)
        self.slab.buf[:] = b"\0" * 64
        self.lock = threading.Lock()
        self.records: dict[tuple, dict] = {}
        self.allocations = 0
        self.quiescing = False

    def bootstrap_descriptor(self) -> dict:
        return {"slab_name": self.slab.name, "slab_size": self.slab.size}

    def is_healthy(self) -> bool:
        return True

    def quiesce(self) -> None:
        self.quiescing = True

    def handle(self, request: StorageRequest) -> StorageResult:
        key = request.identity.key
        method = request.method.lower()
        with self.lock:
            record = self.records.setdefault(
                key, {"cancelled": False, "phase": "UNKNOWN"}
            )
            if method in ("status", "operation_status"):
                return StorageResult(
                    request.identity, record["phase"], {"allocations": self.allocations}
                )
            if method == "cancel":
                record["cancelled"] = True
                if record["phase"] == "UNKNOWN":
                    record["phase"] = "CANCELLED_NOT_STARTED"
                return StorageResult(request.identity, record["phase"])
            if method == "retire":
                record["phase"] = "DONE"
                return StorageResult(request.identity, "DONE")
            if method == "acknowledge_terminal":
                return StorageResult(request.identity, "ACKNOWLEDGED")
            if method in ("health", "memory_ready", "registration_status"):
                return StorageResult(
                    request.identity, "HEALTHY", {"quiescing": self.quiescing}
                )
            if method == "stop":
                return StorageResult(
                    request.identity, "COMPLETED", {"status": "STOPPED"}
                )
            if record["cancelled"]:
                return StorageResult(request.identity, "CANCELLED_NOT_STARTED")
            if "result" in record:
                return record["result"]
            record["phase"] = "PREPARING"
        payload = request.payload or {}
        time.sleep(payload.get("delay", 0))
        with self.lock:
            if record["cancelled"]:
                record["phase"] = "CANCELLED_NOT_STARTED"
                return StorageResult(request.identity, record["phase"])
            if method == "prepare":
                value = payload.get("bytes", b"cpu bytes")
                self.slab.buf[: len(value)] = value
                self.allocations += 1
                value = {
                    "slab_name": self.slab.name,
                    "size": len(value),
                    "allocations": self.allocations,
                }
            else:
                value = payload
            record["phase"] = "EXPORTED"
            result = StorageResult(request.identity, "EXPORTED", value)
            record["result"] = result
            return result

    def close(self) -> None:
        self.slab.close()
        self.slab.unlink()


def _endpoint() -> str:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{reservation.getsockname()[1]}"


def _client(work: str, lifecycle: str, **kwargs: Any) -> StorageClient:
    return StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker-0",
        session_nonce=NONCE,
        **kwargs,
    )


def _request(client: StorageClient, method: str, payload: Any = None):
    return client.request(
        method,
        payload,
        request_id="req",
        request_attempt="attempt",
        load_generation=1,
        purpose="test",
    )


@contextmanager
def _process(
    tmp_path: Path, *, max_work: int = 2, watch_parent: bool = False
) -> Iterator[tuple]:
    work, lifecycle = _endpoint(), _endpoint()
    bootstrap = tmp_path / "storage-bootstrap.json"
    bootstrap.write_text(
        json.dumps(
            {
                "runtime_factory": "test_storage_service_process:ProbeRuntime",
                "work_endpoint": work,
                "lifecycle_endpoint": lifecycle,
                "cache_instance_id": "cache",
                "service_epoch": "epoch",
                "sessions": {"worker-0": NONCE.hex()},
                "max_work": max_work,
                "socket_hwm": 1,
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(__file__).parents[2]),
            str(Path(__file__).parent),
            env.get("PYTHONPATH", ""),
        )
    )
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.storage_service",
        "--bootstrap",
        str(bootstrap),
    ]
    if watch_parent:
        command.append("--watch-parent-stdin")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    ready: Queue = Queue()
    assert process.stdout is not None
    threading.Thread(
        target=lambda: ready.put(process.stdout.readline()), daemon=True
    ).start()
    try:
        line = ready.get(timeout=15)
        if not line:
            assert process.stderr is not None
            pytest.fail(f"Storage process exited: {process.stderr.read()}")
        descriptor = json.loads(line)
        assert descriptor["status"] == "READY"
        yield work, lifecycle, process, descriptor
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=10)


def test_real_process_shared_bytes_replay_and_terminal_ack(tmp_path: Path) -> None:
    with _process(tmp_path) as (work, lifecycle, process, ready):
        client = _client(work, lifecycle, max_pending=1)
        try:
            payload = {"bytes": b"exact shared CPU bytes"}
            pending = _request(client, "prepare", payload)
            payload["bytes"] = b"mutated caller container"
            result = pending.result(5)
            assert result.payload_digest == pending.payload_digest
            assert result.value["allocations"] == 1
            slab = SharedMemory(name=ready["bootstrap_descriptor"]["slab_name"])
            try:
                assert (
                    bytes(slab.buf[: result.value["size"]]) == b"exact shared CPU bytes"
                )
            finally:
                slab.close()
            assert client.replay(pending.identity) is pending
            assert (
                client.operation_status(pending.identity).result(2).value["allocations"]
                == 1
            )
            with pytest.raises(BufferError):
                _request(client, "lookup")
            client.retire(
                pending.identity, pending.payload_digest, "lease", "never_imported"
            ).result(2)
            client.acknowledge_terminal(pending.identity).result(2)
            stop = _request(client, "stop")
            assert stop.result(2).value["status"] == "STOPPED"
            assert (
                client.acknowledge_terminal(stop.identity).result(2).status
                == "ACKNOWLEDGED"
            )
            process.wait(timeout=5)
        finally:
            client.close()


def test_timeout_keeps_original_operation_and_cancellation_progress(
    tmp_path: Path,
) -> None:
    with _process(tmp_path, max_work=1) as (work, lifecycle, _, _):
        client = _client(work, lifecycle)
        try:
            pending = _request(client, "prepare", {"delay": 0.5})
            with pytest.raises(TimeoutError):
                pending.result(0.05)
            assert not pending.cancel()
            cancelled = client.cancel(pending.identity).result(0.3)
            assert cancelled.identity == pending.identity
            assert client.replay(pending.identity) is pending
            assert pending.result(2).status == "CANCELLED_NOT_STARTED"
            assert (
                client.operation_status(pending.identity).result(1).value["allocations"]
                == 0
            )
        finally:
            assert (
                client.close() is False
            )  # No acknowledgement means retained evidence.


def test_lifecycle_progresses_when_work_socket_cannot_send(tmp_path: Path) -> None:
    with _process(tmp_path) as (_, lifecycle, _, _):
        client = _client(_endpoint(), lifecycle, socket_hwm=1)
        try:
            pending = _request(client, "prepare", {"bytes": b"unsent"})
            cancelled = client.cancel(pending.identity).result(2)
            assert cancelled.status == "CANCELLED_NOT_STARTED"
            assert client.send_backpressure > 0
            with pytest.raises(TimeoutError):
                pending.result(0.05)
            assert (
                client.operation_status(pending.identity).result(2).value["allocations"]
                == 0
            )
        finally:
            client.close()


def test_cancel_before_work_arrival_is_not_resurrected(tmp_path: Path) -> None:
    with _process(tmp_path) as (work, lifecycle, _, _):
        context = zmq.Context()
        sockets = [context.socket(zmq.DEALER) for _ in range(2)]
        try:
            for sock, endpoint in zip(sockets, (work, lifecycle), strict=True):
                sock.setsockopt(zmq.RCVTIMEO, 2000)
                sock.connect(endpoint)
            identity = OperationIdentity(
                "cache", "epoch", "worker-0", 1, "req", "attempt", 1, "test"
            )
            payload = {"bytes": b"must not allocate"}
            digest = content_digest(("prepare", payload))
            cancel = StorageRequest(
                identity, "cancel", None, digest, NONCE, correlation_id=1
            )
            sockets[1].send_multipart(encode_message(cancel))
            assert (
                decode_message(sockets[1].recv_multipart(), StorageResult).status
                == "CANCELLED_NOT_STARTED"
            )
            sockets[0].send_multipart(
                encode_message(
                    StorageRequest(identity, "prepare", payload, digest, NONCE)
                )
            )
            assert (
                decode_message(sockets[0].recv_multipart(), StorageResult).status
                == "CANCELLED_NOT_STARTED"
            )
        finally:
            for sock in sockets:
                sock.close(linger=0)
            context.term()


def test_full_bulk_lane_preserves_lookup_and_lifecycle(tmp_path: Path) -> None:
    with _process(tmp_path, max_work=1) as (work, lifecycle, _, _):
        client = _client(work, lifecycle, socket_hwm=1)
        try:
            pending = _request(client, "prepare", {"delay": 1})
            deadline = time.monotonic() + 2
            while (
                client.operation_status(pending.identity).result(1).status
                != "PREPARING"
            ):
                assert time.monotonic() < deadline
            other = _request(client, "prepare")
            assert other.result(0.5).status == "BUSY"
            assert _request(client, "lookup", {"metadata": True}).result(0.5).value == {
                "metadata": True
            }
            assert (
                client.operation_status(pending.identity).result(0.5).status
                == "PREPARING"
            )
            assert pending.result(2).status == "EXPORTED"
        finally:
            client.close()


def test_parent_eof_quiesces_without_unlinking_slab(tmp_path: Path) -> None:
    with _process(tmp_path, watch_parent=True) as (work, lifecycle, process, ready):
        client = _client(work, lifecycle)
        try:
            assert process.stdin is not None
            process.stdin.close()
            process.stdin = None
            deadline = time.monotonic() + 2
            while not _request(client, "health").result(1).value["quiescing"]:
                assert time.monotonic() < deadline
            assert _request(client, "prepare").result(1).status == "BUSY"
            slab = SharedMemory(name=ready["bootstrap_descriptor"]["slab_name"])
            slab.close()
            assert process.poll() is None
        finally:
            client.close()


def test_callback_cannot_block_io_or_drop_retirement(tmp_path: Path) -> None:
    with _process(tmp_path) as (work, lifecycle, _, _):
        client = _client(work, lifecycle)
        release = threading.Event()
        entered = threading.Event()
        try:
            pending = _request(client, "prepare", {"delay": 0.05})

            def blocked_callback(_future) -> None:
                entered.set()
                release.wait(3)

            pending.add_done_callback(blocked_callback)
            assert entered.wait(2)
            retirement = client.retire(
                pending.identity, pending.payload_digest, "lease", "never_imported"
            )
            status = client.operation_status(pending.identity)
            # Both responses can be received while completion callbacks are paused.
            time.sleep(0.1)
            assert not retirement.done()
            release.set()
            assert retirement.result(1).status == "DONE"
            assert status.result(1).status == "DONE"
        finally:
            release.set()
            client.close()


def test_slow_work_result_reader_preserves_lifecycle() -> None:
    runtime = ProbeRuntime()
    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"worker-0": NONCE},
        max_work=2,
        socket_hwm=1,
    )
    service.start()
    context = zmq.Context()
    slow = context.socket(zmq.DEALER)
    slow.setsockopt(zmq.RCVHWM, 1)
    slow.setsockopt(zmq.SNDHWM, 32)
    slow.connect(work)
    client = _client(_endpoint(), lifecycle)
    try:
        for operation in range(1, 13):
            identity = OperationIdentity(
                "cache", "epoch", "worker-0", operation, "req", "attempt", 1, "echo"
            )
            payload = {"bytes": b"x" * (1024 * 1024)}
            slow.send_multipart(
                encode_message(
                    StorageRequest(
                        identity,
                        "echo",
                        payload,
                        content_digest(("echo", payload)),
                        NONCE,
                    )
                )
            )
        pending = _request(client, "prepare")
        deadline = time.monotonic() + 3
        while service.send_backpressure == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert client.cancel(pending.identity).result(1).status in (
            "PREPARING",
            "CANCELLED_NOT_STARTED",
            "EXPORTED",
        )
    finally:
        client.close()
        # Drain queued replies so test shutdown can release its fake CPU-only slab.
        slow.setsockopt(zmq.RCVTIMEO, 200)
        while True:
            try:
                slow.recv_multipart()
            except zmq.Again:
                break
        service.close(5)
        slow.close(linger=0)
        context.term()


def test_service_death_fails_future_without_discarding_ownership(
    tmp_path: Path,
) -> None:
    with _process(tmp_path) as (work, lifecycle, process, _):
        client = _client(work, lifecycle, failure_timeout_s=0.3)
        try:
            pending = _request(client, "prepare", {"delay": 5})
            process.kill()
            process.wait(timeout=5)
            with pytest.raises(StorageUnavailableError, match="completion unknown"):
                pending.result(2)
            assert not client.is_healthy()
            assert client.close() is False
        finally:
            client.close()


def test_peer_retirement_uses_own_nonce_and_full_operation_identity() -> None:
    from lmcache.v1.cpu_cache import StorageOperationRegistry

    sessions = {"worker-0": NONCE, "worker-1": b"peer-nonce"}
    released = []
    registry = StorageOperationRegistry("cache", "epoch", sessions)

    class RegistryRuntime:
        def handle(self, request: StorageRequest) -> StorageResult:
            def dispatch(work: StorageRequest) -> dict:
                if work.method != "prepare":
                    return {"metadata": True}
                lease = registry.add_lease(
                    work.identity,
                    (object(),),
                    participants=sessions,
                    release=lambda objects: released.append(objects),
                )
                return {"lease_id": lease}

            return registry.handle(request, dispatch)

        def close(self) -> None:
            assert registry.snapshot()["leases"] == 0

        def is_healthy(self) -> bool:
            return registry.is_healthy()

    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        RegistryRuntime(),
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions=sessions,
    )
    service.start()
    owner = _client(work, lifecycle)
    peer = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker-1",
        session_nonce=b"peer-nonce",
    )
    try:
        source = _request(owner, "prepare")
        lease_id = source.result(2).value["lease_id"]
        # Both sessions have operation 1; peer controls still target the owner's ID.
        own_peer_work = _request(peer, "lookup")
        assert own_peer_work.result(2).status == "COMPLETED"
        owner.retire(
            source.identity, source.payload_digest, lease_id, "never_imported"
        ).result(2)
        assert not released
        assert owner.acknowledge_terminal(source.identity).result(2).status == "PENDING"
        peer.retire(source.identity, source.payload_digest, lease_id, "fenced").result(
            2
        )
        assert len(released) == 1
        assert (
            owner.acknowledge_terminal(source.identity).result(2).status
            == "ACKNOWLEDGED"
        )
        assert (
            peer.acknowledge_terminal(own_peer_work.identity).result(2).status
            == "ACKNOWLEDGED"
        )
    finally:
        owner.close()
        peer.close()
        service.close()


def test_stale_or_wrong_digest_reply_cannot_resolve_future() -> None:
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.RCVTIMEO, 2000)
    work = _endpoint()
    router.bind(work)
    client = _client(work, _endpoint())
    try:
        pending = _request(client, "prepare")
        route, *frames = router.recv_multipart()
        request = decode_message(frames, StorageRequest)
        wrong = StorageResult(request.identity, "EXPORTED", payload_digest=b"wrong")
        router.send_multipart([route, *encode_message(wrong)])
        wrong = StorageResult(
            request.identity,
            "EXPORTED",
            payload_digest=request.payload_digest,
            correlation_id=1,
        )
        router.send_multipart([route, *encode_message(wrong)])
        with pytest.raises(TimeoutError):
            pending.result(0.05)
        valid = StorageResult(
            request.identity, "EXPORTED", payload_digest=request.payload_digest
        )
        router.send_multipart([route, *encode_message(valid)])
        assert pending.result(2) == valid
        assert client.ignored_responses == 2
    finally:
        client.close()
        router.close(linger=0)
        context.term()


def test_refused_runtime_drain_keeps_service_and_slab_alive() -> None:
    class HeldRuntime(ProbeRuntime):
        held = True

        def close(self) -> None:
            if self.held:
                raise RuntimeError("Native producer completion is unknown")
            super().close()

    runtime = HeldRuntime()
    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"worker-0": NONCE},
    )
    service.start()
    client = _client(work, lifecycle)
    try:
        with pytest.raises(TimeoutError, match="unresolved"):
            service.close(0.05)
        assert service.drain_error is not None
        assert _request(client, "health").result(1).status == "HEALTHY"
        slab = SharedMemory(name=runtime.slab.name)
        slab.close()
    finally:
        runtime.held = False
        client.close()
        service.close()


def test_cached_runtime_fatal_reaches_idle_client_once_without_rpc() -> None:
    class UnhealthyRuntime(ProbeRuntime):
        healthy = True

        def is_healthy(self) -> bool:
            return self.healthy

    runtime = UnhealthyRuntime()
    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"worker-0": NONCE},
    )
    service.start()
    client = _client(work, lifecycle)
    notices = []
    stopped = threading.Event()

    def failed(reason: str) -> None:
        notices.append(reason)
        client.mark_fatal(reason)  # Worker context's callback is reentrant.
        stopped.set()

    client.set_failure_callback(failed)
    try:
        assert _request(client, "health").result(2).status == "HEALTHY"
        runtime.healthy = False
        assert stopped.wait(2)
        assert not client.is_healthy()
        time.sleep(0.05)
        assert len(notices) == 1
        slab = SharedMemory(name=runtime.slab.name)
        slab.close()  # Fatal notification did not close M's retained mapping.
    finally:
        client.close()
        service.close()


def test_forged_fatal_notice_is_rejected_and_close_does_not_notify() -> None:
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.RCVTIMEO, 2000)
    work = _endpoint()
    router.bind(work)
    client = _client(work, _endpoint())
    notices = []
    client.set_failure_callback(notices.append)
    try:
        _request(client, "prepare")
        route, *_ = router.recv_multipart()
        identity = OperationIdentity(
            "cache", "epoch", "worker-0", 0, "session-health", "worker-0", 0, "fatal"
        )
        forged = StorageResult(
            identity, "FATAL", error="forged", session_nonce=b"wrong"
        )
        router.send_multipart([route, *encode_message(forged)])
        time.sleep(0.05)
        assert client.is_healthy()
        assert not notices
    finally:
        client.close()
        assert not notices
        router.close(linger=0)
        context.term()
