# SPDX-License-Identifier: Apache-2.0
"""Real TCP event-driven service tests; device/native ordering is not exercised."""

# Standard
from collections.abc import Callable
from contextlib import contextmanager
from threading import Event, current_thread
from typing import Any, Iterator
import time

# Third Party
import pytest
import zmq

# First Party
from lmcache.v1.cpu_cache import StorageOperationRegistry
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    StorageResult,
    content_digest,
    decode_message,
    encode_message,
)
from lmcache.v1.storage_service import StorageService

# Local
from tests.v1.test_storage_service_process import NONCE, _client, _endpoint, _request


def eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    """Wait for observed transport progress without issuing additional RPCs."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.002)


class Runtime:
    """Controllable CPU-only runtime with the complete notification contract."""

    def __init__(self) -> None:
        self.notifier: Callable[[], None] | None = None
        self.healthy = True
        self.entered, self.release = Event(), Event()
        self.closed = 0

    def set_transport_notifier(self, notifier: Callable[[], None] | None) -> bool:
        """Install the transport's nonblocking signal callback."""
        self.notifier = notifier
        return True

    def handle(self, request: StorageRequest) -> StorageResult:
        """Return values directly, optionally holding preparation until released."""
        if request.method == "prepare" and (request.payload or {}).get("hold"):
            self.entered.set()
            assert self.release.wait(5)
        return StorageResult(request.identity, "COMPLETED", request.payload)

    def is_healthy(self) -> bool:
        """Return the cached health state without external work."""
        return self.healthy

    def close(self) -> None:
        """Record closure after every admitted job and response has drained."""
        self.closed += 1


@contextmanager
def serving(runtime: Any, **kwargs: Any) -> Iterator[tuple[StorageService, str, str]]:
    """Own one real service and keep its worker alive through explicit drain."""
    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"worker-0": NONCE},
        **kwargs,
    )
    service.start()
    try:
        yield service, work, lifecycle
    finally:
        runtime.release.set()
        service.close()


@pytest.fixture
def poll_trace(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Observe actual service poll masks/timeouts without replacing polling."""
    original = zmq.Poller
    trace: dict[str, Any] = {"timeouts": [], "disabled": Event(), "blocking": Event()}

    class ObservedPoller(original):
        def register(self, socket: Any, flags: int = zmq.POLLIN | zmq.POLLOUT) -> None:
            if current_thread().name == "storage-service-io" and flags == 0:
                trace["disabled"].set()
            super().register(socket, flags)

        def poll(self, timeout: int | None = None) -> Any:
            if current_thread().name == "storage-service-io":
                trace["timeouts"].append(timeout)
                if timeout is None:
                    trace["blocking"].set()
            return super().poll(timeout)

    monkeypatch.setattr(zmq, "Poller", ObservedPoller)
    return trace


def test_job_completion_and_close_interrupt_unbounded_poll(poll_trace: dict) -> None:
    runtime = Runtime()
    with serving(runtime) as (_, work, lifecycle):
        client = _client(work, lifecycle)
        try:
            pending = _request(client, "prepare", {"hold": True})
            assert runtime.entered.wait(2)
            poll_trace["blocking"].clear()
            # A lifecycle exchange places the socket owner back in an indefinite
            # poll after the held job has been inserted into its jobs map.
            assert _request(client, "health").result(2).status == "COMPLETED"
            assert poll_trace["blocking"].wait(2)
            runtime.release.set()
            assert pending.result(1).value == {"hold": True}
            assert None in poll_trace["timeouts"]
        finally:
            client.close()
    assert runtime.closed == 1
    assert runtime.notifier is None


def test_immediate_jobs_and_budget_leftovers_need_no_timer(poll_trace: dict) -> None:
    runtime = Runtime()
    with serving(runtime, max_work=64) as (_, work, lifecycle):
        client = _client(work, lifecycle, max_pending=96)
        try:
            pending = [_request(client, "lookup", {"item": i}) for i in range(64)]
            assert [future.result(3).value["item"] for future in pending] == list(
                range(64)
            )
            assert None in poll_trace["timeouts"]
            # Healthy notifier-aware operation uses no periodic five-ms health poll.
            assert 5 not in poll_trace["timeouts"]
        finally:
            client.close()


def test_idle_fatal_notification_interrupts_unbounded_poll(poll_trace: dict) -> None:
    runtime = Runtime()
    with serving(runtime) as (_, work, lifecycle):
        client = _client(work, lifecycle)
        try:
            assert _request(client, "health").result(2).status == "COMPLETED"
            assert poll_trace["blocking"].wait(2)
            runtime.healthy = False
            assert runtime.notifier is not None
            runtime.notifier()
            eventually(lambda: not client.is_healthy())
        finally:
            client.close()


def test_custom_runtime_without_notifier_retains_health_polling(
    poll_trace: dict,
) -> None:
    runtime = Runtime()
    runtime.set_transport_notifier = None  # type: ignore[method-assign]
    with serving(runtime) as (_, work, lifecycle):
        client = _client(work, lifecycle)
        try:
            assert _request(client, "health").result(2).status == "COMPLETED"
            runtime.healthy = False
            eventually(lambda: not client.is_healthy())
            assert any(
                timeout is not None and timeout > 0
                for timeout in poll_trace["timeouts"]
            )
        finally:
            client.close()


def test_terminal_release_notifies_idle_owner_without_another_rpc() -> None:
    registry = StorageOperationRegistry("cache", "epoch", {"worker-0": NONCE})
    released = []

    class LeaseRuntime(Runtime):
        def set_transport_notifier(self, notifier: Callable[[], None] | None) -> bool:
            return registry.set_transport_notifier(notifier)

        def handle(self, request: StorageRequest) -> StorageResult:
            def dispatch(work: StorageRequest) -> str:
                return registry.add_lease(
                    work.identity,
                    (object(),),
                    participants=("worker-0",),
                    release=lambda objects: released.append(objects),
                    producer_holds=1,
                )

            return registry.handle(request, dispatch)

        def drain_terminal_notifications(self) -> list[StorageResult]:
            return registry.drain_terminal_notifications()

    runtime = LeaseRuntime()
    with serving(runtime) as (_, work, lifecycle):
        client = _client(work, lifecycle)
        try:
            source = _request(client, "prepare", {})
            lease_id = source.result(2).value
            assert (
                client.retire(
                    source.identity, source.payload_digest, lease_id, "fenced"
                )
                .result(2)
                .status
                == "RETIRED"
            )
            assert (
                client.acknowledge_terminal(source.identity).result(2).status
                == "PENDING"
            )
            assert registry.snapshot()["leases"] == 1
            registry.producer_done(lease_id, known_complete=True)
            eventually(lambda: registry.snapshot()["operations"] == 0)
            assert len(released) == 1
        finally:
            client.close()


def send_raw(socket: zmq.Socket, operation: int, size: int) -> None:
    """Send a frozen value to a deliberately backpressured ROUTER destination."""
    identity = OperationIdentity(
        "cache", "epoch", "worker-0", operation, "req", "attempt", 1, "test"
    )
    payload = {"bytes": b"x" * size}
    socket.send_multipart(
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


def test_blocked_route_does_not_starve_another_destination(poll_trace: dict) -> None:
    runtime = Runtime()
    with serving(runtime, max_work=16, socket_hwm=1) as (service, work, lifecycle):
        context = zmq.Context()
        slow = context.socket(zmq.DEALER)
        slow.setsockopt(zmq.RCVHWM, 1)
        slow.connect(work)
        client = _client(work, lifecycle)
        try:
            for operation in range(100, 108):
                send_raw(slow, operation, 1024 * 1024)
            eventually(lambda: service.send_backpressure > 0)
            assert _request(client, "lookup", {"fast": True}).result(2).value == {
                "fast": True
            }
            before = len(poll_trace["timeouts"])
            time.sleep(0.1)
            # One retry per blocked route per five milliseconds, rather than a
            # writable-ROUTER spin because the other destination is writable.
            assert len(poll_trace["timeouts"]) - before < 100
        finally:
            slow.setsockopt(zmq.RCVTIMEO, 200)
            while True:
                try:
                    slow.recv_multipart()
                except zmq.Again:
                    break
            client.close()
            slow.close(linger=0)
            context.term()


def test_full_outbox_disables_input_and_recovers(poll_trace: dict) -> None:
    runtime = Runtime()
    with serving(runtime, max_work=2, socket_hwm=1) as (service, work, lifecycle):
        context = zmq.Context()
        slow = context.socket(zmq.DEALER)
        slow.setsockopt(zmq.RCVHWM, 1)
        slow.connect(work)
        client = _client(_endpoint(), lifecycle)
        try:
            for operation in range(100, 124):
                send_raw(slow, operation, 1024 * 1024)
            eventually(lambda: service.send_backpressure > 0)
            assert poll_trace["disabled"].wait(3)
            before = len(poll_trace["timeouts"])
            time.sleep(0.1)
            assert len(poll_trace["timeouts"]) - before < 100
            assert _request(client, "health").result(2).status == "COMPLETED"
            slow.setsockopt(zmq.RCVTIMEO, 3000)
            replies = [
                decode_message(slow.recv_multipart(), StorageResult) for _ in range(24)
            ]
            assert {reply.identity.operation_id for reply in replies} == set(
                range(100, 124)
            )
            assert all(reply.status in ("BUSY", "COMPLETED") for reply in replies)
        finally:
            client.close()
            slow.close(linger=0)
            context.term()


def test_generic_runtime_execution_valueerror_remains_unknown() -> None:
    class BrokenRuntime(Runtime):
        def handle(self, request: StorageRequest) -> StorageResult:
            raise ValueError("allocator dispatch failed after submission")

    with serving(BrokenRuntime()) as (_, work, lifecycle):
        client = _client(work, lifecycle)
        try:
            result = _request(client, "prepare", {}).result(2)
            assert result.status == "FAILED_UNKNOWN"
            assert "allocator dispatch failed" in result.error
        finally:
            client.close()
