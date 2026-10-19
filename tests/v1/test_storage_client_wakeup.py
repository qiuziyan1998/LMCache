# SPDX-License-Identifier: Apache-2.0
"""Real poller/TCP tests for queue wakeups; these do not qualify NPU ordering."""

# Standard
from contextlib import contextmanager
from queue import Empty, Queue
from typing import Any, Iterator
import gc
import socket
import threading
import time

# Third Party
import pytest
import zmq

# First Party
from lmcache.v1.rpc import storage_client
from lmcache.v1.rpc.storage_client import StorageClient, StorageUnavailableError
from lmcache.v1.rpc.storage_protocol import (
    StorageRequest,
    StorageResult,
    decode_message,
    encode_message,
)
from lmcache.v1.rpc_utils import PollerWakeup


def test_notification_before_poll_and_drain() -> None:
    wake = PollerWakeup()
    poller = zmq.Poller()
    poller.register(wake.fileno(), zmq.POLLIN)
    try:
        wake.notify()
        wake.notify()
        assert dict(poller.poll(1000))[wake.fileno()] & zmq.POLLIN
        wake.drain()
        assert not poller.poll(0)
        wake.notify()
        assert dict(poller.poll(1000))[wake.fileno()] & zmq.POLLIN
    finally:
        wake.close()
        wake.close()
    assert wake.fileno() == -1
    wake.notify()
    wake.drain()


def test_concurrent_publication_during_drain_without_gc() -> None:
    wake = PollerWakeup()
    poller = zmq.Poller()
    poller.register(wake.fileno(), zmq.POLLIN)
    queue: Queue[tuple[int, int]] = Queue()
    start = threading.Barrier(5)

    def publish(publisher: int) -> None:
        start.wait()
        for value in range(200):
            queue.put((publisher, value))
            wake.notify()

    threads = [threading.Thread(target=publish, args=(i,)) for i in range(4)]
    enabled = gc.isenabled()
    gc.disable()
    try:
        for thread in threads:
            thread.start()
        start.wait()
        received = set()
        while len(received) < 800:
            assert poller.poll(2000)
            wake.drain()
            while True:
                try:
                    received.add(queue.get_nowait())
                except Empty:
                    break
        assert received == {
            (publisher, value) for publisher in range(4) for value in range(200)
        }
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
    finally:
        wake.close()
        if enabled:
            gc.enable()


def test_close_racing_notify_does_not_write_reused_descriptor() -> None:
    for _ in range(10):
        wake = PollerWakeup()
        running = threading.Event()
        running.set()
        errors = []

        def notify(
            wake: PollerWakeup = wake,
            running: threading.Event = running,
            errors: list[BaseException] = errors,
        ) -> None:
            try:
                while running.is_set():
                    wake.notify()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=notify)
        thread.start()
        wake.close()
        reader, writer = socket.socketpair()
        reader.setblocking(False)
        try:
            for _ in range(50):
                wake.notify()
            with pytest.raises(BlockingIOError):
                reader.recv(1)
        finally:
            running.clear()
            thread.join(2)
            reader.close()
            writer.close()
        assert not errors
        assert not thread.is_alive()


def test_full_wakeup_buffer_is_nonblocking_and_drains() -> None:
    wake = PollerWakeup()
    poller = zmq.Poller()
    poller.register(wake.fileno(), zmq.POLLIN)
    try:
        # Far more notifications than a socket-pair send buffer can hold. A
        # saturated wake is coalesced; no request data is carried in these bytes.
        for _ in range(100_000):
            wake.notify()
        assert poller.poll(0)
        wake.drain()
        assert not poller.poll(0)
    finally:
        wake.close()


@contextmanager
def _routers(**kwargs: Any) -> Iterator[tuple[StorageClient, list[zmq.Socket]]]:
    context = zmq.Context()
    routers = [context.socket(zmq.ROUTER) for _ in range(2)]
    endpoints = []
    for router in routers:
        router.setsockopt(zmq.RCVTIMEO, 1500)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        endpoints.append(f"tcp://127.0.0.1:{port}")
    client = StorageClient(
        *endpoints,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker",
        session_nonce=b"nonce",
        **kwargs,
    )
    try:
        yield client, routers
    finally:
        client.close()
        for router in routers:
            router.close(linger=0)
        context.term()


def _request(client: StorageClient, payload: Any = None):
    return client.request(
        "prepare",
        payload,
        request_id="req",
        request_attempt="attempt",
        load_generation=1,
        purpose="test",
    )


def _receive(router: zmq.Socket) -> tuple[bytes, list[bytes], StorageRequest]:
    route, *frames = router.recv_multipart()
    return route, frames, decode_message(frames, StorageRequest)


def _reply(router: zmq.Socket, route: bytes, request: StorageRequest) -> None:
    result = StorageResult(
        request.identity,
        "EXPORTED",
        request.payload,
        payload_digest=request.payload_digest,
        correlation_id=request.correlation_id,
    )
    router.send_multipart([route, *encode_message(result)])


def _observe_polls(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    original = zmq.Poller
    calls = []

    class ObservedPoller(original):
        def poll(self, timeout: int | None = None):
            calls.append(timeout)
            # An old fixed 5ms poll becomes 2s, proving request/replay/control
            # dispatch must interrupt the actual poll rather than use its timer.
            return super().poll(2000 if timeout == 5 else timeout)

    monkeypatch.setattr(storage_client.zmq, "Poller", ObservedPoller)
    return calls


def test_request_replay_and_control_interrupt_long_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _observe_polls(monkeypatch)
    with _routers() as (client, routers):
        time.sleep(0.05)
        future = _request(client, {"immutable": [1, 2]})
        route, original_frames, request = _receive(routers[0])
        assert client.replay(future.identity) is future
        _, replay_frames, replay = _receive(routers[0])
        assert replay_frames == original_frames
        assert replay.identity == request.identity
        control = client.cancel(future.identity)
        control_route, _, control_request = _receive(routers[1])
        _reply(routers[1], control_route, control_request)
        assert control.result(1).identity == future.identity
        _reply(routers[0], route, request)
        assert future.result(1).value == {"immutable": [1, 2]}
        time.sleep(0.05)
        before = len(calls)
        time.sleep(0.1)
        assert len(calls) == before  # No permanently-writable POLLOUT spin.
        assert calls and all(timeout is None for timeout in calls)


def test_bounded_send_budget_does_not_lose_aggregated_wakeup() -> None:
    with _routers(max_pending=64, socket_hwm=1) as (client, routers):
        futures = [_request(client, {"value": i}) for i in range(48)]
        for _ in range(48):
            route, _, request = _receive(routers[0])
            _reply(routers[0], route, request)
        assert [future.result(2).value for future in futures] == [
            {"value": i} for i in range(48)
        ]


def test_disconnected_work_does_not_spin_or_block_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _observe_polls(monkeypatch)
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.RCVTIMEO, 1000)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        absent = f"tcp://127.0.0.1:{reservation.getsockname()[1]}"
    client = StorageClient(
        absent,
        f"tcp://127.0.0.1:{port}",
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker",
        session_nonce=b"nonce",
        socket_hwm=1,
    )
    try:
        future = _request(client)
        control = client.cancel(future.identity)
        route, _, request = _receive(router)
        _reply(router, route, request)
        assert control.result(1).identity == future.identity
        time.sleep(0.05)
        before = len(calls)
        time.sleep(0.1)
        assert len(calls) == before
        assert not future.done()
        assert client.send_backpressure > 0
    finally:
        client.close()
        router.close(linger=0)
        context.term()


def test_response_deadline_is_not_extended_by_replay() -> None:
    with _routers(failure_timeout_s=0.3) as (client, routers):
        start = time.monotonic()
        future = _request(client)
        _receive(routers[0])
        time.sleep(0.18)
        client.replay(future.identity)
        _receive(routers[0])
        with pytest.raises(StorageUnavailableError, match="completion unknown"):
            future.result(0.22)
        assert time.monotonic() - start < 0.42
        assert client.close() is False  # Failure retains unknown ownership.


def test_fatal_drops_bounded_queue_leftovers_without_disconnected_pollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = zmq.Poller
    masks = {}

    class ObservedPoller(original):
        def modify(self, socket: Any, flags: int = zmq.POLLIN | zmq.POLLOUT) -> None:
            masks[socket] = flags
            super().modify(socket, flags)

    monkeypatch.setattr(storage_client.zmq, "Poller", ObservedPoller)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{reservation.getsockname()[1]}"
    client = StorageClient(
        endpoint,
        endpoint,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker",
        session_nonce=b"nonce",
    )
    try:
        futures = [_request(client) for _ in range(40)]
        deadline = time.monotonic() + 1
        while not any(mask & zmq.POLLOUT for mask in masks.values()):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        client.mark_fatal("injected uncertainty")
        for future in futures:
            with pytest.raises(StorageUnavailableError):
                future.result(1)
        while any(mask & zmq.POLLOUT for mask in masks.values()):
            assert time.monotonic() < deadline
            time.sleep(0.001)
    finally:
        client.close()


def test_completion_callback_can_close_without_self_join() -> None:
    with _routers() as (client, routers):
        future = _request(client)
        completed = threading.Event()

        def close(_future: Any) -> None:
            client.close()
            completed.set()

        future.add_done_callback(close)
        route, _, request = _receive(routers[0])
        _reply(routers[0], route, request)
        assert completed.wait(1)
        assert future.done()


def test_close_timeout_retains_wakeup_until_blocked_completion_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = threading.Event()

    class ObservedWakeup(PollerWakeup):
        def close(self) -> None:
            super().close()
            closed.set()

    monkeypatch.setattr(storage_client, "PollerWakeup", ObservedWakeup)
    with _routers() as (client, routers):
        future = _request(client)
        entered = threading.Event()
        release = threading.Event()

        def block(_future: Any) -> None:
            entered.set()
            release.wait(3)

        future.add_done_callback(block)
        try:
            route, _, request = _receive(routers[0])
            _reply(routers[0], route, request)
            assert entered.wait(1)
            client.close(0.02)
            assert not closed.is_set()
            release.set()
            assert closed.wait(1)
        finally:
            release.set()


def test_failure_callback_can_close_from_io_thread() -> None:
    with _routers(failure_timeout_s=0.05) as (client, _routers_):
        completed = threading.Event()

        def close(_reason: str) -> None:
            client.close()
            completed.set()

        client.set_failure_callback(close)
        future = _request(client)
        assert completed.wait(1)
        with pytest.raises(StorageUnavailableError):
            future.result(1)
