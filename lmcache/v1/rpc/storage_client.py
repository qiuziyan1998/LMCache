# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous storage transport; device ownership stays with the caller."""

# Standard
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable
import hmac
import math
import threading
import time

# Third Party
import zmq

# First Party
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    StorageResult,
    content_digest,
    decode_message,
    encode_message,
)
from lmcache.v1.rpc_utils import PollerWakeup

LIFECYCLE_METHODS = frozenset(
    (
        "status",
        "operation_status",
        "cancel",
        "retire",
        "acknowledge_terminal",
        "health",
        "stop",
        "memory_ready",
        "register_participant",
        "unregister_participant",
        "registration_status",
    )
)


class StorageUnavailableError(RuntimeError):
    """The transport failed; this exception does not prove native completion."""


class StorageFuture(Future):
    """Future exposing the original identity for cancellation and reconciliation."""

    def __init__(self, request: StorageRequest) -> None:
        super().__init__()
        self.identity = request.identity
        self.payload_digest = request.payload_digest

    def cancel(self) -> bool:
        """Require explicit remote cancellation instead of losing local ownership."""
        return False


@dataclass
class _Pending:
    request: StorageRequest
    frames: list[bytes]
    future: StorageFuture
    admitted_at: float
    queued: bool = True
    received: bool = False
    acknowledged: bool = False
    terminal_digest: bytes | None = None
    terminal_ack_queued: bool = False


class StorageClient:
    """Own two DEALER sockets in one I/O thread, with bounded caller admission.

    A caller timeout leaves its operation and lease evidence retained. Replies and
    retries use the same original identity; no timer allocates replacement work.
    Future callbacks run in a separate completion thread so they cannot block I/O.
    """

    def __init__(
        self,
        work_endpoint: str,
        lifecycle_endpoint: str,
        *,
        cache_instance_id: str,
        service_epoch: str,
        client_session: str,
        session_nonce: bytes,
        max_pending: int = 64,
        socket_hwm: int = 64,
        failure_timeout_s: float | None = None,
        failure_callback: Callable[[str], None] | None = None,
    ) -> None:
        if max_pending <= 0 or socket_hwm <= 0 or not session_nonce:
            raise ValueError("Storage bounds and session nonce must be positive")
        if failure_timeout_s is not None and failure_timeout_s <= 0:
            raise ValueError("Storage failure deadline must be positive")
        self.cache_instance_id = cache_instance_id
        self.service_epoch = service_epoch
        self.client_session = client_session
        self.session_nonce = session_nonce
        self.max_pending = max_pending
        self.socket_hwm = socket_hwm
        self.failure_timeout_s = failure_timeout_s
        self._endpoints = (work_endpoint, lifecycle_endpoint)
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._poll_wakeup = PollerWakeup()
        self._records: dict[tuple[str, str, int], _Pending] = {}
        self._terminal_waiting: set[tuple[str, str, int]] = set()
        self._pending: dict[tuple[tuple[str, str, int], int], _Pending] = {}
        self._outboxes: tuple[deque, deque] = (deque(), deque())
        self._completions: deque[tuple[_Pending, StorageResult | BaseException]] = (
            deque()
        )
        self._sequence = 0
        self._correlation = 0
        self._closing = False
        self._io_stopped = False
        self._completion_stopped = False
        self._fatal: str | None = None
        self._failure_callback = failure_callback
        self._failure_notified = False
        self.send_backpressure = 0
        self.ignored_responses = 0
        self._io = threading.Thread(
            target=self._run, name="storage-client-io", daemon=True
        )
        self._completion = threading.Thread(
            target=self._complete, name="storage-client-completion", daemon=True
        )
        self._completion.start()
        self._io.start()

    def request(
        self,
        method: str,
        payload: Any,
        *,
        request_id: str,
        request_attempt: str,
        load_generation: int,
        purpose: str,
    ) -> StorageFuture:
        """Admit one operation or raise BufferError before assigning its identity."""
        with self._lock:
            self._check_admission()
            if len(self._records) >= self.max_pending:
                raise BufferError("Storage operation admission is full")
            identity = OperationIdentity(
                self.cache_instance_id,
                self.service_epoch,
                self.client_session,
                self._sequence + 1,
                request_id,
                request_attempt,
                load_generation,
                purpose,
            )
            identity.validate()
            request = StorageRequest(
                identity,
                method,
                payload,
                content_digest((method, payload)),
                self.session_nonce,
            )
            # Freeze caller-owned containers once; retries reuse these exact bytes.
            frames = encode_message(request)
            request = decode_message(frames, StorageRequest)
            self._sequence += 1
            pending = _Pending(
                request, frames, StorageFuture(request), time.monotonic()
            )
            self._records[identity.key] = pending
            self._pending[(identity.key, 0)] = pending
            self._outboxes[int(method.lower() in LIFECYCLE_METHODS)].append(pending)
            self._poll_wakeup.notify()
            return pending.future

    def replay(self, identity: OperationIdentity) -> StorageFuture:
        """Resend unresolved work with the original ID and immutable payload."""
        with self._lock:
            self._check_admission()
            pending = self._record(identity)
            if not pending.received and not pending.queued:
                pending.queued = True
                self._outboxes[
                    int(pending.request.method.lower() in LIFECYCLE_METHODS)
                ].append(pending)
                self._poll_wakeup.notify()
            return pending.future

    def operation_status(self, identity: OperationIdentity) -> StorageFuture:
        """Query the original operation; UNKNOWN is not allocation-free proof."""
        return self._control(identity, "operation_status", None)

    def cancel(self, identity: OperationIdentity) -> StorageFuture:
        """Request cancellation, retaining records until terminal acknowledgement."""
        return self._control(identity, "cancel", None)

    def retire(
        self,
        identity: OperationIdentity,
        payload_digest: bytes,
        lease_id: str,
        proof: str,
    ) -> StorageFuture:
        """Submit participant retirement proof without waiting in the caller."""
        return self._control(
            identity,
            "retire",
            {"lease_id": lease_id, "proof": proof},
            payload_digest=payload_digest,
        )

    def acknowledge_terminal(
        self, identity: OperationIdentity, payload: Any = None
    ) -> StorageFuture:
        """Acknowledge a resource-free terminal result; this is not a DMA fence."""
        with self._lock:
            if (
                identity.key not in self._records
                and identity.cache_instance_id == self.cache_instance_id
                and identity.service_epoch == self.service_epoch
                and identity.client_session == self.client_session
                and 0 < identity.operation_id <= self._sequence
            ):
                # IDs are assigned without gaps and records are removed only
                # after ACKNOWLEDGED. No separate unbounded tombstone cache.
                future = StorageFuture(
                    StorageRequest(
                        identity,
                        "acknowledge_terminal",
                        payload,
                        b"",
                        self.session_nonce,
                    )
                )
                future.set_result(StorageResult(identity, "ACKNOWLEDGED"))
                return future
        if payload is None:
            with self._lock:
                original = self._record(identity).future
                if not original.done():
                    raise ValueError(
                        "Reconcile the original result before acknowledging it"
                    )
                result = original.result()
                payload = {
                    "terminal_result_digest": content_digest(
                        (result.status, result.value, result.error)
                    )
                }
        return self._control(identity, "acknowledge_terminal", payload)

    def is_healthy(self) -> bool:
        """Return cached transport health without issuing a service query."""
        with self._lock:
            return not self._closing and self._fatal is None

    def set_failure_callback(self, callback: Callable[[str], None]) -> None:
        """Bind worker supervision after its bootstrap context exists."""
        with self._lock:
            self._failure_callback = callback
            reason = (
                self._fatal
                if not self._closing and not self._failure_notified
                else None
            )
            if reason is not None:
                self._failure_notified = True
        if reason is not None:
            callback(reason)

    def mark_fatal(self, reason: str, *, notify: bool = True) -> None:
        """Fail pending CPU futures while retaining original ownership evidence."""
        callback = None
        with self._lock:
            if self._fatal is None:
                self._fatal = reason
                callback = (
                    self._failure_callback if notify and not self._closing else None
                )
                self._failure_notified = callback is not None
                for pending in self._pending.values():
                    if not pending.received:
                        pending.received = True
                        self._completions.append(
                            (pending, StorageUnavailableError(reason))
                        )
                self._wake.notify_all()
                self._poll_wakeup.notify()
        if callback is not None:
            callback(reason)

    def close(self, timeout: float = 5.0) -> bool:
        """Stop transport threads; return whether all operations were acknowledged.

        Unacknowledged records remain retained and must enter coordinated recovery.
        Closing transport never retires leases or proves native completion.
        """
        with self._lock:
            self._closing = True
            self._wake.notify_all()
            self._poll_wakeup.notify()
        self.mark_fatal(
            "Storage client closed before terminal acknowledgement", notify=False
        )
        if threading.current_thread() is not self._io:
            self._io.join(timeout)
        if threading.current_thread() not in (self._completion, self._io):
            self._completion.join(timeout)
        with self._lock:
            if self._io_stopped and self._completion_stopped:
                self._poll_wakeup.close()
            return not self._records and not self._io.is_alive()

    def _check_admission(self) -> None:
        if self._closing or self._fatal is not None:
            raise StorageUnavailableError(self._fatal or "Storage client is closing")

    def _record(self, identity: OperationIdentity) -> _Pending:
        pending = self._records.get(identity.key)
        if pending is None or pending.request.identity != identity:
            raise ValueError("Unknown storage operation identity")
        return pending

    def _control(
        self,
        identity: OperationIdentity,
        method: str,
        payload: Any,
        *,
        payload_digest: bytes | None = None,
    ) -> StorageFuture:
        with self._lock:
            self._check_admission()
            if payload_digest is None:
                payload_digest = self._record(identity).request.payload_digest
            elif method != "retire":
                raise ValueError(
                    "Only retirement may target another participant's operation"
                )
            controls = [
                entry
                for (op, corr), entry in self._pending.items()
                if op == identity.key and corr
            ]
            control_digest = content_digest((method, payload))
            for pending in controls:
                if (
                    content_digest((pending.request.method, pending.request.payload))
                    == control_digest
                ):
                    if not pending.received and not pending.queued:
                        pending.queued = True
                        self._outboxes[1].append(pending)
                        self._poll_wakeup.notify()
                    return pending.future
            if len(self._pending) >= self.max_pending * 9:
                raise BufferError("Storage lifecycle admission is full")
            if len(controls) >= 8:
                raise BufferError(
                    "Storage lifecycle admission is full for this operation"
                )
            self._correlation += 1
            request = StorageRequest(
                identity,
                method,
                payload,
                payload_digest,
                self.session_nonce,
                correlation_id=self._correlation,
                participant_session=self.client_session,
            )
            frames = encode_message(request)
            request = decode_message(frames, StorageRequest)
            pending = _Pending(
                request, frames, StorageFuture(request), time.monotonic()
            )
            self._pending[(identity.key, request.correlation_id)] = pending
            self._outboxes[1].append(pending)
            self._poll_wakeup.notify()
            return pending.future

    def _run(self) -> None:
        context = zmq.Context()
        sockets = []
        try:
            poller = zmq.Poller()
            poller.register(self._poll_wakeup.fileno(), zmq.POLLIN)
            for endpoint in self._endpoints:
                socket = context.socket(zmq.DEALER)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.IMMEDIATE, 1)
                socket.setsockopt(zmq.SNDHWM, self.socket_hwm)
                socket.setsockopt(zmq.RCVHWM, self.socket_hwm)
                socket.connect(endpoint)
                sockets.append(socket)
                poller.register(socket, zmq.POLLIN)
            events: dict[Any, int] = {}
            while not self._closing:
                # Drain first. A concurrent enqueue is either included below or
                # leaves a fresh signal for the next poll.
                self._poll_wakeup.drain()
                for channel in (1, 0):
                    socket = sockets[channel]
                    if events.get(socket, 0) & zmq.POLLIN:
                        for _ in range(16):
                            try:
                                result = decode_message(
                                    socket.recv_multipart(zmq.NOBLOCK), StorageResult
                                )
                            except zmq.Again:
                                break
                            self._receive(result)
                    with self._lock:
                        for _ in range(min(8, len(self._outboxes[channel]))):
                            pending = self._outboxes[channel][0]
                            if pending.received:
                                self._outboxes[channel].popleft()
                                pending.queued = False
                                continue
                            try:
                                socket.send_multipart(
                                    pending.frames, zmq.NOBLOCK, copy=False
                                )
                            except zmq.Again:
                                self.send_backpressure += 1
                                break
                            self._outboxes[channel].popleft()
                            pending.queued = False
                with self._lock:
                    for channel, socket in enumerate(sockets):
                        poller.modify(
                            socket,
                            zmq.POLLIN
                            | (zmq.POLLOUT if self._outboxes[channel] else 0),
                        )
                    deadline = (
                        min(
                            (
                                p.admitted_at + self.failure_timeout_s
                                for p in self._pending.values()
                                if not p.received
                            ),
                            default=None,
                        )
                        if self.failure_timeout_s is not None
                        else None
                    )
                    closing = self._closing
                    runnable = any(box and box[0].received for box in self._outboxes)
                if closing:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self.mark_fatal(
                        "Storage response failure deadline expired; completion unknown"
                    )
                    continue
                # Only unsent channels request POLLOUT. Budget leftovers remain
                # writable, while a disconnected work peer cannot spin or stall
                # the independently connected lifecycle channel.
                timeout = None if remaining is None else math.ceil(remaining * 1000)
                if runnable:
                    timeout = 0
                events = dict(poller.poll(timeout))
        except BaseException as error:
            self.mark_fatal(f"Storage transport failed: {error}")
        finally:
            for socket in sockets:
                socket.close(linger=0)
            context.term()
            with self._lock:
                self._io_stopped = True
                self._wake.notify_all()
                if self._completion_stopped:
                    self._poll_wakeup.close()

    def _receive(self, result: StorageResult) -> None:
        if result.status == "TERMINAL":
            with self._lock:
                original = self._records.get(result.identity.key)
                if (
                    original is not None
                    and result.identity == original.request.identity
                    and result.identity.client_session == self.client_session
                    and result.payload_digest == original.request.payload_digest
                    and result.correlation_id == 0
                    and hmac.compare_digest(result.session_nonce, self.session_nonce)
                    and isinstance(result.value, dict)
                ):
                    original.terminal_digest = result.value.get(
                        "terminal_result_digest"
                    )
                    self._ack_notified(original)
                    return
                self.ignored_responses += 1
                return
        if result.status == "FATAL":
            identity = result.identity
            if (
                identity.cache_instance_id == self.cache_instance_id
                and identity.service_epoch == self.service_epoch
                and identity.client_session == self.client_session
                and identity.operation_id == 0
                and identity.purpose == "fatal"
                and result.correlation_id == 0
                and hmac.compare_digest(result.session_nonce, self.session_nonce)
            ):
                self.mark_fatal(result.error or "Storage runtime became unhealthy")
                return
        with self._lock:
            key = (result.identity.key, result.correlation_id)
            pending = self._pending.get(key)
            if (
                pending is None
                or pending.received
                or result.identity != pending.request.identity
                or result.payload_digest != pending.request.payload_digest
            ):
                self.ignored_responses += 1
                return
            pending.received = True
            self._completions.append((pending, result))
            self._wake.notify_all()

    def _complete(self) -> None:
        try:
            self._complete_pending()
        finally:
            with self._lock:
                self._completion_stopped = True
                if self._io_stopped:
                    self._poll_wakeup.close()

    def _complete_pending(self) -> None:
        while True:
            with self._wake:
                self._wake.wait_for(lambda: self._completions or self._io_stopped)
                if not self._completions:
                    return
                pending, result = self._completions.popleft()
                request = pending.request
                if request.correlation_id:
                    self._pending.pop(
                        (request.identity.key, request.correlation_id), None
                    )
                    original = self._records.get(request.identity.key)
                    if (
                        original is not None
                        and request.method == "acknowledge_terminal"
                        and isinstance(result, StorageResult)
                        and result.status == "ACKNOWLEDGED"
                    ):
                        original.acknowledged = True
                    if (
                        original is not None
                        and original.acknowledged
                        and not any(
                            op == request.identity.key and corr
                            for op, corr in self._pending
                        )
                    ):
                        self._records.pop(request.identity.key, None)
                        self._terminal_waiting.discard(request.identity.key)
                        self._pending.pop((request.identity.key, 0), None)
            if isinstance(result, BaseException):
                pending.future.set_exception(result)
            else:
                pending.future.set_result(result)
            with self._lock:
                original = self._records.get(request.identity.key)
                if original is not None:
                    self._ack_notified(original)
                # Peer controls can release global lifecycle credit without
                # naming any operation owned by this client.
                for key in tuple(self._terminal_waiting):
                    waiting = self._records.get(key)
                    if waiting is not None:
                        self._ack_notified(waiting)

    def _ack_notified(self, original: _Pending) -> None:
        # Bootstrap and scheduler lookup retain their existing explicit ACK path.
        if (
            original.request.method not in ("prepare", "prepare_store", "reserve_write")
            or original.terminal_ack_queued
            or original.terminal_digest is None
            or not original.future.done()
        ):
            return
        try:
            result = original.future.result()
            if original.terminal_digest != content_digest(
                (result.status, result.value, result.error)
            ):
                self.ignored_responses += 1
                return
            self.acknowledge_terminal(
                original.request.identity,
                {"terminal_result_digest": original.terminal_digest},
            )
            original.terminal_ack_queued = True
            self._terminal_waiting.discard(original.request.identity.key)
        except (BufferError, StorageUnavailableError):
            self._terminal_waiting.add(original.request.identity.key)
            return  # Retain the notice and retry when another completion frees credit.
        except BaseException:
            return  # A locally failed Future cannot acknowledge an unseen result.
