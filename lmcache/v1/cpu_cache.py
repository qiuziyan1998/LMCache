# SPDX-License-Identifier: Apache-2.0
"""CPU storage ownership and replay-safe operation lifecycle.

Device work stays in the inference worker. A cancelled CPU operation does not
certify completion of any worker's independently submitted native work.
"""

# Standard
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Iterable
from uuid import uuid4
import hmac
import os

# First Party
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    StorageResult,
    content_digest,
)


class StorageBusy(RuntimeError):
    """Admission must defer until live ownership or terminal history drains."""


class StorageFailedSafe(RuntimeError):
    """A handler has explicitly proved its CPU producer ownership is drained."""

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.details = details


@dataclass
class _Operation:
    identity: OperationIdentity
    digest: bytes
    phase: str = "RUNNING"
    cancelled: bool = False
    result: StorageResult | None = None
    terminal_digest: bytes | None = None
    acknowledged: bool = False
    leases: set[str] = field(default_factory=set)
    ticket: str | None = None
    quarantined: bool = False
    had_leases: bool = False
    notified: bool = False


@dataclass
class _PhysicalLease:
    operation_key: tuple[str, str, int]
    objects: tuple[Any, ...]
    participants: frozenset[str]
    release: Callable[[tuple[Any, ...]], None]
    retired: set[str] = field(default_factory=set)
    producers: int = 0
    quarantined: bool = False
    releasing: bool = False
    exported: bool = False


class StorageOperationRegistry:
    """Bounded idempotent work, cancellation tombstones and physical leases.

    Handlers and allocator releases run outside the registry lock. Lifecycle
    calls cannot wait for CPU I/O; unknown native completion remains retained.
    """

    _CONTROLS = frozenset(
        {"cancel", "operation_status", "acknowledge_terminal", "retire"}
    )

    def __init__(
        self,
        cache_instance_id: str,
        service_epoch: str,
        sessions: dict[str, bytes],
        *,
        max_operations: int = 256,
    ) -> None:
        if not sessions or max_operations < 1:
            raise ValueError(
                "Storage requires authenticated sessions and bounded admission"
            )
        self.cache_instance_id = cache_instance_id
        self.service_epoch = service_epoch
        self.sessions = dict(sessions)
        self.max_operations = max_operations
        self._lock = RLock()
        self._operations: dict[tuple[str, str, int], _Operation] = {}
        self._watermarks = {session: 0 for session in sessions}
        self._acknowledged_ranges: dict[str, list[tuple[int, int]]] = {
            session: [] for session in sessions
        }
        self._leases: dict[str, _PhysicalLease] = {}
        self._terminal_notifications: set[tuple[str, str, int]] = set()
        self._fatal: str | None = None

    def _authenticate(self, request: StorageRequest) -> None:
        request.identity.validate()
        identity = request.identity
        session = request.participant_session or identity.client_session
        expected = self.sessions.get(session)
        if (
            identity.cache_instance_id != self.cache_instance_id
            or identity.service_epoch != self.service_epoch
            or identity.client_session not in self.sessions
            or expected is None
            or not hmac.compare_digest(expected, request.session_nonce)
        ):
            raise ValueError("Storage epoch, instance or session authentication failed")
        if identity.operation_id == 0 or len(request.payload_digest) != 32:
            raise ValueError(
                "Storage operation requires a positive ID and original digest"
            )
        if session != identity.client_session and request.method != "retire":
            raise ValueError("Only the original session may reconcile storage work")

    @staticmethod
    def _reply(
        request: StorageRequest,
        status: str,
        value: Any = None,
        error: str | None = None,
    ) -> StorageResult:
        return StorageResult(
            request.identity,
            status,
            value,
            error,
            payload_digest=request.payload_digest,
            correlation_id=request.correlation_id,
        )

    def _find(self, request: StorageRequest) -> _Operation | None:
        identity = request.identity
        if self._was_acknowledged(identity):
            raise ValueError(
                "Storage operation is below an acknowledged watermark/range"
            )
        operation = self._operations.get(identity.key)
        if operation is not None and (
            operation.identity != identity or operation.digest != request.payload_digest
        ):
            raise ValueError(
                "Storage operation identity was reused with different inputs"
            )
        return operation

    def _was_acknowledged(self, identity: OperationIdentity) -> bool:
        operation = self._operations.get(identity.key)
        return (
            (operation is not None and operation.acknowledged)
            or identity.operation_id <= self._watermarks[identity.client_session]
            or any(
                start <= identity.operation_id <= end
                for start, end in self._acknowledged_ranges[identity.client_session]
            )
        )

    def handle(
        self, request: StorageRequest, dispatch: Callable[[StorageRequest], Any]
    ) -> StorageResult:
        """Execute original work once, or reconcile an original operation."""
        self._authenticate(request)
        if request.method in self._CONTROLS:
            return self._control(request)
        if request.payload_digest != content_digest((request.method, request.payload)):
            raise ValueError("Storage work digest disagrees with its immutable payload")
        with self._lock:
            operation = self._find(request)
            if operation is not None:
                if operation.result is not None:
                    return self._reply(
                        request,
                        operation.result.status,
                        operation.result.value,
                        operation.result.error,
                    )
                return self._reply(request, "PENDING", self._summary(operation))
            if self._fatal is not None:
                raise RuntimeError(self._fatal)
            if len(self._operations) >= self.max_operations:
                raise StorageBusy(
                    "Storage operation history is full; reconcile live operations"
                )
            operation = _Operation(request.identity, request.payload_digest)
            self._operations[request.identity.key] = operation
        try:
            value = dispatch(request)
            result = self._reply(request, "COMPLETED", value)
        except StorageFailedSafe as exc:
            result = self._reply(request, "FAILED_SAFE", exc.details, error=str(exc))
        except Exception as exc:
            # A generic failure is not producer-completion evidence.
            result = self._reply(
                request, "FAILED_UNKNOWN", error=f"{type(exc).__name__}: {exc}"
            )
            with self._lock:
                operation.quarantined = True
                self._fatal = result.error
        with self._lock:
            operation.phase = result.status
            operation.result = result
            operation.terminal_digest = content_digest(
                (result.status, result.value, result.error)
            )
            self._queue_terminal(operation)
        return result

    def _summary(self, operation: _Operation) -> dict[str, Any]:
        leases = {}
        for lease_id in operation.leases:
            lease = self._leases[lease_id]
            leases[lease_id] = {
                "participants": tuple(sorted(lease.participants)),
                "retired": tuple(sorted(lease.retired)),
                "producer_holds": lease.producers,
                "quarantined": lease.quarantined,
                "exported": lease.exported,
            }
        return {
            "phase": operation.phase,
            "cancellation_requested": operation.cancelled,
            "terminal_result_digest": operation.terminal_digest,
            "ticket_id": operation.ticket,
            "leases": leases,
            "quarantined": operation.quarantined,
            "result": operation.result,
        }

    def _control(self, request: StorageRequest) -> StorageResult:
        with self._lock:
            if self._was_acknowledged(request.identity):
                # Compaction proves all leases and the original result were
                # acknowledged. A lost control reply must remain retryable;
                # old work still goes through _find and cannot allocate again.
                if request.method == "acknowledge_terminal":
                    return self._reply(request, "ACKNOWLEDGED")
                if request.method == "retire":
                    if request.payload.get("proof") not in ("never_imported", "fenced"):
                        raise ValueError("Retirement requires a safety proof")
                    return self._reply(request, "RETIRED")
            operation = self._find(request)
            if request.method == "cancel":
                if operation is None:
                    if len(self._operations) >= self.max_operations:
                        raise StorageBusy("Storage cancellation history is full")
                    operation = _Operation(
                        request.identity,
                        request.payload_digest,
                        phase="CANCELLED_NOT_STARTED",
                        cancelled=True,
                    )
                    operation.result = self._reply(request, "CANCELLED_NOT_STARTED")
                    operation.terminal_digest = content_digest(
                        ("CANCELLED_NOT_STARTED", None, None)
                    )
                    self._operations[request.identity.key] = operation
                operation.cancelled = True
                return self._reply(request, operation.phase, self._summary(operation))
            if operation is None:
                return self._reply(request, "UNKNOWN")
            if request.method == "operation_status":
                return self._reply(request, operation.phase, self._summary(operation))
            if request.method == "acknowledge_terminal":
                if (
                    operation.result is None
                    or operation.leases
                    or operation.ticket
                    or operation.quarantined
                    or request.payload.get("terminal_result_digest")
                    != operation.terminal_digest
                ):
                    return self._reply(request, "PENDING", self._summary(operation))
                operation.acknowledged = True
                self._compact(request.identity.client_session)
                return self._reply(request, "ACKNOWLEDGED")
            lease_id = request.payload["lease_id"]
            lease = self._leases.get(lease_id)
            if lease is None:
                if lease_id not in operation.leases:
                    return self._reply(request, "RETIRED")
                raise RuntimeError("Storage lease registry is inconsistent")
            if lease.operation_key != request.identity.key:
                raise ValueError("Retirement targets another storage operation")
            session = request.participant_session or request.identity.client_session
            if session not in lease.participants:
                raise ValueError("Retirement came from an unintended participant")
            if request.payload.get("proof") not in ("never_imported", "fenced"):
                raise ValueError(
                    "Retirement requires explicit per-participant safety proof"
                )
            lease.retired.add(session)
        self._try_release(lease_id)
        return self._reply(request, "RETIRED")

    def _compact(self, session: str) -> None:
        watermark = self._watermarks[session]
        acknowledged = sorted(
            key
            for key, operation in self._operations.items()
            if key[1] == session and operation.acknowledged
        )
        for key in acknowledged:
            ranges = sorted([*self._acknowledged_ranges[session], (key[2], key[2])])
            merged: list[tuple[int, int]] = []
            for start, end in ranges:
                if merged and start <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            if merged and merged[0][0] <= watermark + 1:
                watermark = max(watermark, merged.pop(0)[1])
            if len(merged) > self.max_operations:
                # Keep bounded evidence even for sparse/unreceived operation IDs.
                continue
            self._acknowledged_ranges[session] = merged
            del self._operations[key]
            self._terminal_notifications.discard(key)
        self._watermarks[session] = watermark

    def cancellation_requested(self, identity: OperationIdentity) -> bool:
        """Check only between safe CPU phases; cancellation is not a native fence."""
        with self._lock:
            return self._operations[identity.key].cancelled

    def set_ticket(self, identity: OperationIdentity, ticket: str | None) -> None:
        """Associate a lookup hold with its original operation until transfer/drop."""
        with self._lock:
            self._operations[identity.key].ticket = ticket
            self._queue_terminal(self._operations[identity.key])

    def add_lease(
        self,
        identity: OperationIdentity,
        objects: Iterable[Any],
        *,
        participants: Iterable[str],
        release: Callable[[tuple[Any, ...]], None],
        producer_holds: int = 0,
    ) -> str:
        """Adopt actual acquired allocator refs before exporting to any participant.

        ``objects`` lists acquisitions, not matrix aliases: merged pages appear
        once, while repeated individually fetched objects retain each get ref.
        """
        participants = frozenset(participants)
        if (
            not participants
            or not participants <= self.sessions.keys()
            or producer_holds < 0
        ):
            raise ValueError("Invalid storage lease participants or producer holds")
        lease_id = uuid4().hex
        lease = _PhysicalLease(
            identity.key,
            tuple(objects),
            participants,
            release,
            producers=producer_holds,
        )
        if not lease.objects:
            raise ValueError("A CPU source lease requires real allocator ownership")
        with self._lock:
            operation = self._operations[identity.key]
            self._leases[lease_id] = lease
            operation.leases.add(lease_id)
            operation.had_leases = True
        return lease_id

    def producer_done(self, lease_id: str, *, known_complete: bool) -> None:
        """Release a producer hold only when its actual native completion is known."""
        with self._lock:
            lease = self._leases[lease_id]
            if not known_complete:
                lease.quarantined = True
                self._operations[lease.operation_key].quarantined = True
                return
            if lease.producers <= 0:
                raise ValueError("Storage producer hold was already released")
            lease.producers -= 1
        self._try_release(lease_id)

    def publish_leases(
        self, identity: OperationIdentity, lease_ids: Iterable[str]
    ) -> None:
        """Seal a complete manifest atomically against cancellation and retirement."""
        with self._lock:
            operation = self._operations[identity.key]
            leases = [self._leases[lease_id] for lease_id in lease_ids]
            if (
                operation.cancelled
                or operation.result is not None
                or any(
                    lease.operation_key != identity.key
                    or lease.retired
                    or lease.producers <= 0
                    or lease.quarantined
                    or lease.exported
                    for lease in leases
                )
            ):
                raise StorageFailedSafe("CPU source export was cancelled or retired")
            for lease in leases:
                lease.exported = True
                lease.producers -= 1

    def discard_unexported_lease(
        self, identity: OperationIdentity, lease_id: str
    ) -> None:
        """Roll back a CPU-only result before any reply or source handle escapes.

        The runtime may call this only during failed preparation, before returning
        a manifest. Completed or possibly exported operations require participant
        retirement, including when the transport loses their reply.
        """
        with self._lock:
            operation = self._operations[identity.key]
            lease = self._leases.get(lease_id)
            if lease is None:
                if operation.result is None and lease_id not in operation.leases:
                    return
                raise ValueError("Cannot discard an exported storage lease")
            if (
                operation.result is not None
                or lease.operation_key != identity.key
                or lease.producers
                or lease.quarantined
                or lease.exported
            ):
                raise ValueError("Cannot discard an exported or producer-owned lease")
            lease.retired.update(lease.participants)
        self._try_release(lease_id)

    def _try_release(self, lease_id: str) -> None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if (
                lease is None
                or lease.quarantined
                or lease.producers
                or lease.releasing
                or lease.retired != lease.participants
            ):
                return
            lease.releasing = True
        try:
            lease.release(lease.objects)
        except Exception as exc:
            with self._lock:
                lease.quarantined = True
                self._operations[lease.operation_key].quarantined = True
                self._fatal = f"Storage lease release failed: {exc}"
            raise
        with self._lock:
            del self._leases[lease_id]
            self._operations[lease.operation_key].leases.remove(lease_id)
            self._queue_terminal(self._operations[lease.operation_key])

    def _queue_terminal(self, operation: _Operation) -> None:
        if (
            operation.result is not None
            and operation.had_leases
            and not operation.leases
            and not operation.ticket
            and not operation.quarantined
            and not operation.acknowledged
            and not operation.notified
        ):
            self._terminal_notifications.add(operation.identity.key)

    def drain_terminal_notifications(self) -> list[StorageResult]:
        """Notify original owners once all physical and producer holds are safe."""
        with self._lock:
            notices = []
            for key in tuple(self._terminal_notifications):
                operation = self._operations.get(key)
                if operation is None or operation.acknowledged:
                    self._terminal_notifications.discard(key)
                    continue
                operation.notified = True
                notices.append(
                    StorageResult(
                        operation.identity,
                        "TERMINAL",
                        {"terminal_result_digest": operation.terminal_digest},
                        payload_digest=operation.digest,
                        session_nonce=self.sessions[operation.identity.client_session],
                    )
                )
                self._terminal_notifications.remove(key)
            return notices

    def snapshot(self) -> dict[str, Any]:
        """Return administration counters without traversing cache contents."""
        with self._lock:
            return {
                "pid": os.getpid(),
                "operations": len(self._operations),
                "leases": len(self._leases),
                "watermarks": dict(self._watermarks),
                "acknowledged_ranges": {
                    session: tuple(ranges)
                    for session, ranges in self._acknowledged_ranges.items()
                },
                "running": sum(op.result is None for op in self._operations.values()),
                "quarantined": sum(op.quarantined for op in self._operations.values()),
                "fatal": self._fatal,
            }

    def is_healthy(self) -> bool:
        """Read the latched fatal state; no device or remote query."""
        with self._lock:
            return self._fatal is None

    def mark_fatal(self, reason: str) -> None:
        """Latch native uncertainty without releasing ownership or waiting on I/O."""
        with self._lock:
            if self._fatal is None:
                self._fatal = reason or "CPU storage native completion is unknown"
