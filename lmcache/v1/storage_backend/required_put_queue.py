# SPDX-License-Identifier: Apache-2.0
"""Bounded, storage-only execution of strict CPU page puts."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Condition
from typing import TYPE_CHECKING, Sequence

# First Party
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.storage_backend.storage_manager import StorageManager


class RequiredPutQueue:
    """Persist already-ready CPU buffers through the unchanged strict put API.

    ``manager`` must be a StorageManager using the standard RemoteBackend with a
    connected, completion-requiring batched connector (the Mooncake path). Keep
    that backend/connector configuration stable until close. Its batched path
    returns one child future, so each queued or running job reserves one future
    slot, regardless of row count. This is not a generic multi-backend queue.

    All limits are positive integers. Bytes are the sum of ``obj.get_size()``
    per input occurrence, including duplicates, not allocator or tensor size.
    At most ``min(max_jobs, max_futures, 2)`` lazy executor threads run storage
    calls. Admission is thread-safe; a capacity-limited caller waits for the
    oldest pending job. Not-yet-admitted caller buffers are outside the budgets.

    Any submission or worker failure permanently poisons the queue. Admitted
    jobs still run and drain; outstanding tickets cannot report success after
    poison, and errors are published only after all admitted jobs finish. A
    previously successful ticket is not retroactively changed. Callers must
    treat poison as fatal to their engine worker, not continue serving/retry.

    No GPU readiness, copies, TP operations, cancellation, or native timeout
    retirement is provided here. Completion means only what the strict API
    guarantees. The connector must retain native I/O references until done;
    existing storage teardown must drain those operations before allocator
    destruction. Close this queue before storage teardown. Call submit/drain/
    close from caller threads, never from storage workers or ticket callbacks.
    """

    def __init__(
        self,
        manager: "StorageManager",
        *,
        max_jobs: int,
        max_bytes: int,
        max_futures: int,
    ) -> None:
        """Set admission limits without starting threads; reject invalid limits."""
        for name, value in (
            ("max_jobs", max_jobs),
            ("max_bytes", max_bytes),
            ("max_futures", max_futures),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._manager = manager
        self._max_jobs = max_jobs
        self._max_bytes = max_bytes
        self._max_futures = max_futures
        self._condition = Condition()
        self._executor: ThreadPoolExecutor | None = None
        self._pending: dict[Future[None], int] = {}
        self._failed_tickets: list[Future[None]] = []
        self._bytes = 0
        self._peak_jobs = 0
        self._peak_bytes = 0
        self._error: BaseException | None = None
        self._closed = False

    def submit(
        self, keys: Sequence["CacheEngineKey"], objects: Sequence["MemoryObj"]
    ) -> Future[None]:
        """Snapshot a batch and consume one borrowed reference per occurrence.

        Keys and objects must have equal lengths; buffers must already be CPU
        ready and remain immutable until completion. Returns a running,
        non-cancellable ticket whose result is None after strict persistence.
        Empty input succeeds without allocating an executor or budget slots.

        Oversize indivisible batches raise ValueError before enqueue (splitting
        belongs to the caller). Closed queues raise RuntimeError. Any error,
        including sizing or scheduling, consumes the references, poisons the
        queue, drains admitted work, and raises the first recorded error.
        """
        owned = objects
        admitted = False
        try:
            owned = list(objects)
            key_snapshot = tuple(keys)
            if len(key_snapshot) != len(owned):
                raise ValueError("Required put requires matching keys and objects")
            size = 0
            for obj in owned:
                obj_size = obj.get_size()
                if type(obj_size) is not int or obj_size < 0:
                    raise ValueError("Required put size must be a nonnegative integer")
                size += obj_size
            if size > self._max_bytes:
                raise ValueError(
                    f"Required put payload {size} exceeds max_bytes={self._max_bytes}"
                )

            with self._condition:
                while True:
                    if self._error is not None:
                        raise self._error
                    if self._closed:
                        raise RuntimeError("Required put queue is closed")
                    if (
                        not owned
                        or len(self._pending) < min(self._max_jobs, self._max_futures)
                        and self._bytes + size <= self._max_bytes
                    ):
                        break
                    oldest = next(iter(self._pending))
                    while (
                        oldest in self._pending
                        and self._error is None
                        and not self._closed
                    ):
                        self._condition.wait()

                ticket: Future[None] = Future()
                ticket.set_running_or_notify_cancel()
                if not owned:
                    ticket.set_result(None)
                    return ticket
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=min(self._max_jobs, self._max_futures, 2),
                        thread_name_prefix="required-put",
                    )
                self._pending[ticket] = size
                self._bytes += size
                admitted = True
                try:
                    self._executor.submit(self._run, ticket, key_snapshot, owned)
                except BaseException as exc:
                    # submit() can enqueue before thread.start() raises. The
                    # worker's locked membership check prevents a late handoff.
                    del self._pending[ticket]
                    self._bytes -= size
                    admitted = False
                    self._poison(exc)
                    raise
                self._peak_jobs = max(self._peak_jobs, len(self._pending))
                self._peak_bytes = max(self._peak_bytes, self._bytes)
                return ticket
        except BaseException as exc:
            self._poison(exc)
            if not admitted:
                self._release(owned)
            self.drain()
            raise

    def drain(self) -> None:
        """Wait for all admitted work, then raise the first error, if any.

        Does not close a healthy queue. Concurrent submissions may extend the
        wait; use close to prevent further admission. No completion timeout is
        imposed and no native operation is cancelled. Interrupted waits also
        poison the queue and finish draining before propagating the interruption.
        """
        with self._condition:
            while self._pending:
                try:
                    self._condition.wait()
                except BaseException as exc:
                    self._poison(exc)
            if self._error is not None:
                raise self._error

    def close(self) -> None:
        """Prevent admission, drain, and shut down even on error; safe to repeat.

        Re-raises the first recorded failure after draining and executor shutdown.
        Does not close the manager or its storage backends.
        """
        with self._condition:
            self._closed = True
            executor = self._executor
            self._condition.notify_all()
        try:
            self.drain()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    @property
    def pending_jobs(self) -> int:
        """Return the number of queued plus running strict calls."""
        with self._condition:
            return len(self._pending)

    @property
    def pending_bytes(self) -> int:
        """Return logical input bytes reserved by queued plus running calls."""
        with self._condition:
            return self._bytes

    @property
    def pending_futures(self) -> int:
        """Return reserved child-future slots (one per queued or running batch)."""
        return self.pending_jobs

    def stats(self) -> dict[str, int]:
        """Return an atomic snapshot of current/peak admission counters and limits."""
        with self._condition:
            return {
                "pending_jobs": len(self._pending),
                "pending_bytes": self._bytes,
                "pending_futures": len(self._pending),
                "peak_jobs": self._peak_jobs,
                "peak_bytes": self._peak_bytes,
                "peak_futures": self._peak_jobs,
                "max_jobs": self._max_jobs,
                "max_bytes": self._max_bytes,
                "max_futures": self._max_futures,
            }

    def _poison(self, error: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def _release(self, objects: Sequence["MemoryObj"]) -> None:
        for obj in objects:
            try:
                obj.ref_count_down()
            except BaseException as exc:
                self._poison(exc)

    def _run(
        self,
        ticket: Future[None],
        keys: tuple["CacheEngineKey", ...],
        objects: list["MemoryObj"],
    ) -> None:
        with self._condition:
            if ticket not in self._pending:
                return
        handed_off = False
        try:
            backend = self._manager.storage_backends.get("RemoteBackend")
            if type(backend) is not RemoteBackend:
                raise TypeError(
                    "Required put queue requires the standard RemoteBackend"
                )
            put = self._manager.batched_put_sync_required
            handed_off = True
            put(
                keys,
                objects,
                required_backends=("RemoteBackend",),
                location="RemoteBackend",
            )
        except BaseException as exc:
            self._poison(exc)
        finally:
            if not handed_off:
                self._release(objects)
            with self._condition:
                self._bytes -= self._pending.pop(ticket)
                try:
                    if self._error is None:
                        ticket.set_result(None)
                    else:
                        self._failed_tickets.append(ticket)
                        if not self._pending:
                            failed, self._failed_tickets = self._failed_tickets, []
                            for failed_ticket in failed:
                                failed_ticket.set_exception(self._error)
                finally:
                    self._condition.notify_all()
