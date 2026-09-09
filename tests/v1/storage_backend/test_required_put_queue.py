# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests using the real strict manager and batched RemoteBackend path."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from queue import Queue
from threading import Condition, Event, Lock, Thread, current_thread
from typing import Any, Callable, Iterator
from unittest.mock import Mock
import asyncio
import gc
import time
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObjMetadata
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.required_put_queue import RequiredPutQueue
from lmcache.v1.storage_backend.storage_manager import StorageManager


class Buffer:
    """Metadata-only buffer with explicit caller and connector references."""

    def __init__(self, size: int = 8, refs: int = 1) -> None:
        self.metadata = MemoryObjMetadata(
            shape=torch.Size([size]),
            dtype=torch.uint8,
            address=0,
            phy_size=size + 4096,
            ref_count=refs,
        )
        self.lock = Lock()
        self.sized = Event()
        self.size_error: BaseException | None = None
        self.release_error: BaseException | None = None

    def get_size(self) -> int:
        self.sized.set()
        if self.size_error is not None:
            raise self.size_error
        return self.metadata.get_size()

    def ref_count_up(self) -> None:
        with self.lock:
            self.metadata.ref_count += 1

    def ref_count_down(self) -> None:
        with self.lock:
            assert self.metadata.ref_count > 0
            self.metadata.ref_count -= 1
        if self.release_error is not None:
            raise self.release_error


class StorageFailure(BaseException):
    """Exercise BaseException handling without stopping the asyncio event loop."""


@dataclass
class Batch:
    keys: tuple[Any, ...]
    objects: tuple[Buffer, ...]
    completion: Future


class Connector:
    def __init__(self) -> None:
        self.batched = True
        self.required = True
        self.calls: Queue[Batch] = Queue()
        self.completions: list[Future] = []
        self.finishing = False

    def support_batched_put(self) -> bool:
        return self.batched

    def requires_put_completion(self) -> bool:
        return self.required

    async def batched_put(self, keys: Any, objects: list[Buffer]) -> Any:
        completion: Future = Future()
        completion.set_running_or_notify_cancel()
        self.completions.append(completion)
        self.calls.put(Batch(tuple(keys), tuple(objects), completion))
        if self.finishing:
            completion.set_result(None)
        try:
            return await asyncio.wrap_future(completion)
        finally:
            for obj in objects:
                obj.ref_count_down()

    def finish(self) -> None:
        self.finishing = True
        for completion in self.completions:
            if not completion.done():
                completion.set_result(None)


class StrictManager(StorageManager):
    """Initialize only the state used by the inherited, unmodified strict API."""

    def __init__(self, remote: RemoteBackend, local: Mock) -> None:
        self.allocator_backend = local
        self.storage_backends = {"RemoteBackend": remote, "LocalCPUBackend": local}
        self._bypass_lock = Lock()
        self._freeze_lock = Lock()
        self._bypassed_backends: set[str] = set()
        self._freeze = False


class Harness:
    def __init__(self, manager: StrictManager, connector: Connector) -> None:
        self.manager = manager
        self.connector = connector
        self.queues: list[RequiredPutQueue] = []
        self.threads: list[Thread] = []

    def queue(
        self, max_jobs: int = 2, max_bytes: int = 64, max_futures: int = 2
    ) -> RequiredPutQueue:
        queue = RequiredPutQueue(
            self.manager,
            max_jobs=max_jobs,
            max_bytes=max_bytes,
            max_futures=max_futures,
        )
        self.queues.append(queue)
        return queue

    def call(self, fn: Callable[[], Any]) -> Future:
        result: Future = Future()

        def run() -> None:
            try:
                value = fn()
            except BaseException as exc:
                result.set_exception(exc)
            else:
                result.set_result(value)

        thread = Thread(target=run, daemon=True)
        self.threads.append(thread)
        thread.start()
        return result

    def batch(self) -> Batch:
        return self.connector.calls.get(timeout=5)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    connector = Connector()
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.remote_backend.CreateConnector",
        lambda *args: connector,
    )
    loop = asyncio.new_event_loop()
    loop_thread = Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    local = Mock()
    remote = RemoteBackend(
        LMCacheEngineConfig.from_defaults(
            remote_url="mooncakestore://test", remote_serde="naive"
        ),
        LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.uint8,
            kv_shape=(1, 1, 1, 1, 1),
        ),
        loop,
        local,
        dst_device="cpu",
    )
    harness = Harness(StrictManager(remote, local), connector)
    try:
        yield harness
    finally:
        finished = Event()

        def finish() -> None:
            connector.finish()
            finished.set()

        loop.call_soon_threadsafe(finish)
        assert finished.wait(5)
        for queue in harness.queues:
            try:
                queue.close()
            except BaseException:
                pass
        for thread in harness.threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()
        local.batched_submit_put_task.assert_not_called()


def wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def assert_blocked(future: Future) -> None:
    with pytest.raises(TimeoutError):
        future.result(timeout=0.03)


@pytest.mark.parametrize("limit", ["max_jobs", "max_bytes", "max_futures"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits(harness: Harness, limit: str, value: Any) -> None:
    limits = dict(max_jobs=2, max_bytes=64, max_futures=2)
    limits[limit] = value
    with pytest.raises(ValueError, match=limit):
        RequiredPutQueue(harness.manager, **limits)


def test_lazy_empty_and_stats(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = Mock(side_effect=AssertionError("executor must remain lazy"))
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.required_put_queue.ThreadPoolExecutor", executor
    )
    queue = harness.queue()
    ticket = queue.submit([], [])
    assert ticket.result() is None
    assert ticket.cancel() is False
    queue.drain()
    queue.close()
    queue.close()
    executor.assert_not_called()
    assert queue.stats() == {
        "pending_jobs": 0,
        "pending_bytes": 0,
        "pending_futures": 0,
        "peak_jobs": 0,
        "peak_bytes": 0,
        "peak_futures": 0,
        "max_jobs": 2,
        "max_bytes": 64,
        "max_futures": 2,
    }


@pytest.mark.parametrize(
    "limits",
    [
        dict(max_jobs=1, max_bytes=64, max_futures=4),
        dict(max_jobs=4, max_bytes=8, max_futures=4),
        dict(max_jobs=4, max_bytes=64, max_futures=1),
    ],
    ids=["jobs", "bytes", "futures"],
)
def test_delayed_future_backpressure(harness: Harness, limits: dict[str, int]) -> None:
    queue = harness.queue(**limits)
    first, second = Buffer(), Buffer()
    first_ticket = queue.submit(["first"], [first])
    first_batch = harness.batch()
    submission = harness.call(lambda: queue.submit(["second"], [second]))
    assert second.sized.wait(5)
    assert_blocked(submission)
    assert queue.pending_jobs == queue.pending_futures == 1
    assert queue.pending_bytes == 8
    assert first.metadata.ref_count == 2  # Caller loan plus connector ownership.
    assert second.metadata.ref_count == 1
    first_batch.completion.set_result(None)
    assert first_ticket.result(timeout=5) is None
    second_ticket = submission.result(timeout=5)
    harness.batch().completion.set_result(None)
    second_ticket.result(timeout=5)
    queue.drain()
    assert first.metadata.ref_count == second.metadata.ref_count == 0
    assert queue.pending_jobs == queue.pending_bytes == queue.pending_futures == 0
    assert queue.stats()["peak_jobs"] == 1


def test_admission_waits_oldest_not_any_completion(harness: Harness) -> None:
    queue = harness.queue()
    oldest = queue.submit(["oldest"], [Buffer()])
    oldest_batch = harness.batch()
    newer = queue.submit(["newer"], [Buffer()])
    newer_batch = harness.batch()
    third = Buffer()
    submission = harness.call(lambda: queue.submit(["third"], [third]))
    assert third.sized.wait(5)
    assert_blocked(submission)
    newer_batch.completion.set_result(None)
    newer.result(timeout=5)
    assert queue.pending_jobs == 1
    assert_blocked(submission)
    oldest_batch.completion.set_result(None)
    oldest.result(timeout=5)
    ticket = submission.result(timeout=5)
    harness.batch().completion.set_result(None)
    ticket.result(timeout=5)


def test_79_row_pages_fit_two_jobs_under_64_mib(harness: Harness) -> None:
    mib = 1024 * 1024
    page_bytes = 22 * mib
    queue = harness.queue(max_bytes=64 * mib)
    pages = [
        [Buffer(page_bytes // 79 + (i < page_bytes % 79)) for i in range(79)]
        for _ in range(2)
    ]
    tickets = [queue.submit(list(range(79)), page) for page in pages]
    batches = [harness.batch(), harness.batch()]
    assert all(len(batch.objects) == 79 for batch in batches)
    assert queue.pending_jobs == queue.pending_futures == 2
    assert queue.pending_bytes == 44 * mib
    for batch in batches:
        batch.completion.set_result(None)
    for ticket in tickets:
        ticket.result(timeout=5)
    queue.drain()
    assert all(obj.metadata.ref_count == 0 for page in pages for obj in page)
    assert queue.stats()["peak_bytes"] == 44 * mib


def test_queued_jobs_count_and_snapshot_inputs(harness: Harness) -> None:
    queue = harness.queue(max_jobs=3, max_futures=3)
    tickets = [queue.submit([i], [Buffer()]) for i in range(2)]
    batches = [harness.batch(), harness.batch()]
    original, replacement = Buffer(), Buffer()
    keys, objects = ["original"], [original]
    tickets.append(queue.submit(keys, objects))
    keys[0] = "mutated"
    objects[0] = replacement
    assert queue.pending_jobs == queue.pending_futures == 3
    assert queue.pending_bytes == 24
    assert harness.connector.calls.empty()
    assert original.metadata.ref_count == 1
    for batch in batches:
        batch.completion.set_result(None)
    queued_batch = harness.batch()
    assert queued_batch.keys == ("original",)
    assert queued_batch.objects == (original,)
    queued_batch.completion.set_result(None)
    for ticket in tickets:
        ticket.result(timeout=5)
        assert ticket.cancel() is False
    assert original.metadata.ref_count == 0
    assert replacement.metadata.ref_count == 1


def test_duplicate_occurrences_consume_loans_and_count_logical_bytes(
    harness: Harness,
) -> None:
    queue = harness.queue()
    obj = Buffer(size=7, refs=3)  # One manifest reference and two loans.
    ticket = queue.submit(["a", "b"], [obj, obj])
    batch = harness.batch()
    assert queue.pending_bytes == 14
    assert obj.metadata.ref_count == 5
    batch.completion.set_result(None)
    ticket.result(timeout=5)
    assert obj.metadata.ref_count == 1


@pytest.mark.parametrize("failure", ["oversize", "mismatch", "size", "negative"])
def test_rejected_input_consumed_and_queue_poisoned(
    harness: Harness, failure: str
) -> None:
    queue = harness.queue(max_bytes=8)
    obj = Buffer(size=5, refs=2)
    keys = ["a", "b"]
    if failure == "mismatch":
        keys = ["a"]
    elif failure == "size":
        obj.size_error = RuntimeError("size error")
    elif failure == "negative":
        obj.metadata.shape = torch.Size([-1])
    with pytest.raises((ValueError, RuntimeError)) as first:
        queue.submit(keys, [obj, obj])
    assert obj.metadata.ref_count == 0
    assert harness.connector.calls.empty()
    assert queue.stats()["peak_jobs"] == 0
    later = Buffer()
    with pytest.raises(type(first.value)) as again:
        queue.submit(["later"], [later])
    assert again.value is first.value
    assert later.metadata.ref_count == 0
    with pytest.raises(type(first.value)):
        queue.close()


def test_key_snapshot_error_still_consumes_all_input_loans(harness: Harness) -> None:
    class BrokenKeys(list):
        def __iter__(self) -> Iterator[Any]:
            raise RuntimeError("key snapshot")

    queue = harness.queue()
    obj = Buffer(refs=2)
    with pytest.raises(RuntimeError, match="key snapshot"):
        queue.submit(BrokenKeys(["a", "b"]), [obj, obj])
    assert obj.metadata.ref_count == 0
    assert queue.pending_jobs == 0
    assert harness.connector.calls.empty()


@pytest.mark.parametrize("failure", ["disconnected", "unbatched", "completion", "type"])
def test_unsupported_remote_fails_closed(harness: Harness, failure: str) -> None:
    queue = harness.queue()
    remote = harness.manager.storage_backends["RemoteBackend"]
    if failure == "disconnected":
        remote.connection = None
    elif failure == "unbatched":
        harness.connector.batched = False
    elif failure == "completion":
        harness.connector.required = False
    else:
        harness.manager.storage_backends["RemoteBackend"] = Mock()
    obj = Buffer()
    ticket = queue.submit(["key"], [obj])
    with pytest.raises((RuntimeError, TypeError)):
        ticket.result(timeout=5)
    assert obj.metadata.ref_count == 0
    assert harness.connector.calls.empty()


@pytest.mark.parametrize("failure", ["exception", "false", "base_exception"])
def test_failure_drains_siblings_and_never_reports_safe_success(
    harness: Harness, failure: str
) -> None:
    queue = harness.queue(max_jobs=3, max_futures=3)
    objects = [Buffer() for _ in range(3)]
    tickets = [queue.submit([i], [obj]) for i, obj in enumerate(objects)]
    failed_batch, sibling_batch = harness.batch(), harness.batch()
    if failure == "false":
        failed_batch.completion.set_result(False)
        error_type = RuntimeError
    else:
        error_type = RuntimeError if failure == "exception" else StorageFailure
        failed_batch.completion.set_exception(error_type("native failure"))
    # The third admitted job runs even after poison; nothing is cancelled.
    queued_batch = harness.batch()
    assert queue.pending_jobs == 2
    assert_blocked(tickets[0])
    assert_blocked(tickets[1])
    draining = harness.call(queue.drain)
    closing = harness.call(queue.close)
    rejected = Buffer()
    submission = harness.call(lambda: queue.submit(["rejected"], [rejected]))
    assert rejected.sized.wait(5)
    wait_for(lambda: rejected.metadata.ref_count == 0)
    for result in (draining, closing, submission):
        assert_blocked(result)
    sibling_batch.completion.set_result(None)
    wait_for(lambda: queue.pending_jobs == 1)
    for ticket in tickets:
        assert not ticket.done()
        assert ticket.cancel() is False
    queued_batch.completion.set_result(None)
    errors = []
    for result in (*tickets, draining, closing, submission):
        with pytest.raises(error_type) as raised:
            result.result(timeout=5)
        errors.append(raised.value)
    assert all(error is errors[0] for error in errors)
    assert all(obj.metadata.ref_count == 0 for obj in objects)
    assert queue.pending_jobs == queue.pending_bytes == queue.pending_futures == 0


def test_oversize_error_waits_for_admitted_work(harness: Harness) -> None:
    queue = harness.queue(max_bytes=8)
    ticket = queue.submit(["running"], [Buffer()])
    batch = harness.batch()
    oversized = Buffer(size=9)
    submission = harness.call(lambda: queue.submit(["oversized"], [oversized]))
    wait_for(lambda: oversized.metadata.ref_count == 0)
    assert_blocked(submission)
    batch.completion.set_result(None)
    for result in (submission, ticket):
        with pytest.raises(ValueError, match="exceeds max_bytes"):
            result.result(timeout=5)


def test_interrupted_drain_still_waits_and_poisons_tickets(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    interrupted = Event()

    class InterruptedCondition(Condition):
        def wait(self, timeout: float | None = None) -> bool:
            if not interrupted.is_set():
                interrupted.set()
                raise KeyboardInterrupt("drain interrupted")
            return super().wait(timeout)

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.required_put_queue.Condition", InterruptedCondition
    )
    queue = harness.queue()
    obj = Buffer()
    ticket = queue.submit(["key"], [obj])
    batch = harness.batch()
    draining = harness.call(queue.drain)
    assert interrupted.wait(5)
    assert_blocked(draining)
    assert obj.metadata.ref_count == 2
    batch.completion.set_result(None)
    for result in (ticket, draining):
        with pytest.raises(KeyboardInterrupt, match="drain interrupted"):
            result.result(timeout=5)
    assert obj.metadata.ref_count == 0


def test_cancel_and_result_timeout_never_retire_io(harness: Harness) -> None:
    queue = harness.queue(max_jobs=3, max_futures=3)
    objects = [Buffer() for _ in range(3)]
    tickets = [queue.submit([i], [obj]) for i, obj in enumerate(objects)]
    batches = [harness.batch(), harness.batch()]
    for ticket in tickets:
        assert ticket.running()
        assert ticket.cancel() is False
        assert_blocked(ticket)
    assert queue.pending_jobs == queue.pending_futures == 3
    assert queue.pending_bytes == 24
    assert all(obj.metadata.ref_count > 0 for obj in objects)
    closing = harness.call(queue.close)
    assert_blocked(closing)
    for batch in batches:
        batch.completion.set_result(None)
    harness.batch().completion.set_result(None)
    closing.result(timeout=5)
    for ticket in tickets:
        ticket.result(timeout=5)
    assert all(obj.metadata.ref_count == 0 for obj in objects)
    queue.close()


def test_close_prevents_submits_consuming_their_references(harness: Harness) -> None:
    queue = harness.queue()
    queue.close()
    obj = Buffer()
    with pytest.raises(RuntimeError, match="closed"):
        queue.submit(["key"], [obj])
    assert obj.metadata.ref_count == 0


def test_close_wakes_blocked_admission_and_drains_before_errors(
    harness: Harness,
) -> None:
    queue = harness.queue(max_jobs=1)
    first, second = Buffer(), Buffer()
    ticket = queue.submit(["first"], [first])
    batch = harness.batch()
    submission = harness.call(lambda: queue.submit(["second"], [second]))
    assert second.sized.wait(5)
    assert_blocked(submission)
    closing = harness.call(queue.close)
    wait_for(lambda: second.metadata.ref_count == 0)
    assert queue.pending_jobs == 1
    assert_blocked(closing)
    assert_blocked(submission)
    batch.completion.set_result(None)
    for result in (ticket, submission, closing):
        with pytest.raises(RuntimeError, match="closed"):
            result.result(timeout=5)
    assert first.metadata.ref_count == second.metadata.ref_count == 0
    assert harness.connector.calls.empty()


@pytest.mark.parametrize("failure", ["constructor", "enqueue", "thread_start"])
def test_executor_failures_consume_without_late_double_handoff(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    queue = harness.queue()
    error = RuntimeError(failure)
    if failure == "constructor":
        monkeypatch.setattr(
            "lmcache.v1.storage_backend.required_put_queue.ThreadPoolExecutor",
            Mock(side_effect=error),
        )
    elif failure == "enqueue":
        monkeypatch.setattr(ThreadPoolExecutor, "submit", Mock(side_effect=error))
    else:
        monkeypatch.setattr(Thread, "start", Mock(side_effect=error))
    obj = Buffer()
    with pytest.raises(RuntimeError) as raised:
        queue.submit(["key"], [obj])
    assert raised.value is error
    assert obj.metadata.ref_count == 0
    assert queue.pending_jobs == queue.pending_bytes == queue.pending_futures == 0
    assert queue.stats()["peak_jobs"] == 0
    monkeypatch.undo()
    with pytest.raises(RuntimeError):
        queue.close()
    assert harness.connector.calls.empty()


def test_failed_second_thread_start_drains_first_and_discards_enqueued_job(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = harness.queue()
    first = Buffer()
    ticket = queue.submit(["first"], [first])
    batch = harness.batch()
    original_start = Thread.start

    def start(thread: Thread) -> None:
        if thread.name.startswith("required-put"):
            raise RuntimeError("thread start")
        original_start(thread)

    monkeypatch.setattr(Thread, "start", start)
    second = Buffer()
    submission = harness.call(lambda: queue.submit(["second"], [second]))
    wait_for(lambda: second.metadata.ref_count == 0)
    assert_blocked(submission)
    assert queue.pending_jobs == 1
    batch.completion.set_result(None)
    for result in (ticket, submission):
        with pytest.raises(RuntimeError, match="thread start"):
            result.result(timeout=5)
    with pytest.raises(RuntimeError, match="thread start"):
        queue.close()
    assert first.metadata.ref_count == second.metadata.ref_count == 0
    assert harness.connector.calls.empty()


@pytest.mark.parametrize("failure", ["serialization", "missing_future", "submission"])
def test_worker_submission_exceptions_consume_references(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    queue = harness.queue()
    remote = harness.manager.storage_backends["RemoteBackend"]
    if failure == "serialization":
        monkeypatch.setattr(
            remote.serializer, "serialize", Mock(side_effect=RuntimeError("serialize"))
        )
    else:
        monkeypatch.setattr(
            remote,
            "batched_submit_put_task",
            Mock(
                return_value=None,
                side_effect=RuntimeError("submit") if failure == "submission" else None,
            ),
        )
    objects = [Buffer(), Buffer()]
    ticket = queue.submit(["a", "b"], objects)
    with pytest.raises(RuntimeError):
        ticket.result(timeout=5)
    assert all(obj.metadata.ref_count == 0 for obj in objects)
    with pytest.raises(RuntimeError):
        queue.drain()


def test_release_error_does_not_skip_rejected_siblings(harness: Harness) -> None:
    queue = harness.queue(max_bytes=1)
    objects = [Buffer(), Buffer()]
    objects[0].release_error = RuntimeError("release")
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        queue.submit(["a", "b"], objects)
    assert all(obj.metadata.ref_count == 0 for obj in objects)


def test_concurrent_submit_respects_global_budgets(harness: Harness) -> None:
    queue = harness.queue(max_jobs=3, max_bytes=24, max_futures=3)
    objects = [Buffer() for _ in range(16)]
    submissions = [
        harness.call(lambda i=i, obj=obj: queue.submit([i], [obj]))
        for i, obj in enumerate(objects)
    ]
    for _ in objects:
        batch = harness.batch()
        stats = queue.stats()
        assert stats["pending_jobs"] <= 3
        assert stats["pending_futures"] <= 3
        assert stats["pending_bytes"] <= 24
        batch.completion.set_result(None)
    for submission in submissions:
        submission.result(timeout=5).result(timeout=5)
    queue.close()
    assert all(obj.metadata.ref_count == 0 for obj in objects)


def test_completed_jobs_release_tickets_and_inputs_without_drain(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = harness.queue()
    threads: list[str] = []
    original = harness.manager.batched_put_sync_required

    def put(*args: Any, **kwargs: Any) -> None:
        threads.append(current_thread().name)
        assert kwargs == {
            "required_backends": ("RemoteBackend",),
            "location": "RemoteBackend",
        }
        original(*args, **kwargs)

    monkeypatch.setattr(harness.manager, "batched_put_sync_required", put)
    references = []
    for i in range(20):
        obj = Buffer()
        ticket = queue.submit([i], [obj])
        batch = harness.batch()
        batch.completion.set_result(None)
        ticket.result(timeout=5)
        references.append((weakref.ref(obj), weakref.ref(ticket)))
        del obj, ticket, batch
    wait_for(lambda: queue.pending_jobs == 0)
    gc.collect()
    assert all(obj() is None and ticket() is None for obj, ticket in references)
    assert all(name.startswith("required-put") for name in threads)
    assert queue.pending_bytes == queue.pending_futures == 0
    assert queue.stats()["peak_jobs"] == 1
