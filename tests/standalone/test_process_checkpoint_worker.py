# SPDX-License-Identifier: Apache-2.0
"""Actual M CPU owner and W page views; native device fences are test boundaries."""

from concurrent.futures import Future
import gc
import importlib
from dataclasses import replace
from threading import Event, Lock, Thread
import time
from types import SimpleNamespace as NS

import pytest
import torch

from lmcache.integration.vllm.preemption_checkpoint import CaptureSpec, SealSpec
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    content_digest,
    StorageResult,
)
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator

from test_ascend_cpu_storage_runtime import ascend_modules, setup as cpu_setup

setup = cpu_setup


class RuntimeClient:
    """Execute the real CPU runtime, with immediate Futures in place of sockets."""

    def __init__(self, runtime):
        self.runtime, self.counter, self.requests = runtime, 0, {}
        self.lock = Lock()
        self.controls = []

    def request(
        self, method, payload, *, request_id, request_attempt, load_generation, purpose
    ):
        with self.lock:
            self.counter += 1
            identity = OperationIdentity(
                "cache",
                "epoch",
                "worker-0",
                self.counter,
                request_id,
                request_attempt,
                load_generation,
                purpose,
            )
        request = StorageRequest(
            identity, method, payload, content_digest((method, payload)), b"a"
        )
        self.requests[identity.key] = request
        self.controls.append((method, payload))
        return self._send(request)

    def _send(self, request):
        future = Future()
        future.identity, future.payload_digest = (
            request.identity,
            request.payload_digest,
        )
        try:
            future.set_result(self.runtime.handle(request))
        except BaseException as error:
            future.set_exception(error)
        return future

    def acknowledge_terminal(self, identity, payload=None):
        original = self.requests[identity.key]
        result = self.runtime.handle(
            StorageRequest(
                identity, "operation_status", {}, original.payload_digest, b"a"
            )
        )
        digest = result.value["terminal_result_digest"]
        return self._send(
            StorageRequest(
                identity,
                "acknowledge_terminal",
                payload or {"terminal_result_digest": digest},
                original.payload_digest,
                b"a",
            )
        )

    def retire(self, identity, digest, lease, proof):
        return self._send(
            StorageRequest(
                identity, "retire", {"lease_id": lease, "proof": proof}, digest, b"a"
            )
        )

    def cancel(self, identity):
        original = self.requests[identity.key]
        return self._send(
            StorageRequest(identity, "cancel", {}, original.payload_digest, b"a")
        )

    def replay(self, identity):
        return self._send(self.requests[identity.key])


@pytest.fixture
def worker(setup, monkeypatch):
    owner, runtime, _ = setup
    ascend_modules()
    base = importlib.import_module("lmcache_ascend.v1.preemption_checkpoint")
    module = importlib.import_module("lmcache_ascend.v1.process_checkpoint")
    tensor = torch.tensor
    monkeypatch.setattr(
        base,
        "torch",
        NS(
            long=torch.long,
            tensor=lambda data, **kw: tensor(
                data, **{k: v for k, v in kw.items() if k != "pin_memory"}
            ),
        ),
    )
    client, queued = RuntimeClient(runtime), []
    fences = NS(fail=False, preparations=0, enqueues=0)

    def synchronize():
        if fences.fail:
            raise RuntimeError("injected unknown device fence")
        for plan in queued:
            for layer, rows in enumerate(plan[0]):
                for row in rows:
                    row.tensor.fill_(plan[1] * 20 + layer + 1)
        queued.clear()

    def prepare(rows, starts, ends, **kw):
        fences.preparations += 1
        return rows, kw["kv_group"]

    def enqueue(plan):
        fences.enqueues += 1
        queued.append(plan)
        return NS(synchronize=synchronize)

    mapping = NS(
        shm_name=owner.allocator.shm_name,
        size=owner.allocator.buffer.numel(),
        generation=1,
    )
    context = NS(client=client, mapping=mapping, drain_guards=[])
    engine = NS(
        config=owner.config,
        metadata=runtime.metadata,
        storage_process=context,
        storage_layout=owner.layout,
        is_frozen=lambda: False,
        num_layers_for_group=runtime.num_layers_for_group,
        compact_source_importer=NS(release_restore=lambda *args: None),
        gpu_connector=NS(
            prepare_group_capture=prepare,
            enqueue_group_capture=enqueue,
            finish_checkpoint_capture=synchronize,
        ),
        shared_cpu_cache_passive_allocator=PassiveSharedViewAllocator(
            slab_tensor=owner.allocator.buffer, shm_name=mapping.shm_name, generation=1
        ),
    )
    instance = module.ProcessCheckpointWorker(engine)
    instance.timeout = 0.2
    yield instance, runtime, client, fences
    if not any(state.job.quarantined for state in instance.reservations.values()):
        instance.close()
    else:
        instance.executor.shutdown(wait=True)


def spec(end=9):
    return CaptureSpec("request", 1, 0, end, 0, ((1, 2, 3), (1, 2, 3)))


def finish(worker):
    deadline, results = time.monotonic() + 2, []
    while time.monotonic() < deadline:
        results.extend(worker.poll())
        if any(result.status in ("ready", "failed") for result in results):
            return results
        time.sleep(0.001)
    pytest.fail("checkpoint worker did not finish")


def wait_drained(instance):
    """Wait for completion callbacks without driving foreground worker polling."""
    deadline = time.monotonic() + 2
    while not instance.sources_drained() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert instance.sources_drained()


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert predicate()


def test_actual_mixed_pages_capture_and_accepted_seal_gc_off(worker):
    instance, runtime, client, fences = worker
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
        assert instance.poll()[0].status == "captured"
        assert runtime.registry.snapshot()["leases"] == 2
        assert not any(method == "commit_write" for method, _ in client.controls)
        instance.seal(SealSpec("request", 1, tuple(range(7))))
        assert [(r.status, r.end) for r in finish(instance)] == [("ready", 7)]
        wait_drained(instance)
        assert runtime.registry.snapshot()["leases"] == 0
        assert runtime.checkpoint_store.manifest("request", 1).tokens == tuple(range(7))
        assert fences.preparations == fences.enqueues == 2
        assert [m for m, p in client.controls].count("reserve_write") == 1
        assert [p["phase"] for m, p in client.controls if m == "commit_write"] == [
            "seal",
        ]
    finally:
        if was_enabled:
            gc.enable()


def test_cancel_private_capture_releases_only_after_fenced_capture(worker):
    instance, runtime, client, _ = worker
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    instance.cancel("request", 1)
    wait_drained(instance)
    assert runtime.registry.snapshot()["leases"] == 0
    assert (
        next(p for m, p in client.controls if m == "cancel_write")["proof"] == "fenced"
    )
    assert runtime.checkpoint_store.manifest("request", 1) is None


def test_invalid_second_group_geometry_retires_both_never_imported(worker, monkeypatch):
    instance, runtime, client, fences = worker
    real = client.request

    def malformed(method, payload, **kw):
        future = real(method, payload, **kw)
        if method == "reserve_write":
            future.result().value["groups"][1]["page_offsets"] = b"bad"
        return future

    monkeypatch.setattr(client, "request", malformed)
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "failed"
    wait_drained(instance)
    assert runtime.registry.snapshot()["leases"] == 0
    assert fences.preparations == 0
    assert (
        next(p for m, p in client.controls if m == "cancel_write")["proof"]
        == "never_imported"
    )


@pytest.mark.parametrize("fault", ["slab_generation", "captured_end"])
def test_rejected_checkpoint_identity_retires_every_unimported_export(
    worker, monkeypatch, fault
):
    instance, runtime, client, fences = worker
    real = client.request

    def malformed(method, payload, **kwargs):
        future = real(method, payload, **kwargs)
        if method == "reserve_write":
            result = future.result()
            result.value[fault] += 1
        return future

    monkeypatch.setattr(client, "request", malformed)
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "failed"
    wait_drained(instance)
    assert runtime.registry.snapshot()["leases"] == 0
    assert fences.preparations == fences.enqueues == 0


def test_unknown_native_fence_retains_real_pages_and_refuses_close(worker):
    instance, runtime, _, fences = worker
    fences.fail = True
    with pytest.raises(RuntimeError, match="source owners retained"):
        instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert runtime.registry.snapshot()["leases"] == 2
    assert not instance.sources_drained()
    with pytest.raises(RuntimeError, match="unresolved"):
        instance.poll()
    with pytest.raises(RuntimeError):
        instance.close()
    # Test cleanup uses a newly successful fake-device fence. Production cannot
    # infer this proof from Python lifetime or an elapsed service deadline.
    fences.fail = False
    instance.engine.gpu_connector.finish_checkpoint_capture()
    for job in instance.jobs.values():
        job.quarantined = False
    instance.cancel("request", 1)


def test_passive_rank_still_retains_restore_gate(worker):
    instance, _, client, _ = worker
    instance.writer = False
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert not client.controls
    instance.begin_restore("request", 1, 2)
    assert not instance.sources_drained()
    with pytest.raises(RuntimeError, match="all-worker"):
        instance.close()
    instance.release_restore("request", 1, 2)
    wait_drained(instance)


def test_cpu_pressure_preserves_paired_partial_capture(worker, monkeypatch):
    instance, runtime, _, fences = worker
    allocate = runtime.allocate_checkpoint_fragment

    def small(group, tokens, caches=None):
        if tokens > 2:
            raise MemoryError("injected pressure")
        return allocate(group, tokens, caches)

    monkeypatch.setattr(runtime, "allocate_checkpoint_fragment", small)
    monkeypatch.setattr(runtime, "reclaim_checkpoint_capacity", lambda *args: False)
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert [(r.status, r.end) for r in instance.poll()] == [("captured", 2)]
    instance.seal(SealSpec("request", 1, (0, 1)))
    assert [(r.status, r.end) for r in finish(instance)] == [("ready", 2)]
    assert fences.preparations == 2
    wait_drained(instance)


def test_duplicate_capture_reuses_existing_job_without_reopening_retired_prefix(worker):
    instance, _, client, fences = worker
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()

    def stale_descriptor(source):
        raise ValueError("original prefix source already retired")

    instance.engine.compact_source_importer.source_descriptor = stale_descriptor
    prefix_state = NS(prepared_sparse_sources={0: object()}, cached_keys=[])
    count = len(client.controls)
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4, prefix_state)
    assert [(r.status, r.end) for r in instance.poll()] == [("captured", 9)]
    assert len(client.controls) == count
    assert fences.preparations == 2
    instance.cancel("request", 1)


def test_reserve_timeout_reconciles_same_original_without_device_import(
    worker, monkeypatch
):
    instance, runtime, client, fences = worker
    real = client.request
    delayed = []

    def pause(method, payload, **kw):
        ready = real(method, payload, **kw)
        if method != "reserve_write":
            return ready
        pending = Future()
        pending.identity, pending.payload_digest = ready.identity, ready.payload_digest
        delayed.append((pending, ready.result()))
        return pending

    monkeypatch.setattr(client, "request", pause)
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "failed"
    assert not instance.sources_drained()
    assert fences.preparations == 0
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "failed"
    assert len(delayed) == 1
    pending, result = delayed[0]
    pending.set_result(result)
    instance.poll()
    wait_drained(instance)
    assert runtime.registry.snapshot()["leases"] == 0
    assert [m for m, p in client.controls].count("reserve_write") == 1
    assert fences.preparations == 0


def test_unknown_cpu_control_preserves_native_failure_class_and_quarantine(
    worker, monkeypatch
):
    from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError

    instance, _, _, _ = worker
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    job = instance.jobs["request", 1]
    original = instance.reservations["request", 1].future
    failed = Future()
    failed.identity, failed.payload_digest = original.identity, original.payload_digest
    failed.set_result(
        StorageResult(
            original.identity, "FAILED_UNKNOWN", error="injected CPU uncertainty"
        )
    )
    with monkeypatch.context() as patch:
        patch.setattr(instance, "submit_control", lambda *args: failed)
        with pytest.raises(NativeExternalPageTransferUnknownError) as error:
            instance._call(job, "commit_write", {})
        assert error.value.terminal_future is failed
        assert "commit_write" in error.value.operation
        assert job.quarantined and not instance.sources_drained()
        with pytest.raises(RuntimeError):
            instance.close()
    # Only the test injected an unexecuted CPU response; recover its fake state.
    job.quarantined = False
    instance.cancel("request", 1)


def test_cancel_returns_pending_and_idle_completion_retires_once_gc_off(
    worker, monkeypatch
):
    instance, runtime, client, _ = worker
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    real, delayed = client.request, []

    def pause(method, payload, **kwargs):
        ready = real(method, payload, **kwargs)
        if method != "cancel_write":
            return ready
        future = Future()
        future.identity, future.payload_digest = ready.identity, ready.payload_digest
        delayed.append((future, ready.result()))
        return future

    monkeypatch.setattr(client, "request", pause)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        instance.cancel("request", 1)
        wait_until(lambda: bool(delayed))
        assert not delayed[0][0].done()
        assert not instance.jobs
        assert runtime.registry.snapshot()["leases"] == 2
        assert not instance.sources_drained()
        instance.cancel("request", 1)
        instance.cancel("request", 1)
        # Cleanup yields while its original Future, views and job remain held;
        # completion must progress without another model step.
        state = instance.reservations["request", 1]
        wait_until(lambda: state.cleanup_future.done())
        assert not state.released and not state.job.quarantined
        assert len(delayed) == 1
        delayed[0][0].set_result(delayed[0][1])
        wait_drained(instance)
        assert runtime.registry.snapshot()["leases"] == 0
        assert runtime.checkpoint_store.manifest("request", 1) is None
        assert [m for m, _ in client.controls].count("cancel_write") == 1
    finally:
        if was_enabled:
            gc.enable()


def test_serving_cancel_returns_while_its_response_is_held(worker, monkeypatch):
    instance, runtime, client, _ = worker
    instance.timeout = 1.0
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    real, delayed = client.request, []
    submitted, returned = Event(), Event()

    def pause(method, payload, **kwargs):
        ready = real(method, payload, **kwargs)
        if method != "cancel_write":
            return ready
        future = Future()
        future.identity, future.payload_digest = ready.identity, ready.payload_digest
        delayed.append((future, ready.result()))
        submitted.set()
        return future

    monkeypatch.setattr(client, "request", pause)

    def cancel():
        instance.cancel("request", 1)
        returned.set()

    caller = Thread(target=cancel)
    caller.start()
    try:
        assert submitted.wait(1)
        assert returned.wait(0.2), "serving caller waited for an unfinished response"
        assert not delayed[0][0].done()
        assert runtime.registry.snapshot()["leases"] == 2
    finally:
        if delayed:
            delayed[0][0].set_result(delayed[0][1])
        caller.join(2)
    wait_drained(instance)


def test_unfinished_cancel_does_not_occupy_the_checkpoint_seal_executor(
    worker, monkeypatch
):
    instance, runtime, client, _ = worker
    instance.timeout = 1.0
    for current in (spec(4), replace(spec(4), req_id="other")):
        instance.capture(current, {0: [None] * 2, 1: [None] * 3}, 4)
    assert [result.status for result in instance.poll()] == ["captured", "captured"]
    real, delayed = client.request, []

    def pause(method, payload, **kwargs):
        ready = real(method, payload, **kwargs)
        if method != "cancel_write":
            return ready
        future = Future()
        future.identity, future.payload_digest = ready.identity, ready.payload_digest
        delayed.append((future, ready.result()))
        return future

    monkeypatch.setattr(client, "request", pause)
    instance.cancel("request", 1)
    wait_until(lambda: bool(delayed))
    try:
        instance.seal(SealSpec("other", 1, tuple(range(3))))
        published = instance.jobs["other", 1].future
        assert published.result(0.2) == 3
        assert not delayed[0][0].done()
        assert [(r.req_id, r.status) for r in instance.poll()] == [("other", "ready")]
        wait_until(lambda: ("other", 1) not in instance.reservations)
        assert runtime.registry.snapshot()["leases"] == 2
    finally:
        delayed[0][0].set_result(delayed[0][1])
    wait_drained(instance)


def test_cancel_during_seal_forgets_final_publication_without_foreground_wait(
    worker, monkeypatch
):
    instance, runtime, client, _ = worker
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    real, delayed = client.request, []

    def pause(method, payload, **kwargs):
        ready = real(method, payload, **kwargs)
        if method != "commit_write":
            return ready
        future = Future()
        future.identity, future.payload_digest = ready.identity, ready.payload_digest
        delayed.append((future, ready.result()))
        return future

    monkeypatch.setattr(client, "request", pause)
    instance.seal(SealSpec("request", 1, tuple(range(7))))
    wait_until(lambda: bool(delayed))
    instance.cancel("request", 1)
    assert not delayed[0][0].done()
    assert not instance.sources_drained()
    delayed[0][0].set_result(delayed[0][1])
    wait_drained(instance)
    assert runtime.checkpoint_store.manifest("request", 1) is None
    assert runtime.registry.snapshot()["leases"] == 0
    assert not any(m == "cancel_write" for m, _ in client.controls)


def test_late_reservation_drains_without_foreground_poll(worker, monkeypatch):
    instance, runtime, client, fences = worker
    real, delayed = client.request, []

    def pause(method, payload, **kwargs):
        ready = real(method, payload, **kwargs)
        if method != "reserve_write":
            return ready
        future = Future()
        future.identity, future.payload_digest = ready.identity, ready.payload_digest
        delayed.append((future, ready.result()))
        return future

    monkeypatch.setattr(client, "request", pause)
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    state = instance.reservations["request", 1]
    wait_until(lambda: state.cleanup_future.done())
    assert not state.released and fences.preparations == 0
    delayed[0][0].set_result(delayed[0][1])
    wait_drained(instance)
    assert runtime.registry.snapshot()["leases"] == 0
    assert fences.preparations == 0


def test_close_drains_pending_cleanup_before_executor_shutdown(worker, monkeypatch):
    instance, _, client, _ = worker
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    real, delayed = client.request, []

    def pause(method, payload, **kwargs):
        ready = real(method, payload, **kwargs)
        if method != "cancel_write":
            return ready
        future = Future()
        future.identity, future.payload_digest = ready.identity, ready.payload_digest
        delayed.append((future, ready.result()))
        return future

    monkeypatch.setattr(client, "request", pause)
    instance.cancel("request", 1)
    wait_until(lambda: bool(delayed))
    closed = Event()
    failures = []

    def close():
        try:
            instance.close()
            closed.set()
        except BaseException as error:
            failures.append(error)

    closer = Thread(target=close)
    closer.start()
    assert not closed.wait(0.01)
    delayed[0][0].set_result(delayed[0][1])
    closer.join(2)
    assert closed.is_set() and not failures
    assert instance.sources_drained()


def test_pending_retirement_counts_against_existing_capture_admission(
    worker, monkeypatch
):
    instance, runtime, client, fences = worker
    instance.max_jobs = 1
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    real, delayed = client.request, []

    def pause(method, payload, **kwargs):
        ready = real(method, payload, **kwargs)
        if method != "cancel_write":
            return ready
        future = Future()
        future.identity, future.payload_digest = ready.identity, ready.payload_digest
        delayed.append((future, ready.result()))
        return future

    monkeypatch.setattr(client, "request", pause)
    instance.cancel("request", 1)
    wait_until(lambda: bool(delayed))
    assert not instance.jobs and not instance.sources_drained()
    instance.capture(replace(spec(), generation=2), {0: [None] * 2, 1: [None] * 3}, 4)
    assert [(r.status, r.reason) for r in instance.poll()] == [
        ("failed", "checkpoint staging is busy")
    ]
    assert fences.preparations == 2
    assert [m for m, _ in client.controls].count("reserve_write") == 1
    assert runtime.registry.snapshot()["leases"] == 2
    delayed[0][0].set_result(delayed[0][1])
    wait_drained(instance)


def test_close_includes_final_cancelled_seal_forget_in_drain(worker, monkeypatch):
    instance, _, client, _ = worker
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    instance.seal(SealSpec("request", 1, tuple(range(7))))
    wait_until(lambda: instance.jobs["request", 1].future.done())
    instance.cancel("request", 1)
    wait_drained(instance)
    assert instance.jobs
    real, forgets, delayed = client.request, [], []

    def pause(method, payload, **kwargs):
        ready = real(method, payload, **kwargs)
        if method != "checkpoint_forget":
            return ready
        forgets.append(ready)
        if len(forgets) == 1:
            return ready
        future = Future()
        future.identity, future.payload_digest = ready.identity, ready.payload_digest
        delayed.append((future, ready.result()))
        return future

    monkeypatch.setattr(client, "request", pause)
    closed = Event()
    closer = Thread(target=lambda: (instance.close(), closed.set()))
    closer.start()
    try:
        wait_until(lambda: bool(delayed))
        assert not closed.wait(0.01)
    finally:
        for future, result in delayed:
            future.set_result(result)
        closer.join(2)
    assert closed.is_set() and instance.sources_drained()


@pytest.mark.parametrize("status", ["BUSY", "BUSY_CANCEL_BUSY", "PENDING"])
def test_cancel_retries_only_after_terminal_refusal(worker, monkeypatch, status):
    instance, runtime, client, _ = worker
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    real, real_cancel, refused = runtime.handle, runtime.cancel_write, []

    def refuse(request):
        if request.method == "cancel_write" and not refused:
            if status.startswith("BUSY"):
                refused.append(request.identity)
                return StorageResult(request.identity, "BUSY")
        return real(request)

    if status == "PENDING":

        def pending(value, identity):
            if not refused:
                refused.append(identity)
                return {"status": "PENDING"}
            return real_cancel(value, identity)

        monkeypatch.setattr(runtime, "cancel_write", pending)

    monkeypatch.setattr(runtime, "handle", refuse)
    cancellations = []
    if status == "BUSY_CANCEL_BUSY":
        real_cancel_control = client.cancel

        def cancel(identity):
            cancellations.append(identity)
            if len(cancellations) == 1:
                future = Future()
                future.set_result(StorageResult(identity, "BUSY"))
                return future
            return real_cancel_control(identity)

        monkeypatch.setattr(client, "cancel", cancel)
    instance.cancel("request", 1)
    wait_drained(instance)
    assert len(refused) == 1
    assert [m for m, _ in client.controls].count("cancel_write") == 2
    if cancellations:
        assert cancellations == [refused[0], refused[0]]
    assert runtime.registry.snapshot()["leases"] == 0
