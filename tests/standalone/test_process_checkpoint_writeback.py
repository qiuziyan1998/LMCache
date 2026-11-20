# SPDX-License-Identifier: Apache-2.0
"""Combined M ownership/writeback and W capture; native transfers are simulated."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
import ctypes
import gc
import time

import pytest

from lmcache.integration.vllm.preemption_checkpoint import SealSpec
from lmcache.v1.rpc.storage_protocol import PrepareSpec, wire_hash
import test_ascend_cpu_storage_runtime as cpu
import test_process_checkpoint_worker as capture


@pytest.fixture
def setup(monkeypatch):
    original = cpu.CPUStorage.__init__

    def initialize(self):
        original(self)
        self.config.decode_checkpoint_retention_max_bytes = 60000
        self.config.decode_checkpoint_free_reserve_bytes = 1024

    monkeypatch.setattr(cpu.CPUStorage, "__init__", initialize)
    # The shared fixture explicitly selects local-only operation; enable the
    # actual M writeback constructor before its runtime is created instead.
    factory, module = cpu.ascend_modules()
    runtime_init = module.AscendCPUCacheRuntime.__init__

    def with_writeback(self, config, *args, **kwargs):
        config.decode_checkpoint_writeback = True
        runtime_init(self, config, *args, **kwargs)

    monkeypatch.setattr(module.AscendCPUCacheRuntime, "__init__", with_writeback)
    generator = cpu.setup.__wrapped__()
    value = next(generator)
    try:
        yield value
    finally:
        runtime = value[1]
        runtime.checkpoint_store.close()
        runtime.checkpoint_retire_executor.shutdown(wait=True)
        with pytest.raises(StopIteration):
            next(generator)


@pytest.fixture
def worker(setup, monkeypatch):
    yield from capture.worker.__wrapped__(setup, monkeypatch)


def wait(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.002)


def test_process_capture_spills_and_restores_both_groups(worker, monkeypatch):
    instance, runtime, client, _ = worker
    assert type(instance.local).__name__ == "_CheckpointControls"
    assert type(runtime.checkpoint_store).__name__ == "CheckpointWritebackStore"
    remote = {}

    def put(keys, pointers, sizes, owners, events, request):
        for key, row, widths in zip(keys, pointers, sizes, strict=True):
            remote[key] = b"".join(
                ctypes.string_at(p, n) for p, n in zip(row, widths, strict=True)
            )
        future = Future()
        future.set_result(None)
        return future

    def get(keys, pointers, sizes, owners, request):
        for key, row, widths in zip(keys, pointers, sizes, strict=True):
            offset = 0
            for pointer, size in zip(row, widths, strict=True):
                ctypes.memmove(pointer, remote[key][offset : offset + size], size)
                offset += size

    monkeypatch.setattr(runtime.storage_manager, "batched_put_external_pages", put)
    monkeypatch.setattr(runtime.storage_manager, "batched_get_external_pages", get)
    monkeypatch.setattr(
        runtime.storage_manager,
        "batched_external_pages_exist",
        lambda keys, strict=False: [key in remote for key in keys],
    )
    instance.capture(capture.spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "captured"
    instance.seal(SealSpec("request", 1, tuple(range(7))))
    assert capture.finish(instance)[0].status == "ready"
    capture.wait_drained(instance)
    store = runtime.checkpoint_store
    assert store.used > 0
    with monkeypatch.context() as pressure:
        capacity = runtime.checkpoint_backend().get_allocator_capacity_bytes()[1]
        pressure.setattr(
            runtime.checkpoint_backend(),
            "get_allocator_capacity_bytes",
            lambda: (0, capacity),
        )
        store.wake.set()
        wait(lambda: bool(store.records["request", 1].remote) and store.used == 0)
    manifest = store.manifest("request", 1)
    for group in manifest.groups:
        for page in group:
            runtime.checkpoint_backend().remove(page.key)
    assert store.available("request", 1, 7) == 7
    end, groups = store.acquire(manifest, 7)
    try:
        assert end == 7
        for group, entries in enumerate(groups):
            for source, page in entries:
                assert (
                    b"".join(
                        ctypes.string_at(
                            page.layer_data_ptr(i), page.layer_size_bytes(i)
                        )
                        for i in range(page.num_layers)
                    )
                    == remote[source.key]
                )
    finally:
        store.release(groups)
    store.restore_event(("request", 1), 3, "begin")
    instance.release_restore("request", 1, 3)
    instance.restore_event("request", 1, 3, "admitted")
    wait(lambda: store.manifest("request", 1) is None)
    capture.wait_drained(instance)


@pytest.mark.parametrize("prefix_end", [0, 4])
@pytest.mark.parametrize("missing_prefix", [False, True])
def test_checkpoint_reservation_retains_older_latent_until_seal(
    setup, worker, monkeypatch, prefix_end, missing_prefix
):
    owner, runtime, work = setup
    instance, _, client, _ = worker
    # The fixture's direct runtime driver and W client must not reuse sequence IDs.
    client.counter = 100
    runtime.config.dsa_group1_load_mode = "persistent_direct_hbm"
    runtime.config.pd_role = "receiver"
    runtime.config.retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]
    contains = runtime.storage_manager.batched_contains_two_group_layer_pages

    def paired(left, right, locations, pin):
        if locations == ["RemoteBackend"]:
            return len(left), {
                "RemoteBackend": [
                    k for pair in zip(left, right, strict=True) for k in pair
                ]
            }
        return contains(left, right, locations, pin)

    runtime.storage_manager.batched_contains_two_group_layer_pages = paired
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(8))))
    pages = [runtime.allocate_checkpoint_fragment(0, b - a)[0] for a, b, _ in chunks]
    owner.local.batched_submit_layer_pages([k for _, _, k in chunks], pages)
    for page in pages:
        page.ref_count_down()
    lookup, _ = work(
        "scheduler",
        "lookup",
        {
            "starts": tuple(a for a, _, _ in chunks),
            "ends": tuple(b for _, b, _ in chunks),
            "hashes": tuple(wire_hash(k.chunk_hash, 64) for _, _, k in chunks),
            "request_configs": {},
        },
    )
    prepared, original = work(
        "worker-0",
        "prepare",
        PrepareSpec(
            lookup.value.ticket_id,
            lookup.value.key_plan_digest,
            8,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
        ),
    )
    manifest = prepared.value.groups[0]
    descriptor = {
        "identity": original.identity,
        "manifest": manifest,
        "payload_digest": original.payload_digest,
    }
    instance.engine.compact_source_importer.source_descriptor = (
        lambda source: descriptor
    )
    prefix_state = SimpleNamespace(
        prepared_sparse_sources={0: object()},
        cached_starts=[a for a, _, _ in chunks],
        cached_ends=[b for _, b, _ in chunks],
        cached_keys=[[k.get_first_layer() for _, _, k in chunks]],
    )
    if missing_prefix:
        owner.local.remove(chunks[-1][2])
    before_refs = [p.get_ref_count() for p in pages]
    calls = [0, 0]
    process_tokens = runtime.token_database.process_tokens
    get_prefix = owner.local.batched_get_layer_page_prefix

    def keys(**kwargs):
        calls[0] += 1
        return process_tokens(**kwargs)

    def acquire(keys):
        calls[1] += 1
        return get_prefix(keys)

    with monkeypatch.context() as probe:
        probe.setattr(runtime.token_database, "process_tokens", keys)
        probe.setattr(owner.local, "batched_get_layer_page_prefix", acquire)
        instance.capture(
            replace(
                capture.spec(12),
                base=prefix_end,
                resident_start=8,
                prefix_end=prefix_end,
            ),
            {0: [None] * 2, 1: [None] * 3},
            4,
            prefix_state,
        )
    assert calls == [1, 1]
    assert not any(s.job.quarantined for s in instance.reservations.values()), (
        [s.future.result() for s in instance.reservations.values()],
        instance.results,
    )
    result = instance.poll()[0]
    if missing_prefix:
        assert result.status == "failed"
        assert [p.get_ref_count() for p in pages] == before_refs
        for participant in ("worker-0", "worker-1"):
            cpu.retire(runtime, original, manifest.lease_id, participant)
        capture.wait_drained(instance)
        return
    assert result.status == "captured", result.reason
    for participant in ("worker-0", "worker-1"):
        cpu.retire(runtime, original, manifest.lease_id, participant)
    assert not pages[-1].can_evict
    instance.seal(SealSpec("request", 1, tuple(range(12))))
    assert capture.finish(instance)[0].status == "ready"
    capture.wait_drained(instance)
    assert runtime.checkpoint_store.available("request", 1, 12) == 12
    assert not pages[-1].can_evict


def test_writeback_capture_cancel_uses_existing_fenced_retirement(worker):
    capture.test_cancel_private_capture_releases_only_after_fenced_capture(worker)


@pytest.mark.parametrize("cancel_credit_refused", [False, True])
def test_restore_notification_recovers_from_registry_pressure(
    worker, monkeypatch, cancel_credit_refused
):
    from lmcache.v1.storage_service import StorageService

    instance, runtime, client, _ = worker
    instance.timeout = 2
    instance.capture(capture.spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "captured"
    instance.seal(SealSpec("request", 1, tuple(range(7))))
    assert capture.finish(instance)[0].status == "ready"
    capture.wait_drained(instance)
    runtime.checkpoint_store.restore_event(("request", 1), 3, "begin")
    service = SimpleNamespace(
        runtime=SimpleNamespace(handle=runtime.handle), _reply=StorageService._reply
    )
    monkeypatch.setattr(
        runtime, "handle", lambda request: StorageService._handle(service, request)
    )
    extra = client.request(
        "admin",
        {"method": "snapshot"},
        request_id="other",
        request_attempt="other",
        load_generation=0,
        purpose="admin",
    )
    assert extra.result().status == "COMPLETED"
    runtime.registry.max_operations = 1
    if cancel_credit_refused:
        original_cancel = client.cancel
        refused = False

        def cancel(identity):
            nonlocal refused
            if not refused:
                refused = True
                raise BufferError("Storage lifecycle admission is full")
            return original_cancel(identity)

        monkeypatch.setattr(client, "cancel", cancel)
    enabled = gc.isenabled()
    gc.disable()
    try:
        instance.restore_event("request", 1, 3, "ack")
        instance.restore_event("request", 1, 3, "admitted")
        assert instance.pending_controls
        assert runtime.checkpoint_store.manifest("request", 1) is not None
        client.acknowledge_terminal(extra.identity).result()
        capture.wait_drained(instance)
        assert runtime.checkpoint_store.manifest("request", 1) is None
        wait(lambda: runtime.checkpoint_store.used == 0)
    finally:
        if enabled:
            gc.enable()
    assert runtime.registry.snapshot()["operations"] == 0


def test_disabled_writeback_sends_no_restore_notifications(worker):
    instance, _, client, _ = worker
    instance.engine.config.decode_checkpoint_writeback = False
    instance.restore_event("request", 1, 3, "failed")
    instance.release_restore("request", 1, 3)
    assert not client.controls


def test_registry_retry_yields_to_checkpoint_retirement(worker, monkeypatch):
    from threading import Event
    from lmcache.v1.storage_service import StorageService

    instance, runtime, client, _ = worker
    instance.timeout = 0.5
    instance.capture(capture.spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "captured"
    instance.seal(SealSpec("request", 1, tuple(range(7))))
    wait(
        lambda: instance.jobs["request", 1].future.done()
        and not instance.pending_controls
    )
    assert runtime.registry.snapshot()["operations"] == 1
    runtime.registry.max_operations = 1
    service = SimpleNamespace(
        runtime=SimpleNamespace(handle=runtime.handle), _reply=StorageService._reply
    )
    refused = Event()

    def dispatch(request):
        result = StorageService._handle(service, request)
        if request.method == "cancel" and result.status == "BUSY":
            refused.set()
        return result

    monkeypatch.setattr(runtime, "handle", dispatch)
    instance.restore_event("other", 1, 3, "ack")
    assert refused.wait(1)
    # Poll queues the fenced reservation's retirement on the same executor.
    # This is the operation that frees registry capacity for the notification.
    assert instance.poll()[0].status == "ready"
    capture.wait_drained(instance)
    assert runtime.registry.snapshot()["operations"] == 0


def test_restore_retry_over_real_storage_sockets(worker):
    import socket
    from lmcache.v1.rpc.storage_client import StorageClient
    from lmcache.v1.storage_service import StorageService

    instance, runtime, _, _ = worker
    endpoints = []
    for _ in range(2):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoints.append(f"tcp://127.0.0.1:{sock.getsockname()[1]}")
    # The fixture owns allocator shutdown; exercise real framing and completion
    # threads against its actual M registry without closing that heap twice.
    service = StorageService(
        SimpleNamespace(
            handle=runtime.handle, close=lambda: None, is_healthy=runtime.is_healthy
        ),
        *endpoints,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"worker-0": b"a"},
    )
    service.start()
    client = StorageClient(
        *endpoints,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker-0",
        session_nonce=b"a",
    )
    instance.client = client
    instance.timeout = 3
    try:
        extra = client.request(
            "admin",
            {"method": "snapshot"},
            request_id="other",
            request_attempt="other",
            load_generation=0,
            purpose="admin",
        )
        assert extra.result(2).status == "COMPLETED"
        runtime.registry.max_operations = 1
        instance.restore_event("request", 1, 3, "ack")
        future = next(iter(instance.pending_controls.values()))
        assert future.result(2).status == "BUSY"
        assert (
            client.acknowledge_terminal(extra.identity).result(2).status
            == "ACKNOWLEDGED"
        )
        capture.wait_drained(instance)
        assert runtime.registry.snapshot()["operations"] == 0
    finally:
        instance.close()
        client.close()
        service.close()


def test_writeback_unknown_capture_keeps_destinations(worker):
    capture.test_unknown_native_fence_retains_real_pages_and_refuses_close(worker)


def test_writeback_restore_retains_all_rank_ack_lease(setup):
    cpu.test_checkpoint_restore_keeps_a_separate_scheduler_ack_lease(setup)


def test_failed_restore_preparation_releases_attempt(worker, monkeypatch):
    instance, runtime, client, _ = worker
    instance.capture(capture.spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "captured"
    instance.seal(SealSpec("request", 1, tuple(range(7))))
    assert capture.finish(instance)[0].status == "ready"
    capture.wait_drained(instance)
    store = runtime.checkpoint_store
    from lmcache.v1.cpu_cache_runtime import CPUCacheRuntime
    from lmcache.v1.rpc.storage_protocol import OperationIdentity

    def fail_after_begin(self, spec, identity, **kwargs):
        store.restore_event(("request", 1), identity.load_generation, "begin")
        raise ValueError("injected source export failure")

    monkeypatch.setattr(CPUCacheRuntime, "prepare", fail_after_begin)
    identity = OperationIdentity(
        "cache", "epoch", "worker-0", 1, "request", "resume", 3, "checkpoint"
    )
    spec = PrepareSpec(
        "ticket",
        b"digest",
        7,
        runtime.layout.digest,
        ("worker-0",),
        mode="checkpoint_restore",
        checkpoint_generation=1,
    )
    with pytest.raises(ValueError, match="source export failure"):
        runtime.prepare(spec, identity)
    assert store.records["request", 1].attempts == {}
    assert store.available("request", 1, 7) == 7
