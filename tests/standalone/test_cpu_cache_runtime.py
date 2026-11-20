# SPDX-License-Identifier: Apache-2.0
"""CPU runtime tests using actual paired lookup, LocalCPU pages and source leases."""

# Standard
from collections import defaultdict
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
import gc

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.cpu_cache_runtime import CPUCacheRuntime
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    PrepareSpec,
    StorageRequest,
    content_digest,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager

# Local
from test_cpu_cache_operations import CPUStorage


class TokenPlan:
    """Convert existing hashes to keys without inspecting request token IDs."""

    def __init__(self, owner):
        self.owner = owner

    def process_tokens(self, *, hashes, offsets, request_configs=None, kv_group=0):
        cursor = 0
        for hash_value, count in zip(hashes, offsets, strict=True):
            yield (
                cursor,
                cursor + count,
                self._make_key_by_hash(
                    hash_value,
                    request_configs,
                    kv_group,
                    count,
                ),
            )
            cursor += count

    def _make_key_by_hash(self, value, configs=None, kv_group=0, valid_tokens=None):
        result = self.owner.key(value, kv_group, valid_tokens or 4)
        if configs:
            result.request_configs.update(configs)
        return result


@pytest.fixture
def setup():
    owner = CPUStorage()
    owner.metadata.kv_shape = (2, 1, 4, 1, 2)
    owner.metadata.kv_dtype = torch.bfloat16
    manager = owner.storage_manager
    manager.close = owner.local.close
    manager.get_active_storage_backends = lambda search_range=None: [
        (name, backend)
        for name, backend in manager.storage_backends.items()
        if search_range is None or name in search_range
    ]
    for name in (
        "batched_contains_two_group_layer_pages",
        "batched_contains_layer_pages",
        "batched_contains",
    ):
        setattr(manager, name, MethodType(getattr(StorageManager, name), manager))
    runtime = CPUCacheRuntime(
        owner.config,
        owner.metadata,
        TokenPlan(owner),
        storage_manager=manager,
        layout=owner.layout,
        cache_instance_id="cache",
        service_epoch="epoch",
        slab_name=owner.allocator.shm_name,
        slab_size=owner.allocator.buffer.numel(),
        sessions={"scheduler": b"s", "worker-0": b"a", "worker-1": b"b"},
    )
    counters = defaultdict(int)

    def work(session, method, payload, *, attempt="attempt", request_id="request"):
        counters[session] += 1
        identity = OperationIdentity(
            "cache",
            "epoch",
            session,
            counters[session],
            request_id,
            attempt,
            1,
            method,
        )
        request = StorageRequest(
            identity,
            method,
            payload,
            content_digest((method, payload)),
            runtime.registry.sessions[session],
        )
        return runtime.handle(request), request

    def bootstrap():
        for rank in range(2):
            session = f"worker-{rank}"
            result, _ = work(
                session,
                "register_participant",
                {
                    "session": session,
                    "tp_rank": rank,
                    "resource": {
                        "slab_name": owner.allocator.shm_name,
                        "slab_size": owner.allocator.buffer.numel(),
                        "slab_generation": 1,
                        "native_registered": True,
                        "reader_ready": True,
                        "device_base": (rank + 1) * 1024,
                        "pid": rank + 1,
                    },
                },
            )
            assert result.status == "COMPLETED"
        for rank in range(2):
            session = f"worker-{rank}"
            result, _ = work(
                session,
                "bind_layout",
                {
                    "session": session,
                    "layout": msgspec.to_builtins(owner.layout),
                },
            )
            assert result.status == "COMPLETED"
        assert result.value["status"] == "READY"

    def populate():
        pages = []
        for group in (0, 1):
            for index, tokens in ((7, 4), (8, 1)):
                key, page = owner.key(index, group, tokens), owner.page(group, tokens)
                owner.local.batched_submit_layer_pages([key], [page])
                page.ref_count_down()
                pages.append(page)
        return pages

    yield owner, runtime, work, bootstrap, populate
    owner.local.close()


def lookup(work):
    """Submit one full chunk plus a one-token tail without raw prompt tokens."""
    return work(
        "scheduler",
        "lookup",
        {
            "starts": (0, 4),
            "ends": (4, 5),
            "hashes": (7, 8),
            "request_configs": {},
        },
    )


def prepare(work, runtime, reply, *, attempt="attempt"):
    """Claim the scheduler's ticket from an independently authenticated worker."""
    return work(
        "worker-0",
        "prepare",
        PrepareSpec(
            reply.ticket_id,
            reply.key_plan_digest,
            5,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
        ),
        attempt=attempt,
    )


def control(runtime, original, method, payload, participant=None):
    """Send a lifecycle control against the original operation and digest."""
    session = participant or original.identity.client_session
    return runtime.handle(
        StorageRequest(
            original.identity,
            method,
            payload,
            original.payload_digest,
            runtime.registry.sessions[session],
            participant_session=participant or "",
        )
    )


def test_all_worker_registration_and_layout_are_required(setup):
    owner, runtime, work, bootstrap, populate = setup
    populate()
    result, _ = lookup(work)
    assert result.status == "FAILED_SAFE"
    bootstrap()
    result, _ = lookup(work)
    assert result.status == "COMPLETED" and result.value.matched_tokens == 5


def test_layout_cannot_advertise_before_the_workers_native_registration(setup):
    owner, runtime, work, bootstrap, _ = setup
    initial = runtime.layout
    result, _ = work(
        "worker-0",
        "bind_layout",
        {
            "session": "worker-0",
            "layout": msgspec.to_builtins(owner.layout),
        },
    )
    assert result.status == "FAILED_SAFE"
    assert runtime.layout is initial
    assert runtime.registration_status()["layout_status"] == "WAITING"
    bootstrap()
    assert runtime.registration_status()["layout_status"] == "READY"


def test_lookup_prepare_keeps_unequal_group_geometry_until_every_rank_retires(setup):
    owner, runtime, work, bootstrap, populate = setup
    pages = populate()
    bootstrap()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        hit, original_lookup = lookup(work)
        result, original_prepare = prepare(work, runtime, hit.value)
        assert result.status == "COMPLETED", result.error
        assert len(result.value.groups) == 2
        assert [len(g.geometries[0].layer_sizes) for g in result.value.groups] == [2, 3]
        assert all(g.chunk_ends == (4, 5) for g in result.value.groups)
        for manifest in result.value.groups:
            manifest.validate(
                runtime.layout,
                owner.allocator.buffer.numel(),
                5,
                key_plan_digest=hit.value.key_plan_digest,
            )
        # Original lookup cleanup cannot retire already-exported sources.
        control(runtime, original_lookup, "cancel", {})
        assert runtime.registry.snapshot()["leases"] == 2
        with pytest.raises(RuntimeError, match="live or quarantined"):
            runtime.close()
        for group in result.value.groups:
            for participant in ("worker-0", "worker-1"):
                retired = control(
                    runtime,
                    original_prepare,
                    "retire",
                    {
                        "lease_id": group.lease_id,
                        "proof": "never_imported",
                    },
                    participant,
                )
                assert retired.status == "RETIRED"
        assert runtime.registry.snapshot()["leases"] == 0
        assert all(page.get_ref_count() == 1 and not page.is_pinned for page in pages)
        runtime.close()
        assert owner.allocator.total_allocated_size == 0
    finally:
        if was_enabled:
            gc.enable()


def test_prepare_merged_pages_does_not_expand_layer_keys(setup, monkeypatch):
    from lmcache.utils import CacheEngineKey

    owner, runtime, work, bootstrap, populate = setup
    populate()
    bootstrap()
    hit, _ = lookup(work)
    split = CacheEngineKey.split_layers
    calls = []

    def tracked(key, layers):
        calls.append((key.kv_group, key.chunk_hash, layers))
        return split(key, layers)

    monkeypatch.setattr(CacheEngineKey, "split_layers", tracked)
    result, original = prepare(work, runtime, hit.value)
    assert result.status == "COMPLETED", result.error
    try:
        assert calls == []
        for group in result.value.groups:
            group.validate(
                runtime.layout,
                owner.allocator.buffer.numel(),
                5,
                key_plan_digest=hit.value.key_plan_digest,
            )
    finally:
        for group in result.value.groups:
            for participant in ("worker-0", "worker-1"):
                control(
                    runtime,
                    original,
                    "retire",
                    {
                        "lease_id": group.lease_id,
                        "proof": "never_imported",
                    },
                    participant,
                )


def test_cancel_lookup_releases_only_its_unclaimed_hold(setup):
    _, runtime, work, bootstrap, populate = setup
    pages = populate()
    bootstrap()
    result, original = lookup(work)
    assert all(page.is_pinned for page in pages)
    control(runtime, original, "cancel", {})
    assert not any(page.is_pinned for page in pages)
    late, _ = prepare(work, runtime, result.value)
    assert late.status == "FAILED_SAFE"


def test_reused_request_id_cannot_claim_another_request_attempt(setup):
    _, runtime, work, bootstrap, populate = setup
    populate()
    bootstrap()
    hit, _ = lookup(work)
    result, _ = prepare(work, runtime, hit.value, attempt="new-attempt")
    assert result.status == "FAILED_SAFE"
    assert runtime.is_healthy()


def test_manifest_failure_rolls_back_every_unexported_group_lease(setup, monkeypatch):
    _, runtime, work, bootstrap, populate = setup
    pages = populate()
    bootstrap()
    hit, _ = lookup(work)
    build = runtime._manifest

    def manifest(group, *args):
        if group == 1:
            raise ValueError("injected invalid group-1 geometry")
        return build(group, *args)

    monkeypatch.setattr(runtime, "_manifest", manifest)
    result, _ = prepare(work, runtime, hit.value)
    assert result.status == "FAILED_SAFE"
    assert runtime.registry.snapshot()["leases"] == 0
    assert all(page.get_ref_count() == 1 and not page.is_pinned for page in pages)
    assert runtime.is_healthy()


def test_early_retirement_cannot_release_a_source_during_manifest_build(
    setup, monkeypatch
):
    _, runtime, work, bootstrap, populate = setup
    pages = populate()
    bootstrap()
    hit, _ = lookup(work)
    spec = PrepareSpec(
        hit.value.ticket_id,
        hit.value.key_plan_digest,
        5,
        runtime.layout.digest,
        ("worker-0", "worker-1"),
    )
    build = runtime._manifest

    def manifest(group, rows, page_count, reply, source_end, lease_id):
        identity = runtime.source_key_plans[lease_id][0]
        original = StorageRequest(
            identity,
            "prepare",
            spec,
            content_digest(("prepare", spec)),
            b"a",
        )
        for participant in ("worker-0", "worker-1"):
            control(
                runtime,
                original,
                "retire",
                {
                    "lease_id": lease_id,
                    "proof": "never_imported",
                },
                participant,
            )
        assert rows[0][0].get_ref_count() >= 2
        return build(group, rows, page_count, reply, source_end, lease_id)

    monkeypatch.setattr(runtime, "_manifest", manifest)
    result, _ = work("worker-0", "prepare", spec)
    assert result.status == "FAILED_SAFE", result.error
    assert runtime.registry.snapshot()["leases"] == 0
    assert all(page.get_ref_count() == 1 and not page.is_pinned for page in pages)


def test_persistent_direct_group1_requires_pair_proof_but_creates_no_cpu_lease(setup):
    owner, runtime, work, bootstrap, populate = setup
    populate()
    owner.config.dsa_group1_load_mode = "persistent_direct_hbm"
    calls = []

    def contains(keys, pin=False):
        calls.append((tuple(key.kv_group for key in keys), pin))
        return len(keys)

    owner.storage_manager.storage_backends["RemoteBackend"] = SimpleNamespace(
        batched_contains_layer_pages=contains,
    )
    bootstrap()
    hit, _ = lookup(work)
    result, _ = prepare(work, runtime, hit.value)
    assert result.status == "COMPLETED", result.error
    assert calls[0] == ((0, 1, 0, 1), False)
    assert [group.group_id for group in result.value.groups] == [0]
    assert result.value.actions == ("load_group1_direct",)


def test_peer_registration_cannot_impersonate_a_different_worker(setup):
    _, runtime, work, _, _ = setup
    result, _ = work(
        "worker-0",
        "register_participant",
        {
            "session": "worker-1",
            "tp_rank": 1,
            "error": "failed",
        },
    )
    assert result.status == "FAILED_UNKNOWN"
    assert not runtime.is_healthy()


def test_unknown_transfer_completion_prevents_storage_teardown(setup, monkeypatch):
    _, runtime, work, bootstrap, populate = setup
    populate()
    bootstrap()
    hit, _ = lookup(work)

    def unknown(**kwargs):
        raise NativeExternalPageTransferUnknownError("read", Future())

    monkeypatch.setattr(runtime, "_resolve_shared_rank0_layer_pages", unknown)
    result, _ = prepare(work, runtime, hit.value)
    assert result.status == "FAILED_UNKNOWN"
    with pytest.raises(RuntimeError, match="live or quarantined"):
        runtime.close()


def test_native_uncertainty_blocks_allocator_close(setup):
    _, runtime, _, _, _ = setup
    closed = []

    def refuse():
        raise RuntimeError("native completion unknown")

    runtime.native_io = SimpleNamespace(close=refuse, is_healthy=lambda: False)
    original = runtime.storage_manager.close
    runtime.storage_manager.close = lambda: closed.append(True)
    try:
        with pytest.raises(RuntimeError, match="completion unknown"):
            runtime.close()
        assert not closed
        assert not runtime.is_healthy()
    finally:
        runtime.native_io = None
        runtime.storage_manager.close = original


def test_native_worker_registration_is_required_before_ready(setup):
    owner, runtime, work, _, _ = setup
    calls = []

    def bind(value, name):
        calls.append((value, name))
        if not value:
            raise ValueError("missing native registration")

    runtime.native_io = SimpleNamespace(bind=bind)
    resource = dict(
        slab_name=runtime.shared_cpu_cache_name,
        slab_size=runtime.shared_cpu_cache_slab_size,
        slab_generation=1,
        native_registered=True,
        reader_ready=True,
        device_base=1000,
    )
    result, _ = work(
        "worker-0",
        "register_participant",
        dict(session="worker-0", tp_rank=0, resource=resource),
    )
    assert result.value["status"] == "FAILED"
    assert "missing native registration" in result.value["error"]
    assert calls == [({}, runtime.shared_cpu_cache_name)]


def test_actual_runtime_stop_remains_acknowledgeable(setup):
    from threading import Event
    from lmcache.v1.rpc.storage_bootstrap import WorkerStorageBootstrap
    from lmcache.v1.rpc.storage_client import StorageClient
    from lmcache.v1.storage_service import StorageService
    from tests.v1.test_worker_mooncake_transport import endpoint

    _, runtime, _, _, _ = setup
    work, lifecycle = endpoint(), endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions=runtime.registry.sessions,
    )
    service.start()
    fatal = Event()
    client = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker-0",
        session_nonce=b"a",
        failure_callback=lambda reason: fatal.set(),
    )
    context = WorkerStorageBootstrap(client, {}, "worker-0", 2)
    try:
        assert context.call("stop", {}) == {"status": "STOPPED"}
        assert not fatal.wait(0.05)
    finally:
        client.close()
        service.close()
