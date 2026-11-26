# SPDX-License-Identifier: Apache-2.0
"""Follow-up CPU source audit with real indexes, allocation, pins and key hashes.

The remote tier uses a second real LocalCPU index over the same allocator, so
these tests cover selection/materialization ownership, not native Mooncake I/O.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import MethodType
import gc

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.rpc.storage_protocol import PrepareSpec, wire_hash
from lmcache.v1.memory_management import TensorMemoryAllocator
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager

# Local
from test_cpu_cache_runtime import control, setup as runtime_setup
from test_ascend_cpu_storage_runtime import setup as ascend_runtime_setup


@pytest.fixture
def setup():
    yield from runtime_setup.__wrapped__()


@pytest.fixture
def ascend_setup():
    yield from ascend_runtime_setup.__wrapped__()


def plan(hashes, ends, *, width=256, kind="bytes"):
    return {
        "starts": (0, *ends[:-1]),
        "ends": ends,
        "hashes": tuple(wire_hash(value, width) for value in hashes),
        "hash_width": width,
        "hash_kind": kind,
        "request_configs": {},
    }


def submit(work, runtime, reply, end):
    return work(
        "worker-0",
        "prepare",
        PrepareSpec(
            reply.ticket_id,
            reply.key_plan_digest,
            end,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
        ),
    )


@pytest.mark.parametrize("kind", ["bytes", "int"])
def test_full_digest_identity_survives_lookup_prepare_and_one_byte_miss(setup, kind):
    owner, runtime, work, bootstrap, _ = setup
    hashes = (b"\0" * 31 + b"\x07", b"\0" * 31 + b"\x08")
    if kind == "int":
        hashes = tuple(int.from_bytes(value, "big") + (1 << 200) for value in hashes)
    runtime.layout = owner.layout = msgspec.structs.replace(
        owner.layout, hash_algorithm="sha256", hash_kind=kind, hash_width=256
    )
    for group in (0, 1):
        for value, count in zip(hashes, (4, 1), strict=True):
            page = owner.page(group, count)
            owner.local.batched_submit_layer_pages(
                [owner.key(value, group, count)], [page]
            )
            page.ref_count_down()
    bootstrap()
    query = plan(hashes, (4, 5), kind=kind)
    hit, _ = work("scheduler", "lookup", query)
    assert hit.status == "COMPLETED" and hit.value.matched_tokens == 5
    assert hit.value.hashes == query["hashes"]
    prepared, original = submit(work, runtime, hit.value, 5)
    assert prepared.status == "COMPLETED", prepared.error
    for group in prepared.value.groups:
        for participant in ("worker-0", "worker-1"):
            control(
                runtime,
                original,
                "retire",
                {"lease_id": group.lease_id, "proof": "never_imported"},
                participant,
            )
    changed = list(hashes)
    changed[0] = (b"\x01" + changed[0][1:]) if kind == "bytes" else changed[0] ^ 1
    miss, _ = work("scheduler", "lookup", plan(changed, (4, 5), kind=kind))
    assert miss.status == "COMPLETED" and miss.value.matched_tokens == 0
    assert miss.value.hashes == ()


def put(owner, backend, value, group, count, *, page, marker):
    key = owner.key(value, group, count)
    if page:
        obj = owner.page(group, count)
        for layer in range(obj.num_layers):
            obj.layer_tensor(layer).fill_(marker + layer)
        backend.batched_submit_layer_pages([key], [obj])
        obj.ref_count_down()
        return [obj] * owner.num_layers_for_group(group)
    objects = []
    for layer, layer_key in enumerate(
        key.split_layers(owner.num_layers_for_group(group))
    ):
        shape, dtype, fmt = owner._expected_shared_cpu_chunk_metadata(
            kv_group=group, num_tokens=count, layer_id=layer
        )
        obj = owner.allocator.allocate(shape, dtype, fmt=fmt)
        obj.tensor.fill_(marker + layer)
        obj.metadata.valid_tokens = count
        backend.submit_put_task(layer_key, obj)
        obj.ref_count_down()
        objects.append(obj)
    return objects


def remote_tier(owner):
    """Adapt the second real index to RemoteBackend's representative key ABI."""
    remote = LocalCPUBackend(owner.config, memory_allocator=owner.allocator)
    manager = owner.storage_manager
    manager.storage_backends["RemoteBackend"] = remote
    manager.batched_unpin = MethodType(StorageManager.batched_unpin, manager)
    manager.batched_get = lambda keys, location: [
        remote.get_blocking(key) for key in keys
    ]
    unpin = remote.batched_unpin
    remote.batched_unpin = lambda keys: unpin(
        [
            key.without_layer()
            if hasattr(key, "without_layer") and remote.contains(key.without_layer())
            else key
            for key in keys
        ]
    )

    def fetch(keys):
        pages, count = remote.batched_get_layer_page_prefix(keys)
        assert count == len(keys)
        return pages

    remote.batched_get_layer_pages = fetch
    return remote


def test_alternating_tiers_and_independent_group_page_positions_keep_exact_plan(setup):
    owner, runtime, work, bootstrap, _ = setup
    # Every small object still consumes one actual allocator alignment unit.
    owner.allocator.buffer = torch.zeros(262144, dtype=torch.uint8)
    owner.allocator.pin_allocator = TensorMemoryAllocator(owner.allocator.buffer)
    runtime.shared_cpu_cache_slab_size = owner.allocator.buffer.numel()
    remote = remote_tier(owner)
    hashes = tuple(b"\0" * 31 + bytes((index,)) for index in range(1, 6))
    runtime.layout = owner.layout = msgspec.structs.replace(
        owner.layout, hash_algorithm="sha256", hash_kind="bytes", hash_width=256
    )
    pattern = (
        (False, False),
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    )
    objects = {0: [], 1: []}
    for chunk, (value, pages) in enumerate(zip(hashes, pattern, strict=True)):
        for group in (0, 1):
            objects[group].append(
                put(
                    owner,
                    remote if chunk % 2 else owner.local,
                    value,
                    group,
                    1 if chunk == 4 else 4,
                    page=pages[group],
                    marker=chunk * 10 + group * 2 + 1,
                )
            )
    bootstrap()
    enabled = gc.isenabled()
    gc.disable()
    try:
        hit, _ = work("scheduler", "lookup", plan(hashes, (4, 8, 12, 16, 17)))
        assert hit.status == "COMPLETED" and hit.value.matched_tokens == 17
        prepared, original = submit(work, runtime, hit.value, 17)
        assert prepared.status == "COMPLETED", prepared.error
        assert [group.page_positions() for group in prepared.value.groups] == [
            (1, 2),
            (1, 3),
        ]
        for manifest in prepared.value.groups:
            for layer, row in enumerate(
                manifest.pointer_rows(
                    0, runtime.num_layers_for_group(manifest.group_id)
                )
            ):
                assert row == tuple(
                    int(column[layer].metadata.address)
                    + (
                        column[layer].group_prefix_sum[layer]
                        if pattern[chunk][manifest.group_id]
                        else 0
                    )
                    for chunk, column in enumerate(objects[manifest.group_id])
                )
        owner.local.clear()
        remote.clear()
        held = owner.allocator.get_capacity_bytes()[0]
        assert held < owner.allocator.buffer.numel()
        for manifest in prepared.value.groups:
            control(
                runtime,
                original,
                "retire",
                {"lease_id": manifest.lease_id, "proof": "never_imported"},
                "worker-0",
            )
        assert owner.allocator.get_capacity_bytes()[0] == held
        for manifest in prepared.value.groups:
            control(
                runtime,
                original,
                "retire",
                {"lease_id": manifest.lease_id, "proof": "never_imported"},
                "worker-1",
            )
        owner.local.clear()
        remote.clear()
        assert runtime.registry.snapshot()["leases"] == 0
        assert (
            owner.allocator.get_capacity_bytes()[0] == owner.allocator.buffer.numel()
        ), [
            (group, chunk, layer, obj.get_ref_count(), obj.metadata.pin_count)
            for group, columns in objects.items()
            for chunk, column in enumerate(columns)
            for layer, obj in enumerate(column)
            if obj.is_valid()
        ]
    finally:
        remote.close()
        if enabled:
            gc.enable()


@pytest.mark.parametrize("source_end", [3, 5])
def test_dense_partial_prepare_keeps_both_groups_and_canonical_object_capacity(
    setup, source_end
):
    owner, runtime, work, bootstrap, _ = setup
    owner.config.dsa_group1_load_mode = "persistent_direct_hbm"
    owner.config.pd_role = "receiver"
    remote = remote_tier(owner)
    try:
        for group in (0, 1):
            for value, count in ((7, 4), (8, 1)):
                put(owner, remote, value, group, count, page=True, marker=group + value)
                if group == 0:
                    put(
                        owner,
                        owner.local,
                        value,
                        group,
                        count,
                        page=True,
                        marker=group + value,
                    )
        bootstrap()
        hit, _ = work("scheduler", "lookup", plan((7, 8), (4, 5), width=64, kind="int"))
        assert hit.status == "COMPLETED" and hit.value.matched_tokens == 5
        spec = PrepareSpec(
            hit.value.ticket_id,
            hit.value.key_plan_digest,
            source_end,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
            mode="dense",
        )
        prepared, original = work("worker-0", "prepare", spec)
        assert prepared.status == "COMPLETED", prepared.error
        assert [manifest.group_id for manifest in prepared.value.groups] == [0, 1]
        assert prepared.value.actions == ()
        assert msgspec.convert(msgspec.to_builtins(spec), type=PrepareSpec) == spec
        expected_ends = (4,) if source_end == 3 else (4, 5)
        for manifest in prepared.value.groups:
            assert manifest.chunk_ends == expected_ends
            assert manifest.geometries[0].token_capacity == 4
            for participant in ("worker-0", "worker-1"):
                control(
                    runtime,
                    original,
                    "retire",
                    {"lease_id": manifest.lease_id, "proof": "never_imported"},
                    participant,
                )
        owner.local.clear()
        remote.clear()
        assert owner.allocator.get_capacity_bytes()[0] == owner.allocator.buffer.numel()
    finally:
        remote.close()


@pytest.mark.parametrize("unknown", [False, True])
def test_persistence_barrier_retains_actual_destination_after_worker_retirement(
    ascend_setup, monkeypatch, unknown
):
    owner, runtime, work = ascend_setup
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(4))))
    reserved, original = work(
        "worker-0",
        "reserve_write",
        {
            "purpose": "store",
            "groups": [
                {
                    "group_id": 0,
                    "representation": "page",
                    "starts": (0,),
                    "ends": (4,),
                    "hashes": (chunks[0][2].chunk_hash,),
                }
            ],
            "request_configs": {},
        },
    )
    assert reserved.status == "COMPLETED", reserved.error
    reservation_id = reserved.value["reservation_id"]
    persistence, started, submitted = Future(), Event(), []

    def put_pages(keys, objects, **kwargs):
        # Stand in only for the remote persistence boundary; M owns real slab
        # objects and the manager's real submission reference until completion.
        submitted.extend(zip(keys, objects, strict=True))
        started.set()
        return [persistence]

    monkeypatch.setattr(runtime.storage_manager, "batched_put_layer_pages", put_pages)
    held = owner.allocator.get_capacity_bytes()[0]
    with ThreadPoolExecutor(1) as executor:
        commit = executor.submit(
            work,
            "worker-0",
            "commit_write",
            {"reservation_id": reservation_id, "proof": "fenced"},
        )
        assert started.wait(2)
        assert not commit.done()
        assert not owner.local.contains(chunks[0][2])
        control(
            runtime,
            original,
            "retire",
            {"lease_id": reserved.value["groups"][0]["lease_id"], "proof": "fenced"},
        )
        assert runtime.registry.snapshot()["leases"] == 1
        assert owner.allocator.get_capacity_bytes()[0] == held
        for key, obj in submitted:
            if not unknown:
                owner.local.batched_submit_layer_pages([key], [obj])
            obj.ref_count_down()
        if unknown:
            persistence.set_exception(
                NativeExternalPageTransferUnknownError("write", Future())
            )
        else:
            persistence.set_result(None)
        result, _ = commit.result(2)
    if unknown:
        assert result.status == "FAILED_UNKNOWN"
        assert not runtime.is_healthy()
        assert owner.allocator.get_capacity_bytes()[0] == held
        with pytest.raises(RuntimeError, match="unknown native completion"):
            work(
                "worker-0",
                "cancel_write",
                {"reservation_id": reservation_id, "proof": "fenced"},
            )
        with pytest.raises(RuntimeError, match="live or quarantined"):
            runtime.close()
    else:
        assert result.status == "COMPLETED" and result.value["status"] == "COMMITTED"
        assert runtime.registry.snapshot()["leases"] == 0
        assert owner.local.contains(chunks[0][2])
        owner.local.clear()
        assert owner.allocator.get_capacity_bytes()[0] == owner.allocator.buffer.numel()


def test_remote_fill_reclaim_checks_actual_free_capacity_with_a_borrowed_page(
    ascend_setup,
):
    owner, runtime, _ = ascend_setup
    runtime.config.remote_fill_min_free_bytes = 0
    runtime.config.remote_fill_min_free_ratio = 0
    _, _, key = next(runtime.token_database.process_tokens(tokens=list(range(4))))
    page, _ = runtime.allocate_checkpoint_fragment(0, 4)
    owner.local.batched_submit_layer_pages([key], [page])
    page.ref_count_down()
    borrowed, count = owner.local.batched_get_layer_page_prefix([key])
    assert count == 1
    requested = owner.allocator.buffer.numel()
    assert not runtime._remote_fill_reclaim_capacity(requested)
    assert owner.local.contains(key)
    assert not runtime._remote_fill_capacity_available(requested)
    borrowed[0].ref_count_down()
    assert runtime._remote_fill_reclaim_capacity(requested)
    assert not owner.local.contains(key)
    assert runtime._remote_fill_capacity_available(requested)
    assert owner.allocator.get_capacity_bytes()[0] == requested
