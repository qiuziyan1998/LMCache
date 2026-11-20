# SPDX-License-Identifier: Apache-2.0
"""CPU compact import and retirement tests; device upload is injected."""

from concurrent.futures import Future
from types import SimpleNamespace
import gc

import msgspec
import pytest
import torch

from lmcache.v1.rpc.storage_import import (
    CompactSourceImporter,
    initialize_worker_storage,
    source_pointer_table,
)
from lmcache.v1.rpc.storage_protocol import (
    GroupLayout,
    GroupManifest,
    LayoutSpec,
    OperationIdentity,
    PreparedReply,
    SourceGeometry,
    StorageResult,
    pack_uints,
)


def fixture(source_end=5, base=4096):
    layout = LayoutSpec(
        "model",
        "ns",
        "sha256",
        "bytes",
        256,
        1,
        4,
        (GroupLayout(0, ("a", "b"), ("bfloat16",) * 2, ((4,),) * 2, 1),),
    )
    identity = OperationIdentity(
        "cache", "epoch", "worker-0", 1, "req", "attempt", 1, "load"
    )
    manifest = GroupManifest(
        0,
        1,
        layout.digest,
        b"key-proof",
        "lease",
        (0, 4),
        (4, 5),
        pack_uints([64]),
        pack_uints([64]),
        pack_uints([0], 32),
        (
            SourceGeometry(
                4,
                (0, 32),
                (16, 16),
                ((4, 2),) * 2,
                ("bfloat16",) * 2,
                (((0, 16),),) * 2,
            ),
        ),
        pack_uints([192, 128]),
        pack_uints([4, 4]),
        pack_uints([4, 4]),
        pack_uints([1, 1], 32),
    )
    reply = PreparedReply(identity, source_end, "/slab", 256, 1, (manifest,))
    client = SimpleNamespace(calls=[])

    def retire(*args):
        future = Future()
        client.calls.append((args, future))
        return future

    client.retire = retire
    importer = CompactSourceImporter(
        layout=layout,
        mapping=SimpleNamespace(shm_name="/slab", size=256, generation=1),
        local_device_base=base,
        client=client,
        upload=lambda table: table,
        fence_upload=lambda: None,
    )
    return importer, reply, client


def imported(importer, reply):
    return importer.import_reply(
        reply,
        identity=reply.identity,
        payload_digest=b"digest",
        key_plan_digest=b"key-proof",
        source_end=reply.source_end,
    )


def test_local_cpu_only_worker_binds_layout_without_remote_reader(monkeypatch):
    from lmcache.v1.rpc import storage_layout

    importer, _, _ = fixture()
    bound = []
    context = SimpleNamespace(
        descriptor={},
        reader=None,
        mapping=SimpleNamespace(passive_allocator=lambda: "mapped-allocator"),
        bind_layout=bound.append,
        device_base=4096,
        client=SimpleNamespace(),
        drain_guards=[],
    )
    engine = SimpleNamespace(
        storage_process=context,
        gpu_connector=SimpleNamespace(
            synchronize_dense_load_stream=lambda: None,
        ),
        config=SimpleNamespace(),
        metadata=SimpleNamespace(),
        token_database=SimpleNamespace(),
        storage_plane_token_bytes={},
    )

    # SimpleNamespace cannot be weakly referenced, while a real engine can.
    class Engine:
        pass

    actual_engine = Engine()
    actual_engine.__dict__.update(engine.__dict__)
    monkeypatch.setattr(
        storage_layout, "freeze_storage_layout", lambda *a, **k: importer.layout
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(Event=lambda: None), raising=False
    )
    initialize_worker_storage(actual_engine)
    assert bound == [importer.layout]
    assert actual_engine.shared_cpu_cache_passive_allocator == "mapped-allocator"
    assert len(context.drain_guards) == 2


def test_actual_compact_builder_has_no_per_chunk_memory_objects_or_tensor_views():
    importer, reply, client = fixture()
    source = imported(importer, reply)[0]
    assert source.layers[0].chunk_ptrs_npu.tolist() == [4160, 4288]
    assert source.layers[1].chunk_ptrs_npu.tolist() == [4192, 4224]
    assert all(layer.tensors == layer.memory_objs == () for layer in source.layers)
    assert not client.calls
    source.owner_lease.release("fenced")
    assert client.calls[0][0][-1] == "fenced"
    assert importer.pending_retirements
    client.calls[0][1].set_result(StorageResult(reply.identity, "RETIRED"))
    assert not importer.pending_retirements


def test_pointer_high_bit_is_preserved_in_kernel_int64_representation():
    importer, reply, _ = fixture(base=2**63)
    table = source_pointer_table(reply.groups[0], 2**63, 2)
    assert table.dtype == torch.int64
    assert table[0, 0].item() == -(2**63) + 64
    assert (
        imported(importer, reply)[0].layers[1].chunk_ptrs_npu[0].item() == -(2**63) + 96
    )


def test_explicit_page_position_preserves_flat_then_page_pointer_order():
    importer, reply, _ = fixture()
    geometry = SourceGeometry(
        1, (0, 4), (4, 4), ((1, 2),) * 2, ("bfloat16",) * 2, (((0, 4),),) * 2
    )
    manifest = msgspec.structs.replace(
        reply.groups[0],
        geometries=(geometry,),
        page_chunk_indices=pack_uints([1], 32),
        flat_extents=pack_uints([16, 16]),
        flat_logical_sizes=pack_uints([16, 16]),
        flat_token_capacities=pack_uints([4, 4], 32),
    )
    manifest.validate(importer.layout, reply.slab_size, reply.source_end)
    assert manifest.pointer_rows(4096, 2) == ((4288, 4160), (4224, 4164))
    assert source_pointer_table(manifest, 4096, 2).tolist() == [
        [4288, 4160],
        [4224, 4164],
    ]


@pytest.mark.parametrize("positions", [[2], [1, 1], [0xFFFFFFFF]])
def test_invalid_explicit_page_positions_are_rejected(positions):
    importer, reply, _ = fixture()
    manifest = msgspec.structs.replace(
        reply.groups[0], page_chunk_indices=pack_uints(positions, 32)
    )
    with pytest.raises(ValueError):
        manifest.validate(importer.layout, reply.slab_size, reply.source_end)


def test_explicit_page_capacity_is_checked_at_its_actual_chunk_position():
    importer, reply, _ = fixture()
    small = SourceGeometry(
        1, (0, 4), (4, 4), ((1, 2),) * 2, ("bfloat16",) * 2, (((0, 4),),) * 2
    )
    manifest = msgspec.structs.replace(
        reply.groups[0], geometries=(small,), page_chunk_indices=pack_uints([0], 32)
    )
    with pytest.raises(ValueError, match="capacity"):
        manifest.validate(importer.layout, reply.slab_size, reply.source_end)


def test_consumption_end_does_not_change_physical_full_chunk_geometry():
    importer, reply, _ = fixture(source_end=3)
    # Only the consumed source chunk belongs in this reply; its capacity stays 4.
    manifest = msgspec.structs.replace(
        reply.groups[0],
        chunk_starts=(0,),
        chunk_ends=(4,),
        flat_offsets=b"",
        flat_extents=b"",
        flat_logical_sizes=b"",
        flat_token_capacities=b"",
    )
    reply = msgspec.structs.replace(reply, groups=(manifest,))
    source = imported(importer, reply)[0]
    assert source.total_tokens == 3
    assert source.chunk_token_counts == (3,)
    assert source.physical_chunk_token_counts == ((4,), (4,))
    assert manifest.geometries[0].token_capacity == 4


def test_gc_off_graph_child_prevents_early_physical_retirement():
    importer, reply, client = fixture()
    enabled = gc.isenabled()
    gc.disable()
    try:
        source = imported(importer, reply)[0]
        graph = source.owner_lease.retain()
        graph.mark_imported()
        source.owner_lease.release("fenced")
        assert not client.calls
        with pytest.raises(RuntimeError):
            graph.release("never_imported")
        graph.release("fenced")
        assert len(client.calls) == 1
        client.calls[0][1].set_exception(ConnectionError("owner lost"))
        assert importer.pending_retirements
    finally:
        if enabled:
            gc.enable()


def test_failed_upload_fences_before_retirement_and_retains_unacked_mapping():
    importer, reply, client = fixture()
    importer.upload = lambda table: (_ for _ in ()).throw(MemoryError("upload"))
    with pytest.raises(MemoryError):
        imported(importer, reply)
    assert client.calls[0][0][-1] == "fenced"
    assert importer.pending_retirements


def test_stale_pointer_epoch_and_key_plan_refused_before_upload():
    importer, reply, _ = fixture()
    importer.upload = lambda _: pytest.fail("must validate before upload")
    with pytest.raises(ValueError, match="stale"):
        imported(importer, msgspec.structs.replace(reply, slab_generation=2))
    manifest = msgspec.structs.replace(reply.groups[0], key_plan_digest=b"other-prefix")
    with pytest.raises(ValueError, match="key plan"):
        imported(importer, msgspec.structs.replace(reply, groups=(manifest,)))


def test_rejected_manifest_retires_all_trusted_unimported_leases():
    importer, reply, client = fixture()
    reply = msgspec.structs.replace(
        reply, slab_generation=2, restore_lease_id="restore"
    )
    with pytest.raises(ValueError, match="stale"):
        imported(importer, reply)
    assert {call[0][2:] for call in client.calls} == {
        ("lease", "never_imported"),
        ("restore", "never_imported"),
    }
    assert not importer.sources_drained()
    for _, future in client.calls:
        future.set_result(StorageResult(reply.identity, "RETIRED"))
    assert importer.sources_drained()


def test_untrusted_operation_mismatch_cannot_retire_another_request():
    importer, reply, client = fixture()
    identity = msgspec.structs.replace(reply.identity, request_attempt="other")
    with pytest.raises(ValueError, match="identity"):
        importer.import_reply(
            reply,
            identity=identity,
            payload_digest=b"digest",
            key_plan_digest=b"key-proof",
            source_end=5,
        )
    assert not client.calls


def test_unknown_pointer_upload_completion_retains_mapping_and_restore_owners():
    importer, reply, client = fixture()
    importer.upload = lambda _: (_ for _ in ()).throw(MemoryError("upload"))
    importer.fence_upload = lambda: (_ for _ in ()).throw(RuntimeError("unknown DMA"))
    with pytest.raises(RuntimeError, match="unknown DMA"):
        imported(importer, msgspec.structs.replace(reply, restore_lease_id="restore"))
    assert not client.calls
    assert not importer.sources_drained()


def test_restore_normalization_lifetime_ends_only_at_explicit_all_rank_ack():
    importer, reply, client = fixture()
    reply = msgspec.structs.replace(reply, restore_lease_id="restore")
    source = imported(importer, reply)[0]
    source.owner_lease.release("fenced")
    assert [call[0][2] for call in client.calls] == ["lease"]
    importer.release_restore("other", 1)
    assert len(client.calls) == 1
    importer.release_restore("req", 1)
    importer.release_restore("req", 1)
    assert [call[0][2] for call in client.calls] == ["lease", "restore"]


def test_lazy_diagnostic_views_preserve_actual_page_offsets_and_parent_lifetime():
    importer, reply, client = fixture()
    importer.mapping.tensor = torch.arange(256, dtype=torch.uint8)
    source = imported(importer, reply)[0]
    row = importer.source_layer_tensors(source, 1)
    assert len(row) == 2
    assert row[0].data_ptr() == importer.mapping.tensor.data_ptr() + 96
    assert row[1].data_ptr() == importer.mapping.tensor.data_ptr() + 128
    assert row[0].shape == (4, 2)
    assert row[0].dtype == torch.bfloat16
    assert not client.calls
    assert importer.is_certified_source(source, "req", 1)
    assert not importer.is_certified_source(source, "req", 2)
    source.owner_lease.release("fenced")
    assert len(client.calls) == 1


def test_merged_source_reuses_one_immutable_capacity_row_across_layers():
    importer, reply, client = fixture()
    manifest = msgspec.structs.replace(
        reply.groups[0],
        page_offsets=pack_uints([64, 128]),
        page_extents=pack_uints([64, 64]),
        geometry_ids=pack_uints([0, 0], 32),
        flat_offsets=b"",
        flat_extents=b"",
        flat_logical_sizes=b"",
        flat_token_capacities=b"",
    )
    source = imported(importer, msgspec.structs.replace(reply, groups=(manifest,)))[0]
    try:
        rows = source.physical_chunk_token_counts
        assert rows == ((4, 4), (4, 4))
        assert rows[0] is rows[1]
        assert source.chunk_token_counts == (4, 1)
        assert source.compact_transfer_metadata == {}
    finally:
        source.owner_lease.release("fenced")
    assert len(client.calls) == 1
