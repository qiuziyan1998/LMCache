# SPDX-License-Identifier: Apache-2.0
"""Source-loaded Ascend CPU runtime; no NPU engine or transport is constructed."""

# Standard
from collections import defaultdict
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
import importlib
import sys

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.integration.vllm.preemption_checkpoint import LOCAL_CHECKPOINT_CONFIG
from lmcache.v1.event_manager import EventManager
from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.memory_management import LayerPageMemoryObj, MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc.storage_protocol import (
    GroupLayout,
    LayoutSpec,
    OperationIdentity,
    PrepareSpec,
    StorageRequest,
    content_digest,
    wire_hash,
)
from lmcache.v1.rpc.storage_layout import freeze_storage_layout
from lmcache.v1.storage_backend.storage_manager import StorageManager

# Local
from test_cpu_cache_operations import CPUStorage


def ascend_modules():
    """Load actual CPU modules without running package NPU monkeypatch setup."""
    root = Path(__file__).resolve().parents[3] / "LMCache-Ascend/lmcache_ascend"
    for name, path in (("lmcache_ascend", root), ("lmcache_ascend.v1", root / "v1")):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    factory = importlib.import_module("lmcache_ascend.v1.cpu_cache").CPUBackendFactory
    module = importlib.import_module("lmcache_ascend.v1.cpu_cache_runtime")
    return factory, module


@pytest.fixture
def setup():
    factory, module = ascend_modules()
    owner = CPUStorage()
    owner.config.decode_preemption_checkpoint = True
    owner.config.decode_checkpoint_writeback = False
    owner.config.use_layerwise = True
    owner.config.save_unfull_chunk = True
    owner.config.enable_sparse_attention = True
    owner.config.enable_remote_lmcache_store = False
    owner.config.remote_fill_model_artifact_id = "model"
    owner.config.remote_fill_cache_namespace = "cache"
    owner.config.store_async = True
    owner.config.store_async_max_queue_size = 2
    owner.config.extra_config = {"mooncake_dsa_raw_token_dims": {0: 576, 1: 128}}
    policy = IndexerC8Layout(c8_layers=(True, False, True))
    owner.layout = LayoutSpec(
        "model",
        "cache",
        "builtin",
        "int",
        64,
        1,
        4,
        (
            GroupLayout(
                0,
                ("l0", "l1"),
                ("bfloat16",) * 2,
                ((1024, 128),) * 2,
                MemoryFormat.KV_MLA_LATENT_FMT.value,
            ),
            GroupLayout(
                1,
                ("i0", "i1", "i2"),
                ("uint8",) * 3,
                ((128, 2), (256,), (128, 2)),
                MemoryFormat.KV_DSA_INDEX_FMT.value,
            ),
        ),
        precision_descriptor=msgspec.msgpack.encode(policy.descriptor()),
        allocator_alignment=owner.allocator.address_manager.compute_aligned_size(1),
    )
    metadata = LMCacheMetadata(
        model_name="model",
        world_size=4,
        local_world_size=4,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 1, 4, 1, 576),
        use_mla=True,
        role="worker",
        mla_cache_tp_replicated=True,
    )
    manager = StorageManager(
        owner.config,
        metadata,
        EventManager(),
        backend_factory=factory(owner.allocator, local_backend=owner.local),
        copy_stream_factory=None,
    )
    runtime = module.AscendCPUCacheRuntime(
        owner.config,
        metadata,
        None,
        storage_manager=manager,
        layout=None,
        cache_instance_id="cache",
        service_epoch="epoch",
        slab_name=owner.allocator.shm_name,
        slab_size=owner.allocator.buffer.numel(),
        sessions={"worker-0": b"a", "worker-1": b"b", "scheduler": b"s"},
    )
    counters, requests = defaultdict(int), []

    def work(session, method, value):
        counters[session] += 1
        identity = OperationIdentity(
            "cache",
            "epoch",
            session,
            counters[session],
            "request",
            "attempt",
            1,
            method,
        )
        request = StorageRequest(
            identity,
            method,
            value,
            content_digest((method, value)),
            runtime.registry.sessions[session],
        )
        requests.append(request)
        return runtime.handle(request), request

    for rank in range(2):
        session = f"worker-{rank}"
        runtime.register_participant(
            session,
            {
                "session": session,
                "tp_rank": rank,
                "resource": {
                    "slab_name": owner.allocator.shm_name,
                    "slab_size": owner.allocator.buffer.numel(),
                    "slab_generation": 1,
                    "native_registered": True,
                    "reader_ready": True,
                    "device_base": 1024 * (rank + 1),
                },
            },
        )
        runtime.bind_layout(owner.layout, session)
    yield owner, runtime, work
    for request in requests:
        result = runtime.handle(
            msgspec.structs.replace(request, method="operation_status", payload={})
        )
        if result.value is None:
            continue
        for lease, state in result.value["leases"].items():
            if state["producer_holds"] or state["quarantined"]:
                continue
            for participant in state["participants"]:
                runtime.handle(
                    StorageRequest(
                        request.identity,
                        "retire",
                        {"lease_id": lease, "proof": "never_imported"},
                        request.payload_digest,
                        runtime.registry.sessions[participant],
                        participant_session=participant,
                    )
                )
    # No device or native DMA exists in this harness. Release deliberately
    # quarantined test allocations only after verifying production close refuses.
    for reservation in runtime._write_reservations.values():
        for fragments in reservation.fragments.values():
            for _, _, page in fragments:
                page.ref_count_down()
    manager.close()


def retire(runtime, original, lease, participant="worker-0"):
    """Retire an actual exported destination after the test's CPU producer finishes."""
    return runtime.handle(
        StorageRequest(
            original.identity,
            "retire",
            {"lease_id": lease, "proof": "fenced"},
            original.payload_digest,
            runtime.registry.sessions[participant],
            participant_session=participant,
        )
    )


def capture(work, *, base=0, end=9, resident=0, prefix=0, **extra):
    return work(
        "worker-0",
        "reserve_write",
        {
            "purpose": "checkpoint",
            "generation": 1,
            "base": base,
            "end": end,
            "resident_start": resident,
            "prefix_end": prefix,
            "request_configs": {},
            **extra,
        },
    )


def fill_capture(runtime, reservation_id):
    """Simulate completed worker writes, preserving each plane's physical offset."""
    reservation = runtime._write_reservations[reservation_id]
    for group, fragments in reservation.fragments.items():
        widths = runtime.checkpoint_plane_widths(group)
        for _, _, page in fragments:
            for layer in range(page.num_layers):
                planes = widths[layer] if isinstance(widths[0], tuple) else widths
                begin = 0
                for plane, width in enumerate(planes):
                    end = begin + width * page.valid_tokens
                    page.layer_tensor(layer)[begin:end].fill_(
                        group * 20 + layer * 3 + plane + 1
                    )
                    begin = end


def test_registered_mixed_geometry_has_no_engine_or_npu_dependency(setup):
    owner, runtime, _ = setup
    assert runtime.metadata.runtime_kv_group_layer_counts == (2, 3)
    assert runtime.checkpoint_plane_widths(0) == (512, 64)
    assert runtime.checkpoint_plane_widths(1) == ((128, 2), (256,), (128, 2))
    assert runtime.storage_manager.local_cpu_backend is owner.local
    assert "torch_npu" not in sys.modules


@pytest.mark.parametrize(
    "representations",
    [
        ("page", "page", "page"),
        ("page", "layers", "page"),
        ("layers", "page", "layers"),
    ],
)
def test_ascend_prepare_expands_only_flat_keys_and_preserves_bytes(
    setup, monkeypatch, representations
):
    import ctypes
    from lmcache.utils import CacheEngineKey

    owner, runtime, work = setup
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(9))))
    expected = {}
    for chunk_index, ((start, end, key), representation) in enumerate(
        zip(chunks, representations, strict=True)
    ):
        reserved, original = work(
            "worker-0",
            "reserve_write",
            {
                "purpose": "store",
                "groups": tuple(
                    {
                        "group_id": group,
                        "representation": representation,
                        "starts": (start,),
                        "ends": (end,),
                        "hashes": (wire_hash(key.chunk_hash, 64),),
                    }
                    for group in (0, 1)
                ),
            },
        )
        assert reserved.status == "COMPLETED", reserved.error
        reservation = runtime._write_reservations[reserved.value["reservation_id"]]
        for group, fragments in reservation.fragments.items():
            for index, (_, _, obj) in enumerate(fragments):
                values = (
                    tuple(obj.layer_tensor(layer) for layer in range(obj.num_layers))
                    if isinstance(obj, LayerPageMemoryObj)
                    else (obj.tensor,)
                )
                for layer, value in enumerate(values):
                    layer_id = layer if isinstance(obj, LayerPageMemoryObj) else index
                    value.fill_(1 + group * 20 + layer_id * 3 + chunk_index)
                    expected[group, layer_id, chunk_index] = (
                        value.view(torch.uint8).numpy().tobytes()
                    )
        committed, _ = work(
            "worker-0",
            "commit_write",
            {
                "reservation_id": reserved.value["reservation_id"],
                "proof": "fenced",
            },
        )
        assert committed.status == "COMPLETED", committed.error
        for group in reserved.value["groups"]:
            retire(runtime, original, group["lease_id"])
    hit, _ = work(
        "scheduler",
        "lookup",
        {
            "starts": tuple(a for a, _, _ in chunks),
            "ends": tuple(b for _, b, _ in chunks),
            "hashes": tuple(wire_hash(key.chunk_hash, 64) for _, _, key in chunks),
            "request_configs": {},
        },
    )
    assert hit.value.matched_tokens == 9, hit.error
    calls, split = [], CacheEngineKey.split_layers

    def tracked(key, layers):
        calls.append((key.kv_group, layers))
        return split(key, layers)

    monkeypatch.setattr(CacheEngineKey, "split_layers", tracked)
    prepared, original = work(
        "worker-0",
        "prepare",
        PrepareSpec(
            hit.value.ticket_id,
            hit.value.key_plan_digest,
            9,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
        ),
    )
    assert prepared.status == "COMPLETED", prepared.error
    try:
        flats = representations.count("layers")
        assert calls == [(0, 2)] * flats + [(1, 3)] * flats
        for manifest in prepared.value.groups:
            manifest.validate(
                runtime.layout,
                owner.allocator.buffer.numel(),
                9,
                key_plan_digest=hit.value.key_plan_digest,
            )
            pointers = manifest.pointer_rows(
                owner.allocator.buffer.data_ptr(),
                runtime.num_layers_for_group(manifest.group_id),
            )
            for layer, row in enumerate(pointers):
                for index, address in enumerate(row):
                    wanted = expected[manifest.group_id, layer, index]
                    assert ctypes.string_at(address, len(wanted)) == wanted
    finally:
        for manifest in prepared.value.groups:
            for participant in ("worker-0", "worker-1"):
                retire(runtime, original, manifest.lease_id, participant)
    assert runtime.registry.snapshot()["leases"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("remote_fill_cache_namespace", "stale-namespace"),
        ("remote_fill_model_artifact_id", "stale-artifact"),
        ("pre_caching_hash_algorithm", "sha256"),
    ],
)
def test_storage_key_identity_mismatch_fails_the_startup_gate(setup, field, value):
    owner, live, _ = setup
    _, module = ascend_modules()
    setattr(owner.config, field, value)
    fresh = module.AscendCPUCacheRuntime(
        owner.config,
        live.metadata,
        None,
        storage_manager=live.storage_manager,
        layout=None,
        cache_instance_id="cache",
        service_epoch="other-epoch",
        slab_name=owner.allocator.shm_name,
        slab_size=owner.allocator.buffer.numel(),
        sessions={"worker-0": b"a"},
    )
    fresh.register_participant("worker-0", live._registrations["worker-0"])
    with pytest.raises(ValueError, match="disagrees"):
        fresh.bind_layout(owner.layout, "worker-0")
    assert fresh.registration_status()["layout_status"] == "FAILED"
    assert fresh.get_remote_fill_placement_info() is None
    assert owner.allocator.get_capacity_bytes() == (65536, 65536)


def test_admin_reuses_manager_and_preserves_the_registered_owner(setup):
    owner, runtime, work = setup
    response, _ = work("scheduler", "admin", {"method": "list_backends"})
    assert response.value == {"LocalCPUBackend": "LocalCPUBackend"}
    work("scheduler", "admin", {"method": "set_freeze", "args": (True,)})
    frozen, _ = work("scheduler", "admin", {"method": "is_frozen"})
    assert frozen.value is True
    work("scheduler", "admin", {"method": "set_freeze", "args": (False,)})
    for method in ("close_backend", "recreate_backend"):
        refused, _ = work(
            "scheduler", "admin", {"method": method, "args": ("LocalCPUBackend",)}
        )
        assert refused.status == "FAILED_SAFE"
        assert runtime.storage_manager.local_cpu_backend is owner.local
    refused, _ = work(
        "scheduler", "admin", {"method": "__getattribute__", "args": ("config",)}
    )
    assert refused.status == "FAILED_SAFE"
    assert runtime.is_healthy()


def test_admin_unknown_native_close_latches_failure_without_freeing_slab(
    setup, monkeypatch
):
    owner, runtime, work = setup
    before = owner.allocator.get_capacity_bytes()

    def unknown(_):
        raise RuntimeError("native close cannot establish DMA completion")

    monkeypatch.setattr(runtime.storage_manager, "close_backend", unknown)
    result, _ = work(
        "scheduler", "admin", {"method": "close_backend", "args": ("RemoteBackend",)}
    )
    assert result.status == "FAILED_UNKNOWN"
    assert not runtime.is_healthy()
    assert owner.allocator.get_capacity_bytes() == before
    assert runtime.storage_manager.local_cpu_backend is owner.local


def test_admin_fs_import_uses_the_existing_allocator_and_reports_http_errors(
    setup, monkeypatch, tmp_path
):
    owner, runtime, work = setup
    import lmcache.v1.storage_backend.remote_backend as remote_module
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.storage_backend.connector.fs_connector import FSConnector

    created = []

    def connector(url, loop, local, config, metadata, **kwargs):
        assert local is owner.local
        fs = FSConnector(str(tmp_path / "chunks"), loop, local, config)
        created.append(fs)
        return fs

    monkeypatch.setattr(remote_module, "CreateConnector", connector)
    cfg_path = tmp_path / "fs.yaml"
    cfg_path.write_text(
        "chunk_size: 4\nremote_url: 'fs://host:0/test'\nremote_serde: naive\n",
        encoding="utf-8",
    )
    config = LMCacheEngineConfig.from_file(str(cfg_path))
    remote = remote_module.RemoteBackend(
        config,
        runtime.metadata,
        runtime.storage_manager.loop,
        owner.local,
        dst_device="cpu",
    )
    key = next(iter(runtime.token_database.process_tokens(tokens=list(range(4)))))[2]
    # Existing FS metadata ABI contains two four-dimensional group records.
    # Test the delegated import; actual retrieval still validates precision.
    obj = owner.local.allocate(
        [torch.Size([1, 2, 4, 576]), torch.Size([1, 3, 4, 128])],
        [torch.bfloat16, torch.bfloat16],
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    obj.get_tensor(0).fill_(19)
    try:
        remote.submit_put_task(key, obj).result(timeout=5)
    finally:
        obj.ref_count_down()
        remote.close()
    result, _ = work(
        "scheduler",
        "admin",
        {
            "method": "load_fs_chunks",
            "kwargs": {
                "config_path": str(cfg_path),
                "max_chunks": 1,
                "max_failed_keys": 2,
            },
        },
    )
    assert result.status == "COMPLETED", result.error
    assert result.value["loaded_chunks"] == 1
    assert len(created) == 2
    restored = owner.local.get_blocking(key)
    assert restored is not None, (key, list(owner.local.hot_cache))
    assert torch.all(restored.get_tensor(0) == 19)
    restored.ref_count_down()
    refused, _ = work(
        "scheduler",
        "admin",
        {
            "method": "load_fs_chunks",
            "kwargs": {
                "config_path": str(tmp_path / "missing.yaml"),
            },
        },
    )
    assert refused.status == "FAILED_SAFE"
    assert (
        refused.value["admin_error"] == "HTTPException"
        and refused.value["status_code"] == 400
    )


def test_admin_canonical_clear_and_cpu_metrics_use_actual_storage(setup):
    owner, runtime, work = setup
    from prometheus_client.parser import text_string_to_metric_families
    from lmcache.v1.rpc.storage_admin import CPU_STORAGE_METRIC_NAMES

    chunks = list(runtime.token_database.process_tokens(tokens=list(range(5))))
    for start, end, key in chunks:
        page, _ = runtime.allocate_checkpoint_fragment(0, end - start)
        owner.local.batched_submit_layer_pages([key], [page])
        page.ref_count_down()
    usage = owner.allocator.address_manager.total_allocated_size
    scraped, _ = work("scheduler", "admin", {"method": "metrics"})
    assert scraped.status == "COMPLETED", scraped.error
    families = list(text_string_to_metric_families(scraped.value.decode()))
    assert {family.name for family in families} <= CPU_STORAGE_METRIC_NAMES
    cache_usage = next(
        family for family in families if family.name == "lmcache:local_cache_usage"
    )
    assert cache_usage.samples[0].value == usage > 0
    cleared, _ = work(
        "scheduler",
        "admin",
        {
            "method": "clear_tokens",
            "kwargs": {
                "hashes": tuple(wire_hash(key.chunk_hash, 64) for _, _, key in chunks),
                "offsets": (4, 1),
                "request_configs": {},
                "locations": ["LocalCPUBackend"],
            },
        },
    )
    assert cleared.value == 2, cleared.error
    assert owner.allocator.address_manager.total_allocated_size == 0
    reset, _ = work("scheduler", "admin", {"method": "reset_metrics"})
    assert reset.status == "COMPLETED"


def test_checkpoint_seal_keeps_accepted_tokens_and_mixed_plane_cropping(setup):
    _, runtime, work = setup
    reserved, original = capture(work)
    assert reserved.status == "COMPLETED", reserved.error
    value = reserved.value
    fill_capture(runtime, value["reservation_id"])
    captured, _ = work(
        "worker-0",
        "commit_write",
        {
            "reservation_id": value["reservation_id"],
            "phase": "captured",
            "proof": "fenced",
        },
    )
    assert captured.value["status"] == "CAPTURED"
    for group in value["groups"]:
        retire(runtime, original, group["lease_id"])
    assert runtime.registry.snapshot()["leases"] == 2  # M's private publication hold.
    sealed, _ = work(
        "scheduler",
        "commit_write",
        {
            "reservation_id": value["reservation_id"],
            "phase": "seal",
            "tokens": tuple(range(7)),
        },
    )
    assert sealed.value["status"] == "COMMITTED", sealed.error
    assert runtime.registry.snapshot()["leases"] == 0
    manifest = runtime.checkpoint_store.manifest("request", 1)
    assert manifest.tokens == tuple(range(7))
    base, owners = runtime.checkpoint_store.normalize("request", 1, list(range(7)), {})
    try:
        assert base == 0
        index_keys = list(
            runtime.token_database.process_tokens(tokens=list(range(7)), kv_group=1)
        )
        pages, count = runtime.checkpoint_backend().batched_get_layer_page_prefix(
            [index_keys[-1][2]]
        )
        assert count == 1
        page = pages[0]
        assert page.valid_tokens == 3
        for layer, planes in enumerate(runtime.checkpoint_plane_widths(1)):
            begin = 0
            for plane, width in enumerate(planes):
                end = begin + width * 3
                assert torch.all(
                    page.layer_tensor(layer)[begin:end] == 20 + layer * 3 + plane + 1
                )
                begin = end
        page.ref_count_down()
    finally:
        runtime._release_shared_retrieve_objs(owners, unpin=False)


def test_old_single_plane_freezer_corrupts_the_actual_checkpoint_rope_crop(setup):
    _, runtime, _ = setup
    source, _ = runtime.allocate_checkpoint_fragment(0, 4)
    old_destination, _ = runtime.allocate_checkpoint_fragment(0, 3)
    corrected_destination, _ = runtime.allocate_checkpoint_fragment(0, 3)
    module = importlib.import_module("lmcache_ascend.v1.local_checkpoint")
    try:
        for layer in range(source.num_layers):
            source.layer_tensor(layer)[: 512 * 4].fill_(1)
            source.layer_tensor(layer)[512 * 4 :].fill_(2)
        # Reproduce the pre-fix freezer: shape576 collapsed latent512+RoPE64
        # into one plane. Byte size still matches, but cropping is different.
        old_layout = freeze_storage_layout(
            runtime.config,
            runtime.metadata,
            runtime.token_database,
            allocator_alignment=runtime.allocator_alignment,
        )
        old_widths = tuple(
            width // 2 for width in old_layout.groups[0].plane_token_bytes[0]
        )
        spans = [(module.CheckpointPage(0, 4, None), source)]
        module.LocalCheckpointStore._assemble(old_destination, old_widths, spans, 0, 3)
        module.LocalCheckpointStore._assemble(
            corrected_destination,
            runtime.checkpoint_plane_widths(0),
            spans,
            0,
            3,
        )
        for layer in range(source.num_layers):
            assert torch.all(old_destination.layer_tensor(layer)[512 * 3 :] == 1)
            assert torch.all(corrected_destination.layer_tensor(layer)[: 512 * 3] == 1)
            assert torch.all(corrected_destination.layer_tensor(layer)[512 * 3 :] == 2)
    finally:
        for page in (source, old_destination, corrected_destination):
            page.ref_count_down()


def test_checkpoint_pressure_preserves_bounded_partial_progress(setup, monkeypatch):
    _, runtime, work = setup
    allocate = runtime.allocate_checkpoint_fragment

    def small(group, tokens, caches=None):
        if tokens > 2:
            raise MemoryError("injected pressure")
        return allocate(group, tokens, caches)

    monkeypatch.setattr(runtime, "allocate_checkpoint_fragment", small)
    monkeypatch.setattr(runtime, "reclaim_checkpoint_capacity", lambda *args: False)
    result, original = capture(work, end=9)
    assert result.status == "COMPLETED", result.error
    assert result.value["captured_end"] == 2
    assert all(group["ends"] == (2,) for group in result.value["groups"])
    cancelled, _ = work(
        "worker-0",
        "cancel_write",
        {
            "reservation_id": result.value["reservation_id"],
            "proof": "never_imported",
        },
    )
    assert cancelled.value["status"] == "CANCELLED"
    for group in result.value["groups"]:
        retire(runtime, original, group["lease_id"])
    assert runtime.registry.snapshot()["leases"] == 0


def test_flat_store_uses_existing_group_shape_and_batched_put_ownership(setup):
    owner, runtime, work = setup
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(5))))
    reserved, original = work(
        "worker-0",
        "reserve_write",
        {
            "purpose": "store",
            "groups": [
                {
                    "group_id": 0,
                    "representation": "flat",
                    "starts": tuple(a for a, _, _ in chunks),
                    "ends": tuple(b for _, b, _ in chunks),
                    "hashes": tuple(
                        wire_hash(key.chunk_hash, 64) for _, _, key in chunks
                    ),
                }
            ],
            "request_configs": {},
        },
    )
    assert reserved.status == "COMPLETED", reserved.error
    reservation = runtime._write_reservations[reserved.value["reservation_id"]]
    objects = [obj for _, _, obj in reservation.fragments[0]]
    assert all(not isinstance(obj, LayerPageMemoryObj) for obj in objects)
    assert [obj.get_shapes() for obj in objects] == [
        [runtime.metadata.get_shapes(n)[0]] for n in (4, 1)
    ]
    for obj in objects:
        obj.tensor.fill_(13)
    committed, _ = work(
        "worker-0",
        "commit_write",
        {
            "reservation_id": reserved.value["reservation_id"],
            "proof": "fenced",
        },
    )
    assert committed.value["status"] == "COMMITTED", committed.error
    retire(runtime, original, reserved.value["groups"][0]["lease_id"])
    for _, _, key in chunks:
        obj = owner.local.get_blocking(key)
        assert torch.all(obj.tensor == 13)
        obj.ref_count_down()
    assert all(obj.get_ref_count() == 1 for obj in objects)


@pytest.mark.parametrize("representation", ["page", "layers"])
def test_layerwise_store_preserves_canonical_keys_and_mixed_planes(
    setup, representation
):
    owner, runtime, work = setup
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(5))))
    entries = [
        {
            "group_id": group,
            "representation": representation,
            "starts": tuple(a for a, _, _ in chunks),
            "ends": tuple(b for _, b, _ in chunks),
            "hashes": tuple(wire_hash(key.chunk_hash, 64) for _, _, key in chunks),
        }
        for group in (0, 1)
    ]
    reserved, original = work(
        "worker-0",
        "reserve_write",
        {
            "purpose": "store",
            "groups": entries,
            "request_configs": {},
        },
    )
    assert reserved.status == "COMPLETED", reserved.error
    reservation = runtime._write_reservations[reserved.value["reservation_id"]]
    for group, fragments in reservation.fragments.items():
        for _, _, obj in fragments:
            if isinstance(obj, LayerPageMemoryObj):
                for layer in range(obj.num_layers):
                    obj.layer_tensor(layer).fill_(13 + group)
            else:
                obj.tensor.fill_(13 + group)
    if representation == "layers":
        assert reserved.value["groups"][1]["layer_ids"] == (0, 1, 2, 0, 1, 2)
        assert [
            item["logical_size"] for item in reserved.value["groups"][1]["objects"]
        ] == [
            520,
            1024,
            520,
            130,
            256,
            130,
        ]
    committed, _ = work(
        "worker-0",
        "commit_write",
        {
            "reservation_id": reserved.value["reservation_id"],
            "proof": "fenced",
        },
    )
    assert committed.status == "COMPLETED", committed.error
    for group in reserved.value["groups"]:
        retire(runtime, original, group["lease_id"])
    for group in (0, 1):
        for _, _, key in runtime.token_database.process_tokens(
            tokens=list(range(5)), kv_group=group
        ):
            if representation == "page":
                pages, count = owner.local.batched_get_layer_page_prefix([key])
                assert count == 1
                for layer in range(pages[0].num_layers):
                    assert torch.all(pages[0].layer_tensor(layer) == 13 + group)
                pages[0].ref_count_down()
            else:
                for layer_key in key.split_layers(runtime.num_layers_for_group(group)):
                    obj = owner.local.get_blocking(layer_key)
                    assert torch.all(obj.tensor == 13 + group)
                    obj.ref_count_down()


def test_page_store_refusal_allows_the_established_layer_fallback(setup, monkeypatch):
    owner, runtime, work = setup
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(5))))
    entry = {
        "group_id": 1,
        "representation": "page",
        "starts": (0, 4),
        "ends": (4, 5),
        "hashes": tuple(wire_hash(key.chunk_hash, 64) for _, _, key in chunks),
    }
    monkeypatch.setattr(
        owner.local, "batched_allocate_layer_pages", lambda *args, **kwargs: None
    )
    refused, _ = work(
        "worker-0", "reserve_write", {"purpose": "store", "groups": [entry]}
    )
    assert refused.status == "FAILED_SAFE"
    assert refused.value == {"fallback": "layers"}
    assert owner.allocator.get_capacity_bytes() == (65536, 65536)
    entry["representation"] = "layers"
    fallback, original = work(
        "worker-0", "reserve_write", {"purpose": "store", "groups": [entry]}
    )
    assert fallback.status == "COMPLETED", fallback.error
    cancelled, _ = work(
        "worker-0",
        "cancel_write",
        {
            "reservation_id": fallback.value["reservation_id"],
            "proof": "never_imported",
        },
    )
    assert cancelled.value["status"] == "CANCELLED"
    retire(runtime, original, fallback.value["groups"][0]["lease_id"])
    monkeypatch.setattr(
        runtime.storage_manager, "allocate", lambda *args, **kwargs: None
    )
    refused, _ = work(
        "worker-0", "reserve_write", {"purpose": "store", "groups": [entry]}
    )
    assert refused.status == "FAILED_SAFE" and refused.value == {"store_refused": True}


def test_store_growth_promotes_only_an_actual_paired_prefix_and_live_source(setup):
    owner, runtime, work = setup

    def plan(end):
        chunks = list(runtime.token_database.process_tokens(tokens=list(range(end))))
        return {
            "starts": tuple(a for a, _, _ in chunks),
            "ends": tuple(b for _, b, _ in chunks),
            "hashes": tuple(wire_hash(key.chunk_hash, 64) for _, _, key in chunks),
            "request_configs": {},
        }

    def populate(end, groups=(0, 1)):
        for group in groups:
            for start, stop, key in runtime.token_database.process_tokens(
                tokens=list(range(end)), kv_group=group
            ):
                if owner.local.contains(key):
                    continue
                page, _ = runtime.allocate_checkpoint_fragment(group, stop - start)
                for layer in range(page.num_layers):
                    page.layer_tensor(layer).fill_(group + 3)
                owner.local.batched_submit_layer_pages([key], [page])
                page.ref_count_down()

    populate(5)
    options = {
        "plan": plan(5),
        "source_end": 5,
        "participants": ("worker-0", "worker-1"),
        "layout_digest": runtime.layout.digest,
        "mode": "resident",
    }
    initial, old = work("worker-0", "prepare_store", options)
    assert initial.status == "COMPLETED", initial.error
    descriptor = {
        "identity": old.identity,
        "manifest": initial.value.groups[0],
        "payload_digest": old.payload_digest,
    }
    populate(9, groups=(0,))
    options.update(plan=plan(9), source_end=9, source_operation=descriptor)
    refused, _ = work("worker-0", "prepare_store", options)
    assert refused.status == "FAILED_SAFE" and refused.value == {
        "promotion_refused": True
    }
    populate(9, groups=(1,))
    promoted, current = work("worker-0", "prepare_store", options)
    assert promoted.status == "COMPLETED", promoted.error
    assert promoted.value.source_end == 9
    for group in initial.value.groups:
        for rank in range(2):
            retire(runtime, old, group.lease_id, f"worker-{rank}")
    stale, _ = work("worker-0", "prepare_store", options)
    assert stale.status == "FAILED_SAFE" and "live ownership" in stale.error
    for group in promoted.value.groups:
        for rank in range(2):
            retire(runtime, current, group.lease_id, f"worker-{rank}")
    assert runtime.registry.snapshot()["leases"] == 0


def test_checkpoint_restore_keeps_a_separate_scheduler_ack_lease(setup):
    _, runtime, work = setup
    reserved, original = capture(work)
    fill_capture(runtime, reserved.value["reservation_id"])
    work(
        "worker-0",
        "commit_write",
        {
            "reservation_id": reserved.value["reservation_id"],
            "phase": "captured",
            "proof": "fenced",
        },
    )
    work(
        "scheduler",
        "commit_write",
        {
            "reservation_id": reserved.value["reservation_id"],
            "phase": "seal",
            "tokens": tuple(range(9)),
        },
    )
    for group in reserved.value["groups"]:
        retire(runtime, original, group["lease_id"])
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(9))))
    hit, _ = work(
        "scheduler",
        "lookup",
        {
            "starts": tuple(a for a, _, _ in chunks),
            "ends": tuple(b for _, b, _ in chunks),
            "hashes": tuple(wire_hash(key.chunk_hash, 64) for _, _, key in chunks),
            "request_configs": {LOCAL_CHECKPOINT_CONFIG: 1},
        },
    )
    assert hit.value.matched_tokens == 9, hit.error
    result, restore = work(
        "worker-0",
        "prepare",
        PrepareSpec(
            hit.value.ticket_id,
            hit.value.key_plan_digest,
            9,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
            mode="checkpoint_restore",
            checkpoint_generation=1,
        ),
    )
    assert result.status == "COMPLETED", result.error
    assert result.value.restore_lease_id is not None
    for group in result.value.groups:
        for rank in range(2):
            retire(runtime, restore, group.lease_id, f"worker-{rank}")
    assert runtime.registry.snapshot()["leases"] == 1
    for rank in range(2):
        retire(runtime, restore, result.value.restore_lease_id, f"worker-{rank}")
    assert runtime.registry.snapshot()["leases"] == 0


def test_unknown_writer_completion_quarantines_real_destinations(setup):
    owner, runtime, work = setup
    before = owner.allocator.get_capacity_bytes()[0]
    reserved, original = capture(work)
    after = owner.allocator.get_capacity_bytes()[0]
    assert after < before
    cancelled, _ = work(
        "worker-0",
        "cancel_write",
        {
            "reservation_id": reserved.value["reservation_id"],
            "proof": "unknown",
        },
    )
    assert cancelled.value["status"] == "QUARANTINED"
    for group in reserved.value["groups"]:
        retire(runtime, original, group["lease_id"])
    assert owner.allocator.get_capacity_bytes()[0] == after
    assert runtime.registry.snapshot()["quarantined"] == 1
    assert not runtime.is_healthy()
    with pytest.raises(RuntimeError, match="quarantined"):
        runtime.close()


def test_close_preflights_tickets_and_retains_remote_owners_discovered_armed(setup):
    owner, runtime, work = setup
    chunks = list(runtime.token_database.process_tokens(tokens=[1]))
    for group in (0, 1):
        key = next(
            iter(runtime.token_database.process_tokens(tokens=[1], kv_group=group))
        )[2]
        page, _ = runtime.allocate_checkpoint_fragment(group, 1)
        owner.local.batched_submit_layer_pages([key], [page])
        page.ref_count_down()
    hit, _ = work(
        "scheduler",
        "lookup",
        {
            "starts": (0,),
            "ends": (1,),
            "hashes": (wire_hash(chunks[0][2].chunk_hash, 64),),
            "request_configs": {},
        },
    )
    stopped = []
    remote = SimpleNamespace(close=lambda: stopped.append(True))
    runtime._remote_fill_runtime = remote
    with pytest.raises(RuntimeError, match="tickets"):
        runtime.close()
    assert stopped == [] and runtime._remote_fill_runtime is remote
    work("scheduler", "drop_lookup", {"ticket_id": hit.value.ticket_id})

    def shutdown_finds_armed():
        stopped.append(True)
        runtime._remote_fill_require_paired_restart(("armed-at-shutdown",))

    remote.close = shutdown_finds_armed
    with pytest.raises(RuntimeError, match="unresolved native"):
        runtime.close()
    assert runtime._remote_fill_runtime is remote
    assert runtime.storage_manager.loop.is_running()
    assert not runtime.is_healthy()
    runtime._remote_fill_runtime = None  # This CPU-only injected receiver owns no DMA.


def test_close_refuses_new_work_before_stopping_backends(setup):
    _, runtime, work = setup
    outcomes = []

    def stopping():
        response, _ = work(
            "scheduler",
            "lookup",
            {
                "starts": (),
                "ends": (),
                "hashes": (),
                "request_configs": {},
            },
        )
        outcomes.append(response.status)

    runtime._remote_fill_runtime = SimpleNamespace(close=stopping)
    runtime.close()
    assert outcomes == ["FAILED_SAFE"]


def test_checkpoint_restore_starts_group1_at_aligned_tail(setup, monkeypatch):
    owner, runtime, work = setup
    runtime.config.dsa_group1_load_mode = "persistent_direct_hbm"
    runtime.config.pd_role = "receiver"
    runtime.config.retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]
    contains = runtime.storage_manager.batched_contains_two_group_layer_pages
    remote_checks = []

    def paired(left, right, locations, pin):
        if locations == ["RemoteBackend"]:
            remote_checks.append((left, right))
            return len(left), {
                "RemoteBackend": [
                    key for pair in zip(left, right, strict=False) for key in pair
                ]
            }
        return contains(left, right, locations, pin)

    monkeypatch.setattr(
        runtime.storage_manager, "batched_contains_two_group_layer_pages", paired
    )
    monkeypatch.setattr(
        runtime.storage_manager,
        "batched_get_external_pages",
        lambda *args: pytest.fail(
            "Group-1 prompt bytes must remain on direct HBM path"
        ),
    )
    prompt = list(range(5))
    chunks = list(runtime.token_database.process_tokens(tokens=prompt))
    pages = [
        runtime.allocate_checkpoint_fragment(0, end - start)[0]
        for start, end, _ in chunks
    ]
    for page in pages:
        for layer in range(page.num_layers):
            page.layer_tensor(layer).fill_(2)
    owner.local.batched_submit_layer_pages([key for _, _, key in chunks], pages)
    for page in pages:
        page.ref_count_down()
    query = {
        "starts": tuple(a for a, _, _ in chunks),
        "ends": tuple(b for _, b, _ in chunks),
        "hashes": tuple(wire_hash(key.chunk_hash, 64) for _, _, key in chunks),
        "request_configs": {},
    }
    hit, _ = work("scheduler", "lookup", query)
    prepared, original_source = work(
        "worker-0",
        "prepare",
        PrepareSpec(
            hit.value.ticket_id,
            hit.value.key_plan_digest,
            5,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
        ),
    )
    assert prepared.status == "COMPLETED", prepared.error
    assert prepared.value.actions == ("load_group1_direct",)
    source_manifest = prepared.value.groups[0]
    source_operation = {
        "identity": original_source.identity,
        "manifest": source_manifest,
        "payload_digest": original_source.payload_digest,
    }
    wrong_proof = dict(source_operation, payload_digest=b"x" * 32)
    refused, _ = capture(
        work, base=4, end=9, resident=5, prefix=5, source_operation=wrong_proof
    )
    assert refused.status == "FAILED_SAFE"
    assert "live ownership" in refused.error
    reserved, original = capture(
        work, base=4, end=9, resident=5, prefix=5, source_operation=source_operation
    )
    assert reserved.status == "COMPLETED", reserved.error
    fill_capture(runtime, reserved.value["reservation_id"])
    work(
        "worker-0",
        "commit_write",
        {
            "reservation_id": reserved.value["reservation_id"],
            "phase": "captured",
            "proof": "fenced",
        },
    )
    sealed, _ = work(
        "scheduler",
        "commit_write",
        {
            "reservation_id": reserved.value["reservation_id"],
            "phase": "seal",
            "tokens": tuple(range(9)),
        },
    )
    assert sealed.status == "COMPLETED", sealed.error
    for group in reserved.value["groups"]:
        retire(runtime, original, group["lease_id"])
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(9))))
    query.update(
        starts=tuple(a for a, _, _ in chunks),
        ends=tuple(b for _, b, _ in chunks),
        hashes=tuple(wire_hash(key.chunk_hash, 64) for _, _, key in chunks),
        request_configs={LOCAL_CHECKPOINT_CONFIG: 1},
    )
    hit, _ = work("scheduler", "lookup", query)
    assert hit.value.matched_tokens == 9, hit.error
    restored, restore_op = work(
        "worker-0",
        "prepare",
        PrepareSpec(
            hit.value.ticket_id,
            hit.value.key_plan_digest,
            9,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
            mode="checkpoint_restore",
            checkpoint_generation=1,
        ),
    )
    assert restored.status == "COMPLETED", restored.error
    assert restored.value.groups[0].source_start == 0
    assert restored.value.groups[1].source_start == 4
    assert set(restored.value.actions) == {
        "checkpoint_group1_tail",
        "load_group1_direct_prefix",
    }
    assert remote_checks
    for lease in (source_manifest.lease_id,):
        for rank in range(2):
            retire(runtime, original_source, lease, f"worker-{rank}")
    for lease in (
        *[group.lease_id for group in restored.value.groups],
        restored.value.restore_lease_id,
    ):
        for rank in range(2):
            retire(runtime, restore_op, lease, f"worker-{rank}")
    assert runtime.registry.snapshot()["leases"] == 0


def test_cpu_remote_fill_reuses_registered_session_and_exact_mixed_layout(
    setup, monkeypatch
):
    _, runtime, _ = setup
    _, module = ascend_modules()
    runtime.config.enable_remote_lmcache_store = True
    runtime.config.enable_shared_cpu_cache = True
    runtime.config.shared_cpu_cache_strict = True
    runtime.config.pd_role = "receiver"
    runtime.config.dsa_group1_load_mode = "persistent_direct_hbm"
    runtime.config.remote_fill_model_artifact_id = "test-artifact"
    runtime.config.remote_fill_cache_namespace = "test-namespace"
    runtime.config.remote_fill_min_free_bytes = 0
    runtime.config.remote_fill_min_free_ratio = 0
    runtime.config.remote_fill_max_control_pages_per_window = 1024
    runtime.config.extra_config.update(
        mooncake_page_first_multi_buffer=True, mooncake_layer_merged_page_objects=True
    )
    runtime.metadata.engine_id = "decoder-dp2"
    runtime.metadata.kv_connector_extra_config = {
        "lmcache_remote_fill_destination_dp_rank": 2,
        "lmcache_remote_fill_destination_dp_size": 4,
    }
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(
        runtime.storage_manager,
        "get_remote_fill_destination_session",
        lambda: "10.0.0.1:20000",
    )

    class IdleServer:
        """CPU control seam; native transport is outside this test's coverage."""

        def __init__(self):
            self.closed = Event()

        def serve_once(self):
            self.closed.wait(0.001)
            return False

        def close(self):
            self.closed.set()

    create = module.create_decoder_remote_fill_runtime
    arguments = {}

    def cpu_receiver(**kwargs):
        arguments.update(kwargs)
        return create(**kwargs, server_factory=lambda service: IdleServer())

    monkeypatch.setattr(
        module,
        "create_decoder_remote_fill_runtime",
        cpu_receiver,
    )
    runtime._initialize_decoder_remote_fill()
    try:
        remote = runtime._remote_fill_runtime
        assert remote.available
        assert arguments["local_backend"] is runtime.checkpoint_backend()
        assert arguments["layout"].group_layer_counts == (2, 3)
        assert arguments["layout"].groups[1].layer_token_dims == (130, 256, 130)
        placement = runtime.get_remote_fill_placement_info()
        assert placement["destination_remote_session"] == "10.0.0.1:20000"
        assert placement["dp_rank"] == 2
        assert placement["destination_tp_size"] == 4
        assert runtime._remote_fill_capacity_available(65536)
        assert not runtime._remote_fill_capacity_available(65537)
        runtime._remote_fill_require_paired_restart(("armed-transfer",))
        assert not runtime.is_healthy()
        assert runtime.get_remote_fill_placement_info() is None
    finally:
        runtime._remote_fill_runtime.close()
        runtime._remote_fill_runtime = None
