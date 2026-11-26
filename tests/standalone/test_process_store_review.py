# SPDX-License-Identifier: Apache-2.0
"""Independent P6 regressions for public store event/statistics semantics."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import ast

import pytest
import torch
from lmcache.v1.rpc.storage_protocol import PrepareSpec, StorageRequest

from test_process_store_worker import engine
from test_ascend_cpu_storage_runtime import setup as runtime_setup


@pytest.fixture
def setup():
    yield from runtime_setup.__wrapped__()


def test_hashes_only_store_preserves_existing_enabled_event_payload(setup):
    module, worker, runtime, _ = engine(setup)
    worker.kv_events_enabled, worker.kv_events = True, []
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(5))))
    hashes = [key.chunk_hash for _, _, key in chunks]
    module.run_store(worker, "request", None, hashes, [4, 1], None, {"sync": True})
    assert [event.token_ids for event in worker.kv_events] == [hashes, []]
    assert [event.block_size for event in worker.kv_events] == [4, 1]
    assert [event.block_hashes for event in worker.kv_events] == [[h] for h in hashes]


def test_zero_requested_tokens_from_an_all_false_mask_stays_zero_in_stats(setup):
    module, worker, _, _ = engine(setup)
    requested = []

    def request(count):
        requested.append(count)
        return SimpleNamespace(profile_process_tokens=nullcontext)

    worker.stats_monitor = SimpleNamespace(on_store_request=request)
    module.run_store(
        worker,
        "request",
        list(range(4)),
        None,
        None,
        torch.zeros(4, dtype=torch.bool),
        {"sync": True},
        num_to_store_tokens=0,
    )
    assert requested == [0]


@pytest.mark.parametrize("promoted,expected", [(0, False), (8, False), (9, True)])
def test_partial_paired_promotion_cannot_complete_a_larger_decode_window(
    promoted, expected
):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_decode_window_save_group_pointer_ready"
    )
    source = ast.parse("from __future__ import annotations")
    source.body.append(method)
    namespace = {}
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), namespace)
    source_owner = object()
    state = SimpleNamespace(
        token_count=promoted,
        indexer_npu_resident=True,
        process_load_generation=2,
        prepared_sparse_sources={0: source_owner},
    )
    adapter = SimpleNamespace(
        _worker_retrieve_state={"request": state},
        lmcache_engine=SimpleNamespace(
            compact_source_importer=SimpleNamespace(
                is_certified_source=lambda source, request, generation: source
                is source_owner,
            )
        ),
    )
    request = SimpleNamespace(
        req_id="request",
        decode_window_start=4,
        decode_window_end=9,
        token_ids=list(range(9)),
    )
    result = SimpleNamespace(
        request_id="request", kv_group=1, process_store_plan={"promoted_end": promoted}
    )
    assert namespace[method.name](adapter, request, 1, result) is expected


def test_actual_layer_fallback_remains_a_retrievable_local_two_group_source(
    setup, monkeypatch
):
    module, worker, runtime, _ = engine(setup)
    monkeypatch.setattr(
        runtime.storage_manager, "supports_batched_put_layer_pages", lambda *args: False
    )
    for group in (0, 1):
        result = list(
            module.store_layer(
                worker,
                list(range(5)),
                None,
                {"req_id": "request", "kv_group": group, "sync": True},
            )
        )[-1]
        assert result.committed_end == 5
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(5))))
    response, _ = setup[2](
        "scheduler",
        "lookup",
        {
            "starts": tuple(a for a, _, _ in chunks),
            "ends": tuple(b for _, b, _ in chunks),
            "hashes": tuple(key.chunk_hash for _, _, key in chunks),
            "request_configs": {},
        },
    )
    assert response.status == "COMPLETED", response.error
    assert response.value.matched_tokens == 5
    future = worker.storage_process.client.request(
        "prepare",
        PrepareSpec(
            response.value.ticket_id,
            response.value.key_plan_digest,
            5,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
        ),
        request_id="request",
        request_attempt="attempt",
        load_generation=1,
        purpose="compact",
    )
    prepared = future.result()
    assert prepared.status == "COMPLETED", prepared.error
    assert all(not group.page_offsets for group in prepared.value.groups)
    for group in prepared.value.groups:
        for participant in ("worker-0", "worker-1"):
            retired = runtime.handle(
                StorageRequest(
                    future.identity,
                    "retire",
                    {"lease_id": group.lease_id, "proof": "never_imported"},
                    future.payload_digest,
                    runtime.registry.sessions[participant],
                    participant_session=participant,
                )
            )
            assert retired.status == "RETIRED"


def test_flat_prefix_then_recovered_page_store_remains_preparable(setup, monkeypatch):
    from lmcache.v1.rpc.storage_import import CompactSourceImporter

    owner = setup[0]
    module, worker, runtime, _ = engine(setup)
    supports_pages = runtime.storage_manager.supports_batched_put_layer_pages
    for end, start, pages in ((4, 0, False), (8, 4, True), (9, 8, False)):
        monkeypatch.setattr(
            runtime.storage_manager,
            "supports_batched_put_layer_pages",
            supports_pages if pages else lambda *args: False,
        )
        mask = torch.ones(end, dtype=torch.bool)
        mask[:start] = False
        for group in (0, 1):
            list(
                module.store_layer(
                    worker,
                    list(range(end)),
                    mask,
                    {"req_id": "request", "kv_group": group, "sync": True},
                )
            )
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(9))))
    response, _ = setup[2](
        "scheduler",
        "lookup",
        {
            "starts": tuple(a for a, _, _ in chunks),
            "ends": tuple(b for _, b, _ in chunks),
            "hashes": tuple(key.chunk_hash for _, _, key in chunks),
            "request_configs": {},
        },
    )
    assert response.value.matched_tokens == 9
    future = worker.storage_process.client.request(
        "prepare",
        PrepareSpec(
            response.value.ticket_id,
            response.value.key_plan_digest,
            9,
            runtime.layout.digest,
            ("worker-0", "worker-1"),
        ),
        request_id="request",
        request_attempt="attempt",
        load_generation=1,
        purpose="compact",
    )
    prepared = future.result()
    assert prepared.status == "COMPLETED", prepared.error
    worker.storage_process.mapping.tensor = owner.allocator.buffer
    importer = CompactSourceImporter(
        layout=runtime.layout,
        mapping=worker.storage_process.mapping,
        local_device_base=4096,
        client=worker.storage_process.client,
        upload=lambda table: table,
        fence_upload=lambda: None,
    )
    sources = importer.import_reply(
        prepared.value,
        identity=future.identity,
        payload_digest=future.payload_digest,
        key_plan_digest=response.value.key_plan_digest,
        source_end=9,
    )
    for manifest in prepared.value.groups:
        group, source = manifest.group_id, sources[manifest.group_id]
        assert manifest.page_positions() == (1,) and manifest.flat_positions() == (0, 2)
        keys = list(
            runtime.token_database.process_tokens(tokens=list(range(9)), kv_group=group)
        )
        middle, count = owner.local.batched_get_layer_page_prefix([keys[1][2]])
        assert count == 1
        rows, child = importer.borrow_source_tensors(source)
        try:
            for layer in range(runtime.num_layers_for_group(group)):
                first = owner.local.get_blocking(
                    keys[0][2].split_layers(runtime.num_layers_for_group(group))[layer]
                )
                last = owner.local.get_blocking(
                    keys[2][2].split_layers(runtime.num_layers_for_group(group))[layer]
                )
                expected = (
                    first.metadata.address,
                    middle[0].metadata.address + middle[0].group_prefix_sum[layer],
                    last.metadata.address,
                )
                assert source.layers[layer].chunk_ptrs_npu.tolist() == [
                    4096 + x for x in expected
                ]
                assert [
                    tensor.data_ptr() - owner.allocator.buffer.data_ptr()
                    for tensor in rows[layer]
                ] == list(expected)
                assert all(torch.all(tensor == layer + 1) for tensor in rows[layer])
                first.ref_count_down()
                last.ref_count_down()
        finally:
            child.release("never_imported")
            middle[0].ref_count_down()
    owner.local.clear()
    held = owner.allocator.get_capacity_bytes()[0]
    assert held < 65536
    for source in sources.values():
        source.owner_lease.release("fenced")
    assert owner.allocator.get_capacity_bytes()[0] == held
    for manifest in prepared.value.groups:
        retired = runtime.handle(
            StorageRequest(
                future.identity,
                "retire",
                {
                    "lease_id": manifest.lease_id,
                    "proof": "never_imported",
                },
                future.payload_digest,
                runtime.registry.sessions["worker-1"],
                participant_session="worker-1",
            )
        )
        assert retired.status == "RETIRED"
    assert runtime.registry.snapshot()["leases"] == 0
    owner.local.clear()
    assert owner.allocator.get_capacity_bytes()[0] == 65536
    assert importer.sources_drained()
