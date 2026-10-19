# SPDX-License-Identifier: Apache-2.0
"""Actual M/compact import and Ascend prepared generator, with CPU native doubles."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy
from queue import Queue
from threading import Barrier
from types import SimpleNamespace as NS
from pathlib import Path
import ast
import gc

import pytest
import torch

from lmcache.integration.vllm.storage_process import (
    ordinary_layerwise_load,
    release_sources,
)
from lmcache.v1.rpc.storage_import import CompactSourceImporter
from lmcache.v1.rpc.storage_lifetime import NativeSourceLifetime
from lmcache.v1.rpc.storage_protocol import StorageLookupTicket, StorageRequest

from test_process_store_worker import done, engine
from test_storage_process_worker import definitions as worker_definitions
from test_storage_process_worker import loaded
from test_ascend_cpu_storage_runtime import setup as runtime_setup


@pytest.fixture
def setup():
    yield from runtime_setup.__wrapped__()


@pytest.fixture
def prepared_definitions(monkeypatch):
    return worker_definitions.__wrapped__(monkeypatch)


@pytest.fixture
def harness(setup, prepared_definitions, monkeypatch):
    owner, runtime, work = setup
    module, writer, _, _ = engine(setup)
    for group in (0, 1):
        list(
            module.store_layer(
                writer,
                list(range(5)),
                None,
                {"req_id": "request", "kv_group": group, "sync": True},
            )
        )
    chunks = list(runtime.token_database.process_tokens(tokens=list(range(5))))
    hit, lookup = work(
        "scheduler",
        "lookup",
        {
            "starts": tuple(a for a, _, _ in chunks),
            "ends": tuple(b for _, b, _ in chunks),
            "hashes": tuple(key.chunk_hash for _, _, key in chunks),
            "request_configs": {},
        },
    )
    assert hit.value.matched_tokens == 5
    ticket = StorageLookupTicket(hit.value, lookup.identity, lookup.payload_digest)
    _, prepared = prepared_definitions
    monkeypatch.setattr(
        torch,
        "npu",
        NS(stream=lambda _: nullcontext(), current_stream=lambda: "compute"),
        raising=False,
    )
    queue, agreement = Queue(), Barrier(2)
    ready, transfers, adapters = [False], [], []

    class Event:
        def record(self, stream):
            assert stream == "compute"

        def query(self):
            return ready[0]

        def synchronize(self):
            if not ready[0]:
                raise RuntimeError("unknown native completion")

    for rank in range(2):
        worker = copy(writer)
        worker.metadata = copy(writer.metadata)
        worker.metadata.worker_id = rank
        worker._is_passive = lambda rank=rank: rank != 0
        worker.storage_manager = None
        worker._should_use_shared_layerwise_retrieve = lambda group: True
        worker.broadcast_object_fn = lambda value, first, rank=rank: (
            queue.put(value) if rank == 0 else queue.get(timeout=3)
        )
        worker._ordered_checkpoint_agreement = lambda *args: (
            agreement.wait(3),
            args[-1],
        )[1]
        client = writer.storage_process.client
        if rank:
            client = NS(
                retire=lambda identity, digest, lease, proof: done(
                    runtime.handle(
                        StorageRequest(
                            identity,
                            "retire",
                            {"lease_id": lease, "proof": proof},
                            digest,
                            b"b",
                            participant_session="worker-1",
                        )
                    )
                ),
            )
        worker.storage_process = NS(
            client=client, descriptor={"sessions": {"worker-0": "a", "worker-1": "b"}}
        )
        mapping = NS(
            shm_name=owner.allocator.shm_name,
            size=owner.allocator.buffer.numel(),
            generation=1,
            tensor=owner.allocator.buffer,
        )
        importer = CompactSourceImporter(
            layout=runtime.layout,
            mapping=mapping,
            local_device_base=4096 * (rank + 1),
            client=client,
            upload=lambda table: table,
            fence_upload=lambda: None,
        )
        worker.compact_source_importer = importer
        lifetime = NativeSourceLifetime(Event, pytest.fail)
        worker.native_source_lifetime = lifetime
        connector = NS(
            load_stream="load",
            synchronize_dense_load_stream=lambda: None,
            record_dense_load_readiness=lambda: object(),
            consume_dense_load_readiness=lambda event: None,
            stage_dense_load_tensor=lambda tensor, dtype: tensor.to(dtype),
        )

        def native(*, rank=rank, importer=importer, lifetime=lifetime, **kwargs):
            source, group = kwargs["prepared_sparse_source"], kwargs["kv_group"]
            for layer in range(len(source.layers)):
                payload = yield
                lifetime.note(source, "compute")
                transfers.append(
                    (rank, group, layer, kwargs["slot_mapping"].tolist(), payload)
                )
                assert importer.is_certified_source(
                    source, "request", hit.value.lookup_generation
                )
            yield
            yield

        connector.batched_to_gpu_head_token_wise = native
        worker.gpu_connector = connector
        worker._retrieve_prepared_sparse_layers = prepared.__get__(worker)
        monitor = NS(requested=[], finished=[])

        def requested(count, monitor=monitor):
            monitor.requested.append(count)
            return len(monitor.requested)

        monitor.on_retrieve_request = requested
        monitor.on_retrieve_finished = lambda identity, count, monitor=monitor: (
            monitor.finished.append((identity, int(count)))
        )
        worker.stats_monitor = monitor
        state_map = {}
        adapter = NS(
            lmcache_engine=worker,
            _worker_retrieve_state=state_map,
            _get_cold_load_coordinator=lambda: NS(last_latent_future=None),
            _num_layers_for_group=runtime.num_layers_for_group,
            _kvcaches_for_group=lambda group: (
                [None] * runtime.num_layers_for_group(group)
            ),
        )
        # Bind each callback to its own state map rather than the final loop map.
        adapter._drop_worker_retrieve_state = lambda req_id, states=state_map: (
            states.pop(req_id, None)
        )
        adapter._set_worker_retrieve_state = lambda req_id, state, states=state_map: (
            states.__setitem__(req_id, state)
        )
        adapters.append(adapter)
    return adapters, ticket, runtime, ready, transfers


def request(ticket, count):
    return NS(
        req_id="request",
        token_ids=list(range(8)),
        request_configs={},
        is_sparse_decode=False,
        load_spec=NS(cpu_source_ticket=ticket, lmcache_cached_tokens=count),
    )


def prepare_pair(adapters, ticket, count, mask):
    def prepare(adapter):
        return ordinary_layerwise_load(
            adapter,
            request(ticket, count),
            list(range(count)),
            mask,
            torch.arange(7, 7 + count),
            torch.arange(17, 17 + count),
            request_ordinal=0,
        )

    with ThreadPoolExecutor(2) as pool:
        return list(pool.map(prepare, adapters))


@pytest.mark.parametrize("count", [3, 5])
def test_paired_partial_prefix_preserves_yields_mask_physical_tail_and_final_fence(
    harness, count
):
    adapters, ticket, runtime, ready, transfers = harness
    enabled = gc.isenabled()
    gc.disable()
    try:
        mask = torch.tensor([False] + [True] * (count - 1))
        pairs = prepare_pair(adapters, ticket, count, mask)
        for rank, (adapter, pair) in enumerate(zip(adapters, pairs, strict=True)):
            state = adapter._worker_retrieve_state["request"]
            assert state.token_count == count and not state.shared_request_active
            assert (
                state.prepared_sparse_sources[0].physical_chunk_token_counts[0][0] == 4
            )
            for group, retriever in enumerate(pair):
                returned = list(retriever)
                assert len(returned) == runtime.num_layers_for_group(group) + 2
                assert returned[-2] is None and torch.equal(returned[-1], mask)
            assert state.shared_request_active and state.indexer_npu_resident
            assert state.dense_prefix_resident_tokens == count
            assert state.process_load_generation == ticket.reply.lookup_generation
            assert state.metadata_token_ids == list(range(count))
            assert adapter.lmcache_engine.stats_monitor.requested == [count - 1] * 2
            assert adapter.lmcache_engine.stats_monitor.finished == [
                (1, count - 1),
                (2, count - 1),
            ]
        assert len(transfers) == 2 * (2 + 3)
        assert all(
            slots[0] == -1 and len(slots) == count for _, _, _, slots, _ in transfers
        )
        for adapter in adapters:
            release_sources(
                adapter._worker_retrieve_state["request"], adapter.lmcache_engine
            )
        assert runtime.registry.snapshot()["leases"] == 2
        ready[0] = True
        for adapter in adapters:
            adapter.lmcache_engine.native_source_lifetime.drain()
        assert runtime.registry.snapshot()["leases"] == 0
        assert all(
            adapter.lmcache_engine.compact_source_importer.sources_drained()
            for adapter in adapters
        )
    finally:
        if enabled:
            gc.enable()


def test_safe_prepare_miss_keeps_ordinary_false_masks_and_no_native_submission(
    harness, monkeypatch
):
    from lmcache.v1.cpu_cache import StorageFailedSafe

    adapters, ticket, runtime, _, transfers = harness
    monkeypatch.setattr(
        runtime,
        "prepare",
        lambda *a, **kw: (_ for _ in ()).throw(StorageFailedSafe("injected safe miss")),
    )
    pairs = prepare_pair(adapters, ticket, 5, torch.ones(5, dtype=torch.bool))
    for pair in pairs:
        for group, retriever in enumerate(pair):
            masks = list(retriever)
            assert len(masks) == runtime.num_layers_for_group(group) + 2
            assert all(mask is None for mask in masks[:-1])
            assert not masks[-1].any()
    assert not transfers and runtime.registry.snapshot()["leases"] == 0


def test_gc_off_early_close_retains_new_slot_operands_until_native_final_fence(harness):
    adapters, ticket, runtime, ready, _ = harness
    pairs = prepare_pair(adapters, ticket, 5, torch.ones(5, dtype=torch.bool))
    enabled = gc.isenabled()
    gc.disable()
    try:
        retained = []
        for adapter, pair in zip(adapters, pairs, strict=True):
            next(pair[0])
            next(pair[0])  # Actual prepared wrapper submits layer zero.
            source = adapter._worker_retrieve_state["request"].prepared_sparse_sources[
                0
            ]
            operand = source.compact_transfer_metadata["ordinary_slots"]
            retained.append((source, operand))
            pair[0].close()
            pair[1].close()
            release_sources(
                adapter._worker_retrieve_state["request"], adapter.lmcache_engine
            )
            assert source.compact_transfer_metadata["ordinary_slots"] is operand
            assert adapter.lmcache_engine.native_source_lifetime.retirements_pending
            assert not adapter.lmcache_engine.stats_monitor.finished
        assert runtime.registry.snapshot()["leases"] == 1
        ready[0] = True
        for adapter in adapters:
            adapter.lmcache_engine.native_source_lifetime.drain()
        assert runtime.registry.snapshot()["leases"] == 0
        assert all(
            source.compact_transfer_metadata["ordinary_slots"] is operand
            for source, operand in retained
        )
    finally:
        if enabled:
            gc.enable()


def test_cpu_group1_slots_are_staged_before_actual_prepared_device_validation(harness):
    from pathlib import Path

    adapters, ticket, _, _, _ = harness
    pairs = prepare_pair(adapters, ticket, 5, torch.ones(5, dtype=torch.bool))
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    validate = loaded(
        path,
        "_get_or_create_sparse_destination_plan",
        {
            "prepare_sparse_direct_destination_state": lambda *a: (_ for _ in ()).throw(
                RuntimeError("device check passed")
            ),
        },
        "VLLMPagedMemLayerwiseNPUConnector",
    )
    staged, validated = [], []
    for adapter, pair in zip(adapters, pairs, strict=True):
        connector = adapter.lmcache_engine.gpu_connector

        def stage(tensor, dtype):
            assert tensor.device.type == "cpu"
            staged.append(tensor)
            return tensor.to(device="meta", dtype=dtype)

        connector.stage_dense_load_tensor = stage

        def native(**kwargs):
            guard = NS(
                _expected_group_layers=lambda group: 3,
                _vllm_layer_cache_identity_signature=lambda layer: (
                    _ for _ in ()
                ).throw(RuntimeError("device check passed")),
            )
            with pytest.raises(RuntimeError, match="device check passed"):
                validate(
                    guard,
                    kvcaches_ref=[None] * 3,
                    kv_group=1,
                    slot_mapping_ref=kwargs["slot_mapping"],
                    sparse_kv_format=0,
                    sparse_k_hidden_dims=0,
                    sparse_v_hidden_dims=0,
                    sparse_dsa_hidden_dims=0,
                    expected_device=torch.device("meta"),
                )
            validated.append(True)
            yield

        connector.batched_to_gpu_head_token_wise = native
        next(pair[1])
        pair[1].close()
        pair[0].close()
        release_sources(
            adapter._worker_retrieve_state["request"], adapter.lmcache_engine
        )
    assert len(staged) == len(validated) == 2


def test_actual_adapter_ordinary_route_bypasses_worker_storage_owner(
    harness, monkeypatch
):
    """Extract only the deployed dispatch block; real M/W behavior is above."""
    from lmcache.integration.vllm import storage_process

    adapters, ticket, _, _, _ = harness
    adapter = adapters[0]
    worker = adapter.lmcache_engine
    tokens, mask = list(range(5)), torch.ones(5, dtype=torch.bool)
    # This is the concrete inherited path selected by the former adapter route.
    with pytest.raises(AssertionError):
        next(worker.retrieve_layer(tokens, mask, kv_group=0))
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    start = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_load_kv"
    )
    dispatch = next(
        node
        for node in ast.walk(start)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.ImportFrom)
            and any(alias.name == "ordinary_layerwise_load" for alias in child.names)
            for child in ast.walk(node)
        )
    )
    wrapper = ast.parse("""
def route(self, request, token_count, retrieve_tokens, token_mask,
          slot_mapping, idx, attn_metadata):
    for once in (None,):
        process_storage = getattr(self.lmcache_engine, 'storage_process', None)
    return self.layerwise_retrievers[-1]
""")
    wrapper.body[0].body[0].body.append(dispatch)
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(wrapper), str(path), "exec"), namespace)
    calls = []

    def prepared(*args, **kwargs):
        calls.append((args, kwargs))
        return iter([None] * 4), iter([None] * 5)

    monkeypatch.setattr(storage_process, "ordinary_layerwise_load", prepared)
    adapter.layerwise_retrievers = []
    adapter._layerwise_requests = []
    adapter._layerwise_retriever_is_sparse = []
    adapter._layerwise_sparse_shared_ordered = []
    adapter._prime_dense_prefix_retrievers = lambda *pair: [
        next(generator) for generator in pair for _ in range(2)
    ]
    req = request(ticket, 5)
    req.indexer_slot_mapping = [torch.arange(5)]
    result = namespace["route"](adapter, req, 5, tokens, mask, torch.arange(5), 0, None)
    assert result is adapter.layerwise_retrievers[0]
    assert len(calls) == 1 and calls[0][0][2] is tokens
    assert not hasattr(req.load_spec, "dsa_cold_compact_load")
    assert req.load_spec.lmcache_cached_tokens == 5
