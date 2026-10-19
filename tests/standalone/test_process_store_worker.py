# SPDX-License-Identifier: Apache-2.0
"""Actual M allocation/publication and W shared views; injected CPU D2H only."""

from concurrent.futures import Future
from types import SimpleNamespace as NS
import gc
import importlib

import msgspec
import pytest
import torch

from lmcache.v1.memory_management import LayerPageMemoryObj
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    StorageResult,
    content_digest,
)
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator
from test_ascend_cpu_storage_runtime import setup as runtime_setup


@pytest.fixture
def setup():
    yield from runtime_setup.__wrapped__()


class Worker(LMCacheEngine):
    def __init__(self, **values):
        self.__dict__.update(values)


def done(result):
    future = Future()
    future.set_result(result)
    return future


class Client:
    def __init__(self, runtime):
        self.runtime, self.operations, self.sequence = runtime, {}, 0

    def request(self, method, payload, **kwargs):
        self.sequence += 1
        identity = OperationIdentity(
            "cache",
            "epoch",
            "worker-0",
            self.sequence,
            kwargs["request_id"],
            kwargs["request_attempt"],
            kwargs["load_generation"],
            kwargs["purpose"],
        )
        request = StorageRequest(
            identity,
            method,
            payload,
            content_digest((method, payload)),
            b"a",
        )
        self.operations[identity.key] = request
        result = self.runtime.handle(request)
        future = done(result)
        future.identity, future.payload_digest = identity, request.payload_digest
        return future

    def acknowledge_terminal(self, identity, payload=None):
        original = self.operations[identity.key]
        result = self.runtime.handle(original)
        return done(
            self.runtime.handle(
                msgspec.structs.replace(
                    original,
                    method="acknowledge_terminal",
                    payload=payload
                    if payload is not None
                    else {
                        "terminal_result_digest": content_digest(
                            (result.status, result.value, result.error)
                        )
                    },
                )
            )
        )

    def cancel(self, identity):
        return done(
            self.runtime.handle(
                msgspec.structs.replace(
                    self.operations[identity.key],
                    method="cancel",
                    payload={},
                )
            )
        )

    def operation_status(self, identity):
        return done(
            self.runtime.handle(
                msgspec.structs.replace(
                    self.operations[identity.key],
                    method="operation_status",
                    payload={},
                )
            )
        )

    def retire(self, identity, digest, lease_id, proof):
        return done(
            self.runtime.handle(
                StorageRequest(
                    identity,
                    "retire",
                    {"lease_id": lease_id, "proof": proof},
                    digest,
                    b"a",
                    participant_session="worker-0",
                )
            )
        )


class Connector:
    def __init__(self):
        self.fences, self.transfers, self.group_calls = 0, [], 0
        self.store_stream = NS(synchronize=self.fence)
        self.fail = False

    def fence(self):
        if self.fail:
            raise RuntimeError("DMA unknown")
        self.fences += 1

    def batched_from_gpu(self, rows, starts, ends, **kwargs):
        for layer, row in enumerate(rows):
            for obj in row:
                tensor = (
                    obj.layer_tensor(layer)
                    if isinstance(obj, LayerPageMemoryObj)
                    else obj.tensor
                )
                tensor.fill_(layer + 1)
                self.transfers.append(obj)
            yield
            self.fence()
        yield

    def supports_batched_from_gpu_group(self, group):
        return True

    def batched_from_gpu_group(self, rows, starts, ends, **kwargs):
        self.group_calls += 1
        for _ in self.batched_from_gpu(rows, starts, ends, **kwargs):
            pass


def engine(setup):
    owner, runtime, _ = setup
    module = importlib.import_module("lmcache_ascend.v1.process_store")
    client, connector = Client(runtime), Connector()
    mapping = NS(
        shm_name=owner.allocator.shm_name,
        size=owner.allocator.buffer.numel(),
        generation=1,
    )
    failures = []
    result = Worker(
        config=runtime.config,
        metadata=runtime.metadata,
        storage_layout=runtime.layout,
        token_database=runtime.token_database,
        store_location=None,
        use_layerwise=True,
        storage_process=NS(
            client=client,
            mapping=mapping,
            timeout_s=2,
            drain_guards=[],
            mark_failed=failures.append,
        ),
        shared_cpu_cache_passive_allocator=PassiveSharedViewAllocator(
            slab_tensor=owner.allocator.buffer,
            shm_name=mapping.shm_name,
            generation=1,
        ),
        gpu_connector=connector,
        shared_cpu_cache_generation=1,
        broadcast_object_fn=lambda value, first: value,
        is_healthy=lambda: True,
        is_frozen=lambda: False,
        _is_passive=lambda: False,
        _get_req_id=lambda kw: kw["req_id"],
        _num_layers_for_kv_group=runtime.num_layers_for_group,
        _ensure_layerwise_connector_layout=lambda **kwargs: None,
        _num_transfer_layers_for_call=lambda group, kwargs: (
            runtime.num_layers_for_group(group)
        ),
        _expected_shared_cpu_chunk_metadata=runtime._expected_shared_cpu_chunk_metadata,
    )
    return module, result, runtime, failures


@pytest.mark.parametrize("group", [0, 1])
def test_layer_store_uses_real_registered_pages_and_preserves_mixed_c8(setup, group):
    module, worker, runtime, _ = engine(setup)
    outputs = list(
        module.store_layer(
            worker,
            list(range(5)),
            None,
            {
                "req_id": "request",
                "kv_group": group,
                "sync": True,
            },
        )
    )
    result = outputs[-1]
    assert result.committed_end == 5
    assert result.memory_objs == [] and result.process_store_plan["ends"] == (4, 5)
    assert not worker._process_store_writes
    assert all(guard() for guard in worker.storage_process.drain_guards)
    keys = list(
        runtime.token_database.process_tokens(tokens=list(range(5)), kv_group=group)
    )
    for _, _, key in keys:
        pages, count = runtime.checkpoint_backend().batched_get_layer_page_prefix([key])
        assert count == 1
        page = pages[0]
        for layer in range(runtime.num_layers_for_group(group)):
            assert torch.all(page.layer_tensor(layer) == layer + 1)
        if group == 1:
            assert page.layer_size_bytes(0) != page.layer_size_bytes(1)
        page.ref_count_down()


def test_page_refusal_reuses_actual_layer_fallback(setup, monkeypatch):
    module, worker, runtime, _ = engine(setup)
    monkeypatch.setattr(
        runtime.storage_manager, "supports_batched_put_layer_pages", lambda *args: False
    )
    output = list(
        module.store_layer(
            worker, list(range(5)), None, {"req_id": "request", "sync": True}
        )
    )[-1]
    assert output.committed_end == 5 and not worker._process_store_writes
    assert all(
        not isinstance(obj, LayerPageMemoryObj)
        for obj in worker.gpu_connector.transfers
    )


def test_group_store_optimization_is_preserved(setup):
    module, worker, _, _ = engine(setup)
    output = list(
        module.store_layer(
            worker,
            list(range(4)),
            None,
            {
                "req_id": "request",
                "sync": True,
                "decode_window_save": True,
                "windowed_sparse_save": True,
            },
        )
    )[-1]
    assert output.committed_end == 4 and worker.gpu_connector.group_calls == 1


def test_early_close_fences_then_cancels_without_publishing(setup):
    module, worker, runtime, _ = engine(setup)
    store = module.store_layer(
        worker, list(range(5)), None, {"req_id": "request", "sync": True}
    )
    next(store)
    store.close()
    assert worker.gpu_connector.fences == 1
    assert not worker._process_store_writes
    assert runtime.registry.snapshot()["leases"] == 0


def test_unknown_completion_retains_destinations_with_gc_disabled(setup):
    module, worker, _, failures = engine(setup)
    enabled = gc.isenabled()
    gc.disable()
    try:
        store = module.store_layer(
            worker, list(range(5)), None, {"req_id": "request", "sync": True}
        )
        next(store)
        worker.gpu_connector.fail = True
        with pytest.raises(RuntimeError, match="DMA unknown"):
            store.close()
        assert worker._process_store_writes
        assert not all(guard() for guard in worker.storage_process.drain_guards)
        assert failures and all(
            hold.is_valid()
            for state in worker._process_store_writes.values()
            for hold in state.holds
        )
    finally:
        if enabled:
            gc.enable()


def test_whole_group_flat_store_reuses_layerwise_transfer_with_exact_offsets(setup):
    module, worker, runtime, _ = engine(setup)
    module.run_store(
        worker, "request", list(range(5)), None, None, None, {"sync": True}
    )
    assert not worker._process_store_writes
    keys = [
        key
        for _, _, key in runtime.token_database.process_tokens(tokens=list(range(5)))
    ]
    for key in keys:
        obj = runtime._shared_local_cpu_backend().get_blocking(key)
        assert torch.all(obj.tensor[:, 0] == 1)
        assert torch.all(obj.tensor[:, 1] == 2)
        obj.ref_count_down()


def test_stale_destination_is_cancelled_and_all_leases_retired_before_ack(setup):
    module, worker, runtime, failures = engine(setup)
    worker.storage_process.mapping.generation = 2
    with pytest.raises(ValueError, match="stale slab"):
        list(module.store_layer(worker, list(range(5)), None, {"req_id": "request"}))
    assert not worker._process_store_writes and not failures
    assert runtime.registry.snapshot()["leases"] == 0


def test_constructor_failure_reaches_passive_completion_slot(setup):
    from queue import Queue
    from dataclasses import replace

    module, first, runtime, _ = engine(setup)
    _, passive, _, _ = engine(setup)
    queue = Queue()
    first.broadcast_object_fn = lambda value, rank: queue.put(value)
    passive.metadata = replace(runtime.metadata, worker_id=1)
    passive._is_passive = lambda: True
    passive.broadcast_object_fn = lambda value, rank: queue.get(timeout=2)
    first._ensure_layerwise_connector_layout = lambda **kw: (_ for _ in ()).throw(
        ValueError("layout constructor failed")
    )
    for participant in (first, passive):
        with pytest.raises(
            (ValueError, RuntimeError), match="layout constructor failed"
        ):
            list(
                module.store_layer(
                    participant, list(range(5)), None, {"req_id": "request"}
                )
            )


def test_busy_reservation_cancels_original_before_refusal_without_allocating(setup):
    module, worker, runtime, failures = engine(setup)
    client = worker.storage_process.client
    original = client.request
    calls = []

    def busy(method, payload, **kw):
        if method != "reserve_write":
            return original(method, payload, **kw)
        client.sequence += 1
        identity = OperationIdentity(
            "cache",
            "epoch",
            "worker-0",
            client.sequence,
            kw["request_id"],
            kw["request_attempt"],
            0,
            "store",
        )
        request = StorageRequest(
            identity, method, payload, content_digest((method, payload)), b"a"
        )
        client.operations[identity.key] = request
        calls.append(identity)
        result = done(StorageResult(identity, "BUSY"))
        result.identity, result.payload_digest = identity, request.payload_digest
        return result

    client.request = busy
    output = list(
        module.store_layer(worker, list(range(5)), None, {"req_id": "request"})
    )[-1]
    assert output.committed_end == 0 and len(calls) == 1 and not failures
    assert runtime.registry.snapshot()["operations"] == 0
    assert not worker._process_store_writes and not worker.gpu_connector.transfers


def test_paired_growth_reprepares_exact_partial_successor_with_actual_cpu_proof(
    setup, monkeypatch
):
    import ast
    from dataclasses import dataclass, field
    from pathlib import Path
    from types import ModuleType
    import sys
    from lmcache.integration.vllm import storage_process as bridge
    from lmcache.v1.rpc.storage_import import CompactSourceImporter
    from lmcache.v1.rpc.storage_lifetime import NativeSourceLifetime

    module, worker, runtime, _ = engine(setup)
    runtime.config.extra_config.update(
        mooncake_page_first_multi_buffer=True,
        mooncake_layer_merged_page_objects=True,
        save_chunk_meta=False,
    )
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    state_node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if getattr(n, "name", None) == "WorkerRetrieveState"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(state_node)
    namespace = {"dataclass": dataclass, "field": field}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    stub = ModuleType("lmcache.integration.vllm.vllm_v1_adapter")
    stub.WorkerRetrieveState = namespace["WorkerRetrieveState"]
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    worker.storage_process.descriptor = {"sessions": {"worker-0": "a"}}
    worker.compact_source_importer = CompactSourceImporter(
        layout=worker.storage_layout,
        mapping=worker.storage_process.mapping,
        local_device_base=4096,
        client=worker.storage_process.client,
        upload=lambda tensor: tensor,
        fence_upload=lambda: None,
    )
    worker.native_source_lifetime = NativeSourceLifetime(lambda: None, pytest.fail)
    worker._ordered_checkpoint_agreement = lambda *args: args[-1]
    states, completed = {}, []
    adapter = NS(
        lmcache_engine=worker,
        _worker_retrieve_state=states,
        _is_decode_window_save_request=lambda request: True,
        _decode_window_save_is_next_expected=lambda request: True,
        _num_layers_for_group=runtime.num_layers_for_group,
        _record_dsa_cold_dense_load_readiness=lambda state: None,
        _set_worker_retrieve_state=lambda req, state: states.update({req: state}),
        _record_decode_window_save_group_completed=lambda req, group, result: (
            completed.append((group, result.process_store_plan.get("promoted_end")))
        ),
        _release_shared_worker_retrieve_state=lambda state, engine: (
            bridge.release_sources(state, engine)
        ),
    )
    request = NS(
        req_id="request",
        load_spec=NS(dsa_cold_load_generation=1, cpu_source_ticket=None),
        request_configs={},
        token_ids=list(range(5)),
        decode_window_start=0,
        decode_window_end=5,
    )
    for end, start in ((5, 0), (9, 4)):
        request.token_ids = list(range(end))
        request.decode_window_start, request.decode_window_end = start, end
        mask = torch.ones(end, dtype=torch.bool)
        mask[:start] = False
        previous = states.get("request")
        for group in (0, 1):
            result = list(
                module.store_layer(
                    worker,
                    request.token_ids,
                    mask,
                    {
                        "req_id": "request",
                        "kv_group": group,
                        "sync": True,
                    },
                )
            )[-1]
            bridge.promote_store(adapter, request, result)
        assert "request" in states, [
            (
                op.method,
                runtime.handle(op).status,
                runtime.handle(op).error,
                runtime.handle(op).value,
            )
            for op in worker.storage_process.client.operations.values()
            if op.method == "prepare_store"
        ]
        state = states["request"]
        assert state.token_count == end and state.indexer_npu_resident
        assert state.prepared_sparse_sources[0].total_tokens == end
        assert state.process_source_plan["ends"][-1] == end
        assert worker.compact_source_importer.is_certified_source(
            state.prepared_sparse_sources[0], "request", state.process_load_generation
        )
        if previous is not None:
            assert not previous.prepared_sparse_sources[0].owner_lease.is_valid()
    bridge.release_sources(states["request"], worker)
    assert worker.compact_source_importer.sources_drained()
    assert completed == [(0, 5), (1, 5), (0, 9), (1, 9)]
