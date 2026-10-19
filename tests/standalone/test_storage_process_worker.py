# SPDX-License-Identifier: Apache-2.0
"""Real worker helpers/importer/envelopes; source-loaded Ascend generator, fake NPU."""

import ast
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Barrier
from types import MethodType, ModuleType, SimpleNamespace as NS
import sys

import msgspec
import pytest
import torch

from lmcache.integration.vllm import storage_process as worker
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.rpc.storage_lifetime import NativeSourceLifetime
from lmcache.v1.rpc.storage_protocol import (
    GroupLayout,
    GroupManifest,
    LookupReply,
    OperationIdentity,
    StorageLookupTicket,
    StorageResult,
    pack_uints,
)
from tests.standalone.test_storage_import import fixture


def loaded(path, name, namespace, cls=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = (
        tree.body
        if cls is None
        else next(n for n in tree.body if getattr(n, "name", None) == cls).body
    )
    node = next(n for n in body if getattr(n, "name", None) == name)
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(node)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def definitions(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    state = loaded(
        root / "lmcache/integration/vllm/vllm_v1_adapter.py",
        "WorkerRetrieveState",
        {"dataclass": dataclass, "field": field},
    )
    module = ModuleType("lmcache.integration.vllm.vllm_v1_adapter")
    module.WorkerRetrieveState = state
    monkeypatch.setitem(sys.modules, module.__name__, module)
    prepared = loaded(
        root.parent / "LMCache-Ascend/lmcache_ascend/v1/cache_engine.py",
        "_retrieve_prepared_sparse_layers",
        {"torch": torch},
        "AscendLMCacheEngine",
    )
    monkeypatch.setattr(
        torch,
        "npu",
        NS(set_device=lambda _: None, stream=lambda _: nullcontext()),
        raising=False,
    )
    return state, prepared


def complete(value=None):
    future = Future()
    future.set_result(value)
    return future


class Connector:
    def __init__(self):
        self.load_stream = object()
        self.payloads = []
        self.slots = []
        self.fail_sync = False

    def batched_to_gpu_head_token_wise(self, **kwargs):
        self.slots.append(kwargs["slot_mapping"])
        for _ in kwargs["prepared_sparse_source"].layers:
            self.payloads.append((kwargs["kv_group"], (yield)))
        yield
        yield

    def stage_dense_load_tensor(self, value, dtype):
        return value.to(dtype)

    def record_dense_load_readiness(self):
        return object()

    def synchronize_dense_load_readiness(self, event):
        if self.fail_sync:
            raise RuntimeError("unknown dense completion")

    def synchronize_dense_load_stream(self):
        if self.fail_sync:
            raise RuntimeError("unknown dense completion")


def setup(definitions, rank=0, queue=None, agreement=None):
    state_cls, prepared = definitions
    importer, reply, client = fixture()
    client.client_session = f"worker-{rank}"
    client.retire = lambda *args: complete(StorageResult(reply.identity, "RETIRED"))
    calls = []

    def request(method, value, **kwargs):
        calls.append((method, value, kwargs))
        result = StorageResult(
            reply.identity, "COMPLETED", reply, payload_digest=b"digest"
        )
        future = complete(result)
        future.identity, future.payload_digest = result.identity, result.payload_digest
        return future

    client.request = request
    engine = object.__new__(LMCacheEngine)
    engine.metadata = NS(first_rank=0, worker_id=rank)
    engine.shared_cpu_cache_generation = 1
    engine.shared_cpu_cache_strict = True
    engine.storage_layout = importer.layout
    engine.compact_source_importer = importer
    engine.storage_process = NS(
        client=client, descriptor={"sessions": {"worker-0": "x", "worker-1": "y"}}
    )
    engine.config = NS(
        blocking_timeout_secs=1,
        lookup_timeout_ms=1000,
        remote_fill_native_hard_timeout_ms=1000,
    )
    engine._is_passive = lambda: rank != 0
    engine.gpu_connector = Connector()
    engine.is_healthy = lambda: True
    engine._retrieve_prepared_sparse_layers = MethodType(prepared, engine)
    engine.native_source_lifetime = NativeSourceLifetime(lambda: None, pytest.fail)
    engine._ordered_checkpoint_agreement = agreement or (lambda *args: args[-1])
    queue = Queue() if queue is None else queue
    engine.broadcast_object_fn = lambda value, first: (
        queue.put(value) if rank == 0 else queue.get(timeout=3)
    )
    adapter = NS(
        lmcache_engine=engine,
        _lmcache_chunk_size=4,
        _num_layers_for_group=lambda group: 2,
        _record_dsa_cold_dense_load_readiness=lambda state: setattr(
            state, "dense_load_readiness", object()
        ),
        _synchronize_dsa_cold_dense_load=engine.gpu_connector.synchronize_dense_load_stream,
    )
    identity = OperationIdentity(
        "cache", "epoch", "scheduler", 1, "req", "attempt", 0, "lookup"
    )
    ticket = StorageLookupTicket(
        LookupReply(
            5, "ticket", 0, b"key-proof", (0, 4), (4, 5), (b"a" * 32, b"b" * 32), {}
        ),
        identity,
        b"lookup-digest",
    )
    spec = NS(
        cpu_source_ticket=ticket,
        dsa_cold_load_generation=1,
        dsa_cold_resident_load=False,
        dsa_group1_direct_hbm=True,
    )
    request = NS(
        req_id="req", load_spec=spec, request_configs={}, slot_mapping=[torch.arange(5)]
    )
    plan = {
        "request": request,
        "tokens": list(range(5)),
        "token_count": 5,
        "token_mask": torch.ones(5, dtype=torch.bool),
        "latent_shared_ready": Future(),
        "latent_kvcaches": [object()] * 2,
        "indexer_kvcaches": [object()] * 2,
        "indexer_slots_cpu": torch.arange(5),
    }
    return adapter, plan, calls, state_cls


def test_prepare_is_tp0_only_and_keeps_scheduler_attempt_and_wide_hashes(definitions):
    first, plan, calls, _ = setup(definitions)
    worker.submit_prepare(first, plan)
    assert calls[0][2]["request_attempt"] == "attempt"
    assert calls[0][2]["load_generation"] == 1
    assert plan["chunk_plan"]()[0][2] == b"a" * 32
    passive, peer, passive_calls, _ = setup(definitions, rank=1)
    worker.submit_prepare(passive, peer)
    assert passive_calls == [] and peer["storage_prepare"] is None


def test_ready_cpu_job_does_not_publish_before_entire_predecessor(definitions):
    adapter, plan, _, _ = setup(definitions)
    worker.submit_prepare(adapter, plan)
    previous = Future()
    with ThreadPoolExecutor(1) as executor:
        job = executor.submit(
            worker.run_cold, adapter, plan, None, complete(), previous
        )
        assert not job.done()
        assert not plan["latent_shared_ready"].done()
        previous.set_result(None)
        state = job.result(3)
    worker.record_state(adapter, state, plan["request"])
    assert state.has_cache() and state.shared_request_active
    assert all(
        layer.memory_objs == () for layer in state.prepared_sparse_sources[0].layers
    )
    worker.release_sources(state, adapter.lmcache_engine)


def test_constructor_failure_is_broadcast_to_every_rank(definitions):
    queue = Queue()
    first, plan, _, _ = setup(definitions, queue=queue)
    second, peer, _, _ = setup(definitions, rank=1, queue=queue)

    def fail(*args, **kwargs):
        raise RuntimeError("prepare constructor failed")

    first.lmcache_engine.storage_process.client.request = fail
    worker.submit_prepare(first, plan)
    worker.submit_prepare(second, peer)
    for adapter, job in ((first, plan), (second, peer)):
        with pytest.raises(RuntimeError, match="prepare constructor failed"):
            worker.run_cold(adapter, job, None, complete())
        assert job["latent_shared_ready"].done()


def test_import_failure_agreement_releases_successful_peer(definitions):
    queue, barrier, votes = Queue(), Barrier(2), {}

    def agreement(rank):
        def agree(*args):
            votes[rank] = args[-1]
            barrier.wait(timeout=3)
            return all(votes.values())

        return agree

    first, plan, _, _ = setup(definitions, queue=queue, agreement=agreement(0))
    second, peer, _, _ = setup(definitions, rank=1, queue=queue, agreement=agreement(1))
    second.lmcache_engine.compact_source_importer.upload = lambda table: (
        _ for _ in ()
    ).throw(ValueError("upload failed"))
    worker.submit_prepare(first, plan)
    worker.submit_prepare(second, peer)
    with ThreadPoolExecutor(2) as executor:
        jobs = [
            executor.submit(worker.run_cold, adapter, item, None, complete())
            for adapter, item in ((first, plan), (second, peer))
        ]
        for job in jobs:
            with pytest.raises((ValueError, RuntimeError)):
                job.result(3)
    assert first.lmcache_engine.compact_source_importer.sources_drained()
    assert second.lmcache_engine.compact_source_importer.sources_drained()


@pytest.mark.parametrize("group", [0, 1])
def test_actual_ascend_prepared_generator_accepts_implicit_dense_payload(
    definitions, group
):
    adapter, plan, _, _ = setup(definitions)
    importer = adapter.lmcache_engine.compact_source_importer
    reply = fixture()[1]
    source = importer.import_reply(
        reply,
        identity=reply.identity,
        payload_digest=b"digest",
        key_plan_digest=b"key-proof",
        source_end=5,
    )[0]
    worker._dense_source(adapter, source, plan, group)
    assert adapter.lmcache_engine.gpu_connector.payloads == [(group, None)] * 2
    assert source.owner_lease.is_valid()
    source.owner_lease.release("fenced")


def test_dense_completion_unknown_retains_child_and_source(definitions):
    adapter, plan, _, _ = setup(definitions)
    importer = adapter.lmcache_engine.compact_source_importer
    reply = fixture()[1]
    source = importer.import_reply(
        reply,
        identity=reply.identity,
        payload_digest=b"digest",
        key_plan_digest=b"key-proof",
        source_end=5,
    )[0]
    adapter.lmcache_engine.gpu_connector.fail_sync = True
    with pytest.raises(RuntimeError, match="unknown dense"):
        worker._dense_source(adapter, source, plan, 0)
    source.owner_lease.release("fenced")
    assert not importer.sources_drained()
    held = adapter.lmcache_engine._failed_process_source_reads[0][0]
    assert held.is_valid()


def test_unreconciled_failure_is_published_before_every_rank_fail_stops(definitions):
    queue = Queue()
    first, plan, _, _ = setup(definitions, queue=queue)
    second, peer, _, _ = setup(definitions, rank=1, queue=queue)
    failed = []
    for rank, adapter in enumerate((first, second)):
        adapter.lmcache_engine.storage_process.mark_failed = lambda reason, rank=rank: (
            failed.append((rank, reason))
        )
    worker.submit_prepare(first, plan)
    worker.submit_prepare(second, peer)
    original = plan["storage_prepare"]
    future = Future()
    future.identity, future.payload_digest = original.identity, original.payload_digest
    future.set_exception(ConnectionError("reply lost"))
    plan["storage_prepare"] = future
    first.lmcache_engine.storage_process.client.operation_status = lambda *args: (
        _ for _ in ()
    ).throw(ConnectionError("status lost"))
    for adapter, job in ((first, plan), (second, peer)):
        with pytest.raises(RuntimeError, match="ownership could not be reconciled"):
            worker.run_cold(adapter, job, None, complete())
    assert [rank for rank, _ in failed] == [0, 1]


def test_checkpoint_miss_frontier_survives_failure_broadcast(definitions):
    queue = Queue()
    first, plan, _, _ = setup(definitions, queue=queue)
    second, peer, _, _ = setup(definitions, rank=1, queue=queue)
    worker.submit_prepare(first, plan)
    worker.submit_prepare(second, peer)
    original = plan["storage_prepare"]
    failed = StorageResult(
        original.identity,
        "FAILED_SAFE",
        {"checkpoint_miss_end": 4},
        "checkpoint source missing",
        payload_digest=original.payload_digest,
    )
    future = complete(failed)
    future.identity, future.payload_digest = original.identity, original.payload_digest
    plan["storage_prepare"] = future
    first.lmcache_engine.storage_process.client.operation_status = lambda *args: (
        complete(StorageResult(original.identity, "TERMINAL", {"leases": {}}))
    )
    for adapter, job in ((first, plan), (second, peer)):
        with pytest.raises(worker.CheckpointRestoreMiss) as caught:
            worker.run_cold(adapter, job, None, complete())
        assert caught.value.available_end == 4


def test_late_cancelled_reply_retires_without_pointer_import(definitions):
    adapter, plan, _, _ = setup(definitions)
    adapter._cold_load_coordinator = NS(aborted={"req"})
    importer = adapter.lmcache_engine.compact_source_importer
    importer.upload = lambda _: pytest.fail("cancelled source must not be imported")
    worker.submit_prepare(adapter, plan)
    with pytest.raises(RuntimeError, match="Cancelled CPU source"):
        worker.run_cold(adapter, plan, None, complete())
    assert importer.sources_drained()


def test_checkpoint_c8_tail_keeps_capacity_and_uses_tail_slots(definitions):
    adapter, plan, _, _ = setup(definitions)
    engine = adapter.lmcache_engine
    importer = engine.compact_source_importer
    reply = fixture()[1]
    c8 = GroupLayout(1, ("indexer",), ("uint8",), ((2, 2),), 1)
    layout = msgspec.structs.replace(
        importer.layout, groups=importer.layout.groups + (c8,)
    )
    importer.layout = layout
    importer.mapping.size = 512
    group0 = msgspec.structs.replace(reply.groups[0], layout_digest=layout.digest)
    tail = GroupManifest(
        1,
        1,
        layout.digest,
        b"key-proof",
        "indexer-tail",
        (4,),
        (5,),
        b"",
        b"",
        b"",
        (),
        pack_uints([256]),
        pack_uints([64]),
        pack_uints([16]),
        pack_uints([4], 32),
        source_start=4,
    )
    reply = msgspec.structs.replace(
        reply, slab_size=512, groups=(group0, tail), actions=("checkpoint_group1_tail",)
    )
    sources = importer.import_reply(
        reply,
        identity=reply.identity,
        payload_digest=b"digest",
        key_plan_digest=b"key-proof",
        source_end=5,
    )
    source = sources[1]
    assert source.total_tokens == 1 and source.chunk_token_counts == (1,)
    plan["storage_sources"] = sources
    plan["request"].load_spec.checkpoint_generation = 2
    plan["request"].load_spec.checkpoint_prefix_end = 4
    plan["indexer_kvcaches"] = [object()]
    plan["latent_shared_ready"].set_result(None)
    plan["chunk_plan"] = lambda: ((0, 4, b"a"), (4, 5, b"b"))
    prefix = []
    engine.load_group1_pages_direct = lambda tokens, slots, *args, **kwargs: (
        prefix.append((tokens, slots.tolist(), kwargs["chunk_plan"]()))
    )
    engine.finish_checkpoint_prefix = lambda request, ready: ready
    worker.run_indexer(adapter, plan, None)
    assert prefix == [([0, 1, 2, 3], [0, 1, 2, 3], ((0, 4, b"a"),))]
    assert engine.gpu_connector.slots[-1].tolist() == [4]
    assert engine.gpu_connector.payloads == [(1, None)]
    assert 1 not in sources and not source.owner_lease.is_valid()
    sources[0].owner_lease.release("fenced")


def test_shutdown_releases_foreground_and_completed_cold_states_once(definitions):
    adapter, plan, _, state_cls = setup(definitions)
    first, second = state_cls(req_id="first"), state_cls(req_id="second")
    adapter._worker_retrieve_state = {"first": first}
    failed = Future()
    failed.set_exception(RuntimeError("uncertain ownership"))
    adapter._cold_load_coordinator = NS(
        futures={
            "first": (1, complete(first)),
            "second": (1, complete(second)),
            "failed": (1, failed),
        }
    )
    released = []
    adapter._release_shared_worker_retrieve_state = lambda state, engine: (
        released.append(state)
    )
    worker.release_shutdown_states(adapter)
    assert released == [first, second]


def test_foreground_save_slot_orders_later_cold_after_older_cold_poll():
    from lmcache.integration.vllm.cold_load import ColdLoadCoordinator
    from threading import Event

    class Owner:
        def publish(self, *args):
            pass

        def fail(self, *args):
            pytest.fail("unexpected cold failure")

        def restart(self):
            return False

    owner = Owner()
    coordinator = ColdLoadCoordinator(owner.publish, owner.fail, owner.restart)
    prior = complete(NS(dense_load_readiness=NS(query=lambda: True)))
    request = NS(req_id="new", load_spec=NS(dsa_cold_load_generation=2))
    old_request = NS(load_spec=NS(dsa_cold_load_generation=1))
    coordinator.last_latent_future = prior
    coordinator.futures["old"] = (1, prior, old_request, set(), 0, complete())
    adapter = NS(_get_cold_load_coordinator=lambda: coordinator)
    assert worker.reserve_store_step(adapter) is prior
    slot = coordinator.last_latent_future
    assert worker.reserve_store_step(adapter) is prior
    assert coordinator.poll() == {"old"}
    assert coordinator.last_latent_future is slot and not slot.done()
    entered, order = Event(), []

    def latent(plan, device, sibling, predecessor):
        entered.set()
        assert predecessor is slot
        predecessor.result(3)
        order.append("later-cold")
        return NS(dense_load_readiness=None)

    with ThreadPoolExecutor(2) as executor:
        coordinator.submit_pair(
            {"request": request, "latent_shared_ready": Future()},
            2,
            set(),
            0,
            None,
            executor,
            lambda *args: None,
            latent,
        )
        assert entered.wait(2)
        assert not coordinator.futures["new"][1].done()
        order.append("foreground-save-and-promotion")
        worker.finish_store_step(adapter)
        coordinator.futures["new"][1].result(3)
    assert order == ["foreground-save-and-promotion", "later-cold"]
