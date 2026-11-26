# SPDX-License-Identifier: Apache-2.0
"""Real service allocations/leases with concurrent owners and a mocked NPU fence."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident
from types import ModuleType, SimpleNamespace as NS
import ctypes
import importlib
import ast
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from lmcache.integration.vllm.dcp import DcpWriteLayout
from lmcache.integration.vllm.preemption_checkpoint import CaptureSpec, SealSpec
from lmcache.v1.rpc.storage_protocol import StorageRequest, StorageResult
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator
from lmcache.v1.memory_management import LayerPageMemoryObj, MemoryFormat
from test_ascend_cpu_storage_runtime import setup as runtime_setup
from test_process_checkpoint_worker import RuntimeClient, finish, wait_drained

setup = runtime_setup


class Client(RuntimeClient):
    def __init__(self, runtime, rank):
        super().__init__(runtime)
        self.rank = rank

    def retire(self, identity, digest, lease, proof):
        return self._send(
            StorageRequest(
                identity,
                "retire",
                {"lease_id": lease, "proof": proof},
                digest,
                b"b" if self.rank else b"a",
                participant_session=f"worker-{self.rank}",
            )
        )


class Engine(NS):
    pass


@pytest.fixture
def peers(setup, monkeypatch):
    owner, runtime, _ = setup
    group_module = importlib.import_module("lmcache_ascend.v1.dcp_write_group")
    checkpoint = importlib.import_module("lmcache_ascend.v1.process_checkpoint")
    capture_path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    # Import the real CPU capture planner without bootstrapping the parent
    # package's NPU factory or requiring a generated native build manifest.
    package = ModuleType("lmcache_ascend.v1.npu_connector")
    package.__path__ = [str(capture_path.parent)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    capture_method = next(
        n
        for n in ast.walk(ast.parse(capture_path.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "prepare_dcp_group_capture"
    )
    capture_ns = dict(
        Any=Any,
        LayerPageMemoryObj=LayerPageMemoryObj,
        lmc_ops=NS(get_device_ptr=lambda pointer, size: pointer),
    )
    exec(
        compile(
            ast.Module(body=[capture_method], type_ignores=[]),
            str(capture_path),
            "exec",
        ),
        capture_ns,
    )
    capture_owner = NS(
        _expected_memory_format=lambda group: MemoryFormat(
            owner.layout.groups[group].memory_format
        ),
        _expected_group_layers=runtime.num_layers_for_group,
    )
    base = importlib.import_module("lmcache_ascend.v1.preemption_checkpoint")
    tensor = torch.tensor
    monkeypatch.setattr(
        base,
        "torch",
        NS(
            long=torch.long,
            tensor=lambda data, **kw: tensor(data, dtype=kw["dtype"]),
            empty=lambda size, **kw: torch.empty(size, dtype=kw["dtype"]),
        ),
    )
    barrier, packets = Barrier(2, timeout=3), [None, None]

    def all_gather(out, value, *, group):
        packets[group] = value
        barrier.wait()
        out[:] = packets
        barrier.wait()

    monkeypatch.setattr(group_module.dist, "all_gather_object", all_gather)
    workers, engines, cache_sets = [], [], []
    failures = NS(prepare_rank=None, fence_rank=None, stopped=[])
    layout = DcpWriteLayout(2, 4, (1, 2, 3))
    for rank in range(2):
        group = group_module.DcpWriteGroup.__new__(group_module.DcpWriteGroup)
        group.rank, group.world_size, group.sessions = rank, 2, ("worker-0", "worker-1")
        group.sequence, group.failed, group.group = 0, False, rank
        mapping = NS(
            shm_name=owner.allocator.shm_name,
            size=owner.allocator.buffer.numel(),
            generation=1,
        )
        client = Client(runtime, rank)
        context = NS(
            client=client,
            mapping=mapping,
            drain_guards=[],
            timeout_s=1,
            mark_failed=failures.stopped.append,
        )
        caches = [
            (torch.zeros((4, 4, 1, width), dtype=torch.bfloat16) for width in (512, 64))
            for _ in range(2)
        ]
        caches = [tuple(row) for row in caches]
        for token, slot, count in layout.runs(0, 9, rank):
            for layer, row in enumerate(caches):
                for plane, buf in enumerate(row):
                    buf.view(-1, buf.shape[-1])[slot : slot + count] = (
                        torch.arange(token, token + count) + layer * 10 + plane * 30
                    )[:, None].bfloat16()

        def prepare(pages, starts, ends, layout, owner_rank, *, kvcaches, **kw):
            if failures.prepare_rank == owner_rank:
                raise ValueError("injected shard preparation failure")
            return capture_ns["prepare_dcp_group_capture"](
                capture_owner,
                pages,
                starts,
                ends,
                layout,
                owner_rank,
                kvcaches=kvcaches,
                **kw,
            )

        def enqueue(plan, rank=rank):
            for dst, src, size in plan.copies:
                ctypes.memmove(dst, src, size)
            return NS(synchronize=lambda: None)

        def fence(rank=rank):
            if failures.fence_rank == rank:
                raise RuntimeError("injected uncertain native completion")

        def enqueue_index(plan):
            for layer, row in enumerate(plan):
                for page in row:
                    page.layer_tensor(layer).fill_(layer + 40)
            return NS(synchronize=lambda: None)

        engine = Engine(
            config=owner.config,
            metadata=NS(is_first_rank=lambda rank=rank: rank == 0),
            storage_process=context,
            storage_layout=owner.layout,
            dcp_write_group=group,
            is_frozen=lambda: False,
            num_layers_for_group=runtime.num_layers_for_group,
            compact_source_importer=NS(release_restore=lambda *a: None),
            gpu_connector=NS(
                prepare_dcp_group_capture=prepare,
                enqueue_dcp_group_capture=enqueue,
                prepare_group_capture=lambda rows, *a, **kw: rows,
                enqueue_group_capture=enqueue_index,
                finish_checkpoint_capture=fence,
            ),
            shared_cpu_cache_passive_allocator=PassiveSharedViewAllocator(
                slab_tensor=owner.allocator.buffer,
                shm_name=mapping.shm_name,
                generation=1,
            ),
        )
        engine.token_database = runtime.token_database
        engine.store_location = None
        engine.kv_events_enabled = False
        instance = checkpoint.ProcessCheckpointWorker(engine)
        instance.timeout = 1
        workers.append(instance)
        engines.append(engine)
        cache_sets.append({0: caches, 1: [None] * 3})

    def run(action):
        def call(rank):
            engines[rank].dcp_write_group.thread = get_ident()
            return action(rank)

        with ThreadPoolExecutor(2) as pool:
            results = [pool.submit(call, rank) for rank in range(2)]
            return [f.result(5) for f in results]

    spec = CaptureSpec(
        "request", 1, 0, 9, 0, ((1, 2, 3), (1, 2, 3)), dcp_write_layout=layout
    )
    yield NS(
        workers=workers,
        engines=engines,
        caches=cache_sets,
        run=run,
        spec=spec,
        failures=failures,
        owner=owner,
        runtime=runtime,
    )
    for instance in workers:
        if failures.stopped:
            instance.executor.shutdown(wait=True)
            # Test-only: all mock DMA is synchronous and worker threads have
            # joined. Drop passive views after asserting production retained
            # them; the runtime fixture separately disposes its CPU owner holds.
            for state in instance.reservations.values():
                for fragments in state.job.fragments.values():
                    for _, _, page in fragments:
                        page.ref_count_down()
                state.job.fragments.clear()
                state.job.plans.clear()
        else:
            instance.close()
    if failures.stopped:
        for engine in engines:
            for state in getattr(engine, "_process_store_writes", {}).values():
                for view in state.views:
                    view.ref_count_down()
                state.views.clear()


def test_cooperative_checkpoint_canonical_bytes_and_accepted_seal(peers):
    p = peers
    p.run(lambda rank: p.workers[rank].capture(p.spec, p.caches[rank], 4))
    leader, peer = p.workers
    assert [(r.status, r.end) for r in leader.poll()] == [("captured", 9)]
    assert not peer.poll() and peer.sources_drained()
    for start, end, page in leader.jobs["request", 1].fragments[0]:
        for layer in range(2):
            raw = page.layer_tensor(layer).view(-1)
            capacity = raw.numel() // 576
            for width, offset, value in ((512, 0, 0), (64, capacity * 512, 30)):
                actual = raw[offset : offset + (end - start) * width].view(-1, width)
                expected = torch.arange(start, end) + layer * 10 + value
                assert torch.equal(
                    actual, expected[:, None].expand_as(actual).bfloat16()
                )
    leader.seal(SealSpec("request", 1, tuple(range(7))))
    assert [(r.status, r.end) for r in finish(leader)] == [("ready", 7)]
    wait_drained(leader)
    assert p.runtime.registry.snapshot()["leases"] == 0


def test_preparation_failure_is_shared_and_releases_without_publication(peers):
    p = peers
    p.failures.prepare_rank = 1
    p.run(lambda rank: p.workers[rank].capture(p.spec, p.caches[rank], 4))
    assert p.workers[0].poll()[0].status == "failed"
    wait_drained(p.workers[0])
    assert not p.failures.stopped
    assert p.runtime.registry.snapshot()["leases"] == 0


def test_busy_checkpoint_reservation_is_a_shared_safe_miss(peers, monkeypatch):
    from concurrent.futures import Future

    p = peers
    client = p.engines[0].storage_process.client
    send = client._send

    def busy(request):
        if request.method != "reserve_write":
            return send(request)
        future = Future()
        future.identity, future.payload_digest = (
            request.identity,
            request.payload_digest,
        )
        future.set_result(StorageResult(request.identity, "BUSY"))
        return future

    monkeypatch.setattr(client, "_send", busy)
    p.run(lambda rank: p.workers[rank].capture(p.spec, p.caches[rank], 4))
    assert p.workers[0].poll()[0].status == "failed"
    assert all(w.sources_drained() for w in p.workers)
    assert not p.failures.stopped and p.runtime.registry.snapshot()["leases"] == 0


def test_checkpoint_rejected_prefix_proof_before_reservation_is_a_safe_miss(peers):
    p = peers

    def reject(source):
        raise ValueError("Source descriptor no longer belongs to this importer")

    p.engines[0].compact_source_importer.source_descriptor = reject
    prefix = NS(prepared_sparse_sources={0: object()})
    p.run(lambda rank: p.workers[rank].capture(p.spec, p.caches[rank], 4, prefix))
    result = p.workers[0].poll()[0]
    assert result.status == "failed" and "Source descriptor" in result.reason
    assert all(w.sources_drained() for w in p.workers)
    assert not p.failures.stopped
    assert all(not e.storage_process.client.controls for e in p.engines)
    assert p.runtime.registry.snapshot()["leases"] == 0


def test_duplicate_capture_does_not_allocate_or_copy_again(peers):
    p = peers
    for _ in range(2):
        p.run(lambda rank: p.workers[rank].capture(p.spec, p.caches[rank], 4))
        assert p.workers[0].poll()[0].status == "captured"
    methods = [m for m, _ in p.engines[0].storage_process.client.controls]
    assert methods.count("reserve_write") == 1


def test_cancel_during_capture_cannot_release_peer_destinations(peers, monkeypatch):
    from threading import Event

    p = peers
    entered, release = Event(), Event()
    connector = p.engines[1].gpu_connector
    prepare = connector.prepare_dcp_group_capture

    def pause(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(connector, "prepare_dcp_group_capture", pause)
    with ThreadPoolExecutor(1) as pool:
        run = pool.submit(
            p.run, lambda rank: p.workers[rank].capture(p.spec, p.caches[rank], 4)
        )
        try:
            assert entered.wait(3)
            p.workers[0].cancel("request", 1)
            assert p.runtime.registry.snapshot()["leases"] == 2
            assert p.workers[0].reservations["request", 1].cooperative_pending
        finally:
            release.set()
        run.result(5)
    assert p.workers[0].poll()[0].status == "failed"
    wait_drained(p.workers[0])
    assert p.runtime.registry.snapshot()["leases"] == 0


def test_uncertain_peer_fence_retains_all_owners_and_fails_both(peers):
    p = peers
    p.failures.fence_rank = 1

    def call(rank):
        with pytest.raises(RuntimeError, match="unknown native completion"):
            p.workers[rank].capture(p.spec, p.caches[rank], 4)

    p.run(call)
    assert len(p.failures.stopped) == 2
    assert all(w.jobs["request", 1].quarantined for w in p.workers)
    assert p.runtime.registry.snapshot()["leases"] == 2


@pytest.mark.parametrize("group", [0, 1])
def test_cooperative_ordinary_store_commits_once_and_releases_all_participants(
    peers, group
):
    p = peers
    module = importlib.import_module("lmcache_ascend.v1.dcp_store")
    request = NS(
        req_id="request",
        token_ids=list(range(9)),
        dcp_write_layout=p.spec.dcp_write_layout,
    )
    options = {"slot_mapping": torch.arange(9), "slot_mapping_base": 0}
    results = p.run(
        lambda rank: module.store_group(
            p.engines[rank],
            request,
            group,
            p.caches[rank][group],
            torch.ones(9, dtype=torch.bool),
            options,
        )
    )
    assert [r.committed_end for r in results] == [9, 9]
    assert results[0].process_store_plan == results[1].process_store_plan
    assert all(not e._process_store_writes for e in p.engines)
    assert p.runtime.registry.snapshot()["leases"] == 0
    methods = [name for name, _ in p.engines[0].storage_process.client.controls]
    assert methods.count("reserve_write") == methods.count("commit_write") == 1
    assert not p.engines[1].storage_process.client.controls


def test_store_peer_preparation_failure_is_drained_on_both_ranks(peers):
    p = peers
    module = importlib.import_module("lmcache_ascend.v1.dcp_store")
    p.failures.prepare_rank = 1
    request = NS(
        req_id="request",
        token_ids=list(range(9)),
        dcp_write_layout=p.spec.dcp_write_layout,
    )

    def run(rank):
        with pytest.raises(RuntimeError, match="capture failed"):
            module.store_group(
                p.engines[rank],
                request,
                0,
                p.caches[rank][0],
                torch.ones(9, dtype=torch.bool),
                {},
            )

    p.run(run)
    assert not p.failures.stopped
    assert all(not e._process_store_writes for e in p.engines)
    assert p.runtime.registry.snapshot()["leases"] == 0


@pytest.mark.parametrize("partial", [False, True])
def test_store_layer_fallback_preserves_partial_capacity_and_gc_off_ownership(
    peers, monkeypatch, partial
):
    import gc
    import weakref

    p = peers
    module = importlib.import_module("lmcache_ascend.v1.dcp_store")
    monkeypatch.setattr(
        p.runtime.storage_manager, "supports_batched_put_layer_pages", lambda *a: False
    )
    allocate = p.runtime.storage_manager.allocate
    allocations = []

    def limited(*args, **kwargs):
        if partial and len(allocations) == 4:
            return None  # Exactly two two-layer chunks fit.
        result = allocate(*args, **kwargs)
        allocations.append(result)
        return result

    monkeypatch.setattr(p.runtime.storage_manager, "allocate", limited)
    reserve, adopt = module.process_store._reserve, module._adopt
    references = []

    def observed(action):
        def call(*args, **kwargs):
            state = action(*args, **kwargs)
            if state is not None:
                references.append(weakref.ref(state))
            return state

        return call

    monkeypatch.setattr(module.process_store, "_reserve", observed(reserve))
    monkeypatch.setattr(module, "_adopt", observed(adopt))
    request = NS(
        req_id="request",
        token_ids=list(range(9)),
        dcp_write_layout=p.spec.dcp_write_layout,
    )
    source = NS(total_tokens=3)
    rows = tuple(
        (
            torch.cat(
                (
                    torch.full((3 * 512,), 100 + layer, dtype=torch.bfloat16),
                    torch.full((3 * 64,), 200 + layer, dtype=torch.bfloat16),
                )
            ),
        )
        for layer in range(2)
    )
    released = []
    importer = p.engines[0].compact_source_importer
    importer.source_manifest = lambda _: NS(
        group_id=0, chunk_starts=(0,), chunk_ends=(3,)
    )
    importer.borrow_source_tensors = lambda _: (rows, NS(release=released.append))
    enabled = gc.isenabled()
    gc.disable()
    try:
        results = p.run(
            lambda rank: module.store_group(
                p.engines[rank],
                request,
                0,
                p.caches[rank][0],
                torch.ones(9, dtype=torch.bool),
                {"dcp_prefix_source": source},
            )
        )
        end = 8 if partial else 9
        assert [r.committed_end for r in results] == [end, end]
        assert results[0].ends[-1] == end
        assert all(ref() is None for ref in references)
        assert all(not e._process_store_writes for e in p.engines)
        assert p.runtime.registry.snapshot()["leases"] == 0
        assert released == ["never_imported"]
        for layer, obj in enumerate(allocations[:2]):
            raw = obj.tensor.view(-1)
            assert (raw[: 3 * 512] == 100 + layer).all()
            assert (raw[4 * 512 : 4 * 512 + 3 * 64] == 200 + layer).all()
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("phase", ["submission", "publication"])
def test_store_failure_after_capture_admission_drains_before_release(
    peers, monkeypatch, phase
):
    p = peers
    module = importlib.import_module("lmcache_ascend.v1.dcp_store")
    before = p.owner.allocator.address_manager.get_free_size()
    if phase == "submission":
        enqueue = p.engines[1].gpu_connector.enqueue_dcp_group_capture

        def fail(plan):
            enqueue(plan)  # Earlier copies may already have been enqueued.
            raise RuntimeError("injected partial native submission")

        monkeypatch.setattr(
            p.engines[1].gpu_connector, "enqueue_dcp_group_capture", fail
        )
    else:
        from lmcache.v1.cpu_cache import StorageFailedSafe

        def fail(reservation):
            raise StorageFailedSafe("injected failure before persistent submission")

        monkeypatch.setattr(p.runtime, "_commit_store", fail)
    request = NS(
        req_id="request",
        token_ids=list(range(9)),
        dcp_write_layout=p.spec.dcp_write_layout,
    )

    def run(rank):
        with pytest.raises(RuntimeError):
            module.store_group(
                p.engines[rank],
                request,
                0,
                p.caches[rank][0],
                torch.ones(9, dtype=torch.bool),
                {},
            )

    p.run(run)
    assert not p.failures.stopped
    assert all(not e._process_store_writes for e in p.engines)
    assert p.runtime.registry.snapshot()["leases"] == 0
    assert p.owner.allocator.address_manager.get_free_size() == before


def test_unknown_publication_keeps_destination_ownership(peers, monkeypatch):
    p = peers
    module = importlib.import_module("lmcache_ascend.v1.dcp_store")

    def fail(reservation):
        # An arbitrary exception cannot establish whether persistence submitted
        # native reads. Unlike StorageFailedSafe, the service quarantines it.
        raise ValueError("unknown persistent submission outcome")

    monkeypatch.setattr(p.runtime, "_commit_store", fail)
    request = NS(
        req_id="request",
        token_ids=list(range(9)),
        dcp_write_layout=p.spec.dcp_write_layout,
    )

    def run(rank):
        with pytest.raises(RuntimeError):
            module.store_group(
                p.engines[rank],
                request,
                0,
                p.caches[rank][0],
                torch.ones(9, dtype=torch.bool),
                {},
            )

    p.run(run)
    assert p.failures.stopped
    state = next(iter(p.engines[0]._process_store_writes.values()))
    assert state.unknown and state.views and state.holds
    assert p.runtime.registry.snapshot()["leases"] > 0
