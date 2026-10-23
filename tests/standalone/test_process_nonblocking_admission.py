# SPDX-License-Identifier: Apache-2.0
"""Private paired loading with actual CPU sources and mocked NPU operations."""

# Standard
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS
import gc

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.cold_load import ColdLoadCoordinator
from lmcache.integration.vllm import storage_process

# Local
import test_process_ordinary_load as ordinary
from test_process_scheduler_audit import method


@pytest.fixture
def setup():
    yield from ordinary.setup.__wrapped__()


@pytest.fixture
def prepared_definitions(monkeypatch):
    return ordinary.prepared_definitions.__wrapped__(monkeypatch)


@pytest.fixture
def harness(setup, prepared_definitions, monkeypatch):
    return ordinary.harness.__wrapped__(setup, prepared_definitions, monkeypatch)


@pytest.mark.parametrize("count", [3, 5])
@pytest.mark.parametrize("aborted", [False, True])
def test_pending_paired_preparation_leaves_foreground_free_and_state_private(
    harness,
    monkeypatch,
    count,
    aborted,
):
    adapters, ticket, runtime, native_ready, transfers = harness
    gate, ready_to_publish = Event(), Event()
    exchange = storage_process._exchange_reply

    def delayed(adapter, plan):
        assert gate.wait(10)
        return exchange(adapter, plan)

    monkeypatch.setattr(storage_process, "_exchange_reply", delayed)

    class Owner:
        def __init__(self, adapter):
            self.adapter, self.published = adapter, []

        def publish(self, req_id, entry, state, aborted, perf):
            assert not self.adapter._worker_retrieve_state
            if not aborted:
                self.adapter._set_worker_retrieve_state(req_id, state)
            else:
                storage_process.release_sources(state, self.adapter.lmcache_engine)
            self.published.append(state)

        def fail(self, *args):
            raise AssertionError("unexpected load failure") from args[3]

        def restart(self):
            return False

    enabled = gc.isenabled()
    gc.disable()
    owners = [Owner(adapter) for adapter in adapters]
    coordinators = [ColdLoadCoordinator(o.publish, o.fail, o.restart) for o in owners]
    plans = []
    reductions = []
    original_sum = torch.Tensor.sum

    def count_sum(tensor, *args, **kwargs):
        if any(tensor is plan["token_mask"] for plan in plans):
            reductions.append(id(tensor))
        return original_sum(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", count_sum)
    try:
        for adapter, coordinator in zip(adapters, coordinators, strict=True):
            adapter.lmcache_engine.gpu_connector.synchronize_dense_load_readiness = (
                lambda event: None
            )
            adapter._record_dsa_cold_dense_load_readiness = lambda state: setattr(
                state, "dense_load_readiness", NS(query=ready_to_publish.is_set)
            )
            adapter._synchronize_dsa_cold_dense_load = lambda: None
            req = ordinary.request(ticket, count)
            req.load_spec.dsa_cold_load_generation = ticket.reply.lookup_generation
            req.slot_mapping = [torch.arange(7, 7 + count)]
            plan = dict(
                request=req,
                tokens=list(range(count)),
                token_count=count,
                token_mask=torch.tensor([False] + [True] * (count - 1)),
                indexer_slots_cpu=torch.arange(17, 17 + count),
                latent_kvcaches=adapter._kvcaches_for_group(0),
                indexer_kvcaches=adapter._kvcaches_for_group(1),
                storage_mode="dense",
                storage_phase="storage_process_dense",
                storage_generation=ticket.reply.lookup_generation,
                storage_allow_safe_miss=True,
            )
            plans.append(plan)
            storage_process.submit_prepare(adapter, plan)
            coordinator.submit_ordinary(
                plan,
                ticket.reply.lookup_generation,
                {0, 1},
                0,
                None,
                lambda p, device, previous, adapter=adapter: (
                    storage_process.run_ordinary(adapter, p, device, previous)
                ),
            )
            if aborted:
                coordinator.aborted = {"request"}
        # Simulate runnable decode callbacks while the original CPU reply is held.
        decoded = 0
        for _ in range(100):
            for coordinator in coordinators:
                assert coordinator.poll() is None
            decoded += 1
        assert decoded == 100 and not transfers
        assert all(not a._worker_retrieve_state for a in adapters)
        gate.set()
        for coordinator in coordinators:
            state = coordinator.futures["request"][1].result(10)
            assert state.dense_prefix_resident_tokens == count
            assert coordinator.poll() is None  # final event still pending
        assert all(not a._worker_retrieve_state for a in adapters)
        ready_to_publish.set()
        for coordinator, owner in zip(coordinators, owners, strict=True):
            assert coordinator.poll() == {"request"}
            assert len(owner.published) == 1
            assert bool(owner.adapter._worker_retrieve_state) == (not aborted)
        assert len(transfers) == 2 * (2 + 3)
        assert all(
            slots[0] == -1 and len(slots) == count for _, _, _, slots, _ in transfers
        )
        native_ready[0] = True
        for adapter, owner in zip(adapters, owners, strict=True):
            if not aborted:
                storage_process.release_sources(
                    owner.published[0], adapter.lmcache_engine
                )
            adapter.lmcache_engine.native_source_lifetime.drain()
        assert runtime.registry.snapshot()["leases"] == 0
        assert sorted(reductions) == sorted(id(plan["token_mask"]) for plan in plans)
    finally:
        gate.set()
        for coordinator in coordinators:
            if coordinator.executor is not None:
                coordinator.executor.shutdown(wait=True)
        if enabled:
            gc.enable()


def test_predecessor_wait_remains_on_loader_and_generation_controls_publication():
    class Owner:
        def publish(self, *args):
            pytest.fail("stale load must not publish")

        def fail(self, req_id, entry, state, error, perf):
            assert "generation mismatch" in str(error)
            return True

        def restart(self):
            return False

    owner = Owner()
    coordinator = ColdLoadCoordinator(owner.publish, owner.fail, owner.restart)
    predecessor, started = Future(), Event()
    coordinator.last_latent_future = predecessor
    req = NS(req_id="r", load_spec=NS(dsa_cold_load_generation=2))

    def job(plan, device, previous):
        started.set()
        previous.result(10)
        return NS(dense_load_readiness=None)

    try:
        coordinator.submit_ordinary(dict(request=req), 1, {0}, 0, None, job)
        assert started.wait(5)
        assert coordinator.poll() is None
        predecessor.set_result(None)
        coordinator.futures["r"][1].result(5)
        assert coordinator.poll() == {"r"}
    finally:
        if not predecessor.done():
            predecessor.set_result(None)
        coordinator.executor.shutdown(wait=True)


def test_background_safe_source_miss_leaves_no_published_state_or_device_access(
    harness,
    monkeypatch,
):
    from lmcache.v1.cpu_cache import StorageFailedSafe

    adapters, ticket, runtime, _, transfers = harness

    def refused(*args, **kwargs):
        raise StorageFailedSafe("injected preparation refusal")

    monkeypatch.setattr(runtime, "prepare", refused)

    def load(adapter):
        adapter._synchronize_dsa_cold_dense_load = lambda: None
        req = ordinary.request(ticket, 5)
        plan = dict(
            request=req,
            tokens=list(range(5)),
            token_count=5,
            token_mask=torch.ones(5, dtype=torch.bool),
            storage_mode="dense",
            storage_generation=ticket.reply.lookup_generation,
            storage_allow_safe_miss=True,
        )
        storage_process.submit_prepare(adapter, plan)
        with pytest.raises(RuntimeError, match="missed safely"):
            storage_process.run_ordinary(adapter, plan, None, None)
        assert not adapter._worker_retrieve_state
        assert adapter.lmcache_engine.stats_monitor.finished == [(1, 0), (2, 0)]

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(load, adapters))
    assert not transfers and runtime.registry.snapshot()["leases"] == 0


def test_ordinary_resume_consumes_generation_without_forcing_sparse_metadata():
    spec = NS(process_ordinary_load=True, can_load=True, dsa_cold_load_generation=9)
    adapter = NS(_dsa_cold_loaded_req_ids={"r"})
    assert not method("_take_completed_cold_load")(adapter, "r", spec)
    assert spec.process_ordinary_resume and not spec.can_load
    assert not spec.process_ordinary_load
    assert not hasattr(spec, "dsa_cold_compact_resume")
    assert spec.dsa_cold_load_generation == 9


@pytest.mark.parametrize(
    "async_loading,compact", [(False, False), (True, False), (True, True)]
)
def test_lookup_keeps_async_timing_independent_from_compact_layout(
    async_loading, compact
):
    ticket = object()
    hit, total = (1024, 2048) if not compact else (129, 129)
    req = NS(
        request_id="r",
        num_tokens=total,
        prompt_token_ids=list(range(total)),
        all_token_ids=list(range(total)),
        num_preemptions=0,
        status="waiting",
    )
    adapter = NS(
        _resident_cold_load_enabled=True,
        kv_role="kv_both",
        lookup_client=NS(lookup_cache=lambda **kw: hit, peek_ticket=lambda _: ticket),
        _cold_perf_lookup_started={},
        _lmcache_chunk_size=1024,
        _block_size=128,
        _dsa_scratch_capacity=4096,
        _dsa_kv_policy_threshold=0,
        config=NS(
            decoder_storage_mode="process",
            enable_async_loading=async_loading,
            pd_role="receiver",
            min_retrieve_tokens=0,
            dsa_group1_load_mode="persistent_direct_hbm",
        ),
        enable_sparse_attention=True,
        supports_dsa_cold_compact_load=lambda: True,
        load_specs={},
        _requests_priority={},
    )
    result = method(
        "get_num_new_matched_tokens", cdiv=lambda a, b: (a + b - 1) // b, LoadSpec=NS
    )(adapter, req, 0)
    spec = adapter.load_specs["r"]
    assert result == hit - int(compact)
    assert spec.cpu_source_ticket is ticket
    assert spec.process_ordinary_load == (async_loading and not compact)
    assert getattr(spec, "dsa_cold_compact_load", False) == compact
    adapter.should_load_kv_async = lambda req_id: method("should_load_kv_async")(
        adapter, req_id
    )
    assert method("is_dsa_cold_compact_load")(adapter, "r") == compact
    assert adapter.should_load_kv_async("r") == (async_loading or compact)
    assert method("get_dsa_cold_load_num_tokens")(adapter, "r") == (
        hit if async_loading or compact else 0
    )


def test_ordinary_meta_uses_absolute_blocks_including_local_prefix():
    # A hashed local prefix is deliberately absent from the unhashed view.
    blocks = NS(
        get_block_ids=lambda: ([10, 11], [20, 21]),
        get_unhashed_block_ids_all_groups=lambda: ([11], [21]),
    )
    spec = NS(process_ordinary_load=True, lmcache_cached_tokens=5, vllm_cached_tokens=4)
    req = NS(request_id="r", all_token_ids=list(range(8)), sampling_params=None)
    adapter = NS(_block_size=4)
    build = method(
        "_build_dsa_cold_compact_meta",
        ReqMeta=NS,
        cdiv=lambda a, b: (a + b - 1) // b,
        _split_kv_group_block_ids=lambda ids: ids,
        _build_slot_mapping=lambda ids, size, n: torch.tensor(
            [ids[i // size] * size + i % size for i in range(n)]
        ),
        extract_mm_features=lambda req: (None, None),
        extract_request_configs=lambda params: None,
    )
    meta = build(adapter, req, blocks, spec)
    assert meta.slot_mapping[0].tolist() == [40, 41, 42, 43, 44]
    assert meta.indexer_slot_mapping[0].tolist() == [80, 81, 82, 83, 84]
    assert not meta.is_sparse_decode
    assert adapter._dsa_cold_indexer_block_ids["r"] == {10, 11, 20, 21}
    blocks.get_block_ids = lambda: ([10], [20, 21])
    with pytest.raises(ValueError, match="complete latent destinations"):
        build(adapter, req, blocks, spec)


def test_submission_does_not_invalidate_valid_local_prefix_on_safe_failure(monkeypatch):
    submitted = []
    coordinator = NS(
        retirements=None,
        futures={},
        submit_ordinary=lambda *args: submitted.append(args),
    )
    spec = NS(
        process_ordinary_load=True,
        dsa_cold_load_generation=3,
        lmcache_cached_tokens=5,
        vllm_cached_tokens=4,
    )
    req = NS(
        req_id="r",
        load_spec=spec,
        token_ids=list(range(8)),
        slot_mapping=[torch.tensor([40, 41, 42, 43, 44])],
        indexer_slot_mapping=[torch.tensor([80, 81, 82, 83, 84])],
    )
    adapter = NS(
        _cold_load_coordinator=coordinator,
        _block_size=4,
        _lmcache_chunk_size=1024,
        _kvcaches_for_group=lambda group: [],
        _get_dsa_cold_load_executor=lambda: None,
        _run_process_ordinary_load=lambda *args: None,
        _load_token_mask_for_retrieve=lambda *args: torch.ones(5, dtype=torch.bool),
        lmcache_engine=NS(storage_process=object()),
    )
    monkeypatch.setattr(torch, "npu", NS(current_device=lambda: 0), raising=False)
    monkeypatch.setattr(storage_process, "submit_prepare", lambda *args: None)
    method(
        "_submit_dsa_cold_compact_load",
        torch=torch,
        Future=Future,
        _cold_indexer_block_ids=method("_cold_indexer_block_ids", torch=torch),
    )(adapter, req)
    plan, _, invalidated, *_ = submitted[0]
    assert plan["token_mask"].tolist() == [False, False, False, False, True]
    assert invalidated == {11, 21}  # local blocks 10/20 remain valid
