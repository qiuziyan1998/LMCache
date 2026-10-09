# SPDX-License-Identifier: Apache-2.0
"""Execute adapter admission/background methods with CPU transfer collaborators."""

from collections.abc import Iterator
from concurrent.futures import Future
import gc
from types import SimpleNamespace as NS

import pytest
import torch
from test_checkpoint_adapter import method


@pytest.mark.parametrize("length", [2, 127, 128, 129, 280, 1025, 4095, 4096, 4097])
@pytest.mark.parametrize("role", ["sender", "receiver"])
@pytest.mark.parametrize("capable", [False, True])
def test_short_full_hit_uses_resident_cold_admission(length, role, capable):
    request = NS(
        request_id="r",
        num_tokens=length,
        prompt_token_ids=[1] * length,
        all_token_ids=[1] * length,
        num_preemptions=0,
        status="waiting",
    )
    adapter = NS(
        _resident_cold_load_enabled=capable,
        kv_role="kv_both",
        lookup_client=NS(lookup_cache=lambda **kw: length),
        _cold_perf_lookup_started={},
        _lmcache_chunk_size=1024,
        _block_size=128,
        _dsa_scratch_capacity=4096,
        _dsa_kv_policy_threshold=0,
        config=NS(
            pd_role=role, min_retrieve_tokens=0, dsa_group1_load_mode="p2p_preferred"
        ),
        enable_sparse_attention=True,
        supports_dsa_cold_compact_load=lambda: True,
        load_specs={},
        _requests_priority={},
    )
    matched = method(
        "get_num_new_matched_tokens", cdiv=lambda a, b: (a + b - 1) // b, LoadSpec=NS
    )(adapter, request, 0)
    spec = adapter.load_specs["r"]
    assert matched == length - 1
    resident = capable and role == "receiver" and length <= 4096
    assert getattr(spec, "dsa_cold_resident_load", False) == resident
    if resident:
        assert spec.dsa_cold_compact_load
        assert spec.dsa_committed_end == length
        assert spec.dsa_remap_frontier == spec.dsa_release_frontier == 0
        assert spec.dsa_cold_resume_computed_end == length - 1
    elif length <= 4096:
        assert not getattr(spec, "dsa_cold_compact_load", False)


@pytest.fixture
def no_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("aborted", [False, True])
def test_prepared_pointer_publication_waits_for_final_device_event(
    aborted, no_cyclic_gc
):
    from lmcache.integration.vllm.cold_load import ColdLoadCoordinator

    actions = []
    ready = [False]
    state = NS(dense_load_readiness=NS(query=lambda: ready[0]))

    class Owner:
        def publish(self, req_id, entry, result, was_aborted, perf):
            assert ready[0] and result is state
            actions.append("retire" if was_aborted else "publish")

        def fail(self, *args):
            pytest.fail("Unexpected cold-load failure")

        def requires_restart(self):
            return False

    owner = Owner()
    coordinator = ColdLoadCoordinator(owner.publish, owner.fail, owner.requires_restart)
    latent, indexer = Future(), Future()
    latent.set_result(state)
    indexer.set_result(None)
    request = NS(load_spec=NS(dsa_cold_load_generation=7))
    coordinator.futures["r"] = (7, latent, request, {12}, 0.0, indexer)
    coordinator.aborted = {"r"} if aborted else None
    assert coordinator.poll() is None
    assert coordinator.futures and not actions
    ready[0] = True
    assert coordinator.poll() == {"r"}
    assert actions == ["retire" if aborted else "publish"]
    assert not coordinator.futures


@pytest.mark.parametrize(
    "failure", [None, "incomplete", "missing_event", "submit", "unknown_completion",
                "final_record", "final_record_unknown"]
)
def test_resident_worker_loads_before_publication_and_unblocks_failures(
    failure, no_cyclic_gc
):
    trace = []
    length = 280
    spec = NS(
        lmcache_cached_tokens=length,
        dsa_cold_resident_load=True,
        dsa_group1_direct_hbm=failure not in ("final_record", "final_record_unknown"),
    )
    slots = torch.arange(length)
    request = NS(req_id="r", load_spec=spec, slot_mapping=[slots])
    state = NS(
        req_id="r",
        dense_load_source_owners=(),
        dense_load_readiness=None,
        prepared_sparse_sources={},
        has_cache=lambda: True,
    )
    readiness = object()

    def dense(tokens, mask, **kwargs):
        assert kwargs["kv_group"] == 0 and kwargs["slot_mapping"] is slots
        assert kwargs["_retain_shared_dense_cache"]
        assert "materialize_only" not in kwargs
        try:
            trace.append("load")
            yield None
            if failure in ("submit", "unknown_completion"):
                raise RuntimeError("injected submit failure")
            yield None
            yield None
            if failure != "missing_event":
                kwargs["_dense_load_readiness_out"].append(readiness)
            yield torch.ones(length - int(failure == "incomplete"), dtype=torch.bool)
        finally:
            trace.append("close")

    connector = NS(
        validate_layerwise_slot_mapping=lambda *a, **kw: trace.append("validate"),
        stage_dense_load_tensor=lambda tensor, **kw: tensor,
    )
    engine = NS(gpu_connector=connector, retrieve_layer=dense)
    gate = Future()
    indexer = Future()
    indexer.set_result((None, None if spec.dsa_group1_direct_hbm else object(), 0, 0))
    plan = dict(
        request=request,
        token_count=length,
        tokens=list(range(length)),
        token_mask=torch.ones(length, dtype=torch.bool),
        latent_kvcaches=[0, 1],
        latent_shared_ready=gate,
        planned_at=0,
        plan_started=0,
    )

    def seal(state, count):
        assert gate.done() and state.dense_prefix_resident_tokens == count
        state.prepared_sparse_sources[0] = object()
        trace.append("seal")

    def fence():
        trace.append("fence")
        if failure in ("unknown_completion", "final_record_unknown"):
            raise RuntimeError("completion unknown")

    def record(*args, **kwargs):
        trace.append("record")
        if failure in ("final_record", "final_record_unknown"):
            raise RuntimeError("final fence failed after graph pointer submission")

    adapter = NS(
        lmcache_engine=engine,
        _num_layers_for_group=lambda group: 2 if group == 0 else 1,
        _sparse_retrieve_kwargs=lambda *a, **kw: (dict(kv_group=0), None, None),
        _synchronize_dsa_cold_dense_readiness=lambda event: trace.append("ready"),
        _record_dsa_cold_dense_load_readiness=record,
        _refresh_prepared_sparse_sources=seal,
        _synchronize_dsa_cold_dense_load=fence,
        _release_dense_load_source_owners=lambda *a, **kw: trace.append("release"),
        _release_unadopted_shared_request_objects=lambda *a: None,
        _release_shared_worker_retrieve_state=lambda *a: None,
    )
    run = method(
        "_run_dsa_cold_compact_load",
        torch=torch,
        WorkerRetrieveState=lambda **kw: state,
        logger=NS(exception=lambda *a: None),
    )
    if failure:
        with pytest.raises(RuntimeError) as error:
            run(adapter, plan, None, indexer)
        assert gate.done()
        after_seal = failure in ("final_record", "final_record_unknown")
        assert (gate.exception() is None) == after_seal
        assert ("seal" in trace) == after_seal
        if failure in ("unknown_completion", "final_record_unknown"):
            assert "release" not in trace
            assert error.value._lmcache_dsa_cold_state is state
        else:
            assert trace.index("fence") < trace.index("release")
    else:
        result = run(adapter, plan, None, indexer)
        assert result is state and gate.result() is None
        assert trace == ["load", "close", "ready", "seal", "record"]
        assert state.indexer_npu_resident and state.token_count == length
        assert state.metadata_token_ids is plan["tokens"]


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("length", [1, 129, 84454])
def test_cold_publication_adopts_snapshot_without_copy(
    reuse: bool, length: int
) -> None:
    class Tokens(list):
        copied = 0

        def __getitem__(self, key: int | slice) -> int | list[int]:
            value = super().__getitem__(key)
            if isinstance(key, slice):
                self.copied += len(value)
            return value

    history = Tokens(range(length + 1))
    snapshot = history[:length]
    history.copied = 0
    state = NS(
        token_count=length, location="LocalCPU", metadata_warm=True,
        prepared_sparse_sources={0: NS(total_tokens=length)},
        metadata_token_ids=snapshot if reuse else [],
    )
    request = NS(req_id="r", token_ids=history, sparse_warm_ref=False)
    adapter = NS(
        _worker_retrieve_state={},
        _refresh_prepared_sparse_sources=lambda *a: None,
        _record_shared_worker_retrieve_state=lambda *a: None,
        _set_worker_retrieve_state=lambda *a: None,
    )
    method("_publish_worker_retrieve_state")(
        adapter, state, request, location=None, metadata_warm=True,
        token_count=length, reuse_prepared_sources=reuse,
    )
    assert history.copied == (0 if reuse else length)
    assert (state.metadata_token_ids is snapshot) == reuse
    history[0] = -1
    assert state.metadata_token_ids == list(range(length))


@pytest.mark.parametrize("live", [False, True])
def test_compact_and_live_workers_publish_their_owned_snapshot(
    live: bool, no_cyclic_gc: None
) -> None:
    length = 84454
    history = list(range(length))
    request = NS(
        req_id="r", token_ids=history, sparse_warm_ref=False,
        load_spec=NS(dsa_group1_direct_hbm=True),
    )
    plan = dict(
        request=request, tokens=history[:], token_count=length,
        token_mask=torch.ones(length, dtype=torch.bool),
        latent_kvcaches=[], latent_shared_ready=Future(),
    )
    state = NS(prepared_sparse_sources={}, has_cache=lambda: True)
    calls = []

    def retrieve(
        tokens: list[int], mask: torch.Tensor, **kwargs: object
    ) -> Iterator[torch.Tensor | None]:
        assert not live and tokens is plan["tokens"]
        assert kwargs["materialize_only"]
        calls.append("retrieve")
        yield None
        yield None
        yield mask

    def seal(result: NS, count: int) -> None:
        assert result.metadata_token_ids is plan["tokens"]
        result.prepared_sparse_sources[0] = NS(total_tokens=count)

    adapter = NS(
        lmcache_engine=NS(retrieve_layer_head_token_wise=retrieve),
        _num_layers_for_group=lambda group: 2,
        _sparse_retrieve_kwargs=lambda *a, **kw: ({}, None, None),
        _refresh_prepared_sparse_sources=seal,
        _record_dsa_cold_dense_load_readiness=lambda *a, **kw: calls.append("fence"),
        _worker_retrieve_state={},
        _record_shared_worker_retrieve_state=lambda *a: calls.append("adopt"),
        _set_worker_retrieve_state=lambda *a: calls.append("publish"),
    )
    indexer = Future()
    indexer.set_result((None, None, 0., 0.))
    run = method("_run_dsa_cold_compact_load", torch=torch,
                 WorkerRetrieveState=lambda **kw: state)
    result = run(adapter, plan, None, indexer, live_state=state if live else None)
    history.append(-1)
    method("_publish_worker_retrieve_state")(
        adapter, result, request, location=None, metadata_warm=True,
        token_count=length, reuse_prepared_sources=True,
    )
    assert result.metadata_token_ids is plan["tokens"]
    history[0] = -1
    assert result.metadata_token_ids[0] == 0
    assert calls == ([] if live else ["retrieve"]) + ["fence", "adopt", "publish"]


@pytest.mark.parametrize("snapshot_length", [0, 3, 5])
def test_cold_publication_rejects_wrong_snapshot_frontier(snapshot_length: int) -> None:
    state = NS(
        token_count=4, location="LocalCPU", metadata_warm=True,
        prepared_sparse_sources={0: NS(total_tokens=4)},
        metadata_token_ids=list(range(snapshot_length)),
    )
    released = []
    adapter = NS(
        _worker_retrieve_state={}, lmcache_engine=None,
        _release_unadopted_shared_request_objects=lambda *a: released.append(
            "unadopted"
        ),
        _release_shared_worker_retrieve_state=lambda *a, **kw: released.append("state"),
    )
    with pytest.raises(RuntimeError, match="token snapshot"):
        method("_publish_worker_retrieve_state")(
            adapter, state, NS(req_id="r"), location=None, metadata_warm=True,
            token_count=4, reuse_prepared_sources=True,
        )
    assert released == ["unadopted", "state"]


def test_cold_pool_reuses_threads_across_bursts_and_shutdown_drains(
    no_cyclic_gc: None,
) -> None:
    import threading
    from lmcache.integration.vllm.cold_load import ColdLoadCoordinator

    actions = []

    class Owner:
        def publish(self, *args: object) -> None:
            actions.append("publish")

        def fail(self, *args: object) -> bool:
            raise AssertionError(args[3])

        def requires_restart(self) -> bool:
            return False

    owner = Owner()
    coordinator = ColdLoadCoordinator(owner.publish, owner.fail, owner.requires_restart)
    executor = coordinator.get_executor()
    try:
        worker = executor.submit(threading.current_thread).result(timeout=5)
        for generation in range(2):
            latent, indexer = Future(), Future()
            latent.set_result(NS(dense_load_readiness=NS(query=lambda: True)))
            indexer.set_result(None)
            coordinator.last_latent_future = latent
            request = NS(load_spec=NS(dsa_cold_load_generation=generation))
            coordinator.futures["r"] = (generation, latent, request, set(), 0., indexer)
            assert coordinator.poll() == {"r"}
            assert coordinator.last_latent_future is None
            assert coordinator.get_executor() is executor
            assert executor.submit(threading.current_thread).result(timeout=5) is worker
        adapter = NS(
            _dsa_kv_policy_states={}, _cold_load_coordinator=coordinator,
            _synchronize_dsa_cold_dense_load=lambda: actions.append("sync"),
            _drain_dense_load_retirements=lambda **kw: actions.append("retire"),
            _manager=NS(stop_services=lambda: actions.append("stop")),
        )
        method("shutdown", logger=NS(info=lambda *a: None))(adapter)
        assert not worker.is_alive()
        assert actions == ["publish", "publish", "sync", "retire", "stop"]
    finally:
        executor.shutdown(wait=True)


def test_retained_pool_drops_completed_request_data_without_gc(
    no_cyclic_gc: None,
) -> None:
    import threading
    import weakref
    from lmcache.integration.vllm.cold_load import ColdLoadCoordinator

    class Payload:
        pass

    class Owner:
        def publish(self, *args: object) -> None:
            pass

        def fail(self, *args: object) -> bool:
            raise AssertionError(args[3])

        def requires_restart(self) -> bool:
            return False

    def indexer(plan: dict, device: object) -> tuple:
        plan["latent_shared_ready"].result(timeout=5)
        return None, None, 0., 0.

    def latent(
        plan: dict, device: object, sibling: Future, previous: Future | None
    ) -> NS:
        assert previous is None
        plan["latent_shared_ready"].set_result(None)
        sibling.result(timeout=5)
        return NS(dense_load_readiness=None, payload=plan["payload"])

    owner = Owner()
    coordinator = ColdLoadCoordinator(owner.publish, owner.fail, owner.requires_restart)
    executor = coordinator.get_executor()

    def burst(generation: int) -> weakref.ReferenceType:
        payload = Payload()
        request = NS(req_id="r", load_spec=NS(dsa_cold_load_generation=generation))
        plan = dict(request=request, payload=payload, latent_shared_ready=Future())
        coordinator.submit_pair(
            plan, generation, set(), 0., None, executor, indexer, latent
        )
        coordinator.futures["r"][1].result(timeout=5)
        assert coordinator.poll() == {"r"}
        assert coordinator.executor is executor
        return weakref.ref(payload)

    try:
        references = [burst(1), burst(2)]
        # Occupy both workers to prove their previous work items were retired.
        barrier = threading.Barrier(3)
        jobs = [executor.submit(barrier.wait, 5) for _ in range(2)]
        barrier.wait(timeout=5)
        for job in jobs:
            job.result(timeout=5)
        assert all(reference() is None for reference in references)
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("resident_tokens", [0, 279, 280])
def test_completed_resident_restore_requires_both_groups(resident_tokens):
    check = method("completed_cold_resume_state")
    spec = NS(
        dsa_cold_compact_resume=True,
        dsa_cold_resident_load=True,
        dsa_cold_load_generation=3,
        lmcache_cached_tokens=280,
    )
    state = NS(
        completed_cold_load_generation=3,
        token_count=280,
        indexer_npu_resident=True,
        prepared_sparse_sources={0: object()},
        dense_prefix_resident_tokens=resident_tokens,
    )
    assert check(NS(load_spec=spec), state) == (resident_tokens == 280)
    state.indexer_npu_resident = False
    assert not check(NS(load_spec=spec), state)


@pytest.mark.parametrize("length", [129, 280, 4096])
def test_resident_metadata_covers_full_hit_in_both_groups(length):
    cdiv = lambda a, b: (a + b - 1) // b
    mapping = method("_build_slot_mapping", torch=torch, utils=NS(cdiv=cdiv))
    build = method(
        "_build_dsa_cold_compact_meta",
        cdiv=cdiv,
        ReqMeta=NS,
        _split_kv_group_block_ids=lambda blocks: blocks,
        _build_slot_mapping=mapping,
        _live_split_source_dp_rank=lambda *a: None,
    )
    adapter = NS(
        _block_size=128,
        _vllm_config=NS(parallel_config=NS(tensor_parallel_size=4)),
        _group1_p2p_preferred=lambda: False,
    )
    request = NS(
        request_id="r", all_token_ids=list(range(length)), sampling_params=None
    )
    spec = NS(
        lmcache_cached_tokens=length,
        dsa_cold_resident_load=True,
        dsa_committed_end=length,
        dsa_remap_frontier=0,
    )
    latent = list(range(3, 3 + cdiv(length, 128)))
    indexer = list(range(50, 50 + cdiv(length, 128)))
    meta = build(adapter, request, (latent, indexer), spec)
    assert len(meta.slot_mapping[0]) == len(meta.indexer_slot_mapping[0]) == length
    assert meta.dsa_nonresident_frontier == 0
    assert (meta.slot_mapping[0] // 128).unique().tolist() == latent
    with pytest.raises(ValueError, match="complete latent"):
        build(adapter, request, (latent[:-1], indexer), spec)


def test_resident_slot_setup_failure_reaches_shared_error_envelope():
    """Use real rank-0 priming so peers receive failure before waiting for KV."""
    import ast
    from pathlib import Path
    from types import MethodType

    events = []
    path = Path(__file__).resolve().parents[2] / "lmcache/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_retrieve_layer_shared_rank0"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    ns = dict(assert_layerwise_gpu_connector=lambda _: None)
    exec(compile(module, str(path), "exec"), ns)

    def setup_failure(*args, **kwargs):
        raise RuntimeError("slot allocation failed")

    connector = NS(
        validate_layerwise_slot_mapping=lambda *a, **kw: None,
        stage_dense_load_tensor=setup_failure,
        batched_to_gpu=setup_failure,
    )
    engine = NS(
        storage_manager=object(),
        gpu_connector=connector,
        _remote_fill_pair_lookup_enabled=lambda: False,
        _shared_layerwise_error_envelope=lambda **kw: kw,
        _broadcast_shared_envelope=events.append,
    )
    rank0 = MethodType(ns[node.name], engine)
    engine.retrieve_layer = lambda tokens, mask, **kwargs: rank0(
        starts=[0],
        ends=[280],
        keys_layer_major=[[object()]],
        chunk_locations_layer_major=[["LocalCPUBackend"]],
        location="LocalCPUBackend",
        ret_mask=mask,
        monitor_req_id=0,
        req_id="r",
        kv_group=0,
        kwargs=kwargs,
    )
    state = NS(req_id="r", dense_load_source_owners=(), dense_load_readiness=None)
    request = NS(
        req_id="r",
        slot_mapping=[torch.arange(280)],
        load_spec=NS(dsa_cold_resident_load=True, dsa_group1_direct_hbm=True),
    )
    gate, indexer = Future(), Future()
    indexer.set_result((None, None, 0, 0))
    plan = dict(
        request=request,
        token_count=280,
        tokens=list(range(280)),
        token_mask=torch.ones(280, dtype=torch.bool),
        latent_kvcaches=[object()],
        latent_shared_ready=gate,
    )
    adapter = NS(
        lmcache_engine=engine,
        _num_layers_for_group=lambda _: 1,
        _sparse_retrieve_kwargs=lambda *a, **kw: ({}, None, None),
        _synchronize_dsa_cold_dense_load=lambda: None,
        _release_dense_load_source_owners=lambda *a, **kw: None,
        _release_unadopted_shared_request_objects=lambda *a: None,
        _release_shared_worker_retrieve_state=lambda *a: None,
    )
    run = method(
        "_run_dsa_cold_compact_load",
        torch=torch,
        WorkerRetrieveState=lambda **kw: state,
    )
    with pytest.raises(RuntimeError, match="slot allocation failed"):
        run(adapter, plan, None, indexer)
    assert gate.exception() is not None
    assert len(events) == 1
    assert events[0]["req_id"] == "r" and events[0]["layer_id"] == 0
    assert "consumer preparation failed" in events[0]["message"]


def test_direct_indexer_does_not_allow_failed_resident_dma_retirement():
    released = []
    state = NS(req_id="r")
    error = RuntimeError("resident DMA completion unknown")
    error._lmcache_dsa_cold_state = state
    request = NS(load_spec=NS(dsa_group1_direct_hbm=True, dsa_cold_resident_load=True))
    entry = (1, Future(), request, {3}, 0, Future())

    def unproven():
        raise RuntimeError("cannot prove stream completion")

    adapter = NS(
        lmcache_engine=NS(remote_fill_requires_paired_restart=lambda: False),
        _record_checkpoint_restore_miss=lambda *a: False,
        _synchronize_dsa_cold_dense_load=unproven,
        _release_unadopted_shared_request_objects=lambda *a: released.append(
            "unadopted"
        ),
        _release_shared_worker_retrieve_state=lambda *a: released.append("shared"),
        _release_request_lookup_pins=lambda *a: released.append("pins"),
        _invalid_block_ids=set(),
    )
    fail = method(
        "_fail_completed_cold_load",
        _clear_terminal_load_tracebacks=lambda *a: None,
        logger=NS(exception=lambda *a: None, critical=lambda *a, **kw: None),
    )
    assert fail(adapter, "r", entry, None, error, False) is False
    assert not released and not adapter._invalid_block_ids
