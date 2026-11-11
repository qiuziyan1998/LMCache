# SPDX-License-Identifier: Apache-2.0
"""Real worker helpers/importer/envelopes; source-loaded Ascend generator, fake NPU."""

import ast
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Barrier
from types import MethodType, ModuleType, SimpleNamespace as NS
import sys

import msgspec
import numpy as np
import pytest
import torch

from lmcache.integration.vllm import storage_process as worker
from lmcache.integration.vllm.decode_window_commit import (
    publish_delayed_decode_window_commit,
)
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
    module.DsaDcpDenseLoad = loaded(
        root / "lmcache/integration/vllm/vllm_v1_adapter.py",
        "DsaDcpDenseLoad",
        {"dataclass": dataclass, "torch": torch},
    )
    module.LoadSpec = loaded(
        root / "lmcache/integration/vllm/vllm_v1_adapter.py",
        "LoadSpec",
        {"dataclass": dataclass},
    )
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


@pytest.mark.parametrize(
    "world,phase,compact",
    [(4, "miss", False), (8, "resume", False), (4, "resident_resume", False),
     (4, "decode", False), (8, "long_decode", False), (4, "decode", True),
     (1, "decode", False), (8, "persisted_decode", False),
     (4, "persisted_decode", False), (8, "persisted_long_decode", False),
     (8, "persisted_decode", True), (1, "persisted_decode", False),
     (1, "persisted_logged_decode", False), (1, "persisted_aligned_decode", False)],
)
@pytest.mark.parametrize("storage_mode", ["local", "process"])
def test_dcp_scheduler_metadata_follows_the_allocated_layout(
    world, phase, compact, storage_mode,
):
    path = Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/vllm_v1_adapter.py"

    def cdiv(n, d):
        return (n + d - 1) // d

    ns = dict(
        dataclass=dataclass, field=field, deque=deque, torch=torch,
        cdiv=cdiv, utils=NS(cdiv=cdiv), KVConnectorMetadata=object,
        _lmcache_nvtx_annotate=lambda fn: fn, SPARSE_DECODE_RETRIEVE_TOKENS=2048,
        extract_mm_features=lambda *a, **kw: (None, None),
        extract_request_configs=lambda _: None,
        logger=NS(debug=lambda *a: None, error=lambda *a: None),
        _mtp_dw_event=lambda *a, **kw: None,
        publish_delayed_decode_window_commit=publish_delayed_decode_window_commit,
    )
    for name in (
        "_flatten_block_ids", "_split_kv_group_block_ids", "_copy_block_ids_by_bank",
        "_block_allocation_mode_value", "_disagg_spec_from_request", "_apply_mm_hashes",
        "_build_slot_mapping", "_sparse_slot_mapping_len", "LoadSpec", "SaveSpec",
        "RequestTracker", "ReqMeta", "LMCacheConnectorMetadata",
    ):
        loaded(path, name, ns)

    resident = phase == "resident_resume"
    persisted = phase.startswith("persisted_")
    prompt_len = {
        "resident_resume": 257, "long_decode": 16385,
        "persisted_decode": 16385, "persisted_long_decode": 49153,
        "persisted_logged_decode": 11577,
        "persisted_aligned_decode": 16384,
    }.get(phase, 6145)
    prompt = list(range(prompt_len))
    first = phase in ("miss", "resume", "resident_resume")
    computed = {"miss": 0, "resume": 4096, "resident_resume": 256}.get(phase, prompt_len)
    scheduled = 1024 if phase == "miss" else len(prompt) - computed if first else 1
    end = computed + scheduled
    # Mixed BF16/C8 paired bundles expose two latent and 18 indexer IDs.
    blocks0 = cdiv(end, 128 * world * 2) * 2
    blocks1 = cdiv(end, 128 * 18) * 18
    if compact:
        blocks0 += 32
    block_ids = (list(range(2, blocks0 + 2)), list(range(200, 200 + blocks1)))
    request = NS(req_id="r", prompt_token_ids=prompt, num_computed_tokens=computed,
                 all_token_ids=[*prompt, 9999], block_ids=block_ids, sampling_params=None,
                 dsa_compact_allocated=compact)
    spec = ns["LoadSpec"](0, len(prompt) if resident else computed, False,
                         process_ordinary_load=phase == "resume")
    spec.dsa_cold_load_generation = 3
    if resident:
        spec.dsa_cold_compact_load = spec.dsa_cold_resident_load = True
        spec.dsa_committed_end = len(prompt)
        spec.dsa_remap_frontier = 0
        spec.dsa_cold_resume_computed_end = computed
    adapter = NS(
        config=NS(dsa_two_groups=True, priority_limit=None, save_decode_cache=False,
                  decoder_storage_mode=storage_mode),
        kv_role="kv_both" if persisted else "kv_consumer", force_skip_save=False,
        enable_sparse_attention=True, _dsa_dcp_size=world if world > 1 else None,
        _dsa_scratch_capacity=4096, _dsa_kv_policy_threshold=0, _dsa_kv_policy_log=False,
        _dsa_dcp_live_origin=32,
        _block_size=128, _lmcache_chunk_size=1024, _discard_partial_chunks=True,
        _decode_window_save_window_size=0, _requests_priority={}, _request_trackers={},
        _unfinished_requests={"r": request}, load_specs={"r": spec} if first else {},
        _dsa_cold_loaded_req_ids={"r"} if phase in ("resume", "resident_resume") else set(),
        _add_decode_window_save_metas=lambda *a: None,
        _windowed_sparse_layerwise_save_enabled=lambda: False,
    )
    if not first:
        tracker = ns["RequestTracker"].from_new_request(
            adapter.config, request, len(prompt),
            0 if persisted and not compact else 4096, not persisted,
        )
        if compact:
            tracker.dsa_nonresident_frontier = tracker.sparse_remap_frontier = len(prompt) - 1
        adapter._request_trackers["r"] = tracker
    for name in ("_take_completed_cold_load", "_build_request_meta", "_dcp_write_layout"):
        setattr(adapter, name, MethodType(loaded(path, name, ns, "LMCacheConnectorV1Impl"), adapter))
    if persisted:
        # Actual final-prefill completion followed by scheduler reconciliation.
        # Saving the prompt also reports completion when decode-window size is 0.
        mark = loaded(path, "_mark_prefill_committed", ns, "LMCacheConnectorV1Impl")
        saved = NS(
            _completed_decode_window_saves={}, _lmcache_chunk_size=1024,
            _is_decode_window_save_request=lambda _: False,
        )
        mark(
            saved, NS(req_id="r", token_ids=prompt,
                      is_last_prefill=True, is_sparse_decode=False), len(prompt),
        )
        adapter._eligible_dsa_release_frontier = MethodType(
            loaded(path, "_eligible_dsa_release_frontier", ns,
                   "LMCacheConnectorV1Impl"), adapter,
        )
        consume = loaded(path, "update_connector_output", ns, "LMCacheConnectorV1Impl")
        consume(
            adapter,
            NS(completed_decode_window_saves=saved._completed_decode_window_saves),
        )
    build = loaded(path, "build_connector_meta", ns, "LMCacheConnectorV1Impl")
    result = build(adapter, NS(
        finished_req_ids=set(), scheduled_new_reqs=[request] if first else [],
        scheduled_cached_reqs=NS(
            req_ids=[] if first else ["r"], resumed_req_ids=set(), new_block_ids=[None],
        ),
        num_scheduled_tokens={"r": scheduled},
    )).requests[0]
    if persisted:
        # Exercise the deployed Ascend graph classifier, not a test-only oracle.
        ascend = path.parents[4] / "vllm-ascend/vllm_ascend"
        route_ns = {
            "Enum": Enum, "dataclass": dataclass, "np": np,
            "sfa_full_graph_enabled": lambda _: True,
            "unwrap_staged_sfa_connector_metadata": lambda metadata: metadata,
            "CUDAGraphMode": NS(PIECEWISE="piecewise"),
        }
        loaded(path.parents[4] / "vllm/vllm/forward_context.py",
               "BatchDescriptor", route_ns)
        for name in ("StagedSFAQueryProfile", "StagedSFAGraphKey"):
            loaded(ascend / "ascend_forward_context.py", name, route_ns)
        for name in (
            "StagedSFARouteAction", "StagedSFARouteReason", "StagedSFARouteDecision",
        ):
            loaded(ascend / "utils.py", name, route_ns)
        loaded(ascend / "attention/attention_v1.py", "AscendAttentionState", route_ns)
        for name in ("_dsa_remap_frontier", "staged_sfa_metadata_sparse_route"):
            loaded(ascend / "attention/utils.py", name, route_ns)
        route = loaded(ascend / "worker/model_runner_v1.py", "_staged_sfa_local_route",
                       route_ns, "NPUModelRunner")
        bind = loaded(ascend / "worker/model_runner_v1.py", "_staged_sfa_live_route",
                      route_ns, "NPUModelRunner")
        apply = loaded(ascend / "worker/model_runner_v1.py", "_apply_staged_sfa_route",
                       route_ns, "NPUModelRunner")
        runner = NS(
            _staged_sfa_graph_capture_sizes=[2, 4, 8, 16, 32],
            speculative_config=NS(num_speculative_tokens=1), decode_threshold=2,
            vllm_config=NS(lora_config=None),
            parallel_config=NS(data_parallel_size=1),
            attn_state=route_ns["AscendAttentionState"].DecodeOnly,
        )
        for query_width in (1, 2):
            decision = route(
                runner, num_tokens_unpadded=query_width, num_reqs=1,
                num_scheduled_tokens=np.array([query_width]),
                index_topk=2048, has_cascade_attention=False, request_ids=["r"],
                kv_connector_metadata=NS(requests=[result]),
                num_computed_tokens=[computed], prompt_lens=[prompt_len],
            )
            assert decision.action == route_ns["StagedSFARouteAction"].STAGED, (
                decision.reason
            )
            live = bind(
                runner, local_route=decision, dp_route_action=decision.action,
                cudagraph_mode="piecewise",
                batch_descriptor=route_ns["BatchDescriptor"](num_tokens=2),
                num_tokens_unpadded=query_width, num_tokens_padded=2,
                num_reqs=1, should_ubatch=False,
            )
            key = apply(runner, live)
            assert key is not None and key.token_capacity == 2
            assert key.query_profile == (
                route_ns["StagedSFAQueryProfile"].DECODE_BOUNDED if query_width == 1
                else route_ns["StagedSFAQueryProfile"].SPEC_FIXED
            )
        release_expected = prompt_len // 1024 * 1024 if compact or world == 1 else 0
        assert tracker.dsa_current_released_frontier == release_expected
        assert tracker.dsa_nonresident_frontier == release_expected
        assert tracker.decode_window_save_committed_end == prompt_len // 1024 * 1024
        assert saved._completed_decode_window_saves == (
            {"r": release_expected} if release_expected else {}
        )
        assert decision.frontiers == (release_expected,)
    if compact or world == 1:
        assert result.is_sparse_decode and result.load_spec.can_load
        assert result.load_spec.dsa_remap_frontier == (
            len(prompt) - 1 if compact else len(prompt) // 1024 * 1024
        )
        if persisted and world == 1:
            extent = (len(prompt) if storage_mode == "process"
                      else len(prompt) // 1024 * 1024)
            assert result.load_spec.lmcache_cached_tokens == extent
            assert result.retrieve_token_count() == extent
            assert result.load_spec.cpu_source_ticket is None
            assert request.num_computed_tokens == computed
            assert result.load_spec.dsa_committed_end == release_expected
            assert result.load_spec.dsa_release_frontier == release_expected
            assert result.indexer_slot_mapping[0].numel() == extent
            last_block = block_ids[1][(extent - 1) // 128]
            assert result.indexer_slot_mapping[0][-1] == (
                last_block * 128 + (extent - 1) % 128
            )
            if storage_mode == "process":
                # A locally stored paired source includes the partial tail.
                # Its exact-length certificate must survive first sparse decode.
                source = NS(total_tokens=len(prompt))
                state = NS(
                    shared_request_active=True, prepared_sparse_sources={0: source}
                )
                worker = NS(lmcache_engine=NS(enable_shared_cpu_cache=True))
                select = loaded(path, "_prepared_sparse_source", ns,
                                "LMCacheConnectorV1Impl")
                assert select(worker, state, 0, extent) is source
                request.num_computed_tokens = len(tracker.token_ids)
                request.all_token_ids.append(9998)
                following = build(adapter, NS(
                    finished_req_ids=set(), scheduled_new_reqs=[],
                    scheduled_cached_reqs=NS(
                        req_ids=["r"], resumed_req_ids=set(), new_block_ids=[None],
                    ),
                    num_scheduled_tokens={"r": 1},
                )).requests[0]
                assert following.sparse_warm_ref and following.token_ids == []
                assert following.retrieve_token_count() == extent
                assert following.load_spec.dsa_remap_frontier == release_expected
                assert following.load_spec.cpu_source_ticket is None
    elif resident:
        assert result.is_sparse_decode and result.load_spec.dsa_cold_compact_resume
        assert result.load_spec.dsa_remap_frontier == 0
        assert spec.process_ordinary_resume and spec.dsa_cold_load_generation == 3
        assert not result.slot_mapping and not result.indexer_slot_mapping
    else:
        assert not result.is_sparse_decode
        assert not result.slot_mapping and not result.indexer_slot_mapping
        assert result.load_spec is (spec if first else None)
        if first:
            assert not spec.can_load and len(result.token_ids) == end // 1024 * 1024
        if phase == "resume":
            assert spec.process_ordinary_resume and spec.dsa_cold_load_generation == 3


@pytest.mark.parametrize("stale", [False, True])
def test_dcp_resident_resume_consumes_readiness_without_a_layer_load(stale):
    path = Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/vllm_v1_adapter.py"
    spec = NS(
        can_load=True, process_ordinary_resume=True, dsa_cold_compact_resume=True,
        dsa_cold_resident_load=True, dsa_cold_load_generation=3, lmcache_cached_tokens=257,
        vllm_cached_tokens=0,
    )
    request = NS(req_id="r", load_spec=spec, resumed_from_preemption=False,
                 is_sparse_decode=True, retrieve_token_count=lambda: 257)
    ready = object()
    state = NS(
        completed_cold_load_generation=2 if stale else 3, token_count=257,
        dense_prefix_resident_tokens=257, indexer_npu_resident=True,
        prepared_sparse_sources={0: object()}, dense_load_readiness=ready,
        dense_load_readiness_consumed=False,
    )
    waits = []
    adapter = NS(
        _parent=NS(_get_connector_metadata=lambda: NS(requests=[request])),
        kv_caches={"latent": object()}, _kvcaches_list=[object()],
        lmcache_engine=NS(gpu_connector=NS(consume_dense_load_readiness=waits.append)),
        _worker_retrieve_state={"r": state}, _layerwise_prefill_p_node=False, use_layerwise=True,
        _is_decode_window_save_request=lambda _: False,
        _prune_worker_retrieve_state=lambda *a: None, _drain_layerwise_retrievers=lambda: None,
        _prepare_p_node_layerwise_save_storers=lambda _: None,
        _stats_monitor=NS(update_interval_vllm_hit_tokens=lambda _: None,
                          update_interval_prompt_tokens=lambda _: None),
    )

    def check_loads(requests):
        assert not list(requests)
        return False

    adapter._is_deferred_layerwise_prefill_load_step = check_loads
    scope = dict(LMCacheConnectorMetadata=NS)
    completed = loaded(path, "completed_cold_resume_state", scope)
    adapter._completed_cold_resume = lambda req: completed(req, state)
    adapter._consume_dsa_cold_dense_load_readiness = MethodType(
        loaded(path, "_consume_dsa_cold_dense_load_readiness", scope, "LMCacheConnectorV1Impl"), adapter,
    )
    start = loaded(path, "_start_load_kv", scope, "LMCacheConnectorV1Impl")
    context = NS(attn_metadata={})
    if stale:
        with pytest.raises(RuntimeError, match="lost its prepared generation"):
            start(adapter, context)
        assert not waits
    else:
        start(adapter, context)
        start(adapter, context)
        assert waits == [ready]


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
    from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec

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
    spec = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=5,
        can_load=True,
        cpu_source_ticket=ticket,
        dsa_group1_direct_hbm=True,
    )
    spec.dsa_cold_load_generation = 1
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


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED_SAFE"])
def test_active_reply_reuses_validated_result_and_passive_decodes_once(
    definitions, monkeypatch, status
):
    queue = Queue()
    first, plan, _, _ = setup(definitions, queue=queue)
    second, peer, _, _ = setup(definitions, rank=1, queue=queue)
    worker.submit_prepare(first, plan)
    worker.submit_prepare(second, peer)
    future = plan["storage_prepare"]
    expected = worker.decode_message(
        worker.encode_message(
            msgspec.structs.replace(
                future.result(),
                status=status,
                error="safe miss" if status == "FAILED_SAFE" else None,
            )
        ),
        StorageResult,
    )
    replacement = complete(expected)
    replacement.identity, replacement.payload_digest = (
        future.identity,
        future.payload_digest,
    )
    plan["storage_prepare"] = replacement
    plan["storage_allow_safe_miss"] = peer["storage_allow_safe_miss"] = True
    first.lmcache_engine.storage_process.client.operation_status = lambda *args: (
        complete(StorageResult(expected.identity, "TERMINAL", {"leases": {}}))
    )
    decoded = []
    decode = worker.decode_message

    def counted(frames, record_type):
        decoded.append(record_type)
        return decode(frames, record_type)

    monkeypatch.setattr(worker, "decode_message", counted)
    assert worker._exchange_reply(first, plan) is expected
    assert decoded == []
    received = worker._exchange_reply(second, peer)
    assert msgspec.to_builtins(received) == msgspec.to_builtins(expected)
    assert decoded == [StorageResult]


@pytest.mark.parametrize("rank", [0, 1])
def test_reply_reuse_preserves_common_envelope_validation(definitions, rank):
    adapter, plan, _, _ = setup(definitions, rank=rank)
    worker.submit_prepare(adapter, plan)
    engine = adapter.lmcache_engine
    make_envelope = engine._shared_layerwise_error_envelope

    def stale(**kwargs):
        return replace(make_envelope(**kwargs), generation=99)

    engine._shared_layerwise_error_envelope = stale
    if rank:
        engine._receive_matching_shared_envelope = lambda **kwargs: stale(
            **kwargs, message="", details={}
        )
    with pytest.raises(ValueError, match="generation=99"):
        worker._exchange_reply(adapter, plan)


def test_passive_reply_still_rejects_unsupported_wire_version(definitions):
    queue = Queue()
    first, plan, _, _ = setup(definitions, queue=queue)
    second, peer, _, _ = setup(definitions, rank=1, queue=queue)
    worker.submit_prepare(first, plan)
    worker.submit_prepare(second, peer)
    worker._exchange_reply(first, plan)
    envelope = queue.get_nowait()
    result = msgspec.structs.replace(plan["storage_prepare"].result(), version=999)
    envelope["error_details"]["storage_result"] = worker.encode_message(result)
    queue.put(envelope)
    with pytest.raises(ValueError, match="Unsupported decoder storage protocol"):
        worker._exchange_reply(second, peer)


@pytest.mark.parametrize("resident", [None, False, True])
def test_cold_load_optional_resident_flag_on_active_and_passive_ranks(
    definitions, resident: bool | None
) -> None:
    queue = Queue()
    for rank in (0, 1):
        adapter, plan, calls, _ = setup(definitions, rank=rank, queue=queue)
        spec = plan["request"].load_spec
        if resident is None:
            assert not hasattr(spec, "dsa_cold_resident_load")
        else:
            spec.dsa_cold_resident_load = resident
        worker.submit_prepare(adapter, plan)
        if rank == 0:
            assert calls[0][1].mode == ("resident" if resident else "compact")
        else:
            assert calls == []
        state = worker.run_cold(adapter, plan, None, complete())
        try:
            assert state.indexer_npu_resident and state.metadata_warm
            assert state.dense_prefix_resident_tokens == (5 if resident else 0)
            assert adapter.lmcache_engine.gpu_connector.payloads == (
                [(0, None)] * 2 if resident else []
            )
            assert plan["latent_shared_ready"].done()
            assert state.prepared_sparse_sources[0].owner_lease.is_valid()
        finally:
            worker.release_sources(state, adapter.lmcache_engine)
        assert adapter.lmcache_engine.compact_source_importer.sources_drained()


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


def test_dcp_workers_restore_owned_rows_from_the_same_prepared_prefix(definitions):
    from lmcache.integration.vllm.vllm_v1_adapter import DsaDcpDenseLoad

    queue = Queue()
    layout = DsaDcpDenseLoad(2, 2, (7, 11))
    for rank, expected in enumerate((([1, 4], [15, 22]), ([2], [14]))):
        adapter, plan, calls, _ = setup(definitions, rank=rank, queue=queue)
        plan["request"].load_spec.dsa_cold_resident_load = True
        plan["request"].dsa_dcp_load = layout
        plan["dsa_dcp_rank"] = rank
        plan["token_mask"] = torch.tensor([False, True, True, False, True])
        worker.submit_prepare(adapter, plan)
        state = worker.run_cold(adapter, plan, None, complete())
        source = state.prepared_sparse_sources[0]
        try:
            assert len(calls) == (1 if rank == 0 else 0)
            assert source.total_tokens == state.dense_prefix_resident_tokens == 5
            assert source.owner_lease.is_valid() and state.indexer_npu_resident
            payloads = adapter.lmcache_engine.gpu_connector.payloads
            assert len(payloads) == len(source.layers)
            for group, (ids, start, slots, counts) in payloads:
                assert group == start == 0
                assert (ids.tolist(), slots.tolist()) == expected
                assert counts.tolist() == [len(expected[0])]
        finally:
            worker.release_sources(state, adapter.lmcache_engine)
        assert adapter.lmcache_engine.compact_source_importer.sources_drained()


@pytest.mark.parametrize("dcp_size", [4, 8])
def test_dcp_dense_selection_covers_exact_prefix_and_partial_tail(definitions, dcp_size):
    from lmcache.integration.vllm.vllm_v1_adapter import DsaDcpDenseLoad

    length = 1025
    span = 128 * dcp_size
    blocks = tuple(range(40, 40 + (length + span - 1) // span))
    mask = torch.arange(length) >= 123
    mask[256:260] = False
    layout = DsaDcpDenseLoad(128, dcp_size, blocks)
    seen = set()
    for rank in range(dcp_size):
        ids, slots = layout.select(rank, mask)
        expected = [t for t in range(length) if mask[t] and t // 128 % dcp_size == rank]
        assert ids.tolist() == expected
        assert slots.tolist() == [blocks[t // span] * 128 + t % 128 for t in expected]
        assert seen.isdisjoint(expected)
        seen.update(expected)
    assert seen == set(mask.nonzero().flatten().tolist())


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
    worker.load_prepared_dense_source(adapter, source, plan, group)
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
        worker.load_prepared_dense_source(adapter, source, plan, 0)
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
