# SPDX-License-Identifier: Apache-2.0
"""Derived save hooks through real CPU publication; model/NPU work is doubled."""

# Standard
import ast
from collections.abc import Iterator, Mapping
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import storage_process as bridge
from lmcache.v1.cache_engine import LayerwiseStoreResult
from lmcache.v1.rpc.storage_import import CompactSourceImporter
from lmcache.v1.rpc.storage_lifetime import NativeSourceLifetime

# Local
from test_process_store_worker import engine, setup as process_setup
from test_storage_process_worker import definitions as worker_definitions, loaded


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "lmcache/integration/vllm/vllm_v1_adapter.py"
ASCEND = ROOT.parent / "LMCache-Ascend/lmcache_ascend"


@pytest.fixture
def setup() -> Iterator[Any]:
    yield from process_setup.__wrapped__()


def source_class(
    path: Path,
    name: str,
    methods: tuple[str, ...],
    namespace: dict[str, Any],
    base: str = "object",
) -> type:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if getattr(node, "name", None) == name)
    cls.bases = [ast.Name(id=base, ctx=ast.Load())]
    cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
    assert {node.name for node in cls.body} == set(methods)
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(cls)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def adapter_class(monkeypatch: pytest.MonkeyPatch) -> type:
    definitions = worker_definitions.__wrapped__(monkeypatch)
    ns = dict(
        torch=torch,
        WorkerRetrieveState=definitions[0],
        LayerwiseStoreResult=LayerwiseStoreResult,
        LMCacheConnectorMetadata=NS,
        logger=logging.getLogger(__name__),
        serving_perf_enabled=lambda: False,
        _lmcache_nvtx_annotate=lambda fn: fn,
        Mapping=Mapping,
    )
    source_class(
        BASE,
        "LMCacheConnectorV1Impl",
        (
            "save_kv_layer",
            "wait_for_save",
            "_wait_for_save_impl",
            "_should_defer_latent_save_under_tp",
            "_layerwise_store_kwargs",
            "_prepare_layerwise_store_inputs",
            "_flush_deferred_latent_store",
            "_store_result_from_yield",
            "_layerwise_storer_drain_limit",
            "_drain_layerwise_storer_fully",
            "_close_layerwise_storer",
            "_finalize_layerwise_storer",
            "_consume_completed_layerwise_store",
            "_promote_layerwise_store_result",
            "_prepared_sparse_source",
            "_layerwise_save_storer_key",
            "_layerwise_save_range",
            "_is_dsa_two_groups",
            "_is_decode_window_save_request",
            "_pad_chunk_local_slot_mapping",
            "_prefill_metadata_cache_enabled",
            "_worker_retrieve_state_for_warm_ref",
            "_worker_retrieve_state_for_request",
            "_materialize_dense_prefix_for_sparse",
            "_validate_shared_worker_retrieve_state",
            "_shared_retrieve_token_count_for_request",
        ),
        ns,
    )
    return source_class(
        ASCEND / "integration/vllm/vllm_v1_adapter.py",
        "LMCacheAscendConnectorV1Impl",
        (
            "_direct_prefill_requests",
            "save_kv_layer",
            "_wait_for_save_impl",
            "_finish_save_batch",
            "_consume_completed_layerwise_store",
            "_source_ready_event",
            "_preflight_direct_store",
            "_direct_selected_groups",
            "_direct_group_caches",
            "_direct_request_inputs",
            "_submit_direct_prefill_requests",
            "_producer_fence_handoff_targets",
        ),
        ns,
        "LMCacheConnectorV1Impl",
    )


def request() -> NS:
    return NS(
        req_id="request",
        token_ids=list(range(4)),
        request_configs={},
        is_sparse_decode=False,
        is_decode_window_save=False,
        is_last_prefill=False,
        live_source_requested=False,
        load_spec=None,
        slot_mapping=[torch.arange(4)],
        indexer_slot_mapping=[torch.arange(4)],
        save_spec=NS(
            can_save=True,
            can_save_latent=True,
            can_save_indexer=True,
            skip_leading_tokens=0,
        ),
    )


def make_adapter(cls: type, worker: Any, req: NS) -> Any:
    metadata = NS(requests=[req])
    obj = cls()
    groups = {0: [object(), object()], 1: [object(), object(), object()]}
    obj.__dict__.update(
        config=worker.config,
        lmcache_engine=worker,
        kv_role="kv_both",
        use_layerwise=True,
        device="cpu",
        num_layers=2,
        _lmcache_chunk_size=4,
        _parent=NS(
            _connector_metadata=metadata, _get_connector_metadata=lambda: metadata
        ),
        _vllm_config=NS(parallel_config=NS()),
        _direct_store_requested=True,
        _remote_store_requested=False,
        _force_layerwise_prefill_store=False,
        _layerwise_prefill_p_node=False,
        _direct_store_observed_layers=set(),
        _direct_store_step_supported=None,
        _latest_live_source_ready_event=None,
        _latest_live_source_ready_event_source="missing",
        _latest_direct_source_ready_events={},
        _unfenced_live_stores={},
        _completed_layerwise_stores={},
        _worker_retrieve_state={},
        _latent_layer_names=["layer.0", "layer.1"],
        _indexer_layer_names=["indexer.0", "indexer.1", "indexer.2"],
        kv_caches={"layer.0": object()},
        _kvcaches_list=groups[0],
        _layerwise_save_storers={},
        _deferred_latent_pending=set(),
        _refresh_kvcaches_list=lambda: None,
        _kvcaches_for_group=groups.get,
        _num_layers_for_group=lambda group: len(groups[group]),
        _windowed_sparse_save_mapping=lambda *a, **kw: None,
        _indexer_save_slot_mapping=lambda r, *a: r.indexer_slot_mapping[0],
        _note_decode_window_save_seen=lambda r: None,
        _record_prefill_save_group_completed=lambda *a: None,
        _record_decode_window_save_group_completed=lambda *a: None,
        _mark_decode_window_save_completed=lambda r: None,
        _mark_prefill_committed=lambda r: None,
        _mark_initial_sparse_release_ready=lambda r: None,
        _maybe_lookup_unpin_for_request=lambda r: None,
        _record_dsa_cold_dense_load_readiness=lambda s: None,
        _synchronize_dsa_cold_dense_load=lambda: None,
        _release_shared_worker_retrieve_state=lambda s, e: bridge.release_sources(s, e),
        _abort_save_step=lambda _: None,
    )
    coordinator = NS(last_latent_future=None)
    obj._get_cold_load_coordinator = lambda: coordinator
    obj._complete_worker_save_step = lambda: bridge.finish_store_step(obj)
    obj._set_worker_retrieve_state = lambda r, s: obj._worker_retrieve_state.update(
        {r: s}
    )
    return obj


@pytest.mark.parametrize("tail", [False, True])
def test_receiver_direct_option_keeps_local_prefill_source(
    setup: Any,
    adapter_class: type,
    tail: bool,
) -> None:
    _, worker, runtime, _ = engine(setup)
    worker.config.pd_role = "receiver"
    worker.config.decoder_storage_mode = "process"
    worker.config.dsa_two_groups = True
    worker.config.enable_remote_lmcache_store = True
    worker.config.extra_config.update(
        mooncake_direct_npu_prefill_store=True,
        use_ascend_direct=True,
        mooncake_page_first_multi_buffer=True,
        mooncake_layer_merged_page_objects=True,
        save_only_first_rank=True,
        save_chunk_meta=False,
    )
    worker.enable_shared_cpu_cache = True
    worker.is_store_async = True
    worker._force_layerwise_prefill_store = False
    # Evaluate the actual engine activation gate, independent of adapter routing.
    path = ASCEND / "v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    expression = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_direct_store_enabled"
            for t in node.targets
        )
    )
    worker._direct_store_enabled = eval(
        compile(ast.Expression(expression), str(path), "eval"),
        {"self": worker, "mooncake_layer_pages_enabled": lambda c: True},
    )
    assert not worker._direct_store_enabled
    for name in (
        "store_layer",
        "direct_prefill_store_enabled",
        "direct_prefill_plan_supported",
    ):
        method = loaded(
            path,
            name,
            {
                "torch": torch,
                "_lmcache_nvtx_annotate": lambda fn: fn,
            },
            "AscendLMCacheEngine",
        )
        setattr(worker, name, MethodType(method, worker))
    worker.gpu_connector.direct_page_layout_supported = lambda *a: True
    worker.wait_for_pending_sync_stores = lambda: None
    worker.wait_for_direct_stores = lambda *a: None
    worker.direct_store_committed_ends = lambda r: {}
    worker.store_direct_prefill = Mock(
        side_effect=AssertionError("receiver direct store")
    )
    worker.storage_process.descriptor = {"sessions": {"worker-0": "a"}}
    worker.compact_source_importer = CompactSourceImporter(
        layout=worker.storage_layout,
        mapping=worker.storage_process.mapping,
        local_device_base=4096,
        client=worker.storage_process.client,
        upload=lambda t: t,
        fence_upload=lambda: None,
    )
    worker.native_source_lifetime = NativeSourceLifetime(lambda: None, pytest.fail)
    worker._ordered_checkpoint_agreement = lambda *a: a[-1]
    req = request()
    obj = make_adapter(adapter_class, worker, req)
    ends = (4, 8, 11) if tail else (4, 8)
    previous = None
    for end in ends:
        req.save_spec.skip_leading_tokens = len(req.token_ids) if previous else 0
        req.token_ids = list(range(end))
        req.slot_mapping = req.indexer_slot_mapping = [torch.arange(end)]
        req.is_last_prefill = end == ends[-1]
        for layer in (*obj._latent_layer_names, *obj._indexer_layer_names):
            obj.save_kv_layer(layer, None, None)
        obj.wait_for_save()
        state = obj._worker_retrieve_state.get(req.req_id)
        # The same dispatch predicate as the first sparse load. Before the fix
        # no store ran, so this takes the ticket loader and raises on null.
        source = obj._prepared_sparse_source(state, 0, end)
        if source is None:
            req.load_spec = NS(cpu_source_ticket=None)
            bridge.ordinary_layerwise_load(
                obj,
                req,
                req.token_ids,
                torch.ones(end, dtype=torch.bool),
                torch.arange(end),
                torch.arange(end),
                request_ordinal=0,
            )
        assert state.token_count == source.total_tokens == end
        assert source.owner_lease.is_valid() and state.shared_request_active
        assert worker.compact_source_importer.is_certified_source(
            source,
            req.req_id,
            state.process_load_generation,
        )
        assert not obj._layerwise_save_storers and not obj._deferred_latent_pending
        if previous is not None:
            assert not previous.owner_lease.is_valid()
        previous = source
    # Full first-decode metadata supplies the scratch mapping; subsequent warm
    # references must find the source, without a lookup ticket or another store.
    req.is_sparse_decode = True
    req.sparse_warm_ref = True
    req.load_spec = NS(can_load=True, lmcache_cached_tokens=end, cpu_source_ticket=None)
    state.slot_mapping = torch.arange(4)
    assert obj._worker_retrieve_state_for_warm_ref(req) is state
    # An ordinary warm decode with saving disabled performs no storage work.
    operations = len(worker.storage_process.client.operations)
    transfers = len(worker.gpu_connector.transfers)
    req.save_spec.can_save = False
    for layer in (*obj._latent_layer_names, *obj._indexer_layer_names):
        obj.save_kv_layer(layer, None, None)
    obj.wait_for_save()
    assert len(worker.storage_process.client.operations) == operations
    assert len(worker.gpu_connector.transfers) == transfers
    assert obj._worker_retrieve_state_for_warm_ref(req) is state
    worker.store_direct_prefill.assert_not_called()
    assert len(
        [
            op
            for op in worker.storage_process.client.operations.values()
            if op.method == "commit_write"
        ]
    ) == 2 * len(ends)
    bridge.release_sources(state, worker)
    assert worker.compact_source_importer.sources_drained()


@pytest.mark.parametrize(
    "role,storage_mode",
    [
        (None, "local"),
        ("sender", "local"),
        ("receiver", "local"),
        ("receiver", "process"),
    ],
)
@pytest.mark.parametrize("passive", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_direct_prefill_selection_is_role_consistent_across_ranks(
    adapter_class: type,
    monkeypatch: pytest.MonkeyPatch,
    role: str | None,
    passive: bool,
    direct: bool,
    storage_mode: str,
) -> None:
    base_save, submit = Mock(), Mock()
    monkeypatch.setattr(adapter_class.__bases__[0], "save_kv_layer", base_save)
    worker = NS(
        config=NS(pd_role=role, dsa_two_groups=True, decoder_storage_mode=storage_mode),
        direct_prefill_store_enabled=lambda: not passive and role != "receiver",
    )
    obj = make_adapter(adapter_class, worker, request())
    obj._direct_store_requested = direct
    obj._preflight_direct_store = lambda _: True
    obj._submit_direct_prefill_requests = submit
    for layer in (*obj._latent_layer_names, *obj._indexer_layer_names):
        obj.save_kv_layer(layer, None, None)
    use_direct = direct and role != "receiver"
    assert base_save.call_count == (0 if use_direct else 5)
    assert submit.call_count == int(use_direct)


@pytest.mark.parametrize("mode", ["dcp", "rotating_prefill"])
def test_receiver_routing_preserves_specialized_save_paths(
    adapter_class: type,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    base_save = Mock()
    monkeypatch.setattr(adapter_class.__bases__[0], "save_kv_layer", base_save)
    worker = NS(config=NS(pd_role="receiver", dsa_two_groups=True))
    obj = make_adapter(adapter_class, worker, request())
    if mode == "dcp":
        obj._dsa_dcp_size = 8
    else:
        obj.config.pd_role = "sender"
        obj._force_layerwise_prefill_store = True
    obj._preflight_direct_store = Mock(side_effect=AssertionError("direct preflight"))
    obj.save_kv_layer("layer.0", None, None)
    assert base_save.call_count == int(mode == "rotating_prefill")
    obj._preflight_direct_store.assert_not_called()
