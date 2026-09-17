# SPDX-License-Identifier: Apache-2.0
"""CPU regressions from scheduled prefill chunks to two-group bank restores.

Execute production metadata and worker dispatch without vLLM/NPU services.
The transfer engine is a CPU stand-in; this does not validate NPU kernels.
"""

# Standard
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import ast
import logging

# Third Party
import pytest
import torch


ADAPTER_PATH = (
    Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/vllm_v1_adapter.py"
)


@pytest.fixture(scope="module")
def api() -> SimpleNamespace:
    """Load actual scheduler/worker definitions, stubbing only service imports."""
    methods = {
        "build_connector_meta",
        "_add_completed_cold_resume",
        "_take_completed_cold_load",
        "_build_request_meta",
        "_start_load_kv",
        "_is_decode_window_save_request",
        "_is_dsa_two_groups",
        "_is_deferred_layerwise_prefill_load_step",
        "supports_layerwise_prefill_transfer_window",
        "_materialize_layerwise_prefill_slot_mappings",
        "_layerwise_prefill_slot_mapping",
        "_load_tokens_for_retrieve",
        "_load_token_mask_for_retrieve",
        "_full_hit_recalc_last_token",
        "_prime_dense_prefix_retrievers",
        "_advance_dense_layerwise_retriever",
    }
    classes = {
        "LoadSpec",
        "SaveSpec",
        "RequestTracker",
        "ReqMeta",
        "WorkerRetrieveState",
        "LMCacheConnectorMetadata",
        "LMCacheConnectorV1Impl",
    }
    functions = {
        "_apply_mm_hashes",
        "_build_slot_mapping",
        "_build_slot_mapping_window",
        "_split_kv_group_block_ids",
        "_flatten_block_ids",
        "_copy_block_ids_by_bank",
        "_block_allocation_mode_value",
        "_disagg_spec_from_request",
        "extract_request_configs",
    }
    nodes = []
    for node in ast.parse(ADAPTER_PATH.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            if node.name == "LMCacheConnectorV1Impl":
                node.body = [
                    n for n in node.body if getattr(n, "name", None) in methods
                ]
                assert {n.name for n in node.body} == methods
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
    namespace = dict(
        __name__=__name__,
        dataclass=dataclass,
        field=field,
        deque=deque,
        torch=torch,
        KVConnectorMetadata=object,
        logger=logging.getLogger(__name__),
        cdiv=lambda n, d: (n + d - 1) // d,
        utils=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
        _lmcache_nvtx_annotate=lambda fn: fn,
        _mtp_dw_diag_enabled=lambda: False,
        _layerwise_prefill_p_node_enabled=lambda: True,
        serving_perf_enabled=lambda: False,
        extract_mm_features=lambda *a, **kw: (None, None),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ADAPTER_PATH), "exec"), namespace
    )
    return SimpleNamespace(**namespace)


def make_scheduler(api: SimpleNamespace, prompt: int, *, banked: bool = True) -> Any:
    """Use real tracking/metadata; preallocate capacity to isolate load intent."""
    adapter = api.LMCacheConnectorV1Impl()
    adapter._layerwise_prefill_p_node = True
    adapter.kv_role = "kv_producer" if banked else "kv_both"
    adapter.force_skip_save = False
    adapter._block_size = 128
    adapter._lmcache_chunk_size = 256
    adapter._discard_partial_chunks = False
    adapter.config = SimpleNamespace(
        save_decode_cache=False,
        dsa_two_groups=True,
        priority_limit=None,
    )
    adapter._windowed_sparse_layerwise_save_enabled = lambda: True
    adapter._add_decode_window_save_metas = Mock()
    adapter._take_completed_cold_load = lambda *a: False
    adapter._requests_priority = {}
    adapter.enable_sparse_attention = True
    adapter._dsa_kv_policy_log = False
    adapter._dsa_kv_policy_threshold = 10000
    adapter.load_specs = {}
    adapter._request_trackers = {}
    request = SimpleNamespace(
        request_id="r",
        num_computed_tokens=0,
        all_token_ids=list(range(prompt)),
        kv_transfer_params=None,
    )
    adapter._unfinished_requests = {"r": request}
    blocks = (prompt + 127) // 128
    banks = tuple(
        tuple(
            list(
                range(
                    1 + bank * 2 * blocks + group * blocks,
                    1 + bank * 2 * blocks + (group + 1) * blocks,
                )
            )
            for group in (0, 1)
        )
        for bank in (0, 1)
    )
    adapter.new_request = SimpleNamespace(
        req_id="r",
        num_computed_tokens=0,
        prompt_token_ids=request.all_token_ids,
        block_ids=banks[0],
        block_ids_by_bank=banks if banked else None,
        block_allocation_mode="prefill_child" if banked else "full_parent",
        sampling_params=None,
    )
    return adapter


def schedule(adapter: Any, computed: int, count: int) -> Any:
    """Invoke production build_connector_meta, including tracker advancement."""
    adapter._unfinished_requests["r"].num_computed_tokens = computed
    new = "r" not in adapter._request_trackers
    adapter.new_request.num_computed_tokens = computed
    return adapter.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=[],
            scheduled_new_reqs=[adapter.new_request] if new else [],
            num_scheduled_tokens={"r": count},
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[] if new else ["r"],
                new_block_ids=[] if new else [None],
                resumed_req_ids=set(),
            ),
        )
    )


@pytest.mark.parametrize(
    "prompt,chunk",
    [(9565, 4096), (8192, 4096), (8193, 4096), (3001, 1000), (257, 256), (256, 128)],
)
@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_every_continued_prefill_restores_exact_history(
    api: SimpleNamespace,
    prompt: int,
    chunk: int,
    role: str,
) -> None:
    adapter = make_scheduler(api, prompt)
    adapter.kv_role = role
    for computed in range(0, prompt, chunk):
        end = min(prompt, computed + chunk)
        meta = schedule(adapter, computed, end - computed).requests[0]
        assert not meta.is_sparse_decode
        assert meta.save_spec.skip_leading_tokens == computed
        assert meta.save_spec.can_save_latent and meta.save_spec.can_save_indexer
        assert adapter._request_trackers["r"].num_saved_tokens == end
        if not computed:
            assert meta.load_spec is None  # No history on a fresh cache miss.
            continue
        assert meta.load_spec is not None, "P lost the previous chunk's history load"
        assert meta.load_spec.can_load
        assert meta.load_spec.lmcache_cached_tokens == computed  # No rounding down.
        assert meta.load_spec.vllm_cached_tokens == 0  # Computed != bank-resident.
        assert len(meta.token_ids) == end  # Keep current save and prior load ranges.
        assert all(
            len(mapping) == end
            for bank in meta.slot_mappings_by_bank
            for mapping in bank
        )
        mask = adapter._load_token_mask_for_retrieve(meta, computed, 256)
        assert mask.all() and mask.numel() == computed
        assert not adapter._full_hit_recalc_last_token(
            meta.load_spec,
            meta.retrieve_token_count(),
            is_sparse_decode=False,
        )


def test_ordinary_prefill_keeps_resident_history_without_new_load(
    api: SimpleNamespace,
) -> None:
    adapter = make_scheduler(api, 9565, banked=False)
    for computed, count in [(0, 4096), (4096, 4096), (8192, 1373)]:
        assert schedule(adapter, computed, count).requests[0].load_spec is None


@pytest.mark.parametrize("banked", [False, True])
def test_cold_resume_preserves_allocation_metadata_until_next_growth(
    api: SimpleNamespace,
    banked: bool,
) -> None:
    adapter = make_scheduler(api, 256, banked=banked)
    adapter._decode_window_save_window_size = 0
    adapter._dsa_scratch_capacity = 128
    schedule(adapter, 0, 256)
    tracker = adapter._request_trackers["r"]
    mode = "prefill_child" if banked else "full_parent"
    assert tracker.block_allocation_mode == mode
    request = adapter._unfinished_requests["r"]
    request.all_token_ids.append(999)
    request.num_computed_tokens = 256
    spec = api.LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=256,
        can_load=False,
        dsa_remap_frontier=256,
    )
    adapter.load_specs["r"] = spec
    del adapter._take_completed_cold_load  # Use the real completion handoff.
    adapter._dsa_cold_loaded_req_ids = {"r"}
    # Keep emission lightweight while executing the real scheduler, tracker,
    # and cold-resume methods all the way through their early-continue branch.
    adapter._build_request_meta = lambda tr, load, **kw: SimpleNamespace(
        block_allocation_mode=tr.block_allocation_mode,
        allocated_block_ids_by_bank=tr.allocated_block_ids_by_bank,
    )
    restored = ([21, 22, 23], [31, 32, 33])
    restored_banks = (restored, ([41, 42, 43], [51, 52, 53])) if banked else None
    cached = SimpleNamespace(
        req_ids=["r"],
        new_block_ids=[restored],
        new_block_ids_by_bank=[restored_banks],
        new_block_allocation_modes=[mode],
        resumed_req_ids={"r"},
    )
    output = SimpleNamespace(
        finished_req_ids=[],
        scheduled_new_reqs=[],
        num_scheduled_tokens={"r": 1},
        scheduled_cached_reqs=cached,
    )
    meta = adapter.build_connector_meta(output).requests[0]
    assert meta.resumed_from_preemption
    assert meta.block_allocation_mode == mode
    assert tracker.allocated_block_ids_by_bank == restored_banks
    assert tracker.allocated_block_ids == restored[0]
    assert tracker.allocated_block_ids_indexer == restored[1]
    assert tracker.sparse_remap_frontier == 256
    assert tracker.token_ids == request.all_token_ids
    if banked:
        assert tracker.allocated_block_ids_by_bank[0][0] is not restored[0]

    # Empty block deltas must not erase the mode before the next allocation.
    cached.resumed_req_ids = set()
    cached.new_block_ids = [None]
    cached.new_block_ids_by_bank = [None]
    cached.new_block_allocation_modes = [None]
    request.num_computed_tokens = 257
    request.all_token_ids.append(1000)
    adapter.build_connector_meta(output)
    assert tracker.block_allocation_mode == mode

    cached.new_block_ids = [([24], [34])]
    cached.new_block_ids_by_bank = [(([24], [34]), ([44], [54])) if banked else None]
    cached.new_block_allocation_modes = [mode]
    request.num_computed_tokens = 258
    request.all_token_ids.append(1001)
    adapter.build_connector_meta(output)
    assert tracker.block_allocation_mode == mode
    assert tracker.allocated_block_ids == [21, 22, 23, 24]
    assert tracker.allocated_block_ids_indexer == [31, 32, 33, 34]
    if banked:
        assert tracker.allocated_block_ids_by_bank[1] == (
            [41, 42, 43, 44],
            [51, 52, 53, 54],
        )
    with pytest.raises(RuntimeError, match="allocation mode changed"):
        tracker.update(
            [],
            None,
            new_block_allocation_mode="full_parent" if banked else "prefill_child",
        )


@pytest.mark.parametrize("can_load", [False, True])
@pytest.mark.parametrize("cached", [4096, 9565])
def test_existing_lookup_or_resume_spec_is_preserved(
    api: SimpleNamespace,
    can_load: bool,
    cached: int,
) -> None:
    adapter = make_scheduler(api, 9565)
    original = api.LoadSpec(
        vllm_cached_tokens=0, lmcache_cached_tokens=cached, can_load=can_load
    )
    original.checkpoint_generation = 3
    adapter.load_specs["r"] = original
    meta = schedule(adapter, min(cached, 9564), 1).requests[0]
    assert meta.load_spec is original
    assert original.checkpoint_generation == 3


def test_decode_does_not_get_prefill_history_load(api: SimpleNamespace) -> None:
    adapter = make_scheduler(api, 256)
    schedule(adapter, 0, 256)
    adapter._unfinished_requests["r"].all_token_ids.append(999)
    meta = schedule(adapter, 256, 1).requests[0]
    assert meta.load_spec is None


def make_worker(api: SimpleNamespace, metadata: Any) -> Any:
    worker = api.LMCacheConnectorV1Impl()
    worker._layerwise_prefill_p_node = True
    worker.kv_role = "kv_producer"
    worker.use_layerwise = True
    worker.enable_sparse_attention = True
    worker.enable_blending = False
    worker.device = torch.device("cpu")
    worker._lmcache_chunk_size = 256
    worker.config = SimpleNamespace(dsa_two_groups=True)
    worker.kv_caches = {"latent": object(), "index": object()}
    worker._kvcaches_list = [worker.kv_caches["latent"]]
    worker._kvcaches_for_group = lambda group: [
        worker.kv_caches[("latent", "index")[group]]
    ]
    worker._indexer_layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
    worker._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    worker._prune_worker_retrieve_state = Mock()
    worker._drain_layerwise_retrievers = Mock()
    worker._prepare_p_node_layerwise_save_storers = Mock()
    worker._worker_retrieve_state = {}
    worker._set_worker_retrieve_state = lambda key, state: (
        worker._worker_retrieve_state.__setitem__(key, state)
    )
    worker._stats_monitor = Mock()
    worker.layerwise_retrievers = []
    worker._layerwise_requests = []
    worker._layerwise_retriever_is_sparse = []
    worker._layerwise_sparse_shared_ordered = []
    worker.lmcache_engine = SimpleNamespace(
        enable_shared_cpu_cache=True,
        supports_dense_sparse_cache_retention=lambda: True,
        gpu_connector=SimpleNamespace(
            supports_layerwise_prefill_transfer_window=True,
            set_layerwise_staging_concurrency=Mock(),
        ),
    )
    return worker


@pytest.mark.parametrize("computed", [4096, 8192, 1000])
def test_scheduler_metadata_starts_both_worker_transfers_and_rotates_banks(
    api: SimpleNamespace,
    computed: int,
) -> None:
    """No hand-written LoadSpec: the actual worker receives scheduler output."""
    scheduler = make_scheduler(api, computed + 100)
    schedule(scheduler, 0, computed)
    metadata = schedule(scheduler, computed, 100)
    worker = make_worker(api, metadata)
    meta = metadata.requests[0]
    hbm = torch.full((100000,), -1, dtype=torch.int64)
    read_rows = []

    def retrieve(tokens: list[int], mask: torch.Tensor, **kwargs: Any) -> Any:
        assert tokens == list(range(computed))
        assert mask.all() and mask.numel() == computed
        assert kwargs["deferred_layerwise_get"]
        assert kwargs["layerwise_prefill_bank_count"] == 2
        assert kwargs["vllm_cached_tokens"] == 0
        group = kwargs["kv_group"]
        command = yield torch.tensor(computed)
        for layer in range(4):
            slots = kwargs["slot_mapping"] if layer == 0 else command["slot_mapping"]
            prefix_slots = slots[:computed]
            hbm[prefix_slots] = group * 100 + layer
            read_rows.append((group, layer))
            command = yield None
        yield torch.ones(computed, dtype=torch.bool)

    worker.lmcache_engine.retrieve_layer = Mock(side_effect=retrieve)
    worker._start_load_kv(SimpleNamespace(attn_metadata={"real": object()}))
    assert worker._deferred_layerwise_prefill_load_active
    assert worker.lmcache_engine.retrieve_layer.call_count == 2
    assert read_rows == [(0, 0), (1, 0)]
    for layer in range(4):
        for group in (0, 1):
            if layer:
                worker._advance_dense_layerwise_retriever(
                    meta,
                    worker.layerwise_retrievers[0],
                    group,
                    layer,
                )
            slots = meta.slot_mappings_by_bank[layer % 2][group]
            assert (hbm[slots[:computed]] == group * 100 + layer).all()
            assert (hbm[slots[computed:]] == -1).all()  # Never clobber new tokens.
    assert len(read_rows) == 8
    for gen in worker.layerwise_retrievers[0]:
        gen.close()
