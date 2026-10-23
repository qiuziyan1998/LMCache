# SPDX-License-Identifier: Apache-2.0
"""Exercise the production prefetch window without importing an NPU runtime."""

# Standard
import ast
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

# Third Party
import pytest


SOURCE = (
    Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/vllm_v1_adapter.py"
)


def adapter() -> NS:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if getattr(n, "name", "") == "LMCacheConnectorV1Impl"
    )
    names = {
        "_submit_initial_layerwise_prefill_load",
        "_advance_deferred_layerwise_prefill_load",
        "_layerwise_prefill_dma_block_ids",
        "_layerwise_prefill_slot_mapping",
        "record_layerwise_prefill_bank_use",
        "_wait_for_layerwise_prefill_bank",
    }
    methods = [n for n in cls.body if getattr(n, "name", "") in names]
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    namespace = {"prefill_start_timing_enabled": lambda: False}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    obj = NS()
    for method in methods:
        setattr(obj, method.name, MethodType(namespace[method.name], obj))
    return obj


@pytest.mark.parametrize("banks", [1, 2, 3, 8])
@pytest.mark.parametrize("layers,indexers", [(19, 11), (4, 1)])
def test_prefetch_window_covers_all_banks_and_clamps_short_groups(
    banks: int,
    layers: int,
    indexers: int,
) -> None:
    obj = adapter()
    requests = [NS(req_id="a", token_ids=[]), NS(req_id="b", token_ids=[])]
    submitted = []
    obj._layerwise_prefill_bank_count = banks
    obj._initial_layerwise_prefill_load_submitted = False
    obj._layerwise_requests = requests
    obj.layerwise_retrievers = [(object(), object()) for _ in requests]
    obj._latent_layer_names = list(range(layers))
    obj._indexer_layer_names = list(range(indexers))
    obj._sparse_retrieve_state_guard = lambda requests: nullcontext()
    obj._advance_dense_layerwise_retriever = lambda request, pair, group, layer: (
        submitted.append((request.req_id, group, layer))
    )
    obj._submit_initial_layerwise_prefill_load()
    assert submitted == [
        (request.req_id, group, layer)
        for request in requests
        for group, count in ((0, layers), (1, indexers))
        for layer in range(1, min(banks, count))
    ]
    before = list(submitted)
    obj._submit_initial_layerwise_prefill_load()
    assert submitted == before


@pytest.mark.parametrize("banks", [1, 2, 3, 8])
@pytest.mark.parametrize("layers", [3, 19])
def test_each_finished_layer_submits_n_plus_bank_count_and_drains_once(
    banks: int,
    layers: int,
) -> None:
    obj = adapter()
    request = NS(
        req_id="r", is_sparse_decode=False, block_allocation_mode="prefill_child"
    )
    pair = (object(), None)
    submitted = list(range(min(banks, layers)))
    results = []
    obj._layerwise_prefill_bank_count = banks
    obj._deferred_layerwise_prefill_load_active = True
    obj._layerwise_requests = [request]
    obj.layerwise_retrievers = [pair]
    obj._layerwise_waited_groups = set()
    obj._sparse_retrieve_state_guard = lambda requests: nullcontext()
    obj._layerwise_wait_group = lambda name: 0
    obj._layerwise_required_wait_groups = lambda: {0}
    obj._layerwise_prefill_load_position = lambda name, group: (name, name, layers - 1)
    obj._complete_layerwise_retrieve_group = lambda *args: None
    obj._validate_dense_retrieve_result = lambda *args, **kwargs: results.append(
        "drained"
    )

    def advance(request: NS, pair: tuple, group: int, layer: int) -> object:
        if layer < layers:
            assert layer == len(submitted)
            submitted.append(layer)
        return object()

    obj._advance_dense_layerwise_retriever = advance
    for layer in range(layers):
        obj.current_layer = layer
        if layer == layers - 1:
            obj._advance_deferred_layerwise_prefill_load(layer, finish_current=True)
        obj._advance_deferred_layerwise_prefill_load(layer)
    assert submitted == list(range(layers))
    assert results == ["drained"]


def test_eight_bank_mapping_and_mixed_phase_event_handoff() -> None:
    obj = adapter()
    obj._layerwise_prefill_p_node = True
    obj._layerwise_prefill_dma = True
    obj._layerwise_prefill_bank_count = 8
    request = NS(block_ids_by_bank=tuple(([bank], [100 + bank]) for bank in range(8)))
    assert obj._layerwise_prefill_dma_block_ids(request, 1) == tuple(
        [100 + b] for b in range(8)
    )
    calls = []
    obj._parent = NS(
        _get_connector_metadata=lambda: NS(
            requests=[NS(layerwise_prefill_bank_offset=phase) for phase in (0, 3, 7, 7)]
        )
    )
    obj._layerwise_wait_group = lambda name: 0
    obj._layerwise_prefill_transfer_layer_id = lambda name, group: 9
    obj.lmcache_engine = NS(
        gpu_connector=NS(
            record_layerwise_prefill_bank_use=lambda *args: calls.append(args),
        )
    )
    event = object()
    obj.record_layerwise_prefill_bank_use("layer", event)
    assert calls[0][:2] == (9, 0)
    assert set(calls[0][2]) == {0, 3, 7}
    assert calls[0][3] is event


@pytest.mark.parametrize(
    "banks,phases,loadable,expected_offset",
    [
        (4, [0, 1], [1], None),
        (4, [0, 1], [0], None),
        (8, [0, 3, 7], [0, 2], None),
        (4, [2, 2], [1], 2),
        (4, [3, 3], [], 3),
        (4, [], [], None),
    ],
)
def test_bank_wait_includes_scheduled_cache_misses(
    banks: int,
    phases: list[int],
    loadable: list[int],
    expected_offset: int | None,
) -> None:
    obj = adapter()
    requests = [NS(layerwise_prefill_bank_offset=phase) for phase in phases]
    obj._layerwise_prefill_bank_count = banks
    obj._layerwise_requests = [requests[index] for index in loadable]
    obj._parent = NS(_get_connector_metadata=lambda: NS(requests=requests))
    obj._layerwise_prefill_transfer_layer_id = lambda name, group: banks
    calls = []
    obj.lmcache_engine = NS(
        gpu_connector=NS(
            wait_for_layerwise_prefill_load=lambda **kwargs: calls.append(kwargs),
        )
    )
    obj._wait_for_layerwise_prefill_bank("layer", 1)
    expected = dict(layer_id=banks, kv_group=1)
    if expected_offset is not None:
        expected["bank_offset"] = expected_offset
    elif phases:
        expected["bank_offsets"] = tuple(sorted(set(phases)))
    assert calls == [expected]


@pytest.mark.parametrize("chunks", [0, 1, 7, 8, 19])
def test_chunk_bank_phase_ignores_remote_prefix_and_wraps_at_eight(chunks: int) -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    phase = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "layerwise_prefill_dma and lmcache_chunk_size > 0"
    )
    namespace = dict(
        layerwise_prefill_dma=True,
        lmcache_chunk_size=1024,
        tracker=NS(
            num_saved_tokens=32768 + chunks * 1024,
            num_lmcache_cached_tokens=32768,
            allocated_block_ids_by_bank=tuple(() for _ in range(8)),
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[phase], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    assert namespace["layerwise_prefill_bank_offset"] == chunks % 8
