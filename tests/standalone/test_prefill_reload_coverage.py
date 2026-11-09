# SPDX-License-Identifier: Apache-2.0
"""Reject missing required history in the production retrieval path."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "lmcache/integration/vllm/vllm_v1_adapter.py"
ENGINE = ROOT / "lmcache/v1/cache_engine.py"


def extracted_class(path, class_name, names, namespace):
    source = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(
        node for node in source.body if getattr(node, "name", None) == class_name
    )
    owner.body = [node for node in owner.body if getattr(node, "name", None) in names]
    owner.bases = []
    for method in owner.body:
        if method.name == "retrieve_layer":
            method.decorator_list = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(owner)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]


class Key:
    def __init__(self, start, layer=0):
        self.start, self.layer = start, layer

    def split_layers(self, count):
        return [Key(self.start, layer) for layer in range(count)]


class ReadStarted(Exception):
    pass


def reader(shared, available, *, missing_layer=None, healthy=True):
    events, envelopes = [], []
    namespace = dict(
        torch=torch,
        CacheEngineKey=Key,
        mooncake_page_layout_enabled=lambda config: False,
        _RemoteFillMaterializationError=type(
            "MaterializationError", (RuntimeError,), {}
        ),
    )
    cls = extracted_class(
        ENGINE,
        "LMCacheEngine",
        {
            "retrieve_layer",
            "_require_complete_prefill_source",
        },
        namespace,
    )
    obj = object.__new__(cls)
    obj.is_healthy = lambda: healthy
    obj._num_transfer_layers_for_call = lambda *args: 2
    obj._should_use_shared_layerwise_retrieve = lambda group: shared
    obj._is_passive = lambda: False
    obj._get_req_id = lambda kwargs: kwargs.get("req_id", "test")
    obj.config = NS()
    obj.gpu_connector = NS()
    obj.stats_monitor = NS(on_retrieve_request=lambda count: 1)
    obj.retrieve_locations = []
    obj.shared_cpu_cache_strict = False
    obj._remote_fill_retrieve_plan = lambda *args: None
    obj._dense_retrieve_token_results = lambda *args: [
        (0, 4, Key(0)),
        (4, 8, Key(4)),
    ]

    def location(key, *args):
        events.append(("lookup", key.start, key.layer))
        if key.start >= available or key.layer == missing_layer:
            return None
        return "LocalCPUBackend"

    def read(*args, **kwargs):
        events.append(("read",))
        raise ReadStarted()

    obj.storage_manager = NS(contains=location, layerwise_batched_get=read)
    obj._find_shared_rank0_chunk_location = location
    obj._retrieve_layer_shared_rank0 = read
    obj._broadcast_shared_envelope = envelopes.append
    obj._shared_layerwise_error_envelope = lambda **kwargs: kwargs
    return obj, events, envelopes


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("available", [0, 4])
def test_required_cache_miss_fails_inside_reader_before_data_read(
    shared, group, available
):
    obj, events, envelopes = reader(shared, available)
    with pytest.raises(RuntimeError, match="incomplete during retrieval") as error:
        next(
            obj.retrieve_layer(
                list(range(8)),
                req_id="request",
                kv_group=group,
                deferred_layerwise_get=True,
                _require_complete_prefill_prefix=True,
            )
        )
    assert f"kv_group={group}" in str(error.value)
    assert f"retrieved_tokens={available}" in str(error.value)
    assert "required_tokens=8" in str(error.value)
    assert all(event[0] == "lookup" for event in events)
    assert len(envelopes) == int(shared)
    if shared:
        assert envelopes[0]["message"] == str(error.value)
        assert envelopes[0]["layer_id"] == 0


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("strict,available", [(False, 4), (True, 8)])
def test_optional_miss_and_complete_required_prefix_reach_existing_read(
    shared, strict, available
):
    obj, events, envelopes = reader(shared, available)
    with pytest.raises(ReadStarted):
        next(
            obj.retrieve_layer(
                list(range(8)),
                deferred_layerwise_get=True,
                _require_complete_prefill_prefix=strict,
            )
        )
    assert events[-1] == ("read",)
    assert envelopes == []


def test_missing_later_layer_reports_through_shared_error_envelope():
    obj, events, envelopes = reader(True, 8, missing_layer=1)
    with pytest.raises(RuntimeError, match="retrieved_tokens=0"):
        next(
            obj.retrieve_layer(
                list(range(8)),
                deferred_layerwise_get=True,
                _require_complete_prefill_prefix=True,
            )
        )
    assert len(envelopes) == 1
    assert all(event[0] == "lookup" for event in events)


def test_unhealthy_required_reader_raises_instead_of_yielding_a_miss():
    obj, events, _ = reader(False, 0, healthy=False)
    with pytest.raises(RuntimeError, match="unhealthy during required"):
        next(obj.retrieve_layer(list(range(8)), _require_complete_prefill_prefix=True))
    assert events == []


@pytest.mark.parametrize(
    "required,available,recalc,passes",
    [
        (8, 8, False, True),
        (8, 7, False, False),
        (8, 7, True, True),
        (8, 6, True, False),
        (0, 0, True, True),
        (1, 0, True, True),
    ],
)
def test_recomputed_tail_and_empty_prefix_contract(required, available, recalc, passes):
    obj, _, _ = reader(False, available)
    starts, ends = ([4], [4 + available]) if available else ([], [])

    def check():
        obj._require_complete_prefill_source(
            starts,
            ends,
            required,
            req_id="r",
            kv_group=0,
            recalculate_last_token=recalc,
        )

    if passes:
        check()
    else:
        with pytest.raises(RuntimeError, match="incomplete during retrieval"):
            check()


def test_coverage_validation_reads_only_range_endpoints():
    class Endpoint:
        def __init__(self, index, value):
            self.index, self.value = index, value

        def __getitem__(self, index):
            assert index == self.index
            return self.value

        def __bool__(self):
            return True

    obj, _, _ = reader(False, 0)
    obj._require_complete_prefill_source(
        Endpoint(0, 1024),
        Endpoint(-1, 513024),
        512000,
        req_id="large",
        kv_group=1,
        recalculate_last_token=False,
    )


def test_adapter_priming_consumes_existing_yields_without_tensor_operations():
    cls = extracted_class(
        ADAPTER,
        "LMCacheConnectorV1Impl",
        {
            "_prime_dense_prefix_retrievers",
        },
        {},
    )
    events = []

    def retrieve(group):
        events.append((group, "prepared"))
        yield object()
        events.append((group, "submitted"))
        yield object()

    cls._prime_dense_prefix_retrievers(retrieve(0), retrieve(1))
    assert events == [
        (0, "prepared"),
        (1, "prepared"),
        (0, "submitted"),
        (1, "submitted"),
    ]


@pytest.mark.parametrize(
    "p_node,deferred", [(False, False), (False, True), (True, False), (True, True)]
)
def test_adapter_marks_only_banked_p_reads_as_required(p_node, deferred):
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    gate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Assign)
            and "_require_complete_prefill_prefix" in ast.unparse(child)
            for child in node.body
        )
    )
    runtime = dict(
        self=NS(
            _layerwise_prefill_p_node=p_node,
            _deferred_layerwise_prefill_load_active=deferred,
        ),
        deferred_prefill_kwargs={},
        recalc_last_applied=True,
    )
    exec(
        compile(ast.Module(body=[gate], type_ignores=[]), str(ADAPTER), "exec"), runtime
    )
    expected = (
        {
            "_require_complete_prefill_prefix": True,
            "_prefill_recalculate_last_token": True,
        }
        if p_node and deferred
        else {}
    )
    assert runtime["deferred_prefill_kwargs"] == expected
