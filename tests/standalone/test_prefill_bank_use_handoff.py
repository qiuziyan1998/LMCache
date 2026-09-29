# SPDX-License-Identifier: Apache-2.0
"""Exercise bank-event routing without importing vLLM or the NPU runtime."""

# Standard
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

# Third Party
import pytest


ROOT = Path(__file__).resolve().parents[2] / "lmcache/integration/vllm"


def load_method(filename: str, class_name: str) -> Callable[..., None]:
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "record_layerwise_prefill_bank_use"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("group,ordinal", [(0, 6), (1, 3), (0, 78), (1, 22)])
def test_handoff_includes_cache_misses_and_passive_rank_requests(
    group: int,
    ordinal: int,
) -> None:
    forward = load_method("lmcache_connector_v1.py", "LMCacheConnectorV1Dynamic")
    route = load_method("vllm_v1_adapter.py", "LMCacheConnectorV1Impl")
    requests = [
        SimpleNamespace(layerwise_prefill_bank_offset=i, load_spec=None)
        for i in (0, 1, 1)
    ]
    calls = []
    event = object()
    impl = SimpleNamespace(
        _layerwise_prefill_p_node=True,
        # This list deliberately excludes cache misses. It must not be used.
        _layerwise_requests=[],
        _parent=SimpleNamespace(
            _get_connector_metadata=lambda: SimpleNamespace(requests=requests),
        ),
        _layerwise_wait_group=lambda name: group,
        _layerwise_prefill_transfer_layer_id=lambda name, group: ordinal,
        lmcache_engine=SimpleNamespace(
            gpu_connector=SimpleNamespace(
                record_layerwise_prefill_bank_use=lambda *args: calls.append(args),
            )
        ),
    )
    impl.record_layerwise_prefill_bank_use = lambda name, event: route(
        impl, name, event
    )
    forward(SimpleNamespace(_lmcache_engine=impl), "registered.layer", event)
    assert len(calls) == 1
    layer, kv_group, offsets, handed_off = calls[0]
    assert (layer, kv_group) == (ordinal, group)
    assert set(offsets) == {0, 1}
    assert handed_off is event


def test_d_node_never_inspects_or_records_prefill_bank_events() -> None:
    route = load_method("vllm_v1_adapter.py", "LMCacheConnectorV1Impl")
    route(SimpleNamespace(_layerwise_prefill_p_node=False), "layer", object())
