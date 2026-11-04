# SPDX-License-Identifier: Apache-2.0
"""Diagnostic calls retain the production layer-to-KV-group mapping."""

# Standard
import ast
from pathlib import Path
from types import SimpleNamespace as NS

# Third Party
import pytest


ROOT = Path(__file__).resolve().parents[2]


def method(path: str) -> object:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef)
        and item.name == "synchronize_prefill_sfa_inputs"
    )
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("p_node", [False, True])
def test_public_connector_routes_only_p_calls_to_the_correct_ordinal(
    p_node: bool,
) -> None:
    calls = []
    gpu = NS(
        synchronize_prefill_sfa_inputs=lambda layer, group: calls.append((layer, group))
    )
    adapter = NS(
        _layerwise_prefill_p_node=p_node,
        _layerwise_wait_group=lambda name: 0,
        _layerwise_prefill_transfer_layer_id=lambda name, group: 7,
        lmcache_engine=NS(gpu_connector=gpu),
    )
    adapter_method = method("lmcache/integration/vllm/vllm_v1_adapter.py")
    adapter.synchronize_prefill_sfa_inputs = lambda name: adapter_method(adapter, name)
    public = NS(_lmcache_engine=adapter)
    public_method = method("lmcache/integration/vllm/lmcache_connector_v1.py")
    public_method(public, "model.layers.7.self_attn")
    assert calls == ([(7, 0)] if p_node else [])
