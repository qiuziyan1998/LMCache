# SPDX-License-Identifier: Apache-2.0
"""Run with Python alone; exercise the actual connector methods without NPU imports."""

# Standard
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
METHOD = "synchronize_prefill_vup_output"


def load_method(path: str) -> Callable:
    """Load the production method without importing vLLM or device packages."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == METHOD
    )
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[METHOD]


class PrefillVupFenceTest(unittest.TestCase):
    def test_connector_preserves_group_local_ordinal(self) -> None:
        adapter_method = load_method("lmcache/integration/vllm/vllm_v1_adapter.py")
        connector_method = load_method(
            "lmcache/integration/vllm/lmcache_connector_v1.py"
        )
        for group, ordinal, model_layer in ((0, 7, 7), (1, 3, 19)):
            with self.subTest(group=group):
                layer_name = f"model.layers.{model_layer}.self_attn"
                resolve = Mock(return_value=ordinal)
                select_group = Mock(return_value=group)
                synchronize = Mock()
                adapter = SimpleNamespace(
                    _layerwise_prefill_p_node=True,
                    _layerwise_wait_group=select_group,
                    _layerwise_prefill_transfer_layer_id=resolve,
                    lmcache_engine=SimpleNamespace(
                        gpu_connector=SimpleNamespace(
                            synchronize_prefill_vup_output=synchronize
                        )
                    ),
                )
                adapter.synchronize_prefill_vup_output = MethodType(
                    adapter_method, adapter
                )
                connector = SimpleNamespace(_lmcache_engine=adapter)
                connector_method(connector, layer_name)
                select_group.assert_called_once_with(layer_name)
                resolve.assert_called_once_with(layer_name, group)
                synchronize.assert_called_once_with(ordinal, group)

    def test_decode_node_does_not_resolve_or_synchronize(self) -> None:
        adapter_method = load_method("lmcache/integration/vllm/vllm_v1_adapter.py")
        connector_method = load_method(
            "lmcache/integration/vllm/lmcache_connector_v1.py"
        )
        # No resolver or device connector: accessing either would fail the test.
        adapter = SimpleNamespace(_layerwise_prefill_p_node=False)
        adapter.synchronize_prefill_vup_output = MethodType(adapter_method, adapter)
        connector_method(
            SimpleNamespace(_lmcache_engine=adapter), "model.layers.7.self_attn"
        )


if __name__ == "__main__":
    unittest.main()
