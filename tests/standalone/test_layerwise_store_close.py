# SPDX-License-Identifier: Apache-2.0
"""Store finalization must propagate an unsuccessful device completion fence."""

# Standard
import ast
from pathlib import Path
from types import SimpleNamespace

# Third Party
import pytest


def adapter_methods() -> type:
    """Load production finalization methods without the accelerator runtime."""
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheConnectorV1Impl"
    )
    names = {"_close_layerwise_storer", "_finalize_layerwise_storer"}
    module = ast.parse("from __future__ import annotations\nclass Adapter: pass")
    module.body[1].body = [
        node for node in cls.body if getattr(node, "name", None) in names
    ]
    namespace: dict = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Adapter"]


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_finalization_reports_failed_device_fence(error_type: type[Exception]) -> None:
    adapter = adapter_methods()()
    adapter._drain_layerwise_storer_fully = lambda storer: (False, None)
    pending_writes = ["unfinished D2H"]

    def close() -> None:
        raise error_type("D2H completion unknown")

    with pytest.raises(error_type, match="D2H completion unknown"):
        adapter._finalize_layerwise_storer(SimpleNamespace(close=close))
        pending_writes.clear()  # Request reuse is only legal after completion.

    assert pending_writes == ["unfinished D2H"]


def test_successful_finalization_keeps_result() -> None:
    adapter = adapter_methods()()
    result = object()
    adapter._drain_layerwise_storer_fully = lambda storer: (True, result)
    closed = []
    assert adapter._finalize_layerwise_storer(
        SimpleNamespace(close=lambda: closed.append(True))
    ) == (True, result)
    assert closed == [True]


def test_empty_storer_is_harmless() -> None:
    adapter_methods()._close_layerwise_storer(None)
