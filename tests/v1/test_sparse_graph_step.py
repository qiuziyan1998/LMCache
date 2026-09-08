# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU contract tests for graph preparation without importing vLLM."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import ast

# Third Party
import pytest
import torch


def load_prepare_method() -> Any:
    """Load the real method without vLLM/NPU import-time dependencies."""
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
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_sparse_graph_step"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {
        "torch": torch,
        "_lmcache_nvtx_annotate": lambda fn: fn,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["prepare_sparse_graph_step"]


class FakeAdapter:
    prepare_sparse_graph_step = load_prepare_method()

    def __init__(self, *, warm: bool = True, draft: bool = True) -> None:
        self.num_layers = 3 if draft else 2
        self._latent_layer_names = [f"layers.{i}.attn" for i in range(self.num_layers)]
        self.current_layer = 0
        self.device = "cpu"
        self.request = SimpleNamespace(
            req_id="r1",
            is_sparse_decode=True,
            load_spec=SimpleNamespace(lmcache_cached_tokens=512),
        )
        self.source = SimpleNamespace(
            layers=(object(),) * self.num_layers, total_tokens=512
        )
        self.state = SimpleNamespace(
            prepared_sparse_sources={0: self.source} if warm else {},
            shared_request_active=warm,
            indexer_npu_resident=warm,
            indexer_npu_materialization_pending=not warm,
            slot_mapping=torch.zeros(512),
            decode_ret_mask=None,
        )
        self._worker_retrieve_state = {"r1": self.state}
        self._layerwise_requests = [self.request]
        self.layerwise_retrievers = [(object(), None if warm else object())]
        self._layerwise_retriever_is_sparse = [True]
        self._layerwise_sparse_req_ids = ["r1"]
        self._layerwise_sparse_shared_ordered = [False]
        self.waits: list[str] = []
        self.suffix_kwargs: dict[str, Any] = {}
        self.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            is_healthy=lambda: True,
            retrieve_layer_head_token_wise=self.retrieve,
        )

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        self.suffix_kwargs = kwargs
        yield None

    def _is_dsa_two_groups(self) -> bool:
        return True

    def _kvcaches_for_group(self, group: int) -> list[Any]:
        return [torch.zeros(1)] * self.num_layers

    def _drain_layerwise_retrievers(self) -> None:
        self.layerwise_retrievers.clear()
        self._layerwise_requests.clear()
        self._layerwise_retriever_is_sparse.clear()
        self._layerwise_sparse_req_ids.clear()
        self._layerwise_sparse_shared_ordered.clear()

    def wait_for_layer_load(self, name: str, **kwargs: Any) -> None:
        assert kwargs["selected_token_counts"].eq(0).all()
        assert kwargs["target_slot_mapping"].eq(-1).all()
        self.waits.append(name)
        self.current_layer += 1
        if self.current_layer == self.num_layers:
            self.state.prepared_sparse_sources = {0: self.source}
            self.state.shared_request_active = True
            self.state.indexer_npu_resident = True
            self.state.indexer_npu_materialization_pending = False
            self._drain_layerwise_retrievers()


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_prepare_keeps_draft_suffix_and_avoids_target_loads(
    warm: bool, draft: bool
) -> None:
    adapter = FakeAdapter(warm=warm, draft=draft)
    source = adapter.prepare_sparse_graph_step(("layers.0.attn", "layers.1.attn"))
    assert source is adapter.source
    assert adapter.current_layer == 2
    assert len(adapter.waits) == (0 if warm else adapter.num_layers)
    assert len(adapter.layerwise_retrievers) == int(draft)
    if draft:
        assert adapter.suffix_kwargs["prepared_start_layer"] == 2
        assert adapter.suffix_kwargs["prepared_sparse_source"] is source


def test_zero_frontier_requires_explicit_authorization() -> None:
    adapter = FakeAdapter()
    adapter._drain_layerwise_retrievers()
    with pytest.raises(RuntimeError, match="one shared-CPU sparse request"):
        adapter.prepare_sparse_graph_step(("layers.0.attn", "layers.1.attn"))
    assert (
        adapter.prepare_sparse_graph_step(
            ("layers.0.attn", "layers.1.attn"), allow_empty=True
        )
        is None
    )


def test_layer_order_and_health_fail_before_using_source() -> None:
    adapter = FakeAdapter()
    with pytest.raises(RuntimeError, match="ordered target-layer prefix"):
        adapter.prepare_sparse_graph_step(("layers.1.attn", "layers.0.attn"))
    adapter.lmcache_engine.is_healthy = lambda: False
    with pytest.raises(RuntimeError, match="unhealthy"):
        adapter.prepare_sparse_graph_step(("layers.0.attn", "layers.1.attn"))


def test_window_growth_prepares_before_returning_source() -> None:
    adapter = FakeAdapter()
    adapter.request.load_spec.lmcache_cached_tokens = 768
    adapter.source = SimpleNamespace(
        layers=(object(),) * adapter.num_layers, total_tokens=768
    )
    assert (
        adapter.prepare_sparse_graph_step(
            ("layers.0.attn", "layers.1.attn")
        ).total_tokens
        == 768
    )
    assert len(adapter.waits) == adapter.num_layers
