# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU contract tests for graph preparation without importing vLLM."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import ast
from copy import deepcopy

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.sparse import build_prepared_sparse_source


def load_prepare_method(name: str = "prepare_sparse_graph_step") -> Any:
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
        if isinstance(node, ast.FunctionDef) and node.name == name
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
        "build_prepared_sparse_source": build_prepared_sparse_source,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class FakeAdapter:
    prepare_sparse_graph_step = load_prepare_method()
    _sparse_decode_requires_index_materialization = load_prepare_method(
        "_sparse_decode_requires_index_materialization"
    )

    def __init__(
        self,
        *,
        warm: bool = True,
        draft: bool = True,
        native_index: bool = False,
        target_layers: int = 2,
    ) -> None:
        self.num_layers = target_layers + int(draft)
        self._latent_layer_names = [f"layers.{i}.attn" for i in range(self.num_layers)]
        self.current_layer = 0
        self.device = "cpu"
        self.kv_role = "kv_both"
        self.native_index = native_index
        self.bootstrap_completes = True
        self.request = SimpleNamespace(
            req_id="r1",
            is_sparse_decode=True,
            load_spec=SimpleNamespace(lmcache_cached_tokens=512),
            shared_index_skipped=native_index,
            resumed_from_preemption=False,
            disagg_spec=None,
        )
        self.source = SimpleNamespace(
            layers=(object(),) * self.num_layers, total_tokens=512
        )
        self.state = SimpleNamespace(
            prepared_sparse_sources={0: self.source} if warm else {},
            shared_request_active=warm,
            indexer_npu_resident=warm and not native_index,
            indexer_npu_materialization_pending=not warm and not native_index,
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
            enable_shared_cpu_cache=not native_index,
            is_healthy=lambda: True,
            retrieve_layer_head_token_wise=self.retrieve,
        )

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        self.suffix_kwargs = kwargs
        yield None

    def _is_dsa_two_groups(self) -> bool:
        return True

    def _shared_cpu_materialize_index_on_decode_cold(self) -> bool:
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
        if self.current_layer == self.num_layers and self.bootstrap_completes:
            self.state.prepared_sparse_sources = {0: self.source}
            self.state.shared_request_active = True
            self.state.indexer_npu_resident = not self.native_index
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
    with pytest.raises(RuntimeError, match="unique sparse requests"):
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


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_batch_sources_follow_model_order_and_keep_draft_suffixes(
    cold: bool, shared: bool
) -> None:
    adapter = FakeAdapter()
    second = deepcopy(adapter.request)
    second.req_id = "r2"
    second.load_spec.lmcache_cached_tokens = 768
    state = deepcopy(adapter.state)
    second_source = SimpleNamespace(
        layers=(object(),) * adapter.num_layers, total_tokens=768
    )
    state.prepared_sparse_sources = {0: second_source} if not cold else {}
    state.indexer_npu_resident = not cold
    state.shared_request_active = shared
    adapter.state.shared_request_active = shared
    adapter.lmcache_engine.enable_shared_cpu_cache = shared
    adapter._worker_retrieve_state["r2"] = state
    adapter._layerwise_requests.append(second)
    adapter.layerwise_retrievers.append((object(), None))
    calls = []

    def retrieve(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        yield None

    def wait(name: str, **kwargs: Any) -> None:
        assert kwargs["request_ids"] == ["r1", "r2"]
        assert kwargs["selected_token_counts"].tolist() == [0, 0]
        assert kwargs["target_slot_mapping"].eq(-1).all()
        adapter.waits.append(name)
        state.prepared_sparse_sources = {0: second_source}
        state.indexer_npu_resident = True

    adapter.lmcache_engine.retrieve_layer_head_token_wise = retrieve
    adapter.wait_for_layer_load = wait
    result = adapter.prepare_sparse_graph_step(
        ("layers.0.attn", "layers.1.attn"),
        request_ids=("r2", "no-history", "r1"),
        frontiers=(768, 0, 512),
    )
    assert result == (second_source, None, adapter.source)
    assert len(adapter.waits) == (adapter.num_layers if cold else 0)
    assert len(calls) == 2
    assert all(call["prepared_start_layer"] == 2 for call in calls)
    assert calls[0]["prepared_sparse_source"] is adapter.source
    assert calls[1]["prepared_sparse_source"] is second_source


def test_batch_cannot_invent_missing_positive_frontier_source() -> None:
    adapter = FakeAdapter()
    with pytest.raises(RuntimeError, match="preparation failed"):
        adapter.prepare_sparse_graph_step(
            ("layers.0.attn", "layers.1.attn"),
            request_ids=("r1", "missing"),
            frontiers=(512, 512),
        )


def test_batch_empty_source_is_explicit_per_request() -> None:
    adapter = FakeAdapter()
    adapter._drain_layerwise_retrievers()
    assert adapter.prepare_sparse_graph_step(
        ("layers.0.attn", "layers.1.attn"),
        request_ids=("r0", "r2"),
        frontiers=(0, 0),
    ) == (None, None)


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("target_layers", [2, 8])
def test_local_prefill_index_does_not_require_lmcache_materialization(
    warm: bool, draft: bool, target_layers: int
) -> None:
    adapter = FakeAdapter(
        warm=warm, draft=draft, native_index=True, target_layers=target_layers
    )
    # prepare_sparse_graph_step executes the real eager materialization policy,
    # loaded above, rather than an unconditional index-ready test stub.
    source = adapter.prepare_sparse_graph_step(
        tuple(f"layers.{i}.attn" for i in range(target_layers)),
        request_ids=("r1",),
        frontiers=(512,),
    )
    assert source == (adapter.source,)
    assert not adapter.state.indexer_npu_resident
    assert len(adapter.waits) == (0 if warm else adapter.num_layers)
    assert len(adapter.layerwise_retrievers) == int(draft)


def test_native_index_still_bootstraps_growing_latent_history() -> None:
    adapter = FakeAdapter(native_index=True)
    adapter.source = SimpleNamespace(
        layers=(object(),) * adapter.num_layers, total_tokens=768
    )
    adapter.request.load_spec.lmcache_cached_tokens = 768
    source = adapter.prepare_sparse_graph_step(
        ("layers.0.attn", "layers.1.attn"),
        request_ids=("r1",),
        frontiers=(768,),
    )
    assert source[0].total_tokens == 768
    assert len(adapter.waits) == adapter.num_layers
    assert not adapter.state.indexer_npu_resident


@pytest.mark.parametrize(
    "case",
    ["shared", "consumer", "resumed", "disaggregated", "no_skip", "pending"],
)
def test_native_index_exception_does_not_bypass_other_readiness_guards(
    case: str,
) -> None:
    adapter = FakeAdapter(native_index=True)
    adapter.bootstrap_completes = False
    if case == "shared":
        adapter.lmcache_engine.enable_shared_cpu_cache = True
    elif case == "consumer":
        adapter.kv_role = "kv_consumer"
    elif case == "resumed":
        adapter.request.resumed_from_preemption = True
    elif case == "disaggregated":
        adapter.request.disagg_spec = object()
    elif case == "no_skip":
        adapter.request.shared_index_skipped = False
    else:
        adapter.state.indexer_npu_materialization_pending = True
    with pytest.raises(RuntimeError, match="index_resident=False"):
        adapter.prepare_sparse_graph_step(
            ("layers.0.attn", "layers.1.attn"),
            request_ids=("r1",),
            frontiers=(512,),
        )


def test_native_index_never_hides_short_or_missing_latent_source() -> None:
    for warm in (False, True):
        adapter = FakeAdapter(warm=warm, native_index=True)
        adapter.bootstrap_completes = False
        with pytest.raises(RuntimeError) as error:
            adapter.prepare_sparse_graph_step(
                ("layers.0.attn", "layers.1.attn"),
                request_ids=("r1",),
                frontiers=(768,),
            )
        assert "req_id=r1 frontier=768" in str(error.value)
        assert "source_tokens=" + ("512" if warm else "None") in str(error.value)
        assert "native_index=True" in str(error.value)


def test_native_index_requires_all_registered_index_layers() -> None:
    class MissingIndexLayer(FakeAdapter):
        def _kvcaches_for_group(self, group: int) -> list[Any]:
            return [torch.zeros(1)] * (self.num_layers - int(group == 1))

    adapter = MissingIndexLayer(native_index=True)
    with pytest.raises(RuntimeError, match="native_index=False"):
        adapter.prepare_sparse_graph_step(
            ("layers.0.attn", "layers.1.attn"),
            request_ids=("r1",),
            frontiers=(512,),
        )


class SourcePublisher:
    """Exercise the actual publication method with CPU cache payloads."""

    refresh_sources = load_prepare_method("_refresh_prepared_sparse_sources")
    _cached_prefix_covered_token_count = load_prepare_method(
        "_cached_prefix_covered_token_count"
    )
    _cached_ranges_cover_prefix = load_prepare_method("_cached_ranges_cover_prefix")

    def __init__(self, layers: int) -> None:
        self.layers = layers
        self.device = torch.device("cpu")
        self._lmcache_chunk_size = 256

    def _is_dsa_two_groups(self) -> bool:
        return True

    def _num_layers_for_group(self, group: int) -> int:
        return self.layers


def make_source_cache(tokens: int, layers: int) -> dict[str, Any]:
    starts = list(range(0, tokens, 256))
    ends = [min(start + 256, tokens) for start in starts]
    tensors = [
        [torch.zeros(end - start) for start, end in zip(starts, ends, strict=True)]
        for _ in range(layers)
    ]
    return {
        "cached_starts": starts,
        "cached_ends": ends,
        "cached_tensors": tensors,
        "cached_memory_objs": [[object() for _ in starts] for _ in range(layers)],
        "cached_chunk_ptrs_npu": [
            torch.tensor([tensor.data_ptr() for tensor in layer], dtype=torch.int64)
            for layer in tensors
        ],
    }


@pytest.mark.parametrize("first_group", [1, 0], ids=["indexer-first", "latent-first"])
@pytest.mark.parametrize("tokens", [1024, 1023], ids=["full-chunks", "partial-tail"])
@pytest.mark.parametrize("layers", [1, 9], ids=["one-layer", "target-plus-mtp"])
def test_source_publication_waits_for_matching_group_frontiers(
    first_group: int, tokens: int, layers: int
) -> None:
    publisher = SourcePublisher(layers)
    caches = {group: make_source_cache(512, layers) for group in (0, 1)}
    state = SimpleNamespace(
        cache_kwargs=lambda group, two_groups: caches[group],
        prepared_sparse_sources={},
    )
    publisher.refresh_sources(state, 512)
    assert set(state.prepared_sparse_sources) == {0, 1}

    # TP's final indexer callback publishes before the deferred latent flush.
    # Also cover the reverse completion order and a final partial CPU chunk.
    caches[first_group] = make_source_cache(tokens, layers)
    first_cache = caches[first_group]
    frontier = tokens if first_group == 0 else 512
    publisher.refresh_sources(state, frontier)
    assert set(state.prepared_sparse_sources) == {0}
    assert state.prepared_sparse_sources[0].total_tokens == frontier
    assert caches[first_group] is first_cache
    assert first_cache["cached_ends"][-1] == tokens

    # The second completion must publish both sources without another load.
    caches[1 - first_group] = make_source_cache(tokens, layers)
    publisher.refresh_sources(state, tokens)
    assert set(state.prepared_sparse_sources) == {0, 1}
    for group, source in state.prepared_sparse_sources.items():
        cache = caches[group]
        assert source.total_tokens == tokens
        assert sum(source.chunk_token_counts) == tokens
        assert source.validated_chunk_size == 256
        for index, layer in enumerate(source.layers):
            assert layer.chunk_ptrs_npu is cache["cached_chunk_ptrs_npu"][index]
            assert layer.memory_objs == tuple(cache["cached_memory_objs"][index])
            assert all(
                actual is expected
                for actual, expected in zip(
                    layer.tensors, cache["cached_tensors"][index], strict=True
                )
            )


@pytest.mark.parametrize("malformed", ["overlap", "pointer-count"])
def test_source_publication_keeps_strict_validation(malformed: str) -> None:
    publisher = SourcePublisher(1)
    caches = {group: make_source_cache(512, 1) for group in (0, 1)}
    if malformed == "overlap":
        caches[1]["cached_starts"] = [0, 0, 256]
        caches[1]["cached_ends"] = [256, 256, 512]
    else:
        caches[1]["cached_chunk_ptrs_npu"] = [torch.zeros(1, dtype=torch.int64)]
    prior_sources = {}
    state = SimpleNamespace(
        cache_kwargs=lambda group, two_groups: caches[group],
        prepared_sparse_sources=prior_sources,
    )
    with pytest.raises(ValueError, match="coverage"):
        publisher.refresh_sources(state, 512)
    assert state.prepared_sparse_sources is prior_sources
