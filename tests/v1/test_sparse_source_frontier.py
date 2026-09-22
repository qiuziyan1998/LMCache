# SPDX-License-Identifier: Apache-2.0
# Standard
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.sparse import build_prepared_sparse_source


def load_prepare_method(name: str) -> Any:
    """Load the real method without vLLM/NPU import-time dependencies."""
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LMCacheConnectorV1Impl"
        )
    )
    method = next(
        (
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
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
        cache_kwargs=lambda group, two_groups: caches[group], prepared_sparse_sources={}
    )
    publisher.refresh_sources(state, 512)
    assert set(state.prepared_sparse_sources) == {0, 1}
    caches[first_group] = make_source_cache(tokens, layers)
    first_cache = caches[first_group]
    frontier = tokens if first_group == 0 else 512
    publisher.refresh_sources(state, frontier)
    assert set(state.prepared_sparse_sources) == {0}
    assert state.prepared_sparse_sources[0].total_tokens == frontier
    assert caches[first_group] is first_cache
    assert first_cache["cached_ends"][-1] == tokens
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
                (
                    actual is expected
                    for actual, expected in zip(
                        layer.tensors, cache["cached_tensors"][index], strict=True
                    )
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
