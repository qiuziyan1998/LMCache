# SPDX-License-Identifier: Apache-2.0
"""Run real store promotion/merge/sealing on CPU with staggered KV groups."""

# Standard
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


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def api() -> SimpleNamespace:
    """Avoid service/native imports, but execute production methods unchanged."""
    namespace = dict(
        __name__=__name__,
        dataclass=dataclass,
        field=field,
        torch=torch,
        logger=logging.getLogger(__name__),
    )
    adapter_methods = {
        "_is_dsa_two_groups",
        "_is_decode_window_save_request",
        "_cached_prefix_covered_token_count",
        "_cached_ranges_cover_prefix",
        "_copy_layer_cache",
        "_snapshot_worker_retrieve_cache_state",
        "_restore_worker_retrieve_cache_state",
        "_state_has_retrieve_tensor_cache",
        "_resolve_store_retrieve_location",
        "_ensure_layer_cache_shape",
        "_merge_cache_group_by_ranges",
        "_merge_store_result_into_worker_state",
        "_retain_shared_store_seed_state",
        "_store_result_has_retrieve_data",
        "_promote_layerwise_store_result",
        "_refresh_prepared_sparse_sources",
        "_prepared_sparse_source",
        "_set_worker_retrieve_state",
        "_mark_worker_retrieve_registry_changed",
    }
    for filename, names in (
        (
            "lmcache/v1/gpu_connector/sparse.py",
            {
                "PreparedSparseSource",
                "PreparedSparseSourceLayer",
                "build_prepared_sparse_source",
            },
        ),
        ("lmcache/v1/cache_engine.py", {"LayerwiseStoreResult"}),
        (
            "lmcache/integration/vllm/vllm_v1_adapter.py",
            {"WorkerRetrieveState", "LMCacheConnectorV1Impl"},
        ),
    ):
        path = ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if getattr(n, "name", None) in names]
        assert {n.name for n in nodes} == names
        for node in nodes:
            if node.name == "LMCacheConnectorV1Impl":
                node.body = [
                    n for n in node.body if getattr(n, "name", None) in adapter_methods
                ]
                assert {n.name for n in node.body} == adapter_methods
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def make_adapter(api: SimpleNamespace, groups: bool = True) -> Any:
    adapter = api.LMCacheConnectorV1Impl()
    adapter.config = SimpleNamespace(dsa_two_groups=groups)
    adapter.device = torch.device("cpu")
    adapter._lmcache_chunk_size = 256
    adapter._worker_retrieve_state = {}
    adapter._num_layers_for_group = lambda group: (2, 1)[group]
    adapter.lmcache_engine = SimpleNamespace(
        enable_shared_cpu_cache=True,
        storage_manager=None,
        store_location="LocalCPUBackend",
        retain_shared_cpu_store_seed=Mock(),
    )
    adapter._release_shared_worker_retrieve_state = Mock()
    return adapter


def store_result(api: SimpleNamespace, group: int, start: int, end: int) -> Any:
    starts = list(range(start, end, 256))
    tensors = [
        [
            torch.full((min(256, end - s), 1), group * 100000 + layer * 20000 + s)
            for s in starts
        ]
        for layer in range((2, 1)[group])
    ]
    pointers = [[t.data_ptr() for t in layer] for layer in tensors]
    return api.LayerwiseStoreResult(
        request_id="req",
        kv_group=group,
        starts=starts,
        ends=[min(s + 256, end) for s in starts],
        keys=[
            [f"{group}/{layer}/{s}" for s in starts] for layer in range(len(tensors))
        ],
        memory_objs=[[object() for _ in starts] for _ in tensors],
        tensors=tensors,
        chunk_dev_ptrs=pointers,
        chunk_ptrs=[torch.tensor(layer, dtype=torch.int64) for layer in pointers],
        committed_end=end,
    )


@pytest.mark.parametrize("order", [(1, 0), (0, 1)])
@pytest.mark.parametrize("tail", [12288, 12301])
def test_chunked_prefill_promotes_each_group_without_relabelling_or_losing_kv(
    api: SimpleNamespace, order: tuple[int, int], tail: int
) -> None:
    adapter = make_adapter(api)
    request = SimpleNamespace(req_id="req", is_sparse_decode=False)
    start = 0
    for end in (4096, 8192, tail):
        old_state = adapter._worker_retrieve_state.get("req")
        old_sources = dict(old_state.prepared_sparse_sources) if old_state else {}
        for group in order:
            result = store_result(api, group, start, end)
            # Before the fix the second indexer-first step raises exactly:
            # coverage=8192, total_tokens=4096.
            adapter._promote_layerwise_store_result(request, result)
            state = adapter._worker_retrieve_state["req"]
            cache = state.cache_kwargs(group, True)
            assert cache["cached_ends"][-1] == end
            for layer, saved in enumerate(result.tensors):
                count = len(saved)
                assert all(
                    a is b
                    for a, b in zip(
                        cache["cached_tensors"][layer][-count:], saved, strict=True
                    )
                )
                assert torch.equal(
                    cache["cached_chunk_ptrs_npu"][layer][-count:],
                    result.chunk_ptrs[layer],
                )
                assert all(
                    a is b
                    for a, b in zip(
                        cache["cached_memory_objs"][layer][-count:],
                        result.memory_objs[layer],
                        strict=True,
                    )
                )
            if group == order[0] and start:
                # No stale prepared pointer table survives for the unfinished
                # common frontier; underlying group data is still retained.
                assert 1 not in state.prepared_sparse_sources
                assert adapter.lmcache_engine.retain_shared_cpu_store_seed.called
        state = adapter._worker_retrieve_state["req"]
        assert state.token_count == end
        assert set(state.prepared_sparse_sources) == {0, 1}
        for group, source in state.prepared_sparse_sources.items():
            assert source.total_tokens == end
            assert sum(source.chunk_token_counts) == end
            assert len(source.layers) == (2, 1)[group]
            assert source is not old_sources.get(group)
            assert source.validated_chunk_size == 256
        start = end
    adapter._release_shared_worker_retrieve_state.assert_not_called()


def test_single_group_chunked_prefill_still_seals_each_frontier(
    api: SimpleNamespace,
) -> None:
    adapter = make_adapter(api, groups=False)
    request = SimpleNamespace(req_id="req", is_sparse_decode=False)
    for start, end in ((0, 4096), (4096, 8192)):
        adapter._promote_layerwise_store_result(
            request, store_result(api, 0, start, end)
        )
        state = adapter._worker_retrieve_state["req"]
        assert set(state.prepared_sparse_sources) == {0}
        assert state.prepared_sparse_sources[0].total_tokens == end


def test_builder_still_rejects_incorrect_total_tokens(api: SimpleNamespace) -> None:
    result = store_result(api, 1, 0, 8192)
    with pytest.raises(ValueError, match="coverage=8192, total_tokens=4096"):
        api.build_prepared_sparse_source(
            result.tensors,
            result.chunk_ptrs,
            num_layers=1,
            total_tokens=4096,
            chunk_token_counts=[256] * 32,
            chunk_size=256,
        )


def test_overlapping_chunks_are_not_mistaken_for_a_group_ahead(
    api: SimpleNamespace,
) -> None:
    adapter = make_adapter(api)
    request = SimpleNamespace(req_id="req", is_sparse_decode=False)
    adapter._promote_layerwise_store_result(request, store_result(api, 0, 0, 4096))
    state = adapter._worker_retrieve_state["req"]
    state.cached_starts[1] = 0  # Same endpoint, but invalid duplicated coverage.
    with pytest.raises(ValueError, match="coverage.*total_tokens"):
        adapter._refresh_prepared_sparse_sources(state, 4096)
