# SPDX-License-Identifier: Apache-2.0
"""Run the production adapter's prefix lifetime rules without vLLM or an NPU."""

# Standard
import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

# Third Party
import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/vllm_v1_adapter.py"
)


def adapter_types() -> tuple[type, type]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    state = next(
        node for node in tree.body if getattr(node, "name", "") == "WorkerRetrieveState"
    )
    adapter = next(
        node
        for node in tree.body
        if getattr(node, "name", "") == "LMCacheConnectorV1Impl"
    )
    methods = {
        "_prepare_dense_prefix_retrieve_state",
        "_materialize_dense_prefix_for_sparse",
        "_worker_retrieve_state_for_request",
        "_worker_retrieve_state_for_warm_ref",
        "_should_invalidate_worker_retrieve_state",
        "_trim_dense_prefix_seed_for_sparse",
    }
    adapter.body = [
        node for node in adapter.body if getattr(node, "name", "") in methods
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, state, adapter], type_ignores=[])
    )
    namespace = dict(dataclass=dataclass, field=field, torch=torch)
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["LMCacheConnectorV1Impl"], namespace["WorkerRetrieveState"]


Adapter, State = adapter_types()


def request(**changes: Any) -> NS:
    values = dict(
        req_id="r",
        block_allocation_mode="prefill_child",
        is_sparse_decode=False,
        resumed_from_preemption=False,
        sparse_warm_ref=False,
        load_spec=NS(can_load=True, lmcache_cached_tokens=6),
    )
    values.update(changes)
    return NS(**values)


def populated_state() -> Any:
    state = State(
        req_id="r",
        dense_prefix_seed=True,
        dense_prefix_generation=7,
        metadata_warm=True,
        token_count=6,
        slot_mapping=torch.arange(6),
    )
    for group in (0, 1):
        cache = state.cache_kwargs(group, True)
        cache["cached_starts"][:] = [0, 4]
        cache["cached_ends"][:] = [4, 6]
        cache["cached_keys"][:] = [["full", "partial"] for _ in range(2)]
        cache["cached_memory_objs"][:] = [[object(), object()] for _ in range(2)]
        cache["cached_chunk_dev_ptrs"][:] = [[101, 102], [201, 202]]
        cache["cached_chunk_ptrs_npu"][:] = [None, None]
    return state


def adapter(state: Any) -> tuple[Any, list, list, list]:
    obj = Adapter()
    released, uploads, refreshed = [], [], []

    def materialize(rows: list, pointers: list, *, kv_group: int) -> None:
        uploads.append((kv_group, rows))
        pointers[:] = list(torch.tensor(rows, dtype=torch.long).unbind(0))

    def refresh(value: Any, count: int) -> None:
        refreshed.append((value, count))
        value.prepared_sparse_sources = {
            0: NS(total_tokens=count),
            1: NS(total_tokens=count),
        }

    obj.kv_role = "kv_both"
    obj._layerwise_prefill_p_node = True
    obj._deferred_layerwise_prefill_load_active = True
    obj._worker_retrieve_state = {"r": state}
    obj.lmcache_engine = NS(
        shared_cpu_cache_generation=7,
        gpu_connector=NS(materialize_sparse_chunk_ptr_cache=materialize),
    )
    obj._release_shared_worker_retrieve_state = lambda value, engine: released.append(
        value
    )
    obj._is_dsa_two_groups = lambda: True
    obj._num_layers_for_group = lambda group: 2
    obj._refresh_prepared_sparse_sources = refresh
    obj._validate_shared_worker_retrieve_state = lambda value, req: None
    obj._prepared_sparse_source = lambda value, group, count: (
        value.prepared_sparse_sources.get(group)
    )
    return obj, released, uploads, refreshed


def prepare(obj: Any, state: Any, req: Any = None, **changes: Any) -> tuple[Any, bool]:
    options = dict(retain_dense_seed=True, dsa_two_groups=True, token_count=8)
    options.update(changes)
    return obj._prepare_dense_prefix_retrieve_state(req or request(), state, **options)


def test_continuous_p_prefill_keeps_both_source_caches_and_partial_tail() -> None:
    state = populated_state()
    obj, released, uploads, _ = adapter(state)
    caches = [state.cache_kwargs(group, True) for group in (0, 1)]
    for next_frontier in (6, 8, 12):
        result, reuse = prepare(obj, state, token_count=next_frontier)
        assert reuse and result is state
        for group, original in enumerate(caches):
            current = result.cache_kwargs(group, True)
            assert all(current[name] is value for name, value in original.items())
        state.token_count = next_frontier
    assert not released and not uploads


@pytest.mark.parametrize(
    "reason", ["generation", "preemption", "shrink", "request", "untracked_seed"]
)
def test_invalid_continuation_starts_with_an_empty_adoption_target(reason: str) -> None:
    state = populated_state()
    obj, released, uploads, _ = adapter(state)
    req = request()
    count = 8
    if reason == "generation":
        obj.lmcache_engine.shared_cpu_cache_generation = 8
    elif reason == "preemption":
        req.resumed_from_preemption = True
    elif reason == "shrink":
        count = 4
    elif reason == "request":
        state.req_id = "previous"
    else:
        state.dense_prefix_generation = None
    result, reuse = prepare(obj, state, req, token_count=count)
    assert reuse and result is not state
    assert result.req_id == req.req_id
    assert (
        result.dense_prefix_generation == obj.lmcache_engine.shared_cpu_cache_generation
    )
    assert not result.group_has_data(0, True)
    assert not result.group_has_data(1, True)
    assert released == [state] and not uploads


@pytest.mark.parametrize(
    "mode", ["d_node", "consumer", "ordinary_dense", "ordinary_layout", "sparse"]
)
def test_other_paths_keep_the_original_replace_semantics(mode: str) -> None:
    state = populated_state()
    obj, released, uploads, _ = adapter(state)
    req = request()
    if mode == "d_node":
        obj._layerwise_prefill_p_node = False
    elif mode == "consumer":
        obj.kv_role = "kv_consumer"
    elif mode == "ordinary_dense":
        obj._deferred_layerwise_prefill_load_active = False
    elif mode == "ordinary_layout":
        req.block_allocation_mode = None
    else:
        req.is_sparse_decode = True
    result, reuse = prepare(obj, state, req)
    assert not reuse and result is not state
    assert result.dense_prefix_generation is None
    assert released == [state] and not uploads


def test_disabled_retention_does_not_change_the_existing_state() -> None:
    state = populated_state()
    obj, released, uploads, _ = adapter(state)
    result, reuse = prepare(obj, state, retain_dense_seed=False)
    assert result is state and not reuse
    assert not released and not uploads


def test_first_sparse_consumer_materializes_each_group_once_before_validation() -> None:
    state = populated_state()
    obj, _, uploads, refreshed = adapter(state)

    def validate(value: Any, req: Any) -> None:
        assert all(isinstance(row, torch.Tensor) for row in value.cached_chunk_ptrs_npu)
        assert all(
            isinstance(row, torch.Tensor) for row in value.cached_chunk_ptrs_npu_indexer
        )

    obj._validate_shared_worker_retrieve_state = validate
    req = request(is_sparse_decode=True)
    assert obj._worker_retrieve_state_for_request(req) is state
    assert [group for group, _ in uploads] == [0, 1]
    assert refreshed == [(state, 6)]
    assert obj._worker_retrieve_state_for_warm_ref(req) is state
    assert len(uploads) == 2 and len(refreshed) == 1


def test_dense_consumption_does_not_materialize_pointers() -> None:
    state = populated_state()
    obj, _, uploads, refreshed = adapter(state)
    assert obj._worker_retrieve_state_for_request(request()) is state
    assert not uploads and not refreshed


def test_ordinary_sparse_without_opt_in_keeps_original_validation() -> None:
    state = populated_state()
    state.dense_prefix_generation = None
    obj, _, uploads, refreshed = adapter(state)
    validated = []
    obj._validate_shared_worker_retrieve_state = lambda *args: validated.append(args)
    assert (
        obj._worker_retrieve_state_for_request(request(is_sparse_decode=True)) is state
    )
    assert len(validated) == 1 and not uploads and not refreshed


def test_generation_change_invalidates_before_sparse_use() -> None:
    state = populated_state()
    obj, _, uploads, _ = adapter(state)
    obj.lmcache_engine.shared_cpu_cache_generation = 8
    req = request(is_sparse_decode=True)
    assert not obj._trim_dense_prefix_seed_for_sparse(state, 4)
    assert state.cached_ends == [4, 6]
    assert obj._should_invalidate_worker_retrieve_state(req, 6)
    with pytest.raises(RuntimeError, match="stale shared dense prefix"):
        obj._worker_retrieve_state_for_warm_ref(req)
    assert not uploads


def test_raw_prefill_missing_materialization_capability_fails_explicitly() -> None:
    state = populated_state()
    obj, _, _, _ = adapter(state)
    obj.lmcache_engine.gpu_connector = NS()
    with pytest.raises(RuntimeError, match="materialization API"):
        obj._worker_retrieve_state_for_request(request(is_sparse_decode=True))


def test_both_dense_groups_receive_the_same_reuse_protocol() -> None:
    """Check integration wiring in addition to the executed lifecycle methods."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_load_kv"
    )
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "retrieve_layer"
        and any(
            keyword.arg == "_retain_shared_dense_cache" for keyword in node.keywords
        )
    ]
    assert len(calls) == 2
    assert all(
        any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "deferred_prefill_kwargs"
            for keyword in call.keywords
        )
        for call in calls
    )
    assert any(
        isinstance(node, ast.Assign)
        and ast.unparse(node)
        == "deferred_prefill_kwargs['_reuse_shared_dense_prefix'] = True"
        for node in ast.walk(function)
    )
