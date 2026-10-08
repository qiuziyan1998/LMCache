# SPDX-License-Identifier: Apache-2.0
"""Exercise store/load/store pointer ownership with real CPU tensors.

Extract the production state methods to avoid importing vLLM or NPU kernels.
The merge boundary is tested directly because it owns the pointer-table
invariant; the dense adoption method is executed without a storage backend.
"""

# Standard
import ast
from dataclasses import dataclass, field
import logging
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

# Third Party
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def production_types() -> tuple[type, type, type, type]:
    namespace = dict(
        dataclass=dataclass,
        field=field,
        torch=torch,
        logger=logging.getLogger(__name__),
    )
    selections = {
        "lmcache/integration/vllm/vllm_v1_adapter.py": {
            "LayerwisePointerTable": None,
            "WorkerRetrieveState": None,
            "LMCacheConnectorV1Impl": {
                "_ensure_layer_cache_shape",
                "_merge_cache_group_by_ranges",
                "_merge_store_result_into_worker_state",
            },
        },
        "lmcache/v1/cache_engine.py": {
            "LMCacheEngine": {"_adopt_dense_shared_retrieve_cache"},
        },
    }
    for relative, classes in selections.items():
        source = ROOT / relative
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in classes:
                continue
            methods = classes[node.name]
            if methods is not None:
                node.body = [
                    item for item in node.body if getattr(item, "name", "") in methods
                ]
            nodes.append(node)
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future, *nodes], type_ignores=[])
        )
        exec(compile(module, str(source), "exec"), namespace)
    return tuple(
        namespace[name]
        for name in (
            "LMCacheConnectorV1Impl",
            "WorkerRetrieveState",
            "LayerwisePointerTable",
            "LMCacheEngine",
        )
    )


Adapter, State, PointerTable, Engine = production_types()


def connector(*, require_pointers: bool = False) -> Any:
    obj = Adapter()
    obj._layerwise_prefill_p_node = True
    obj._is_dsa_two_groups = lambda: True
    obj._is_decode_window_save_request = lambda req: require_pointers
    obj.lmcache_engine = NS(enable_shared_cpu_cache=True)
    return obj


def store_result(
    group: int, starts: list[int], ends: list[int], pointers: torch.Tensor
) -> NS:
    layers, chunks = pointers.shape
    assert len(starts) == len(ends) == chunks
    return NS(
        kv_group=group,
        starts=starts,
        ends=ends,
        keys=[
            [f"{layer}:{start}:{end}" for start, end in zip(starts, ends, strict=True)]
            for layer in range(layers)
        ],
        memory_objs=[[object() for _ in starts] for _ in range(layers)],
        tensors=[],
        chunk_dev_ptrs=pointers.tolist(),
        chunk_ptrs=list(pointers.unbind(0)),
    )


def adopt_dense_prefix(
    state: Any, group: int, pointers: torch.Tensor, *, deferred: bool
) -> None:
    cache = state.cache_kwargs(group, True)
    layers = pointers.shape[0]
    engine = Engine()
    engine.supports_dense_sparse_cache_retention = lambda: True
    engine.num_layers_for_group = lambda group: layers
    engine.register_shared_cpu_sparse_request = lambda *args, **kwargs: None
    engine.get_shared_cpu_request_lease = lambda req: None

    # Output of the load pointer preparer: raw DMA has host addresses and
    # deferred device rows, while a materialized load publishes new row views.
    cache["cached_chunk_dev_ptrs"][:] = pointers.tolist()
    cache["cached_chunk_ptrs_npu"][:] = (
        [None] * layers if deferred else list(pointers.unbind(0))
    )
    adopted = engine._adopt_dense_shared_retrieve_cache(
        req_id="r",
        starts=[0, 1024, 2048, 3072],
        ends=[1024, 2048, 3072, 3979],
        keys_layer_major=[[f"loaded:{i}" for i in range(4)] for _ in range(layers)],
        memory_objs=[[object() for _ in range(4)] for _ in range(layers)],
        handles=[[object() for _ in range(4)] for _ in range(layers)],
        kv_group=group,
        kwargs=dict(
            cache,
            _retain_shared_dense_cache=True,
            _reuse_shared_dense_prefix=True,
            deferred_layerwise_get=True,
            prefill_dma_block_ids_by_bank=((1,),),
        ),
    )
    assert adopted


@pytest.mark.parametrize(("group", "layers"), [(0, 79), (1, 22)])
@pytest.mark.parametrize("deferred", [True, False])
@pytest.mark.parametrize("seed_chunks", [1, 4])
def test_store_load_store_replaces_partial_tail_using_current_prefix(
    group: int, layers: int, deferred: bool, seed_chunks: int
) -> None:
    obj, state, req = connector(), State(req_id="r"), NS(req_id="r")
    starts = [3072] if seed_chunks == 1 else [0, 1024, 2048, 3072]
    ends = [3979] if seed_chunks == 1 else [1024, 2048, 3072, 3979]
    seed = store_result(
        group, starts, ends, torch.arange(layers * seed_chunks).reshape(layers, -1)
    )
    assert obj._merge_store_result_into_worker_state(state, seed, req) == seed_chunks
    assert state.pointer_tables[group].length == seed_chunks

    loaded = torch.arange(layers * 4).reshape(layers, 4) + 10000
    adopt_dense_prefix(state, group, loaded, deferred=deferred)
    cache = state.cache_kwargs(group, True)
    prefix_owners = [row[:3] for row in cache["cached_memory_objs"]]
    old_tail = [row[3] for row in cache["cached_memory_objs"]]
    suffix = torch.arange(layers * 17).reshape(layers, 17) + 20000
    result = store_result(
        group,
        list(range(3072, 20363, 1024)),
        list(range(4096, 20363, 1024)) + [20363],
        suffix,
    )
    assert obj._merge_store_result_into_worker_state(state, result, req) == 17
    assert cache["cached_starts"] == list(range(0, 20363, 1024))
    assert cache["cached_ends"] == list(range(1024, 20363, 1024)) + [20363]
    assert [row[:3] for row in cache["cached_memory_objs"]] == prefix_owners
    assert all(
        row[3] is not tail
        for row, tail in zip(cache["cached_memory_objs"], old_tail, strict=True)
    )
    expected = torch.cat((loaded[:, :3], suffix), dim=1)
    assert cache["cached_chunk_dev_ptrs"] == expected.tolist()
    if deferred:
        assert cache["cached_chunk_ptrs_npu"] == [None] * layers
        assert state.pointer_tables[group].table is None
    else:
        assert torch.equal(torch.stack(cache["cached_chunk_ptrs_npu"]), expected)
        assert state.pointer_tables[group].length == 20
    assert not state.group_has_data(1 - group, True)
    # A duplicate completion must not append the same suffix again.
    assert obj._merge_store_result_into_worker_state(state, result, req) == 0


def test_required_pointer_cache_rejects_deferred_prefix_without_mutation() -> None:
    state, req = State(req_id="r"), NS(req_id="r")
    seed = store_result(0, [3072], [3979], torch.tensor([[42]]))
    connector()._merge_store_result_into_worker_state(state, seed, req)
    old_table = state.pointer_tables[0].table
    adopt_dense_prefix(state, 0, torch.tensor([[10, 20, 30, 40]]), deferred=True)
    cache = state.cache_kwargs(0, True)
    owners = list(cache["cached_memory_objs"][0])
    result = store_result(0, [3072], [4096], torch.tensor([[50]]))
    assert (
        connector(require_pointers=True)._merge_store_result_into_worker_state(
            state, result, req
        )
        == 0
    )
    assert cache["cached_ends"] == [1024, 2048, 3072, 3979]
    assert cache["cached_memory_objs"][0] == owners
    assert cache["cached_chunk_ptrs_npu"] == [None]
    assert state.pointer_tables[0].table is old_table
    assert state.pointer_tables[0].length == 1


def test_continuous_store_growth_reuses_backing_capacity() -> None:
    obj, state, req = connector(), State(req_id="r"), NS(req_id="r")
    for index in range(12):
        table = state.pointer_tables.get(0)
        prior_storage = None if table is None else table.table
        prior_capacity = 0 if table is None else table.capacity
        result = store_result(
            0,
            [index * 1024],
            [(index + 1) * 1024],
            torch.tensor([[index + 100], [index + 200]]),
        )
        assert obj._merge_store_result_into_worker_state(state, result, req) == 1
        if prior_capacity >= index + 1:
            assert state.pointer_tables[0].table is prior_storage
        expected = torch.stack(
            (torch.arange(index + 1) + 100, torch.arange(index + 1) + 200)
        )
        assert torch.equal(torch.stack(state.cached_chunk_ptrs_npu), expected)


def test_truncate_still_rejects_extension() -> None:
    table = PointerTable()
    table.append([torch.tensor([42])], [], 0)
    with pytest.raises(ValueError, match="Cannot extend pointer table by truncation"):
        table.truncate(3)


def test_growing_tail_keeps_current_prefix_and_reuses_its_table() -> None:
    obj, state, req = connector(), State(req_id="r"), NS(req_id="r")
    seed = store_result(0, [0, 1024], [1024, 1931], torch.tensor([[10, 11], [20, 21]]))
    assert obj._merge_store_result_into_worker_state(state, seed, req) == 2
    storage = state.pointer_tables[0].table
    owners = [row[0] for row in state.cached_memory_objs]
    result = store_result(0, [1024], [2048], torch.tensor([[12], [22]]))
    assert obj._merge_store_result_into_worker_state(state, result, req) == 1
    assert state.pointer_tables[0].table is storage
    assert [row[0] for row in state.cached_memory_objs] == owners
    assert state.cached_starts == [0, 1024]
    assert state.cached_ends == [1024, 2048]
    assert torch.equal(
        torch.stack(state.cached_chunk_ptrs_npu), torch.tensor([[10, 12], [20, 22]])
    )
