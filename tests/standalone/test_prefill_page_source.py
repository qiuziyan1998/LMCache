# SPDX-License-Identifier: Apache-2.0
"""Check local and remote page resolution on production resolver branches."""

# Standard
import ast
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

# Third Party
import pytest


@dataclass(frozen=True)
class Key:
    chunk: int

    def split_layers(self, count: int) -> list:
        return [(self.chunk, layer) for layer in range(count)]


class Page:
    num_layers = 2

    def get_shape(self) -> tuple:
        return (4,)

    def get_size(self) -> int:
        return 16

    @staticmethod
    def pin_many(pages: list) -> bool:
        return True


def resolver(local_positions: set[int]) -> tuple:
    """Load actual resolver methods, with only backend I/O substituted."""
    path = Path(__file__).resolve().parents[2] / "lmcache/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if getattr(node, "name", "") == "LMCacheEngine"
    )
    names = {
        "_resolve_shared_rank0_layer_pages",
        "_resolve_shared_rank0_exact_layer_pages",
    }
    nodes = [node for node in cls.body if getattr(node, "name", "") in names]
    module = ast.parse("from __future__ import annotations")
    module.body.extend(nodes)
    trace, io = [], []
    runtime = dict(
        LayerPageMemoryObj=Page,
        LayerCacheEngineKey=Key,
        mooncake_valid_tokens=lambda key, size: size,
    )
    exec(compile(module, str(path), "exec"), runtime)
    pages = [Page() for _ in range(3)]

    def get_local(keys):
        io.append(("local", tuple(keys)))
        result = []
        for key in keys:
            if key.chunk not in local_positions:
                break
            result.append(pages[key.chunk])
        return result, len(result)

    def contains_remote(keys):
        result = 0
        for key in keys:
            if key.chunk in local_positions:
                break
            result += 1
        return result

    def get_remote(keys):
        io.append(("remote", tuple(keys)))
        return [pages[key.chunk] for key in keys]

    local = NS(
        batched_get_layer_page_prefix=get_local, contains_all_exact=lambda keys: False
    )
    remote = NS(
        batched_contains_layer_pages=contains_remote, batched_get_layer_pages=get_remote
    )
    engine = NS(
        _force_layerwise_prefill_store=True,
        config=NS(chunk_size=4),
        metadata=NS(worker_id=0),
        _shared_local_cpu_backend=lambda: local,
        storage_manager=NS(storage_backends={"RemoteBackend": remote}),
        num_layers_for_group=lambda group: 2,
        _expected_shared_cpu_chunk_metadata=lambda **kwargs: ((4,),),
        _validate_rank0_shared_mem_obj=lambda *args, **kwargs: None,
        _resolve_shared_rank0_layer_mem_objs=lambda **kwargs: [],
    )
    for name in names:
        setattr(engine, name, MethodType(runtime[name], engine))
    return engine, pages, trace, io


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("local_positions", [set(), {0, 1, 2}, {0, 2}])
def test_page_resolution_preserves_backend_order(exact, local_positions):
    engine, pages, trace, _ = resolver(local_positions)
    keys = [Key(index) for index in range(3)]
    resolved, count = engine._resolve_shared_rank0_layer_pages(
        req_id="followup",
        phase="prefill",
        kv_group=0,
        keys_layer_major=[keys, keys],
        page_chunks=3,
        base_page_keys=keys,
        exact_chunk_locations=[
            "LocalCPUBackend" if index in local_positions else "RemoteBackend"
            for index in range(3)
        ]
        if exact
        else None,
    )
    assert count == 3 and resolved == [pages, pages]
    assert trace == []
