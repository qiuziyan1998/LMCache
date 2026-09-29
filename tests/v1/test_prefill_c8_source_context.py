# SPDX-License-Identifier: Apache-2.0
"""Trusted P source identity and incomplete-group rejection before yielding."""

# Standard
import ast
from pathlib import Path
from types import SimpleNamespace as NS

# Third Party
import pytest

# First Party
from lmcache.v1.indexer_c8 import IndexerC8Layout


@pytest.fixture
def source_nodes():
    path = Path(__file__).resolve().parents[2] / "lmcache/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    engine = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheEngine"
    )
    return engine.body


def compile_nodes(nodes):
    module = ast.parse("from __future__ import annotations")
    module.body.extend(nodes)
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(module), "<actual engine nodes>", "exec"),
        namespace,
    )
    return namespace


def test_full_mixed_policy_changes_trusted_source_identity(source_nodes):
    node = next(
        n
        for n in source_nodes
        if getattr(n, "name", "") == "_shared_prefill_source_context"
    )
    context = compile_nodes([node])[node.name]
    first = IndexerC8Layout(128, (True, False, True))
    changed = IndexerC8Layout(128, (True, True, False))
    engine = NS(
        shared_cpu_cache_generation=7,
        shared_cpu_cache_name="slab",
        shared_cpu_cache_passive_allocator=object(),
        num_layers_for_group=lambda _: 3,
        _expected_shared_cpu_chunk_metadata=lambda **_: ((130,), "uint8", "index"),
        metadata=NS(indexer_c8_layout=first),
    )
    original = context(engine, 1)
    latent = context(engine, 0)
    engine.metadata.indexer_c8_layout = changed
    assert context(engine, 1) != original
    assert context(engine, 0) == latent
    assert sum(first.layer_bytes(127, i) for i in range(3)) == sum(
        changed.layer_bytes(127, i) for i in range(3)
    )
    engine.metadata.indexer_c8_layout = IndexerC8Layout(128, first.c8_layers)
    assert context(engine, 1) == original
    engine.metadata.indexer_c8_layout = None
    assert context(engine, 1) == original[:-1]
    assert len(context(engine, 1)) == 7


@pytest.mark.parametrize(
    "packets,rows,sources,reject",
    [
        (True, [[1]], [[1], None], True),
        (True, [[1]], [None, [1]], True),
        (True, [[1]], [[1]], True),
        (True, [], [None, None], False),
        (True, [[1], [2]], [[1], [2]], False),
        (False, [[1]], [[1], None], False),
    ],
)
def test_incomplete_packet_guard_precedes_first_success_yield(
    source_nodes, packets, rows, sources, reject
):
    guard = next(
        n
        for method in source_nodes
        for n in ast.walk(method)
        if isinstance(n, ast.If)
        and any(
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and child.value.startswith("Incomplete shared C8 prefill source group")
            for statement in n.body
            if isinstance(statement, ast.Raise)
            for child in ast.walk(statement)
        )
    )
    wrapper = ast.parse(
        "def check(self, prepare_c8_packets, resolved_layers, "
        "prepared_sources, kv_group):\n"
        "    yield 'success'\n"
    ).body[0]
    wrapper.body.insert(0, guard)
    check = compile_nodes([wrapper])["check"]
    engine = NS(num_layers_for_group=lambda _: 2)
    stream = check(engine, packets, rows, sources, 1)
    if reject:
        with pytest.raises(RuntimeError, match="Incomplete shared C8"):
            next(stream)
    else:
        assert next(stream) == "success"
