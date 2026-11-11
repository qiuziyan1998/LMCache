# SPDX-License-Identifier: Apache-2.0
"""Canonical coverage and physical addresses of cooperative DCP writes."""

import importlib.util
from pathlib import Path
import sys

import pytest

path = Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/dcp.py"
spec = importlib.util.spec_from_file_location("dcp_write_layout_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
DcpWriteLayout = module.DcpWriteLayout


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("origin", [0, 32])
@pytest.mark.parametrize(
    "start,end",
    [(0, 0), (0, 1), (127, 129), (129, 1025), (2048, 4097), (1048500, 1048576)],
)
def test_disjoint_complete_canonical_coverage(world, origin, start, end):
    block = 128
    columns = (end + block * world - 1) // (block * world)
    ids = tuple(range(20, 20 + origin + columns))
    layout = DcpWriteLayout(world, block, ids, origin)
    actual = {}
    for rank in range(world):
        for first, slot, count in layout.runs(start, end, rank):
            assert 0 < count <= block
            for delta in range(count):
                token = first + delta
                assert token not in actual
                assert token // block % world == rank
                expected = (
                    ids[origin + token // (block * world)] * block + token % block
                )
                assert slot + delta == expected
                actual[token] = rank
    assert sorted(actual) == list(range(start, end))


@pytest.mark.parametrize("blocks,error", [((), "allocated"), ((0,), "null")])
def test_missing_source_is_not_clamped(blocks, error):
    with pytest.raises(ValueError, match=error):
        tuple(DcpWriteLayout(8, 128, blocks).runs(0, 1, 0))


def test_empty_owner_does_not_need_a_source_block():
    assert tuple(DcpWriteLayout(8, 128, ()).runs(0, 1, 7)) == ()
