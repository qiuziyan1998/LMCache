# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.sparse import build_prepared_sparse_source
from lmcache.v1.memory_management import MemoryObj


def test_build_prepared_sparse_source_seals_complete_layers() -> None:
    tensors = [
        [torch.zeros(4), torch.ones(2)],
        [torch.full((4,), 2), torch.full((2,), 3)],
    ]
    pointer_tables = [
        torch.tensor([101, 102], dtype=torch.int64),
        torch.tensor([201, 202], dtype=torch.int64),
    ]

    source = build_prepared_sparse_source(
        tensors,
        pointer_tables,
        num_layers=2,
        total_tokens=6,
        chunk_token_counts=(4, 2),
        expected_pointer_device=torch.device("cpu"),
    )

    assert source is not None
    assert source.total_tokens == 6
    assert source.layers[0].tensors == tuple(tensors[0])
    assert source.layers[1].chunk_ptrs_npu is pointer_tables[1]
    assert source.chunk_token_counts == (4, 2)
    assert source.pointer_device == torch.device("cpu")


def test_build_prepared_sparse_source_waits_for_complete_bootstrap() -> None:
    source = build_prepared_sparse_source(
        [[torch.zeros(4)], []],
        [torch.tensor([101], dtype=torch.int64), None],
        num_layers=2,
        total_tokens=4,
    )

    assert source is None


def test_build_prepared_sparse_source_accepts_memory_obj_owners() -> None:
    owners = [
        [MagicMock(spec=MemoryObj), MagicMock(spec=MemoryObj)],
        [MagicMock(spec=MemoryObj), MagicMock(spec=MemoryObj)],
    ]
    pointer_tables = [
        torch.tensor([101, 102], dtype=torch.int64),
        torch.tensor([201, 202], dtype=torch.int64),
    ]

    source = build_prepared_sparse_source(
        [],
        pointer_tables,
        num_layers=2,
        total_tokens=6,
        chunk_token_counts=(4, 2),
        cached_memory_objs=owners,
    )

    assert source is not None
    assert source.layers[0].tensors == ()
    assert source.layers[0].memory_objs == tuple(owners[0])
    assert source.layers[1].chunk_ptrs_npu is pointer_tables[1]


def test_build_prepared_sparse_source_rejects_owner_pointer_mismatch() -> None:
    with pytest.raises(ValueError, match="pointer coverage"):
        build_prepared_sparse_source(
            [],
            [torch.tensor([101], dtype=torch.int64)],
            num_layers=1,
            total_tokens=6,
            cached_memory_objs=[
                [MagicMock(spec=MemoryObj), MagicMock(spec=MemoryObj)]
            ],
        )


def test_build_prepared_sparse_source_rejects_partial_pointer_coverage() -> None:
    with pytest.raises(ValueError, match="pointer coverage"):
        build_prepared_sparse_source(
            [[torch.zeros(4), torch.ones(2)]],
            [torch.tensor([101], dtype=torch.int64)],
            num_layers=1,
            total_tokens=6,
        )


def test_build_prepared_sparse_source_waits_for_token_coverage() -> None:
    source = build_prepared_sparse_source(
        [[torch.zeros(4)]],
        [torch.tensor([101], dtype=torch.int64)],
        num_layers=1,
        total_tokens=4,
        chunk_token_counts=(3,),
    )

    assert source is None


@pytest.mark.parametrize("chunk_counts", [(4, 3), (3, 4, 1)])
def test_build_prepared_sparse_source_rejects_invalid_tail_coverage(
    chunk_counts: tuple[int, ...],
) -> None:
    chunks = len(chunk_counts)
    with pytest.raises(ValueError, match="coverage|non-tail chunks"):
        build_prepared_sparse_source(
            [[torch.empty(1)] * chunks],
            [torch.arange(chunks, dtype=torch.int64)],
            num_layers=1,
            total_tokens=6 if chunks == 2 else 8,
            chunk_token_counts=chunk_counts,
        )


def test_build_prepared_sparse_source_rejects_wrong_pointer_device() -> None:
    with pytest.raises(ValueError, match="wrong device"):
        build_prepared_sparse_source(
            [[torch.zeros(4)]],
            [torch.tensor([101], dtype=torch.int64)],
            num_layers=1,
            total_tokens=4,
            chunk_token_counts=(4,),
            expected_pointer_device=torch.device("cuda"),
        )


def test_build_prepared_sparse_source_rejects_noncontiguous_pointer_table() -> None:
    with pytest.raises(ValueError, match="must be contiguous"):
        build_prepared_sparse_source(
            [[torch.zeros(4), torch.ones(2)]],
            [torch.tensor([101, 0, 102, 0], dtype=torch.int64)[::2]],
            num_layers=1,
            total_tokens=6,
        )
