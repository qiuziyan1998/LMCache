# SPDX-License-Identifier: Apache-2.0
"""Page-first layout validation must reject ABI mismatches without native I/O."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)


@pytest.fixture
def connector() -> Iterator[MooncakestoreConnector]:
    result = object.__new__(MooncakestoreConnector)
    result._page_first_multi_buffer = True
    result.save_chunk_meta = False
    result.store = Mock()
    result._dsa_raw_token_dims = {0: 576, 1: 128}
    result.local_cpu_backend = SimpleNamespace(
        config=SimpleNamespace(dsa_two_groups=True),
        metadata=SimpleNamespace(
            model_name="test",
            chunk_size=256,
            kv_shape=(79, 1, 256, 1, 576),
            runtime_kv_group_layer_counts=(79, 22),
        ),
    )
    result.meta_shapes = [torch.Size([1, 1, 256, 576])]
    result.meta_dtypes = [torch.bfloat16]
    result.meta_fmt = MemoryFormat.KV_MLA_LATENT_FMT
    result.single_token_size = 576 * 2
    yield result
    assert not result.store.mock_calls


@pytest.mark.parametrize("group,count,width", [(0, 79, 576), (1, 22, 128)])
def test_page_layout_matches_live_raw_metadata(
    connector: MooncakestoreConnector, group: int, count: int, width: int
) -> None:
    fmt = (
        MemoryFormat.KV_DSA_INDEX_FMT if group else MemoryFormat.KV_MLA_LATENT_FMT
    )
    connector.validate_page_first_layout(
        group, count, torch.Size([256 * width]), torch.bfloat16, fmt
    )


@pytest.mark.parametrize(
    "mismatch",
    [
        "width",
        "cardinality",
        "raw_dtype",
        "dtype",
        "format",
        "token_bytes",
        "legacy_shape",
        "singleton_shape",
        "tail",
        "chunk_size",
        "disabled",
    ],
)
def test_page_layout_rejects_mismatch(
    connector: MooncakestoreConnector, mismatch: str
) -> None:
    shape = torch.Size([256 * 576])
    dtype = torch.bfloat16
    fmt = MemoryFormat.KV_MLA_LATENT_FMT
    if mismatch == "width":
        connector._dsa_raw_token_dims[0] = 512
    elif mismatch == "cardinality":
        connector.local_cpu_backend.metadata.runtime_kv_group_layer_counts = (78, 22)
    elif mismatch == "raw_dtype":
        connector.meta_dtypes = [torch.float16]
    elif mismatch == "dtype":
        dtype = torch.float16
    elif mismatch == "format":
        fmt = MemoryFormat.KV_DSA_INDEX_FMT
    elif mismatch == "token_bytes":
        connector._dsa_raw_token_dims.clear()
        connector.meta_shapes = [shape]
        connector.single_token_size += 2
    elif mismatch == "legacy_shape":
        connector._dsa_raw_token_dims.clear()
    elif mismatch == "singleton_shape":
        shape = torch.Size([1, 256 * 576])
    elif mismatch == "tail":
        shape = torch.Size([255 * 576])
    elif mismatch == "chunk_size":
        connector.local_cpu_backend.metadata.chunk_size = 128
    else:
        connector._page_first_multi_buffer = False
    with pytest.raises(ValueError, match="Page-first layout"):
        connector.validate_page_first_layout(0, 79, shape, dtype, fmt)


def test_base_connector_rejects_unvalidated_page_layout() -> None:
    with pytest.raises(ValueError, match="does not support"):
        RemoteConnector.validate_page_first_layout(
            SimpleNamespace(),
            0,
            79,
            torch.Size([256 * 576]),
            torch.bfloat16,
            MemoryFormat.KV_MLA_LATENT_FMT,
        )


def test_instrumented_connector_delegates_layout_validation(
    connector: MooncakestoreConnector,
) -> None:
    wrapped = object.__new__(InstrumentedRemoteConnector)
    wrapped._connector = connector
    shape = torch.Size([256 * 576])
    wrapped.validate_page_first_layout(
        0, 79, shape, torch.bfloat16, MemoryFormat.KV_MLA_LATENT_FMT
    )
    connector._dsa_raw_token_dims[0] = 512
    with pytest.raises(ValueError, match="Page-first layout mismatch"):
        wrapped.validate_page_first_layout(
            0, 79, shape, torch.bfloat16, MemoryFormat.KV_MLA_LATENT_FMT
        )
