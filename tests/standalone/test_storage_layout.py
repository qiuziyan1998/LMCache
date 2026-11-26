# SPDX-License-Identifier: Apache-2.0
"""Frozen layout derives from actual registered group metadata on CPU."""

from types import SimpleNamespace

import msgspec
import pytest
import torch

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc.storage_layout import freeze_storage_layout


def fixture(tp_size=4):
    cfg = LMCacheEngineConfig.from_defaults()
    cfg.dsa_two_groups = True
    cfg.chunk_size = 1024
    meta = LMCacheMetadata(
        "model",
        tp_size,
        tp_size,
        0,
        0,
        torch.bfloat16,
        (3, 1, 1024, 1, 576),
        use_mla=True,
        runtime_kv_group_layer_counts=(3, 2),
        runtime_kv_group_layer_names=(("m0", "m1", "mtp"), ("i0", "i2")),
        indexer_c8_layout=IndexerC8Layout(c8_layers=(True, False)),
    )
    meta.kv_layer_groups_manager.kv_layer_groups = [
        KVLayerGroupInfo(
            ["m0", "m1", "mtp"], [0, 1, 2], torch.Size([10, 16, 576]), torch.bfloat16
        ),
        KVLayerGroupInfo(
            ["i0", "i2"], [0, 2], torch.Size([10, 16, 128]), torch.bfloat16
        ),
    ]
    database = SimpleNamespace(
        chunk_hash_type=bytes,
        chunk_hash_bytes=32,
        save_only_first_rank=True,
        mooncake_payload_layout="source-abi",
    )
    return cfg, meta, database


def test_registered_unequal_groups_and_mixed_c8_scales_keep_exact_bytes():
    cfg, meta, database = fixture()
    layout = freeze_storage_layout(cfg, meta, database)
    assert tuple(len(group.layer_names) for group in layout.groups) == (3, 2)
    assert layout.groups[0].plane_token_bytes == ((1152,),) * 3
    assert layout.groups[1].plane_token_bytes == ((128, 2), (256,))
    assert layout.groups[1].memory_format == MemoryFormat.KV_DSA_INDEX_FMT.value
    assert msgspec.msgpack.decode(layout.precision_descriptor)["c8_layers"] == [
        True,
        False,
    ]
    assert b"indexer_hbm_block_map" not in msgspec.msgpack.encode(layout)


def test_replicated_source_identity_is_independent_of_destination_tp_size():
    layout4 = freeze_storage_layout(*fixture(4))
    layout2 = freeze_storage_layout(*fixture(2))
    assert layout4 == layout2
    assert layout4.source_tp_size == 1


def test_missing_or_conflicting_runtime_registration_rejected():
    cfg, meta, database = fixture()
    meta.runtime_kv_group_layer_counts = (3, 3)
    with pytest.raises(ValueError):
        freeze_storage_layout(cfg, meta, database)
    meta.runtime_kv_group_layer_names = None
    with pytest.raises(ValueError):
        freeze_storage_layout(cfg, meta, database)


def test_registered_latent_rope_plane_split_and_allocator_alignment_are_preserved():
    layout = freeze_storage_layout(
        *fixture(),
        allocator_alignment=4096,
        plane_token_bytes={0: (1024, 128), 1: ((128, 2), (256,))},
    )
    assert layout.groups[0].plane_token_bytes == ((1024, 128),) * 3
    assert layout.groups[1].plane_token_bytes == ((128, 2), (256,))
    assert layout.allocator_alignment == 4096


@pytest.mark.parametrize(
    "planes",
    [
        {0: (1024, 130)},
        {0: ((1024, 128),)},
        {1: ((130,), (256,))},
        {0: (True, 1151)},
        {2: (1152,)},
    ],
)
def test_unproved_registered_plane_geometry_is_rejected(planes):
    with pytest.raises(ValueError, match="plane"):
        freeze_storage_layout(*fixture(), plane_token_bytes=planes)
