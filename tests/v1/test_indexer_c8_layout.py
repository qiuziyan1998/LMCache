# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import replace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import (
    mooncake_legacy_key,
    mooncake_payload_layout,
    resolve_mooncake_dsa_raw_token_dims,
)
from lmcache.v1.token_database import ChunkedTokenDatabase, TokenDatabase


@pytest.fixture
def metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test-model",
        world_size=4,
        local_world_size=4,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(3, 1, 1024, 1, 576),
        use_mla=True,
        runtime_kv_group_layer_counts=(3, 1),
        runtime_kv_group_layer_names=(("a", "b", "c"), ("a.indexer",)),
    )


@pytest.mark.parametrize("tokens", [0, 1, 7, 127, 128, 129, 754, 1023, 1024])
def test_planar_packet_preserves_key_and_scale_bits(tokens: int) -> None:
    layout = IndexerC8Layout()
    key = torch.arange(tokens * 128).to(torch.int8)
    scale_bits = torch.arange(tokens).to(torch.int16)
    scales = scale_bits.view(torch.float16)
    packet = torch.empty(layout.layer_bytes(tokens), dtype=torch.uint8)
    for (offset, length), tensor in zip(
        layout.plane_ranges(tokens), (key, scales), strict=True
    ):
        packet[offset : offset + length].copy_(tensor.view(torch.uint8))
    assert torch.equal(packet[: tokens * 128].view(torch.int8), key)
    assert torch.equal(packet[tokens * 128 :].view(torch.int16), scale_bits)


def test_c8_metadata_keeps_latent_dtype_and_distinct_indexer_bytes(
    metadata: LMCacheMetadata,
) -> None:
    original = metadata.get_shapes(7)[0]
    c8 = replace(metadata, indexer_c8_layout=IndexerC8Layout())
    assert c8.get_shapes(7) == [original, torch.Size([1, 1, 7, 130])]
    assert c8.get_dtypes() == [torch.bfloat16, torch.uint8]
    assert c8.get_num_groups() == 2
    assert metadata.get_dtypes() == [torch.bfloat16]


def test_execution_identity_isolates_both_groups(metadata: LMCacheMetadata) -> None:
    config = LMCacheEngineConfig.from_defaults(
        dsa_two_groups=True,
        remote_fill_model_artifact_id="weights",
        remote_fill_cache_namespace="test",
        extra_config={"mooncake_dsa_raw_token_dims": {0: 576, 1: 128}},
    )
    original, descriptor = mooncake_payload_layout(config, metadata)
    c8 = replace(metadata, indexer_c8_layout=IndexerC8Layout())
    quantized, c8_descriptor = mooncake_payload_layout(config, c8)
    assert original != quantized
    assert "indexer_execution" not in descriptor
    assert c8_descriptor["indexer_execution"] == IndexerC8Layout().descriptor()
    assert resolve_mooncake_dsa_raw_token_dims(config, metadata)[0] == {0: 576, 1: 128}
    assert resolve_mooncake_dsa_raw_token_dims(config, c8)[0] == {0: 576, 1: 130}
    config.extra_config["mooncake_dsa_raw_token_dims"][1] = 256
    with pytest.raises(ValueError, match="conflicts"):
        resolve_mooncake_dsa_raw_token_dims(config, c8)


def test_invalid_c8_geometry_rejected() -> None:
    with pytest.raises(ValueError):
        IndexerC8Layout(256)
    with pytest.raises(ValueError):
        IndexerC8Layout().plane_ranges(-1)


@pytest.mark.parametrize("tokens", [-1, 1.5, True])
def test_packet_rejects_nonintegral_or_negative_token_counts(tokens):
    with pytest.raises(ValueError, match="nonnegative integer"):
        IndexerC8Layout().plane_ranges(tokens)


@pytest.mark.parametrize("page_first", [False, True])
def test_c8_namespace_survives_legacy_tail_lookup(metadata, monkeypatch, page_first):
    monkeypatch.setattr(TokenDatabase, "_get_vllm_hash_func", lambda *args: hash)
    config = LMCacheEngineConfig.from_defaults(
        dsa_two_groups=True,
        save_unfull_chunk=True,
        remote_fill_model_artifact_id="weights",
        remote_fill_cache_namespace="test",
        extra_config={
            "mooncake_dsa_raw_token_dims": {0: 576, 1: 128},
            "mooncake_page_first_multi_buffer": page_first,
        },
    )
    baseline = ChunkedTokenDatabase(config, metadata)
    c8 = ChunkedTokenDatabase(
        config, replace(metadata, indexer_c8_layout=IndexerC8Layout())
    )
    assert c8.mooncake_payload_layout is not None
    if not page_first:
        assert baseline.mooncake_payload_layout is None
    for group in (0, 1):
        old = list(baseline.process_tokens(tokens=[1, 2, 3], kv_group=group))[0][2]
        new = list(c8.process_tokens(tokens=[1, 2, 3], kv_group=group))[0][2]
        assert old.to_string() != new.to_string()
        assert c8.mooncake_payload_layout in mooncake_legacy_key(new)
        assert mooncake_legacy_key(old) != mooncake_legacy_key(new)


def test_c8_identity_records_equal_sized_physical_layer_maps(metadata):
    config = LMCacheEngineConfig.from_defaults(
        dsa_two_groups=True,
        remote_fill_model_artifact_id="weights",
        remote_fill_cache_namespace="test",
    )
    first = replace(
        metadata,
        indexer_c8_layout=IndexerC8Layout(),
        runtime_kv_group_layer_counts=(1, 1),
        runtime_kv_group_layer_names=(("layer.0",), ("layer.0.indexer",)),
    )
    second = replace(
        first, runtime_kv_group_layer_names=(("layer.1",), ("layer.1.indexer",))
    )
    assert (
        mooncake_payload_layout(config, first)[0]
        != mooncake_payload_layout(config, second)[0]
    )


def test_mixed_mooncake_layer_and_page_metadata(metadata):
    from types import SimpleNamespace
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
        MooncakestoreConnector,
    )

    policy = IndexerC8Layout(c8_layers=(False, True, False))
    mixed = replace(
        metadata,
        indexer_c8_layout=policy,
        runtime_kv_group_layer_counts=(3, 3),
        runtime_kv_group_layer_names=(
            ("a", "b", "c"),
            ("a.indexer", "b.indexer", "c.indexer"),
        ),
    )
    connector = object.__new__(MooncakestoreConnector)
    connector._mixed_indexer_layout = policy
    connector.local_cpu_backend = SimpleNamespace(metadata=mixed)
    connector._lmcache_chunk_size = lambda: 1024
    key = CacheEngineKey("model", 1, 0, 7, torch.uint8, kv_group=1)
    shapes, dtypes, _, width = connector._metadata_for_raw_key(key)
    assert shapes == mixed.indexer_layer_shapes(1024)
    assert dtypes == [torch.uint8] * 3
    assert width == 642
    for i, layer_key in enumerate(key.split_layers(3)):
        layer_shapes, layer_dtypes, _, layer_width = connector._metadata_for_raw_key(
            layer_key
        )
        assert layer_shapes == [shapes[i]]
        assert layer_dtypes == [torch.uint8]
        assert layer_width == policy.token_bytes_for(i)
    config = LMCacheEngineConfig.from_defaults(
        dsa_two_groups=True,
        remote_fill_model_artifact_id="weights",
        remote_fill_cache_namespace="test",
    )
    other = replace(
        mixed, indexer_c8_layout=IndexerC8Layout(c8_layers=(True, False, False))
    )
    assert (
        mooncake_payload_layout(config, mixed)[0]
        != mooncake_payload_layout(config, other)[0]
    )


def test_worker_hbm_permutation_does_not_change_persistent_identity(metadata):
    policy = IndexerC8Layout(c8_layers=(False, True, False))
    base = replace(
        metadata,
        runtime_kv_group_layer_counts=(3, 3),
        runtime_kv_group_layer_names=(
            ("a", "b", "c"),
            ("a.indexer", "b.indexer", "c.indexer"),
        ),
        indexer_c8_layout=policy,
    )
    cfg = LMCacheEngineConfig.from_defaults(
        dsa_two_groups=True,
        remote_fill_model_artifact_id="weights",
        remote_fill_cache_namespace="ns",
    )
    remapped = replace(base, indexer_hbm_block_map=tuple(range(36)))
    assert mooncake_payload_layout(cfg, base) == mooncake_payload_layout(cfg, remapped)
    assert base.get_shapes(7) == remapped.get_shapes(7)
    assert base.get_dtypes() == remapped.get_dtypes()
    assert "indexer_hbm_block_map" not in repr(remapped)
